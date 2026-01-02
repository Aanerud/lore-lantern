"""
Voice Processing Service

Handles speech-to-text and text-to-speech for real-time voice interactions.

TTS Provider Priority:
1. ElevenLabs (primary) - Best quality, handles expressiveness natively
2. Speechify (fallback) - Norwegian support
3. OpenAI TTS (dialogue) - gpt-4o-mini-tts for CompanionAgent voice

NOTE: TTS services remain as direct API calls even when Foundry is enabled.
Microsoft Foundry Model Router does NOT support audio input/output, so:
- ElevenLabs stays as direct API (external provider)
- OpenAI TTS for dialogue stays as direct API
- Speechify stays as direct API (external provider)
This is by design - Model Router is for text-based LLM operations only.
"""

import base64
import io
import re
from typing import Optional, AsyncGenerator, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import logging

logger = logging.getLogger(__name__)

# Google Cloud TTS has a 5000 byte limit per request
# Use 4500 to leave buffer for SSML wrapper tags
GOOGLE_TTS_MAX_BYTES = 4500

try:
    from google.cloud import speech_v1p1beta1 as speech
    from google.cloud import texttospeech
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    print("⚠️  Google Cloud Speech libraries not installed. Install with: pip install google-cloud-speech google-cloud-texttospeech")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")

try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️  ElevenLabs library not installed. Install with: pip install elevenlabs")


class VoiceService:
    """
    Voice processing service for speech-to-text and text-to-speech.

    Features:
    - Streaming speech recognition
    - Child-friendly voice synthesis
    - Audio format conversion
    - Async processing with thread pool
    - Multi-provider support (Google Cloud, OpenAI)
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.provider = None  # Legacy - will be set but use_case routing takes precedence
        self.fallback_provider = None  # Fallback when primary fails

        print("🎤 Initializing VoiceService...")
        print("   📋 Architecture: Dialogue→OpenAI, Narration→Speechify (or ElevenLabs if enabled)")

        # Get settings
        from src.config import get_settings
        settings = get_settings()

        # Initialize clients (we may use multiple)
        self.elevenlabs_client = None
        self.openai_client = None
        self.speechify_api_key = None
        self.speech_client = None  # Google STT
        self.tts_client = None     # Google TTS

        # Track which providers are available for each use case
        self.dialogue_provider = None   # OpenAI gpt-4o-mini-tts
        self.narration_provider = None  # Speechify or ElevenLabs

        # 1. Initialize OpenAI (REQUIRED for dialogue)
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.dialogue_provider = 'openai'
                    self.provider = 'openai'  # Legacy compatibility
                    print("✅ OpenAI TTS initialized for DIALOGUE")
                    print(f"   📊 Model: {settings.openai_tts_model}")
                    print(f"   🎤 Voice: {settings.openai_tts_voice}")
                else:
                    print("⚠️  OPENAI_API_KEY not set - dialogue TTS unavailable")
            except Exception as e:
                print(f"⚠️  Could not initialize OpenAI: {e}")

        # 2. Initialize Speechify (for narration when ElevenLabs disabled)
        speechify_key = settings.speechify_api_key or os.getenv("SPEECHIFY_API_KEY")
        if speechify_key:
            self.speechify_api_key = speechify_key
            if not settings.use_elevenlabs:
                self.narration_provider = 'speechify'
                print("✅ Speechify API initialized for NARRATION")
                print("   📊 Norwegian: nb-NO (beta)")
            else:
                print("✅ Speechify API available (backup for narration)")

        # 3. Initialize ElevenLabs (for narration when USE_ELEVENLABS=true)
        if ELEVENLABS_AVAILABLE and settings.elevenlabs_api_key:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=settings.elevenlabs_api_key)
                if settings.use_elevenlabs:
                    self.narration_provider = 'elevenlabs'
                    print("✅ ElevenLabs TTS initialized for NARRATION (USE_ELEVENLABS=true)")
                    print(f"   📊 Model: {settings.elevenlabs_model}")
                    print("   🎵 Audio tags: [whispers], [laughing], etc.")
                else:
                    print("✅ ElevenLabs available but DISABLED (set USE_ELEVENLABS=true to enable)")
            except Exception as e:
                print(f"⚠️  Could not initialize ElevenLabs: {e}")

        # 4. Google Cloud for STT only (not used for TTS in new architecture)
        if GOOGLE_SPEECH_AVAILABLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                self.speech_client = speech.SpeechClient()
                self.tts_client = texttospeech.TextToSpeechClient()
                print("✅ Google Cloud Speech available for STT")
            except Exception as e:
                print(f"⚠️  Could not initialize Google Speech: {e}")

        # Set fallback chain for narration
        if self.narration_provider == 'elevenlabs' and self.speechify_api_key:
            self.fallback_provider = 'speechify'
        elif self.narration_provider == 'speechify' and self.openai_client:
            self.fallback_provider = 'openai'

        # Summary
        print("─" * 50)
        print(f"   🗣️  Dialogue provider: {self.dialogue_provider or 'NONE'}")
        print(f"   📖 Narration provider: {self.narration_provider or 'NONE'}")
        print(f"   🔄 Fallback: {self.fallback_provider or 'none'}")
        if not self.dialogue_provider and not self.narration_provider:
            print("❌ No TTS providers available!")
        else:
            print("✅ VoiceService ready")

    async def speech_to_text(
        self,
        audio_data: bytes,
        language_code: str = "en-US",
        sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Convert speech audio to text.

        Args:
            audio_data: Raw audio bytes (WAV, FLAC, or LINEAR16)
            language_code: Language code (e.g., "en-US", "no-NO")
            sample_rate: Audio sample rate in Hz

        Returns:
            Transcribed text or None if failed
        """
        if not self.provider:
            print("⚠️  Speech-to-text not available (no provider configured)")
            return None

        try:
            loop = asyncio.get_event_loop()

            if self.provider == 'google':
                result = await loop.run_in_executor(
                    self.executor,
                    self._do_speech_recognition_google,
                    audio_data,
                    language_code,
                    sample_rate
                )
            elif self.provider == 'openai':
                result = await loop.run_in_executor(
                    self.executor,
                    self._do_speech_recognition_openai,
                    audio_data,
                    language_code
                )
            else:
                return None

            return result
        except Exception as e:
            print(f"❌ Speech-to-text error: {e}")
            return None

    def _do_speech_recognition_google(
        self,
        audio_data: bytes,
        language_code: str,
        sample_rate: int
    ) -> Optional[str]:
        """Synchronous Google speech recognition (runs in thread pool)"""
        audio = speech.RecognitionAudio(content=audio_data)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            # Enhanced models for better child voice recognition
            use_enhanced=True,
            model="default",
            enable_automatic_punctuation=True,
        )

        response = self.speech_client.recognize(config=config, audio=audio)

        # Return first result if available
        if response.results:
            return response.results[0].alternatives[0].transcript
        return None

    def _do_speech_recognition_openai(
        self,
        audio_data: bytes,
        language_code: str
    ) -> Optional[str]:
        """Synchronous OpenAI Whisper speech recognition (runs in thread pool)"""
        # OpenAI Whisper expects audio file, so save to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        try:
            with open(temp_audio_path, "rb") as audio_file:
                # Map language codes to Whisper format (ISO 639-1)
                whisper_lang = language_code.split('-')[0] if '-' in language_code else language_code

                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=whisper_lang
                )
                return transcript.text
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    def _is_ssml(self, text: str) -> bool:
        """Check if text is SSML formatted (starts with <speak>)"""
        return text.strip().startswith('<speak>')

    def _ssml_to_plain_text(self, ssml: str) -> str:
        """
        Convert SSML to plain text for providers that don't support SSML.

        Preserves natural reading cues:
        - <break time="Xms"/> → "..." for pauses
        - <emphasis> → preserved text
        - <prosody> → preserved text
        - All other tags → stripped

        Args:
            ssml: SSML-formatted text

        Returns:
            Plain text with natural pause cues
        """
        text = ssml

        # Convert break tags to ellipsis for natural pauses
        # Short pauses (<500ms) → single pause
        text = re.sub(r'<break\s+time="[0-4]\d{0,2}ms"\s*/>', ' ', text)
        # Longer pauses → ellipsis
        text = re.sub(r'<break\s+time="\d+ms"\s*/>', '... ', text)

        # Remove prosody tags but keep content
        text = re.sub(r'<prosody[^>]*>', '', text)
        text = re.sub(r'</prosody>', '', text)

        # Remove emphasis tags but keep content
        text = re.sub(r'<emphasis[^>]*>', '', text)
        text = re.sub(r'</emphasis>', '', text)

        # Remove speak wrapper
        text = re.sub(r'</?speak>', '', text)

        # Remove any remaining XML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _split_ssml_into_chunks(self, ssml: str, max_bytes: int = GOOGLE_TTS_MAX_BYTES) -> List[str]:
        """
        Split SSML content into chunks that fit within Google TTS byte limit.

        Strategy:
        1. Remove outer <speak> tags
        2. Split at natural break points (sentences, then prosody tags)
        3. Re-wrap each chunk in <speak> tags
        4. Ensure each chunk is under max_bytes

        Args:
            ssml: Full SSML content
            max_bytes: Maximum bytes per chunk (default 4500)

        Returns:
            List of SSML chunks, each wrapped in <speak> tags
        """
        # Remove outer <speak> tags and prosody wrapper if present
        content = re.sub(r'^\s*<speak>\s*', '', ssml)
        content = re.sub(r'\s*</speak>\s*$', '', content)

        # If content fits, return as single chunk
        wrapped = f"<speak>{content}</speak>"
        if len(wrapped.encode('utf-8')) <= max_bytes:
            return [wrapped]

        logger.info(f"📄 Content is {len(wrapped.encode('utf-8'))} bytes, need to split...")

        # Extract any outer prosody wrapper to reapply to each chunk
        prosody_match = re.match(r'^(<prosody[^>]*>)(.*)(</prosody>)$', content, re.DOTALL)
        if prosody_match:
            prosody_open = prosody_match.group(1)
            prosody_close = prosody_match.group(3)
            inner_content = prosody_match.group(2)
            logger.info(f"📄 Found prosody wrapper: {prosody_open}")
        else:
            prosody_open = ""
            prosody_close = ""
            inner_content = content

        # Split by sentences - look for sentence-ending punctuation
        # This regex captures the punctuation to preserve it
        sentence_parts = re.split(r'(?<=[.!?])\s+', inner_content)

        chunks = []
        current_chunk = ""

        for part in sentence_parts:
            # Build test chunk with prosody wrapper
            if prosody_open:
                test_content = f"{prosody_open}{current_chunk} {part}{prosody_close}" if current_chunk else f"{prosody_open}{part}{prosody_close}"
            else:
                test_content = f"{current_chunk} {part}" if current_chunk else part

            test_wrapped = f"<speak>{test_content}</speak>"

            if len(test_wrapped.encode('utf-8')) <= max_bytes:
                current_chunk = f"{current_chunk} {part}" if current_chunk else part
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    if prosody_open:
                        chunk_content = f"{prosody_open}{current_chunk.strip()}{prosody_close}"
                    else:
                        chunk_content = current_chunk.strip()
                    chunks.append(f"<speak>{chunk_content}</speak>")

                # Check if the new part itself is too big
                if prosody_open:
                    part_wrapped = f"<speak>{prosody_open}{part}{prosody_close}</speak>"
                else:
                    part_wrapped = f"<speak>{part}</speak>"

                if len(part_wrapped.encode('utf-8')) > max_bytes:
                    # Part is too big even alone - split by character count
                    logger.warning(f"⚠️ Sentence too long ({len(part_wrapped.encode('utf-8'))} bytes), splitting mid-sentence")
                    sub_chunks = self._split_by_char_count(part, max_bytes - 100, prosody_open, prosody_close)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1].replace("<speak>", "").replace("</speak>", "")
                    if prosody_open:
                        current_chunk = current_chunk.replace(prosody_open, "").replace(prosody_close, "")
                else:
                    current_chunk = part

        # Don't forget the last chunk
        if current_chunk.strip():
            if prosody_open:
                chunk_content = f"{prosody_open}{current_chunk.strip()}{prosody_close}"
            else:
                chunk_content = current_chunk.strip()
            chunks.append(f"<speak>{chunk_content}</speak>")

        logger.info(f"📄 Split SSML into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunk_bytes = len(chunk.encode('utf-8'))
            logger.info(f"   Chunk {i+1}/{len(chunks)}: {chunk_bytes} bytes")

        return chunks

    def _split_by_char_count(self, text: str, max_bytes: int, prosody_open: str = "", prosody_close: str = "") -> List[str]:
        """Split text by approximate character count when sentences are too long."""
        chunks = []
        # Rough estimate: 1 char ≈ 1-2 bytes for ASCII/Latin, more for other scripts
        # Be conservative with chunk size
        safe_char_limit = max_bytes // 2

        words = text.split()
        current = ""

        for word in words:
            test = f"{current} {word}" if current else word
            if prosody_open:
                test_wrapped = f"<speak>{prosody_open}{test}{prosody_close}</speak>"
            else:
                test_wrapped = f"<speak>{test}</speak>"

            if len(test_wrapped.encode('utf-8')) <= max_bytes:
                current = test
            else:
                if current:
                    if prosody_open:
                        chunks.append(f"<speak>{prosody_open}{current}{prosody_close}</speak>")
                    else:
                        chunks.append(f"<speak>{current}</speak>")
                current = word

        if current:
            if prosody_open:
                chunks.append(f"<speak>{prosody_open}{current}{prosody_close}</speak>")
            else:
                chunks.append(f"<speak>{current}</speak>")

        return chunks

    def _concatenate_mp3_audio(self, audio_chunks: List[bytes]) -> bytes:
        """
        Concatenate multiple MP3 audio chunks into a single audio stream.

        MP3 is designed to allow simple byte concatenation for streaming.

        Args:
            audio_chunks: List of MP3 audio bytes

        Returns:
            Combined MP3 audio bytes
        """
        if not audio_chunks:
            return b""
        if len(audio_chunks) == 1:
            return audio_chunks[0]

        # Simple concatenation works for MP3
        return b"".join(audio_chunks)

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        use_ssml: bool = False,
        use_case: str = "dialogue"  # NEW: "dialogue" | "narration"
    ) -> Optional[bytes]:
        """
        Convert text to speech audio with use-case-based provider routing.

        Args:
            text: Text to convert to speech (plain text or SSML)
            language_code: Language code (e.g., "en-US", "nb-NO")
            voice_name: Specific voice name (optional)
            speaking_rate: Speed (0.25 to 4.0, default 1.0)
            pitch: Pitch adjustment (-20.0 to 20.0, default 0.0)
            use_ssml: If True, treat input as SSML
            use_case: "dialogue" (OpenAI) or "narration" (Speechify/ElevenLabs)

        Returns:
            Audio bytes (MP3 format) or None if failed
        """
        # Check if TTS is disabled (development mode)
        from src.config import get_settings
        settings = get_settings()
        if settings.disable_tts:
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"🔇 TTS DISABLED (dev mode): '{text_preview}'")
            return None

        # Auto-detect SSML
        is_ssml = use_ssml or self._is_ssml(text)

        # Log TTS request with use_case
        text_preview = text[:50] + "..." if len(text) > 50 else text
        provider_for_case = self.dialogue_provider if use_case == "dialogue" else self.narration_provider
        logger.info(f"🎵 TTS [{use_case.upper()}]: '{text_preview}'")
        logger.info(f"   Provider: {provider_for_case}, Lang: {language_code}")

        try:
            loop = asyncio.get_event_loop()
            result = None

            # Route based on use_case
            if use_case == "narration":
                # Story chapters → Speechify or ElevenLabs
                result = await self._tts_narration(text, language_code, is_ssml, loop)
            else:
                # Dialogue → OpenAI gpt-4o-mini-tts
                result = await self._tts_dialogue(text, language_code, loop)

            # Log result
            if result:
                logger.info(f"✅ TTS Success [{use_case}]: Generated {len(result)} bytes of audio")
            else:
                logger.warning(f"❌ TTS Failed [{use_case}]: No audio bytes returned")

            return result
        except Exception as e:
            logger.error(f"❌ Text-to-speech error [{use_case}]: {e}", exc_info=True)
            return None

    async def _tts_dialogue(self, text: str, language_code: str, loop) -> Optional[bytes]:
        """Handle dialogue TTS with OpenAI gpt-4o-mini-tts."""
        if not self.dialogue_provider or not self.openai_client:
            logger.error("❌ Dialogue TTS not available (OpenAI not configured)")
            return None

        try:
            result = await loop.run_in_executor(
                self.executor,
                self._do_text_to_speech_openai_dialogue,
                text,
                language_code
            )
            return result
        except Exception as e:
            logger.error(f"❌ OpenAI dialogue TTS failed: {e}")
            return None

    async def _tts_narration(self, text: str, language_code: str, is_ssml: bool, loop) -> Optional[bytes]:
        """Handle narration TTS with Speechify or ElevenLabs."""
        from src.config import get_settings
        settings = get_settings()

        # Sanitize control characters that can corrupt text (e.g., \x08 backspace)
        # These break sentence detection and cause chunking issues
        import re
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # TTS timeout scales with text length:
        # - Base: 60 seconds
        # - Per 2000 chars (roughly 1 chunk): 30 seconds (allows for retries on 503)
        # - Minimum: 120 seconds
        # For 22689 chars (14 chunks): 60 + (12 * 30) = 420 seconds (7 minutes)
        estimated_chunks = max(1, len(text) // 2000 + 1)
        TTS_TIMEOUT_SECONDS = max(120, 60 + estimated_chunks * 30)
        logger.info(f"🎵 TTS timeout set to {TTS_TIMEOUT_SECONDS}s for {len(text)} chars (~{estimated_chunks} chunks)")

        # Try primary narration provider
        if self.narration_provider == 'elevenlabs' and self.elevenlabs_client:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self._do_text_to_speech_elevenlabs,
                        text,
                        language_code
                    ),
                    timeout=TTS_TIMEOUT_SECONDS
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"❌ ElevenLabs TTS timed out after {TTS_TIMEOUT_SECONDS}s")
                # Fall through to fallback
            except Exception as e:
                error_str = str(e)
                if "quota_exceeded" in error_str or "401" in error_str:
                    logger.warning(f"⚠️ ElevenLabs quota exceeded, trying fallback...")
                else:
                    logger.error(f"❌ ElevenLabs narration failed: {e}")
                    # Fall through to fallback

        elif self.narration_provider == 'speechify' and self.speechify_api_key:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self._do_text_to_speech_speechify,
                        text,
                        language_code
                    ),
                    timeout=TTS_TIMEOUT_SECONDS
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"❌ Speechify TTS timed out after {TTS_TIMEOUT_SECONDS}s")
                # Fall through to fallback
            except Exception as e:
                logger.warning(f"⚠️ Speechify narration failed: {e}, trying fallback...")

        # Try fallback
        if self.fallback_provider == 'speechify' and self.speechify_api_key:
            try:
                logger.info("🔄 Using Speechify as fallback for narration...")
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self._do_text_to_speech_speechify,
                        text,
                        language_code
                    ),
                    timeout=TTS_TIMEOUT_SECONDS
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"❌ Speechify fallback timed out after {TTS_TIMEOUT_SECONDS}s")
            except Exception as e:
                logger.warning(f"⚠️ Speechify fallback failed: {e}")

        elif self.fallback_provider == 'openai' and self.openai_client:
            try:
                logger.info("🔄 Using OpenAI tts-1 as fallback for narration...")
                tts_text = self._ssml_to_plain_text(text) if is_ssml else text

                # OpenAI tts-1 has 4096 character limit - chunk if needed
                OPENAI_TTS_MAX_CHARS = 4000  # Leave buffer for safety
                if len(tts_text) > OPENAI_TTS_MAX_CHARS:
                    logger.info(f"📄 Text too long for OpenAI ({len(tts_text)} chars), chunking...")
                    chunks = self._split_text_for_elevenlabs(tts_text, OPENAI_TTS_MAX_CHARS)
                    logger.info(f"📄 Split into {len(chunks)} chunks")

                    audio_chunks = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"🎵 OpenAI chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                        chunk_audio = await asyncio.wait_for(
                            loop.run_in_executor(
                                self.executor,
                                self._do_text_to_speech_openai,
                                chunk,
                                None,  # voice_name
                                1.0    # speaking_rate
                            ),
                            timeout=TTS_TIMEOUT_SECONDS
                        )
                        if chunk_audio:
                            audio_chunks.append(chunk_audio)

                    result = self._concatenate_mp3_audio(audio_chunks)
                    logger.info(f"✅ OpenAI fallback generated {len(result)} bytes ({len(chunks)} chunks)")
                    return result
                else:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor,
                            self._do_text_to_speech_openai,
                            tts_text,
                            None,  # voice_name
                            1.0    # speaking_rate
                        ),
                        timeout=TTS_TIMEOUT_SECONDS
                    )
                    return result
            except asyncio.TimeoutError:
                logger.error(f"❌ OpenAI fallback timed out after {TTS_TIMEOUT_SECONDS}s")
            except Exception as e:
                logger.error(f"❌ OpenAI fallback failed: {e}")

        logger.error("❌ No narration provider available")
        return None

    def _do_text_to_speech_google(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str],
        speaking_rate: float,
        pitch: float,
        is_ssml: bool = False
    ) -> Optional[bytes]:
        """Synchronous Google text-to-speech (runs in thread pool)"""

        # Voice selection
        if not voice_name:
            # Default to child-friendly voices
            voice_name = self._get_default_voice_google(language_code)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        # Audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch
        )

        # Check if we need to chunk the SSML
        if is_ssml and len(text.encode('utf-8')) > GOOGLE_TTS_MAX_BYTES:
            logger.info(f"📄 SSML too long ({len(text.encode('utf-8'))} bytes), splitting into chunks...")
            chunks = self._split_ssml_into_chunks(text)

            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"   🔊 Generating audio for chunk {i+1}/{len(chunks)} ({len(chunk.encode('utf-8'))} bytes)")
                synthesis_input = texttospeech.SynthesisInput(ssml=chunk)
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                audio_chunks.append(response.audio_content)

            # Concatenate all audio chunks
            combined_audio = self._concatenate_mp3_audio(audio_chunks)
            logger.info(f"✅ Combined {len(chunks)} chunks into {len(combined_audio)} bytes of audio")
            return combined_audio

        # Single chunk - normal processing
        if is_ssml:
            synthesis_input = texttospeech.SynthesisInput(ssml=text)
        else:
            synthesis_input = texttospeech.SynthesisInput(text=text)

        # Generate speech
        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        return response.audio_content

    def _do_text_to_speech_openai(
        self,
        text: str,
        voice_name: Optional[str],
        speaking_rate: float
    ) -> Optional[bytes]:
        """Synchronous OpenAI TTS (runs in thread pool)"""

        # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
        # Child-friendly defaults: nova (warm female), shimmer (soft female)
        if not voice_name:
            voice_name = "nova"  # Default to warm, friendly female voice

        response = self.openai_client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=voice_name,
            input=text,
            speed=speaking_rate
        )

        # OpenAI returns audio content directly
        return response.content

    def _do_text_to_speech_openai_dialogue(
        self,
        text: str,
        language_code: str
    ) -> Optional[bytes]:
        """
        OpenAI gpt-4o-mini-tts for companion dialogue.

        Uses the steerable gpt-4o-mini-tts model with instructions
        for warm, child-friendly speech.
        """
        from src.config import get_settings
        settings = get_settings()

        # Get language-appropriate instruction
        instructions = self._get_dialogue_instructions(language_code)

        try:
            response = self.openai_client.audio.speech.create(
                model=settings.openai_tts_model,  # gpt-4o-mini-tts
                voice=settings.openai_tts_voice,   # nova
                input=text,
                instructions=instructions
            )
            return response.content
        except Exception as e:
            # Fallback to tts-1 if gpt-4o-mini-tts fails
            logger.warning(f"⚠️ gpt-4o-mini-tts failed, falling back to tts-1: {e}")
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=settings.openai_tts_voice,
                input=text
            )
            return response.content

    def _get_dialogue_instructions(self, language_code: str) -> str:
        """Get voice style instructions for dialogue based on language."""
        if language_code.startswith("nb") or language_code.startswith("no"):
            return (
                "Speak warmly and gently like a friendly storyteller for children. "
                "Use clear Norwegian pronunciation. Be enthusiastic but calm. "
                "Speak at a pace suitable for young children."
            )
        elif language_code.startswith("es"):
            return (
                "Habla con calidez y dulzura como un narrador amigable para niños. "
                "Sé entusiasta pero tranquilo. Habla a un ritmo adecuado para niños pequeños."
            )
        else:
            return (
                "Speak warmly and gently like a friendly storyteller for children. "
                "Be enthusiastic but calm. Speak at a pace suitable for young children."
            )

    async def generate_dialogue_with_audio(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list = None,
        voice: str = None,
        audio_format: str = "mp3"
    ) -> tuple:
        """
        Generate dialogue text + audio in a SINGLE API call using GPT-4o-audio-preview.

        This eliminates the sequential LLM → TTS round-trip, reducing latency by ~50%.

        Args:
            system_prompt: System prompt defining the assistant persona
            user_message: The user's message to respond to
            conversation_history: Optional list of prior messages for context
            voice: Voice to use (default from settings)
            audio_format: Output format - mp3, wav, aac, flac, opus, pcm16

        Returns:
            Tuple of (text_response: str, audio_bytes: bytes, transcript: str)
        """
        from src.config import get_settings
        settings = get_settings()

        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        # Use configured voice or default
        voice = voice or settings.direct_audio_voice

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        # Make single API call that returns both text and audio
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model=settings.direct_audio_model,  # gpt-4o-audio-preview or gpt-4o-mini-audio-preview
            modalities=["text", "audio"],
            audio={
                "voice": voice,
                "format": audio_format
            },
            messages=messages
        )

        # Extract response components
        choice = response.choices[0]

        # Audio is returned as base64-encoded data
        # IMPORTANT: When audio modality is used, the spoken text is in audio.transcript,
        # NOT in message.content (which may be empty)
        audio_data = choice.message.audio
        if audio_data:
            audio_bytes = base64.b64decode(audio_data.data)
            transcript = audio_data.transcript or ""
            # Use transcript as the text response since message.content is often empty with audio
            text_response = transcript
        else:
            # Fallback if audio generation failed
            audio_bytes = b""
            text_response = choice.message.content or ""
            transcript = text_response

        logger.info(f"🎵 Direct audio: Generated {len(audio_bytes)} bytes audio + {len(text_response)} chars text")

        return text_response, audio_bytes, transcript

    # Speechify Speech endpoint has 2,000 character limit (including SSML tags)
    # See: https://docs.sws.speechify.com/docs/api-limits
    # Stream endpoint supports 20,000 chars but we use the Speech endpoint
    SPEECHIFY_MAX_CHARS = 1900  # Leave buffer for SSML tags

    def _do_text_to_speech_speechify(
        self,
        text: str,
        language_code: str
    ) -> Optional[bytes]:
        """
        Speechify TTS for story narration.

        Uses Speechify's REST API for high-quality narration.
        Norwegian (nb-NO) is in beta but supported.

        API Response format:
        {
            "audio_data": "base64_encoded_string",
            "audio_format": "mp3",
            "billable_characters_count": 123,
            "speech_marks": {...}
        }
        """
        import requests
        import base64

        # Map language code to Speechify format
        speechify_lang = self._get_speechify_language_code(language_code)

        # Get voice_id based on language (required by Speechify API)
        voice_id = self._get_speechify_voice_id(language_code)

        # Select model: simba-multilingual for non-English, simba-english for English
        lang_prefix = language_code.split('-')[0].lower() if '-' in language_code else language_code.lower()
        model = "simba-english" if lang_prefix == "en" else "simba-multilingual"

        # Input validation
        if not text or not text.strip():
            logger.error("❌ Speechify: Empty text input")
            return None

        # If text exceeds limit, split into chunks and concatenate audio
        if len(text) > self.SPEECHIFY_MAX_CHARS:
            logger.info(f"📄 Speechify: Text is {len(text)} chars, splitting into chunks...")
            chunks = self._split_text_for_speechify(text)
            logger.info(f"📄 Split into {len(chunks)} chunks")

            audio_parts = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🎵 Speechify chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                chunk_audio = self._speechify_single_request(chunk, voice_id, model, speechify_lang)
                if chunk_audio:
                    audio_parts.append(chunk_audio)
                else:
                    logger.error(f"❌ Speechify chunk {i+1} failed")
                    return None

            # Concatenate all audio parts
            combined_audio = b"".join(audio_parts)
            logger.info(f"✅ Speechify combined {len(audio_parts)} chunks into {len(combined_audio)} bytes")
            return combined_audio

        # Single request for short text
        return self._speechify_single_request(text, voice_id, model, speechify_lang)

    def _split_text_for_speechify(self, text: str) -> list:
        """Split text into chunks that fit within Speechify's character limit."""
        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= self.SPEECHIFY_MAX_CHARS:
                chunks.append(remaining)
                break

            # Find a good break point (sentence end, paragraph, or word boundary)
            chunk = remaining[:self.SPEECHIFY_MAX_CHARS]

            # Try to break at sentence end
            for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > self.SPEECHIFY_MAX_CHARS // 2:  # Only if past halfway
                    chunk = remaining[:last_sep + len(sep)]
                    break
            else:
                # Fall back to word boundary
                last_space = chunk.rfind(' ')
                if last_space > self.SPEECHIFY_MAX_CHARS // 2:
                    chunk = remaining[:last_space + 1]

            chunks.append(chunk.strip())
            remaining = remaining[len(chunk):].strip()

        return chunks

    def _speechify_single_request(self, text: str, voice_id: str, model: str, language: str) -> Optional[bytes]:
        """Make a single Speechify TTS request with retry logic for transient errors."""
        import requests
        import base64
        import time

        url = "https://api.sws.speechify.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.speechify_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "voice_id": voice_id,
            "audio_format": "mp3",
            "model": model,
            "language": language
        }

        logger.info(f"🎵 Speechify TTS: {len(text)} chars, voice={voice_id}, model={model}, lang={language}")

        # Retry configuration for transient errors (503, 429, 502, 504)
        MAX_RETRIES = 3
        RETRY_CODES = {429, 502, 503, 504}  # Rate limit, bad gateway, unavailable, timeout
        INITIAL_DELAY = 2  # seconds

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=120)

                # Check for retryable errors
                if response.status_code in RETRY_CODES:
                    delay = INITIAL_DELAY * (2 ** (attempt - 1))  # Exponential backoff: 2s, 4s, 8s
                    logger.warning(f"⚠️ Speechify returned {response.status_code}, retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(delay)
                    continue

                # Non-retryable error
                if not response.ok:
                    error_detail = response.text[:500] if response.text else "No error details"
                    logger.error(f"❌ Speechify API error {response.status_code}: {error_detail}")
                    logger.error(f"   Payload: voice_id={voice_id}, model={model}, lang={language}, text_len={len(text)}")
                    logger.error(f"   Text preview: {text[:200]}...")
                    response.raise_for_status()

                # Success - parse response
                response_data = response.json()
                audio_base64 = response_data.get("audio_data")

                if not audio_base64:
                    logger.error("❌ Speechify response missing audio_data field")
                    return None

                # Decode Base64 to bytes
                audio_bytes = base64.b64decode(audio_base64)

                billable_chars = response_data.get("billable_characters_count", 0)
                if attempt > 1:
                    logger.info(f"✅ Speechify succeeded on retry {attempt}: {len(audio_bytes)} bytes ({billable_chars} billable chars)")
                else:
                    logger.info(f"✅ Speechify generated {len(audio_bytes)} bytes of audio ({billable_chars} billable chars)")
                return audio_bytes

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = INITIAL_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"⚠️ Speechify request failed: {e}, retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Speechify failed after {MAX_RETRIES} attempts: {e}")
                    raise

        # All retries exhausted
        logger.error(f"❌ Speechify failed after {MAX_RETRIES} retries")
        if last_error:
            raise last_error
        return None

    def _get_speechify_language_code(self, language_code: str) -> str:
        """Map BCP-47 language codes to Speechify format."""
        mapping = {
            # Norwegian
            "nb-NO": "nb-NO",
            "no-NO": "nb-NO",
            "no": "nb-NO",
            # English (full ISO format for API compliance)
            "en-US": "en-US",
            "en-GB": "en-GB",
            "en": "en-US",
            # Other supported languages
            "es-ES": "es-ES",
            "es": "es-ES",
            "de-DE": "de-DE",
            "de": "de-DE",
            "fr-FR": "fr-FR",
            "fr": "fr-FR",
            "pt-BR": "pt-BR",
            "pt-PT": "pt-PT",
        }
        result = mapping.get(language_code, "en-US")  # Default to en-US for unknown languages
        logger.debug(f"Speechify language mapping: {language_code} → {result}")
        return result

    def _get_speechify_voice_id(self, language_code: str) -> str:
        """
        Get Speechify voice ID based on language.

        Norwegian voices tested (nb-NO):
        - vegard, eirik, tor, roar - male voices that sound more Norwegian
        - sunniva, sigrid, frida - female voices
        - harald - sounds more Danish than Norwegian (avoided)

        English voices:
        - oliver - clear English narrator
        """
        # Extract base language
        lang = language_code.split('-')[0].lower() if '-' in language_code else language_code.lower()

        # Map language to voice ID
        voice_map = {
            "nb": "vegard",   # Norwegian Bokmål - clearer Norwegian accent
            "no": "vegard",   # Norwegian
            "nn": "vegard",   # Norwegian Nynorsk
            "en": "oliver",   # English
        }

        voice_id = voice_map.get(lang, "oliver")  # Default to English voice
        logger.info(f"🗣️ Speechify voice for {language_code}: {voice_id}")
        return voice_id

    # ElevenLabs has a 5000 character limit per request
    ELEVENLABS_MAX_CHARS = 4800  # Leave buffer for safety

    def _do_text_to_speech_elevenlabs(
        self,
        text: str,
        language_code: str
    ) -> Optional[bytes]:
        """
        Synchronous ElevenLabs TTS (runs in thread pool).

        Uses Eleven v3 for all languages:
        - Supports [audio tags] for expressive narration
        - Supports 74 languages with explicit language_code (ISO 639-1 format)
        - language_code ensures Norwegian sounds Norwegian (not Danish!)
        - Automatically chunks text > 5000 chars

        Args:
            text: Plain text to convert (can include [audio tags])
            language_code: Language code (e.g., "en-US", "nb-NO")

        Returns:
            Audio bytes (MP3 format)
        """
        from src.config import get_settings
        settings = get_settings()

        # Get voice ID for language
        voice_id = self._get_elevenlabs_voice_id(language_code)

        # Get ElevenLabs language code (ISO 639-1 format, e.g., "en", "no")
        el_language = self._get_elevenlabs_language_code(language_code)

        # Strip any SSML tags - ElevenLabs uses [audio tags] instead
        clean_text = self._ssml_to_plain_text(text) if self._is_ssml(text) else text

        # Use Eleven v3 for all languages
        model = settings.elevenlabs_model  # eleven_v3

        # Check if we need to chunk the text (5000 char limit)
        if len(clean_text) > self.ELEVENLABS_MAX_CHARS:
            logger.info(f"📄 Text too long ({len(clean_text)} chars), splitting into chunks...")
            chunks = self._split_text_for_elevenlabs(clean_text)
            logger.info(f"📄 Split into {len(chunks)} chunks")

            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🎵 ElevenLabs chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                audio_generator = self.elevenlabs_client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=chunk,
                    model_id=model,
                    output_format=settings.elevenlabs_output_format,
                    language_code=el_language
                )
                chunk_audio = b"".join(audio_generator)
                audio_chunks.append(chunk_audio)
                logger.info(f"   ✅ Chunk {i+1}: {len(chunk_audio)} bytes")

            # Concatenate all audio chunks
            audio_bytes = self._concatenate_mp3_audio(audio_chunks)
            logger.info(f"✅ ElevenLabs generated {len(audio_bytes)} bytes ({len(chunks)} chunks)")
            return audio_bytes

        # Single request for short text
        logger.info(f"🎵 ElevenLabs TTS (v3): {len(clean_text)} chars, voice={voice_id}, lang={el_language}")

        audio_generator = self.elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=clean_text,
            model_id=model,
            output_format=settings.elevenlabs_output_format,
            language_code=el_language  # ISO 639-1 format (e.g., "en", "no", "es")
        )

        # Convert generator to bytes
        audio_bytes = b"".join(audio_generator)

        logger.info(f"✅ ElevenLabs generated {len(audio_bytes)} bytes of audio")
        return audio_bytes

    def _split_text_for_elevenlabs(self, text: str, max_chars: int = None) -> List[str]:
        """
        Split text into chunks under character limit.

        Splits at paragraph boundaries first, then sentences, then by character count.
        Preserves [audio tags] at the start of chunks where possible.

        Args:
            text: Full text to split
            max_chars: Maximum characters per chunk (default: ELEVENLABS_MAX_CHARS)

        Returns:
            List of text chunks, each under max_chars
        """
        if max_chars is None:
            max_chars = self.ELEVENLABS_MAX_CHARS
        chunks = []

        # Helper to force-split oversized text at word boundaries
        def force_split_text(text: str, limit: int) -> List[str]:
            """Split text at word boundaries to fit within limit."""
            if len(text) <= limit:
                return [text]

            result = []
            words = text.split(' ')
            current = ""

            for word in words:
                test = f"{current} {word}" if current else word
                if len(test) <= limit:
                    current = test
                else:
                    if current:
                        result.append(current)
                    # Handle single words longer than limit (rare but possible)
                    if len(word) > limit:
                        # Split at limit boundaries
                        for i in range(0, len(word), limit):
                            result.append(word[i:i+limit])
                        current = ""
                    else:
                        current = word

            if current:
                result.append(current)

            return result

        # First, split by paragraphs (double newline)
        paragraphs = text.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            # Check if adding this paragraph exceeds limit
            test_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Check if paragraph itself is too long
                if len(para) > max_chars:
                    # Split paragraph by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    para_chunk = ""
                    for sentence in sentences:
                        # If sentence itself exceeds limit, force-split it
                        if len(sentence) > max_chars:
                            if para_chunk.strip():
                                chunks.append(para_chunk.strip())
                                para_chunk = ""
                            # Force-split the oversized sentence
                            forced_chunks = force_split_text(sentence, max_chars)
                            # Add all but last to chunks, keep last as para_chunk
                            for fc in forced_chunks[:-1]:
                                chunks.append(fc)
                            para_chunk = forced_chunks[-1] if forced_chunks else ""
                        else:
                            test = f"{para_chunk} {sentence}" if para_chunk else sentence
                            if len(test) <= max_chars:
                                para_chunk = test
                            else:
                                if para_chunk.strip():
                                    chunks.append(para_chunk.strip())
                                para_chunk = sentence
                    if para_chunk.strip():
                        current_chunk = para_chunk.strip()
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _get_elevenlabs_language_code(self, language_code: str) -> str:
        """
        Convert BCP-47 language code to ElevenLabs ISO 639-1 format.

        ElevenLabs API uses ISO 639-1 codes (2 letters) for language specification.
        This ensures Norwegian sounds Norwegian (not Danish!).

        Args:
            language_code: BCP-47 code (e.g., "en-US", "nb-NO", "no-NO")

        Returns:
            ElevenLabs language code (e.g., "en", "no", "es")
        """
        # Map BCP-47 prefixes to ISO 639-1 codes (2-letter)
        language_map = {
            "en": "en",  # English
            "no": "no",  # Norwegian
            "nb": "no",  # Norwegian Bokmål -> no
            "nn": "no",  # Norwegian Nynorsk -> no
            "es": "es",  # Spanish
            "de": "de",  # German
            "fr": "fr",  # French
            "it": "it",  # Italian
            "pt": "pt",  # Portuguese
            "nl": "nl",  # Dutch
            "sv": "sv",  # Swedish
            "da": "da",  # Danish
            "fi": "fi",  # Finnish
            "pl": "pl",  # Polish
            "ru": "ru",  # Russian
            "ja": "ja",  # Japanese
            "ko": "ko",  # Korean
            "zh": "zh",  # Chinese
        }

        # Extract base language
        lang = language_code.split('-')[0].lower() if '-' in language_code else language_code.lower()

        el_code = language_map.get(lang, "en")  # Default to English
        logger.info(f"🌍 Language mapping: {language_code} → {el_code}")
        return el_code

    def _get_elevenlabs_voice_id(self, language_code: str) -> str:
        """
        Get ElevenLabs voice ID for language.

        Uses settings.elevenlabs_voices mapping with fallback to English.

        Args:
            language_code: Full language code (e.g., "en-US", "nb-NO")

        Returns:
            ElevenLabs voice ID
        """
        from src.config import get_settings
        settings = get_settings()

        # Extract base language (en-US -> en, nb-NO -> nb)
        lang = language_code.split('-')[0].lower() if '-' in language_code else language_code.lower()

        # Handle Norwegian variants (nb, nn, no all map to "no")
        if lang in ('nb', 'nn'):
            lang = 'no'

        # Look up in settings
        voice_id = settings.elevenlabs_voices.get(lang)

        if voice_id:
            logger.info(f"🗣️ ElevenLabs voice for {language_code}: {voice_id}")
            return voice_id

        # Fallback to English
        fallback = settings.elevenlabs_voices.get("en", "21m00Tcm4TlvDq8ikWAM")
        logger.warning(f"⚠️ No ElevenLabs voice for {language_code}, using English: {fallback}")
        return fallback

    def _get_default_voice_google(self, language_code: str) -> str:
        """
        Get default child-friendly Google Wavenet voice for language.

        All voices are Wavenet (neural) quality for natural narration.
        Female voices selected for warm, friendly storytelling tone.

        Supported languages from parent preferences:
        - en-US: English (United States)
        - nb-NO: Norwegian Bokmål
        - es-ES: Spanish (Spain)
        """
        # Child-friendly Wavenet voices for storytelling
        # All verified to exist in Google Cloud TTS
        voices = {
            # Primary supported languages (from parent preferences)
            "en-US": "en-US-Wavenet-F",   # Female, warm and friendly
            "nb-NO": "nb-NO-Wavenet-F",   # Norwegian female
            "es-ES": "es-ES-Wavenet-F",   # Spanish female

            # Additional languages (fallbacks)
            "en-GB": "en-GB-Wavenet-A",   # British English female
            "en-AU": "en-AU-Wavenet-A",   # Australian English female
            "sv-SE": "sv-SE-Wavenet-A",   # Swedish female
            "da-DK": "da-DK-Wavenet-A",   # Danish female
            "fi-FI": "fi-FI-Wavenet-A",   # Finnish female
            "de-DE": "de-DE-Wavenet-C",   # German female
            "fr-FR": "fr-FR-Wavenet-C",   # French female
            "it-IT": "it-IT-Wavenet-A",   # Italian female
            "pt-BR": "pt-BR-Wavenet-A",   # Portuguese (Brazil) female
            "nl-NL": "nl-NL-Wavenet-A",   # Dutch female
            "pl-PL": "pl-PL-Wavenet-A",   # Polish female

            # Fallback mappings for alternate codes
            "no-NO": "nb-NO-Wavenet-F",   # Map no-NO to nb-NO voice
            "es-MX": "es-US-Wavenet-A",   # Mexican Spanish
            "es-US": "es-US-Wavenet-A",   # US Spanish
        }

        voice = voices.get(language_code)
        if voice:
            logger.info(f"🗣️ Selected Wavenet voice: {voice} for {language_code}")
            return voice

        # Fallback: try to find a matching language prefix
        lang_prefix = language_code.split('-')[0] if '-' in language_code else language_code
        for code, voice in voices.items():
            if code.startswith(lang_prefix):
                logger.info(f"🗣️ Fallback Wavenet voice: {voice} for {language_code} (matched {code})")
                return voice

        # Ultimate fallback
        logger.warning(f"⚠️ No Wavenet voice for {language_code}, using en-US-Wavenet-F")
        return "en-US-Wavenet-F"

    async def stream_text_to_speech(
        self,
        text: str,
        chunk_size: int = 4096,
        **tts_kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text-to-speech audio in chunks.

        Useful for sending audio over WebSocket as it's generated.

        Args:
            text: Text to convert
            chunk_size: Size of audio chunks to yield
            **tts_kwargs: Arguments for text_to_speech()

        Yields:
            Audio chunks (bytes)
        """
        # Generate full audio
        audio_data = await self.text_to_speech(text, **tts_kwargs)

        if not audio_data:
            return

        # Stream in chunks
        audio_stream = io.BytesIO(audio_data)
        while True:
            chunk = audio_stream.read(chunk_size)
            if not chunk:
                break
            yield chunk
            await asyncio.sleep(0)  # Allow other tasks to run

    def get_chapter_tts_content(self, chapter) -> tuple[str, bool]:
        """
        Get the appropriate text content for TTS from a Chapter.

        Prefers tts_content (SSML-optimized by VoiceDirectorAgent) when available,
        falls back to raw prose content.

        This is the audiobook production model:
        - tts_content: SSML-optimized narration with prosody, breaks, emphasis
        - content: Original prose meant for reading

        Args:
            chapter: Chapter model with content and optional tts_content

        Returns:
            Tuple of (text, is_ssml) where:
            - text: The content to speak
            - is_ssml: True if content is SSML-formatted
        """
        if hasattr(chapter, 'tts_content') and chapter.tts_content:
            return chapter.tts_content, True
        elif hasattr(chapter, 'content') and chapter.content:
            return chapter.content, False
        else:
            return "", False

    async def text_to_speech_for_chapter(
        self,
        chapter,
        language_code: str = "en-US",
        voice_name: Optional[str] = None,
        speaking_rate: float = 0.9,  # Slightly slower for children
        pitch: float = 0.0
    ) -> Optional[bytes]:
        """
        Generate audio for a chapter, preferring SSML tts_content when available.

        This convenience method implements the audiobook production pattern:
        - If chapter has tts_content (from VoiceDirectorAgent): use SSML-optimized narration
        - Otherwise: fall back to raw prose content

        Args:
            chapter: Chapter model with content and optional tts_content
            language_code: Language code (e.g., "en-US")
            voice_name: Specific voice name (e.g., "nova")
            speaking_rate: Speed (0.25 to 4.0, default 0.9 for children)
            pitch: Pitch adjustment (-20.0 to 20.0, default 0.0)

        Returns:
            Audio bytes (MP3 format) or None if failed
        """
        text, is_ssml = self.get_chapter_tts_content(chapter)

        if not text:
            logger.warning("Chapter has no content for TTS")
            return None

        # Log which content version we're using
        content_type = "SSML (voice-directed)" if is_ssml else "raw prose"
        logger.info(f"📖 Generating chapter audio from {content_type}")

        return await self.text_to_speech(
            text=text,
            language_code=language_code,
            voice_name=voice_name,
            speaking_rate=speaking_rate,
            pitch=pitch,
            use_ssml=is_ssml,
            use_case="narration"  # Chapters are always narration
        )

    def encode_audio_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string for WebSocket transmission"""
        return base64.b64encode(audio_bytes).decode('utf-8')

    def decode_audio_base64(self, audio_base64: str) -> bytes:
        """Decode base64 audio string to bytes"""
        return base64.b64decode(audio_base64)


# Global voice service instance (lazy initialized)
_voice_service_instance = None

class VoiceServiceProxy:
    """Proxy to lazily initialize voice service after env vars are set"""
    def __getattr__(self, name):
        global _voice_service_instance
        if _voice_service_instance is None:
            _voice_service_instance = VoiceService()
        return getattr(_voice_service_instance, name)

voice_service = VoiceServiceProxy()
