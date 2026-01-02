# WebSocket Real-Time Voice Storytelling System

**Version:** 2.0.0
**Last Updated:** December 2025

## 🎉 What's Been Built

This system solves the critical problem: **"2-4-6-8 year olds don't have time to wait 10 minutes"**

The storyteller now keeps the conversation flowing in real-time while agents work in the background!

---

## 🔊 TTS Provider Architecture

### Why Multiple TTS Providers?

Different use cases have different requirements:

| Use Case | Provider | Model | Rationale |
|----------|----------|-------|-----------|
| **Story Narration** | ElevenLabs | eleven_v3 | Best expressiveness with [audio tags], multilingual, high quality |
| **Narration Fallback** | Speechify | simba-multilingual | Good Norwegian (nb-NO beta), reliable backup |
| **Companion Dialogue** | OpenAI | gpt-4o-mini-tts | Fast response, steerable voice instructions, child-friendly |
| **Direct Audio** | OpenAI | gpt-4o-mini-audio-preview | Single API call for text+audio, 50% latency reduction |

### Provider Priority Chain

```
Narration (chapters):
  1. ElevenLabs (if USE_ELEVENLABS=true and API key available)
  2. Speechify (fallback if ElevenLabs fails/quota exceeded)
  3. OpenAI tts-1 (emergency fallback)

Dialogue (companion responses):
  1. Direct Audio (if use_direct_audio=true) → Single LLM+TTS call
  2. OpenAI gpt-4o-mini-tts (separate TTS call after LLM)
```

### ElevenLabs - Premium Story Narration

**Why ElevenLabs for narration:**
- **Audio Tags**: Supports expressive markers like `[whispers]`, `[laughing]`, `[excited]`
- **VoiceDirectorAgent**: Adds these tags to prose for dramatic effect
- **eleven_v3 Model**: Most expressive model, handles 74 languages
- **Language Code**: Explicit `language_code` ensures Norwegian sounds Norwegian (not Danish!)

**Configuration (`.env`):**
```
ELEVENLABS_API_KEY=sk_xxxxx
USE_ELEVENLABS=true  # Enable for premium narration
```

**Per-language voice IDs (`src/config/settings.py`):**
```python
elevenlabs_voices = {
    "en": "7p1Ofvcwsv7UBPoFNcpI",  # Custom English narrator
    "no": "xF681s0UeE04gsf0mVsJ",  # Norwegian narrator
    "es": "21m00Tcm4TlvDq8ikWAM",  # Spanish narrator
}
```

### Speechify - Reliable Fallback

**Why Speechify as fallback:**
- **Norwegian Support**: nb-NO in beta, but functional
- **API Stability**: Reliable when ElevenLabs quota exceeded
- **simba-multilingual**: Good for non-English content

**Configuration (`.env`):**
```
SPEECHIFY_API_KEY=xxxxx
```

**Voice selection logic:**
```python
voice_map = {
    "nb": "vegard",   # Norwegian - clearer accent
    "no": "vegard",
    "en": "oliver",   # English narrator
}
```

### OpenAI TTS - Companion Dialogue

**Why OpenAI for dialogue:**
- **Speed**: gpt-4o-mini-tts responds quickly for real-time conversation
- **Steerable**: Accepts `instructions` parameter for voice style
- **Child-Friendly**: "nova" voice is warm and friendly

**Direct Audio Mode (Recommended):**
- Uses `gpt-4o-mini-audio-preview`
- Single API call generates BOTH text response AND audio
- 50% latency reduction compared to LLM → TTS pipeline
- Perfect for CompanionAgent real-time responses

**Configuration (`src/config/settings.py`):**
```python
use_direct_audio = True  # Enable single-call mode
direct_audio_model = "gpt-4o-mini-audio-preview"
direct_audio_voice = "nova"  # Warm, friendly
```

### Important: Foundry Does NOT Support TTS

Microsoft Azure AI Foundry Model Router is for text-based LLM operations only.
TTS remains as direct API calls to:
- ElevenLabs (external)
- Speechify (external)
- OpenAI Audio API (direct)

## 🏗️ Architecture

### Backend Components

1. **Event System** (`src/services/events.py`)
   - Event emitter pattern for story progress
   - Per-story event queues for WebSocket connections
   - Events: `dialogue_ready`, `structure_ready`, `character_ready`, `chapter_ready`

2. **WebSocket Endpoint** (`src/api/websocket.py`)
   - Route: `ws://localhost:3000/ws/story/{story_id}`
   - Bidirectional communication
   - Handles voice input and streams events with audio

3. **Voice Processing** (`src/services/voice.py`)
   - Speech-to-Text: OpenAI Whisper or Google Cloud
   - Text-to-Speech: ElevenLabs (narration), Speechify (fallback), OpenAI (dialogue)
   - Direct Audio: gpt-4o-mini-audio-preview for single-call LLM+TTS
   - Base64 audio encoding for WebSocket transmission

4. **Real-Time Commentary** (`src/crew/coordinator.py`)
   - DialogueAgent generates live commentary as agents complete work
   - Events emitted at every milestone:
     - Initial response (5 seconds) ✅
     - Structure complete: "I've got 5 exciting chapters!"
     - Each character: "Meet Harald's father, King Halfdan!"
     - Each chapter: "Chapter 1 is ready!"

### Frontend Components

1. **WebSocket Client** (`static/js/voice-websocket.js`)
   - Connects to WebSocket endpoint
   - Handles audio streaming (speech-to-text and text-to-speech)
   - Event display and logging

2. **Test Interface** (`static/ws-test.html`)
   - Full-featured test page
   - Voice recording controls
   - Real-time event log
   - Connection status monitoring

## 📋 Dependencies

Key voice-related dependencies in `requirements.txt`:
- `elevenlabs` - ElevenLabs TTS SDK
- `openai` - OpenAI TTS and Whisper
- `websockets` - WebSocket support
- `httpx` - Async HTTP for Speechify API
- All existing dependencies (FastAPI, CrewAI, LiteLLM, etc.)

## 🚀 How It Works - The Flow

### Initial Request (Immediate - ~5 seconds)
```
User: "Tell me a story about Vikings!"
   ↓
DialogueAgent responds immediately
   ↓
WebSocket emits: dialogue_ready + audio
   ↓
Child hears: "Oh wow! Vikings! What an exciting choice! Did you know Vikings
sailed all the way to North America over a thousand years ago? I'm so excited
to create this adventure!"
```

### Structure Phase (~30 seconds)
```
StructureAgent creates outline
   ↓
Event: structure_ready (5 chapters, 8 characters)
   ↓
DialogueAgent generates commentary
   ↓
Child hears: "I've planned out 5 thrilling chapters with 8 amazing characters!
This is going to be epic!"
```

### Character Phase (~1-2 minutes)
```
CharacterAgent creates each character
   ↓
Event: character_ready (for each)
   ↓
DialogueAgent generates introduction
   ↓
Child hears: "Meet Harald Fairhair, a young and impulsive prince who dreams
of uniting Norway!"
Child hears: "Here comes King Halfdan, Harald's wise father who rules with
both strength and kindness!"
```

### Chapter Phase (~2-3 minutes per chapter)
```
NarrativeAgent writes Chapter 1
   ↓
FactCheckAgent verifies facts
   ↓
Event: chapter_ready
   ↓
DialogueAgent generates hook
   ↓
Child hears: "Chapter 1 is ready! You won't believe what happens when young
Harald first picks up a sword!"
```

**Total time from request to first chapter: ~3-5 minutes**
**But the child is engaged THE ENTIRE TIME with real-time commentary!**

## 🧪 Testing the System

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure TTS providers in `.env`:**
   ```bash
   # ElevenLabs (primary narration)
   ELEVENLABS_API_KEY=sk_xxxxx
   USE_ELEVENLABS=true

   # Speechify (fallback narration)
   SPEECHIFY_API_KEY=xxxxx

   # OpenAI (dialogue + direct audio)
   OPENAI_API_KEY=sk-xxxxx
   ```

### Test Steps

1. **Start the server:**
   ```bash
   python3 -m src.main
   ```

2. **Open the WebSocket test page:**
   ```
   http://localhost:3000/static/ws-test.html
   ```

3. **Test the flow:**
   - Enter a story request (default: "Tell me a story about Harald Hårfagre!")
   - Click "Create Story & Connect"
   - Watch the real-time events stream in!
   - Use the voice recording button to test speech-to-text

### Expected Behavior

You should see events arrive in this order:
1. ✅ WebSocket connected
2. 🗣️ Narrator: Initial dialogue response (with audio)
3. 📚 Story structure ready (with commentary)
4. 🎭 Character introduced (multiple times, with commentary)
5. 📖 Chapter ready (with commentary)

Each narrator commentary includes:
- Text display in the UI
- Audio playback (via ElevenLabs/Speechify for narration, OpenAI for dialogue)
- Event log entry with timestamp

## 🎤 Voice Features

### Speech-to-Text
- Records user voice input via microphone
- Converts to text using OpenAI Whisper or Google Cloud Speech API
- Supports multiple languages (configurable)

### Text-to-Speech Providers

#### ElevenLabs (Story Narration)
- **Model**: eleven_v3 (most expressive)
- **Audio Tags**: `[whispers]`, `[laughing]`, `[excited]`, `[sad]`
- **VoiceDirectorAgent** adds these for dramatic storytelling
- **Languages**: Full Norwegian support with explicit language_code
- **Output**: mp3_44100_128 quality

#### Speechify (Fallback Narration)
- **Model**: simba-multilingual
- **Norwegian**: nb-NO in beta, good quality
- **Use Case**: When ElevenLabs quota exceeded

#### OpenAI TTS (Companion Dialogue)
- **Model**: gpt-4o-mini-tts
- **Voice**: nova (warm, child-friendly)
- **Steerable**: Instructions parameter for voice style
- **Speed**: Optimized for real-time conversation

#### Direct Audio (Recommended for Dialogue)
- **Model**: gpt-4o-mini-audio-preview
- **Benefit**: Single API call for LLM response + audio generation
- **Latency**: ~50% reduction vs separate LLM→TTS calls
- **Use Case**: CompanionAgent real-time responses

### Audio Streaming
- All audio sent as base64-encoded MP3 over WebSocket
- Browser decodes and plays audio automatically
- Audio stored in Azure Blob Storage for replay

---

## 🎭 VoiceDirectorAgent

The VoiceDirectorAgent transforms plain prose into expressive narration by adding ElevenLabs audio tags.

### How It Works

1. **NarrativeAgent** writes the chapter prose
2. **VoiceDirectorAgent** reads the prose and adds audio tags
3. **ElevenLabs eleven_v3** interprets the tags for dramatic effect

### Audio Tag Examples

```
Original:  "The dragon roared and breathed fire."
Enhanced:  "[dramatic pause] The dragon [growling] roared [/growling] and breathed fire."

Original:  "She whispered the secret password."
Enhanced:  "[whispers] She whispered the secret password. [/whispers]"

Original:  "Everyone laughed at the funny joke."
Enhanced:  "[laughing] Everyone laughed at the funny joke! [/laughing]"
```

### Supported ElevenLabs Tags

| Tag | Effect | Use Case |
|-----|--------|----------|
| `[whispers]...[/whispers]` | Soft, hushed speech | Secrets, suspense |
| `[laughing]...[/laughing]` | Speech with laughter | Humor, joy |
| `[dramatic pause]` | Significant pause | Tension, revelation |
| `[excited]...[/excited]` | Energetic delivery | Action, discovery |
| `[sad]...[/sad]` | Somber tone | Emotional moments |
| `[scary]...[/scary]` | Ominous delivery | Spooky scenes |

### Why This Matters

- **Engagement**: Audio tags add emotion that plain TTS lacks
- **Storytelling**: Proper pauses and emphasis match written intent
- **Age-Appropriate**: Tailored to keep young listeners engaged
- **Differentiation**: Better than generic TTS narration

## 📁 Files Created/Modified

### New Files
- `src/services/events.py` - Event emitter system
- `src/services/voice.py` - Voice processing service
- `src/api/websocket.py` - WebSocket endpoint
- `static/js/voice-websocket.js` - Frontend WebSocket client
- `static/ws-test.html` - Test interface
- `requirements.txt` - Python dependencies

### Modified Files
- `src/main.py` - Added WebSocket router
- `src/crew/coordinator.py` - Added event emissions and commentary generation

## 🎯 Key Benefits

1. **No more silent waiting!** DialogueAgent narrates progress in real-time
2. **Universal device support** - WebSockets work on browsers, HomePod, Alexa, etc.
3. **Voice-first interface** - Full speech-to-text and text-to-speech
4. **Progressive enhancement** - Story elements appear as they're ready
5. **Fact-checked before presentation** - Only verified content reaches the child

## 🔮 What's Next

The core system is complete! Remaining optional enhancements:

1. **Auto-Continue with Pauses** - After chapter commentary, automatically wait 3-5
   seconds, then start reading the chapter content

2. **Interrupt Handling** - Allow child to interrupt storytelling with questions

3. **Multi-Chapter Narration** - After chapter 1 completes, automatically generate
   chapters 2-5 with real-time updates

4. **Voice Activity Detection** - Detect when child stops speaking to process input

## 🐛 Troubleshooting

### WebSocket won't connect
- Check server is running: `http://localhost:3000/api/health`
- Verify WebSocket URL: `ws://localhost:3000/ws/story/{story_id}`
- Check browser console for errors

### No audio playback
- TTS API key not configured → Check ELEVENLABS_API_KEY, SPEECHIFY_API_KEY, or OPENAI_API_KEY
- Browser audio context not initialized → Click page first to enable audio
- Check browser audio permissions
- Check console for TTS provider fallback messages

### Events not arriving
- Check coordinator is emitting events (look for log messages)
- Verify WebSocket connection status in test UI
- Check event log for connection errors

## 🎓 Technical Details

### WebSocket Message Format

**Client → Server:**
```json
{
  "type": "audio_chunk",
  "audio": "base64_encoded_audio"
}
```

**Server → Client:**
```json
{
  "type": "dialogue_ready",
  "story_id": "uuid",
  "data": {
    "message": "Narrator text",
    "audio": "base64_encoded_mp3",
    "phase": "initialization"
  },
  "timestamp": "2025-10-27T..."
}
```

### Event Types

- `connection_established` - WebSocket connected
- `dialogue_ready` - Narrator commentary with audio
- `structure_ready` - Story outline complete
- `character_ready` - Character created
- `chapter_ready` - Chapter written and verified
- `transcript` - Speech-to-text result
- `pong` - Ping response

## 🎉 Success Criteria

✅ Child hears response within 5 seconds of request
✅ Continuous engagement throughout story creation (3-5 minutes)
✅ Real-time narrator commentary at every milestone
✅ Voice input/output works seamlessly
✅ Works on all platforms (browser, voice assistants)
✅ Only fact-checked content presented

**Mission accomplished!** The 2-8 year olds won't have to wait anymore. 🚀
