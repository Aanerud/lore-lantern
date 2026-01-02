"""
SSML Processor Service

⚠️  DEPRECATED: This module is deprecated now that ElevenLabs is the primary TTS provider.
ElevenLabs handles expressiveness natively without needing SSML markup.
This code is kept for backwards compatibility with Google Cloud TTS fallback.

Converts narrative text into rich SSML (Speech Synthesis Markup Language)
for enhanced text-to-speech output with prosody, pauses, and emphasis.

OpenAI TTS supports a subset of SSML tags:
- <speak> wrapper
- <break> for pauses
- <emphasis> for stress
- <prosody> for pitch/rate/volume
- <say-as> for specific interpretations

Note: Some advanced SSML features (like Amazon Polly's whisper effect)
are not supported by OpenAI TTS. We use compatible alternatives.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DialogueSegment:
    """Represents a piece of dialogue with speaker context"""
    text: str
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    is_question: bool = False
    is_exclamation: bool = False


class SSMLProcessor:
    """
    Convert narrative text to SSML for rich TTS output.

    Features:
    - Dialogue detection with pitch/rate variations
    - Dramatic pauses after key sentences
    - Emphasis on character names and important words
    - Age-appropriate pacing adjustments
    - XML character escaping for SSML compatibility
    """

    # Characters that must be escaped in SSML (XML standard)
    @staticmethod
    def escape_for_ssml(text: str) -> str:
        """
        Escape special characters for SSML/XML compatibility.

        Required by Speechify and other SSML parsers.
        Order matters: & must be first to avoid double-escaping.

        Note: We don't escape quotes (" and ') in text content because:
        1. They're only required in attribute values
        2. Escaping them would break dialogue detection patterns
        """
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    # Emotion keywords that suggest dialogue tone
    EMOTION_MARKERS = {
        'whispered': ('pitch', '-15%', 'rate', '85%'),
        'shouted': ('pitch', '+20%', 'rate', '110%'),
        'yelled': ('pitch', '+20%', 'rate', '115%'),
        'cried': ('pitch', '+10%', 'rate', '95%'),
        'muttered': ('pitch', '-10%', 'rate', '90%'),
        'exclaimed': ('pitch', '+15%', 'rate', '105%'),
        'asked': ('pitch', '+5%', 'rate', '100%'),
        'demanded': ('pitch', '+10%', 'rate', '105%'),
        'pleaded': ('pitch', '+5%', 'rate', '90%'),
        'declared': ('pitch', '+5%', 'rate', '95%'),
        'laughed': ('pitch', '+10%', 'rate', '105%'),
        'sighed': ('pitch', '-5%', 'rate', '85%'),
        'gasped': ('pitch', '+15%', 'rate', '120%'),
    }

    # Words that deserve emphasis in storytelling
    EMPHASIS_WORDS = {
        'suddenly', 'finally', 'never', 'always', 'forever',
        'secret', 'mysterious', 'ancient', 'powerful', 'magical',
        'dangerous', 'incredible', 'impossible', 'amazing', 'terrible',
        'first', 'last', 'only', 'greatest', 'bravest',
    }

    # Dramatic sentence endings that warrant pauses
    DRAMATIC_ENDINGS = [
        r'\.{3}$',  # Ellipsis
        r'!$',       # Exclamation
        r'\?$',      # Question
        r'—$',       # Em dash
    ]

    # Patterns for dramatic sentences (warrant longer pauses)
    DRAMATIC_PATTERNS = [
        r'And then[,.]',
        r'But suddenly[,.]',
        r'At that moment[,.]',
        r'Without warning[,.]',
        r'In an instant[,.]',
        r'To .+ surprise[,.]',
        r'Little did .+ know[,.]',
    ]

    def __init__(self, character_names: List[str] = None):
        """
        Initialize the SSML processor.

        Args:
            character_names: List of character names to emphasize
        """
        self.character_names = character_names or []
        logger.info(f"SSMLProcessor initialized with {len(self.character_names)} character names")

    def process_chapter(self, text: str, target_age: int = 8) -> str:
        """
        Convert chapter text to SSML with all enhancements.

        Args:
            text: Raw chapter text
            target_age: Child's age (affects pacing)

        Returns:
            SSML-formatted string
        """
        logger.info(f"Processing chapter for SSML (age: {target_age}, length: {len(text)} chars)")

        # Apply transformations in order (order matters to avoid tag interference!)
        processed = text

        # 0. Escape special XML characters FIRST (before any tags are added)
        # This prevents 400 errors from Speechify and other SSML parsers
        processed = self.escape_for_ssml(processed)

        # 1. Process dialogue FIRST (before other tags are added)
        processed = self._add_dialogue_prosody(processed)

        # 2. Add dramatic pauses (affects sentence endings)
        processed = self._add_dramatic_pauses(processed, target_age)

        # 3. Add emphasis to character names and key words LAST
        # (so we don't match text inside prosody attributes)
        processed = self._add_emphasis(processed)

        # 4. Adjust overall pacing for age
        processed = self._adjust_pacing_for_age(processed, target_age)

        # 5. Wrap in speak tags
        ssml = f'<speak>\n{processed}\n</speak>'

        logger.info(f"SSML processing complete (output: {len(ssml)} chars)")
        return ssml

    def _add_dialogue_prosody(self, text: str) -> str:
        """
        Detect dialogue and add pitch/rate changes.

        Handles patterns like:
        - "Hello!" said the Viking.
        - The Viking said, "Hello!"
        - "Hello," she whispered, "I'm here."

        Returns:
            Text with prosody tags around dialogue
        """
        # Pattern for dialogue: "text" followed by speech verb
        # This regex captures: "dialogue" verb_phrase
        dialogue_pattern = r'"([^"]+)"(\s*,?\s*)(\w+(?:\s+\w+)?(?:\s*,)?)'

        def replace_dialogue(match):
            dialogue = match.group(1)
            separator = match.group(2)
            attribution = match.group(3).lower()

            # Determine emotion from attribution
            pitch = '+5%'  # Default slight pitch increase for dialogue
            rate = '100%'

            for marker, adjustments in self.EMOTION_MARKERS.items():
                if marker in attribution:
                    pitch = adjustments[1]
                    rate = adjustments[3]
                    break

            # Check if question or exclamation
            if dialogue.endswith('?'):
                pitch = '+10%'  # Questions go up
            elif dialogue.endswith('!'):
                rate = '105%'  # Exclamations slightly faster

            # Build prosody-wrapped dialogue
            prosody_dialogue = f'<prosody pitch="{pitch}" rate="{rate}">"{dialogue}"</prosody>'

            return f'{prosody_dialogue}{separator}{match.group(3)}'

        processed = re.sub(dialogue_pattern, replace_dialogue, text)

        # Also handle dialogue at start: "Hello!" The Viking smiled.
        start_dialogue_pattern = r'^"([^"]+)"'

        def replace_start_dialogue(match):
            dialogue = match.group(1)
            pitch = '+5%'
            if dialogue.endswith('?'):
                pitch = '+10%'
            return f'<prosody pitch="{pitch}">"{dialogue}"</prosody>'

        # Apply to each paragraph
        paragraphs = processed.split('\n')
        processed_paragraphs = []
        for para in paragraphs:
            if para.strip().startswith('"'):
                para = re.sub(start_dialogue_pattern, replace_start_dialogue, para)
            processed_paragraphs.append(para)

        return '\n'.join(processed_paragraphs)

    def _add_dramatic_pauses(self, text: str, target_age: int = 8) -> str:
        """
        Add breaks after dramatic sentences.

        Pause durations vary by age:
        - Younger: Longer pauses for comprehension
        - Older: Shorter, more natural pauses

        Returns:
            Text with break tags inserted
        """
        # Calculate pause duration based on age
        if target_age <= 6:
            short_pause = '600ms'
            long_pause = '900ms'
        elif target_age <= 9:
            short_pause = '500ms'
            long_pause = '750ms'
        else:
            short_pause = '400ms'
            long_pause = '600ms'

        processed = text

        # Add pauses after dramatic patterns
        for pattern in self.DRAMATIC_PATTERNS:
            processed = re.sub(
                pattern,
                lambda m: f'{m.group(0)}<break time="{long_pause}"/>',
                processed,
                flags=re.IGNORECASE
            )

        # Add pauses after ellipsis
        processed = re.sub(
            r'\.{3}',
            f'...<break time="{long_pause}"/>',
            processed
        )

        # Add short pauses after exclamations and questions (if not in dialogue)
        # Only match those not inside quotes
        processed = re.sub(
            r'([^"])\!(\s+[A-Z])',
            f'\\1!<break time="{short_pause}"/>\\2',
            processed
        )
        processed = re.sub(
            r'([^"])\?(\s+[A-Z])',
            f'\\1?<break time="{short_pause}"/>\\2',
            processed
        )

        # Add pause after paragraph breaks (natural reading pause)
        processed = re.sub(
            r'\n\n',
            f'\n<break time="{short_pause}"/>\n',
            processed
        )

        return processed

    def _add_emphasis(self, text: str) -> str:
        """
        Add emphasis tags to character names and key words.

        Returns:
            Text with emphasis tags
        """
        processed = text

        def is_inside_tag(full_text: str, match_start: int) -> bool:
            """Check if match position is inside an XML tag (between < and >)"""
            # Look backwards for < or >
            before = full_text[:match_start]
            last_open = before.rfind('<')
            last_close = before.rfind('>')
            # If last < is after last >, we're inside a tag
            return last_open > last_close

        # Emphasize character names (moderate emphasis)
        # Process each name one at a time, rebuilding text after each
        for name in self.character_names:
            if name and len(name) > 1:
                pattern = rf'\b({re.escape(name)})\b'
                result_parts = []
                last_end = 0

                for match in re.finditer(pattern, processed, flags=re.IGNORECASE):
                    # Add text from last_end to this match
                    result_parts.append(processed[last_end:match.start()])

                    # Check if this match is inside a tag
                    if not is_inside_tag(processed, match.start()):
                        # Wrap in emphasis
                        result_parts.append(f'<emphasis level="moderate">{match.group(1)}</emphasis>')
                    else:
                        # Keep original text (inside a tag, don't modify)
                        result_parts.append(match.group(0))

                    last_end = match.end()

                # Add remaining text
                result_parts.append(processed[last_end:])
                processed = ''.join(result_parts)

        # Emphasize dramatic words (strong emphasis, but only first occurrence)
        for word in self.EMPHASIS_WORDS:
            pattern = rf'\b({word})\b'

            # Find first match not inside a tag
            for match in re.finditer(pattern, processed, flags=re.IGNORECASE):
                if not is_inside_tag(processed, match.start()):
                    # Replace only this occurrence
                    before = processed[:match.start()]
                    after = processed[match.end():]
                    processed = f'{before}<emphasis level="strong">{match.group(1)}</emphasis>{after}'
                    break  # Only first occurrence

        return processed

    def _adjust_pacing_for_age(self, text: str, target_age: int) -> str:
        """
        Wrap entire text in age-appropriate pacing.

        Younger children benefit from slightly slower narration.

        Returns:
            Text wrapped in prosody tag if needed
        """
        # Calculate reading rate based on age
        if target_age <= 5:
            rate = '85%'
        elif target_age <= 7:
            rate = '90%'
        elif target_age <= 9:
            rate = '95%'
        else:
            rate = '100%'  # Default speed for older kids

        if rate != '100%':
            return f'<prosody rate="{rate}">\n{text}\n</prosody>'

        return text

    def set_character_names(self, names: List[str]):
        """
        Update the list of character names to emphasize.

        Args:
            names: List of character names from the story
        """
        # Filter out empty names and very short names
        self.character_names = [n for n in names if n and len(n) > 1]
        logger.info(f"Updated character names: {self.character_names}")

    def process_dialogue_only(self, text: str) -> str:
        """
        Process only the dialogue portions (for CompanionAgent responses).

        This is a lighter version for real-time responses.

        Returns:
            SSML-formatted string
        """
        processed = self._add_dialogue_prosody(text)
        return f'<speak>{processed}</speak>'

    def extract_dialogue_segments(self, text: str) -> List[DialogueSegment]:
        """
        Extract all dialogue segments from text for analysis.

        Useful for understanding the dialogue distribution in a chapter.

        Returns:
            List of DialogueSegment objects
        """
        segments = []

        # Pattern: "dialogue" attribution
        pattern = r'"([^"]+)"\s*,?\s*(\w+(?:\s+\w+)*)?'

        for match in re.finditer(pattern, text):
            dialogue = match.group(1)
            attribution = match.group(2) or ''

            # Detect emotion from attribution
            emotion = None
            for marker in self.EMOTION_MARKERS:
                if marker in attribution.lower():
                    emotion = marker
                    break

            segment = DialogueSegment(
                text=dialogue,
                speaker=attribution.split()[0] if attribution else None,
                emotion=emotion,
                is_question=dialogue.endswith('?'),
                is_exclamation=dialogue.endswith('!')
            )
            segments.append(segment)

        return segments


# Singleton instance for easy import
_ssml_processor: Optional[SSMLProcessor] = None


def get_ssml_processor(character_names: List[str] = None) -> SSMLProcessor:
    """
    Get or create the global SSMLProcessor instance.

    Args:
        character_names: Optional list of character names to emphasize

    Returns:
        SSMLProcessor instance
    """
    global _ssml_processor

    if _ssml_processor is None:
        _ssml_processor = SSMLProcessor(character_names or [])
    elif character_names:
        _ssml_processor.set_character_names(character_names)

    return _ssml_processor
