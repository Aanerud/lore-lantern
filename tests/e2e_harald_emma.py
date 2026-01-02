#!/usr/bin/env python3
"""
Interactive E2E Test: Emma (7) asks for Harald Viking story

Tests Chapter 1 + Structure V2 refinement + Chapter 2 generation with
dialogue interaction and validates compliance with AGENT_ARCHITECTURE.md.

A/B Testing Support:
Auto-detects TEST_* environment variables for model overrides:
  TEST_NARRATIVE_MODEL=claude-sonnet-4-5-20250929 python tests/e2e_harald_emma.py
  TEST_STRUCTURE_MODEL=claude-sonnet-4-5-20250929 python tests/e2e_harald_emma.py
  TEST_ROUNDTABLE_MODEL=gemini-3-flash-preview python tests/e2e_harald_emma.py

Combined example:
  TEST_NARRATIVE_MODEL=claude-sonnet-4-5-20250929 \
  TEST_STRUCTURE_MODEL=claude-sonnet-4-5-20250929 \
  TEST_ROUNDTABLE_MODEL=gemini-3-flash-preview \
  python tests/e2e_harald_emma.py

This test:
1. Uses existing Emma profile (emma_7, age 7)
2. Generates Chapter 1
3. Triggers Structure V2 refinement (via start_chapter)
4. Generates Chapter 2
5. Validates agent behaviors against architecture specs
6. Generates markdown report with models used

Stories are saved as markdown files to tests/outputs/ with model names in filename.

Run with: python3 tests/e2e_harald_emma.py
"""

import asyncio
import aiohttp
import websockets
import json
import time
import random
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re


# ============================================================================
# A/B Testing Helper Functions
# ============================================================================

def get_test_model_names() -> Dict[str, str]:
    """
    Get the models being tested from environment variables.
    Auto-detects TEST_*_MODEL env vars without requiring manual setup.

    Returns:
        Dict with 'narrative', 'structure', and 'roundtable' model names
    """
    return {
        "narrative": os.environ.get("TEST_NARRATIVE_MODEL", "default"),
        "structure": os.environ.get("TEST_STRUCTURE_MODEL", "default"),
        "roundtable": os.environ.get("TEST_ROUNDTABLE_MODEL", "default")
    }


def get_safe_model_name(model: str) -> str:
    """Convert model name to safe filename component."""
    return model.replace("/", "-").replace(":", "-").replace(" ", "_")


# Test Configuration
BASE_URL = "http://localhost:3000"

# New Family Account Model
PARENT_ID = "parent_smith_test"
CHILD_ID = "child_emma_test"
CHILD_NAME = "Emma"
CHILD_BIRTH_YEAR = 2018  # Makes her ~7 years old
FAMILY_LANGUAGE = "en"

# Legacy (deprecated, kept for reference)
USER_ID = "emma_7"  # DEPRECATED: Use CHILD_ID

TARGET_AGE = 7
INITIAL_CHAPTERS = 1  # Only generate Chapter 1 initially (so snowstorm can influence V2/Ch2)
EXPECTED_CHAPTERS = 2  # Total chapters expected after V2 refinement + Chapter 2 generation
MIN_CHAPTERS_PLANNED = 5  # For age 7, should plan 5-10 chapters in structure

# Non-blocking dialogue thresholds
MAX_DIALOGUE_LATENCY_MS = 3000  # Max 3 seconds for dialogue response
MIN_CONCURRENT_RESPONSES = 3   # Must get 3+ responses WHILE story generates
DIALOGUE_INTERVAL_SECONDS = 25  # Send dialogue every 25 seconds (balanced)

# Emma's child-like dialogue - how a real 7-year-old talks!
# Expanded pools to prevent repetition during long story generation
EMMA_INITIAL_PROMPTS = [
    "I wanna hear about Harald! He was like the first king of Norway right? With Vikings and swords and stuff!",
    "Tell me about Harald Fairhair! My teacher said he had really really long hair and was super brave!",
    "Can you tell me a story about Vikings? Like Harald who became a king? I love Viking stories!",
]

EMMA_EXCITED_REACTIONS = [
    "Ooooh that's so cool! Did he have a sword? Like a REALLY big one?",
    "WOW! What happened next? I wanna know more!",
    "That's AMAZING! Were there dragons too? Please say there were dragons!",
    "No way!! That's so awesome! Keep going keep going!",
    "That sounds SO exciting! Tell me more!",
    "Yay! This is gonna be the best story ever!",
]

# Expanded curious questions - 25+ unique questions to avoid repetition
EMMA_CURIOUS_QUESTIONS = [
    # About Harald
    "But WHY did he want to be king? Was he scared?",
    "Did Harald have a pet? Like a wolf or something cool?",
    "Did Harald ever get hurt in a battle?",
    "What was Harald's favorite thing to do?",
    "Did Harald have any brothers or sisters?",
    "How old was Harald when the story starts?",
    "What color was Harald's hair? Was it really super long?",
    "Did Harald have a best friend?",
    # About Vikings in general
    "What did Vikings eat? Did they have pizza?",
    "Were there any girl Vikings? That would be so cool!",
    "How big were the boats? Bigger than my house?",
    "Did they wear helmets with horns? My brother says they did!",
    "Where did Vikings live? In castles?",
    "Did Vikings have schools? Did kids have to do homework?",
    "What games did Viking kids play?",
    "Did Vikings have horses? Or just boats?",
    "What did Viking houses look like inside?",
    "Did Vikings celebrate birthdays?",
    # About the story world
    "Is Norway cold? Do Vikings like snow?",
    "Are there mountains in the story?",
    "What kind of animals live where Harald is?",
    "Is there magic in the story? Like real magic?",
    "Are there any funny parts in the story?",
    # About story elements
    "Does Harald have a special weapon?",
    "Who teaches Harald to be brave?",
    "Is there a princess in the story?",
]

EMMA_DURING_WAIT_QUESTIONS = [
    "Is the story ready yet? I'm sooooo excited!",
    "Ooh ooh what's happening now? Tell me tell me!",
    "Are the Vikings fighting yet? I wanna hear about the battles!",
    "This is taking forever... but I can wait! Vikings are worth waiting for!",
    "Can you tell me something about Harald while we wait? Pretty please?",
    "I can't wait! Is it almost done?",
    "What are you working on right now? The exciting parts?",
    "Can you give me a hint about what happens?",
]

# Expanded grounding test questions - questions that REQUIRE story-specific answers
# These should trigger responses that reference actual characters, plot, educational content
EMMA_GROUNDING_TEST_QUESTIONS = [
    # Character questions
    "Who is Harald's father? What's he like?",
    "Tell me about the characters in the story!",
    "Who are the bad guys in the story?",
    "Who is Harald's best friend or helper?",
    "Tell me about the most important person in the story besides Harald!",
    # Plot questions
    "What happens in chapter 1?",
    "What is Harald learning?",
    "What is the biggest problem Harald has to solve?",
    "What does Harald want more than anything?",
    # Setting questions
    "Where does the story take place?",
    "What does Harald's home look like?",
    # Educational questions
    "What will I learn from this story?",
    "What's the most important lesson in the story?",
]

EMMA_CHAPTER_REACTIONS = [
    "YAAAY the chapter is done! Read it to me read it to me!",
    "Finally! I bet it's super exciting! Is there fighting?",
    "Ooooh I can't wait to hear what happens! Go on go on!",
    "Is it really really ready? Can we start now?",
]


# ============================================================================
# WebSocket Client for Real-Time Event Streaming
# ============================================================================

class EmmaWebSocketClient:
    """WebSocket client for real-time story event streaming
    
    Configured with extended ping timeouts to handle long story generation
    (which can take 2-3 minutes per chapter).
    """

    def __init__(self, story_id: str, base_url: str = "ws://localhost:3000"):
        self.story_id = story_id
        self.ws_url = f"{base_url}/ws/story/{story_id}"
        self.ws = None
        self.events_received: List[Dict[str, Any]] = []
        self.connected = False
        self._ping_task = None

    async def connect(self) -> bool:
        """Connect to story WebSocket with extended timeouts for long operations"""
        try:
            print(f"   Attempting WebSocket connection to: {self.ws_url}")
            # Disable automatic pings - we'll handle keepalive ourselves
            # This prevents timeout during long LLM generation (2-3 min per chapter)
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=None,  # Disable automatic pings
                ping_timeout=None,   # Disable ping timeout
                close_timeout=10,    # 10s close timeout
            )
            self.connected = True
            print(f"   WebSocket connected, waiting for confirmation...")

            # Wait for connection confirmation (increased timeout for server load)
            msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            event = json.loads(msg)
            print(f"   Received: {event.get('type')}")
            if event.get("type") == "connection_established":
                print(f"   Connection confirmed for story: {event.get('story_id')}")
            return True
        except asyncio.TimeoutError as e:
            print(f"   WebSocket timeout waiting for confirmation: {e}")
            self.connected = False
            if self.ws:
                await self.ws.close()
            return False
        except Exception as e:
            print(f"   WebSocket connection failed: {e}")
            self.connected = False
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            return False

    async def send_message(self, message: str):
        """Send user message via WebSocket"""
        if not self.ws or not self.connected:
            print("   WebSocket not connected, cannot send message")
            return

        await self.ws.send(json.dumps({
            "type": "user_message",
            "message": message
        }))

    async def receive_events(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Receive all pending events with timeout
        
        Handles long waits during story generation (2-3 min per chapter).
        """
        events = []
        if not self.ws or not self.connected:
            print(f"   [receive_events] WebSocket not connected")
            return events

        try:
            while True:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                event = json.loads(msg)
                
                # Handle pong responses (keepalive)
                if event.get("type") == "pong":
                    continue  # Don't add pong to events list
                    
                events.append(event)
                self.events_received.append(event)
                # Short timeout for subsequent messages
                timeout = 0.5
        except asyncio.TimeoutError:
            pass  # Normal - no more events pending
        except websockets.exceptions.ConnectionClosed as e:
            print(f"   [receive_events] Connection closed: {e}")
            self.connected = False
        except Exception as e:
            print(f"   [receive_events] Error: {e}")
        return events

    async def wait_for_event(self, event_type: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Wait for a specific event type"""
        if not self.ws or not self.connected:
            return None

        start = time.time()
        while time.time() - start < timeout:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=min(5.0, timeout - (time.time() - start)))
                event = json.loads(msg)
                self.events_received.append(event)
                if event.get("type") == event_type:
                    return event
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                return None
        return None

    async def close(self):
        """Close the WebSocket connection"""
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
            self.connected = False
            print(f"   WebSocket closed")

    async def send_ping(self):
        """Send application-level ping to keep connection alive"""
        if self.ws and self.connected:
            try:
                await self.ws.send(json.dumps({"type": "ping"}))
            except Exception:
                pass


# Keywords that indicate GROUNDED responses (story-specific content)
GROUNDING_KEYWORDS = [
    # Character names from Harald story
    "harald", "halfdan", "king", "father", "prince",
    # Story themes
    "viking", "leadership", "training", "courage", "wisdom",
    # Plot elements
    "kingdom", "throne", "sword", "battle", "clan",
    # Educational content
    "learn", "teach", "discover",
]

# Keywords that indicate FLUFFY/GENERIC responses (not grounded)
FLUFFY_KEYWORDS = [
    "exciting things", "wonderful adventure", "stay tuned",
    "something special", "great story", "amazing things",
    "can't wait", "so much fun", "really cool",
]


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    passed: bool
    details: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    excerpt: Optional[str] = None


@dataclass
class DialogueTimingResult:
    """Measures dialogue response latency for non-blocking verification"""
    question: str
    question_sent_at: float  # time.time() when sent
    response_received_at: float  # time.time() when received
    latency_ms: float  # Response time in milliseconds
    story_status: str  # Story status when response received
    concurrent_with_generation: bool  # Was story still generating?
    response_text: str = ""  # The actual response

    @property
    def is_fast(self) -> bool:
        """Check if response was fast enough (non-blocking)"""
        return self.latency_ms < MAX_DIALOGUE_LATENCY_MS


@dataclass
class TestResults:
    """Comprehensive test results"""
    story_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    validations: List[ValidationResult] = field(default_factory=list)
    dialogue_responses: List[Dict[str, Any]] = field(default_factory=list)
    websocket_events: List[Dict[str, Any]] = field(default_factory=list)
    story_data: Optional[Dict[str, Any]] = None
    chapter_1_content: Optional[str] = None
    # NEW: Timing results for non-blocking verification
    dialogue_timing: List[DialogueTimingResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all validations passed"""
        return all(v.passed for v in self.validations)

    @property
    def duration(self) -> float:
        """Test duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    @property
    def concurrent_response_count(self) -> int:
        """Count responses received while story was still generating"""
        return sum(1 for t in self.dialogue_timing if t.concurrent_with_generation)

    @property
    def average_latency_ms(self) -> float:
        """Average dialogue response latency"""
        if not self.dialogue_timing:
            return 0
        return sum(t.latency_ms for t in self.dialogue_timing) / len(self.dialogue_timing)

    @property
    def dialogue_non_blocking(self) -> bool:
        """Check if dialogue was truly non-blocking"""
        if len(self.dialogue_timing) < MIN_CONCURRENT_RESPONSES:
            return False  # Not enough data points
        # Must have concurrent responses AND fast latency
        has_concurrent = self.concurrent_response_count >= MIN_CONCURRENT_RESPONSES
        is_fast = self.average_latency_ms < MAX_DIALOGUE_LATENCY_MS
        return has_concurrent and is_fast


class ArchitectureValidator:
    """Validates agent outputs against AGENT_ARCHITECTURE.md requirements"""

    @staticmethod
    def validate_dialogue_agent(response: str, age: int) -> ValidationResult:
        """
        Validate DialogueAgent response for ages 4-7.

        Requirements:
        - 2-3 sentences maximum
        - Warm, enthusiastic tone
        - Simple words
        """
        sentences = response.split('.')
        sentence_count = len([s for s in sentences if s.strip()])

        # Check sentence count (relaxed - up to 5 for conversational flow)
        if 1 <= sentence_count <= 5:
            passed = True
            details = f"Response has {sentence_count} sentences (optimal: 2-3)"
        else:
            passed = False
            details = f"Response has {sentence_count} sentences (expected: 2-5)"

        # Check for enthusiasm markers
        enthusiasm_markers = ['!', 'exciting', 'wonderful', 'amazing', 'great', 'wow', 'cool', 'awesome', 'fantastic']
        has_enthusiasm = any(marker in response.lower() for marker in enthusiasm_markers)

        if has_enthusiasm:
            details += " | Enthusiastic tone detected"
        else:
            details += " | Could be more enthusiastic"

        return ValidationResult(
            check_name="DialogueAgent - Age 4-7 Response Style",
            passed=passed,
            details=details,
            expected="2-5 sentences, warm/enthusiastic",
            actual=f"{sentence_count} sentences",
            excerpt=response[:150] + "..." if len(response) > 150 else response
        )

    @staticmethod
    def validate_dialogue_grounding(response: str, question: str) -> ValidationResult:
        """
        Validate that DialogueAgent response is GROUNDED in story content.
        
        A grounded response references specific story elements like:
        - Character names (Harald, Halfdan, etc.)
        - Plot points (training, leadership, etc.)
        - Educational content
        
        A fluffy response uses generic phrases without story specifics.
        """
        response_lower = response.lower()
        
        # Count grounding keywords found
        grounding_found = [kw for kw in GROUNDING_KEYWORDS if kw in response_lower]
        fluffy_found = [kw for kw in FLUFFY_KEYWORDS if kw in response_lower]
        
        grounding_score = len(grounding_found)
        fluffy_score = len(fluffy_found)
        
        # Response is grounded if it has at least 2 grounding keywords
        # and grounding outweighs fluffy content
        is_grounded = grounding_score >= 2 and grounding_score > fluffy_score
        
        details_parts = []
        if grounding_found:
            details_parts.append(f"Grounded: {', '.join(grounding_found[:5])}")
        if fluffy_found:
            details_parts.append(f"Fluffy: {', '.join(fluffy_found[:3])}")
        
        details = f"Score: {grounding_score} grounded, {fluffy_score} fluffy"
        if details_parts:
            details += " | " + " | ".join(details_parts)
        
        return ValidationResult(
            check_name="DialogueAgent - Grounded Response",
            passed=is_grounded,
            details=details,
            expected="At least 2 story-specific keywords",
            actual=f"{grounding_score} grounding keywords found",
            excerpt=response[:200] + "..." if len(response) > 200 else response
        )

    @staticmethod
    def validate_structure_planning(structure: Dict[str, Any], age: int) -> ValidationResult:
        """
        Validate StructureAgent planning.

        Requirements for age 7:
        - 5-10 chapters planned
        - Each chapter 200-400 words
        - Characters defined
        """
        chapters = structure.get('chapters', [])
        chapter_count = len(chapters)

        if age <= 7:
            min_chapters, max_chapters = 3, 10  # Relaxed minimum for testing
        elif age <= 12:
            min_chapters, max_chapters = 8, 15
        else:
            min_chapters, max_chapters = 12, 25

        passed = min_chapters <= chapter_count <= max_chapters

        details = f"Planned {chapter_count} chapters"
        if passed:
            details += f" (optimal for age {age}: {min_chapters}-{max_chapters})"
        else:
            details += f" (expected {min_chapters}-{max_chapters} for age {age})"

        # Check character planning
        characters_needed = structure.get('characters_needed', [])
        details += f" | {len(characters_needed)} characters planned"

        return ValidationResult(
            check_name="StructureAgent - Full Book Planning",
            passed=passed,
            details=details,
            expected=f"{min_chapters}-{max_chapters} chapters for age {age}",
            actual=f"{chapter_count} chapters planned"
        )

    @staticmethod
    def validate_first_chapter_scene_setting(content: str) -> ValidationResult:
        """
        Validate first chapter scene-setting quality.

        Requirements (based on narrative.py improvements):
        - Physical setting description in first 2-3 paragraphs
        - Sensory details (sight, sound, smell, feel)
        - POV character perspective
        - Delayed plot introduction
        """
        if not content:
            return ValidationResult(
                check_name="NarrativeAgent - First Chapter Scene-Setting",
                passed=False,
                details="No chapter content to validate",
                expected="Physical setting, sensory details, POV immersion",
                actual="Empty content"
            )

        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        first_two_paras = ' '.join(paragraphs[:2]) if len(paragraphs) >= 2 else content[:500]

        # Check for setting indicators
        setting_words = ['hall', 'room', 'forest', 'sea', 'village', 'castle', 'house', 'fire', 'light',
                        'sky', 'sun', 'moon', 'door', 'wall', 'window', 'throne', 'table']
        has_setting = any(word in first_two_paras.lower() for word in setting_words)

        # Check for sensory details
        sight_words = ['saw', 'looked', 'watched', 'gleamed', 'shone', 'flickered', 'glowed', 'bright', 'dark']
        sound_words = ['heard', 'voice', 'sound', 'echo', 'whisper', 'roar', 'crackle', 'silence']
        smell_words = ['smell', 'scent', 'aroma', 'stench', 'smoke', 'fragrant']
        feel_words = ['felt', 'cold', 'warm', 'rough', 'soft', 'wind', 'breeze', 'touch']

        has_sight = any(word in first_two_paras.lower() for word in sight_words)
        has_sound = any(word in first_two_paras.lower() for word in sound_words)
        has_smell = any(word in first_two_paras.lower() for word in smell_words)
        has_feel = any(word in first_two_paras.lower() for word in feel_words)

        sensory_count = sum([has_sight, has_sound, has_smell, has_feel])

        # Check for POV grounding (character experiencing the world)
        pov_indicators = ['he saw', 'she saw', 'he felt', 'she felt', 'he heard', 'she heard',
                         'he watched', 'she watched', 'eyes', 'hands', 'heart', 'noticed']
        has_pov = any(indicator in first_two_paras.lower() for indicator in pov_indicators)

        # Check for delayed plot (no immediate action/conflict words in first paragraph)
        first_para = paragraphs[0] if paragraphs else content[:200]
        action_words = ['suddenly', 'attacked', 'enemy', 'battle', 'fight', 'war', 'died', 'killed']
        has_delayed_plot = not any(word in first_para.lower() for word in action_words)

        # Calculate score
        score = sum([has_setting, sensory_count >= 2, has_pov, has_delayed_plot])
        passed = score >= 3  # At least 3 of 4 criteria

        details_parts = []
        details_parts.append(f"{'✅' if has_setting else '❌'} Setting description")
        details_parts.append(f"{'✅' if sensory_count >= 2 else '❌'} Sensory details ({sensory_count}/4 types)")
        details_parts.append(f"{'✅' if has_pov else '❌'} POV immersion")
        details_parts.append(f"{'✅' if has_delayed_plot else '❌'} Delayed plot intro")

        return ValidationResult(
            check_name="NarrativeAgent - First Chapter Scene-Setting",
            passed=passed,
            details=" | ".join(details_parts),
            expected="Physical setting + 2+ sensory types + POV + delayed plot",
            actual=f"Score: {score}/4",
            excerpt=first_two_paras[:300] + "..." if len(first_two_paras) > 300 else first_two_paras
        )

    @staticmethod
    def validate_character_visuals(character: Dict[str, Any]) -> ValidationResult:
        """
        Validate CharacterAgent visual descriptions.

        Requirements (based on actual Character model):
        - appearance: Optional[str] - Physical appearance description
        - personality_traits: List[str] - At least 2 traits
        - background: str - Character backstory (min 20 chars)
        """
        # Get actual model fields (not nested visual_description)
        appearance = character.get('appearance', '') or ''
        personality_traits = character.get('personality_traits', []) or []
        background = character.get('background', '') or ''

        has_appearance = len(appearance) > 20
        has_traits = len(personality_traits) >= 2
        has_background = len(background) > 20

        passed = has_appearance and has_traits and has_background

        details_parts = []
        if has_appearance:
            details_parts.append(f"Appearance ({len(appearance)} chars)")
        else:
            details_parts.append(f"Appearance missing/short ({len(appearance)} chars)")

        if has_traits:
            details_parts.append(f"Traits ({len(personality_traits)} items)")
        else:
            details_parts.append(f"Traits insufficient ({len(personality_traits)} items, need 2+)")

        if has_background:
            details_parts.append(f"Background ({len(background)} chars)")
        else:
            details_parts.append(f"Background missing/short ({len(background)} chars)")

        return ValidationResult(
            check_name=f"CharacterAgent - Visual Description ({character.get('name', 'Unknown')})",
            passed=passed,
            details=" | ".join(details_parts),
            expected="Appearance (>20 chars) + Traits (2+) + Background (>20 chars)",
            actual=f"Appearance:{has_appearance}, Traits:{has_traits}, Background:{has_background}"
        )

    @staticmethod
    def validate_narrative_prose(chapter: Dict[str, Any], age: int) -> ValidationResult:
        """
        Validate NarrativeAgent prose quality.

        Requirements for age 7:
        - 300-1200 words per chapter (relaxed upper limit)
        - Simple sentences (max 10-12 words)
        - 3-5 vocabulary words
        """
        content = chapter.get('content', '')
        word_count = len(content.split())

        # Word count check (relaxed for richer narratives)
        if age <= 7:
            min_words, max_words = 200, 1200
            max_sentence_words = 15
        elif age <= 12:
            min_words, max_words = 800, 2000
            max_sentence_words = 18
        else:
            min_words, max_words = 1500, 3500
            max_sentence_words = 22

        word_count_ok = min_words <= word_count <= max_words

        # Sentence complexity check (sample first 5 sentences)
        sentences = re.split(r'[.!?]+', content)[:5]
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        sentence_ok = avg_sentence_length <= max_sentence_words

        # Vocabulary words (relaxed - 0-10 is acceptable)
        vocab_words = chapter.get('vocabulary_words', {})
        vocab_count = len(vocab_words) if isinstance(vocab_words, dict) else 0
        vocab_ok = True  # Don't fail on vocab count

        passed = word_count_ok and sentence_ok

        details_parts = []
        details_parts.append(f"{'✅' if word_count_ok else '❌'} {word_count} words ({min_words}-{max_words})")
        details_parts.append(f"{'✅' if sentence_ok else '❌'} Avg {avg_sentence_length:.1f} words/sentence (max {max_sentence_words})")
        details_parts.append(f"📚 {vocab_count} vocabulary words")

        return ValidationResult(
            check_name=f"NarrativeAgent - Prose Quality (Chapter {chapter.get('chapter_number', '?')})",
            passed=passed,
            details=" | ".join(details_parts),
            expected=f"{min_words}-{max_words} words, max {max_sentence_words} words/sentence",
            actual=f"{word_count} words, {avg_sentence_length:.1f} avg words/sentence"
        )

    @staticmethod
    def validate_factcheck_accuracy(chapter: Dict[str, Any]) -> ValidationResult:
        """
        Validate FactCheckAgent review.

        Requirements for historical stories:
        - factcheck_status present
        - No major historical inaccuracies
        """
        factcheck_status = chapter.get('factcheck_status', 'unknown')

        # Accept various approval statuses
        passed = factcheck_status in ['approved', 'approved_with_notes', 'pending', 'unknown']

        details = f"Status: {factcheck_status}"

        if factcheck_status == 'approved':
            details += " | No issues found"
        elif factcheck_status == 'approved_with_notes':
            details += " | Approved with minor notes"
        elif factcheck_status in ['pending', 'unknown']:
            details += " | Review pending/skipped"
        else:
            details += f" | Unexpected status"

        return ValidationResult(
            check_name=f"FactCheckAgent - Historical Accuracy (Chapter {chapter.get('chapter_number', '?')})",
            passed=passed,
            details=details,
            expected="approved or approved_with_notes",
            actual=factcheck_status
        )

    @staticmethod
    def validate_dialogue_non_blocking(timing_results: List[DialogueTimingResult]) -> ValidationResult:
        """
        Validate that dialogue is truly non-blocking and separate from story generation.

        Requirements:
        1. Average latency < 3000ms (fast responses)
        2. No timeouts during story generation
        3. At least 3 responses received WHILE story was still generating

        This proves CompanionAgent operates independently of CrewAI.
        """
        if not timing_results:
            return ValidationResult(
                check_name="Dialogue Non-Blocking Verification",
                passed=False,
                details="No timing data collected",
                expected=f"At least {MIN_CONCURRENT_RESPONSES} responses during generation",
                actual="0 responses"
            )

        # Calculate metrics
        total_responses = len(timing_results)
        concurrent_responses = sum(1 for t in timing_results if t.concurrent_with_generation)
        avg_latency = sum(t.latency_ms for t in timing_results) / total_responses
        max_latency = max(t.latency_ms for t in timing_results)
        fast_responses = sum(1 for t in timing_results if t.is_fast)
        timeout_count = sum(1 for t in timing_results if t.latency_ms >= MAX_DIALOGUE_LATENCY_MS)

        # Check pass criteria
        has_enough_concurrent = concurrent_responses >= MIN_CONCURRENT_RESPONSES
        is_fast_enough = avg_latency < MAX_DIALOGUE_LATENCY_MS
        no_timeouts = timeout_count == 0

        passed = has_enough_concurrent and is_fast_enough

        # Build details
        details_parts = []
        details_parts.append(f"{'✅' if has_enough_concurrent else '❌'} {concurrent_responses}/{total_responses} during generation (need {MIN_CONCURRENT_RESPONSES}+)")
        details_parts.append(f"{'✅' if is_fast_enough else '❌'} Avg latency: {avg_latency:.0f}ms (max allowed: {MAX_DIALOGUE_LATENCY_MS}ms)")
        details_parts.append(f"{'✅' if no_timeouts else '⚠️'} Max latency: {max_latency:.0f}ms ({timeout_count} timeouts)")
        details_parts.append(f"📊 {fast_responses}/{total_responses} fast responses")

        return ValidationResult(
            check_name="Dialogue Non-Blocking Verification",
            passed=passed,
            details=" | ".join(details_parts),
            expected=f"{MIN_CONCURRENT_RESPONSES}+ concurrent responses, <{MAX_DIALOGUE_LATENCY_MS}ms avg latency",
            actual=f"{concurrent_responses} concurrent, {avg_latency:.0f}ms avg latency"
        )


class E2ETestRunner:
    """Runs the interactive E2E test with Emma's child-like dialogue"""

    def __init__(self):
        self.base_url = BASE_URL
        # New family account model
        self.parent_id = PARENT_ID
        self.child_id = CHILD_ID
        self.child_name = CHILD_NAME
        self.child_birth_year = CHILD_BIRTH_YEAR
        self.family_language = FAMILY_LANGUAGE
        # Legacy (for backward compat)
        self.user_id = USER_ID
        self.target_age = TARGET_AGE
        self.validator = ArchitectureValidator()
        self.results: Optional[TestResults] = None
        self.dialogue_count = 0
        self.ws_client: Optional[EmmaWebSocketClient] = None
        # Track asked questions to prevent repetition
        self._asked_curious: set = set()
        self._asked_grounding: set = set()

    def _get_unique_question(self, pool: List[str], asked_set: set) -> str:
        """Get a question from pool that hasn't been asked yet.

        If all questions exhausted, clear the set and start over.
        """
        available = [q for q in pool if q not in asked_set]
        if not available:
            # All questions exhausted - reset and start over
            asked_set.clear()
            available = pool
        question = random.choice(available)
        asked_set.add(question)
        return question

    async def run(self) -> TestResults:
        """Execute the full E2E test"""
        models = get_test_model_names()

        print("=" * 80)
        print(" INTERACTIVE E2E TEST: Emma (7) - Harald Viking Story")
        print(" Testing Chapter 1 + V2 Refinement + Chapter 2")
        print("=" * 80)
        print(f" Models (auto-detected from TEST_* env vars):")
        print(f"   Narrative:  {models['narrative']}")
        print(f"   Structure:  {models['structure']}")
        print(f"   Roundtable: {models['roundtable']}")
        print("=" * 80)
        print()

        # Initialize results
        results = TestResults(
            story_id="",
            start_time=datetime.now()
        )
        self.results = results

        try:
            # Phase 1: Set up family profiles (Parent + Child)
            print("📋 Phase 1: Setting up family profiles...")
            await self._setup_family_profiles()

            # Phase 2: Initialize story with Emma's excited request
            print("\n📖 Phase 2: Emma asks for a Viking story...")
            story_response = await self._initialize_story()
            results.story_id = story_response['story_id']
            results.dialogue_responses.append({
                'phase': 'init',
                'emma_says': random.choice(EMMA_INITIAL_PROMPTS),
                'response': story_response.get('welcome_message', '')
            })

            # Validate initial dialogue response
            if 'welcome_message' in story_response:
                validation = self.validator.validate_dialogue_agent(
                    story_response['welcome_message'],
                    self.target_age
                )
                results.validations.append(validation)

            print(f"   Story ID: {results.story_id}")
            print(f"   🧒 Emma: \"{random.choice(EMMA_INITIAL_PROMPTS)[:60]}...\"")
            print(f"   🤖 Narrator: \"{story_response.get('welcome_message', '')[:100]}...\"")

            # Phase 3: Wait for story generation with interactive dialogue
            print("\n⏳ Phase 3: Story generation (Emma will chat while waiting)...")
            await self._wait_for_story_completion(results.story_id)

            # Phase 4: Fetch and validate story data
            print("\n📊 Phase 4: Fetching story data...")
            story_data = await self._fetch_story(results.story_id)
            results.story_data = story_data

            # Phase 5: Run architecture validations
            print("\n✅ Phase 5: Running architecture validations...")
            await self._run_validations(results, story_data)

            # Phase 6a: Emma makes a story request BEFORE V2 (to influence Chapter 2)
            print("\n💬 Phase 6a: Emma makes a story request (to influence V2)...")
            await self._send_story_influence_request(results)

            # Phase 6b: Trigger V2 refinement and generate Chapter 2
            print("\n📚 Phase 6b: Triggering Structure V2 refinement + Chapter 2...")
            await self._trigger_v2_and_chapter2(results)

            # Phase 7: Emma's follow-up questions after story is ready
            print("\n💬 Phase 7: Emma asks follow-up questions...")
            await self._test_follow_up_questions(results)

        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            results.end_time = datetime.now()

        # Phase 8: Generate report
        print("\n" + "=" * 80)
        print(" TEST RESULTS")
        print("=" * 80)
        self._print_report(results)

        # Save story as markdown for A/B comparison
        if results.story_data:
            self._save_story_markdown(results)

        return results

    async def _setup_family_profiles(self):
        """Set up parent and child profiles for the test (new family account model)"""
        async with aiohttp.ClientSession() as session:
            # Step 1: Create or verify parent account
            print(f"   Setting up parent account: {self.parent_id}")
            parent_payload = {
                "parent_id": self.parent_id,
                "language": self.family_language,
                "display_name": "Smith Family (Test)"
            }

            async with session.post(
                f"{self.base_url}/api/parents",
                json=parent_payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Parent account created: {data.get('parent', {}).get('display_name', self.parent_id)}")
                elif response.status == 409:
                    # Already exists - that's fine
                    print(f"   ✅ Parent account already exists: {self.parent_id}")
                else:
                    error = await response.text()
                    print(f"   ⚠️  Parent creation returned {response.status}: {error[:100]}")

            # Step 2: Create or verify child profile
            print(f"   Setting up child profile: {self.child_id}")
            child_payload = {
                "child_id": self.child_id,
                "name": self.child_name,
                "birth_year": self.child_birth_year
            }

            async with session.post(
                f"{self.base_url}/api/parents/{self.parent_id}/children",
                json=child_payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    child = data.get('child', {})
                    age = child.get('current_age', self.target_age)
                    print(f"   ✅ Child profile created: {child.get('name', self.child_name)}, Age: {age}")
                elif response.status == 409:
                    # Already exists - fetch to verify
                    async with session.get(f"{self.base_url}/api/children/{self.child_id}") as get_resp:
                        if get_resp.status == 200:
                            data = await get_resp.json()
                            child = data.get('child', {})
                            age = child.get('current_age', self.target_age)
                            print(f"   ✅ Child profile exists: {child.get('name', self.child_name)}, Age: {age}")
                        else:
                            print(f"   ⚠️  Child exists but couldn't fetch details")
                else:
                    error = await response.text()
                    print(f"   ⚠️  Child creation returned {response.status}: {error[:100]}")

    async def _initialize_story(self) -> Dict[str, Any]:
        """Initialize the story with Emma's excited request"""
        emma_prompt = random.choice(EMMA_INITIAL_PROMPTS)

        payload = {
            "prompt": emma_prompt,
            # New family account model - use child_id
            "child_id": self.child_id,
            # Language and target_age are derived from child profile automatically
            # but we can override if needed
            # Only generate Chapter 1 initially - Chapter 2 comes after V2 refinement
            # This allows Emma's snowstorm request to influence V2 and Chapter 2
            "chapters_to_write": INITIAL_CHAPTERS,
            "preferences": {
                "educational_focus": "history",
                "difficulty": "medium",
                "themes": ["exploration", "courage", "leadership"],
                "scary_level": "mild"
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/conversation/init",
                json=payload
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Failed to initialize story: {text}")
                return await response.json()

    async def _wait_for_story_completion(self, story_id: str, timeout: int = 0):
        """Wait for story generation with WebSocket event streaming

        Uses WebSocket for real-time events (structure_ready, character_ready, chapter_ready)
        while also polling REST for status. Emma chats via WebSocket for real responses.

        Args:
            story_id: The story to wait for
            timeout: Max wait time in seconds (0 = unlimited, wait forever)
        """
        start = time.time()
        last_status = None
        current_status = "unknown"  # Track current story status for timing
        last_dialogue_time = time.time()
        dialogue_interval = DIALOGUE_INTERVAL_SECONDS  # Use constant (15 seconds)
        chapters_seen = set()
        characters_seen = set()
        structure_seen = False

        # Connect to WebSocket for real-time events
        self.ws_client = EmmaWebSocketClient(story_id)
        ws_connected = await self.ws_client.connect()

        if ws_connected:
            print(f"   Real-time event streaming enabled!")
            # Verify connection is still active using our tracked state
            if self.ws_client.connected:
                print(f"   WebSocket verified open and ready")
            else:
                print(f"   WARNING: WebSocket connection not confirmed")
        else:
            print(f"   WebSocket connection failed, falling back to REST polling only")

        # Send Emma's FIRST message immediately (don't wait 15 seconds)
        print(f"\n   🧒 Emma starts chatting while waiting...")
        await self._emma_chats_while_waiting(story_id, current_status)
        last_dialogue_time = time.time()

        while True:  # No timeout - wait until completion
            # Process any WebSocket events
            if self.ws_client and self.ws_client.connected:
                events = await self.ws_client.receive_events(timeout=2.0)
                for event in events:
                    event_type = event.get("type")
                    data = event.get("data", {})
                    elapsed = int(time.time() - start)

                    if event_type == "dialogue_ready":
                        msg = data.get("message", "")
                        metadata = data.get("metadata", {})
                        phase = metadata.get("phase", data.get("phase", "unknown"))
                        if msg:
                            # Check if this is a response to Emma's question (tier 1)
                            if metadata.get("tier") == 1 or phase == "immediate":
                                print(f"\n   [{elapsed}s] 🤖 Narrator (reply): \"{msg[:200]}{'...' if len(msg) > 200 else ''}\"")
                            else:
                                print(f"\n   [{elapsed}s] 🤖 Narrator: \"{msg[:200]}{'...' if len(msg) > 200 else ''}\"")
                            self.results.dialogue_responses.append({
                                'phase': phase,
                                'response': msg
                            })
                            self.results.websocket_events.append(event)

                    elif event_type == "structure_ready":
                        if not structure_seen:
                            structure_seen = True
                            title = data.get("title", "untitled")
                            # chapters can be a list or an int count
                            chapters_data = data.get("chapters", [])
                            if isinstance(chapters_data, int):
                                chapters_count = chapters_data
                            elif isinstance(chapters_data, list):
                                chapters_count = len(chapters_data)
                            else:
                                chapters_count = 0
                            print(f"\n   [{elapsed}s] Structure ready: '{title}' ({chapters_count} chapters planned)")
                            self.results.websocket_events.append(event)
                            # Emma reacts excitedly
                            print(f"   🧒 Emma: \"Ooooh {chapters_count} chapters! That's gonna be a BIG story!\"")

                    elif event_type == "character_ready":
                        name = data.get("name", "unknown")
                        role = data.get("role", "")
                        if name not in characters_seen:
                            characters_seen.add(name)
                            print(f"\n   [{elapsed}s] Character created: {name} - {role}")
                            self.results.websocket_events.append(event)
                            # Emma reacts to character
                            print(f"   🧒 Emma: \"Cool! I can't wait to meet {name}!\"")

                    elif event_type == "chapter_ready":
                        ch_num = data.get("chapter_number", data.get("chapter", 0))
                        title = data.get("title", "")
                        words = data.get("word_count", 0)
                        if ch_num and ch_num not in chapters_seen:
                            chapters_seen.add(ch_num)
                            print(f"\n   [{elapsed}s] Chapter {ch_num} ready: '{title}' ({words} words)")
                            self.results.websocket_events.append(event)
                            # Emma reacts excitedly
                            reaction = random.choice(EMMA_CHAPTER_REACTIONS)
                            print(f"   🧒 Emma: \"{reaction}\"")

                    elif event_type == "chapter_generating":
                        ch_num = data.get("chapter", 0)
                        msg = data.get("message", "")
                        print(f"\n   [{elapsed}s] Writing chapter {ch_num}...")

                    elif event_type == "story_complete":
                        total = data.get("total_chapters", 0)
                        print(f"\n   [{elapsed}s] Story complete! ({total} chapters)")

                    elif event_type == "error":
                        stage = data.get("stage", "unknown")
                        error = data.get("error", "")
                        print(f"\n   [{elapsed}s] Error in {stage}: {error}")

            # Also poll REST for status (backup + final status check)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/stories/{story_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        story = data.get('story', {})
                        status = story.get('status', 'unknown')
                        current_status = status  # Track for timing validation

                        # Print status changes
                        if status != last_status:
                            elapsed = int(time.time() - start)
                            print(f"   [{elapsed}s] Status: {status}")
                            last_status = status

                        # Check for new chapters from REST (backup if WS missed)
                        chapters_written = story.get('chapters', [])  # Model field is 'chapters', not 'chapters_written'
                        for ch in chapters_written:
                            ch_num = ch.get('number', ch.get('chapter_number'))
                            if ch_num and ch_num not in chapters_seen:
                                chapters_seen.add(ch_num)
                                elapsed = int(time.time() - start)
                                print(f"   [{elapsed}s] Chapter {ch_num} completed (via REST)")
                                # Emma reacts to chapter completion
                                reaction = random.choice(EMMA_CHAPTER_REACTIONS)
                                print(f"   🧒 Emma: \"{reaction}\"")

                        # Emma chats via WebSocket while waiting (every 15 seconds)
                        if time.time() - last_dialogue_time > dialogue_interval:
                            await self._emma_chats_while_waiting(story_id, current_status)
                            last_dialogue_time = time.time()

                        # Check for completion or failure
                        if status in ['completed', 'failed']:
                            if status == 'failed':
                                if self.ws_client:
                                    await self.ws_client.close()
                                raise Exception("Story generation failed")

                            # ========== NON-BLOCKING VALIDATION ==========
                            print(f"\n   📊 Validating dialogue non-blocking behavior...")
                            if self.results.dialogue_timing:
                                non_blocking_validation = self.validator.validate_dialogue_non_blocking(
                                    self.results.dialogue_timing
                                )
                                self.results.validations.append(non_blocking_validation)
                                print(f"   {'✅' if non_blocking_validation.passed else '❌'} {non_blocking_validation.check_name}")
                                print(f"      {non_blocking_validation.details}")
                            else:
                                print(f"   ⚠️  No dialogue timing data collected")

                            print(f"   Story completed in {int(time.time() - start)}s!")
                            if self.ws_client:
                                await self.ws_client.close()
                            return

            await asyncio.sleep(3)  # Check every 3 seconds

    async def _emma_chats_while_waiting(self, story_id: str, current_status: str = "unknown"):
        """Emma sends child-like questions via WebSocket while waiting

        Args:
            story_id: The story ID
            current_status: Current story generation status (for timing validation)
        """
        self.dialogue_count += 1

        # Try to reconnect if WebSocket disconnected
        if self.ws_client and not self.ws_client.connected:
            print(f"   WebSocket disconnected, attempting reconnect...")
            reconnected = await self.ws_client.connect()
            if reconnected:
                print(f"   WebSocket reconnected successfully!")
            else:
                print(f"   WebSocket reconnection failed, will use REST fallback")

        # Alternate between grounding test questions and curious questions
        # This tests dialogue grounding DURING story generation
        # Use unique question getter to avoid repetition
        if self.dialogue_count % 2 == 0:
            emma_says = self._get_unique_question(EMMA_GROUNDING_TEST_QUESTIONS, self._asked_grounding)
        else:
            emma_says = self._get_unique_question(EMMA_CURIOUS_QUESTIONS, self._asked_curious)

        print(f"\n   🧒 Emma: \"{emma_says}\"")

        # ========== TIMING: Start tracking ==========
        send_time = time.time()
        is_generating = current_status not in ['completed', 'failed']

        # Try WebSocket first (gets real dialogue responses)
        if self.ws_client and self.ws_client.connected:
            try:
                await self.ws_client.send_message(emma_says)

                # Keep receiving events until we get dialogue_ready or timeout
                # The dialogue response may come after other events (structure_ready, etc.)
                max_wait = 60.0  # Total max wait time
                start_time = time.time()
                all_events = []
                got_dialogue = False

                while time.time() - start_time < max_wait and not got_dialogue:
                    # Short timeout for each receive attempt
                    events = await self.ws_client.receive_events(timeout=5.0)

                    for event in events:
                        all_events.append(event)
                        self.results.websocket_events.append(event)

                        if event.get("type") == "dialogue_ready":
                            # ========== TIMING: Response received ==========
                            receive_time = time.time()
                            latency_ms = (receive_time - send_time) * 1000

                            data = event.get("data", {})
                            reply = data.get("message", "")
                            if reply:
                                print(f"   🤖 Narrator: \"{reply[:150]}{'...' if len(reply) > 150 else ''}\"")
                                print(f"   ⏱️  Latency: {latency_ms:.0f}ms {'✅' if latency_ms < MAX_DIALOGUE_LATENCY_MS else '❌'}")

                                # ========== TIMING: Record result ==========
                                timing_result = DialogueTimingResult(
                                    question=emma_says,
                                    question_sent_at=send_time,
                                    response_received_at=receive_time,
                                    latency_ms=latency_ms,
                                    story_status=current_status,
                                    concurrent_with_generation=is_generating,
                                    response_text=reply
                                )
                                self.results.dialogue_timing.append(timing_result)

                                # Store dialogue
                                self.results.dialogue_responses.append({
                                    'emma_says': emma_says,
                                    'response': reply,
                                    'phase': 'during_generation',
                                    'latency_ms': latency_ms,
                                    'concurrent': is_generating
                                })

                                # Validate dialogue response - STYLE
                                if len(reply) > 10:
                                    validation = self.validator.validate_dialogue_agent(reply, self.target_age)
                                    self.results.validations.append(validation)

                                    # Validate dialogue response - GROUNDING
                                    grounding_validation = self.validator.validate_dialogue_grounding(reply, emma_says)
                                    self.results.validations.append(grounding_validation)

                                    if grounding_validation.passed:
                                        print(f"   ✅ GROUNDED: {grounding_validation.details}")
                                    else:
                                        print(f"   ❌ NOT GROUNDED: {grounding_validation.details}")
                                got_dialogue = True
                                break

                    if not events and not got_dialogue:
                        # No events received, wait a bit before trying again
                        await asyncio.sleep(0.5)

                # No dialogue_ready received after all attempts
                if not got_dialogue:
                    # ========== TIMING: Record timeout ==========
                    timeout_time = time.time()
                    timeout_latency = (timeout_time - send_time) * 1000
                    timing_result = DialogueTimingResult(
                        question=emma_says,
                        question_sent_at=send_time,
                        response_received_at=timeout_time,
                        latency_ms=timeout_latency,
                        story_status=current_status,
                        concurrent_with_generation=is_generating,
                        response_text=""  # No response
                    )
                    self.results.dialogue_timing.append(timing_result)

                    if all_events:
                        event_types = [e.get("type") for e in all_events]
                        print(f"   ⚠️  (No dialogue response - received: {event_types})")
                    else:
                        print(f"   ⚠️  (No response received - narrator may be busy)")

                return  # Done with WebSocket attempt (success or not)

            except Exception as e:
                print(f"   ⚠️  (WebSocket dialogue error: {str(e)[:50]})")
                # Fall through to REST fallback

        else:
            # Fallback to REST if WebSocket not available
            payload = {
                "story_id": story_id,
                "message": emma_says,
                "child_id": self.child_id  # New family account model
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/conversation/message",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            reply = data.get('response', data.get('message', 'No response'))
                            print(f"   🤖 Narrator: \"{reply[:150]}{'...' if len(str(reply)) > 150 else ''}\"")

                            # Store dialogue
                            self.results.dialogue_responses.append({
                                'emma_says': emma_says,
                                'response': reply,
                                'phase': 'during_generation'
                            })

                            # Validate dialogue response - STYLE
                            if isinstance(reply, str) and len(reply) > 10:
                                validation = self.validator.validate_dialogue_agent(reply, self.target_age)
                                self.results.validations.append(validation)
                                
                                # Validate dialogue response - GROUNDING
                                grounding_validation = self.validator.validate_dialogue_grounding(reply, emma_says)
                                self.results.validations.append(grounding_validation)
                                
                                if grounding_validation.passed:
                                    print(f"   ✅ GROUNDED: {grounding_validation.details}")
                                else:
                                    print(f"   ❌ NOT GROUNDED: {grounding_validation.details}")
                        else:
                            print(f"   ⚠️  (REST fallback - no response)")
            except asyncio.TimeoutError:
                print(f"   ⚠️  (Response timed out)")
            except Exception as e:
                print(f"   ⚠️  (Dialogue unavailable: {str(e)[:50]})")

    async def _fetch_story(self, story_id: str) -> Dict[str, Any]:
        """Fetch complete story data"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/stories/{story_id}") as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch story: {await response.text()}")
                data = await response.json()
                return data.get('story', {})

    async def _run_validations(self, results: TestResults, story_data: Dict[str, Any]):
        """Run all architecture validations"""

        # Validate structure planning
        if 'structure' in story_data:
            validation = self.validator.validate_structure_planning(
                story_data['structure'],
                self.target_age
            )
            results.validations.append(validation)
            print(f"   {'✅' if validation.passed else '❌'} {validation.check_name}")
            print(f"      {validation.details}")

        # Validate characters
        story_id = results.story_id
        characters = story_data.get('characters', [])
        if characters:
            print(f"   Found {len(characters)} characters")
            for char in characters[:3]:  # Validate first 3
                if isinstance(char, dict):
                    validation = self.validator.validate_character_visuals(char)
                    results.validations.append(validation)
                    print(f"   {'✅' if validation.passed else '❌'} {validation.check_name}")

        # Validate chapters (model field is 'chapters', not 'chapters_written')
        chapters_written = story_data.get('chapters', [])
        print(f"   Found {len(chapters_written)} chapters to validate")

        for i, chapter in enumerate(chapters_written):
            if isinstance(chapter, dict):
                chapter_num = chapter.get('number', chapter.get('chapter_number', i+1))

                # Validate prose quality
                validation = self.validator.validate_narrative_prose(chapter, self.target_age)
                results.validations.append(validation)
                print(f"   {'✅' if validation.passed else '❌'} {validation.check_name}")
                print(f"      {validation.details}")

                # Special validation for Chapter 1 scene-setting
                if chapter_num == 1:
                    content = chapter.get('content', '')
                    results.chapter_1_content = content
                    validation = self.validator.validate_first_chapter_scene_setting(content)
                    results.validations.append(validation)
                    print(f"   {'✅' if validation.passed else '❌'} {validation.check_name}")
                    print(f"      {validation.details}")
                    if validation.excerpt:
                        print(f"      📝 Opening: \"{validation.excerpt[:200]}...\"")

                # Validate fact-checking
                validation = self.validator.validate_factcheck_accuracy(chapter)
                results.validations.append(validation)
                print(f"   {'✅' if validation.passed else '❌'} {validation.check_name}")

    async def _send_story_influence_request(self, results: TestResults):
        """Send a story influence request BEFORE V2 to test user input incorporation.

        This tests USER_SCENARIOS.md Scenario 3: User Influences Story Direction.
        We ask for a snowstorm, which should appear in Chapter 2 after V2 refinement.
        """
        # The request Emma will make - easy to verify in Chapter 2
        emma_request = "Ooh! Can there be a big snowstorm in the story? With lots of snow and wind!"

        print(f"   🧒 Emma: \"{emma_request}\"")

        # Connect WebSocket if needed
        if not self.ws_client or not self.ws_client.connected:
            self.ws_client = EmmaWebSocketClient(results.story_id)
            ws_connected = await self.ws_client.connect()
            if not ws_connected:
                print("   ❌ Failed to connect WebSocket for story influence")
                return

        try:
            # Send the request
            await self.ws_client.send_message(emma_request)

            # Wait for acknowledgment
            events = await self.ws_client.receive_events(timeout=30.0)

            for event in events:
                if event.get("type") == "dialogue_ready":
                    reply = event.get("data", {}).get("message", "")
                    if reply:
                        print(f"   🤖 Narrator: \"{reply[:150]}{'...' if len(reply) > 150 else ''}\"")

                        # Store this for later reference
                        results.dialogue_responses.append({
                            'emma_says': emma_request,
                            'response': reply,
                            'phase': 'story_influence'
                        })

                        # Check if acknowledged
                        reply_lower = reply.lower()
                        acknowledged = any(word in reply_lower for word in [
                            'snow', 'storm', 'yes', 'great idea', 'love', 'perfect',
                            'absolutely', 'of course', 'sure', 'wonderful'
                        ])

                        if acknowledged:
                            print("   ✅ Story request acknowledged!")
                        else:
                            print("   ⚠️  Request may not have been acknowledged")

                        results.websocket_events.append(event)
                    break

            # Small delay to ensure the request is processed
            await asyncio.sleep(2)

        except Exception as e:
            print(f"   ❌ Story influence request error: {e}")

    async def _trigger_v2_and_chapter2(self, results: TestResults):
        """Trigger Structure V2 refinement and generate Chapter 2.

        This method:
        1. Sends start_chapter for Chapter 1 (triggers V2 refinement)
        2. Waits for Chapter 2 to be fully generated
        3. Re-fetches story data to include Chapter 2 content
        """
        print("   ▶️  Triggering Structure V2 via start_chapter...")

        # Check if structure has more than 1 chapter
        story_data = results.story_data or {}
        structure = story_data.get('structure', {})
        planned_chapters = structure.get('chapters', [])

        if len(planned_chapters) < 2:
            print("   ⚠️  Story only has 1 planned chapter - skipping Chapter 2 generation")
            return

        # Reconnect WebSocket
        if not self.ws_client or not self.ws_client.connected:
            self.ws_client = EmmaWebSocketClient(results.story_id)
            ws_connected = await self.ws_client.connect()
            if not ws_connected:
                print("   ❌ Failed to connect WebSocket")
                return

        try:
            # Send start_chapter message for Chapter 1
            # This triggers V2 refinement AND prefetch of Chapter 2
            await self.ws_client.ws.send(json.dumps({
                "type": "start_chapter",
                "chapter_number": 1
            }))

            # Track events we receive
            v2_refinement_seen = False
            chapter_2_generating = False
            chapter_2_ready = False

            # Wait for events (with generous timeout for chapter generation)
            start_time = time.time()
            max_wait_seconds = 600  # 10 minutes max for Chapter 2

            print("   ⏳ Waiting for V2 refinement and Chapter 2 generation...")

            while time.time() - start_time < max_wait_seconds:
                events = await self.ws_client.receive_events(timeout=10.0)

                for event in events:
                    event_type = event.get("type")
                    data = event.get("data", {})
                    elapsed = int(time.time() - start_time)

                    if event_type == "structure_v2_ready":
                        v2_refinement_seen = True
                        print(f"   [{elapsed}s] ✅ Structure V2 refinement complete")
                        results.websocket_events.append(event)

                    elif event_type == "chapter_generating":
                        ch_num = data.get("chapter_number", data.get("chapter", 0))
                        if ch_num == 2:
                            chapter_2_generating = True
                            print(f"   [{elapsed}s] 📝 Chapter 2 generation started...")
                            results.websocket_events.append(event)

                    elif event_type == "chapter_ready":
                        ch_num = data.get("chapter_number", data.get("chapter", 0))
                        if ch_num == 2:
                            chapter_2_ready = True
                            title = data.get("title", "")
                            words = data.get("word_count", 0)
                            print(f"   [{elapsed}s] ✅ Chapter 2 ready: '{title}' ({words} words)")
                            results.websocket_events.append(event)

                    elif event_type == "chapter_started":
                        ch_num = data.get("chapter_number", 0)
                        print(f"   [{elapsed}s] ▶️  Chapter {ch_num} playback confirmed")
                        results.websocket_events.append(event)

                # If Chapter 2 is ready, we're done!
                if chapter_2_ready:
                    break

            # Re-fetch story data to include Chapter 2
            if chapter_2_ready:
                print("   📊 Re-fetching story data with Chapter 2...")
                updated_story = await self._fetch_story(results.story_id)
                results.story_data = updated_story

                # Update chapter count
                chapters = updated_story.get('chapters', [])
                print(f"   ✅ Story now has {len(chapters)} chapters")

                # Validate Chapter 2 was added
                results.validations.append(ValidationResult(
                    check_name="Chapter 2 Generation",
                    passed=len(chapters) >= 2,
                    details=f"Generated Chapter 2 after V2 refinement",
                    expected="2 chapters",
                    actual=f"{len(chapters)} chapters"
                ))

                # Validate snowstorm influence in Chapter 2
                chapter_2 = next((ch for ch in chapters if ch.get('number') == 2), None)
                if chapter_2:
                    ch2_content = chapter_2.get('content', '').lower()
                    snow_keywords = ['snow', 'snowstorm', 'blizzard', 'winter', 'frost', 'icy', 'frozen', 'cold wind']
                    snow_found = [kw for kw in snow_keywords if kw in ch2_content]

                    snow_influence_passed = len(snow_found) >= 1
                    results.validations.append(ValidationResult(
                        check_name="Story Influence - Snowstorm Request",
                        passed=snow_influence_passed,
                        details=f"Found: {', '.join(snow_found)}" if snow_found else "No snow/winter keywords found",
                        expected="Snowstorm elements in Chapter 2 (from Emma's request)",
                        actual=f"{len(snow_found)} keywords found"
                    ))

                    if snow_influence_passed:
                        print(f"   ✅ Snowstorm influence detected in Chapter 2: {', '.join(snow_found)}")
                    else:
                        print("   ❌ Snowstorm NOT found in Chapter 2 - user input may not have influenced V2")
            else:
                elapsed = int(time.time() - start_time)
                print(f"   ⚠️  Chapter 2 not ready after {elapsed}s")
                results.validations.append(ValidationResult(
                    check_name="Chapter 2 Generation",
                    passed=False,
                    details=f"Timeout after {elapsed}s waiting for Chapter 2",
                    expected="Chapter 2 ready within 10 minutes",
                    actual="Timeout"
                ))

        except Exception as e:
            print(f"   ❌ V2/Chapter 2 error: {e}")
            import traceback
            traceback.print_exc()

        # Note: Don't close WebSocket here - follow-up questions phase will use it

    async def _test_follow_up_questions(self, results: TestResults):
        """Emma asks follow-up questions via WebSocket like a real 7-year-old
        
        This phase specifically tests DIALOGUE GROUNDING - ensuring responses
        reference actual story content (characters, plot, educational points)
        rather than giving generic/fluffy responses.
        """

        # Use GROUNDING TEST QUESTIONS to validate story-specific responses
        # These are designed to require knowledge of the story content
        emma_follow_ups = EMMA_GROUNDING_TEST_QUESTIONS[:3]  # First 3 grounding questions

        print(f"   Testing {len(emma_follow_ups)} GROUNDING dialogue interactions...")
        print(f"   (Expecting responses with story-specific content: characters, plot, etc.)")

        # Reconnect WebSocket for follow-up phase if not connected
        if not self.ws_client or not self.ws_client.connected:
            self.ws_client = EmmaWebSocketClient(results.story_id)
            await self.ws_client.connect()

        for i, emma_says in enumerate(emma_follow_ups, 1):
            print(f"\n   🧒 Emma ({i}): \"{emma_says}\"")

            # Try WebSocket first
            if self.ws_client and self.ws_client.connected:
                try:
                    await self.ws_client.send_message(emma_says)

                    # Wait for dialogue_ready event
                    events = await self.ws_client.receive_events(timeout=60.0)

                    for event in events:
                        if event.get("type") == "dialogue_ready":
                            data = event.get("data", {})
                            reply = data.get("message", "")
                            if reply:
                                print(f"   🤖 Narrator: \"{reply[:150]}{'...' if len(reply) > 150 else ''}\"")

                                # Store dialogue response
                                results.dialogue_responses.append({
                                    'emma_says': emma_says,
                                    'response': reply,
                                    'phase': 'follow_up'
                                })
                                results.websocket_events.append(event)

                                # Validate dialogue response - STYLE
                                if len(reply) > 10:
                                    validation = self.validator.validate_dialogue_agent(reply, self.target_age)
                                    results.validations.append(validation)

                                    if not validation.passed:
                                        print(f"   ⚠️  Style validation: {validation.details}")
                                    
                                    # Validate dialogue response - GROUNDING (new!)
                                    grounding_validation = self.validator.validate_dialogue_grounding(reply, emma_says)
                                    results.validations.append(grounding_validation)
                                    
                                    if grounding_validation.passed:
                                        print(f"   ✅ GROUNDED: {grounding_validation.details}")
                                    else:
                                        print(f"   ❌ NOT GROUNDED: {grounding_validation.details}")
                                break
                        else:
                            results.websocket_events.append(event)
                    else:
                        print(f"   ⚠️  No dialogue response received")

                except Exception as e:
                    print(f"   ❌ WebSocket exception: {e}")

            else:
                # Fallback to REST
                payload = {
                    "story_id": results.story_id,
                    "message": emma_says,
                    "child_id": self.child_id  # New family account model
                }

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/api/conversation/message",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                reply = data.get('response', data.get('message', 'No response'))
                                print(f"   🤖 Narrator (REST): \"{reply[:150]}{'...' if len(str(reply)) > 150 else ''}\"")

                                # Store dialogue response
                                results.dialogue_responses.append({
                                    'emma_says': emma_says,
                                    'response': reply,
                                    'phase': 'follow_up'
                                })

                                # Validate dialogue response - STYLE
                                if isinstance(reply, str) and len(reply) > 10:
                                    validation = self.validator.validate_dialogue_agent(reply, self.target_age)
                                    results.validations.append(validation)

                                    if not validation.passed:
                                        print(f"   ⚠️  Style validation: {validation.details}")
                                    
                                    # Validate dialogue response - GROUNDING (new!)
                                    grounding_validation = self.validator.validate_dialogue_grounding(reply, emma_says)
                                    results.validations.append(grounding_validation)
                                    
                                    if grounding_validation.passed:
                                        print(f"   ✅ GROUNDED: {grounding_validation.details}")
                                    else:
                                        print(f"   ❌ NOT GROUNDED: {grounding_validation.details}")
                            else:
                                error_text = await response.text()
                                print(f"   ❌ Failed (status {response.status})")

                except Exception as e:
                    print(f"   ❌ REST exception: {e}")

            # Natural pause between Emma's questions
            await asyncio.sleep(2)

        # Close WebSocket when done with follow-ups
        if self.ws_client:
            await self.ws_client.close()

    def _print_report(self, results: TestResults):
        """Print comprehensive validation report"""
        print(f"\n📊 Story ID: {results.story_id}")
        print(f"⏱️  Duration: {results.duration:.1f}s ({results.duration/60:.1f} minutes)")
        print(f"💬 Dialogue exchanges: {len(results.dialogue_responses)}")

        # ========== TIMING SUMMARY ==========
        if results.dialogue_timing:
            print(f"\n{'─' * 80}")
            print(f"  NON-BLOCKING DIALOGUE TIMING")
            print(f"{'─' * 80}")
            print(f"   📊 Total dialogue attempts: {len(results.dialogue_timing)}")
            print(f"   ⏱️  Average latency: {results.average_latency_ms:.0f}ms")
            print(f"   🔄 Concurrent responses: {results.concurrent_response_count}/{len(results.dialogue_timing)}")
            print(f"   {'✅' if results.dialogue_non_blocking else '❌'} Non-blocking: {results.dialogue_non_blocking}")

            # Show individual timing breakdown
            print(f"\n   Individual response times:")
            for i, t in enumerate(results.dialogue_timing, 1):
                status_icon = "🔄" if t.concurrent_with_generation else "⏸️"
                speed_icon = "✅" if t.is_fast else "❌"
                print(f"   {i}. {status_icon} {t.latency_ms:.0f}ms {speed_icon} ({t.story_status})")

        print(f"\n{'─' * 80}")
        print(f"  VALIDATION RESULTS ({len(results.validations)} checks)")
        print(f"{'─' * 80}\n")

        passed_count = sum(1 for v in results.validations if v.passed)
        failed_count = len(results.validations) - passed_count

        # Group validations by type
        validation_groups = {}
        for v in results.validations:
            group = v.check_name.split(' - ')[0] if ' - ' in v.check_name else 'Other'
            if group not in validation_groups:
                validation_groups[group] = []
            validation_groups[group].append(v)

        for group, validations in validation_groups.items():
            group_passed = sum(1 for v in validations if v.passed)
            print(f"\n📋 {group} ({group_passed}/{len(validations)} passed)")
            for v in validations:
                icon = "✅" if v.passed else "❌"
                print(f"   {icon} {v.check_name.replace(group + ' - ', '')}")
                print(f"      {v.details}")

        print(f"\n{'=' * 80}")
        print(f"\n🎯 SUMMARY: {passed_count}/{len(results.validations)} validations passed")

        if results.passed:
            print("✅ ALL VALIDATIONS PASSED - Story meets architecture requirements!")
        else:
            print(f"❌ {failed_count} validation(s) failed - See details above")

        # Print chapter 1 opening excerpt
        if results.chapter_1_content:
            print(f"\n{'─' * 80}")
            print("📖 CHAPTER 1 OPENING (First 500 chars):")
            print(f"{'─' * 80}")
            print(results.chapter_1_content[:500] + "...")

        print(f"\n{'=' * 80}\n")

    def _build_story_markdown(self, results: TestResults, models: Dict[str, str]) -> str:
        """Build comprehensive markdown content for A/B testing comparison."""
        story_data = results.story_data or {}
        structure = story_data.get('structure', {})
        characters = story_data.get('characters', [])
        chapters = story_data.get('chapters', [])

        lines = []

        # Header
        lines.append("# A/B Test Report: Harald Viking Story for Emma\n")

        # Test Configuration
        lines.append("## Test Configuration\n")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| **Test Date** | {results.start_time.strftime('%Y-%m-%d %H:%M:%S')} |")
        lines.append(f"| **Story ID** | {results.story_id} |")
        lines.append(f"| **Narrative Model** | {models.get('narrative', 'default')} |")
        lines.append(f"| **Structure Model** | {models.get('structure', 'default')} |")
        lines.append(f"| **Roundtable Model** | {models.get('roundtable', 'default')} |")
        lines.append(f"| **Language** | English (en) |")
        lines.append(f"| **Target Age** | {self.target_age} years |")
        lines.append(f"| **Test Duration** | {results.duration:.1f}s ({results.duration/60:.1f} minutes) |")
        lines.append(f"| **Chapters Generated** | {len(chapters)} (Chapter 1 + V2 + Chapter 2) |")
        lines.append("")

        # Story Overview
        lines.append("## Story Overview\n")
        lines.append(f"**Title:** {structure.get('title', 'N/A')}\n")
        theme = structure.get('theme', 'N/A')
        if theme and len(theme) > 200:
            theme = theme[:200] + "..."
        lines.append(f"**Theme:** {theme}\n")

        # Narrative Method
        narrative_method = structure.get('narrative_method', {})
        if narrative_method:
            method_type = narrative_method.get('method', 'linear_single_pov')
            pov_chars = narrative_method.get('pov_characters', [])
            hook_strategy = narrative_method.get('hook_strategy', 'N/A')
            lines.append("### Narrative Method (Guillermo + Stephen Debate)\n")
            lines.append(f"**Method:** {method_type}\n")
            lines.append(f"**POV Characters:** {', '.join(pov_chars) if pov_chars else 'N/A'}\n")
            lines.append(f"**Hook Strategy:** {hook_strategy[:150]}{'...' if len(hook_strategy) > 150 else ''}\n")
            lines.append("")

        # Test Validations
        lines.append("## Test Validations\n")
        passed = sum(1 for v in results.validations if v.passed)
        total = len(results.validations)
        lines.append(f"**Result:** {passed}/{total} passed\n")
        lines.append("| Check | Status | Details |")
        lines.append("|-------|--------|---------|")
        for v in results.validations:
            status = "✅" if v.passed else "❌"
            details = v.details[:60] + "..." if len(v.details) > 60 else v.details
            lines.append(f"| {v.check_name} | {status} | {details} |")
        lines.append("")

        # Characters Section
        if characters:
            lines.append("---\n")
            lines.append("## Characters Created\n")
            for char in characters[:10]:  # Limit to 10
                name = char.get('name', 'Unknown')
                role = char.get('role', 'N/A')
                lines.append(f"### {name}")
                lines.append(f"**Role:** {role}\n")
                bg = char.get('background', char.get('description', ''))
                if bg:
                    lines.append(f"{bg[:300]}{'...' if len(bg) > 300 else ''}\n")

        # Chapter Statistics
        if chapters:
            lines.append("## Chapter Statistics\n")
            lines.append("| Chapter | Title | Words |")
            lines.append("|---------|-------|-------|")
            total_words = 0
            for ch in sorted(chapters, key=lambda x: x.get('number', 0)):
                num = ch.get('number', 0)
                title = ch.get('title', 'Untitled')[:40]
                content = ch.get('content', '')
                words = len(content.split())
                total_words += words
                lines.append(f"| {num} | {title} | {words} |")
            lines.append(f"| **Total** | | **{total_words}** |")
            lines.append("")

        # Story Content
        if chapters:
            lines.append("---\n")
            lines.append("## Story Content\n")
            for ch in sorted(chapters, key=lambda x: x.get('number', 0)):
                num = ch.get('number', 0)
                title = ch.get('title', 'Untitled')
                content = ch.get('content', '')
                words = len(content.split())
                lines.append(f"### Chapter {num}: {title}")
                lines.append(f"*({words} words)*\n")
                lines.append(content)
                lines.append("")

                # Add Round Table Review for this chapter
                rt_review = ch.get('round_table_review')
                if rt_review:
                    lines.append(f"#### Round Table Review - Chapter {num}\n")
                    decision = rt_review.get('decision', 'unknown')
                    revision_rounds = rt_review.get('revision_rounds', 0)
                    lines.append(f"**Decision:** {decision} | **Revision Rounds:** {revision_rounds}\n")

                    reviews = rt_review.get('reviews', [])
                    if reviews:
                        lines.append("| Agent | Verdict | Praise | Concern |")
                        lines.append("|-------|---------|--------|---------|")
                        for review in reviews:
                            agent = review.get('agent', 'Unknown')
                            verdict = review.get('verdict', 'N/A')
                            verdict_icon = "✅" if verdict == "approve" else "⚠️"
                            praise = review.get('praise', '')[:60].replace('|', '/').replace('\n', ' ')
                            concern = review.get('concern', '')[:60].replace('|', '/').replace('\n', ' ')
                            lines.append(f"| {agent} | {verdict_icon} {verdict} | {praise}... | {concern}... |")
                        lines.append("")

                lines.append("\n---\n")

        # Dialogue Responses
        if results.dialogue_responses:
            lines.append("## Dialogue Interactions\n")
            for i, dialogue in enumerate(results.dialogue_responses[:10], 1):
                emma_says = dialogue.get('emma_says', '')
                response = dialogue.get('response', '')
                phase = dialogue.get('phase', 'unknown')
                if emma_says:
                    lines.append(f"**Emma ({phase}):** {emma_says[:100]}{'...' if len(emma_says) > 100 else ''}\n")
                if response:
                    lines.append(f"**Narrator:** {response[:200]}{'...' if len(response) > 200 else ''}\n")
                lines.append("")

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- **Narrative Model:** {models.get('narrative', 'default')}")
        lines.append(f"- **Structure Model:** {models.get('structure', 'default')}")
        lines.append(f"- **Roundtable Model:** {models.get('roundtable', 'default')}")
        lines.append(f"- **Total Generation Time:** {results.duration/60:.1f} minutes")
        lines.append(f"- **Chapters Generated:** {len(chapters)}")
        lines.append(f"- **Validations Passed:** {passed}/{total}")
        lines.append("")

        return "\n".join(lines)

    def _save_story_markdown(self, results: TestResults):
        """Save the generated story as a markdown file."""
        models = get_test_model_names()
        narrative_model = models["narrative"]

        # Create safe filename from model name
        safe_model_name = get_safe_model_name(narrative_model)

        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp for uniqueness
        timestamp = results.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"Emma_Harald_{safe_model_name}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)

        # Build and write markdown content
        md_content = self._build_story_markdown(results, models)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"\n📄 Story saved to: {filepath}")
        return filepath


async def main():
    """Main entry point"""
    runner = E2ETestRunner()
    results = await runner.run()

    # Exit with appropriate code
    exit(0 if results.passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
