"""
Intent Classifier Service

Classifies child messages to determine what they want to do:
- Continue an existing story
- Start a new story
- Explore/discover what they want
- Simple greeting

This enables the "conversation-first" UX where CompanionAgent
engages before story generation begins.
"""

import re
from enum import Enum
from typing import Optional
from dataclasses import dataclass
import google.generativeai as genai
import os


class ConversationIntent(Enum):
    """Types of conversation intents from a child's message."""
    CONTINUE_STORY = "continue"      # "Continue my story", "What happens next?"
    NEW_STORY = "new_story"          # "Tell me about dragons", "I want a viking story"
    EXPLORING = "exploring"          # "I don't know", "What can we do?", vague messages
    GREETING = "greeting"            # "Hi!", "Hello Hanan"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: ConversationIntent
    confidence: float  # 0.0 - 1.0
    suggested_action: str  # "resume_story", "init_story", "ask_preference", "greet"
    extracted_topic: Optional[str] = None  # For NEW_STORY: "dragons", "vikings", etc.
    reason: Optional[str] = None  # Why this classification was made


# Fast-path regex patterns for each intent (multilingual)
CONTINUE_PATTERNS = [
    # English
    r"\bcontinue\b", r"\blast time\b", r"\bwhat happens next\b",
    r"\bkeep going\b", r"\bmore story\b", r"\bresume\b",
    r"\bwhere (was I|were we|did we stop)\b",
    # Norwegian
    r"\bfortsett(e)?\b", r"\bvidere\b", r"\bsist (gang)?\b",
    r"\bhva skjer (nå|videre)\b", r"\bforrige historie\b",
    r"\bfortsette historien\b",
    # Spanish
    r"\bcontinuar?\b", r"\bla última vez\b", r"\bseguir\b",
    r"qué pasa (ahora|después)", r"después\b",
]

NEW_STORY_PATTERNS = [
    # English
    r"\bnew story\b", r"\bstory about\b", r"\btell me about\b",
    r"\bI want a story\b", r"\bcan (you|we) (make|create|tell)\b",
    r"\babout (a |an )?(dragon|viking|princess|knight|pirate|animal|bear|cat|dog)\b",
    r"\bhear about\b", r"\blike to hear\b",  # "I'd like to hear about..."
    r"\bancient\b", r"\bmedieval\b", r"\bspace\b", r"\bocean\b",  # Topic keywords
    r"\begypt\b", r"\bpharaoh\b", r"\bpyramid\b", r"\bmummy\b",  # Egyptian topics
    # Norwegian - NOTE: "ny" (common) vs "nytt" (neuter) grammatical gender
    r"\b(ny|nytt) (historie|eventyr|fortelling)\b",  # "ny historie", "nytt eventyr"
    r"\b(historie|eventyr|fortelling) om\b",  # "historie om", "eventyr om"
    r"\bfortell meg om\b",
    r"\bjeg (vil ha|ønsker) (en |et )?(ny|nytt)?\b",  # "jeg vil ha", "jeg ønsker et nytt"
    r"\bkan (du|vi) lage\b",
    r"\bhøre om\b",  # "I want to hear about"
    r"\bhelt (ny|nytt)\b",  # "helt nytt eventyr" (completely new)
    r"\begypt\b", r"\bfarao\b", r"\bpyramide\b", r"\bmumie\b",  # Norwegian Egyptian topics
    # Spanish
    r"\bnueva historia\b", r"\bhistoria (sobre|de)\b",
    r"\bquiero (una |un )?historia\b", r"\bcuéntame (sobre|de)\b",
    r"\bescuchar sobre\b",  # "I want to listen about"
    r"\begipto\b", r"\bfaraón\b", r"\bpirámide\b", r"\bmomia\b",  # Spanish Egyptian topics
]

EXPLORING_PATTERNS = [
    # English
    r"\bI don'?t know\b", r"\bwhat can\b", r"\bhelp me\b",
    r"\bnot sure\b", r"\bmaybe\b", r"\bum+\b", r"\bhmm+\b",
    # Norwegian
    r"\bvet ikke\b", r"\bhjelp\b", r"\bkanskje\b",
    r"\bikke sikker\b", r"\behm+\b",
    # Spanish
    r"\bno sé\b", r"\bayuda\b", r"\btal vez\b",
    r"\bno estoy segur[oa]\b",
]

GREETING_PATTERNS = [
    # English
    r"^(hi|hello|hey|hiya)[\s!.]*$",
    r"^good (morning|afternoon|evening)[\s!.]*$",
    # Norwegian
    r"^(hei|hallo|heisann)[\s!.]*$",
    r"^god (morgen|dag|kveld)[\s!.]*$",
    # Spanish
    r"^(hola|buenos días|buenas tardes|buenas noches)[\s!.]*$",
]


def _match_patterns(message: str, patterns: list) -> bool:
    """Check if message matches any pattern in the list."""
    message_lower = message.lower().strip()
    for pattern in patterns:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return True
    return False


def _extract_story_topic(message: str) -> Optional[str]:
    """Extract potential story topic from a new story request."""
    message_lower = message.lower()

    # Common story topics
    topics = [
        "dragon", "viking", "princess", "prince", "knight", "pirate",
        "animal", "bear", "cat", "dog", "horse", "dinosaur", "robot",
        "space", "ocean", "forest", "castle", "adventure", "mystery",
        "magic", "fairy", "monster", "superhero", "explorer",
        # Norwegian
        "drage", "prinsesse", "ridder", "sjørøver", "bjørn", "katt",
        "hund", "eventyr", "magi", "troll", "nisse",
        # Spanish
        "dragón", "princesa", "caballero", "pirata", "oso", "gato",
        "perro", "aventura", "magia", "monstruo",
    ]

    for topic in topics:
        if topic in message_lower:
            return topic

    return None


async def classify_intent_with_llm(
    message: str,
    has_active_story: bool,
    child_age: int,
    language: str = "en"
) -> IntentResult:
    """
    Use LLM to classify ambiguous messages.

    This is called when fast-path patterns don't match.
    """
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        context = f"Child has an active story to continue" if has_active_story else "No active story"

        prompt = f"""Classify this child's message into ONE of these intents:
- CONTINUE: Child wants to continue/resume their existing story
- NEW_STORY: Child wants a new story about a specific topic
- EXPLORING: Child is unsure, vague, or needs help deciding
- GREETING: Just a greeting with no story intent

Child's message: "{message}"
Child's age: {child_age} years old
Context: {context}
Language: {language}

Consider:
- Younger children (3-5) with vague messages → likely EXPLORING
- Messages with specific topics (dragons, pirates) → likely NEW_STORY
- References to "last time", "continue", "what happens next" → likely CONTINUE
- Just "hi" or "hello" without more → GREETING

Respond with ONLY one word: CONTINUE, NEW_STORY, EXPLORING, or GREETING"""

        response = await model.generate_content_async(prompt)
        result_text = response.text.strip().upper()

        intent_map = {
            "CONTINUE": ConversationIntent.CONTINUE_STORY,
            "NEW_STORY": ConversationIntent.NEW_STORY,
            "EXPLORING": ConversationIntent.EXPLORING,
            "GREETING": ConversationIntent.GREETING,
        }

        intent = intent_map.get(result_text, ConversationIntent.EXPLORING)

        extracted_topic = _extract_story_topic(message) if intent == ConversationIntent.NEW_STORY else None
        return IntentResult(
            intent=intent,
            confidence=0.8,  # LLM classification
            suggested_action=_get_suggested_action(intent, has_active_story, has_topic=extracted_topic is not None),
            extracted_topic=extracted_topic,
            reason=f"LLM classified as {result_text}"
        )

    except Exception as e:
        print(f"⚠️ Intent classification LLM error: {e}")
        # Fallback: exploring for young children, greeting for simple messages
        if child_age <= 5 or len(message.split()) <= 2:
            return IntentResult(
                intent=ConversationIntent.EXPLORING,
                confidence=0.5,
                suggested_action="ask_preference",
                reason="Fallback due to LLM error"
            )
        return IntentResult(
            intent=ConversationIntent.GREETING,
            confidence=0.5,
            suggested_action="greet",
            reason="Fallback due to LLM error"
        )


def _get_suggested_action(intent: ConversationIntent, has_active_story: bool, has_topic: bool = False) -> str:
    """Get the suggested frontend action based on intent.

    Args:
        intent: The classified conversation intent
        has_active_story: Whether child has an active story
        has_topic: For NEW_STORY intent, whether a specific topic was extracted
    """
    if intent == ConversationIntent.CONTINUE_STORY:
        return "resume_story" if has_active_story else "ask_preference"
    elif intent == ConversationIntent.NEW_STORY:
        # Only init_story if child gave a clear topic, otherwise ask what they want
        return "init_story" if has_topic else "ask_preference"
    elif intent == ConversationIntent.EXPLORING:
        return "ask_preference"
    else:  # GREETING
        return "greet"


async def classify_conversation_intent(
    message: str,
    has_active_story: bool,
    child_age: int,
    language: str = "en"
) -> IntentResult:
    """
    Classify what the child wants to do.

    Uses fast-path pattern matching first, then LLM for ambiguous cases.

    Args:
        message: The child's message
        has_active_story: Whether the child has an active story to continue
        child_age: Child's age in years
        language: Language code (en, no, es)

    Returns:
        IntentResult with classification and suggested action
    """
    message_clean = message.strip()

    # Empty or very short messages from young children → exploring
    if len(message_clean) < 3 or (child_age <= 5 and len(message_clean.split()) <= 1):
        return IntentResult(
            intent=ConversationIntent.EXPLORING,
            confidence=0.9,
            suggested_action="ask_preference",
            reason="Very short message from young child"
        )

    # Fast-path: Check greeting patterns first (exact matches)
    if _match_patterns(message_clean, GREETING_PATTERNS):
        return IntentResult(
            intent=ConversationIntent.GREETING,
            confidence=0.95,
            suggested_action="greet",
            reason="Matched greeting pattern"
        )

    # Fast-path: Check continue patterns
    if _match_patterns(message_clean, CONTINUE_PATTERNS):
        return IntentResult(
            intent=ConversationIntent.CONTINUE_STORY,
            confidence=0.95,
            suggested_action="resume_story" if has_active_story else "ask_preference",
            reason="Matched continue pattern"
        )

    # Fast-path: Check new story patterns
    if _match_patterns(message_clean, NEW_STORY_PATTERNS):
        topic = _extract_story_topic(message_clean)
        return IntentResult(
            intent=ConversationIntent.NEW_STORY,
            confidence=0.9,
            suggested_action=_get_suggested_action(ConversationIntent.NEW_STORY, has_active_story, has_topic=topic is not None),
            extracted_topic=topic,
            reason=f"Matched new story pattern, topic: {topic}"
        )

    # Fast-path: Check exploring patterns
    if _match_patterns(message_clean, EXPLORING_PATTERNS):
        return IntentResult(
            intent=ConversationIntent.EXPLORING,
            confidence=0.9,
            suggested_action="ask_preference",
            reason="Matched exploring pattern"
        )

    # Young children (3-5) with ambiguous messages → default to exploring
    if child_age <= 5:
        return IntentResult(
            intent=ConversationIntent.EXPLORING,
            confidence=0.7,
            suggested_action="ask_preference",
            reason="Young child with ambiguous message"
        )

    # No fast-path match → use LLM for classification
    return await classify_intent_with_llm(
        message=message_clean,
        has_active_story=has_active_story,
        child_age=child_age,
        language=language
    )
