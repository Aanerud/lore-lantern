"""
Test Intent Classifier

Tests for the conversation-first UX intent classification.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.intent_classifier import (
    classify_conversation_intent,
    ConversationIntent,
    _match_patterns,
    CONTINUE_PATTERNS,
    NEW_STORY_PATTERNS,
    EXPLORING_PATTERNS,
    GREETING_PATTERNS
)


async def test_continue_intent():
    """Test detection of continue/resume intent."""
    print("\n=== Testing CONTINUE intent ===")

    test_cases = [
        # English
        ("I want to continue my story", True, 7),
        ("what happens next?", True, 8),
        ("continue from last time", True, 6),
        ("keep going with the story", True, 9),
        # Norwegian
        ("Jeg vil fortsette historien", True, 7),
        ("Hva skjer videre?", True, 8),
        ("fortsett fra sist gang", True, 6),
        # Spanish
        ("Quiero continuar la historia", True, 7),
        ("¿Qué pasa después?", True, 8),
    ]

    for message, has_active, age in test_cases:
        result = await classify_conversation_intent(message, has_active, age)
        status = "✅" if result.intent == ConversationIntent.CONTINUE_STORY else "❌"
        print(f"  {status} '{message}' → {result.intent.value} (conf: {result.confidence:.2f})")


async def test_new_story_intent():
    """Test detection of new story intent."""
    print("\n=== Testing NEW_STORY intent ===")

    test_cases = [
        # English
        ("I want a story about dragons", False, 7),
        ("Tell me about Vikings", False, 10),
        ("Can you make a pirate adventure?", False, 8),
        ("new story about a princess", False, 5),
        # Norwegian
        ("Jeg vil ha en historie om drager", False, 7),
        ("Fortell meg om vikinger", False, 10),
        ("ny historie om en prinsesse", False, 5),
        # Spanish
        ("Quiero una historia de dragones", False, 7),
        ("Cuéntame sobre piratas", False, 8),
    ]

    for message, has_active, age in test_cases:
        result = await classify_conversation_intent(message, has_active, age)
        status = "✅" if result.intent == ConversationIntent.NEW_STORY else "❌"
        print(f"  {status} '{message}' → {result.intent.value} (topic: {result.extracted_topic})")


async def test_exploring_intent():
    """Test detection of exploring/unsure intent."""
    print("\n=== Testing EXPLORING intent ===")

    test_cases = [
        # English
        ("I don't know", False, 4),
        ("hmm", False, 5),
        ("maybe...", False, 4),
        ("help me choose", False, 6),
        # Norwegian
        ("vet ikke", False, 4),
        ("ehm...", False, 5),
        ("kanskje", False, 4),
        # Spanish
        ("no sé", False, 4),
        ("tal vez", False, 5),
        # Very short/vague from young children
        ("um", False, 4),
        ("hi", False, 3),  # Should be greeting but might be exploring for 3yo
    ]

    for message, has_active, age in test_cases:
        result = await classify_conversation_intent(message, has_active, age)
        is_exploring = result.intent in [ConversationIntent.EXPLORING, ConversationIntent.GREETING]
        status = "✅" if is_exploring else "❌"
        print(f"  {status} '{message}' (age {age}) → {result.intent.value}")


async def test_greeting_intent():
    """Test detection of simple greetings."""
    print("\n=== Testing GREETING intent ===")

    test_cases = [
        # English
        ("hi!", False, 7),
        ("hello", False, 8),
        ("hey", False, 6),
        ("good morning", False, 5),
        # Norwegian
        ("hei!", False, 7),
        ("hallo", False, 8),
        ("god morgen", False, 5),
        # Spanish
        ("hola", False, 7),
        ("buenos días", False, 8),
    ]

    for message, has_active, age in test_cases:
        result = await classify_conversation_intent(message, has_active, age)
        status = "✅" if result.intent == ConversationIntent.GREETING else "❌"
        print(f"  {status} '{message}' → {result.intent.value}")


async def test_edge_cases():
    """Test edge cases and ambiguous messages."""
    print("\n=== Testing EDGE CASES ===")

    test_cases = [
        # Mixed intent - should prioritize based on context
        ("Hi, I want to continue my story", True, 7, ConversationIntent.CONTINUE_STORY),
        ("Hello! Tell me about dragons", False, 8, ConversationIntent.NEW_STORY),

        # Young child with vague message
        ("story", False, 4, ConversationIntent.EXPLORING),
        ("...", False, 3, ConversationIntent.EXPLORING),

        # Older child with clear intent
        ("I'd like to hear about ancient Egypt", False, 12, ConversationIntent.NEW_STORY),
    ]

    for message, has_active, age, expected in test_cases:
        result = await classify_conversation_intent(message, has_active, age)
        status = "✅" if result.intent == expected else "⚠️"
        print(f"  {status} '{message}' (age {age}) → {result.intent.value} (expected: {expected.value})")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Intent Classifier Tests")
    print("=" * 60)

    await test_continue_intent()
    await test_new_story_intent()
    await test_exploring_intent()
    await test_greeting_intent()
    await test_edge_cases()

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
