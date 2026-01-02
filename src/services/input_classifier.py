"""
Input Classifier Service for Hybrid Chapter Generation

Classifies user input into tiers during story reading:
- Tier 1 (IMMEDIATE): Questions/comments with no story impact
- Tier 2 (PREFERENCE): Minor adjustments for next chapter
- Tier 3 (STORY_CHOICE): Plot forks affecting N+2 chapter
- Tier 4 (ADDITION): New subplot elements

Uses LLM to classify ambiguous input, with fast-path patterns for common cases.
Supports Microsoft Foundry Model Router for intelligent model selection.
"""

import re
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from src.models import InputTier, QueuedInput

logger = logging.getLogger(__name__)


# Fast-path patterns for immediate classification (no LLM needed)
# These bypass LLM classification entirely for ~200ms+ savings
TIER_1_PATTERNS = [
    # =================================================================
    # QUESTIONS - The vast majority of child questions are Tier 1
    # These patterns catch 90%+ of questions without LLM classification
    # =================================================================

    # What questions (English)
    r"^what\b",                                    # "What did Vikings eat?"
    r"what('s| is| are| was| were| did| do| does)\b",  # "What's a fjord?"

    # Who/Where/When questions (English)
    r"^who\b",                                     # "Who is Harald's father?"
    r"^where\b",                                   # "Where did they go?"
    r"^when\b",                                    # "When did Vikings live?"

    # Why/How questions (English)
    r"^why\b",                                     # "Why did he want to be king?"
    r"^how\b",                                     # "How did they build ships?"

    # Did/Do/Does questions (English)
    r"^did\b",                                     # "Did Vikings have horses?"
    r"^do\b",                                      # "Do they have magic?"
    r"^does\b",                                    # "Does Harald have a sword?"

    # Is/Are/Was/Were questions (English)
    r"^is\b",                                      # "Is there magic in the story?"
    r"^are\b",                                     # "Are Vikings scary?"
    r"^was\b",                                     # "Was Harald brave?"
    r"^were\b",                                    # "Were there dragons?"

    # Norwegian question words
    r"^hva\b",                                     # "Hva spiste vikingene?"
    r"^hvem\b",                                    # "Hvem er Harald?"
    r"^hvor\b",                                    # "Hvor bor de?"
    r"^når\b",                                     # "Når skjedde dette?"
    r"^hvorfor\b",                                 # "Hvorfor gjorde han det?"
    r"^hvordan\b",                                 # "Hvordan bygget de skip?"
    r"^(gjorde|gjør|hadde|har|var|er)\b.+\?",     # "Hadde vikingene hester?"

    # Questions about vocabulary/meaning
    r"what('s| is| does).*mean",
    r"how do you (say|spell|pronounce)",
    r"can you explain",
    r"tell me (about|more)",

    # =================================================================
    # ACKNOWLEDGMENTS - Simple reactions that need immediate response
    # =================================================================

    # Simple acknowledgments (English)
    r"^(cool|nice|wow|awesome|great|ok|okay|yay|yeah|ooh|whoa|huh)[\!\.\?]?$",
    r"^that('s| is) (cool|nice|awesome|great|interesting|funny|scary|amazing)",
    r"^i (like|love|liked|loved) (that|this|it|the)",
    r"^(really|so|that's|thats)\?$",              # "Really?" "So?"

    # Simple acknowledgments (Norwegian)
    r"^(kult|fint|wow|flott|bra|ok|ja|nei|ooh|oi)[\!\.\?]?$",
    r"^det (er|var) (kult|fint|flott|bra|interessant|morsomt|skummelt)",
    r"^jeg (liker|elsker|likte) (det|dette|den|historien)",

    # =================================================================
    # CONTINUE PROMPTS - User wants to proceed
    # =================================================================

    # Continue/next prompts (English)
    r"^(yes|yeah|sure|next|continue|go on|keep going|more|and then)[\!\.\?]?$",
    r"what happens next",
    r"tell me more",
    r"then what",
    r"and then\??$",

    # Continue/next prompts (Norwegian)
    r"^(ja|jo|videre|fortsett|mer|så)[\!\.\?]?$",
    r"hva skjer (videre|nå|så)",
    r"fortell mer",
    r"og så\??$",
]

# Style preferences - checked before TIER_3
TIER_2_PATTERNS = [
    # Style/tone preferences (in English)
    r"make it (more |less )?(funny|scary|exciting|sad|happy|silly|serious|adventurous|mysterious|romantic|dark|light|spooky)",
    r"(i want|let'?s have) (a |it to be )?(funny|scary|exciting|sad|happy|silly|serious|adventurous|mysterious)",
    r"it should be (more |less )?(funny|scary|exciting|sad|happy|silly)",
    r"(funnier|scarier|more exciting|less scary|more fun)",

    # Style/tone preferences (Norwegian)
    r"(gjør|lag) (den|det|historien) (mer |mindre )?(morsom|skummel|spennende|trist|glad|morsomt|skummelt|spennende)",
    r"(jeg vil|la oss ha) (en |at det skal være )?(morsom|skummel|spennende|trist|glad)",
    r"(den|det) (må|skal|bør) være (mer |mindre )?(morsom|skummel|spennende|morsomt|skummelt)",
    r"(morsommere|skumlere|mer spennende|mindre skummelt|mer gøy)",

    # Speed/pacing preferences
    r"(slower|faster|simpler|easier|harder)",
    r"(saktere|raskere|enklere|lettere|vanskeligere)",
    r"use (simpler|easier|harder) words",
    r"bruk (enklere|lettere|vanskeligere) ord",
]

TIER_3_PATTERNS = [
    # Explicit story requests with "can" or "add"
    r"can (she|he|they|the \w+|we) have",
    r"can there be",
    r"add a",
    r"i want (a|the|her|him|them) to",
    r"make (her|him|them|the \w+) (have|get|find|meet)",
    r"(she|he|they) should (have|get|find|meet)",

    # Norwegian story requests
    r"kan (hun|han|de|vi) (ha|få)",
    r"kan det være",
    r"legg til",
    r"jeg vil (at |ha )(hun|han|de|den|det) (skal )?",
]


@dataclass
class InputClassification:
    """Result of classifying user input"""
    tier: InputTier
    classified_intent: str
    preference_updates: Optional[Dict[str, Any]] = None
    story_direction: Optional[str] = None
    confidence: float = 1.0


class InputClassifier:
    """
    Classifies user input during story reading into tiers.

    Uses pattern matching for fast classification of common cases,
    falls back to LLM for ambiguous input.
    Supports Microsoft Foundry Model Router for intelligent model selection.
    """

    def __init__(
        self,
        foundry_endpoint: Optional[str] = None,
        foundry_api_key: Optional[str] = None,
        use_foundry: bool = False
    ):
        """
        Initialize the classifier.

        Args:
            foundry_endpoint: Azure AI Foundry endpoint URL
            foundry_api_key: Foundry API key (None for Managed Identity)
            use_foundry: Feature flag to enable Foundry (vs legacy direct APIs)
        """
        self._foundry_service = None
        self._use_foundry = use_foundry

        if use_foundry and foundry_endpoint:
            from src.services.foundry import FoundryService
            self._foundry_service = FoundryService(
                endpoint=foundry_endpoint,
                api_key=foundry_api_key,
                routing_mode="balanced"  # Fast, cost-effective for classification
            )
            logger.info("InputClassifier initialized with Foundry Model Router")

    async def classify(
        self,
        user_input: str,
        story_context: Dict[str, Any],
        current_chapter: int,
        generating_chapter: Optional[int]
    ) -> InputClassification:
        """
        Classify user input into a tier.

        Args:
            user_input: The user's message
            story_context: Current story state (title, characters, plot summary)
            current_chapter: Chapter user is currently reading
            generating_chapter: Chapter currently being generated (if any)

        Returns:
            InputClassification with tier and extracted intent
        """
        import time
        t_start = time.perf_counter()
        normalized = user_input.lower().strip()

        # Fast-path: Check Tier 1 patterns
        for pattern in TIER_1_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                t_end = time.perf_counter()
                logger.info(f"⚡ FAST PATTERN MATCH: Tier 1 in {(t_end - t_start) * 1000:.1f}ms (matched: '{pattern[:30]}...')")
                return InputClassification(
                    tier=InputTier.TIER_1_IMMEDIATE,
                    classified_intent=self._extract_question_intent(user_input),
                    confidence=0.95
                )

        # Fast-path: Check Tier 2 patterns (style preferences)
        # This must be BEFORE Tier 3 to catch "make it funny" vs "make him find"
        for pattern in TIER_2_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return InputClassification(
                    tier=InputTier.TIER_2_PREFERENCE,
                    classified_intent=user_input,
                    preference_updates={"style_adjustment": user_input},
                    confidence=0.9
                )

        # Fast-path: Check Tier 3 patterns
        for pattern in TIER_3_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return InputClassification(
                    tier=InputTier.TIER_3_STORY_CHOICE,
                    classified_intent=user_input,
                    story_direction=user_input,
                    confidence=0.9
                )

        # Slow-path: Use LLM for classification
        logger.info(f"🐢 NO PATTERN MATCH - falling back to LLM classification...")
        result = await self._classify_with_llm(user_input, story_context)
        t_end = time.perf_counter()
        logger.info(f"🐢 LLM CLASSIFICATION: Tier {result.tier.value} in {(t_end - t_start) * 1000:.0f}ms")
        return result

    def _extract_question_intent(self, user_input: str) -> str:
        """Extract the key question/intent from Tier 1 input."""
        # Try to extract the subject of the question
        match = re.search(r"what('s| is| does)\s+(.+?)(\s+mean|\?|$)", user_input, re.IGNORECASE)
        if match:
            return f"Definition: {match.group(2).strip()}"

        match = re.search(r"(explain|tell me about)\s+(.+)", user_input, re.IGNORECASE)
        if match:
            return f"Explanation: {match.group(2).strip()}"

        return user_input

    async def _classify_with_llm(
        self,
        user_input: str,
        story_context: Dict[str, Any]
    ) -> InputClassification:
        """Use LLM to classify ambiguous input via Foundry Model Router."""

        prompt = f"""You are classifying a child's input during an interactive story.

STORY CONTEXT:
Title: {story_context.get('title', 'Unknown')}
Current plot: {story_context.get('plot_summary', 'A story in progress')}
Characters: {', '.join(story_context.get('characters', []))}

CHILD'S MESSAGE:
"{user_input}"

Classify this into ONE of these tiers:

TIER_1_IMMEDIATE - No story impact. Examples:
- Questions about words/facts ("What's a fjord?", "Who was that?")
- Simple reactions ("Cool!", "That's scary", "I like the part where...")
- Commands to continue ("Yes", "Next", "Keep going")

TIER_2_PREFERENCE - Minor style adjustment. Examples:
- "Make it less scary"
- "I like the brave parts"
- "Can you use simpler words?"

TIER_3_STORY_CHOICE - Requests a specific plot change. Examples:
- "Can she have a pet wolf?"
- "Make him save the princess"
- "Add a dragon!"

TIER_4_ADDITION - New subplot or major element. Examples:
- "Can there be pirates too?"
- "I want a whole new adventure"
- "Add a mystery to solve"

Respond in this exact format:
TIER: [1, 2, 3, or 4]
INTENT: [What the child wants, in a brief phrase]
STORY_DIRECTION: [For Tier 3/4 only: What should change in the story]
PREFERENCE: [For Tier 2 only: What preference to adjust]
CONFIDENCE: [0.0-1.0 how certain you are]
"""

        # Use Foundry Model Router for classification
        if self._foundry_service:
            try:
                response = await self._foundry_service.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    routing_mode="balanced",  # Fast, cost-effective for classification
                    max_tokens=200,
                    temperature=0.1  # Low temperature for consistent classification
                )
                logger.debug(f"Classification used model: {response.get('model', 'unknown')}")
                return self._parse_llm_response(response["content"], user_input)
            except Exception as foundry_error:
                logger.warning(f"Foundry classification failed: {foundry_error}")

        # Fallback to Tier 1 if Foundry unavailable or fails
        logger.warning("No LLM available for classification, defaulting to Tier 1")
        return InputClassification(
            tier=InputTier.TIER_1_IMMEDIATE,
            classified_intent=user_input,
            confidence=0.5
        )

    def _parse_llm_response(self, response: str, original_input: str) -> InputClassification:
        """Parse the LLM's classification response."""
        lines = response.strip().split('\n')

        tier_num = 1
        intent = original_input
        story_direction = None
        preference_updates = None
        confidence = 0.8

        for line in lines:
            line = line.strip()
            if line.startswith('TIER:'):
                try:
                    tier_num = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('INTENT:'):
                intent = line.split(':', 1)[1].strip() if ':' in line else intent
            elif line.startswith('STORY_DIRECTION:'):
                story_direction = line.split(':', 1)[1].strip() if ':' in line else None
            elif line.startswith('PREFERENCE:'):
                pref = line.split(':', 1)[1].strip() if ':' in line else None
                if pref and pref.lower() not in ['none', 'n/a', '-']:
                    preference_updates = {'adjustment': pref}
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

        # Map tier number to enum
        tier_map = {
            1: InputTier.TIER_1_IMMEDIATE,
            2: InputTier.TIER_2_PREFERENCE,
            3: InputTier.TIER_3_STORY_CHOICE,
            4: InputTier.TIER_4_ADDITION,
        }
        tier = tier_map.get(tier_num, InputTier.TIER_1_IMMEDIATE)

        return InputClassification(
            tier=tier,
            classified_intent=intent,
            story_direction=story_direction if tier_num in [3, 4] else None,
            preference_updates=preference_updates if tier_num == 2 else None,
            confidence=confidence
        )

    def create_queued_input(
        self,
        classification: InputClassification,
        raw_input: str,
        target_chapter: int
    ) -> QueuedInput:
        """Create a QueuedInput from classification result."""
        return QueuedInput(
            tier=classification.tier,
            raw_input=raw_input,
            classified_intent=classification.classified_intent,
            target_chapter=target_chapter,
            preference_updates=classification.preference_updates,
            story_direction=classification.story_direction,
        )

    def calculate_target_chapter(
        self,
        classification: InputClassification,
        current_chapter: int,
        generating_chapter: Optional[int]
    ) -> int:
        """
        Calculate which chapter should be affected by this input.

        Rules:
        - Tier 1: N/A (immediate response, no chapter affected)
        - Tier 2: Next chapter (current + 1), unless that's generating
        - Tier 3/4: Chapter N+2 or next available after generating
        """
        if classification.tier == InputTier.TIER_1_IMMEDIATE:
            # Tier 1 doesn't affect chapters
            return 0

        if classification.tier == InputTier.TIER_2_PREFERENCE:
            # Prefer next chapter, but skip if it's generating
            target = current_chapter + 1
            if generating_chapter and target == generating_chapter:
                target = generating_chapter + 1
            return target

        # Tier 3 and 4: At least N+2, but never modify what's generating
        target = current_chapter + 2
        if generating_chapter and target <= generating_chapter:
            target = generating_chapter + 1

        return target


# Convenience function for quick classification
async def classify_input(
    user_input: str,
    story_context: Dict[str, Any],
    current_chapter: int = 1,
    generating_chapter: Optional[int] = None,
    foundry_endpoint: Optional[str] = None,
    foundry_api_key: Optional[str] = None,
    use_foundry: Optional[bool] = None
) -> InputClassification:
    """
    Quick classification function for use in handlers.

    Uses pattern matching first, LLM only if needed.
    Uses Microsoft Foundry Model Router when enabled.
    """
    from src.config import get_settings
    settings = get_settings()

    # Use settings if not provided
    if foundry_endpoint is None:
        foundry_endpoint = settings.foundry_endpoint
    if foundry_api_key is None:
        foundry_api_key = settings.foundry_api_key
    if use_foundry is None:
        use_foundry = settings.use_foundry

    classifier = InputClassifier(
        foundry_endpoint=foundry_endpoint,
        foundry_api_key=foundry_api_key,
        use_foundry=use_foundry
    )
    return await classifier.classify(
        user_input,
        story_context,
        current_chapter,
        generating_chapter
    )
