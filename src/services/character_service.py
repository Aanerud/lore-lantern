"""
Character Service for Norse-Aware Identity Matching and Deduplication

This module provides character management functionality:
- Norse-aware name matching (patronymics, epithets, spelling variants)
- Character deduplication using LLM reasoning
- Character merging operations

Architecture:
- Pure functions for matching logic (no external dependencies)
- CharacterService class for operations requiring storage/LLM access
- Called by coordinator for character creation and management
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable

from src.models.models import Character
from src.services.text_processing import (
    normalize_norse_name,
    extract_patronymic,
    extract_first_name
)

logger = logging.getLogger(__name__)


# =========================================================================
# PURE CHARACTER MATCHING FUNCTIONS
# =========================================================================

def normalize_character_names(
    featured_names: List[str],
    existing_characters: List[Character]
) -> List[str]:
    """
    Normalize character names from NarrativeAgent output to match existing characters.

    This prevents character variants like "Harald Halfdansson" when "Harald Fairhair" exists.
    Uses fuzzy matching to find the closest existing character name.

    Args:
        featured_names: List of character names from NarrativeAgent output
        existing_characters: List of existing Character objects

    Returns:
        List of normalized character names that match existing characters
    """
    if not existing_characters or not featured_names:
        return featured_names

    # Build lookup of existing character names
    existing_names = [c.name for c in existing_characters]
    existing_lower = {n.lower().strip(): n for n in existing_names}
    existing_first_names = {}
    for name in existing_names:
        parts = name.split()
        if parts:
            first = parts[0].lower().strip()
            if first not in existing_first_names:
                existing_first_names[first] = name

    normalized = []
    for name in featured_names:
        name_lower = name.lower().strip()

        # Check 1: Exact match
        if name_lower in existing_lower:
            normalized.append(existing_lower[name_lower])
            continue

        # Check 2: First name match (e.g., "Harald Halfdansson" → "Harald")
        name_parts = name_lower.split()
        if name_parts and name_parts[0] in existing_first_names:
            normalized.append(existing_first_names[name_parts[0]])
            logger.debug(f"CharacterNorm: Normalized '{name}' → '{existing_first_names[name_parts[0]]}' (first name match)")
            continue

        # Check 3: Partial match (name contains or is contained by existing)
        matched = False
        for ex_lower, ex_original in existing_lower.items():
            if ex_lower in name_lower or name_lower in ex_lower:
                normalized.append(ex_original)
                logger.debug(f"CharacterNorm: Normalized '{name}' → '{ex_original}' (partial match)")
                matched = True
                break

        if not matched:
            # No match found - keep original (may be a new character introduced in prose)
            normalized.append(name)

    return normalized


def check_relationship_match(
    new_name: str,
    char: Character
) -> bool:
    """
    Check if new_name's patronymic matches character's known father.

    Example:
    - new_name: "Harald Halfdansson"
    - char.relationships: {"father": "King Halfdan the Black"}
    - Returns: True (Halfdansson = son of Halfdan)

    Args:
        new_name: Name to check for patronymic
        char: Existing character with potential relationship data

    Returns:
        True if patronymic matches father relationship
    """
    new_patronymic = extract_patronymic(new_name)
    if not new_patronymic:
        return False

    # Check if character has a father relationship
    relationships = getattr(char, 'relationships', {}) or {}
    father_name = relationships.get('father', '')
    if not father_name:
        return False

    # Extract father's first name and compare
    father_first = extract_first_name(father_name)
    if not father_first:
        return False

    # Patronymic matches father's name?
    return new_patronymic == father_first


# Known spelling variations (Norwegian/English equivalents)
SPELLING_EQUIVALENTS = {
    'harald fairhair': ['harald hårfagre', 'harald harfagre', 'harald tanglehair'],
    'harald hårfagre': ['harald fairhair', 'harald harfagre', 'harald tanglehair'],
    'halfdan the black': ['halvdan svarte', 'halfdan svarte', 'halfdan svartr'],
    'halvdan svarte': ['halfdan the black', 'halfdan svarte', 'halfdan svartr'],
    'eric bloodaxe': ['eirik blodøks', 'erik bloodaxe', 'eirik bloodaxe'],
    'eirik blodøks': ['eric bloodaxe', 'erik bloodaxe', 'eirik bloodaxe'],
}


# =========================================================================
# SEMANTIC BATCHING FUNCTIONS
# =========================================================================

def get_similarity_key(name: str) -> str:
    """
    Generate similarity key for name-based grouping.
    Characters with same key must go to different batches.

    Uses existing functions:
    - extract_first_name() - skip titles, epithets
    - normalize_norse_name() - handle accents (í→i, å→a)
    - SPELLING_EQUIVALENTS - known variants

    Example:
        "Harald Fairhair" -> "harald"
        "Harald Harfagre" -> "harald" (same key - same person!)
        "Astrid Eiriksdottir" -> "astrid"

    Args:
        name: Character name to generate key for

    Returns:
        Normalized similarity key (lowercase first name)
    """
    first_name = extract_first_name(name)
    normalized = normalize_norse_name(first_name)

    # Check spelling equivalents for canonical form
    full_lower = name.lower().strip()
    for canonical, variants in SPELLING_EQUIVALENTS.items():
        if full_lower == canonical or full_lower in variants:
            # Use canonical form's first name
            return normalize_norse_name(extract_first_name(canonical.split()[0]))

    return normalized


def assign_to_semantic_batches(
    characters: list,
    max_batch_size: int = 4,
    get_name=None
) -> list:
    """
    Assign characters to semantic batches where similar characters
    go to DIFFERENT batches to ensure visibility during creation.

    Rules:
    1. Same normalized first name -> different batches
    2. Known spelling variants -> different batches
    3. Each batch capped at max_batch_size

    This allows:
    - Sequential processing within each batch (full visibility)
    - Parallel processing across batches (performance)

    Args:
        characters: List of character specs (dicts or objects with 'name')
        max_batch_size: Maximum characters per batch (default 4)
        get_name: Optional function to extract name from character spec

    Returns:
        List of batches, where each batch is a list of character specs
    """
    if not characters:
        return []

    # Default name extractor
    if get_name is None:
        def get_name(c):
            if isinstance(c, dict):
                return c.get('name', '')
            return getattr(c, 'name', '')

    batches = []
    similarity_to_batch_indices = {}  # sim_key -> list of batch indices containing that key

    for char in characters:
        name = get_name(char)
        sim_key = get_similarity_key(name)

        # Find eligible batch (no same sim_key already, and has room)
        used_batches = similarity_to_batch_indices.get(sim_key, [])
        assigned = False

        for batch_idx, batch in enumerate(batches):
            if batch_idx not in used_batches and len(batch) < max_batch_size:
                batch.append(char)
                similarity_to_batch_indices.setdefault(sim_key, []).append(batch_idx)
                assigned = True
                break

        if not assigned:
            # Create new batch
            batches.append([char])
            similarity_to_batch_indices.setdefault(sim_key, []).append(len(batches) - 1)

    return batches


def match_character_identity(
    new_name: str,
    existing_characters: List[Character]
) -> Optional[Character]:
    """
    Norse-aware character identity matching.

    SAFE matching that respects patronymics:
    - Different patronymics = NEVER match (different fathers)
    - Same patronymic OR compatible = consider matching

    Matching Rules:
    1. Exact name match (case-insensitive)
    2. Known spelling variations (Harald Fairhair = Harald Hårfagre)
    3. Same first name + compatible patronymics (no conflict)
    4. Patronymic matches known father relationship

    NEVER matches if patronymics conflict:
    - "Olaf Halfdansson" CANNOT match "Olaf Sigurdsson" (different fathers)

    Args:
        new_name: The new character name to match
        existing_characters: List of existing Character objects

    Returns:
        Matching Character object if found, else None
    """
    if not existing_characters or not new_name:
        return None

    new_lower = new_name.lower().strip()
    new_normalized = normalize_norse_name(new_name)
    new_first = normalize_norse_name(extract_first_name(new_name))
    new_patronymic = normalize_norse_name(extract_patronymic(new_name) or "")

    logger.debug(f"CharacterMatch: 🔍 Match check: new='{new_name}' → first='{new_first}', patronymic='{new_patronymic}'")

    for char in existing_characters:
        existing_lower = char.name.lower().strip()
        existing_normalized = normalize_norse_name(char.name)
        existing_first = normalize_norse_name(extract_first_name(char.name))
        existing_patronymic = normalize_norse_name(extract_patronymic(char.name) or "")

        logger.debug(f"CharacterMatch:    🔍 vs '{char.name}' → first='{existing_first}', patronymic='{existing_patronymic}'")

        # Rule 1: Exact match (case-insensitive, or normalized match)
        if new_lower == existing_lower or new_normalized == existing_normalized:
            logger.info(f"✅ Exact match: '{new_name}' = '{char.name}'")
            return char

        # Rule 2: Known spelling variations
        if new_lower in SPELLING_EQUIVALENTS:
            if existing_lower in SPELLING_EQUIVALENTS[new_lower]:
                logger.info(
                    f"📝 Character spelling match: '{new_name}' = '{char.name}' (known variation)"
                )
                return char
        if existing_lower in SPELLING_EQUIVALENTS:
            if new_lower in SPELLING_EQUIVALENTS[existing_lower]:
                logger.info(
                    f"📝 Character spelling match: '{new_name}' = '{char.name}' (known variation)"
                )
                return char

        # Rule 3: CONFLICT CHECK - Different patronymics = different people!
        if new_patronymic and existing_patronymic:
            if new_patronymic != existing_patronymic:
                # "Halfdansson" vs "Sigurdsson" = DIFFERENT people - skip
                continue

        # Rule 4: Same first name + compatible patronymics
        if new_first and existing_first and new_first == existing_first:
            logger.debug(f"CharacterMatch:    ✓ First name match: '{new_first}' = '{existing_first}'")
            # Patronymics are compatible if: same, or one/both missing
            patronymics_compatible = (
                new_patronymic == existing_patronymic or  # Same patronymic
                not new_patronymic or  # New has no patronymic
                not existing_patronymic  # Existing has no patronymic
            )

            if patronymics_compatible:
                # Extra confidence: check if patronymic matches relationship
                if new_patronymic and check_relationship_match(new_name, char):
                    logger.info(
                        f"📝 Patronymic+relationship match: '{new_name}' → '{char.name}'"
                    )
                    return char

                # If same role, more likely same character
                char_role = getattr(char, 'role', '') or ''
                if char_role:
                    logger.info(
                        f"📝 First name match (compatible patronymics): '{new_name}' → '{char.name}'"
                    )
                    return char

    # No match found - this is likely a DIFFERENT character
    return None


# =========================================================================
# CHARACTER SERVICE CLASS (for operations needing external dependencies)
# =========================================================================

# Type alias for the LLM caller function
LLMCaller = Callable[[str, str, str], Awaitable[str]]


class CharacterService:
    """
    Character management service for operations requiring storage or LLM access.

    This class handles:
    - LLM-based character deduplication (using Bill/FactCheck agent)
    - Character merging in the database

    The simpler matching functions are standalone (above) for easier testing.
    """

    def __init__(
        self,
        storage,
        llm_caller: LLMCaller,
        app_logger=None
    ):
        """
        Initialize CharacterService with dependencies.

        Args:
            storage: Storage service (Firebase or Azure SQL) with delete_character method
            llm_caller: Async function to call LLM: (prompt, system_prompt, model_preference) -> response
            app_logger: Optional custom logger (uses module logger if not provided)
        """
        self.storage = storage
        self.llm_caller = llm_caller
        self.logger = app_logger or logger

    async def dedupe_characters_with_llm(
        self,
        story_id: str,
        characters: List[Character],
        story_context: str
    ) -> List[Tuple[str, str]]:
        """
        Ask Bill (FactCheck agent) to identify duplicate characters using LLM reasoning.

        This is a safety net that runs AFTER rule-based matching.
        Bill reviews all characters with Norse naming expertise.

        Args:
            story_id: Story ID for logging
            characters: List of all Character objects
            story_context: Brief story description for context

        Returns:
            List of (duplicate_name, original_name) pairs to merge
        """
        if len(characters) <= 1:
            return []

        self.logger.info(f"🔍 Bill reviewing {len(characters)} characters for duplicates...")

        # Build character list for Bill
        char_list = []
        for c in characters:
            char_info = {
                "name": c.name,
                "role": getattr(c, 'role', '') or '',
                "personality_traits": (getattr(c, 'personality_traits', []) or [])[:3]
            }
            relationships = getattr(c, 'relationships', {}) or {}
            if relationships:
                char_info["relationships"] = relationships
            char_list.append(char_info)

        prompt = f"""NORSE NAMING EXPERT TASK - Identify DUPLICATE Characters

You are Bill, a historical fact-checker with expertise in Norse naming conventions.

NORSE NAMING RULES:
- "-sson" suffix means "son of" (e.g., Halfdansson = son of Halfdan)
- "-dottir" suffix means "daughter of" (e.g., Håkonsdottir = daughter of Haakon)
- DIFFERENT patronymics = DIFFERENT people (Olaf Halfdansson ≠ Olaf Sigurdsson - they have different fathers!)
- Same first name + same/compatible patronymic = likely SAME person
- Epithets can vary (Fairhair = Hårfagre, the Black = Svarte)
- Titles can change (Jarl → King)

STORY CONTEXT: {story_context}

CHARACTERS TO REVIEW:
{json.dumps(char_list, indent=2, ensure_ascii=False)}

TASK: Identify DEFINITE duplicates only.

CRITICAL RULES:
1. NEVER mark two characters as duplicates if they have DIFFERENT patronymics
   - "Jarl Haakon Sigurdsson" and "Jarl Haakon Eiriksson" are DIFFERENT people
2. Only mark as duplicate if you are CERTAIN they are the same person
3. When in doubt, do NOT include - better to have extra characters than merge wrong ones

OUTPUT: Return ONLY valid JSON (no markdown, no explanation):
{{
    "duplicates": [
        {{"duplicate": "name to remove", "original": "name to keep", "reason": "brief explanation"}}
    ]
}}

If no duplicates found, return: {{"duplicates": []}}
"""

        try:
            # Use the factcheck agent's LLM for Bill
            response = await self.llm_caller(
                prompt,
                "You are Bill, a historical Norse naming expert. Respond ONLY with valid JSON.",
                "balanced"
            )

            # Parse response
            response_text = response.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            duplicates = result.get("duplicates", [])

            if duplicates:
                self.logger.info(f"📝 Bill found {len(duplicates)} potential duplicates:")
                for dup in duplicates:
                    self.logger.info(f"   - '{dup.get('duplicate')}' → '{dup.get('original')}' ({dup.get('reason', 'no reason')})")

                # Convert to tuple list
                return [(d["duplicate"], d["original"]) for d in duplicates if "duplicate" in d and "original" in d]
            else:
                self.logger.info("✅ Bill found no duplicates - all characters are unique")
                return []

        except json.JSONDecodeError as e:
            self.logger.warning(f"⚠️ Bill dedup response not valid JSON: {e}")
            return []
        except Exception as e:
            self.logger.error("CharacterService", f"Bill dedup check failed: {e}")
            return []

    async def merge_characters(
        self,
        story_id: str,
        duplicate_name: str,
        original_name: str,
        characters: List[Character]
    ) -> bool:
        """
        Merge a duplicate character into the original.

        This removes the duplicate from the database and logs the merge.
        The original character is kept with its existing data.

        Args:
            story_id: Story ID
            duplicate_name: Name of character to remove
            original_name: Name of character to keep
            characters: List of all Character objects

        Returns:
            True if merge was successful, False otherwise
        """
        try:
            # Find the duplicate character
            duplicate_char = None
            original_char = None

            for char in characters:
                if char.name.lower().strip() == duplicate_name.lower().strip():
                    duplicate_char = char
                elif char.name.lower().strip() == original_name.lower().strip():
                    original_char = char

            if not duplicate_char:
                self.logger.warning(f"⚠️ Merge failed: duplicate '{duplicate_name}' not found")
                return False

            if not original_char:
                self.logger.warning(f"⚠️ Merge failed: original '{original_name}' not found")
                return False

            # Get character ID for deletion
            char_id = getattr(duplicate_char, 'id', None) or getattr(duplicate_char, 'character_id', None)
            if not char_id:
                self.logger.warning(f"⚠️ Merge failed: no ID found for '{duplicate_name}'")
                return False

            # Delete the duplicate from storage
            await self.storage.delete_character(story_id, char_id)

            self.logger.info(f"🔗 Merged character: '{duplicate_name}' → '{original_name}'")
            return True

        except Exception as e:
            self.logger.error("CharacterService", f"CharacterMerge: Failed to merge '{duplicate_name}' → '{original_name}': {e}")
            return False
