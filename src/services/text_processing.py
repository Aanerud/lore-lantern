"""
Text Processing Service for Norse Name Handling and Text Utilities

This module provides pure text processing functions for:
- Norse name normalization (Unicode handling, accent removal)
- Patronymic extraction (-sson, -dottir suffixes)
- First name extraction (skipping titles and epithets)
- Reviewer name normalization (LLM output cleanup)

These are standalone functions (not class methods) for easy reuse
across agents and services.

Architecture:
- Pure functions with no external dependencies (except unicodedata)
- Called by CharacterService for identity matching
- Called by Coordinator for reviewer name normalization
"""

import unicodedata
from typing import Optional


# =========================================================================
# REVIEWER NAME NORMALIZATION
# =========================================================================

# LLM sometimes returns full names in JSON - map to consistent short names
# for frontend data-reviewer matching
REVIEWER_NAME_MAP = {
    "guillermo del toro": "Guillermo",
    "guillermo": "Guillermo",
    "bill nye": "Bill",
    "bill bryson": "Bill",
    "bill": "Bill",
    "clarissa pinkola estés": "Clarissa",
    "clarissa pinkola estes": "Clarissa",
    "clarissa": "Clarissa",
    "benjamin zander": "Benjamin",
    "benjamin dreyer": "Benjamin",
    "benjamin": "Benjamin",
    "continuity checker": "Continuity",
    "continuity": "Continuity",
    "stephen king": "Stephen",
    "stephen": "Stephen",
}


def normalize_reviewer_name(name: str) -> str:
    """
    Normalize reviewer name to short form for frontend matching.

    LLM agents sometimes return full names like "Bill Nye" or
    "Guillermo del Toro" in their JSON output. This normalizes
    to consistent short names (Bill, Guillermo) for frontend matching.

    Args:
        name: Full or partial reviewer name from LLM output

    Returns:
        Short normalized name (e.g., "Bill", "Guillermo", "Clarissa")
        Returns "Unknown" if name is empty
        Returns original name if no mapping found
    """
    if not name:
        return "Unknown"
    return REVIEWER_NAME_MAP.get(name.lower(), name)


# =========================================================================
# NORSE NAME UTILITIES
# =========================================================================

def normalize_norse_name(name: str) -> str:
    """
    Normalize Norse names for comparison.

    Handles:
    - Accented characters (í → i, å → a, ø → o)
    - Case normalization (to lowercase)
    - Special Nordic characters that don't decompose well

    Example: "Eiríksdottir" → "eiriksdottir"
    Example: "Håkon" → "hakon"
    Example: "Þórr" → "thorr"

    Args:
        name: Norse name with possible accents/special chars

    Returns:
        Lowercase ASCII-normalized version for comparison
    """
    if not name:
        return ""

    # NFKD decomposition separates base char from combining chars
    normalized = unicodedata.normalize('NFKD', name)
    # Remove combining characters (accents)
    ascii_only = ''.join(c for c in normalized if not unicodedata.combining(c))

    # Handle special Nordic characters that don't decompose well
    replacements = {
        'ø': 'o', 'Ø': 'O',
        'æ': 'ae', 'Æ': 'AE',
        'ð': 'd', 'Ð': 'D',
        'þ': 'th', 'Þ': 'TH',
    }
    for old, new in replacements.items():
        ascii_only = ascii_only.replace(old, new)

    return ascii_only.lower().strip()


def extract_patronymic(name: str) -> Optional[str]:
    """
    Extract patronymic from Norse name.

    Norse naming convention:
    - "-sson" = son of (e.g., Halfdansson = son of Halfdan)
    - "-dottir" = daughter of (e.g., Håkonsdottir = daughter of Håkon)

    This extracts the father's name from the patronymic suffix.

    Examples:
        "Halfdansson" → "halfdan" (father's name)
        "Håkonsdottir" → "håkon" (father's name)
        "Harald Fairhair" → None (no patronymic)

    Args:
        name: Full Norse name possibly containing patronymic

    Returns:
        Father's name if patronymic found, else None
    """
    if not name:
        return None

    name_lower = name.lower()

    # Find -sson or -dottir suffix in any word
    # Strip punctuation from each part to handle names like "Eiríksdottir,"
    for raw_part in name_lower.split():
        part = raw_part.strip('.,;:()[]')

        # Handle -sson variants (son of)
        if part.endswith('sson'):
            return part[:-4]  # Remove "sson"
        if part.endswith('son') and len(part) > 3:
            # Be careful not to match names that just end in "son"
            # Check if it looks like a patronymic (has a name before 'son')
            potential_father = part[:-3]
            if len(potential_father) >= 3:  # Reasonable name length
                return potential_father

        # Handle -dottir variants (daughter of)
        if part.endswith('dottir'):
            return part[:-6]  # Remove "dottir"
        if part.endswith('dotter'):
            return part[:-6]  # Remove "dotter"
        if part.endswith('sdottir'):
            return part[:-7]  # Remove "sdottir" (e.g., "Håkonsdottir")
        if part.endswith('sdotter'):
            return part[:-7]  # Remove "sdotter"

    return None


# Common Norse titles (English and Norwegian)
NORSE_TITLES = frozenset([
    'king', 'queen', 'jarl', 'earl', 'skald', 'prince', 'princess',
    'kong', 'dronning', 'hersker'
])

# Common epithets/descriptors in Norse names
NORSE_EPITHETS = frozenset([
    'black', 'svarte', 'fairhair', 'hårfagre', 'harfagre',
    'bloodaxe', 'blodøks', 'ironbrow', 'storm-brow'
])

# Words to skip when extracting first name
SKIP_WORDS = frozenset(['the', 'den', 'of', 'av'])


def extract_first_name(name: str) -> str:
    """
    Extract first name from Norse name, skipping titles and epithets.

    Handles complex Norse names with titles and descriptors:
    - "King Halfdan the Black" → "halfdan"
    - "Jarl Eirik Bloodaxe" → "eirik"
    - "Harald Hårfagre" → "harald"
    - "Astrid Håkonsdottir" → "astrid"

    Args:
        name: Full Norse name with possible titles/epithets

    Returns:
        First name (lowercase) or empty string if not found
    """
    if not name:
        return ""

    # Split and strip punctuation from each part
    parts = [p.strip('.,;:()[]') for p in name.lower().split()]

    for part in parts:
        # Skip titles
        if part in NORSE_TITLES:
            continue
        # Skip patronymics
        if (part.endswith('sson') or part.endswith('son') or
            part.endswith('dottir') or part.endswith('dotter')):
            continue
        # Skip common articles/prepositions
        if part in SKIP_WORDS:
            continue
        # Skip common epithets
        if part in NORSE_EPITHETS:
            continue
        # This is likely the first name
        return part

    # Fallback to first part if nothing else matches
    return parts[0] if parts else ""
