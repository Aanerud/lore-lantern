"""
Norwegian Language Refinement Prompts

This module provides prompts for refining Norwegian text using the NB AI Lab
Borealis model (or any Norwegian-capable LLM).

The goal is to make translated/AI-generated Norwegian sound more natural by:
- Fixing unnatural word choices common in machine translation
- Adjusting word order from English patterns to Norwegian patterns
- Replacing overly formal words with more conversational alternatives
- Substituting English loanwords with Norwegian equivalents

Usage:
    from src.prompts.refine.no import get_refinement_prompt, validate_response

    prompt = get_refinement_prompt(paragraph)
    # ... call LLM ...
    cleaned = validate_response(response, original_paragraph)
"""

# Language metadata
LANGUAGE_CODE = "no"
LANGUAGE_NAME = "Norwegian"
LANGUAGE_NATIVE_NAME = "Norsk"

# Recommended model (from models.yaml)
RECOMMENDED_MODEL = "ollama/hf.co/NbAiLab/borealis-4b-instruct-preview-gguf:BF16"


def get_refinement_prompt(paragraph: str) -> str:
    """
    Generate a refinement prompt for Norwegian text.

    Optimized for small models like Borealis 4B:
    - Specific, concise instructions
    - Few-shot examples
    - Clear output expectations

    Args:
        paragraph: Norwegian text to refine

    Returns:
        Complete prompt for the LLM
    """
    return f"""Forbedre denne norske teksten. Se etter:

1. Unaturlige ordvalg (f.eks. "gjøre en beslutning" → "ta en beslutning")
2. Stiv ordstilling fra engelsk (f.eks. "Han var veldig glad" → "Han ble kjempeglad")
3. Formelle ord som kan være mer muntlige (f.eks. "imidlertid" → "men", "dessuten" → "og")
4. Engelske lånord med norske alternativer (f.eks. "basically" → "egentlig")
5. Verbbøying: Behold riktig tid. Etter sanseverb (hørte, så, kjente) bruk infinitiv.

Eksempler:
ORIGINAL: Hun gjorde en beslutning om å forlate stedet.
BEDRE: Hun bestemte seg for å dra.

ORIGINAL: Det var veldig interessant for ham å se dette.
BEDRE: Han syntes det var spennende å se.

ORIGINAL: Han hadde ikke noen idé om hva som skjedde.
BEDRE: Han ante ikke hva som foregikk.

ORIGINAL: Han hørte døren slå igjen.
BEDRE: Han hørte døren smelle igjen. (infinitiv etter "hørte" er korrekt)

Nå, forbedre denne teksten. Behold handlingen og stemningen. Skriv KUN den forbedrede teksten:

{paragraph}"""


def validate_response(response: str, original: str) -> str:
    """
    Validate and clean the LLM's response.

    Small models sometimes add preambles like "Her er den forbedrede teksten:"
    This function strips those out and returns just the refined content.

    Args:
        response: Raw LLM response
        original: Original paragraph (fallback if response is invalid)

    Returns:
        Cleaned refined text, or original if response is invalid
    """
    if not response:
        return original

    refined = response.strip()

    # Too short - probably failed
    if len(refined) < 20:
        return original

    # Check for common preamble patterns that small models add
    problem_indicators = [
        "her er",
        "forbedret tekst:",
        "endringer:",
        "jeg har",
        "teksten er",
        "original:",
        "###",
        "bedre:",
        "resultat:",
    ]

    refined_lower = refined.lower()

    for indicator in problem_indicators:
        if refined_lower.startswith(indicator):
            # Model added preamble - try to extract just the text
            lines = refined.split('\n')
            for line in lines:
                line = line.strip()
                # Find a substantial line without preamble words
                if len(line) > 50 and not any(ind in line.lower() for ind in problem_indicators):
                    return line
            # Couldn't extract clean text
            return original

    # Check if response is too different (model hallucinated new content)
    original_words = set(original.lower().split())
    refined_words = set(refined.lower().split())
    overlap = len(original_words & refined_words) / max(len(original_words), 1)

    if overlap < 0.3:
        # Less than 30% word overlap - probably hallucination
        return original

    return refined


# Pattern examples for documentation/testing
PATTERN_EXAMPLES = [
    {
        "original": "Hun gjorde en beslutning om å forlate stedet.",
        "refined": "Hun bestemte seg for å dra.",
        "pattern": "gjøre en beslutning → ta en beslutning / bestemme seg",
    },
    {
        "original": "Det var veldig interessant for ham å se dette.",
        "refined": "Han syntes det var spennende å se.",
        "pattern": "X var Y for ham → Han syntes X var Y",
    },
    {
        "original": "Han hadde ikke noen idé om hva som skjedde.",
        "refined": "Han ante ikke hva som foregikk.",
        "pattern": "hadde ikke noen idé → ante ikke",
    },
    {
        "original": "Hun var veldig glad for å se ham.",
        "refined": "Hun ble kjempeglad for å se ham.",
        "pattern": "var veldig glad → ble kjempeglad",
    },
    {
        "original": "De skulle basically bare gå hjem.",
        "refined": "De skulle egentlig bare gå hjem.",
        "pattern": "basically → egentlig",
    },
    {
        "original": "Han hørte staven slå mot steinene.",
        "refined": "Han hørte staven smelle mot steinene.",
        "pattern": "sanseverb + infinitiv (hørte X slå → hørte X smelle)",
        "note": "Infinitiv er grammatisk korrekt etter sanseverb som hørte, så, kjente",
    },
]
