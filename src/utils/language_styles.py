"""
Language Style Definitions for Multilingual Story Generation

This module provides language-specific instructions for:
- Prose generation (NarrativeAgent)
- Dialogue generation (CompanionAgent)
- Text-to-Speech (TTS)

Architecture:
- Internal operations (Structure, Character, FactCheck) stay in English
- Child-facing output (Prose, Dialogue, TTS) uses parent's language
"""

from typing import Dict, Any


LANGUAGE_STYLES: Dict[str, Dict[str, Any]] = {
    "en": {
        "name": "English",
        "tts_code": "en-US",
        "style_instruction": "Write in clear, engaging English prose.",
        "dialogue_instruction": "Speak in warm, friendly English.",
        "storytelling_tradition": "Use classic storytelling techniques with vivid imagery."
    },
    "no": {
        "name": "Norwegian",
        "tts_code": "nb-NO",
        "style_instruction": (
            "Write in Norwegian (Bokmål). "
            "Use warm Nordic storytelling traditions with cozy, atmospheric descriptions. "
            "Embrace 'kos' (coziness) in your narrative style."
        ),
        "dialogue_instruction": (
            "Snakk norsk på en varm og vennlig måte. "
            "Bruk enkelt språk som passer for barn."
        ),
        "storytelling_tradition": (
            "Draw from Scandinavian folklore traditions - "
            "atmospheric nature descriptions, magical realism, and cozy indoor scenes."
        )
    },
    "es": {
        "name": "Spanish",
        "tts_code": "es-ES",
        "style_instruction": (
            "Write in Spanish (Castilian). "
            "Use vivid, expressive language with warmth and emotion. "
            "Embrace the musicality of Spanish prose."
        ),
        "dialogue_instruction": (
            "Habla español de manera cálida y amigable. "
            "Usa lenguaje simple apropiado para niños."
        ),
        "storytelling_tradition": (
            "Draw from rich Latin storytelling traditions - "
            "expressive emotion, family bonds, and magical realism."
        )
    }
}


def get_language_style(lang_code: str) -> Dict[str, Any]:
    """
    Get language style configuration for the given language code.

    Args:
        lang_code: Two-letter language code (en, no, es)

    Returns:
        Dictionary with language name, TTS code, and style instructions.
        Falls back to English if language code not found.
    """
    return LANGUAGE_STYLES.get(lang_code, LANGUAGE_STYLES["en"])


def get_prose_instruction(lang_code: str) -> str:
    """
    Get the prose writing instruction for NarrativeAgent.

    Args:
        lang_code: Two-letter language code

    Returns:
        Full instruction string for prose generation
    """
    style = get_language_style(lang_code)
    return f"""
=== LANGUAGE REQUIREMENT ===
Write this chapter in {style['name']}.
{style['style_instruction']}
{style['storytelling_tradition']}
"""


def get_dialogue_instruction(lang_code: str) -> str:
    """
    Get the dialogue instruction for CompanionAgent.

    Args:
        lang_code: Two-letter language code

    Returns:
        Instruction string for dialogue generation
    """
    style = get_language_style(lang_code)
    return f"""
LANGUAGE: You MUST respond in {style['name']}.
{style['dialogue_instruction']}
"""


def get_tts_language_code(lang_code: str) -> str:
    """
    Get the TTS language code for voice synthesis.

    Args:
        lang_code: Two-letter language code (en, no, es)

    Returns:
        Full TTS language code (e.g., 'nb-NO' for Norwegian)
    """
    style = get_language_style(lang_code)
    return style['tts_code']
