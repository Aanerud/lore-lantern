"""
Language Refinement Prompts

This package contains per-language refinement prompts for post-processing
AI-generated story content.

Structure:
    refine/
    ├── __init__.py     # This file - dynamic loader
    ├── no.py           # Norwegian (Borealis model)
    ├── sv.py           # Swedish (future)
    └── ...             # Other languages

Each language module must export:
    - LANGUAGE_CODE: str (e.g., "no")
    - LANGUAGE_NAME: str (e.g., "Norwegian")
    - get_refinement_prompt(paragraph: str) -> str
    - validate_response(response: str, original: str) -> str

Usage:
    from src.prompts.refine import get_refiner_for_language, is_language_supported

    if is_language_supported("no"):
        refiner = get_refiner_for_language("no")
        prompt = refiner.get_refinement_prompt(paragraph)
"""

import importlib
from pathlib import Path
from typing import Optional, List


def get_supported_languages() -> List[str]:
    """
    Get list of supported language codes based on available prompt files.

    Returns:
        List of language codes (e.g., ["no", "sv"])
    """
    refine_dir = Path(__file__).parent
    languages = []

    for file in refine_dir.glob("*.py"):
        if file.name.startswith("_"):
            continue
        lang_code = file.stem  # "no.py" -> "no"
        languages.append(lang_code)

    return sorted(languages)


def is_language_supported(language_code: str) -> bool:
    """
    Check if a language has refinement prompts available.

    Args:
        language_code: ISO 639-1 code (e.g., "no", "sv", "en")

    Returns:
        True if prompts exist for this language
    """
    refine_dir = Path(__file__).parent
    prompt_file = refine_dir / f"{language_code}.py"
    return prompt_file.exists()


def get_refiner_for_language(language_code: str):
    """
    Dynamically load the refinement module for a language.

    Args:
        language_code: ISO 639-1 code (e.g., "no")

    Returns:
        Module with get_refinement_prompt() and validate_response() functions

    Raises:
        ValueError: If language is not supported
    """
    if not is_language_supported(language_code):
        supported = get_supported_languages()
        raise ValueError(
            f"Language '{language_code}' not supported for refinement. "
            f"Supported: {supported}. "
            f"Add prompts at src/prompts/refine/{language_code}.py"
        )

    module_name = f"src.prompts.refine.{language_code}"
    return importlib.import_module(module_name)
