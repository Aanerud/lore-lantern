"""
Validation Service for LLM Output Processing

This module provides utilities for cleaning and validating LLM-generated
JSON output. Handles common issues:
- Trailing commas, JavaScript comments
- Unescaped quotes inside strings
- Truncated JSON (attempts to close brackets)
- Invalid escape sequences
- Missing/wrong-type fields in character data

These functions handle the variability in LLM output formats across
different providers (GPT, Claude, Gemini, DeepSeek, etc.).

Architecture:
- Called by CharacterService for character creation
- Called by Coordinator for structure/review parsing
- Stateless utility functions (no class needed)
"""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# JSON CLEANING UTILITIES
# =========================================================================

def clean_json_output(output: str) -> str:
    """
    Clean markdown code blocks from LLM output and repair common JSON issues.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Preamble text before JSON (e.g., "Here is my response:\n{...}")
    - CrewAI "Final Answer:" format
    - Unescaped quotes in string values
    - Invalid escape sequences (\\» → »)
    - Invalid control characters (0x00-0x1f except tab, newline, carriage return)

    Args:
        output: Raw LLM output possibly containing markdown

    Returns:
        Cleaned JSON string ready for parsing
    """
    result_str = output.strip()

    # Remove invalid control characters that break JSON parsing
    # Keep \t (0x09), \n (0x0a), \r (0x0d) as they are valid when escaped
    result_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', result_str)

    # Strategy 1: Try to extract JSON from markdown code blocks
    # Matches: ```json\n{...}\n``` or ```{...}```
    # IMPORTANT: Use greedy * (not non-greedy *?) to capture complete nested JSON
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', result_str)
    if json_match:
        extracted = json_match.group(1).strip()
        try:
            json.loads(extracted, strict=False)
            return extracted
        except json.JSONDecodeError:
            pass  # Continue trying other strategies

    # Strategy 2: Find JSON object anywhere in the text
    # This handles preamble like "Here is the JSON:\n{...}"
    # Use a balanced brace matching approach
    extracted = _extract_json_object(result_str)
    if extracted:
        try:
            json.loads(extracted, strict=False)
            return extracted
        except json.JSONDecodeError:
            pass  # Continue with the extracted string for repair
        result_str = extracted
    else:
        # Strategy 3: Fallback - simple strip if no JSON found
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        if result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        result_str = result_str.strip()

    # Try to parse - if it works, return as-is
    # Use strict=False to allow control characters like unescaped newlines in strings
    try:
        json.loads(result_str, strict=False)
        return result_str
    except json.JSONDecodeError:
        pass  # Continue with repair

    # Repair unescaped quotes inside JSON string values
    # Example: "synopsis": "...nickname "Harald Tanglehair" and..."
    # Should become: "synopsis": "...nickname \"Harald Tanglehair\" and..."
    repaired = _repair_unescaped_quotes(result_str)

    # Try to parse again - if it works, return
    try:
        json.loads(repaired, strict=False)
        return repaired
    except json.JSONDecodeError as e:
        # If any escape-related error, try to fix invalid escape sequences
        # Catches both "Invalid \escape" and "Invalid \uXXXX escape" (malformed unicode)
        error_str = str(e).lower()
        if "invalid" in error_str and "escape" in error_str:
            repaired = _fix_invalid_escapes(repaired)

    return repaired


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract a JSON object from text by finding balanced braces.

    This handles cases where LLMs output preamble text before JSON:
    - "Here is the JSON response:\n{...}"
    - "Based on the analysis...\n```json\n{...}\n```"
    - "Thought: ...\nFinal Answer:\n{...}"

    Args:
        text: Raw text that may contain JSON somewhere within it

    Returns:
        The extracted JSON string, or None if no valid JSON object found
    """
    # Find the first '{' character
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    # Track brace depth to find matching '}'
    depth = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        return None

    return text[start_idx:end_idx + 1]


def clean_json_for_character(json_str: str, char_name: str) -> str:
    """
    Clean common JSON issues from LLM output before parsing character data.

    Handles:
    - Trailing commas before ] or }
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - JavaScript-style comments
    - Truncated JSON (attempts to close brackets)

    Args:
        json_str: Raw JSON string from LLM
        char_name: Character name for logging

    Returns:
        Cleaned JSON string
    """
    original_len = len(json_str)

    # Remove JavaScript-style comments
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    # Fix trailing commas before ] or }
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # Fix unescaped newlines within strings (common LLM error)
    # Note: Variable-width look-behinds not supported in Python re module
    # Use a simpler approach: replace literal newlines between quotes with \n
    # This handles the common case of ": "some\nvalue"
    json_str = re.sub(r'(": ")([^"]*?)\n([^"]*?")', r'\1\2\\n\3', json_str)

    # Try to close truncated JSON
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')

    if open_braces > 0 or open_brackets > 0:
        logger.warning(f"JSON appears truncated for {char_name}: {open_braces} unclosed braces, {open_brackets} unclosed brackets")
        # Add closing brackets/braces
        json_str = json_str.rstrip()
        # Remove trailing comma if present
        if json_str.endswith(','):
            json_str = json_str[:-1]
        json_str += ']' * open_brackets + '}' * open_braces

    if len(json_str) != original_len:
        logger.info(f"JSON cleaned for {char_name} ({original_len} → {len(json_str)} chars)")

    return json_str


# =========================================================================
# CHARACTER DATA AUTO-FIX
# =========================================================================

def auto_fix_character_data(
    char_data: Dict[str, Any],
    char_name: str,
    validation_error: Optional[Exception] = None
) -> Dict[str, Any]:
    """
    Attempt to auto-fix common validation issues in character data.

    Common issues fixed:
    - Missing 'progression' field
    - Wrong type for 'relationships' (list instead of dict)
    - Missing 'skills_learned' in progression
    - Wrong type for 'personality_traits' (string instead of list)
    - Missing required fields with sensible defaults

    Args:
        char_data: Character data dict from LLM
        char_name: Character name for logging
        validation_error: Optional validation error for context

    Returns:
        Fixed character data dict
    """
    fixes_applied = []

    # Ensure name is set
    if 'name' not in char_data or not char_data['name']:
        char_data['name'] = char_name
        fixes_applied.append("name")

    # Ensure role has a default
    if 'role' not in char_data or not char_data['role']:
        char_data['role'] = "supporting"
        fixes_applied.append("role")

    # Fix relationships - must be dict, not list
    if 'relationships' in char_data:
        rel = char_data['relationships']
        if isinstance(rel, list):
            # Convert list to dict
            char_data['relationships'] = {f"character_{i}": str(r) for i, r in enumerate(rel)}
            fixes_applied.append("relationships (list→dict)")
        elif rel is None:
            char_data['relationships'] = {}
            fixes_applied.append("relationships (null→empty)")

    # Fix personality_traits - must be list
    if 'personality_traits' in char_data:
        traits = char_data['personality_traits']
        if isinstance(traits, str):
            # Split comma-separated string into list
            char_data['personality_traits'] = [t.strip() for t in traits.split(',')]
            fixes_applied.append("personality_traits (str→list)")
        elif traits is None:
            char_data['personality_traits'] = ["determined"]
            fixes_applied.append("personality_traits (null→default)")

    # Ensure progression exists with required fields
    if 'progression' not in char_data or char_data['progression'] is None:
        char_data['progression'] = {
            'skills_learned': [
                {"name": "Awareness", "level": 1, "acquired_chapter": 0, "description": "Basic alertness"}
            ],
            'personality_evolution': [],
            'relationship_changes': [],
            'current_emotional_state': "curious",
            'chapters_featured': []
        }
        fixes_applied.append("progression (missing)")
    else:
        prog = char_data['progression']

        # Ensure skills_learned exists and is a list
        if 'skills_learned' not in prog or not isinstance(prog.get('skills_learned'), list):
            prog['skills_learned'] = [
                {"name": "Awareness", "level": 1, "acquired_chapter": 0, "description": "Basic alertness"}
            ]
            fixes_applied.append("skills_learned")

        # Ensure current_emotional_state exists
        if 'current_emotional_state' not in prog or not prog['current_emotional_state']:
            prog['current_emotional_state'] = "curious"
            fixes_applied.append("current_emotional_state")

        # Ensure list fields exist
        for list_field in ['personality_evolution', 'relationship_changes', 'chapters_featured']:
            if list_field not in prog or not isinstance(prog.get(list_field), list):
                prog[list_field] = []
                fixes_applied.append(list_field)

    # Ensure background exists
    if 'background' not in char_data or not char_data['background']:
        char_data['background'] = f"A {char_data.get('role', 'supporting')} character in the story."
        fixes_applied.append("background")

    if fixes_applied:
        logger.info(f"Auto-fixed {len(fixes_applied)} fields for {char_name}: {', '.join(fixes_applied)}")

    return char_data


# =========================================================================
# INTERNAL HELPER FUNCTIONS
# =========================================================================

def _sanitize_control_characters(json_str: str) -> str:
    """
    Sanitize control characters in JSON string using simple regex replacement.

    Control characters (0x00-0x1F) are NEVER valid raw in JSON - they must
    always be escaped. This method uses a simple regex approach that doesn't
    rely on tracking string state (which can desync on unescaped quotes).
    """
    def escape_control_char(match):
        """Convert a raw control character to its JSON escape sequence."""
        char = match.group(0)
        # Map common control chars to short escape sequences
        escape_map = {
            '\x08': '\\b',   # Backspace
            '\x09': '\\t',   # Tab
            '\x0a': '\\n',   # Newline
            '\x0c': '\\f',   # Form feed
            '\x0d': '\\r',   # Carriage return
        }
        if char in escape_map:
            return escape_map[char]
        # Use unicode escape for other control characters
        return f'\\u{ord(char):04x}'

    # Match any control character (0x00-0x1F) that is NOT preceded by a backslash
    # This avoids double-escaping already-escaped sequences like \\n
    pattern = r'(?<!\\)[\x00-\x1f]'

    return re.sub(pattern, escape_control_char, json_str)


def _fix_invalid_escapes(json_str: str) -> str:
    """
    Fix invalid escape sequences in JSON.

    JSON only allows these escape sequences after a backslash:
    - " \\ / b f n r t and uXXXX (exactly 4 hex digits)

    This method:
    - Removes backslash from invalid escapes like \\» -> »
    - Removes malformed unicode escapes like \\u00 (incomplete) -> empty
    - Fixes \\uXYZQ where not all are hex -> empty
    """
    # First, fix malformed unicode escapes (\u not followed by exactly 4 hex digits)
    # This handles cases like \u00, \u0, \u followed by non-hex, etc.
    # The negative lookahead ensures we only match \u NOT followed by 4 hex chars
    fixed = re.sub(r'\\u(?![0-9a-fA-F]{4})[0-9a-fA-F]{0,3}', '', json_str)

    # Valid JSON escape characters (after backslash)
    valid_escapes = set('"\\bfnrt/')

    def fix_escape(match):
        """Fix invalid escape by removing the backslash."""
        char_after_backslash = match.group(1)
        # If it's a valid escape char, keep the backslash
        if char_after_backslash in valid_escapes:
            return match.group(0)  # Keep as-is
        # If it's 'u' followed by valid unicode, the regex above already handled it
        # So any remaining \u is part of a valid sequence
        if char_after_backslash == 'u':
            return match.group(0)  # Keep as-is (valid unicode was preserved)
        # For invalid escapes, remove the backslash
        return char_after_backslash

    # Match backslash followed by any character
    fixed = re.sub(r'\\(.)', fix_escape, fixed)

    return fixed


def _repair_unescaped_quotes(json_str: str) -> str:
    """
    Repair unescaped double quotes inside JSON string values.

    Uses a state machine to track whether we're inside a string,
    and escapes quotes that appear to be embedded within string content.
    """
    result = []
    i = 0
    in_string = False

    while i < len(json_str):
        char = json_str[i]

        if char == '\\' and i + 1 < len(json_str):
            # Escaped character - keep as-is and skip next char
            result.append(char)
            result.append(json_str[i + 1])
            i += 2
            continue

        if char == '"':
            if not in_string:
                # Starting a string
                in_string = True
                result.append(char)
            else:
                # We're in a string and hit a quote - check if it's the end or embedded
                # Look ahead to see if this looks like a string boundary
                next_chars = json_str[i + 1:i + 50] if i + 1 < len(json_str) else ""
                stripped = next_chars.lstrip()

                is_boundary = False

                # End of string
                if next_chars == "":
                    is_boundary = True
                # Followed by } or ] - definitely boundary
                elif stripped.startswith(('}', ']')):
                    is_boundary = True
                # Followed by : - this quote ends a key
                elif stripped.startswith(':'):
                    is_boundary = True
                # Followed by comma - check what comes AFTER the comma
                elif stripped.startswith(','):
                    # After a real JSON boundary comma, we'd see:
                    # - A new key starting with "
                    # - A number, boolean, null
                    # - An object { or array [
                    after_comma = stripped[1:].lstrip()
                    if after_comma.startswith('"'):
                        # Could be a new key "key": or just a string value in array
                        # Check if there's a colon pattern "key":
                        match = re.match(r'^"[^"]*"\s*:', after_comma)
                        if match:
                            is_boundary = True  # It's a new key
                        else:
                            # Check if it's followed by }, ], or , (array of strings)
                            match2 = re.match(r'^"[^"]*"\s*[}\],]', after_comma)
                            if match2:
                                is_boundary = True  # Array of strings
                            else:
                                # Extended check: if the next string doesn't look like a key
                                extended_match = re.match(r'^"[^":]{0,100}"', after_comma)
                                if extended_match:
                                    is_boundary = True  # Likely array element
                                else:
                                    is_boundary = True  # Default safer
                    elif after_comma.startswith(('{', '[', 'true', 'false', 'null')) or \
                         (after_comma and after_comma[0].isdigit()):
                        is_boundary = True
                    else:
                        # After comma we see regular text - NOT a boundary
                        is_boundary = False
                # Newline followed by whitespace then " (new key on next line)
                elif stripped.startswith('\n') or re.match(r'^\s*"[^"]*"\s*:', stripped):
                    is_boundary = True

                if is_boundary:
                    # This is the end of the string
                    in_string = False
                    result.append(char)
                else:
                    # This is an embedded quote that needs escaping
                    result.append('\\')
                    result.append(char)
        else:
            result.append(char)

        i += 1

    return ''.join(result)
