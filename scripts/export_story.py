#!/usr/bin/env python3
"""
Export a story and its D&D-style character data to markdown.

Usage:
    python scripts/export_story.py "Story Title"
    python scripts/export_story.py "Echoes of Synthesis"  # Partial match works
    python scripts/export_story.py --list                 # List all stories

Examples:
    python scripts/export_story.py "Echoes of Synthesis: The Divergence Protocol"
    python scripts/export_story.py "Viking"  # Matches any story with "Viking" in title
"""

import sys
import argparse
import json
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_azure_config():
    """Load Azure SQL configuration from environment."""
    from dotenv import load_dotenv
    import os

    load_dotenv(project_root / ".env")

    config = {
        "sql_server": os.getenv("AZURE_SQL_SERVER"),
        "sql_database": os.getenv("AZURE_SQL_DATABASE"),
        "sql_username": os.getenv("AZURE_SQL_USERNAME"),
        "sql_password": os.getenv("AZURE_SQL_PASSWORD"),
    }

    if not all(config.values()):
        print("Error: Azure SQL configuration incomplete in .env")
        print("  Required: AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_USERNAME, AZURE_SQL_PASSWORD")
        sys.exit(1)

    return config


def get_sql_connection(config):
    """Create Azure SQL connection."""
    import pyodbc

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={config['sql_server']};"
        f"DATABASE={config['sql_database']};"
        f"UID={config['sql_username']};"
        f"PWD={config['sql_password']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except Exception as e:
        print(f"Error connecting to Azure SQL: {e}")
        sys.exit(1)


def list_all_stories(conn):
    """List all stories in the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, structure, status, created_at
        FROM stories
        ORDER BY created_at DESC
    """)

    stories = []
    for row in cursor.fetchall():
        story_id = str(row[0])
        structure = json.loads(row[1]) if row[1] else {}
        title = structure.get("title", "(Untitled)")
        status = row[2]
        created = row[3]
        stories.append({
            "id": story_id,
            "title": title,
            "status": status,
            "created": created
        })

    cursor.close()
    return stories


def find_story_by_title(conn, search_term: str):
    """Find stories matching the search term (case-insensitive partial match)."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, structure FROM stories")

    matches = []
    search_lower = search_term.lower()

    for row in cursor.fetchall():
        story_id = str(row[0])
        structure = json.loads(row[1]) if row[1] else {}
        title = structure.get("title", "")

        if search_lower in title.lower():
            matches.append({"id": story_id, "title": title})

    cursor.close()
    return matches


def get_story_data(conn, story_id: str):
    """Fetch complete story data including chapters and characters."""
    cursor = conn.cursor()

    # Get story
    cursor.execute("""
        SELECT id, prompt, status, preferences, structure, created_at, updated_at
        FROM stories WHERE id = ?
    """, (story_id,))
    story_row = cursor.fetchone()

    if not story_row:
        return None

    story = {
        "id": str(story_row[0]),
        "prompt": story_row[1],
        "status": story_row[2],
        "preferences": json.loads(story_row[3]) if story_row[3] else {},
        "structure": json.loads(story_row[4]) if story_row[4] else {},
        "created_at": story_row[5],
        "updated_at": story_row[6],
    }

    # Get chapters
    cursor.execute("""
        SELECT id, number, title, synopsis, content, characters_featured,
               word_count, reading_time_minutes, status, created_at
        FROM chapters
        WHERE story_id = ?
        ORDER BY number
    """, (story_id,))

    chapters = []
    for row in cursor.fetchall():
        chapters.append({
            "id": str(row[0]),
            "number": row[1],
            "title": row[2],
            "synopsis": row[3] or "",
            "content": row[4] or "",
            "characters_featured": json.loads(row[5]) if row[5] else [],
            "word_count": row[6] or 0,
            "reading_time_minutes": row[7] or 0,
            "status": row[8],
            "created_at": row[9],
        })
    story["chapters"] = chapters

    # Get characters
    cursor.execute("""
        SELECT id, name, role, age, background, appearance, motivation,
               personality_traits, relationships, progression, character_arc
        FROM characters
        WHERE story_id = ?
    """, (story_id,))

    characters = []
    for row in cursor.fetchall():
        characters.append({
            "id": str(row[0]),
            "name": row[1],
            "role": row[2],
            "age": row[3],
            "background": row[4] or "",
            "appearance": row[5] or "",
            "motivation": row[6] or "",
            "personality_traits": json.loads(row[7]) if row[7] else [],
            "relationships": json.loads(row[8]) if row[8] else {},
            "progression": json.loads(row[9]) if row[9] else {},
            "character_arc": json.loads(row[10]) if row[10] else {},
        })
    story["characters"] = characters

    cursor.close()
    return story


def generate_markdown(story: dict) -> str:
    """Generate markdown content from story data."""
    md = []

    structure = story.get("structure", {})
    title = structure.get("title", "Untitled Story")
    theme = structure.get("theme", "")
    reading_time = structure.get("estimated_reading_time_minutes", 0)

    # Header
    md.append(f"# {title}")
    md.append("")
    md.append(f"**Status:** {story.get('status', 'Unknown')}")
    if story.get("created_at"):
        created = story["created_at"]
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M")
        md.append(f"**Created:** {created}")
    if theme:
        md.append(f"**Theme:** {theme}")
    if reading_time:
        md.append(f"**Estimated Reading Time:** {reading_time} minutes")
    md.append("")

    # Story prompt
    if story.get("prompt"):
        md.append("## Original Prompt")
        md.append("")
        md.append(f"> {story['prompt']}")
        md.append("")

    # Characters section
    characters = story.get("characters", [])
    if characters:
        md.append("## Characters")
        md.append("")

        for char in characters:
            # Character header
            name = char.get("name", "Unknown")
            role = char.get("role", "")
            md.append(f"### {name}" + (f" ({role})" if role else ""))
            md.append("")

            # Basic info
            if char.get("age"):
                md.append(f"**Age:** {char['age']}")
            if char.get("background"):
                md.append(f"**Background:** {char['background']}")
            if char.get("appearance"):
                md.append(f"**Appearance:** {char['appearance']}")
            if char.get("motivation"):
                md.append(f"**Motivation:** {char['motivation']}")

            # Personality traits
            traits = char.get("personality_traits", [])
            if traits:
                md.append(f"**Personality:** {', '.join(traits)}")

            md.append("")

            # D&D-style progression
            progression = char.get("progression", {})

            # Skills table
            skills = progression.get("skills_learned", [])
            if skills:
                md.append("#### Skills (D&D Stats)")
                md.append("")
                md.append("| Skill | Level | Acquired |")
                md.append("|-------|-------|----------|")
                for skill in skills:
                    skill_name = skill.get("name", "Unknown")
                    level = skill.get("level", 1)
                    acquired = skill.get("acquired_chapter", 0)
                    acquired_str = f"Ch. {acquired}" if acquired > 0 else "Backstory"
                    md.append(f"| {skill_name} | {level}/10 | {acquired_str} |")
                md.append("")

            # Relationships table
            rel_changes = progression.get("relationship_changes", [])
            relationships = char.get("relationships", {})
            if rel_changes or relationships:
                md.append("#### Relationships")
                md.append("")
                md.append("| Character | Type | Strength | Notes |")
                md.append("|-----------|------|----------|-------|")

                # From progression (detailed)
                for rel in rel_changes:
                    other = rel.get("other_character", "Unknown")
                    rel_type = rel.get("relationship_type", "")
                    strength = rel.get("strength", 5)
                    desc = rel.get("description", "")[:50]
                    md.append(f"| {other} | {rel_type} | {strength}/10 | {desc} |")

                # From static relationships dict
                for other, desc in relationships.items():
                    if not any(r.get("other_character") == other for r in rel_changes):
                        md.append(f"| {other} | - | - | {desc[:50]} |")

                md.append("")

            # Personality evolution
            evolutions = progression.get("personality_evolution", [])
            if evolutions:
                md.append("#### Character Development")
                md.append("")
                for evo in evolutions:
                    chapter = evo.get("chapter_number", "?")
                    from_trait = evo.get("from_trait", "")
                    to_trait = evo.get("to_trait", "")
                    trigger = evo.get("trigger_event", "")
                    if from_trait and to_trait:
                        md.append(f"- **Ch. {chapter}:** {from_trait} → {to_trait}")
                        if trigger:
                            md.append(f"  - *Trigger:* {trigger}")
                    elif evo.get("change"):
                        md.append(f"- **Ch. {chapter}:** {evo['change']}")
                md.append("")

            # Character arc summary
            arc = char.get("character_arc", {})
            if arc:
                md.append("#### Arc Summary")
                md.append("")
                if isinstance(arc, dict):
                    for chapter, milestone in sorted(arc.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                        md.append(f"- **Ch. {chapter}:** {milestone}")
                md.append("")

            # Current emotional state
            emotional_state = progression.get("current_emotional_state", "")
            if emotional_state:
                md.append(f"**Current Emotional State:** {emotional_state}")
                md.append("")

            md.append("* * *")
            md.append("")

    # Chapters section
    chapters = story.get("chapters", [])
    if chapters:
        md.append("## Chapters")
        md.append("")

        for chapter in chapters:
            number = chapter.get("number", 0)
            ch_title = chapter.get("title", "Untitled Chapter")
            synopsis = chapter.get("synopsis", "")
            content = chapter.get("content", "")
            featured = chapter.get("characters_featured", [])
            word_count = chapter.get("word_count", 0)

            md.append(f"### Chapter {number}: {ch_title}")
            md.append("")

            if synopsis:
                md.append(f"> {synopsis}")
                md.append("")

            if content:
                # Replace --- scene breaks with * * * to avoid YAML parsing issues
                content = content.replace("\n---\n", "\n* * *\n")
                md.append(content)
                md.append("")

            # Chapter footer
            footer_parts = []
            if featured:
                footer_parts.append(f"**Characters:** {', '.join(featured)}")
            if word_count:
                footer_parts.append(f"**Words:** {word_count:,}")

            if footer_parts:
                md.append("* * *")
                md.append(f"{' | '.join(footer_parts)}")
            md.append("")

    return "\n".join(md)


def sanitize_filename(title: str) -> str:
    """Convert title to a safe filename."""
    # Replace spaces and special chars with underscores
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'[\s]+', '_', safe)
    return safe[:100]  # Limit length


def main():
    parser = argparse.ArgumentParser(
        description="Export a story to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("title", nargs="?", help="Story title to search for (partial match)")
    parser.add_argument("--list", "-l", action="store_true", help="List all stories")
    parser.add_argument("--output", "-o", help="Output file path (default: {title}.md)")
    args = parser.parse_args()

    if not args.title and not args.list:
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("  LORE LANTERN - STORY EXPORTER")
    print("=" * 60)
    print()

    # Connect to database
    config = get_azure_config()
    conn = get_sql_connection(config)
    print(f"Connected to Azure SQL: {config['sql_database']}")
    print()

    # List all stories
    if args.list:
        stories = list_all_stories(conn)
        if not stories:
            print("No stories found in database.")
        else:
            print(f"Found {len(stories)} stories:\n")
            for s in stories:
                created = s['created']
                if isinstance(created, datetime):
                    created = created.strftime("%Y-%m-%d")
                print(f"  [{s['status']}] {s['title']}")
                print(f"           Created: {created}")
                print(f"           ID: {s['id'][:8]}...")
                print()
        conn.close()
        return

    # Search for story
    matches = find_story_by_title(conn, args.title)

    if not matches:
        print(f"No stories found matching: '{args.title}'")
        print("\nUse --list to see all available stories.")
        conn.close()
        sys.exit(1)

    if len(matches) > 1:
        print(f"Multiple stories match '{args.title}':\n")
        for i, m in enumerate(matches, 1):
            print(f"  {i}. {m['title']}")
        print(f"\nPlease be more specific.")
        conn.close()
        sys.exit(1)

    # Get the story
    story_id = matches[0]["id"]
    story_title = matches[0]["title"]
    print(f"Exporting: {story_title}")
    print()

    story = get_story_data(conn, story_id)
    conn.close()

    if not story:
        print(f"Error: Could not load story data.")
        sys.exit(1)

    # Generate markdown
    markdown = generate_markdown(story)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        filename = sanitize_filename(story_title) + ".md"
        output_path = Path.cwd() / filename

    # Write file
    output_path.write_text(markdown, encoding="utf-8")

    # Summary
    num_chapters = len(story.get("chapters", []))
    num_characters = len(story.get("characters", []))
    total_words = sum(ch.get("word_count", 0) for ch in story.get("chapters", []))

    print(f"Exported successfully!")
    print(f"  File: {output_path}")
    print(f"  Chapters: {num_chapters}")
    print(f"  Characters: {num_characters}")
    print(f"  Total words: {total_words:,}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
