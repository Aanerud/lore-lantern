"""
Azure SQL Database service for LoreLantern.

Replaces Firebase Realtime Database with Azure SQL for proper relational
data storage and household-level data isolation.
"""

import pyodbc
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4, UUID
import uuid as uuid_module
import sys
from pathlib import Path


# =========================================================================
# UUID Helpers - Convert string IDs to valid UUIDs for Azure SQL
# =========================================================================

# Namespace UUID for generating deterministic UUIDs from string IDs
LORELANTERN_NAMESPACE = UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    if not value:
        return False
    try:
        UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def to_uuid(value: str) -> str:
    """
    Convert a string to a valid UUID string.

    If the value is already a valid UUID, returns it as-is.
    If not, generates a deterministic UUID5 from the string.
    This ensures the same input always produces the same UUID.

    Args:
        value: String ID (may or may not be a valid UUID)

    Returns:
        Valid UUID string
    """
    if not value:
        return str(uuid4())  # Generate new UUID if empty

    if is_valid_uuid(value):
        return str(UUID(value))  # Normalize format

    # Generate deterministic UUID from string
    return str(uuid_module.uuid5(LORELANTERN_NAMESPACE, value))

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models import (
    Story, Chapter, Character, DialogueEntry, FactCheckReport,
    ChapterStatus, StoryReadingState, QueuedInput, ChapterAudioState,
    ParentAccount, ChildProfile, LearningProgress, StoryStructure,
    StoryPreferences, StoryMetadata, StoryStatus
)


class DatabaseService:
    """Azure SQL Database service with household isolation."""

    def __init__(
        self,
        server: str,
        database: str,
        username: str,
        password: str,
        logger=None
    ):
        """
        Initialize database service.

        Args:
            server: Azure SQL server (e.g., lorelantern-db.database.windows.net)
            database: Database name
            username: SQL admin username
            password: SQL admin password
            logger: Optional logger instance
        """
        self.connection_string = (
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{server},1433;"
            f"Database={database};"
            f"Uid={username};"
            f"Pwd={password};"
            f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )
        self.logger = logger
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._initialized = False

    def initialize(self):
        """Initialize the database connection (verify connectivity)."""
        if self._initialized:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            self._initialized = True
            print("   Azure SQL connection verified")
        except Exception as e:
            print(f"   Warning: Azure SQL connection failed: {e}")
            raise

    def _get_connection(self) -> pyodbc.Connection:
        """Get a database connection."""
        return pyodbc.connect(self.connection_string)

    async def _run_async(self, func, *args, **kwargs):
        """Run a sync function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )

    def _log(self, operation: str, path: str, summary: str, duration: float = 0):
        """Log database operation."""
        if self.logger:
            self.logger.info(f"[DB {operation}] {path}: {summary} ({duration:.3f}s)")

    # =========================================================================
    # Household Operations (formerly ParentAccount)
    # =========================================================================

    def _save_household_sync(self, household: ParentAccount) -> ParentAccount:
        """Save or update a household."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(household.parent_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT id FROM households WHERE id = ?",
                (uuid_id,)
            )
            exists = cursor.fetchone() is not None

            if exists:
                cursor.execute("""
                    UPDATE households
                    SET display_name = ?, language = ?, updated_at = GETUTCDATE()
                    WHERE id = ?
                """, (
                    household.display_name,
                    household.language or 'en',
                    uuid_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO households (id, display_name, language)
                    VALUES (?, ?, ?)
                """, (
                    uuid_id,
                    household.display_name,
                    household.language or 'en'
                ))

            conn.commit()

        # Update the parent_id with the UUID for consistency
        household.parent_id = uuid_id
        return household

    async def save_household(self, household: ParentAccount) -> ParentAccount:
        """Save or update a household."""
        return await self._run_async(self._save_household_sync, household)

    # Alias for compatibility
    async def save_parent_account(self, parent: ParentAccount) -> ParentAccount:
        return await self.save_household(parent)

    def _get_household_sync(self, household_id: str) -> Optional[ParentAccount]:
        """Get a household by ID."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(household_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, display_name, language, created_at, updated_at
                FROM households WHERE id = ?
            """, (uuid_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Get child IDs
            cursor.execute(
                "SELECT id FROM children WHERE household_id = ?",
                (uuid_id,)
            )
            child_ids = [str(r[0]) for r in cursor.fetchall()]

            return ParentAccount(
                parent_id=str(row[0]),
                display_name=row[1],
                language=row[2],
                child_ids=child_ids,
                created_at=row[3],
                updated_at=row[4]
            )

    async def get_household(self, household_id: str) -> Optional[ParentAccount]:
        """Get a household by ID."""
        return await self._run_async(self._get_household_sync, household_id)

    # Alias for compatibility
    async def get_parent_account(self, parent_id: str) -> Optional[ParentAccount]:
        return await self.get_household(parent_id)

    def _list_households_sync(self) -> List[ParentAccount]:
        """List all households."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, display_name, language, created_at, updated_at
                FROM households ORDER BY created_at DESC
            """)

            households = []
            for row in cursor.fetchall():
                # Get child IDs for each household
                cursor2 = conn.cursor()
                cursor2.execute(
                    "SELECT id FROM children WHERE household_id = ?",
                    (row[0],)
                )
                child_ids = [str(r[0]) for r in cursor2.fetchall()]

                households.append(ParentAccount(
                    parent_id=str(row[0]),
                    display_name=row[1],
                    language=row[2],
                    child_ids=child_ids,
                    created_at=row[3],
                    updated_at=row[4]
                ))

            return households

    async def list_households(self) -> List[ParentAccount]:
        """List all households."""
        return await self._run_async(self._list_households_sync)

    # Alias for compatibility
    async def list_parent_accounts(self) -> List[ParentAccount]:
        return await self.list_households()

    async def update_parent_account(self, parent_id: str, updates: Dict[str, Any]) -> Optional[ParentAccount]:
        """Update a household."""
        household = await self.get_household(parent_id)
        if not household:
            return None

        for key, value in updates.items():
            if hasattr(household, key) and value is not None:
                setattr(household, key, value)

        return await self.save_household(household)

    # =========================================================================
    # Child Operations
    # =========================================================================

    def _save_child_sync(self, child: ChildProfile) -> ChildProfile:
        """Save or update a child profile."""
        # Convert string IDs to valid UUIDs for Azure SQL
        uuid_child_id = to_uuid(child.child_id)
        uuid_parent_id = to_uuid(child.parent_id)
        uuid_story_id = to_uuid(child.active_story_id) if child.active_story_id else None

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT id FROM children WHERE id = ?",
                (uuid_child_id,)
            )
            exists = cursor.fetchone() is not None

            if exists:
                cursor.execute("""
                    UPDATE children
                    SET name = ?, birth_year = ?, active_story_id = ?, updated_at = GETUTCDATE()
                    WHERE id = ?
                """, (
                    child.name,
                    child.birth_year,
                    uuid_story_id,
                    uuid_child_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO children (id, household_id, name, birth_year, active_story_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    uuid_child_id,
                    uuid_parent_id,  # household_id
                    child.name,
                    child.birth_year,
                    uuid_story_id
                ))

            conn.commit()

        # Update IDs with UUIDs for consistency
        child.child_id = uuid_child_id
        child.parent_id = uuid_parent_id
        return child

    async def save_child_profile(self, child: ChildProfile) -> ChildProfile:
        """Save or update a child profile."""
        return await self._run_async(self._save_child_sync, child)

    def _get_child_sync(self, child_id: str) -> Optional[ChildProfile]:
        """Get a child profile by ID."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(child_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, household_id, name, birth_year, active_story_id, created_at, updated_at
                FROM children WHERE id = ?
            """, (uuid_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Get story IDs
            cursor.execute(
                "SELECT id FROM stories WHERE child_id = ?",
                (uuid_id,)
            )
            story_ids = [str(r[0]) for r in cursor.fetchall()]

            return ChildProfile(
                child_id=str(row[0]),
                parent_id=str(row[1]),  # household_id
                name=row[2],
                birth_year=row[3],
                active_story_id=str(row[4]) if row[4] else None,
                story_ids=story_ids,
                created_at=row[5],
                updated_at=row[6]
            )

    async def get_child_profile(self, child_id: str) -> Optional[ChildProfile]:
        """Get a child profile by ID."""
        return await self._run_async(self._get_child_sync, child_id)

    def _get_children_by_household_sync(self, household_id: str) -> List[ChildProfile]:
        """Get all children for a household."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(household_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, household_id, name, birth_year, active_story_id, created_at, updated_at
                FROM children WHERE household_id = ?
                ORDER BY created_at
            """, (uuid_id,))

            children = []
            for row in cursor.fetchall():
                child_id = str(row[0])
                # Get story IDs
                cursor2 = conn.cursor()
                cursor2.execute(
                    "SELECT id FROM stories WHERE child_id = ?",
                    (child_id,)
                )
                story_ids = [str(r[0]) for r in cursor2.fetchall()]

                children.append(ChildProfile(
                    child_id=child_id,
                    parent_id=str(row[1]),
                    name=row[2],
                    birth_year=row[3],
                    active_story_id=str(row[4]) if row[4] else None,
                    story_ids=story_ids,
                    created_at=row[5],
                    updated_at=row[6]
                ))

            return children

    async def get_children_by_household(self, household_id: str) -> List[ChildProfile]:
        """Get all children for a household."""
        return await self._run_async(self._get_children_by_household_sync, household_id)

    # Alias for compatibility
    async def get_children_by_parent(self, parent_id: str) -> List[ChildProfile]:
        return await self.get_children_by_household(parent_id)

    async def update_child_profile(self, child_id: str, updates: Dict[str, Any]) -> Optional[ChildProfile]:
        """Update a child profile."""
        child = await self.get_child_profile(child_id)
        if not child:
            return None

        for key, value in updates.items():
            if hasattr(child, key) and value is not None:
                setattr(child, key, value)

        return await self.save_child_profile(child)

    async def set_active_story(self, child_id: str, story_id: Optional[str]) -> Optional[ChildProfile]:
        """Set active story for a child."""
        return await self.update_child_profile(child_id, {'active_story_id': story_id})

    async def add_story_to_child(self, child_id: str, story_id: str) -> Optional[ChildProfile]:
        """Add a story to child's story list (no-op in SQL - stories linked via FK)."""
        return await self.get_child_profile(child_id)

    # =========================================================================
    # Learning Progress Operations
    # =========================================================================

    def _save_learning_progress_sync(self, progress: LearningProgress) -> LearningProgress:
        """Save or update learning progress."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_child_id = to_uuid(progress.child_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT child_id FROM learning_progress WHERE child_id = ?",
                (uuid_child_id,)
            )
            exists = cursor.fetchone() is not None

            # Serialize complex types to JSON
            # vocabulary_bank is Dict[str, VocabularyEntry] - need to serialize VocabularyEntry objects
            vocab_json = None
            if progress.vocabulary_bank:
                vocab_dict = {}
                for word, entry in progress.vocabulary_bank.items():
                    if hasattr(entry, 'model_dump'):
                        vocab_dict[word] = entry.model_dump(mode='json')
                    else:
                        vocab_dict[word] = entry
                vocab_json = json.dumps(vocab_dict)

            # concepts_mastered is Dict[str, ConceptMastery]
            concepts_json = None
            if progress.concepts_mastered:
                concepts_dict = {}
                for concept, entry in progress.concepts_mastered.items():
                    if hasattr(entry, 'model_dump'):
                        concepts_dict[concept] = entry.model_dump(mode='json')
                    else:
                        concepts_dict[concept] = entry
                concepts_json = json.dumps(concepts_dict)

            interests_json = json.dumps(progress.detected_interests) if progress.detected_interests else None
            prefs_json = json.dumps({
                'preferred_scary_level': progress.preferred_scary_level,
                'preferred_story_length': progress.preferred_story_length
            })

            if exists:
                cursor.execute("""
                    UPDATE learning_progress
                    SET vocabulary = ?, concepts = ?, reading_level = ?,
                        detected_interests = ?, preferences = ?,
                        total_stories = ?, total_chapters_read = ?,
                        curiosity_score = ?, updated_at = GETUTCDATE()
                    WHERE child_id = ?
                """, (
                    vocab_json,
                    concepts_json,
                    progress.reading_level,
                    interests_json,
                    prefs_json,
                    progress.total_stories_completed,
                    progress.total_chapters_read,
                    progress.total_questions_asked,  # This maps to curiosity_score in DB
                    uuid_child_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO learning_progress
                    (child_id, vocabulary, concepts, reading_level, detected_interests,
                     preferences, total_stories, total_chapters_read, curiosity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    uuid_child_id,
                    vocab_json,
                    concepts_json,
                    progress.reading_level,
                    interests_json,
                    prefs_json,
                    progress.total_stories_completed,
                    progress.total_chapters_read,
                    progress.total_questions_asked  # This maps to curiosity_score in DB
                ))

            conn.commit()
        return progress

    async def save_learning_progress(self, progress: LearningProgress) -> LearningProgress:
        """Save learning progress."""
        return await self._run_async(self._save_learning_progress_sync, progress)

    def _get_learning_progress_sync(self, child_id: str) -> Optional[LearningProgress]:
        """Get learning progress for a child."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(child_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT child_id, vocabulary, concepts, reading_level,
                       detected_interests, preferences, total_stories,
                       total_chapters_read, curiosity_score, updated_at
                FROM learning_progress WHERE child_id = ?
            """, (uuid_id,))
            row = cursor.fetchone()

            if not row:
                return None

            prefs = json.loads(row[5]) if row[5] else {}

            return LearningProgress(
                child_id=str(row[0]),
                vocabulary_bank=json.loads(row[1]) if row[1] else {},
                concepts_mastered=json.loads(row[2]) if row[2] else {},
                reading_level=row[3],
                detected_interests=json.loads(row[4]) if row[4] else [],
                preferred_scary_level=prefs.get('preferred_scary_level', 'mild'),
                preferred_story_length=prefs.get('preferred_story_length', 'medium'),
                total_stories_completed=row[6],
                total_chapters_read=row[7],
                total_questions_asked=row[8],  # curiosity_score in DB maps to total_questions_asked
                updated_at=row[9]
            )

    async def get_learning_progress(self, child_id: str) -> Optional[LearningProgress]:
        """Get learning progress for a child."""
        return await self._run_async(self._get_learning_progress_sync, child_id)

    async def get_or_create_learning_progress(self, child_id: str) -> LearningProgress:
        """Get or create learning progress."""
        progress = await self.get_learning_progress(child_id)
        if progress:
            return progress

        progress = LearningProgress(child_id=child_id)
        return await self.save_learning_progress(progress)

    # =========================================================================
    # Story Operations
    # =========================================================================

    def _save_story_sync(self, story: Story) -> Story:
        """Save or update a story."""
        # Convert string IDs to valid UUIDs for Azure SQL
        uuid_story_id = to_uuid(story.id)
        uuid_child_id = to_uuid(story.metadata.child_id) if story.metadata.child_id else None

        if not uuid_child_id:
            raise ValueError("Story must have a child_id in metadata")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute("SELECT id FROM stories WHERE id = ?", (uuid_story_id,))
            exists = cursor.fetchone() is not None

            # Get household_id from child
            cursor.execute("SELECT household_id FROM children WHERE id = ?", (uuid_child_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Child {uuid_child_id} not found")
            household_id = str(row[0])

            prefs_json = json.dumps(story.preferences.model_dump(mode='json')) if story.preferences else None
            structure_json = json.dumps(story.structure.model_dump(mode='json')) if story.structure else None

            if exists:
                cursor.execute("""
                    UPDATE stories
                    SET prompt = ?, status = ?, preferences = ?, structure = ?, updated_at = GETUTCDATE()
                    WHERE id = ?
                """, (
                    story.prompt,
                    story.status.value,
                    prefs_json,
                    structure_json,
                    uuid_story_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO stories (id, household_id, child_id, prompt, status, preferences, structure)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    uuid_story_id,
                    household_id,
                    uuid_child_id,
                    story.prompt,
                    story.status.value,
                    prefs_json,
                    structure_json
                ))

            conn.commit()

        # Update story ID with UUID for consistency
        story.id = uuid_story_id
        return story

    async def save_story(self, story: Story) -> Story:
        """Save or update a story."""
        return await self._run_async(self._save_story_sync, story)

    def _get_story_sync(self, story_id: str) -> Optional[Story]:
        """Get a story by ID."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(story_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, household_id, child_id, prompt, status, preferences, structure,
                       created_at, updated_at
                FROM stories WHERE id = ?
            """, (uuid_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Parse JSON fields
            preferences = StoryPreferences(**json.loads(row[5])) if row[5] else StoryPreferences()
            structure = StoryStructure(**json.loads(row[6])) if row[6] else None

            # Get chapters
            cursor.execute("""
                SELECT id, number, title, synopsis, content, tts_content, audio_blob_url,
                       status, characters_featured, educational_points, vocabulary_words,
                       facts, statements, user_inputs_applied, round_table_review,
                       voice_direction, word_count, reading_time_minutes, created_at,
                       language_refined, pre_refinement_content
                FROM chapters WHERE story_id = ? ORDER BY number
            """, (uuid_id,))
            chapters = []
            for ch_row in cursor.fetchall():
                chapters.append(Chapter(
                    id=str(ch_row[0]),
                    number=ch_row[1],
                    title=ch_row[2],
                    synopsis=ch_row[3] or "",
                    content=ch_row[4] or "",
                    tts_content=ch_row[5],
                    characters_featured=json.loads(ch_row[8]) if ch_row[8] else [],
                    educational_points=json.loads(ch_row[9]) if ch_row[9] else [],
                    vocabulary_words=json.loads(ch_row[10]) if ch_row[10] else [],
                    facts=json.loads(ch_row[11]) if ch_row[11] else [],
                    statements=json.loads(ch_row[12]) if ch_row[12] else [],
                    user_inputs_applied=json.loads(ch_row[13]) if ch_row[13] else [],
                    status=ChapterStatus(ch_row[7]),
                    word_count=ch_row[16],
                    reading_time_minutes=ch_row[17],
                    created_at=ch_row[18],
                    language_refined=bool(ch_row[19]) if ch_row[19] is not None else False,
                    pre_refinement_content=ch_row[20]
                ))

            # Get characters
            cursor.execute("""
                SELECT id, name, role, age, background, appearance, motivation,
                       personality_traits, relationships, progression, character_arc
                FROM characters WHERE story_id = ?
            """, (uuid_id,))
            characters = []
            for char_row in cursor.fetchall():
                characters.append(Character(
                    id=str(char_row[0]),
                    name=char_row[1],
                    role=char_row[2],
                    age=char_row[3],
                    background=char_row[4] or "",
                    appearance=char_row[5],
                    motivation=char_row[6] or "",
                    personality_traits=json.loads(char_row[7]) if char_row[7] else [],
                    relationships=json.loads(char_row[8]) if char_row[8] else {},
                    character_arc=json.loads(char_row[10]) if char_row[10] else None
                ))

            return Story(
                id=str(row[0]),
                prompt=row[3],
                status=StoryStatus(row[4]),
                preferences=preferences,
                structure=structure,
                chapters=chapters,
                characters=characters,
                metadata=StoryMetadata(
                    child_id=str(row[2]),
                    created_at=row[7],
                    updated_at=row[8]
                )
            )

    async def get_story(self, story_id: str) -> Optional[Story]:
        """Get a story by ID."""
        return await self._run_async(self._get_story_sync, story_id)

    async def update_story_status(self, story_id: str, status: str):
        """Update story status."""
        uuid_id = to_uuid(story_id)
        def _update():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE stories SET status = ?, updated_at = GETUTCDATE() WHERE id = ?",
                    (status, uuid_id)
                )
                conn.commit()
        await self._run_async(_update)

    def _get_stories_by_child_sync(self, child_id: str) -> List[Story]:
        """Get all stories for a child."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(child_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM stories WHERE child_id = ? ORDER BY created_at DESC",
                (uuid_id,)
            )
            story_ids = [str(r[0]) for r in cursor.fetchall()]

        # Fetch each story fully
        stories = []
        for sid in story_ids:
            story = self._get_story_sync(sid)
            if story:
                stories.append(story)
        return stories

    async def get_stories_by_child(self, child_id: str) -> List[Story]:
        """Get all stories for a child."""
        return await self._run_async(self._get_stories_by_child_sync, child_id)

    # Alias for compatibility
    async def get_stories_by_user(self, user_id: str) -> List[Story]:
        return await self.get_stories_by_child(user_id)

    async def delete_story(self, story_id: str):
        """Delete a story."""
        uuid_id = to_uuid(story_id)
        def _delete():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM stories WHERE id = ?", (uuid_id,))
                conn.commit()
        await self._run_async(_delete)

    def _get_household_id_for_story_sync(self, story_id: str) -> Optional[str]:
        """Get household_id for a story (for blob storage paths)."""
        uuid_id = to_uuid(story_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT household_id FROM stories WHERE id = ?", (uuid_id,))
            row = cursor.fetchone()
            return str(row[0]) if row else None

    async def get_household_id_for_story(self, story_id: str) -> Optional[str]:
        """Get household_id for a story (for blob storage paths)."""
        return await self._run_async(self._get_household_id_for_story_sync, story_id)

    # =========================================================================
    # Chapter Operations
    # =========================================================================

    def _save_chapter_sync(self, story_id: str, chapter: Chapter) -> Chapter:
        """Save or update a chapter."""
        # Convert string IDs to valid UUIDs for Azure SQL
        uuid_story_id = to_uuid(story_id)
        uuid_chapter_id = to_uuid(chapter.id)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT id FROM chapters WHERE story_id = ? AND number = ?",
                (uuid_story_id, chapter.number)
            )
            exists = cursor.fetchone() is not None

            chars_json = json.dumps(chapter.characters_featured) if chapter.characters_featured else None
            edu_json = json.dumps(chapter.educational_points) if chapter.educational_points else None
            vocab_json = json.dumps([v.model_dump(mode='json') for v in chapter.vocabulary_words]) if chapter.vocabulary_words else None
            facts_json = json.dumps([f.model_dump(mode='json') for f in chapter.facts]) if chapter.facts else None
            stmts_json = json.dumps([s.model_dump(mode='json') for s in chapter.statements]) if chapter.statements else None
            inputs_json = json.dumps(chapter.user_inputs_applied) if chapter.user_inputs_applied else None
            review_json = json.dumps(chapter.round_table_review.model_dump(mode='json')) if chapter.round_table_review else None
            voice_json = json.dumps(chapter.voice_direction_metadata.model_dump(mode='json')) if chapter.voice_direction_metadata else None
            gen_meta_json = json.dumps(chapter.generation_metadata.model_dump(mode='json')) if chapter.generation_metadata else None

            if exists:
                cursor.execute("""
                    UPDATE chapters
                    SET title = ?, synopsis = ?, content = ?, tts_content = ?,
                        audio_blob_url = ?,
                        status = ?, characters_featured = ?, educational_points = ?,
                        vocabulary_words = ?, facts = ?, statements = ?,
                        user_inputs_applied = ?, round_table_review = ?, voice_direction = ?,
                        generation_metadata = ?, tts_status = ?, tts_error = ?,
                        word_count = ?, reading_time_minutes = ?,
                        language_refined = ?, pre_refinement_content = ?
                    WHERE story_id = ? AND number = ?
                """, (
                    chapter.title,
                    chapter.synopsis,
                    chapter.content,
                    chapter.tts_content,
                    chapter.audio_blob_url,
                    chapter.status.value,
                    chars_json,
                    edu_json,
                    vocab_json,
                    facts_json,
                    stmts_json,
                    inputs_json,
                    review_json,
                    voice_json,
                    gen_meta_json,
                    chapter.tts_status,
                    chapter.tts_error,
                    chapter.word_count,
                    chapter.reading_time_minutes,
                    1 if chapter.language_refined else 0,
                    chapter.pre_refinement_content,
                    uuid_story_id,
                    chapter.number
                ))
            else:
                cursor.execute("""
                    INSERT INTO chapters
                    (id, story_id, number, title, synopsis, content, tts_content,
                     audio_blob_url,
                     status, characters_featured, educational_points, vocabulary_words,
                     facts, statements, user_inputs_applied, round_table_review,
                     voice_direction, generation_metadata, tts_status, tts_error,
                     word_count, reading_time_minutes, language_refined, pre_refinement_content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    uuid_chapter_id,
                    uuid_story_id,
                    chapter.number,
                    chapter.title,
                    chapter.synopsis,
                    chapter.content,
                    chapter.tts_content,
                    chapter.audio_blob_url,
                    chapter.status.value,
                    chars_json,
                    edu_json,
                    vocab_json,
                    facts_json,
                    stmts_json,
                    inputs_json,
                    review_json,
                    voice_json,
                    gen_meta_json,
                    chapter.tts_status,
                    chapter.tts_error,
                    chapter.word_count,
                    chapter.reading_time_minutes,
                    1 if chapter.language_refined else 0,
                    chapter.pre_refinement_content
                ))

            conn.commit()
        return chapter

    async def save_chapter(self, story_id: str, chapter: Chapter) -> Chapter:
        """Save or update a chapter."""
        return await self._run_async(self._save_chapter_sync, story_id, chapter)

    def _get_chapter_sync(self, story_id: str, chapter_number: int) -> Optional[Chapter]:
        """Get a specific chapter."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_story_id = to_uuid(story_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, number, title, synopsis, content, tts_content, audio_blob_url,
                       status, characters_featured, educational_points, vocabulary_words,
                       facts, statements, user_inputs_applied, round_table_review,
                       voice_direction, generation_metadata, tts_status, tts_error,
                       word_count, reading_time_minutes, created_at,
                       language_refined, pre_refinement_content
                FROM chapters WHERE story_id = ? AND number = ?
            """, (uuid_story_id, chapter_number))
            row = cursor.fetchone()

            if not row:
                return None

            # Parse generation_metadata if present
            gen_metadata = None
            if row[16]:  # generation_metadata column
                from src.models.models import GenerationMetadata
                gen_metadata = GenerationMetadata(**json.loads(row[16]))

            return Chapter(
                id=str(row[0]),
                number=row[1],
                title=row[2],
                synopsis=row[3] or "",
                content=row[4] or "",
                tts_content=row[5],
                audio_blob_url=row[6],
                status=ChapterStatus(row[7]),
                characters_featured=json.loads(row[8]) if row[8] else [],
                educational_points=json.loads(row[9]) if row[9] else [],
                vocabulary_words=json.loads(row[10]) if row[10] else [],
                facts=json.loads(row[11]) if row[11] else [],
                statements=json.loads(row[12]) if row[12] else [],
                user_inputs_applied=json.loads(row[13]) if row[13] else [],
                generation_metadata=gen_metadata,
                tts_status=row[17] or "pending",
                tts_error=row[18],
                word_count=row[19],
                reading_time_minutes=row[20],
                created_at=row[21],
                language_refined=bool(row[22]) if row[22] is not None else False,
                pre_refinement_content=row[23]
            )

    async def get_chapter(self, story_id: str, chapter_number: int) -> Optional[Chapter]:
        """Get a specific chapter."""
        return await self._run_async(self._get_chapter_sync, story_id, chapter_number)

    def _get_chapters_sync(self, story_id: str) -> List[Chapter]:
        """Get all chapters for a story."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(story_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, number, title, synopsis, content, tts_content, audio_blob_url,
                       status, characters_featured, educational_points, vocabulary_words,
                       facts, statements, user_inputs_applied, round_table_review,
                       voice_direction, generation_metadata, tts_status, tts_error,
                       word_count, reading_time_minutes, created_at,
                       language_refined, pre_refinement_content
                FROM chapters WHERE story_id = ? ORDER BY number
            """, (uuid_id,))

            chapters = []
            for row in cursor.fetchall():
                # Parse generation_metadata if present
                gen_metadata = None
                if row[16]:  # generation_metadata column
                    from src.models.models import GenerationMetadata
                    gen_metadata = GenerationMetadata(**json.loads(row[16]))

                chapters.append(Chapter(
                    id=str(row[0]),
                    number=row[1],
                    title=row[2],
                    synopsis=row[3] or "",
                    content=row[4] or "",
                    tts_content=row[5],
                    audio_blob_url=row[6],
                    status=ChapterStatus(row[7]),
                    characters_featured=json.loads(row[8]) if row[8] else [],
                    educational_points=json.loads(row[9]) if row[9] else [],
                    vocabulary_words=json.loads(row[10]) if row[10] else [],
                    facts=json.loads(row[11]) if row[11] else [],
                    statements=json.loads(row[12]) if row[12] else [],
                    user_inputs_applied=json.loads(row[13]) if row[13] else [],
                    generation_metadata=gen_metadata,
                    tts_status=row[17] or "pending",
                    tts_error=row[18],
                    word_count=row[19],
                    reading_time_minutes=row[20],
                    created_at=row[21],
                    language_refined=bool(row[22]) if row[22] is not None else False,
                    pre_refinement_content=row[23]
                ))
            return chapters

    async def get_chapters(self, story_id: str) -> List[Chapter]:
        """Get all chapters for a story."""
        return await self._run_async(self._get_chapters_sync, story_id)

    async def update_chapter_status(self, story_id: str, chapter_num: int, status: ChapterStatus):
        """Update chapter status."""
        uuid_id = to_uuid(story_id)
        def _update():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE chapters SET status = ? WHERE story_id = ? AND number = ?",
                    (status.value, uuid_id, chapter_num)
                )
                conn.commit()
        await self._run_async(_update)

    # =========================================================================
    # Character Operations
    # =========================================================================

    def _save_character_sync(self, story_id: str, character: Character) -> Character:
        """Save or update a character."""
        # Convert string IDs to valid UUIDs for Azure SQL
        uuid_story_id = to_uuid(story_id)
        uuid_char_id = to_uuid(character.id)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT id FROM characters WHERE id = ?",
                (uuid_char_id,)
            )
            exists = cursor.fetchone() is not None

            traits_json = json.dumps(character.personality_traits) if character.personality_traits else None
            rels_json = json.dumps(character.relationships) if character.relationships else None
            prog_json = json.dumps(character.progression.model_dump(mode='json')) if character.progression else None
            arc_json = json.dumps(character.character_arc) if character.character_arc else None

            if exists:
                cursor.execute("""
                    UPDATE characters
                    SET name = ?, role = ?, age = ?, background = ?, appearance = ?,
                        motivation = ?, personality_traits = ?, relationships = ?,
                        progression = ?, character_arc = ?
                    WHERE id = ?
                """, (
                    character.name,
                    character.role,
                    str(character.age) if character.age else None,
                    character.background,
                    character.appearance,
                    character.motivation,
                    traits_json,
                    rels_json,
                    prog_json,
                    arc_json,
                    uuid_char_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO characters
                    (id, story_id, name, role, age, background, appearance, motivation,
                     personality_traits, relationships, progression, character_arc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    uuid_char_id,
                    uuid_story_id,
                    character.name,
                    character.role,
                    str(character.age) if character.age else None,
                    character.background,
                    character.appearance,
                    character.motivation,
                    traits_json,
                    rels_json,
                    prog_json,
                    arc_json
                ))

            conn.commit()
        return character

    async def save_character(self, story_id: str, character: Character) -> Character:
        """Save or update a character."""
        return await self._run_async(self._save_character_sync, story_id, character)

    async def update_character(self, story_id: str, character: Character) -> Character:
        """Update a character (alias for save_character)."""
        return await self.save_character(story_id, character)

    def _get_characters_sync(self, story_id: str) -> List[Character]:
        """Get all characters for a story."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(story_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, role, age, background, appearance, motivation,
                       personality_traits, relationships, progression, character_arc
                FROM characters WHERE story_id = ?
            """, (uuid_id,))

            characters = []
            for row in cursor.fetchall():
                characters.append(Character(
                    id=str(row[0]),
                    name=row[1],
                    role=row[2],
                    age=row[3],
                    background=row[4] or "",
                    appearance=row[5],
                    motivation=row[6] or "",
                    personality_traits=json.loads(row[7]) if row[7] else [],
                    relationships=json.loads(row[8]) if row[8] else {},
                    character_arc=json.loads(row[10]) if row[10] else None
                ))
            return characters

    async def get_characters(self, story_id: str) -> List[Character]:
        """Get all characters for a story."""
        return await self._run_async(self._get_characters_sync, story_id)

    def _delete_character_sync(self, story_id: str, character_id: str) -> bool:
        """Delete a character from a story (used for deduplication)."""
        uuid_story_id = to_uuid(story_id)
        uuid_char_id = to_uuid(character_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM characters WHERE story_id = ? AND character_id = ?",
                (uuid_story_id, uuid_char_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    async def delete_character(self, story_id: str, character_id: str) -> bool:
        """
        Delete a character from a story (used for deduplication).

        Args:
            story_id: ID of the story
            character_id: ID of the character to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        return await self._run_async(self._delete_character_sync, story_id, character_id)

    # =========================================================================
    # Reading State Operations
    # =========================================================================

    def _get_reading_state_sync(self, story_id: str) -> Optional[StoryReadingState]:
        """Get reading state for a story."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(story_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT story_id, session_id, current_chapter, chapter_position,
                       generating_chapter, chapter_statuses, queued_inputs,
                       chapter_audio_states, queued_messages, playback_phase,
                       discussion_started, started_at, last_active
                FROM reading_states WHERE story_id = ?
            """, (uuid_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Parse JSON fields
            chapter_statuses = {}
            if row[5]:
                raw_statuses = json.loads(row[5])
                chapter_statuses = {k: ChapterStatus(v) for k, v in raw_statuses.items()}

            queued_inputs = []
            if row[6]:
                raw_inputs = json.loads(row[6])
                queued_inputs = [QueuedInput(**inp) for inp in raw_inputs]

            chapter_audio_states = {}
            if row[7]:
                raw_audio = json.loads(row[7])
                chapter_audio_states = {k: ChapterAudioState(**v) for k, v in raw_audio.items()}

            queued_messages = json.loads(row[8]) if row[8] else []

            return StoryReadingState(
                story_id=str(row[0]),
                session_id=row[1],
                current_chapter=row[2],
                chapter_position=row[3],
                generating_chapter=row[4],
                chapter_statuses=chapter_statuses,
                queued_inputs=queued_inputs,
                chapter_audio_states=chapter_audio_states,
                queued_messages=queued_messages,
                playback_phase=row[9],
                discussion_started=bool(row[10]),
                started_at=row[11],
                last_active=row[12]
            )

    async def get_reading_state(self, story_id: str) -> Optional[StoryReadingState]:
        """Get reading state for a story."""
        return await self._run_async(self._get_reading_state_sync, story_id)

    def _update_reading_state_sync(self, story_id: str, state: StoryReadingState) -> StoryReadingState:
        """Save or update reading state."""
        # Convert string ID to valid UUID for Azure SQL
        uuid_id = to_uuid(story_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT story_id FROM reading_states WHERE story_id = ?",
                (uuid_id,)
            )
            exists = cursor.fetchone() is not None

            statuses_json = json.dumps({k: v.value for k, v in state.chapter_statuses.items()})
            inputs_json = json.dumps([inp.model_dump(mode='json') for inp in state.queued_inputs])
            audio_json = json.dumps({k: v.model_dump(mode='json') for k, v in state.chapter_audio_states.items()})
            messages_json = json.dumps(state.queued_messages)

            if exists:
                cursor.execute("""
                    UPDATE reading_states
                    SET session_id = ?, current_chapter = ?, chapter_position = ?,
                        generating_chapter = ?, chapter_statuses = ?, queued_inputs = ?,
                        chapter_audio_states = ?, queued_messages = ?, playback_phase = ?,
                        discussion_started = ?, last_active = GETUTCDATE()
                    WHERE story_id = ?
                """, (
                    state.session_id,
                    state.current_chapter,
                    state.chapter_position,
                    state.generating_chapter,
                    statuses_json,
                    inputs_json,
                    audio_json,
                    messages_json,
                    state.playback_phase.value if hasattr(state.playback_phase, 'value') else state.playback_phase,
                    1 if state.discussion_started else 0,
                    uuid_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO reading_states
                    (story_id, session_id, current_chapter, chapter_position,
                     generating_chapter, chapter_statuses, queued_inputs,
                     chapter_audio_states, queued_messages, playback_phase, discussion_started)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    uuid_id,
                    state.session_id,
                    state.current_chapter,
                    state.chapter_position,
                    state.generating_chapter,
                    statuses_json,
                    inputs_json,
                    audio_json,
                    messages_json,
                    state.playback_phase.value if hasattr(state.playback_phase, 'value') else state.playback_phase,
                    1 if state.discussion_started else 0
                ))

            conn.commit()
        return state

    async def update_reading_state(self, story_id: str, state: StoryReadingState) -> StoryReadingState:
        """Save or update reading state."""
        return await self._run_async(self._update_reading_state_sync, story_id, state)

    async def create_reading_state(self, story_id: str) -> StoryReadingState:
        """Create a new reading state."""
        state = StoryReadingState(
            story_id=story_id,
            started_at=datetime.now(),
            last_active=datetime.now()
        )
        return await self.update_reading_state(story_id, state)

    async def queue_user_input(self, story_id: str, queued_input: QueuedInput) -> QueuedInput:
        """Queue user input for future chapter."""
        state = await self.get_reading_state(story_id)
        if not state:
            state = await self.create_reading_state(story_id)

        state.queued_inputs.append(queued_input)
        state.last_active = datetime.now()
        await self.update_reading_state(story_id, state)
        return queued_input

    async def get_queued_inputs(self, story_id: str, chapter_num: int) -> List[QueuedInput]:
        """Get queued inputs for a chapter."""
        state = await self.get_reading_state(story_id)
        if not state:
            return []
        return [
            inp for inp in state.queued_inputs
            if inp.target_chapter == chapter_num and not inp.applied
        ]

    async def mark_input_applied(self, story_id: str, input_id: str, chapter_num: int):
        """Mark an input as applied."""
        state = await self.get_reading_state(story_id)
        if not state:
            return

        for inp in state.queued_inputs:
            if inp.id == input_id:
                inp.applied = True
                inp.applied_at = datetime.now()
                break

        await self.update_reading_state(story_id, state)

    async def update_reading_position(self, story_id: str, chapter_num: int, position: float):
        """Update reading position."""
        state = await self.get_reading_state(story_id)
        if not state:
            state = await self.create_reading_state(story_id)

        state.current_chapter = chapter_num
        state.chapter_position = position
        state.last_active = datetime.now()

        if position < 0.1:
            state.set_chapter_status(chapter_num, ChapterStatus.READING)

        await self.update_reading_state(story_id, state)

    async def mark_chapter_completed(self, story_id: str, chapter_num: int):
        """Mark chapter as completed."""
        state = await self.get_reading_state(story_id)
        if not state:
            return

        state.set_chapter_status(chapter_num, ChapterStatus.COMPLETED)
        state.chapter_position = 1.0
        state.last_active = datetime.now()

        await self.update_reading_state(story_id, state)
        await self.update_chapter_status(story_id, chapter_num, ChapterStatus.COMPLETED)

    async def set_generating_chapter(self, story_id: str, chapter_num: int):
        """Set which chapter is generating."""
        state = await self.get_reading_state(story_id)
        if not state:
            state = await self.create_reading_state(story_id)

        state.generating_chapter = chapter_num
        state.set_chapter_status(chapter_num, ChapterStatus.GENERATING)
        state.last_active = datetime.now()

        await self.update_reading_state(story_id, state)
        await self.update_chapter_status(story_id, chapter_num, ChapterStatus.GENERATING)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def generate_id(self, prefix: str = "story") -> str:
        """Generate a new unique ID."""
        return f"{prefix}_{uuid4().hex[:12]}"

    async def save_structure(self, story_id: str, structure: Dict[str, Any]):
        """Save story structure."""
        uuid_id = to_uuid(story_id)
        def _save():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                structure_json = json.dumps(structure)
                cursor.execute(
                    "UPDATE stories SET structure = ?, updated_at = GETUTCDATE() WHERE id = ?",
                    (structure_json, uuid_id)
                )
                conn.commit()
        await self._run_async(_save)

    async def get_structure(self, story_id: str) -> Optional[Dict[str, Any]]:
        """Get story structure."""
        uuid_id = to_uuid(story_id)
        def _get():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT structure FROM stories WHERE id = ?", (uuid_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return None
        return await self._run_async(_get)

    async def append_style_request(self, story_id: str, style_request: str) -> bool:
        """Append a style request to story preferences."""
        uuid_id = to_uuid(story_id)
        def _append():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT preferences FROM stories WHERE id = ?", (uuid_id,))
                row = cursor.fetchone()
                if not row:
                    return False

                prefs = json.loads(row[0]) if row[0] else {}
                style_requests = prefs.get('user_style_requests', [])
                if style_request not in style_requests:
                    style_requests.append(style_request)
                    prefs['user_style_requests'] = style_requests
                    cursor.execute(
                        "UPDATE stories SET preferences = ?, updated_at = GETUTCDATE() WHERE id = ?",
                        (json.dumps(prefs), uuid_id)
                    )
                    conn.commit()
                return True
        return await self._run_async(_append)

    # =========================================================================
    # Dialogue Operations
    # =========================================================================

    async def save_dialogue(self, story_id: str, dialogue: DialogueEntry) -> DialogueEntry:
        """Save a dialogue entry."""
        uuid_story_id = to_uuid(story_id)
        uuid_dialogue_id = to_uuid(dialogue.id)
        def _save():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO dialogues (id, story_id, speaker, message, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    uuid_dialogue_id,
                    uuid_story_id,
                    dialogue.speaker,
                    dialogue.message,
                    json.dumps(dialogue.metadata) if dialogue.metadata else None
                ))
                conn.commit()
            return dialogue
        return await self._run_async(_save)

    async def get_dialogues(self, story_id: str, limit: int = 50) -> List[DialogueEntry]:
        """Get dialogue history for a story."""
        uuid_id = to_uuid(story_id)
        def _get():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT TOP (?) id, speaker, message, metadata, timestamp
                    FROM dialogues WHERE story_id = ?
                    ORDER BY timestamp DESC
                """, (limit, uuid_id))

                dialogues = []
                for row in cursor.fetchall():
                    dialogues.append(DialogueEntry(
                        id=str(row[0]),
                        speaker=row[1],
                        message=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        timestamp=row[4]
                    ))
                return list(reversed(dialogues))  # Return in chronological order
        return await self._run_async(_get)

    # =========================================================================
    # Fact Check Operations
    # =========================================================================

    async def save_factcheck_report(self, story_id: str, report: FactCheckReport):
        """Save a fact-check report (only if issues found)."""
        if len(report.issues_found) == 0:
            return

        uuid_id = to_uuid(story_id)
        def _save():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO fact_check_reports
                    (story_id, chapter_number, issues_found, approval_status,
                     overall_confidence, revision_round)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    uuid_id,
                    report.chapter_number,
                    json.dumps([i.model_dump(mode='json') for i in report.issues_found]),
                    report.approval_status,
                    report.overall_confidence,
                    report.revision_round
                ))
                conn.commit()
        await self._run_async(_save)

    async def get_factcheck_reports(
        self,
        story_id: str,
        chapter_number: Optional[int] = None
    ) -> List[FactCheckReport]:
        """Get fact-check reports for a story."""
        uuid_id = to_uuid(story_id)
        def _get():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if chapter_number is not None:
                    cursor.execute("""
                        SELECT chapter_number, issues_found, approval_status,
                               overall_confidence, timestamp, revision_round
                        FROM fact_check_reports
                        WHERE story_id = ? AND chapter_number = ?
                    """, (uuid_id, chapter_number))
                else:
                    cursor.execute("""
                        SELECT chapter_number, issues_found, approval_status,
                               overall_confidence, timestamp, revision_round
                        FROM fact_check_reports WHERE story_id = ?
                    """, (uuid_id,))

                reports = []
                for row in cursor.fetchall():
                    from src.models import FactCheckIssue
                    issues = [FactCheckIssue(**i) for i in json.loads(row[1])] if row[1] else []
                    reports.append(FactCheckReport(
                        chapter_number=row[0],
                        issues_found=issues,
                        approval_status=row[2],
                        overall_confidence=row[3],
                        timestamp=row[4].isoformat() if row[4] else datetime.now().isoformat(),
                        revision_round=row[5]
                    ))
                return reports
        return await self._run_async(_get)
