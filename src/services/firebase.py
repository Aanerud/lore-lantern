"""
Firebase service for Kids Storyteller V2

Handles all Firebase Realtime Database operations for stories, characters, chapters, etc.
"""

import firebase_admin
from firebase_admin import credentials, db
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import os
import time
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models import (
    Story, Chapter, Character, DialogueEntry, FactCheckReport, UserProfile,
    ChapterStatus, StoryReadingState, QueuedInput, ChapterAudioState,
    # New profile models
    ParentAccount, ChildProfile, LearningProgress
)
import re


def sanitize_firebase_key(key: str) -> str:
    """
    Sanitize a string to be a valid Firebase Realtime Database key.
    Firebase keys cannot contain: . $ # [ ] /
    """
    if not isinstance(key, str):
        return str(key)
    # Replace forbidden characters with underscores
    return re.sub(r'[.$#\[\]/]', '_', key)


def sanitize_firebase_data(data: Any) -> Any:
    """
    Recursively sanitize data for Firebase storage.
    Cleans keys in dicts and handles nested structures.
    """
    if isinstance(data, dict):
        return {
            sanitize_firebase_key(k): sanitize_firebase_data(v)
            for k, v in data.items()
            if k  # Skip empty keys
        }
    elif isinstance(data, list):
        return [sanitize_firebase_data(item) for item in data]
    else:
        return data


class FirebaseService:
    """Service for Firebase Realtime Database operations"""

    def __init__(self, database_url: str, credentials_dict: Optional[Dict[str, Any]] = None, credentials_path: Optional[str] = None, logger=None):
        """
        Initialize Firebase service.

        Args:
            database_url: Firebase Realtime Database URL
            credentials_dict: Optional dict with Firebase credentials (project_id, client_email, private_key)
            credentials_path: Optional path to Firebase service account JSON
            logger: Optional logger instance for debug logging
        """
        self.database_url = database_url
        self.credentials_dict = credentials_dict
        self.credentials_path = credentials_path
        self.logger = logger
        self._initialized = False
        # Thread pool for async Firebase operations (firebase_admin is synchronous)
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="firebase")

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous Firebase operation in the thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def shutdown(self):
        """Shutdown the thread pool executor. Call during app shutdown."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def initialize(self):
        """Initialize Firebase app (call once at startup)"""
        if self._initialized:
            return

        try:
            # Check if app already exists
            firebase_admin.get_app()
            print("   Firebase app already initialized")
        except ValueError:
            # App doesn't exist, initialize it
            try:
                if self.credentials_dict:
                    # Use credentials from dict (environment variables)
                    print("   Initializing Firebase with credentials from environment variables")
                    cred = credentials.Certificate(self.credentials_dict)
                elif self.credentials_path and os.path.exists(self.credentials_path):
                    # Use credentials from file (path redacted for security)
                    print("   Initializing Firebase with credentials file: [REDACTED]")
                    cred = credentials.Certificate(self.credentials_path)
                else:
                    # Try Application Default Credentials
                    print("   Initializing Firebase with Application Default Credentials")
                    cred = credentials.ApplicationDefault()

                firebase_admin.initialize_app(cred, {
                    'databaseURL': self.database_url
                })
            except Exception as e:
                print(f"   Warning: Firebase initialization error: {e}")
                print("   Attempting to initialize without credentials (for testing)...")
                # For testing, try to initialize with minimal config
                firebase_admin.initialize_app(options={
                    'databaseURL': self.database_url
                })

        self._initialized = True
        self.db = db.reference()

    # Story Operations

    async def save_story(self, story: Story) -> Story:
        """
        Save or update a story in Firebase.

        Args:
            story: Story object to save

        Returns:
            The saved story
        """
        start_time = time.time()
        story_data = story.model_dump(mode='json')

        def _sync_save():
            story_ref = self.db.child(f'stories/{story.id}')
            story_ref.set(story_data)

        await self._run_sync(_sync_save)

        # Log Firebase operation
        if self.logger:
            duration = time.time() - start_time
            data_size = len(str(story_data))
            self.logger.firebase_operation(
                operation="set",
                path=f"stories/{story.id}",
                data_summary=f"Story: {story.structure.title if story.structure else 'untitled'}",
                size_bytes=data_size,
                duration=duration
            )

        return story

    async def get_story(self, story_id: str) -> Optional[Story]:
        """
        Retrieve a story from Firebase.

        Args:
            story_id: ID of the story

        Returns:
            Story object or None if not found
        """
        start_time = time.time()

        def _sync_get():
            story_ref = self.db.child(f'stories/{story_id}')
            return story_ref.get()

        data = await self._run_sync(_sync_get)

        # Log Firebase read
        if self.logger:
            duration = time.time() - start_time
            result_summary = "Story found" if data else "Story not found"
            data_size = len(str(data)) if data else 0
            self.logger.firebase_read(
                path=f"stories/{story_id}",
                result_summary=result_summary,
                size_bytes=data_size,
                duration=duration
            )

        if data:
            # FIX: Clean character_arc fields that may be lists (convert to dict)
            if 'characters' in data and data['characters']:
                for char in data['characters']:
                    if 'character_arc' in char and isinstance(char['character_arc'], list):
                        # Filter out None values and convert to dict with chapter numbers as keys
                        arc_list = char['character_arc']
                        if arc_list:
                            char['character_arc'] = {str(i+1): arc_list[i] for i in range(len(arc_list)) if arc_list[i] is not None}
                            if not char['character_arc']:  # If empty after filtering
                                char['character_arc'] = None
                        else:
                            char['character_arc'] = None
            return Story(**data)
        return None

    async def update_story_status(self, story_id: str, status: str):
        """Update story status"""
        self.db.child(f'stories/{story_id}/status').set(status)

    async def append_style_request(self, story_id: str, style_request: str) -> bool:
        """
        Append a user style request to the story preferences.

        These are collected during CompanionAgent conversation (e.g., "make it funny")
        and applied to ALL chapters during generation.

        Args:
            story_id: The story ID
            style_request: The style preference to add (e.g., "make it funny")

        Returns:
            True if successful
        """
        try:
            # Get current style requests
            pref_ref = self.db.child(f'stories/{story_id}/preferences/user_style_requests')
            current = pref_ref.get() or []

            # Avoid duplicates
            if style_request not in current:
                current.append(style_request)
                pref_ref.set(current)
                self._log_operation("WRITE", f"stories/{story_id}/preferences/user_style_requests",
                                   f"Added style request: {style_request}")

            return True
        except Exception as e:
            self._log_operation("ERROR", f"stories/{story_id}/preferences/user_style_requests",
                               f"Failed to append style request: {e}")
            return False

    async def get_stories_by_user(self, user_id: str) -> List[Story]:
        """Get all stories for a user"""
        stories_ref = self.db.child('stories')
        all_stories = stories_ref.order_by_child('metadata/user_id').equal_to(user_id).get()

        if not all_stories:
            return []

        return [Story(**data) for data in all_stories.values()]

    # Chapter Operations

    async def save_chapter(self, story_id: str, chapter: Chapter) -> Chapter:
        """
        Save a chapter to a story.

        Args:
            story_id: ID of the story
            chapter: Chapter object to save

        Returns:
            The saved chapter
        """
        start_time = time.time()
        chapter_ref = self.db.child(f'stories/{story_id}/chapters')

        # Get existing chapters
        existing_chapters = chapter_ref.get() or []
        chapter_data = chapter.model_dump(mode='json')

        # Sanitize chapter data to prevent Firebase InvalidArgumentError
        # (handles smart quotes, em-dashes, control characters from LLM output)
        chapter_data = self._sanitize_for_firebase(chapter_data)

        # Check if chapter already exists (by number)
        chapter_exists = False
        for i, existing_chapter in enumerate(existing_chapters):
            if existing_chapter.get('number') == chapter.number:
                existing_chapters[i] = chapter_data
                chapter_exists = True
                break

        if not chapter_exists:
            existing_chapters.append(chapter_data)

        chapter_ref.set(existing_chapters)

        # Log Firebase operation
        if self.logger:
            duration = time.time() - start_time
            data_size = len(str(chapter_data))
            operation = "update" if chapter_exists else "append"
            self.logger.firebase_operation(
                operation=operation,
                path=f"stories/{story_id}/chapters",
                data_summary=f"Chapter #{chapter.number}: {chapter.title} ({chapter.word_count} words)",
                size_bytes=data_size,
                duration=duration
            )

        return chapter

    async def get_chapter(self, story_id: str, chapter_number: int) -> Optional[Chapter]:
        """Get a specific chapter"""
        def _sync_get():
            chapters_ref = self.db.child(f'stories/{story_id}/chapters')
            return chapters_ref.get() or []

        chapters = await self._run_sync(_sync_get)

        for chapter_data in chapters:
            if chapter_data.get('number') == chapter_number:
                return Chapter(**chapter_data)

        return None

    async def get_chapters(self, story_id: str) -> List[Chapter]:
        """
        Get all chapters for a story.

        Args:
            story_id: ID of the story

        Returns:
            List of Chapter objects
        """
        def _sync_get():
            chapters_ref = self.db.child(f'stories/{story_id}/chapters')
            return chapters_ref.get() or []

        chapters_data = await self._run_sync(_sync_get)

        return [Chapter(**chapter) for chapter in chapters_data]

    # Character Operations

    def _sanitize_for_firebase(self, data: Any) -> Any:
        """
        Sanitize data for Firebase storage.

        Handles:
        - datetime objects → ISO format strings
        - Control characters (0x00-0x1F)
        - Smart quotes/apostrophes → standard ASCII
        - Em/en dashes → regular dashes
        - Non-breaking spaces → regular spaces

        Args:
            data: Any data structure (str, dict, list, or other)

        Returns:
            Sanitized data safe for Firebase storage
        """
        # Handle datetime objects first (convert to ISO string)
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, str):
            result = data
            # Replace smart quotes with standard quotes
            result = result.replace('"', '"').replace('"', '"')  # Smart double quotes
            result = result.replace(''', "'").replace(''', "'")  # Smart single quotes
            # Replace em/en dashes with regular dashes
            result = result.replace('—', '-').replace('–', '-')
            # Replace non-breaking space with regular space
            result = result.replace('\u00a0', ' ')
            # Remove control characters (keep newline, carriage return, tab)
            result = ''.join(
                char if ord(char) >= 32 or char in '\n\r\t' else ''
                for char in result
            )
            return result
        elif isinstance(data, dict):
            # Sanitize both keys AND values
            # Firebase keys cannot contain: . $ # [ ] /
            sanitized_dict = {}
            for k, v in data.items():
                # Sanitize the key - replace invalid characters
                safe_key = str(k)
                safe_key = safe_key.replace('/', '-')  # Forward slash → dash
                safe_key = safe_key.replace('.', '_')  # Period → underscore
                safe_key = safe_key.replace('$', '_')  # Dollar → underscore
                safe_key = safe_key.replace('#', '_')  # Hash → underscore
                safe_key = safe_key.replace('[', '(')  # Brackets → parentheses
                safe_key = safe_key.replace(']', ')')
                sanitized_dict[safe_key] = self._sanitize_for_firebase(v)
            return sanitized_dict
        elif isinstance(data, list):
            return [self._sanitize_for_firebase(item) for item in data]
        else:
            return data

    async def save_character(self, story_id: str, character: Character) -> Character:
        """
        Save a character to a story.

        Args:
            story_id: ID of the story
            character: Character object to save

        Returns:
            The saved character
        """
        start_time = time.time()
        characters_ref = self.db.child(f'stories/{story_id}/characters')

        # Get existing characters
        existing_characters = characters_ref.get() or []
        character_data = character.model_dump(mode='json')

        # Sanitize character data to prevent Firebase InvalidArgumentError
        # (handles smart quotes, em-dashes, control characters from LLM output)
        character_data = self._sanitize_for_firebase(character_data)

        # Check if character already exists (by ID)
        character_exists = False
        for i, existing_char in enumerate(existing_characters):
            if existing_char.get('id') == character.id:
                existing_characters[i] = character_data
                character_exists = True
                break

        if not character_exists:
            existing_characters.append(character_data)

        characters_ref.set(existing_characters)

        # Log Firebase operation
        if self.logger:
            duration = time.time() - start_time
            data_size = len(str(character_data))
            operation = "update" if character_exists else "append"
            self.logger.firebase_operation(
                operation=operation,
                path=f"stories/{story_id}/characters",
                data_summary=f"Character: {character.name} ({character.role})",
                size_bytes=data_size,
                duration=duration
            )

        return character

    async def update_character(self, story_id: str, character: Character) -> Character:
        """
        Update an existing character (alias for save_character with progression updates).

        Args:
            story_id: ID of the story
            character: Character object with updated progression data

        Returns:
            The updated character
        """
        return await self.save_character(story_id, character)

    async def get_characters(self, story_id: str) -> List[Character]:
        """Get all characters for a story"""
        start_time = time.time()
        characters_ref = self.db.child(f'stories/{story_id}/characters')
        characters_data = characters_ref.get() or []

        # Log Firebase read
        if self.logger:
            duration = time.time() - start_time
            count = len(characters_data)
            data_size = len(str(characters_data))
            self.logger.firebase_read(
                path=f"stories/{story_id}/characters",
                result_summary=f"{count} characters found",
                size_bytes=data_size,
                duration=duration
            )

        # FIX: Clean character_arc fields that may be lists (convert to dict)
        for char in characters_data:
            if 'character_arc' in char and isinstance(char['character_arc'], list):
                # Filter out None values and convert to dict with chapter numbers as keys
                arc_list = char['character_arc']
                if arc_list:
                    char['character_arc'] = {str(i+1): arc_list[i] for i in range(len(arc_list)) if arc_list[i] is not None}
                    if not char['character_arc']:  # If empty after filtering
                        char['character_arc'] = None
                else:
                    char['character_arc'] = None

        return [Character(**char) for char in characters_data]

    async def delete_character(self, story_id: str, character_id: str) -> bool:
        """
        Delete a character from a story (used for deduplication).

        Args:
            story_id: ID of the story
            character_id: ID of the character to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        start_time = time.time()
        characters_ref = self.db.child(f'stories/{story_id}/characters')
        characters_data = characters_ref.get() or []

        # Find and remove the character by ID
        original_count = len(characters_data)
        characters_data = [c for c in characters_data if c.get('id') != character_id]

        if len(characters_data) == original_count:
            # Character not found by 'id', try 'character_id' field
            characters_data = [c for c in characters_data if c.get('character_id') != character_id]

        if len(characters_data) < original_count:
            # Character was removed, save the updated list
            characters_ref.set(characters_data)

            if self.logger:
                duration = time.time() - start_time
                self.logger.firebase_write(
                    path=f"stories/{story_id}/characters",
                    data_summary=f"Deleted character {character_id}",
                    duration=duration
                )
            return True

        return False

    # Dialogue Operations

    async def save_dialogue(self, story_id: str, dialogue: DialogueEntry) -> DialogueEntry:
        """
        Save a dialogue entry.

        Args:
            story_id: ID of the story
            dialogue: DialogueEntry object

        Returns:
            The saved dialogue entry
        """
        dialogues_ref = self.db.child(f'stories/{story_id}/dialogues')

        # Get existing dialogues
        existing_dialogues = dialogues_ref.get() or []
        dialogue_data = dialogue.model_dump(mode='json')
        existing_dialogues.append(dialogue_data)

        dialogues_ref.set(existing_dialogues)
        return dialogue

    async def get_dialogues(self, story_id: str, limit: int = 50) -> List[DialogueEntry]:
        """Get dialogue history for a story"""
        dialogues_ref = self.db.child(f'stories/{story_id}/dialogues')
        dialogues_data = dialogues_ref.limit_to_last(limit).get() or []

        return [DialogueEntry(**dlg) for dlg in dialogues_data]

    # User Profile Operations

    async def save_user_profile(self, profile: UserProfile) -> UserProfile:
        """
        Save or update a user profile in Firebase.

        Args:
            profile: UserProfile object to save

        Returns:
            The saved profile
        """
        profile_ref = self.db.child(f'users/{profile.user_id}')
        profile_data = profile.model_dump(mode='json')
        profile_ref.set(profile_data)
        return profile

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile from Firebase.

        Args:
            user_id: ID of the user

        Returns:
            UserProfile object or None if not found
        """
        profile_ref = self.db.child(f'users/{user_id}')
        data = profile_ref.get()

        if data:
            return UserProfile(**data)
        return None

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> UserProfile:
        """
        Update specific fields of a user profile.

        Args:
            user_id: ID of the user
            updates: Dictionary of fields to update

        Returns:
            Updated UserProfile
        """
        profile_ref = self.db.child(f'users/{user_id}')

        # Update only specified fields
        for key, value in updates.items():
            profile_ref.child(key).set(value)

        # Get and return updated profile
        updated_data = profile_ref.get()
        return UserProfile(**updated_data) if updated_data else None

    async def user_exists(self, user_id: str) -> bool:
        """Check if a user profile exists"""
        profile_ref = self.db.child(f'users/{user_id}')
        return profile_ref.get() is not None

    # Structure Operations

    async def save_structure(self, story_id: str, structure: Dict[str, Any]):
        """Save story structure (sanitizes keys for Firebase compatibility)"""
        structure_ref = self.db.child(f'stories/{story_id}/structure')
        # Sanitize data to remove invalid Firebase key characters
        sanitized = sanitize_firebase_data(structure)
        structure_ref.set(sanitized)

    async def get_structure(self, story_id: str) -> Optional[Dict[str, Any]]:
        """Get story structure"""
        structure_ref = self.db.child(f'stories/{story_id}/structure')
        return structure_ref.get()

    # FactCheck Operations

    async def save_factcheck_report(self, story_id: str, report: FactCheckReport) -> None:
        """
        Save a fact-check report (only if issues were found).

        Args:
            story_id: ID of the story
            report: FactCheckReport object

        Note:
            Only saves if issues_found list is not empty, to reduce database clutter.
        """
        # Only save if issues were found
        if len(report.issues_found) > 0:
            reports_ref = self.db.child(f'stories/{story_id}/factcheck_reports')

            # Get existing reports or initialize empty list
            existing_reports = reports_ref.get() or []
            report_data = report.model_dump(mode='json')
            existing_reports.append(report_data)

            reports_ref.set(existing_reports)
            print(f"   ⚠️  Saved fact-check report: {len(report.issues_found)} issues found for chapter {report.chapter_number}")

    async def get_factcheck_reports(self, story_id: str, chapter_number: Optional[int] = None) -> List[FactCheckReport]:
        """
        Get fact-check reports for a story.

        Args:
            story_id: ID of the story
            chapter_number: Optional filter for specific chapter

        Returns:
            List of FactCheckReport objects
        """
        reports_ref = self.db.child(f'stories/{story_id}/factcheck_reports')
        reports_data = reports_ref.get() or []

        reports = [FactCheckReport(**report) for report in reports_data]

        # Filter by chapter if specified
        if chapter_number is not None:
            reports = [r for r in reports if r.chapter_number == chapter_number]

        return reports

    # Utility Methods

    def generate_id(self, prefix: str = "story") -> str:
        """Generate a new Firebase ID"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    async def delete_story(self, story_id: str):
        """Delete a story (use with caution)"""
        self.db.child(f'stories/{story_id}').delete()

    # =====================================================================
    # Hybrid Generation Reading State Operations
    # =====================================================================

    async def update_chapter_status(self, story_id: str, chapter_num: int, status: ChapterStatus) -> None:
        """
        Update the status of a specific chapter in the hybrid generation pipeline.

        Args:
            story_id: ID of the story
            chapter_num: Chapter number (1-indexed)
            status: New ChapterStatus value
        """
        start_time = time.time()

        # Update in reading state
        reading_state_ref = self.db.child(f'reading_states/{story_id}/chapter_statuses/{chapter_num}')
        reading_state_ref.set(status.value)

        # Also update in the chapter itself if it exists
        chapters_ref = self.db.child(f'stories/{story_id}/chapters')
        chapters = chapters_ref.get() or []
        for i, chapter in enumerate(chapters):
            if chapter.get('number') == chapter_num:
                chapters[i]['status'] = status.value
                chapters_ref.set(chapters)
                break

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="update",
                path=f"reading_states/{story_id}/chapter_statuses/{chapter_num}",
                data_summary=f"Chapter {chapter_num} -> {status.value}",
                size_bytes=len(status.value),
                duration=duration
            )

    async def get_reading_state(self, story_id: str) -> Optional[StoryReadingState]:
        """
        Get the reading state for a story session.

        Args:
            story_id: ID of the story

        Returns:
            StoryReadingState or None if not found
        """
        start_time = time.time()

        def _sync_get():
            state_ref = self.db.child(f'reading_states/{story_id}')
            return state_ref.get()

        data = await self._run_sync(_sync_get)

        if self.logger:
            duration = time.time() - start_time
            result_summary = "Reading state found" if data else "No reading state"
            data_size = len(str(data)) if data else 0
            self.logger.firebase_read(
                path=f"reading_states/{story_id}",
                result_summary=result_summary,
                size_bytes=data_size,
                duration=duration
            )

        if data:
            # Convert chapter_statuses string values back to enum
            if 'chapter_statuses' in data:
                chapter_statuses = data['chapter_statuses']
                # Handle case where chapter_statuses is stored as list instead of dict
                if isinstance(chapter_statuses, list):
                    # Convert list to dict: index becomes key
                    chapter_statuses = {str(i+1): status for i, status in enumerate(chapter_statuses) if status}
                elif not isinstance(chapter_statuses, dict):
                    chapter_statuses = {}
                data['chapter_statuses'] = {
                    k: ChapterStatus(v) for k, v in chapter_statuses.items()
                }
            # Convert queued_inputs dicts to QueuedInput objects
            if 'queued_inputs' in data:
                data['queued_inputs'] = [QueuedInput(**inp) for inp in data['queued_inputs']]
            # Convert chapter_audio_states dicts
            if 'chapter_audio_states' in data:
                data['chapter_audio_states'] = {
                    k: ChapterAudioState(**v) for k, v in data['chapter_audio_states'].items()
                }
            return StoryReadingState(**data)
        return None

    async def update_reading_state(self, story_id: str, state: StoryReadingState) -> StoryReadingState:
        """
        Save or update the reading state for a story.

        Args:
            story_id: ID of the story
            state: StoryReadingState to save

        Returns:
            The saved state
        """
        start_time = time.time()
        state_ref = self.db.child(f'reading_states/{story_id}')
        state_data = state.model_dump(mode='json')
        state_ref.set(state_data)

        if self.logger:
            duration = time.time() - start_time
            data_size = len(str(state_data))
            self.logger.firebase_operation(
                operation="set",
                path=f"reading_states/{story_id}",
                data_summary=f"Reading state: ch{state.current_chapter}, gen={state.generating_chapter}",
                size_bytes=data_size,
                duration=duration
            )

        return state

    async def create_reading_state(self, story_id: str) -> StoryReadingState:
        """
        Create a new reading state for a story session.

        Args:
            story_id: ID of the story

        Returns:
            New StoryReadingState
        """
        state = StoryReadingState(
            story_id=story_id,
            started_at=datetime.now(),
            last_active=datetime.now()
        )
        return await self.update_reading_state(story_id, state)

    async def queue_user_input(self, story_id: str, queued_input: QueuedInput) -> QueuedInput:
        """
        Queue a user input for future chapter generation.

        Args:
            story_id: ID of the story
            queued_input: QueuedInput to add to the queue

        Returns:
            The queued input
        """
        start_time = time.time()
        state = await self.get_reading_state(story_id)

        if not state:
            state = await self.create_reading_state(story_id)

        state.queued_inputs.append(queued_input)
        state.last_active = datetime.now()
        await self.update_reading_state(story_id, state)

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="append",
                path=f"reading_states/{story_id}/queued_inputs",
                data_summary=f"Tier {queued_input.tier.value} input for ch{queued_input.target_chapter}",
                size_bytes=len(queued_input.raw_input),
                duration=duration
            )

        return queued_input

    async def get_queued_inputs(self, story_id: str, chapter_num: int) -> List[QueuedInput]:
        """
        Get all unapplied inputs queued for a specific chapter.

        Args:
            story_id: ID of the story
            chapter_num: Target chapter number

        Returns:
            List of QueuedInput objects
        """
        state = await self.get_reading_state(story_id)
        if not state:
            return []

        return [
            inp for inp in state.queued_inputs
            if inp.target_chapter == chapter_num and not inp.applied
        ]

    async def mark_input_applied(self, story_id: str, input_id: str, chapter_num: int) -> None:
        """
        Mark a queued input as applied to a chapter.

        Args:
            story_id: ID of the story
            input_id: ID of the QueuedInput
            chapter_num: Chapter where input was applied
        """
        state = await self.get_reading_state(story_id)
        if not state:
            return

        for inp in state.queued_inputs:
            if inp.id == input_id:
                inp.applied = True
                inp.applied_at = datetime.now()
                break

        # Also update the chapter's user_inputs_applied list
        chapters_ref = self.db.child(f'stories/{story_id}/chapters')
        chapters = chapters_ref.get() or []
        for i, chapter in enumerate(chapters):
            if chapter.get('number') == chapter_num:
                if 'user_inputs_applied' not in chapters[i]:
                    chapters[i]['user_inputs_applied'] = []
                chapters[i]['user_inputs_applied'].append(input_id)
                chapters_ref.set(chapters)
                break

        await self.update_reading_state(story_id, state)

    async def update_reading_position(self, story_id: str, chapter_num: int, position: float) -> None:
        """
        Update the user's reading position within a chapter.

        Args:
            story_id: ID of the story
            chapter_num: Chapter being read
            position: Progress through chapter (0.0-1.0)
        """
        state = await self.get_reading_state(story_id)
        if not state:
            state = await self.create_reading_state(story_id)

        state.current_chapter = chapter_num
        state.chapter_position = position
        state.last_active = datetime.now()

        # If starting to read a chapter, update its status
        if position < 0.1:
            state.set_chapter_status(chapter_num, ChapterStatus.READING)

        await self.update_reading_state(story_id, state)

    async def mark_chapter_completed(self, story_id: str, chapter_num: int) -> None:
        """
        Mark a chapter as completed (user finished reading).

        Args:
            story_id: ID of the story
            chapter_num: Chapter that was completed
        """
        state = await self.get_reading_state(story_id)
        if not state:
            return

        state.set_chapter_status(chapter_num, ChapterStatus.COMPLETED)
        state.chapter_position = 1.0
        state.last_active = datetime.now()

        await self.update_reading_state(story_id, state)
        await self.update_chapter_status(story_id, chapter_num, ChapterStatus.COMPLETED)

    async def set_generating_chapter(self, story_id: str, chapter_num: int) -> None:
        """
        Set which chapter is currently being generated.

        Args:
            story_id: ID of the story
            chapter_num: Chapter being generated
        """
        state = await self.get_reading_state(story_id)
        if not state:
            state = await self.create_reading_state(story_id)

        state.generating_chapter = chapter_num
        state.set_chapter_status(chapter_num, ChapterStatus.GENERATING)
        state.last_active = datetime.now()

        await self.update_reading_state(story_id, state)
        await self.update_chapter_status(story_id, chapter_num, ChapterStatus.GENERATING)

    # ========================================================================
    # Parent Account Operations
    # ========================================================================

    async def save_parent_account(self, parent: ParentAccount) -> ParentAccount:
        """
        Save or update a parent account.

        Args:
            parent: ParentAccount object to save

        Returns:
            The saved parent account
        """
        start_time = time.time()
        parent_ref = self.db.child(f'parents/{parent.parent_id}')
        parent_data = parent.model_dump(mode='json')
        parent_data = self._sanitize_for_firebase(parent_data)
        parent_ref.set(parent_data)

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="write",
                path=f"parents/{parent.parent_id}",
                data_summary=f"Parent: {parent.display_name or parent.parent_id}",
                size_bytes=len(str(parent_data)),
                duration=duration
            )

        return parent

    async def get_parent_account(self, parent_id: str) -> Optional[ParentAccount]:
        """
        Get a parent account by ID.

        Args:
            parent_id: The parent account ID

        Returns:
            ParentAccount or None if not found
        """
        start_time = time.time()
        parent_ref = self.db.child(f'parents/{parent_id}')
        parent_data = parent_ref.get()

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="read",
                path=f"parents/{parent_id}",
                data_summary="Parent account" if parent_data else "Not found",
                size_bytes=len(str(parent_data)) if parent_data else 0,
                duration=duration
            )

        if not parent_data:
            return None

        return ParentAccount(**parent_data)

    async def list_parent_accounts(self) -> List[ParentAccount]:
        """
        List all parent accounts.

        Returns:
            List of ParentAccount objects
        """
        start_time = time.time()
        parents_ref = self.db.child('parents')
        parents_data = parents_ref.get()

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="read",
                path="parents",
                data_summary=f"{len(parents_data) if parents_data else 0} parent accounts",
                size_bytes=len(str(parents_data)) if parents_data else 0,
                duration=duration
            )

        if not parents_data:
            return []

        parents = []
        for parent_id, parent_data in parents_data.items():
            try:
                parents.append(ParentAccount(**parent_data))
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to parse parent {parent_id}: {e}")

        return parents

    async def update_parent_account(self, parent_id: str, updates: Dict[str, Any]) -> Optional[ParentAccount]:
        """
        Update specific fields of a parent account.

        Args:
            parent_id: The parent account ID
            updates: Dict of fields to update

        Returns:
            Updated ParentAccount or None if not found
        """
        parent = await self.get_parent_account(parent_id)
        if not parent:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(parent, key) and value is not None:
                setattr(parent, key, value)

        parent.updated_at = datetime.now()
        return await self.save_parent_account(parent)

    # ========================================================================
    # Child Profile Operations
    # ========================================================================

    async def save_child_profile(self, child: ChildProfile) -> ChildProfile:
        """
        Save or update a child profile.

        Args:
            child: ChildProfile object to save

        Returns:
            The saved child profile
        """
        start_time = time.time()
        child_ref = self.db.child(f'children/{child.child_id}')
        child_data = child.model_dump(mode='json')
        child_data = self._sanitize_for_firebase(child_data)
        child_ref.set(child_data)

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="write",
                path=f"children/{child.child_id}",
                data_summary=f"Child: {child.name}",
                size_bytes=len(str(child_data)),
                duration=duration
            )

        return child

    async def get_child_profile(self, child_id: str) -> Optional[ChildProfile]:
        """
        Get a child profile by ID.

        Args:
            child_id: The child profile ID

        Returns:
            ChildProfile or None if not found
        """
        start_time = time.time()
        child_ref = self.db.child(f'children/{child_id}')
        child_data = child_ref.get()

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="read",
                path=f"children/{child_id}",
                data_summary="Child profile" if child_data else "Not found",
                size_bytes=len(str(child_data)) if child_data else 0,
                duration=duration
            )

        if not child_data:
            return None

        return ChildProfile(**child_data)

    async def get_household_id_for_story(self, story_id: str) -> Optional[str]:
        """
        Get household_id (parent_id) for a story.

        Traverses story -> child -> parent_id to find the household.
        Used for blob storage path organization.

        Args:
            story_id: The story ID

        Returns:
            household_id (parent_id) or None if not found
        """
        story = await self.get_story(story_id)
        if not story or not story.metadata or not story.metadata.child_id:
            return None
        child = await self.get_child_profile(story.metadata.child_id)
        return child.parent_id if child else None

    async def get_children_by_parent(self, parent_id: str) -> List[ChildProfile]:
        """
        Get all children for a parent account.

        Uses parent's child_ids list to fetch children directly,
        avoiding Firebase index requirements.

        Args:
            parent_id: The parent account ID

        Returns:
            List of ChildProfile objects
        """
        start_time = time.time()

        # Get parent to access child_ids list
        parent = await self.get_parent_account(parent_id)
        if not parent or not parent.child_ids:
            if self.logger:
                duration = time.time() - start_time
                self.logger.firebase_operation(
                    operation="read",
                    path=f"children (via parent {parent_id})",
                    data_summary="0 children (no child_ids)",
                    size_bytes=0,
                    duration=duration
                )
            return []

        # Fetch each child by ID
        children = []
        for child_id in parent.child_ids:
            child = await self.get_child_profile(child_id)
            if child:
                children.append(child)

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="read",
                path=f"children (via parent {parent_id})",
                data_summary=f"{len(children)} children fetched",
                size_bytes=len(str(children)),
                duration=duration
            )

        return children

    async def update_child_profile(self, child_id: str, updates: Dict[str, Any]) -> Optional[ChildProfile]:
        """
        Update specific fields of a child profile.

        Args:
            child_id: The child profile ID
            updates: Dict of fields to update

        Returns:
            Updated ChildProfile or None if not found
        """
        child = await self.get_child_profile(child_id)
        if not child:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(child, key) and value is not None:
                setattr(child, key, value)

        child.updated_at = datetime.now()
        return await self.save_child_profile(child)

    async def add_story_to_child(self, child_id: str, story_id: str) -> Optional[ChildProfile]:
        """
        Add a story ID to a child's story list.

        Args:
            child_id: The child profile ID
            story_id: The story ID to add

        Returns:
            Updated ChildProfile or None if not found
        """
        child = await self.get_child_profile(child_id)
        if not child:
            return None

        if story_id not in child.story_ids:
            child.story_ids.append(story_id)
            child.updated_at = datetime.now()
            return await self.save_child_profile(child)

        return child

    async def set_active_story(self, child_id: str, story_id: Optional[str]) -> Optional[ChildProfile]:
        """
        Set the active story for a child (for quick resume).

        Args:
            child_id: The child profile ID
            story_id: The story ID to set as active (or None to clear)

        Returns:
            Updated ChildProfile or None if not found
        """
        child = await self.get_child_profile(child_id)
        if not child:
            return None

        child.active_story_id = story_id
        child.updated_at = datetime.now()
        return await self.save_child_profile(child)

    async def delete_child_profile(self, child_id: str, delete_learning: bool = True) -> bool:
        """
        Soft delete a child profile.

        Args:
            child_id: The child profile ID
            delete_learning: Whether to also delete learning progress (GDPR compliance)

        Returns:
            True if deleted, False if not found
        """
        child = await self.get_child_profile(child_id)
        if not child:
            return False

        # Remove from parent's child_ids list
        parent = await self.get_parent_account(child.parent_id)
        if parent and child_id in parent.child_ids:
            parent.child_ids.remove(child_id)
            await self.save_parent_account(parent)

        # Delete learning progress if requested (GDPR compliance)
        if delete_learning:
            learning_ref = self.db.child(f'learning_progress/{child_id}')
            learning_ref.delete()

        # Delete the child profile
        child_ref = self.db.child(f'children/{child_id}')
        child_ref.delete()

        if self.logger:
            self.logger.info(f"Deleted child profile {child_id} (learning data: {'deleted' if delete_learning else 'preserved'})")

        return True

    # ========================================================================
    # Learning Progress Operations (Backend-Only)
    # ========================================================================

    async def save_learning_progress(self, progress: LearningProgress) -> LearningProgress:
        """
        Save or update learning progress for a child.

        Args:
            progress: LearningProgress object to save

        Returns:
            The saved learning progress
        """
        start_time = time.time()
        progress_ref = self.db.child(f'learning_progress/{progress.child_id}')
        progress_data = progress.model_dump(mode='json')
        progress_data = self._sanitize_for_firebase(progress_data)
        progress_ref.set(progress_data)

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="write",
                path=f"learning_progress/{progress.child_id}",
                data_summary=f"Learning progress: level {progress.reading_level}",
                size_bytes=len(str(progress_data)),
                duration=duration
            )

        return progress

    async def get_learning_progress(self, child_id: str) -> Optional[LearningProgress]:
        """
        Get learning progress for a child.

        Args:
            child_id: The child profile ID

        Returns:
            LearningProgress or None if not found
        """
        start_time = time.time()
        progress_ref = self.db.child(f'learning_progress/{child_id}')
        progress_data = progress_ref.get()

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="read",
                path=f"learning_progress/{child_id}",
                data_summary="Learning progress" if progress_data else "Not found",
                size_bytes=len(str(progress_data)) if progress_data else 0,
                duration=duration
            )

        if not progress_data:
            return None

        return LearningProgress(**progress_data)

    async def get_or_create_learning_progress(self, child_id: str) -> LearningProgress:
        """
        Get learning progress for a child, creating if it doesn't exist.

        Args:
            child_id: The child profile ID

        Returns:
            LearningProgress (existing or newly created)
        """
        progress = await self.get_learning_progress(child_id)
        if progress:
            return progress

        # Create new learning progress
        progress = LearningProgress(child_id=child_id)
        return await self.save_learning_progress(progress)

    async def update_learning_progress(self, child_id: str, updates: Dict[str, Any]) -> Optional[LearningProgress]:
        """
        Update specific fields of learning progress.

        Args:
            child_id: The child profile ID
            updates: Dict of fields to update

        Returns:
            Updated LearningProgress or None if not found
        """
        progress = await self.get_learning_progress(child_id)
        if not progress:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(progress, key) and value is not None:
                setattr(progress, key, value)

        progress.updated_at = datetime.now()
        return await self.save_learning_progress(progress)

    # ========================================================================
    # Story Operations by Child
    # ========================================================================

    async def get_stories_by_child(self, child_id: str) -> List[Story]:
        """
        Get all stories belonging to a child.

        Uses the child's story_ids list to fetch stories individually,
        avoiding the need for Firebase indexing.

        Args:
            child_id: The child profile ID

        Returns:
            List of Story objects
        """
        start_time = time.time()

        # First get the child profile to get their story_ids
        child = await self.get_child_profile(child_id)
        if not child or not child.story_ids:
            if self.logger:
                duration = time.time() - start_time
                self.logger.firebase_operation(
                    operation="query",
                    path=f"children/{child_id}",
                    data_summary=f"No stories for child {child_id}",
                    size_bytes=0,
                    duration=duration
                )
            return []

        # Fetch each story by ID
        stories = []
        for story_id in child.story_ids:
            try:
                story = await self.get_story(story_id)
                if story:
                    stories.append(story)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to fetch story {story_id}: {e}")
                continue

        if self.logger:
            duration = time.time() - start_time
            self.logger.firebase_operation(
                operation="query",
                path="stories",
                data_summary=f"Stories for child {child_id}: {len(stories)} found",
                size_bytes=len(str(stories)) if stories else 0,
                duration=duration
            )

        return stories

    async def update_story_child_id(self, story_id: str, child_id: str) -> bool:
        """
        Update the child_id for a story (used during migration).

        Args:
            story_id: The story ID
            child_id: The child profile ID

        Returns:
            True if updated, False if story not found
        """
        story = await self.get_story(story_id)
        if not story:
            return False

        story.metadata.child_id = child_id
        story.metadata.updated_at = datetime.now()
        await self.save_story(story)
        return True
