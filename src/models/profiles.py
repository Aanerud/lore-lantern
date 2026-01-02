"""
Profile Models for Lore Lantern Family Account System

This module implements a Parent/Child profile hierarchy:
- ParentAccount: Owns the Lantern, manages children, sets family language
- ChildProfile: Simple profile (name, birth_year) linked to stories
- LearningProgress: Backend-only tracking (not exposed to parents)

GDPR Considerations:
- ChildProfile stores minimal data (name, birth_year only)
- LearningProgress is stored separately and can be deleted independently
- No personal identifiers beyond what's needed for personalization
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime
import uuid


# ============================================================================
# Parent Account Model
# ============================================================================

class ParentAccount(BaseModel):
    """
    Parent account - the Lantern owner.

    One parent account owns the physical Lantern device and manages
    all child profiles. Language is set at this level for the whole family.
    """
    parent_id: str = Field(
        default_factory=lambda: f"parent_{uuid.uuid4().hex[:12]}",
        description="Unique parent account identifier"
    )

    # Family-wide settings
    language: str = Field(
        default="en",
        pattern="^(en|es|no)$",
        description="Family language preference for all stories"
    )

    # Children managed by this parent
    child_ids: List[str] = Field(
        default_factory=list,
        description="List of child profile IDs owned by this parent"
    )

    # Optional display name
    display_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Parent's display name or family name (optional)"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ParentAccountCreate(BaseModel):
    """Request model for creating a parent account"""
    parent_id: Optional[str] = Field(default=None, max_length=100, description="Optional custom parent ID. If not provided, auto-generated.")
    language: str = Field(default="en", pattern="^(en|es|no)$")
    display_name: Optional[str] = Field(default=None, max_length=100)


class ParentAccountUpdate(BaseModel):
    """Request model for updating parent account"""
    language: Optional[str] = Field(default=None, pattern="^(en|es|no)$")
    display_name: Optional[str] = Field(default=None, max_length=100)


# ============================================================================
# Child Profile Model
# ============================================================================

class ChildProfile(BaseModel):
    """
    Child profile - a young user of the Lantern.

    Contains only GDPR-compliant minimal data visible to parents:
    - Name (for personalization in stories)
    - Birth year (for age-appropriate content)

    Learning progress is stored SEPARATELY in LearningProgress model
    and is NOT exposed via parent-facing APIs.
    """
    child_id: str = Field(
        default_factory=lambda: f"child_{uuid.uuid4().hex[:12]}",
        description="Unique child profile identifier"
    )

    # Link to parent
    parent_id: str = Field(
        ...,
        description="Parent account that owns this child profile"
    )

    # Minimal GDPR-compliant data
    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Child's first name for story personalization"
    )

    birth_year: int = Field(
        ...,
        ge=1920,
        le=2025,
        description="Birth year for age calculation (more stable than storing age)"
    )

    # Story library (references, not full stories)
    story_ids: List[str] = Field(
        default_factory=list,
        description="All story IDs belonging to this child"
    )

    # Active story tracking (for "continue" functionality)
    active_story_id: Optional[str] = Field(
        default=None,
        description="Currently active/paused story for quick resume"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def current_age(self) -> int:
        """Calculate current age from birth_year"""
        return datetime.now().year - self.birth_year

    def get_difficulty_level(self) -> str:
        """
        Return difficulty level based on age.
        Used by story generation for content adaptation.
        """
        age = self.current_age
        if age <= 5:
            return "easy"      # 3-5: Simple language, shorter stories
        elif age <= 9:
            return "medium"    # 6-9: Building vocabulary
        else:
            return "hard"      # 10+: Complex themes


class ChildProfileCreate(BaseModel):
    """Request model for creating a user profile"""
    child_id: Optional[str] = Field(default=None, max_length=100, description="Optional custom user ID. If not provided, auto-generated.")
    name: str = Field(..., min_length=1, max_length=50)
    birth_year: int = Field(..., ge=1920, le=2025)  # Allow users of any age


class ChildProfileUpdate(BaseModel):
    """Request model for updating a user profile"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=50)
    birth_year: Optional[int] = Field(default=None, ge=1920, le=2025)  # Allow users of any age


class ChildProfileResponse(BaseModel):
    """
    Response model for child profile (parent-facing).

    Intentionally EXCLUDES learning progress data.
    Parents see only: name, age, story count.
    """
    child_id: str
    name: str
    current_age: int
    story_count: int
    active_story_id: Optional[str]
    created_at: datetime


# ============================================================================
# Learning Progress Model (Backend-Only)
# ============================================================================

class VocabularyEntry(BaseModel):
    """A word in the child's vocabulary bank"""
    word: str
    definition: str
    first_encountered_story_id: str
    times_encountered: int = Field(default=1, ge=1)
    mastery_level: int = Field(default=1, ge=1, le=5)  # 1=new, 5=mastered


class ConceptMastery(BaseModel):
    """Tracking mastery of an educational concept"""
    concept: str
    description: str
    stories_encountered: List[str] = Field(default_factory=list)
    mastery_level: int = Field(default=1, ge=1, le=5)
    first_learned_at: datetime = Field(default_factory=datetime.now)


class LearningProgress(BaseModel):
    """
    Backend learning progress tracking per child.

    IMPORTANT: This data is NOT exposed to parents via API.
    It's used internally by the story generation system to:
    - Adapt vocabulary complexity
    - Track mastered concepts (avoid repetition)
    - Adjust reading level
    - Personalize educational content

    GDPR Note: This is aggregate/derived data from story interactions.
    Stored separately for privacy-by-design and easy deletion.
    """
    child_id: str = Field(
        ...,
        description="Child this progress belongs to"
    )

    # Vocabulary tracking
    vocabulary_bank: Dict[str, VocabularyEntry] = Field(
        default_factory=dict,
        description="Words learned: word -> entry with definition, mastery"
    )

    # Concept mastery
    concepts_mastered: Dict[str, ConceptMastery] = Field(
        default_factory=dict,
        description="Educational concepts: concept -> mastery details"
    )

    # Reading progression
    reading_level: int = Field(
        default=1,
        ge=1,
        le=10,
        description="1-10 scale based on complexity handled"
    )

    # Engagement metrics (for internal adaptation, not gamification)
    total_stories_completed: int = Field(default=0, ge=0)
    total_chapters_read: int = Field(default=0, ge=0)
    total_questions_asked: int = Field(default=0, ge=0)

    # Theme/interest detection (for story recommendations)
    detected_interests: List[str] = Field(
        default_factory=list,
        description="Auto-detected from story choices and engagement"
    )

    # Preference learning
    preferred_story_length: str = Field(
        default="medium",
        description="short/medium/long based on completion patterns"
    )
    preferred_scary_level: str = Field(
        default="mild",
        description="mild/medium/exciting based on story choices"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

    def add_vocabulary(self, word: str, definition: str, story_id: str):
        """Add or update a word in vocabulary bank"""
        if word.lower() in self.vocabulary_bank:
            entry = self.vocabulary_bank[word.lower()]
            entry.times_encountered += 1
            # Increase mastery if seen multiple times
            if entry.times_encountered >= 3 and entry.mastery_level < 5:
                entry.mastery_level = min(entry.mastery_level + 1, 5)
        else:
            self.vocabulary_bank[word.lower()] = VocabularyEntry(
                word=word,
                definition=definition,
                first_encountered_story_id=story_id
            )
        self.updated_at = datetime.now()

    def add_concept(self, concept: str, description: str, story_id: str):
        """Add or update a concept in mastery tracking"""
        if concept.lower() in self.concepts_mastered:
            entry = self.concepts_mastered[concept.lower()]
            if story_id not in entry.stories_encountered:
                entry.stories_encountered.append(story_id)
            # Increase mastery if encountered in multiple stories
            if len(entry.stories_encountered) >= 2 and entry.mastery_level < 5:
                entry.mastery_level = min(entry.mastery_level + 1, 5)
        else:
            self.concepts_mastered[concept.lower()] = ConceptMastery(
                concept=concept,
                description=description,
                stories_encountered=[story_id]
            )
        self.updated_at = datetime.now()

    def record_story_completion(self):
        """Record a completed story"""
        self.total_stories_completed += 1
        self.last_activity = datetime.now()
        self.updated_at = datetime.now()

    def record_chapter_read(self):
        """Record a chapter being read"""
        self.total_chapters_read += 1
        self.last_activity = datetime.now()
        self.updated_at = datetime.now()

    def record_question_asked(self):
        """Record a question being asked during story"""
        self.total_questions_asked += 1
        self.last_activity = datetime.now()
        self.updated_at = datetime.now()


# ============================================================================
# Story Library Response Models
# ============================================================================

class StoryLibraryItem(BaseModel):
    """A story in a child's library (summary view)"""
    story_id: str
    title: str
    prompt: str
    status: str  # "in_progress", "paused", "completed"
    chapters_completed: int
    total_chapters: int
    last_activity: datetime
    can_continue: bool


class StoryLibraryResponse(BaseModel):
    """Response for child's story library"""
    child_id: str
    total_stories: int
    stories: List[StoryLibraryItem]
