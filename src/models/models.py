"""
Pydantic data models for Kids Storyteller V2

These models replace the Joi schemas from Node.js with Python's Pydantic,
providing runtime type validation, automatic API documentation, and better
IDE support.

API LIMITS (user-facing)
========================
When modifying these limits, also update:
- src/api/routes.py (PROMPT_MAX_LENGTH, MESSAGE_MAX_LENGTH constants)
- Frontend error handling in static/js/app.js

| Field              | Min   | Max     | Model            | Notes                              |
|--------------------|-------|---------|------------------|------------------------------------|
| Story.prompt       | 10    | 20,000  | Story            | Initial story request + reference  |
| StoryInitRequest   | 10    | 20,000  | StoryInitRequest | API request validation             |
| QueuedInput.raw    | 1     | 20,000  | QueuedInput      | Queued pre-story msgs for Ch1      |
| DialogueEntry.msg  | 1     | 20,000  | DialogueEntry    | Chat messages during story         |
| UserQuestion       | 1     | 1,000   | UserQuestion     | Questions to companion             |
| VoiceCommand       | 1     | 1,000   | VoiceCommand     | Voice inputs                       |
| Child/User names   | 1-2   | 100     | Various          | Profile names                      |
| Story/Chapter title| 3     | 100     | Various          | Auto-generated titles              |
| Age                | 1-4   | 120     | Various          | User age in years                  |
"""

from pydantic import BaseModel, Field, validator, root_validator, field_serializer, constr
from typing import List, Optional, Dict, Literal, Union, Any
from datetime import datetime, timezone
from enum import Enum
import uuid
import re


# ============================================================================
# LLM Response Normalizer - Handles inconsistent LLM output formats
# ============================================================================

def normalize_llm_evolution_item(value: Any, item_type: str = "evolution") -> Dict[str, Any]:
    """
    Convert LLM output to proper dict format.

    LLMs often return strings instead of dicts for nested items:
    - "Chapter 4: Harald becomes brave" (string)
    - {"chapter_number": 4, "change": "Harald becomes brave"} (expected dict)

    This function normalizes both to the expected dict format.

    Args:
        value: The value from LLM (could be str, dict, or other)
        item_type: "evolution" for personality_evolution, "relationship" for relationship_changes

    Returns:
        Normalized dict with standard keys
    """
    # Already a dict - return as-is
    if isinstance(value, dict):
        return value

    # Handle string format: "Chapter X: description"
    if isinstance(value, str):
        # Try to extract chapter number and description
        match = re.match(r'(?:Chapter\s*)?(\d+)[:\s]*(.+)', value, re.IGNORECASE)

        if match:
            chapter_num = int(match.group(1))
            description = match.group(2).strip()
        else:
            # No chapter number found, default to chapter 1
            chapter_num = 1
            description = value.strip()

        if item_type == "evolution":
            return {
                'chapter_number': chapter_num,
                'change': description,
                'from_trait': 'initial',
                'to_trait': description[:100] if len(description) > 100 else description,
                'trigger_event': description
            }
        elif item_type == "relationship":
            return {
                'chapter_number': chapter_num,
                'other_character': 'others',
                'relationship_type': 'evolving',
                'strength': 5,
                'description': description
            }

    # Fallback for other types
    return {
        'chapter_number': 1,
        'description': str(value) if value else 'Unknown change'
    }


# ============================================================================
# Enums
# ============================================================================

class StoryStatus(str, Enum):
    INITIALIZING = "initializing"
    IN_DIALOGUE = "in_dialogue"
    STRUCTURE_READY = "structure_ready"
    GENERATING_CHAPTER = "generating_chapter"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


class TaskType(str, Enum):
    DIALOGUE_RESPONSE = "dialogue_response"
    STRUCTURE_GENERATION = "structure_generation"
    CHARACTER_CREATION = "character_creation"
    NARRATIVE_GENERATION = "narrative_generation"
    FACT_CHECK = "fact_check"
    STORY_REVISION = "story_revision"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    CRITICAL = "critical"  # Dialogue - respond immediately
    HIGH = "high"  # Structure - story foundation
    MEDIUM = "medium"  # Characters, narrative
    LOW = "low"  # Fact-checking, refinement


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ChapterStatus(str, Enum):
    """Status of an individual chapter in the hybrid generation system"""
    PENDING = "pending"           # Planned but not started
    QUEUED = "queued"             # In generation queue
    GENERATING = "generating"     # Being written by NarrativeAgent
    DRAFT = "draft"              # Written, awaiting fact-check
    FACT_CHECKING = "fact_checking"  # Being reviewed by FactCheckAgent
    REVISION = "revision"         # Being revised based on fact-check
    READY = "ready"              # Fact-checked, first audio segment ready
    READING = "reading"          # Currently being read to user
    COMPLETED = "completed"      # User finished reading


class InputTier(int, Enum):
    """Classification tier for user input during story reading"""
    TIER_1_IMMEDIATE = 1    # No story impact - acknowledgment/vocab ("That's cool!", "What's a fjord?")
    TIER_2_PREFERENCE = 2   # Minor adjustment next chapter ("Make it scarier", "I like the brave part")
    TIER_3_STORY_CHOICE = 3 # Plot fork - affects N+2 chapter ("Add a wolf!", "Make her save him")
    TIER_4_ADDITION = 4     # New subplot element ("Can there be pirates too?")


class PlaybackPhase(str, Enum):
    """Phase of chapter playback for state management.

    Used to track whether the user can interact, is listening to audio,
    or is in post-chapter discussion with the CompanionAgent.
    """
    PRE_CHAPTER = "pre_chapter"       # Before playback - can ask questions, hear teasers
    PLAYING_CHAPTER = "playing"       # Audio playing - messages queued, no interruption
    POST_CHAPTER = "post_chapter"     # After playback - discussion phase with teacher
    TRANSITIONING = "transitioning"   # Moving to next chapter


# ============================================================================
# Story Models
# ============================================================================

class StoryPreferences(BaseModel):
    """User preferences for story generation"""
    language: str = Field(default="en", pattern="^(en|es|no)$")
    educational_focus: Optional[str] = Field(
        None,
        description="History, science, language, math, culture"
    )
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    themes: List[str] = Field(default_factory=list)
    scary_level: str = Field(default="mild", description="How scary/intense (mild, medium, exciting, etc.)")
    target_age: Optional[int] = Field(
        default=None,
        ge=3, le=120,
        description="Target age for content. Derived from user profile if not set."
    )
    user_style_requests: List[str] = Field(
        default_factory=list,
        description="User's style preferences collected during conversation (e.g., 'make it funny', 'more exciting', 'less scary'). Applied to ALL chapters."
    )

    @validator('themes')
    def validate_themes(cls, v):
        # Limit to 5 themes
        if len(v) > 5:
            raise ValueError("Maximum 5 themes allowed")
        return v


class EducationalGoal(BaseModel):
    """Educational objective for the story"""
    concept: str = Field(..., description="What to teach")
    description: str = Field(..., description="How it's taught")
    age_appropriate: bool = Field(default=True)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flexible key:value storage for LLM-generated extras (integration_approach, examples, etc.)"
    )

    @root_validator(pre=True)
    def capture_extra_fields_to_metadata(cls, values):
        """Capture any extra fields LLMs add (like integration_approach) into metadata."""
        if not isinstance(values, dict):
            return values

        known_fields = {'concept', 'description', 'age_appropriate', 'metadata'}
        extra_fields = {k: v for k, v in values.items() if k not in known_fields}

        if extra_fields:
            existing_metadata = values.get('metadata') or {}
            values['metadata'] = {**existing_metadata, **extra_fields}
            for key in extra_fields:
                del values[key]

        return values


class VerifiedFact(BaseModel):
    """Historical or scientific fact with verification"""
    fact: str = Field(..., description="The fact statement (also accepts 'text' as alias)")
    verified: bool = Field(default=False)
    source: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @root_validator(pre=True)
    def accept_text_as_fact(cls, values):
        """Accept 'text' as an alias for 'fact' field.

        Also handles string input from LLMs returning just the fact text.
        """
        # Handle string input: LLM returns "Vikings discovered America" instead of {"fact": "..."}
        if isinstance(values, str):
            return {'fact': values.strip(), 'verified': False, 'confidence': 0.5}

        # Handle non-dict types
        if not isinstance(values, dict):
            return {'fact': str(values) if values else 'Unknown fact', 'verified': False, 'confidence': 0.0}

        # Dict input: check for 'text' alias
        if 'text' in values and 'fact' not in values:
            values['fact'] = values.pop('text')
        return values


class Statement(BaseModel):
    """A factual claim extracted from narrative for tracking and verification"""
    id: str = Field(..., description="Unique statement ID (e.g., 'stmt_1', 'stmt_2')")
    text: str = Field(..., min_length=3, max_length=500, description="The actual factual claim")
    paragraph_number: int = Field(..., ge=1, description="Which paragraph this statement appears in")
    sentence_number: int = Field(default=0, ge=0, description="Position within paragraph (0 if unknown)")


class FactCheckIssue(BaseModel):
    """Issue found during fact-checking process"""
    fact_claimed: str = Field(..., description="What the narrative stated")
    statement_id: Optional[str] = Field(None, description="ID of the tagged statement if applicable (e.g., 'stmt_3')")
    paragraph_number: Optional[int] = Field(None, ge=1, description="Which paragraph contains this issue")
    issue_type: str = Field(..., description="Type of factual issue (incorrect, uncertain, impossible, inconsistent, etc.)")
    explanation: str = Field(..., description="Why this is problematic")
    correction: Optional[str] = Field(None, description="Suggested correction")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in this assessment")


class FactCheckReport(BaseModel):
    """Report from FactCheckAgent reviewing a chapter"""
    chapter_number: int = Field(..., ge=1)
    issues_found: List[FactCheckIssue] = Field(default_factory=list)
    approval_status: Literal["approved", "needs_revision", "major_issues"] = Field(...)
    overall_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    revision_round: int = Field(default=1, ge=1, description="Which revision attempt this is")


# ============================================================================
# Round Table Review Models
# ============================================================================

class AgentReview(BaseModel):
    """Single agent's review in the Round Table discussion"""
    agent: str = Field(..., description="Agent persona name: 'Guillermo', 'Bill', 'Clarissa', 'Benjamin', 'Continuity', or 'Stephen'")
    domain: str = Field(..., description="Review domain: 'structure', 'facts', 'characters', 'prose', 'continuity', or 'tension'")
    verdict: Literal["approve", "concern", "block"] = Field(
        ...,
        description="approve=good to go, concern=minor issues, block=needs revision"
    )
    praise: str = Field(default="", description="What works beautifully in this chapter")
    concern: str = Field(default="", description="What troubles the reviewer")
    suggestion: str = Field(default="", description="Specific suggestion for improvement")
    # Stephen (tension agent) specific fields
    chapter_ending_score: Optional[str] = Field(
        default=None,
        description="Stephen's assessment of chapter ending: hook (pulls forward), adequate (OK), allows_stop (reader can put book down)"
    )
    tension_arc: Optional[str] = Field(
        default=None,
        description="Stephen's assessment of tension throughout chapter"
    )


class RoundTableReview(BaseModel):
    """
    Collective Round Table review result.

    After Nnedi writes a draft, all agents gather around the table:
    - Guillermo (Structure): Pacing, themes, visual coherence
    - Bill (Facts): Historical/scientific accuracy
    - Clarissa (Characters): Psychology, arc consistency, voice distinctiveness
    - Benjamin (Prose): Line editing, read-aloud quality, humor/tonal variation
    - Continuity: Plot thread tracking, setup/payoff consistency
    - Stephen (Tension): Page-turning momentum, chapter endings, cliffhangers

    If ANY agent blocks → Discussion phase → Revision (max 3 rounds)
    """
    decision: Literal["approved", "approved_with_notes", "revise"] = Field(
        ...,
        description="Final collective decision"
    )
    reviews: List[AgentReview] = Field(default_factory=list, description="Individual agent reviews")
    discussion: Optional[str] = Field(None, description="Nnedi's response to concerns if revision needed")
    revision_guidance: Optional[str] = Field(None, description="Compiled guidance for Nnedi's revision")
    collective_notes: List[str] = Field(default_factory=list, description="Notes from approved_with_notes decisions")
    revision_rounds: int = Field(default=0, ge=0, description="How many revision rounds occurred")


class VocabularyWord(BaseModel):
    """Vocabulary word for educational tracking"""
    word: str
    definition: str
    age_appropriate_level: int = Field(..., ge=1, description="Age level for content appropriateness")
    context_in_story: Optional[str] = None


class ChapterOutline(BaseModel):
    """Outline for a single chapter"""
    number: int = Field(..., ge=1)  # No upper limit on chapter count
    title: str = Field(..., min_length=3, max_length=100)
    synopsis: str = Field(..., min_length=1, max_length=10000, description="Chapter synopsis - can be brief or detailed")
    characters_featured: List[str] = Field(default_factory=list)
    educational_points: List[str] = Field(default_factory=list)
    facts_to_verify: List[str] = Field(default_factory=list)
    character_development_milestones: Dict[str, str] = Field(default_factory=dict, description="Character development for this chapter (e.g., {'Harald': 'learns the cost of war'})")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flexible key:value storage for LLM-generated extras (visual_thematic_notes, pacing_notes, etc.)"
    )

    @root_validator(pre=True)
    def capture_extra_fields_to_metadata(cls, values):
        """Capture any extra fields LLMs add (like visual_thematic_notes) into metadata."""
        if not isinstance(values, dict):
            return values

        known_fields = {
            'number', 'title', 'synopsis', 'characters_featured', 'educational_points',
            'facts_to_verify', 'character_development_milestones', 'metadata'
        }
        extra_fields = {k: v for k, v in values.items() if k not in known_fields}

        if extra_fields:
            existing_metadata = values.get('metadata') or {}
            values['metadata'] = {**existing_metadata, **extra_fields}
            for key in extra_fields:
                del values[key]

        return values


class CharacterNeeded(BaseModel):
    """Character specification from StructureAgent with arc planning"""
    name: str = Field(..., min_length=2)  # No max - allow elaborate titles like "Harald Fairhair, Son of Halfdan"
    role: str = Field(..., description="Character role (protagonist, antagonist, mentor, etc.)")
    importance: str = Field(default="supporting", description="Character importance level (major, supporting, minor, thematic, cameo, etc.)")
    importance_qualifier: Optional[str] = Field(
        default=None,
        description="Qualifier for importance level (e.g., 'thematic', 'historical', 'deceased'). Extracted from 'major (thematic)' style inputs."
    )
    arc_milestones: Optional[Dict[str, str]] = Field(None, description="Character development milestones by chapter number")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flexible key:value storage for LLM-generated character details (description, species, age, visual_notes, etc.)"
    )

    @root_validator(pre=True)
    def extract_importance_qualifier(cls, values):
        """Extract qualifiers like 'thematic' from 'major (thematic)' before validation.

        This preserves the semantic meaning the LLM intended:
        - 'major (thematic)' → importance='major', importance_qualifier='thematic'
        - 'minor (historical)' → importance='minor', importance_qualifier='historical'
        """
        if not isinstance(values, dict):
            return values

        importance = values.get('importance', '')
        if isinstance(importance, str) and '(' in importance:
            # Extract the qualifier from parentheses
            import re
            match = re.match(r'(\w+)\s*\(([^)]+)\)', importance)
            if match:
                base_importance = match.group(1).strip()
                qualifier = match.group(2).strip()
                values['importance'] = base_importance
                # Only set if not already set
                if not values.get('importance_qualifier'):
                    values['importance_qualifier'] = qualifier

        return values

    @validator('importance', pre=True)
    def normalize_importance(cls, v):
        """Normalize importance to lowercase. Accept any value - this is a creative system."""
        if isinstance(v, str):
            return v.lower().strip()
        return v or "supporting"

    @validator('arc_milestones', pre=True)
    def convert_arc_milestones(cls, v):
        """Convert array format to dict format if needed.

        Also handles string input from LLMs like "Chapter 3: becomes confident".
        """
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, list):
            # Convert array [None, "text", None, "text2", ...] to dict {"1": "text", "3": "text2", ...}
            result = {}
            for i, value in enumerate(v, 1):
                if value is not None and value != "":
                    result[str(i)] = str(value)
            return result if result else None
        if isinstance(v, str):
            # Handle string format: "Chapter 3: becomes confident" or just "becomes confident"
            match = re.match(r'(?:Chapter\s*)?(\d+)[:\s]*(.+)', v, re.IGNORECASE)
            if match:
                chapter_num = match.group(1)
                milestone = match.group(2).strip()
                return {chapter_num: milestone}
            else:
                # No chapter number found, assume chapter 1
                return {'1': v.strip()}
        return v

    @root_validator(pre=True)
    def capture_extra_fields_to_metadata(cls, values):
        """Capture any extra fields LLMs add and store them in metadata.

        LLMs often add useful fields like 'description', 'species', 'age',
        'visual_notes' etc. Instead of losing them, we preserve them in metadata.
        """
        if not isinstance(values, dict):
            return values

        # Known fields that are part of the schema
        known_fields = {'name', 'role', 'importance', 'importance_qualifier', 'arc_milestones', 'metadata'}

        # Find extra fields
        extra_fields = {k: v for k, v in values.items() if k not in known_fields}

        if extra_fields:
            # Merge into existing metadata or create new
            existing_metadata = values.get('metadata') or {}
            values['metadata'] = {**existing_metadata, **extra_fields}

            # Remove extra fields from top level (they're now in metadata)
            for key in extra_fields:
                del values[key]

        return values


class PlotElementType(str, Enum):
    """Types of plot elements that can be tracked for continuity."""
    MYSTERY = "mystery"      # Unanswered question
    OBJECT = "object"        # Physical item introduced
    CONFLICT = "conflict"    # Tension to resolve
    PROMISE = "promise"      # Narrative promise (foreshadowing)
    SECRET = "secret"        # Hidden information
    RELATIONSHIP = "relationship"  # Character relationship development


class PlotElement(BaseModel):
    """
    Tracks narrative threads that need resolution (setup → payoff).

    Examples:
    - A mysterious letter is discovered (setup) → Contents revealed (payoff)
    - A promise is made (setup) → Promise is kept/broken (payoff)
    - A secret is hinted at (setup) → Secret is unveiled (payoff)

    This enables the ContinuityAgent to track plot threads across chapters
    and flag unresolved setups before story completion.
    """
    id: str = Field(default_factory=lambda: f"plot_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., min_length=3, max_length=100, description="Brief name for the plot element (e.g., 'The ancient letter')")
    element_type: Union[PlotElementType, str] = Field(..., description="Type: mystery, object, conflict, promise, relationship, secret")
    introduced_chapter: int = Field(..., ge=1, description="Chapter where this element was introduced")
    setup_text: str = Field(..., min_length=1, description="The setup text that introduces this element")
    resolution_chapter: Optional[int] = Field(None, ge=1, description="Chapter where this element was resolved")
    resolution_text: Optional[str] = Field(None, description="How the element was resolved")
    status: str = Field(default="pending", description="Status: pending | resolved | abandoned")
    importance: str = Field(default="major", description="Importance: major | minor - major must be resolved")


class StructureRefinementV2(BaseModel):
    """Metadata tracking Structure V2 refinement pass.

    Structure V2 runs after Chapter 1 playback starts to refine synopses
    for chapters 2-N using actual Chapter 1 content, D&D character cards,
    and user preferences from the dialogue phase.
    """
    refined_at: datetime = Field(default_factory=datetime.now)
    chapters_refined: int = Field(..., ge=0, description="Number of chapters whose synopses were refined")
    user_inputs_incorporated: int = Field(default=0, ge=0, description="Count of user preferences woven into refined synopses")
    skills_leveraged: List[str] = Field(default_factory=list, description="Character skills specifically leveraged in plot points")
    notes: str = Field(default="", description="Brief explanation of major refinement changes")


class NarrativeMethod(BaseModel):
    """
    Narrative storytelling method chosen by Guillermo + Stephen debate.

    Determines HOW the story will be told - single POV (Harry Potter style)
    vs multi-POV with parallel action (Da Vinci Code style).

    This is decided BEFORE chapter writing begins to ensure consistent approach.
    """
    method: str = Field(
        default="linear_single_pov",
        description="Storytelling method (linear_single_pov, linear_dual_thread, multi_pov_alternating, frame_narrative, or LLM-invented methods)"
    )
    pov_characters: List[str] = Field(
        default_factory=list,
        description="Which character(s) tell the story - for single_pov, usually just protagonist"
    )
    hook_strategy: str = Field(
        default="Each chapter ends with unanswered question or discovery",
        description="Strategy for chapter endings: cliffhangers, mystery breadcrumbs, emotional hooks, etc."
    )
    chapter_rhythm: str = Field(
        default="alternating action/quiet",
        description="Pacing rhythm across chapters: alternating action/quiet, escalating tension, etc."
    )
    rationale: str = Field(
        default="",
        description="Why this method suits this particular story and target audience"
    )


class StoryStructure(BaseModel):
    """Complete story structure from StructureAgent"""
    title: str = Field(..., min_length=3, max_length=100)
    theme: str
    chapters: List[ChapterOutline] = Field(..., min_items=1, max_items=100)  # Allow epic sagas
    characters_needed: List[Union[CharacterNeeded, str]] = Field(..., min_items=1, description="Characters needed (can be CharacterNeeded objects or strings)")
    educational_goals: List[EducationalGoal]
    estimated_reading_time_minutes: int = Field(..., ge=1)  # No upper limit - epic stories welcome
    plot_elements: List[PlotElement] = Field(default_factory=list, description="Plot threads to track for continuity (setup → payoff)")
    refinement_v2: Optional[StructureRefinementV2] = Field(default=None, description="Metadata from Structure V2 refinement pass (if applied)")
    narrative_method: Optional[NarrativeMethod] = Field(default=None, description="Storytelling method chosen by Guillermo + Stephen debate")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flexible key:value storage for LLM-generated story extensions (series_potential, themes_and_visual_leitmotifs, world_building_notes, etc.)"
    )

    @validator('chapters')
    def validate_chapters(cls, v):
        if len(v) < 1:
            raise ValueError("Story must have at least 1 chapter")
        # Ensure sequential numbering
        for i, chapter in enumerate(v, 1):
            if chapter.number != i:
                raise ValueError(f"Chapter numbers must be sequential, expected {i}, got {chapter.number}")
        return v

    @root_validator(pre=True)
    def capture_extra_fields_to_metadata(cls, values):
        """Capture any extra fields LLMs add and store them in metadata.

        LLMs often add creative fields like 'series_potential', 'themes_and_visual_leitmotifs',
        'world_building_notes', 'color_symbolism' etc. Instead of losing them, we preserve them.
        """
        if not isinstance(values, dict):
            return values

        # Known fields that are part of the schema
        known_fields = {
            'title', 'theme', 'chapters', 'characters_needed', 'educational_goals',
            'estimated_reading_time_minutes', 'plot_elements', 'refinement_v2',
            'narrative_method', 'metadata'
        }

        # Find extra fields
        extra_fields = {k: v for k, v in values.items() if k not in known_fields}

        if extra_fields:
            # Merge into existing metadata or create new
            existing_metadata = values.get('metadata') or {}
            values['metadata'] = {**existing_metadata, **extra_fields}

            # Remove extra fields from top level (they're now in metadata)
            for key in extra_fields:
                del values[key]

        return values


class CharacterSpec(BaseModel):
    """Character specification (minimal info for creation)"""
    name: str = Field(..., min_length=2)  # No max - fictional characters can have long names
    role: str = Field(default="supporting", description="Character role (protagonist, antagonist, sidekick, mentor, supporting, or custom roles)")
    age_range: Optional[str] = None
    key_trait: Optional[str] = None


class CharacterSkill(BaseModel):
    """Skill that a character can learn or improve (D&D-style)"""
    name: str = Field(..., min_length=2, max_length=50, description="Skill name (e.g., 'Battle Strategy', 'Diplomacy')")
    level: int = Field(default=1, ge=1, le=10, description="Skill level from 1 (novice) to 10 (master)")
    acquired_chapter: int = Field(default=0, ge=0, description="Chapter number when skill was learned (0 = pre-story/backstory)")
    description: str = Field(..., min_length=1)  # No max - skills can have rich descriptions


class PersonalityEvolution(BaseModel):
    """Track how a personality trait evolved through the story.

    Accepts multiple input formats from LLM agents:
    - Standard: {chapter_number, from_trait, to_trait, trigger_event}
    - Agent format: {chapter, change, trigger} or {chapter, changes: [...]}
    - Simple format: {chapter, state} - just describes state at that chapter
    """
    chapter_number: int = Field(..., ge=1)
    from_trait: str = Field(default="", description="Original trait (e.g., 'impulsive')")
    to_trait: str = Field(default="", description="New trait (e.g., 'thoughtful')")
    trigger_event: str = Field(default="", description="What caused the change")
    change: Optional[str] = Field(None, description="Alternative: combined change description")
    state: Optional[str] = Field(None, description="Alternative: character state at this chapter")

    @root_validator(pre=True)
    def normalize_personality_evolution(cls, values):
        """Accept alternate formats from LLM agents."""
        # Handle string input first (LLMs often return strings instead of dicts)
        if isinstance(values, str):
            return normalize_llm_evolution_item(values, item_type="evolution")

        # Handle non-dict types
        if not isinstance(values, dict):
            return normalize_llm_evolution_item(values, item_type="evolution")

        # Now values is guaranteed to be a dict
        # Convert 'chapter' to 'chapter_number'
        if 'chapter' in values and 'chapter_number' not in values:
            chapter_val = values.pop('chapter')
            # Handle string chapters like "baseline", "chapter_3", "end"
            if isinstance(chapter_val, str):
                chapter_match = re.search(r'(\d+)', chapter_val)
                if chapter_match:
                    values['chapter_number'] = int(chapter_match.group(1))
                elif chapter_val.lower() in ('baseline', 'start', 'beginning', 'initial'):
                    values['chapter_number'] = 1
                elif chapter_val.lower() in ('end', 'final', 'conclusion'):
                    values['chapter_number'] = 99  # High number for end state
                else:
                    values['chapter_number'] = 1  # Default to chapter 1
            else:
                values['chapter_number'] = int(chapter_val) if chapter_val else 1

        # Handle 'changes' array format: {"chapter": 1, "changes": ["...", "..."]}
        if 'changes' in values:
            changes = values.pop('changes')
            if isinstance(changes, list) and changes:
                # Combine array into single change description
                values['change'] = ' | '.join(str(c) for c in changes)

        # Handle 'state' format: {"chapter": 1, "state": "Harald is impulsive..."}
        if 'state' in values and values.get('state'):
            state_text = values['state']
            if not values.get('change'):
                values['change'] = state_text
            if not values.get('to_trait'):
                # Extract a short trait from the state description
                values['to_trait'] = state_text[:100] if len(state_text) > 100 else state_text

        # If we have 'change' but not from/to traits, use change as both
        if 'change' in values and values.get('change'):
            change_text = values['change']
            if not values.get('from_trait'):
                values['from_trait'] = "initial"
            if not values.get('to_trait'):
                values['to_trait'] = change_text[:100] if len(change_text) > 100 else change_text
            if not values.get('trigger_event'):
                values['trigger_event'] = change_text

        # Ensure defaults for missing fields
        # CRITICAL: chapter_number is required - provide default if missing
        if 'chapter_number' not in values or values.get('chapter_number') is None:
            values['chapter_number'] = 1  # Default to chapter 1

        if not values.get('from_trait'):
            values['from_trait'] = "unknown"
        if not values.get('to_trait'):
            values['to_trait'] = "evolved"
        if not values.get('trigger_event'):
            values['trigger_event'] = "story events"

        return values


class RelationshipChange(BaseModel):
    """Track relationship evolution between characters.

    Accepts multiple input formats from LLM agents:
    - Standard: {chapter_number, other_character, relationship_type, strength, description}
    - Agent format: {chapter, character, change} or {chapter, character, change, new_dynamic}
    - Simple format: {with, evolution} - just names the character and describes evolution
    """
    chapter_number: int = Field(default=1, ge=1, description="Chapter where relationship is introduced/changes")
    other_character: str = Field(default="others", description="Name of the other character")
    relationship_type: str = Field(default="evolving", description="Type of relationship (e.g., 'rival', 'ally', 'mentor')")
    strength: int = Field(default=5, ge=1, le=10, description="Relationship strength/depth (1-10)")
    description: str = Field(default="Relationship evolving", description="Description of the change")
    new_dynamic: Optional[str] = Field(None, description="Alternative: new relationship dynamic")

    @root_validator(pre=True)
    def normalize_relationship_change(cls, values):
        """Accept alternate formats from LLM agents."""
        # Handle string input first (LLMs often return strings instead of dicts)
        if isinstance(values, str):
            return normalize_llm_evolution_item(values, item_type="relationship")

        # Handle non-dict types
        if not isinstance(values, dict):
            return normalize_llm_evolution_item(values, item_type="relationship")

        # Now values is guaranteed to be a dict
        # Convert 'chapter' to 'chapter_number'
        if 'chapter' in values and 'chapter_number' not in values:
            chapter_val = values.pop('chapter')
            if isinstance(chapter_val, str):
                chapter_match = re.search(r'(\d+)', chapter_val)
                values['chapter_number'] = int(chapter_match.group(1)) if chapter_match else 1
            else:
                values['chapter_number'] = int(chapter_val) if chapter_val else 1

        # Convert 'character' to 'other_character'
        if 'character' in values and 'other_character' not in values:
            values['other_character'] = values.pop('character')

        # Convert 'with' to 'other_character' (LLM format: {"with": "King Halfdan", ...})
        if 'with' in values and 'other_character' not in values:
            values['other_character'] = values.pop('with')

        # Handle complex format: {"with": "Harald", "before": "...", "chapter_3_change": "...", "chapter_6_change": "..."}
        # Extract all chapter_X_change fields and combine them
        chapter_changes = []
        keys_to_remove = []
        for key in list(values.keys()):
            if '_change' in key:
                match = re.search(r'chapter_?(\d+)_change', key)
                if match:
                    chapter_num = int(match.group(1))
                    change_text = values[key]
                    chapter_changes.append(f"Ch{chapter_num}: {change_text}")
                    keys_to_remove.append(key)

        # Remove processed keys
        for key in keys_to_remove:
            values.pop(key, None)

        # Handle 'before' field - add to description
        before_text = values.pop('before', None)
        after_text = values.pop('after', None)

        # Build description from all available sources
        desc_parts = []
        if before_text:
            desc_parts.append(f"Before: {before_text}")
        if after_text:
            desc_parts.append(f"After: {after_text}")
        if chapter_changes:
            desc_parts.extend(chapter_changes)

        if desc_parts and not values.get('description'):
            values['description'] = ' | '.join(desc_parts)

        # Handle 'evolution' field as description (LLM format: {"evolution": "From tense to mutual respect"})
        if 'evolution' in values and not values.get('description'):
            evolution_text = values.pop('evolution')
            values['description'] = evolution_text if evolution_text else "Relationship evolving"

        # Handle 'change' field as description AND try to extract other_character from it
        if 'change' in values:
            change_text = values.pop('change')
            if change_text and not values.get('description'):
                values['description'] = change_text

            # Try to extract character name from change text if other_character not set
            if change_text and not values.get('other_character'):
                # Common patterns: "Father-son tensions", "Skald Eirik's stories", "Rivalry with Hakon"
                # Note: 're' is already imported at module level
                # Look for possessive patterns: "Name's" or "with Name"
                patterns = [
                    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s",  # "Skald Eirik's"
                    r"with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "with Hakon"
                    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:tensions|bond|relationship)",  # "Father-son tensions"
                ]
                for pattern in patterns:
                    match = re.search(pattern, change_text)
                    if match:
                        values['other_character'] = match.group(1)
                        break

        # Handle 'new_dynamic' by appending to description
        if 'new_dynamic' in values and values.get('new_dynamic'):
            new_dynamic = values.get('new_dynamic')
            if values.get('description'):
                values['description'] = f"{values['description']} → {new_dynamic}"
            else:
                values['description'] = new_dynamic

        # Infer relationship_type from description if missing
        if not values.get('relationship_type') or values.get('relationship_type') == "evolving":
            desc = values.get('description', '').lower()
            other_char = values.get('other_character', '').lower()
            combined = f"{desc} {other_char}"
            if 'rival' in combined:
                values['relationship_type'] = 'rival'
            elif 'ally' in combined or 'alliance' in combined:
                values['relationship_type'] = 'ally'
            elif 'mentor' in combined or 'seer' in combined or 'teacher' in combined:
                values['relationship_type'] = 'mentor'
            elif 'friend' in combined or 'counselor' in combined:
                values['relationship_type'] = 'friend'
            elif 'father' in combined or 'mother' in combined or 'parent' in combined:
                values['relationship_type'] = 'family'
            elif 'respect' in combined:
                values['relationship_type'] = 'respectful'
            else:
                values['relationship_type'] = 'evolving'

        # Ensure description has minimum content
        if not values.get('description') or len(values.get('description', '')) < 10:
            values['description'] = f"Relationship with {values.get('other_character', 'unknown')} is changing"

        return values


class CharacterProgression(BaseModel):
    """Complete progression history for a character throughout the story"""
    skills_learned: List[CharacterSkill] = Field(default_factory=list, description="Skills acquired through the story")
    personality_evolution: List[PersonalityEvolution] = Field(default_factory=list, description="How personality changed")
    relationship_changes: List[RelationshipChange] = Field(default_factory=list, description="Relationship developments")
    current_emotional_state: str = Field(default="neutral", description="Current emotional state")
    chapters_featured: List[int] = Field(default_factory=list, description="Chapters where character appeared")

    @validator('chapters_featured', pre=True)
    def normalize_chapters_featured(cls, v):
        """Accept chapter numbers as integers, strings, or objects with 'chapter' key."""
        if not v:
            return []

        result = []
        for item in v:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, dict):
                # Handle object format: {"chapter": 1, "focus": "..."}
                chapter_num = item.get('chapter') or item.get('chapter_number') or item.get('number')
                if chapter_num is not None:
                    if isinstance(chapter_num, int):
                        result.append(chapter_num)
                    elif isinstance(chapter_num, str):
                        import re
                        match = re.search(r'(\d+)', str(chapter_num))
                        if match:
                            result.append(int(match.group(1)))
            elif isinstance(item, str):
                import re
                # Match various patterns: "Chapter 1", "chapter_3", "Ch. 2", just "3"
                match = re.search(r'[Cc]h(?:apter)?[_\s\.]*(\d+)', item)
                if match:
                    result.append(int(match.group(1)))
                else:
                    # Try to find any number in the string
                    match = re.search(r'(\d+)', item)
                    if match:
                        result.append(int(match.group(1)))
                    else:
                        try:
                            result.append(int(item.strip()))
                        except ValueError:
                            pass
        return result


class Character(BaseModel):
    """Fully developed character profile with progression tracking"""
    id: str = Field(default_factory=lambda: f"char_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., min_length=2)  # No max - allow elaborate titles like "Gandalf the Grey, Mithrandir, Servant of the Secret Fire"
    name_history: List[str] = Field(
        default_factory=list,
        description="Previous names/epithets this character has had (tracks evolution like 'Jarl Eirik Ironbrow' → 'Jarl Eirik Bloodaxe')"
    )
    role: str
    age: Optional[Union[int, str]] = Field(None, description="Character age (integer) or description ('Ageless', 'Ancient')")
    background: str = Field(..., min_length=1)  # No max - backstories can be as rich as needed
    personality_traits: List[str] = Field(default_factory=list)  # Characters can have any number of traits
    motivation: str = Field(..., min_length=1)  # No max - deep psychological motivations need space
    appearance: Optional[str] = None
    relationships: Dict[str, str] = Field(default_factory=dict)  # character_id -> relationship

    # NEW: Progression system
    progression: CharacterProgression = Field(default_factory=CharacterProgression, description="Character development tracker")
    character_arc: Optional[Dict[str, str]] = Field(None, description="Planned arc milestones by chapter (e.g., {'1': 'reckless', '5': 'wise'})")

    @root_validator(pre=True)
    def normalize_character_fields(cls, values):
        """Accept flat structure from LLM and move progression fields into nested object.

        Also handles string input from LLMs returning just a character name or description.
        """
        # Handle string input: LLM returns "Harald Fairhair" instead of {"name": "Harald Fairhair", ...}
        if isinstance(values, str):
            # Extract name from string, provide minimal defaults for required fields
            name = values.strip()
            return {
                'name': name,
                'role': 'supporting',
                'background': f'{name} is a character in this story.',
                'personality_traits': ['determined', 'adaptable'],
                'motivation': f'{name} seeks to fulfill their role in the story.',
                'progression': {}
            }

        # Handle non-dict types
        if not isinstance(values, dict):
            return {
                'name': str(values) if values else 'Unknown Character',
                'role': 'supporting',
                'background': 'A character in this story.',
                'personality_traits': ['determined', 'adaptable'],
                'motivation': 'To play their part in the story.',
                'progression': {}
            }

        # LLM may return progression fields at top level - move them into 'progression' dict
        progression_fields = {
            'skills': 'skills_learned',  # LLM uses 'skills', model expects 'skills_learned'
            'skills_learned': 'skills_learned',
            'personality_evolution': 'personality_evolution',
            'relationship_changes': 'relationship_changes',
            'current_emotional_state': 'current_emotional_state',
            'chapters_featured': 'chapters_featured',
        }

        # Initialize progression dict if not present
        if 'progression' not in values or values['progression'] is None:
            values['progression'] = {}

        # If progression is already a CharacterProgression object, convert to dict
        if hasattr(values['progression'], 'dict'):
            values['progression'] = values['progression'].dict()

        # Move fields from top level to progression
        for llm_field, model_field in progression_fields.items():
            if llm_field in values and llm_field != 'progression':
                field_value = values.pop(llm_field)
                if field_value is not None:
                    values['progression'][model_field] = field_value

        # Handle 'age_appearance' as an alias for 'age'
        if 'age_appearance' in values and 'age' not in values:
            values['age'] = values.pop('age_appearance')

        # Handle 'physical_traits' - merge into appearance if present
        if 'physical_traits' in values:
            physical_traits = values.pop('physical_traits')
            if physical_traits and isinstance(physical_traits, list):
                traits_str = ', '.join(physical_traits)
                if values.get('appearance'):
                    values['appearance'] = f"{values['appearance']}. {traits_str}"
                else:
                    values['appearance'] = traits_str

        return values

    @validator('personality_traits')
    def validate_traits(cls, v):
        if len(v) < 2:
            raise ValueError("Character must have at least 2 personality traits")
        return v


# ============================================================================
# Voice Direction Models (for TTS/Audiobook Production)
# ============================================================================

class CharacterVoiceMapping(BaseModel):
    """Voice characteristics for a character in TTS narration"""
    character_name: str = Field(..., description="Name of the character")
    pitch_adjustment: str = Field(default="+0%", description="Pitch adjustment, e.g., '+10%' for higher, '-5%' for lower")
    rate_adjustment: str = Field(default="100%", description="Speaking rate, e.g., '90%' for slower, '110%' for faster")
    voice_quality: str = Field(default="neutral", description="Voice quality descriptor, e.g., 'gruff', 'melodic', 'childlike'")
    sample_dialogue: Optional[str] = Field(None, description="Example dialogue showing their voice style")


class EmotionalBeat(BaseModel):
    """Emotional moment in narration requiring special TTS treatment"""
    paragraph_index: int = Field(..., ge=0, description="Paragraph number (0-indexed)")
    emotion: str = Field(..., description="Emotional quality, e.g., 'tension', 'joy', 'revelation', 'sorrow'")
    pacing: str = Field(default="normal", description="Pacing adjustment: 'slow', 'normal', 'fast'")
    pause_after_ms: int = Field(default=0, ge=0, le=3000, description="Pause duration after this beat in milliseconds")


class VoiceDirectionMetadata(BaseModel):
    """Metadata from VoiceDirectorAgent's analysis of a chapter"""
    character_voice_mappings: List[CharacterVoiceMapping] = Field(default_factory=list)
    emotional_beats: List[EmotionalBeat] = Field(default_factory=list)
    target_age: int = Field(default=10, ge=4, le=120)
    total_estimated_duration_seconds: int = Field(default=0, ge=0)
    processed_at: datetime = Field(default_factory=datetime.now)
    processor_version: str = Field(default="1.0")

    @field_serializer('processed_at')
    def serialize_processed_at(self, v: datetime, _info):
        """Serialize datetime to ISO format string for JSON compatibility."""
        return v.isoformat() if v else None


class GenerationMetadata(BaseModel):
    """
    Tracks LLM generation details for analytics and improvement.

    Stored with each chapter to enable:
    - Model performance comparison (which models produce better prose?)
    - Token usage tracking (cost optimization)
    - Generation timing analysis (bottleneck identification)
    - A/B testing of different models/configurations
    """
    model_config = {"protected_namespaces": ()}  # Allow model_* field names

    model_used: str = Field(..., description="Model identifier (e.g., 'gpt-5-chat-2025-08-07')")
    model_provider: str = Field(default="azure_foundry", description="Provider: azure_foundry, openai, anthropic")
    generation_started_at: datetime = Field(..., description="When generation began")
    generation_completed_at: datetime = Field(..., description="When generation finished")
    duration_seconds: float = Field(..., ge=0, description="Total generation time in seconds")
    input_tokens: Optional[int] = Field(None, ge=0, description="Input token count (if available)")
    output_tokens: Optional[int] = Field(None, ge=0, description="Output token count (if available)")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Temperature setting used")
    routing_mode: Optional[str] = Field(None, description="Azure Foundry routing mode: quality, speed, cost")
    agent_name: Optional[str] = Field(None, description="Agent that generated this content (e.g., 'NarrativeAgent')")

    @field_serializer('generation_started_at', 'generation_completed_at')
    def serialize_datetime(self, v: datetime, _info):
        """Serialize datetime to ISO format string for JSON compatibility."""
        return v.isoformat() if v else None


class Chapter(BaseModel):
    """Completed chapter content with hybrid generation status tracking"""
    id: str = Field(default_factory=lambda: f"ch_{uuid.uuid4().hex[:8]}")
    number: int = Field(..., ge=1)
    title: str = Field(..., min_length=3, max_length=100)
    synopsis: str
    content: str = Field(..., min_length=1, max_length=100000)  # Allow any chapter length
    characters_featured: List[str] = Field(default_factory=list)
    educational_points: List[str] = Field(default_factory=list)
    vocabulary_words: List[VocabularyWord] = Field(default_factory=list)
    facts: List[VerifiedFact] = Field(default_factory=list)
    statements: List[Statement] = Field(default_factory=list, description="Factual claims extracted from narrative for verification tracking")
    educational_facts_embedded: List[str] = Field(default_factory=list, description="Educational facts naturally embedded in narrative (no XML tags)")
    word_count: int = Field(..., ge=100)
    reading_time_minutes: int = Field(..., ge=1)
    created_at: datetime = Field(default_factory=datetime.now)

    # Hybrid generation fields
    status: ChapterStatus = Field(default=ChapterStatus.PENDING, description="Current status in hybrid generation pipeline")
    user_inputs_applied: List[str] = Field(default_factory=list, description="IDs of QueuedInputs that affected this chapter")

    # Round Table review result
    round_table_review: Optional["RoundTableReview"] = Field(None, description="Collective review from Guillermo, Bill, and Clarissa")

    # Voice Direction (for TTS/Audiobook production - separate from prose writing)
    tts_content: Optional[str] = Field(None, description="SSML-optimized narration for TTS, generated by VoiceDirectorAgent")
    voice_direction_metadata: Optional[VoiceDirectionMetadata] = Field(None, description="Voice direction analysis: character voices, emotional beats, pacing")
    tts_status: str = Field(default="pending", description="TTS generation status: pending | generating | ready | failed")
    tts_error: Optional[str] = Field(None, description="Error message if TTS generation failed")
    audio_blob_url: Optional[str] = Field(None, description="Azure Blob Storage URL for cached chapter audio")

    # Generation Analytics (for model comparison and optimization)
    generation_metadata: Optional[GenerationMetadata] = Field(None, description="LLM generation details: model, tokens, timing")

    @validator('word_count', always=True)
    def calculate_word_count(cls, v, values):
        if 'content' in values:
            return len(values['content'].split())
        return v

    @field_serializer('created_at')
    def serialize_created_at(self, v: datetime, _info):
        """Serialize datetime to ISO format string for JSON compatibility."""
        return v.isoformat() if v else None


class DialogueEntry(BaseModel):
    """Single dialogue exchange"""
    id: str = Field(default_factory=lambda: f"dlg_{uuid.uuid4().hex[:8]}")
    speaker: Literal["user", "agent", "system"] = "user"
    # Allow 20000 chars to support long story prompts with reference material
    message: str = Field(..., min_length=1, max_length=20000)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)  # For context, emotions, etc.


class EducationalProgress(BaseModel):
    """Tracking of user's educational progress"""
    concepts_learned: List[str] = Field(default_factory=list)
    vocabulary_words: Dict[str, str] = Field(default_factory=dict)  # word -> definition
    questions_asked: int = Field(default=0, ge=0)
    chapters_read: int = Field(default=0, ge=0)
    time_spent_seconds: int = Field(default=0, ge=0)
    quiz_results: Dict[str, Dict] = Field(default_factory=dict)
    last_activity: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Hybrid Generation Reading State Models
# ============================================================================

class QueuedInput(BaseModel):
    """User input queued for future chapter generation"""
    id: str = Field(default_factory=lambda: f"input_{uuid.uuid4().hex[:8]}")
    tier: InputTier = Field(..., description="Classification tier for this input")
    raw_input: str = Field(..., min_length=1, max_length=20000, description="Original user message (supports long prompts with reference material)")
    classified_intent: str = Field(..., description="LLM-extracted intent/meaning")
    target_chapter: int = Field(..., ge=1, description="Chapter number this input should affect")
    preference_updates: Optional[Dict[str, Any]] = Field(None, description="Tier 2: preference changes to apply")
    story_direction: Optional[str] = Field(None, description="Tier 3: plot direction to incorporate")
    created_at: datetime = Field(default_factory=datetime.now)
    applied: bool = Field(default=False, description="Whether this input has been applied to a chapter")
    applied_at: Optional[datetime] = Field(None, description="When this input was applied")


class ChapterAudioState(BaseModel):
    """Audio generation state for a chapter"""
    chapter_number: int = Field(..., ge=1)
    segments: List[str] = Field(default_factory=list, description="Base64 audio segments (~500 words each)")
    total_segments: int = Field(default=0, ge=0, description="Expected total segments")
    generated_segments: int = Field(default=0, ge=0, description="Segments generated so far")

    @property
    def is_playable(self) -> bool:
        """Chapter is playable if at least first segment is ready"""
        return self.generated_segments >= 1


class StoryReadingState(BaseModel):
    """
    Persistent reading state for a story session.

    Tracks:
    - Current reading position
    - Chapter generation status
    - Queued user inputs
    - Audio generation state
    """
    story_id: str = Field(..., description="Story this reading state belongs to")
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    current_chapter: int = Field(default=0, ge=0, description="Chapter being read (0 = none)")
    chapter_position: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress through chapter (0.0-1.0)")
    generating_chapter: Optional[int] = Field(None, ge=1, description="Chapter currently being generated")
    chapter_statuses: Dict[str, ChapterStatus] = Field(default_factory=dict, description="Status of each chapter by number")
    queued_inputs: List[QueuedInput] = Field(default_factory=list, description="User inputs waiting for future chapters")
    chapter_audio_states: Dict[str, ChapterAudioState] = Field(default_factory=dict, description="Audio state per chapter")
    started_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)

    # Playback phase tracking for CompanionAgent interaction
    playback_phase: PlaybackPhase = Field(
        default=PlaybackPhase.PRE_CHAPTER,
        description="Current phase of chapter playback"
    )
    discussion_started: bool = Field(
        default=False,
        description="Whether post-chapter discussion has begun for current chapter"
    )
    queued_messages: List[str] = Field(
        default_factory=list,
        description="User messages received during playback, processed after chapter ends"
    )

    def get_chapter_status(self, chapter_num: int) -> ChapterStatus:
        """Get status of a specific chapter"""
        return self.chapter_statuses.get(str(chapter_num), ChapterStatus.PENDING)

    def set_chapter_status(self, chapter_num: int, status: ChapterStatus):
        """Update status of a specific chapter"""
        self.chapter_statuses[str(chapter_num)] = status
        self.last_active = datetime.now()

    def get_inputs_for_chapter(self, chapter_num: int) -> List[QueuedInput]:
        """Get all unapplied inputs targeting a specific chapter"""
        return [
            inp for inp in self.queued_inputs
            if inp.target_chapter == chapter_num and not inp.applied
        ]


class AgentState(BaseModel):
    """State of an individual agent"""
    agent_name: str
    status: Literal["idle", "busy", "waiting", "error"] = "idle"
    current_task_id: Optional[str] = None
    tasks_completed: int = Field(default=0, ge=0)
    last_activity: datetime = Field(default_factory=datetime.now)


class AgentCoordination(BaseModel):
    """Coordination metadata for the story"""
    current_phase: str = "initialization"
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    active_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)


class StoryMetadata(BaseModel):
    """Metadata for a story"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Child profile this story belongs to (new parent/child model)
    child_id: Optional[str] = Field(
        default=None,
        description="Child profile this story belongs to"
    )

    # DEPRECATED: Keep for backward compatibility during migration
    user_id: Optional[str] = Field(
        default=None,
        description="DEPRECATED: Use child_id. Kept for backward compatibility."
    )

    educational_goals: List[EducationalGoal] = Field(default_factory=list)
    agent_coordination: AgentCoordination = Field(default_factory=AgentCoordination)
    educational_progress: EducationalProgress = Field(default_factory=EducationalProgress)


class Story(BaseModel):
    """Complete story object"""
    id: str = Field(default_factory=lambda: f"story_{uuid.uuid4().hex[:12]}")
    prompt: str = Field(..., min_length=1, max_length=20000)  # Allow any prompt length
    preferences: StoryPreferences = Field(default_factory=StoryPreferences)
    status: StoryStatus = StoryStatus.INITIALIZING

    # Story content
    structure: Optional[StoryStructure] = None
    characters: List[Character] = Field(default_factory=list)
    chapters: List[Chapter] = Field(default_factory=list)
    dialogues: List[DialogueEntry] = Field(default_factory=list)

    # Metadata
    metadata: StoryMetadata = Field(default_factory=StoryMetadata)

    # Computed properties
    @property
    def completion_percentage(self) -> float:
        if not self.structure:
            return 0.0
        total_chapters = len(self.structure.chapters)
        completed_chapters = len([ch for ch in self.chapters if ch.content])
        return (completed_chapters / total_chapters * 100) if total_chapters > 0 else 0.0

    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Prompt must be at least 10 characters")
        return v.strip()


# ============================================================================
# User Profile Models
# ============================================================================

class UserProfile(BaseModel):
    """
    User profile with D&D-style progression tracking.

    Tracks learning, preferences, and stats across all stories.
    """
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    display_name: str = Field(..., min_length=1, max_length=100)
    current_age: int = Field(..., ge=1, description="User's current age")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Story Progress
    total_stories_completed: int = Field(default=0, ge=0)
    total_chapters_read: int = Field(default=0, ge=0)
    reading_level: int = Field(default=1, ge=1, le=10, description="1-10 scale based on complexity")

    # Cross-Story Learning (aggregated from all user's stories)
    concepts_mastered: Dict[str, int] = Field(
        default_factory=dict,
        description="Concept -> number of stories where encountered"
    )
    vocabulary_bank: Dict[str, str] = Field(
        default_factory=dict,
        description="All unique words learned across stories (word -> definition)"
    )
    favorite_themes: List[str] = Field(
        default_factory=list,
        description="Auto-detected from story choices"
    )

    # D&D-Style User Stats
    curiosity_score: int = Field(default=0, ge=0, description="Total questions asked across all stories")
    persistence_score: int = Field(default=0, ge=0, description="Stories completed")
    creativity_score: int = Field(default=0, ge=0, description="Story choices made")
    exploration_score: int = Field(default=0, ge=0, description="Different story types tried")

    # Learned Preferences (evolve over time)
    preferred_scary_level: Optional[str] = Field(default="mild")
    preferred_story_length: Optional[str] = Field(default="medium")
    interests: List[str] = Field(default_factory=list, description="Auto-detected interests")

    # Story Library References
    story_ids: List[str] = Field(default_factory=list, description="All user's story IDs for quick lookup")

    def add_story_progress(self, story: 'Story'):
        """Update user profile based on completed story"""
        self.total_stories_completed += 1
        self.total_chapters_read += len(story.chapters)
        self.persistence_score += 1

        # Aggregate concepts
        if story.metadata.educational_progress:
            for concept in story.metadata.educational_progress.concepts_learned:
                self.concepts_mastered[concept] = self.concepts_mastered.get(concept, 0) + 1

            # Aggregate vocabulary
            for word, definition in story.metadata.educational_progress.vocabulary_words.items():
                if word not in self.vocabulary_bank:
                    self.vocabulary_bank[word] = definition

            # Track curiosity
            self.curiosity_score += story.metadata.educational_progress.questions_asked

        # Update reading level based on completion
        if self.total_stories_completed % 5 == 0 and self.reading_level < 10:
            self.reading_level += 1

        self.updated_at = datetime.now()

    def get_age_band(self) -> str:
        """Return age band for targeting content"""
        if self.current_age <= 5:
            return "toddler"
        elif self.current_age <= 8:
            return "early_reader"
        elif self.current_age <= 12:
            return "middle_reader"
        elif self.current_age <= 18:
            return "teen"
        else:
            return "adult"


# ============================================================================
# Task Models
# ============================================================================

class TaskMetadata(BaseModel):
    """Metadata for a task"""
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agent: Optional[str] = None
    educational_goals: List[str] = Field(default_factory=list)
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)


class Task(BaseModel):
    """Task for agent execution"""
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    type: TaskType
    story_id: str
    status: TaskStatus = TaskStatus.QUEUED
    priority: TaskPriority = TaskPriority.MEDIUM

    # Task data
    dependencies: List[str] = Field(default_factory=list)
    data: Dict = Field(default_factory=dict)
    result: Optional[Dict] = None
    error: Optional[str] = None

    # Metadata
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)

    @property
    def can_retry(self) -> bool:
        return self.metadata.attempts < self.metadata.max_attempts


# ============================================================================
# API Request/Response Models
# ============================================================================

class StoryInitRequest(BaseModel):
    """Request to initialize a new story"""
    prompt: str = Field(..., min_length=1, max_length=20000)  # Allow any prompt length

    # New family account model: use child_id
    child_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Child profile ID (new family account model)"
    )

    # DEPRECATED: Keep for backward compatibility, use child_id instead
    user_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="DEPRECATED: Use child_id. User identifier for story ownership"
    )

    type: str = Field(default="adventure", description="Story type/genre (historical, fantasy, science, adventure, sci-fi, etc.)")
    language: Optional[str] = Field(
        default=None,
        pattern="^(en|es|no)$",
        description="Language override. If not provided, uses parent account language."
    )
    target_age: Optional[int] = Field(
        default=None,
        ge=4,
        le=120,
        description="Target age. If not provided, uses user's age from profile."
    )
    preferences: Optional[StoryPreferences] = None
    chapters_to_write: Optional[int] = Field(default=None, ge=1, description="Number of chapters to write in full detail")

    # Pre-story conversation messages to incorporate into Chapter 1
    # Allow same length as prompt (20000) since these come from conversation flow
    pre_story_messages: Optional[List[constr(max_length=20000)]] = Field(
        default=None,
        description="User messages from pre-story conversation to incorporate into the first chapter"
    )

    @validator('prompt')
    def clean_prompt(cls, v):
        return v.strip()


class StoryInitResponse(BaseModel):
    """Response after initializing a story"""
    success: bool
    story_id: str
    message: str
    story: Optional[Story] = None
    welcome_message: Optional[str] = None


class QuestionRequest(BaseModel):
    """User question about the story"""
    question: str = Field(..., min_length=1, max_length=1000)
    context: Optional[str] = None  # chapter_id, character_id, etc.


class QuestionResponse(BaseModel):
    """Response to user question"""
    success: bool
    answer: str
    educational_notes: List[str] = Field(default_factory=list)
    related_chapters: List[int] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class StoryStatusResponse(BaseModel):
    """Story status information"""
    success: bool
    story: Story
    agent_states: Dict[str, AgentState]
    tasks: Dict[str, int]  # {status: count}


class VoiceCommand(BaseModel):
    """Voice command from speaker"""
    command: str = Field(..., min_length=1, max_length=1000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    language: str = "en"
    timestamp: datetime = Field(default_factory=datetime.now)


class VoiceResponse(BaseModel):
    """Response for voice output"""
    speech: str = Field(..., description="Text to speak")
    action: Optional[str] = None  # "start_story", "continue_chapter", etc.
    story_id: Optional[str] = None
    should_wait_for_response: bool = False


# ============================================================================
# Utility functions
# ============================================================================

def example_story() -> Story:
    """Generate an example story for testing"""
    return Story(
        prompt="A story about Vikings exploring North America",
        preferences=StoryPreferences(
            language="en",
            educational_focus="history",
            difficulty=DifficultyLevel.MEDIUM,
            themes=["exploration", "courage"]
        ),
        status=StoryStatus.IN_DIALOGUE
    )


if __name__ == "__main__":
    # Test the models
    story = example_story()
    print(f"Created story: {story.id}")
    print(f"Status: {story.status}")
    print(f"Prompt: {story.prompt}")

    # Test validation
    try:
        bad_story = Story(prompt="too short")  # Should fail
    except ValueError as e:
        print(f"Validation works: {e}")

    # Test JSON serialization
    story_json = story.model_dump_json(indent=2)
    print("\nStory as JSON:")
    print(story_json[:200] + "...")
