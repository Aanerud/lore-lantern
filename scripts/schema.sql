-- ============================================================================
-- LoreLantern Azure SQL Schema
-- Migrated from Firebase Realtime Database
-- ============================================================================

-- Drop tables if they exist (for clean re-creation)
IF OBJECT_ID('reading_states', 'U') IS NOT NULL DROP TABLE reading_states;
IF OBJECT_ID('characters', 'U') IS NOT NULL DROP TABLE characters;
IF OBJECT_ID('chapters', 'U') IS NOT NULL DROP TABLE chapters;
IF OBJECT_ID('stories', 'U') IS NOT NULL DROP TABLE stories;
IF OBJECT_ID('learning_progress', 'U') IS NOT NULL DROP TABLE learning_progress;
IF OBJECT_ID('children', 'U') IS NOT NULL DROP TABLE children;
IF OBJECT_ID('households', 'U') IS NOT NULL DROP TABLE households;

-- ============================================================================
-- HOUSEHOLDS (formerly ParentAccount)
-- Top-level tenant for data isolation
-- ============================================================================
CREATE TABLE households (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    display_name NVARCHAR(255) NOT NULL,
    language NVARCHAR(10) NOT NULL DEFAULT 'en',
    created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

CREATE INDEX IX_households_created ON households(created_at);

-- ============================================================================
-- CHILDREN (formerly ChildProfile)
-- Child profiles belong to a household
-- ============================================================================
CREATE TABLE children (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    household_id UNIQUEIDENTIFIER NOT NULL,
    name NVARCHAR(255) NOT NULL,
    birth_year INT NOT NULL,
    active_story_id UNIQUEIDENTIFIER NULL,  -- FK added after stories table
    created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_children_household FOREIGN KEY (household_id)
        REFERENCES households(id) ON DELETE CASCADE
);

CREATE INDEX IX_children_household ON children(household_id);

-- ============================================================================
-- LEARNING_PROGRESS (1:1 with child)
-- Tracks vocabulary, concepts, reading level over time
-- ============================================================================
CREATE TABLE learning_progress (
    child_id UNIQUEIDENTIFIER PRIMARY KEY,
    vocabulary NVARCHAR(MAX) NULL,          -- JSON: {word: definition}
    concepts NVARCHAR(MAX) NULL,            -- JSON: {concept: count}
    reading_level INT NOT NULL DEFAULT 1,
    detected_interests NVARCHAR(MAX) NULL,  -- JSON: [interests]
    preferences NVARCHAR(MAX) NULL,         -- JSON: {scary_level, story_length}
    total_stories INT NOT NULL DEFAULT 0,
    total_chapters_read INT NOT NULL DEFAULT 0,
    curiosity_score INT NOT NULL DEFAULT 0,
    updated_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_learning_child FOREIGN KEY (child_id)
        REFERENCES children(id) ON DELETE CASCADE
);

-- ============================================================================
-- STORIES
-- Main story objects
-- ============================================================================
CREATE TABLE stories (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    household_id UNIQUEIDENTIFIER NOT NULL,
    child_id UNIQUEIDENTIFIER NOT NULL,
    prompt NVARCHAR(MAX) NOT NULL,
    status NVARCHAR(50) NOT NULL DEFAULT 'initializing',

    -- Preferences (JSON)
    preferences NVARCHAR(MAX) NULL,

    -- Structure (JSON) - title, theme, chapters outline
    structure NVARCHAR(MAX) NULL,

    created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_stories_household FOREIGN KEY (household_id)
        REFERENCES households(id) ON DELETE NO ACTION,
    CONSTRAINT FK_stories_child FOREIGN KEY (child_id)
        REFERENCES children(id) ON DELETE NO ACTION
);

CREATE INDEX IX_stories_household ON stories(household_id);
CREATE INDEX IX_stories_child ON stories(child_id);
CREATE INDEX IX_stories_status ON stories(status);

-- Now add the active_story FK to children
ALTER TABLE children ADD CONSTRAINT FK_children_active_story
    FOREIGN KEY (active_story_id) REFERENCES stories(id) ON DELETE SET NULL;

-- ============================================================================
-- CHAPTERS
-- Story chapters with content and TTS data
-- ============================================================================
CREATE TABLE chapters (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    story_id UNIQUEIDENTIFIER NOT NULL,
    number INT NOT NULL,
    title NVARCHAR(500) NOT NULL,
    synopsis NVARCHAR(MAX) NULL,
    content NVARCHAR(MAX) NULL,

    -- Voice/Audio
    tts_content NVARCHAR(MAX) NULL,         -- SSML/audio tags optimized content
    audio_blob_url NVARCHAR(1000) NULL,     -- Azure Blob Storage URL

    -- Status
    status NVARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Metadata (JSON)
    characters_featured NVARCHAR(MAX) NULL,  -- JSON array
    educational_points NVARCHAR(MAX) NULL,   -- JSON array
    vocabulary_words NVARCHAR(MAX) NULL,     -- JSON array
    facts NVARCHAR(MAX) NULL,                -- JSON array
    statements NVARCHAR(MAX) NULL,           -- JSON array (fact claims)
    user_inputs_applied NVARCHAR(MAX) NULL,  -- JSON array of input IDs
    round_table_review NVARCHAR(MAX) NULL,   -- JSON object
    voice_direction NVARCHAR(MAX) NULL,      -- JSON object
    generation_metadata NVARCHAR(MAX) NULL,  -- JSON: model, tokens, timing for analytics

    -- TTS status tracking
    tts_status NVARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending | generating | ready | failed
    tts_error NVARCHAR(MAX) NULL,            -- Error message if TTS failed

    word_count INT NOT NULL DEFAULT 0,
    reading_time_minutes INT NOT NULL DEFAULT 0,
    created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_chapters_story FOREIGN KEY (story_id)
        REFERENCES stories(id) ON DELETE CASCADE,
    CONSTRAINT UQ_chapters_story_number UNIQUE (story_id, number)
);

CREATE INDEX IX_chapters_story ON chapters(story_id);
CREATE INDEX IX_chapters_status ON chapters(status);

-- ============================================================================
-- CHARACTERS
-- Story characters with progression tracking
-- ============================================================================
CREATE TABLE characters (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    story_id UNIQUEIDENTIFIER NOT NULL,
    name NVARCHAR(500) NOT NULL,
    role NVARCHAR(100) NOT NULL,
    age NVARCHAR(50) NULL,                  -- Can be int or string like "Ancient"
    background NVARCHAR(MAX) NULL,
    appearance NVARCHAR(MAX) NULL,
    motivation NVARCHAR(MAX) NULL,

    -- JSON fields
    personality_traits NVARCHAR(MAX) NULL,   -- JSON array
    relationships NVARCHAR(MAX) NULL,        -- JSON: {character_id: relationship}
    progression NVARCHAR(MAX) NULL,          -- JSON: CharacterProgression
    character_arc NVARCHAR(MAX) NULL,        -- JSON: {chapter: milestone}

    created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_characters_story FOREIGN KEY (story_id)
        REFERENCES stories(id) ON DELETE CASCADE
);

CREATE INDEX IX_characters_story ON characters(story_id);

-- ============================================================================
-- READING_STATES
-- Session state for story reading/playback
-- ============================================================================
CREATE TABLE reading_states (
    story_id UNIQUEIDENTIFIER PRIMARY KEY,
    session_id NVARCHAR(100) NOT NULL,
    current_chapter INT NOT NULL DEFAULT 0,
    chapter_position FLOAT NOT NULL DEFAULT 0.0,
    generating_chapter INT NULL,

    -- JSON fields
    chapter_statuses NVARCHAR(MAX) NULL,     -- JSON: {chapter_num: status}
    queued_inputs NVARCHAR(MAX) NULL,        -- JSON array of QueuedInput
    chapter_audio_states NVARCHAR(MAX) NULL, -- JSON: {chapter_num: audio_state}
    queued_messages NVARCHAR(MAX) NULL,      -- JSON array

    -- Playback state
    playback_phase NVARCHAR(50) NOT NULL DEFAULT 'pre_chapter',
    discussion_started BIT NOT NULL DEFAULT 0,

    started_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    last_active DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_reading_states_story FOREIGN KEY (story_id)
        REFERENCES stories(id) ON DELETE CASCADE
);

-- ============================================================================
-- DIALOGUES (optional - can store conversation history)
-- ============================================================================
CREATE TABLE dialogues (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    story_id UNIQUEIDENTIFIER NOT NULL,
    speaker NVARCHAR(50) NOT NULL,           -- 'user', 'agent', 'system'
    message NVARCHAR(MAX) NOT NULL,
    metadata NVARCHAR(MAX) NULL,             -- JSON
    timestamp DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_dialogues_story FOREIGN KEY (story_id)
        REFERENCES stories(id) ON DELETE CASCADE
);

CREATE INDEX IX_dialogues_story ON dialogues(story_id);
CREATE INDEX IX_dialogues_timestamp ON dialogues(timestamp);

-- ============================================================================
-- FACT_CHECK_REPORTS
-- ============================================================================
CREATE TABLE fact_check_reports (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    story_id UNIQUEIDENTIFIER NOT NULL,
    chapter_number INT NOT NULL,
    issues_found NVARCHAR(MAX) NULL,         -- JSON array
    approval_status NVARCHAR(50) NOT NULL,   -- 'approved', 'needs_revision', 'major_issues'
    overall_confidence FLOAT NOT NULL DEFAULT 0.9,
    revision_round INT NOT NULL DEFAULT 1,
    timestamp DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_factcheck_story FOREIGN KEY (story_id)
        REFERENCES stories(id) ON DELETE CASCADE
);

CREATE INDEX IX_factcheck_story ON fact_check_reports(story_id);

-- ============================================================================
-- Helper function to get computed age from birth_year
-- ============================================================================
-- Note: In Azure SQL, we compute age in the application layer
-- Formula: YEAR(GETUTCDATE()) - birth_year

-- ============================================================================
-- Sample queries for common operations
-- ============================================================================

-- Get all children for a household with computed age:
-- SELECT id, name, YEAR(GETUTCDATE()) - birth_year AS age
-- FROM children WHERE household_id = @household_id

-- Get all stories for a child:
-- SELECT * FROM stories WHERE child_id = @child_id ORDER BY created_at DESC

-- Get story with all chapters:
-- SELECT s.*, c.* FROM stories s
-- LEFT JOIN chapters c ON c.story_id = s.id
-- WHERE s.id = @story_id ORDER BY c.number

PRINT 'Schema created successfully!';

-- ============================================================================
-- MIGRATION SCRIPTS (for existing databases)
-- Run these if upgrading an existing database
-- ============================================================================

-- Migration: Add generation_metadata column to chapters table (Phase 1)
-- IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS
--                WHERE TABLE_NAME = 'chapters' AND COLUMN_NAME = 'generation_metadata')
-- BEGIN
--     ALTER TABLE chapters ADD generation_metadata NVARCHAR(MAX) NULL;
--     PRINT 'Added generation_metadata column to chapters table';
-- END

-- Migration: Add tts_status and tts_error columns to chapters table
-- IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS
--                WHERE TABLE_NAME = 'chapters' AND COLUMN_NAME = 'tts_status')
-- BEGIN
--     ALTER TABLE chapters ADD tts_status NVARCHAR(50) NOT NULL DEFAULT 'pending';
--     ALTER TABLE chapters ADD tts_error NVARCHAR(MAX) NULL;
--     PRINT 'Added tts_status and tts_error columns to chapters table';
-- END
