# Lore Lantern Log Analysis Guide

## Overview

The `log.txt` file contains the complete execution trace of a story generation session. This guide explains how to extract analytics data for comparing multiple test runs.

---

## Log Structure - Timeline of Events

A typical e2e test follows this pipeline:

```
1. STARTUP (lines 1-280)
   └── Server initialization, model configuration

2. STORY INIT (lines ~280-290)
   └── Child profile, age grounding, story ID creation

3. STRUCTURE GENERATION (lines ~290-900)
   └── StructureAgent (Guillermo) creates chapters + character list

4. NARRATIVE METHOD (lines ~900-960)
   └── Guillermo + Stephen debate POV and hooks

5. CHARACTER CREATION (lines ~960-3500)
   └── Semantic batching → CharacterAgent per character → Bill validation

6. STRUCTURE V2 (lines ~3500-4000)
   └── Refinement pass with full character context

7. CHAPTER 1 GENERATION (lines ~4000-6000)
   └── NarrativeAgent → Round Table Review → Polish

8. CHAPTER 2+ GENERATION (if reached)
   └── Same pipeline per chapter
```

---

## Key Markers for Extraction

### 1. **Test Metadata**

| Pattern | Description | Example |
|---------|-------------|---------|
| `Story initialized: <UUID>` | Story ID | `Story initialized: dc834062-fb85-551e-8726-409fa1355a8d` |
| `Emma is X years old` | Child age | `Emma is 8 years old (born 2018)` |
| `Language GROUNDED` | Language | `Language GROUNDED from parent: en` |

**Regex:**
```
Story initialized: ([a-f0-9-]{36})
is (\d+) years old
Language GROUNDED from parent: (\w+)
```

---

### 2. **Model Configuration**

Models are logged at startup under "Agent models:":

```
🤖 Agent models:
    • Guillermo (structure): claude-opus-4-5-20251101 [agent override]
    • Bill (factcheck): claude-sonnet-4-5-20250929 [agent override]
    • Clarissa (character): claude-opus-4-5-20251101 [agent override]
    ...
```

**Regex:**
```
• (\w+) \((\w+)\): ([^\[]+) \[([^\]]+)\]
```
Captures: display_name, agent_id, model, source

---

### 3. **StructureAgent Output**

```
[01:02:28] ℹ️ 📤 StructureAgent raw output length: 20034 chars
[01:02:28] ℹ️    ✅ Successfully parsed structure JSON
[01:02:28] ℹ️    💾 Structure saved to Firebase
```

**Extract:**
- Timestamp: `[HH:MM:SS]`
- Output length: `StructureAgent raw output length: (\d+) chars`
- Duration: Calculate from first StructureAgent log to "Structure saved"

---

### 4. **Character Creation Analytics**

#### Character Batching Info
```
[01:02:48] ℹ️ 📊 Creating ALL 9 unique characters for complete story (deduped from 9)
[01:02:48] ℹ️ 🚀 Starting SMART BATCHED character generation:
[01:02:48] ℹ️    📌 Major: 4 characters
[01:02:48] ℹ️    📌 Supporting: 3 characters
[01:02:48] ℹ️    📌 Minor: 2 characters
```

**Regex:**
```
Creating ALL (\d+) unique characters.*\(deduped from (\d+)\)
📌 Major: (\d+) characters
📌 Supporting: (\d+) characters
📌 Minor: (\d+) characters
```

#### Semantic Batches
```
📦 Batch 1: [Harald, King Halfdan, Guthorm, Gyda]
```

**Regex:**
```
📦 Batch (\d+): \[([^\]]+)\]
```

#### Per-Character Creation
```
[01:02:48] ➡️ Batch 1/1: Creating Harald (1/4) - sees 0 existing chars
[01:03:28] ✅ Character created: Harald Sigurdsson, called 'Fairhair'... (protagonist)
```

**Regex:**
```
Creating (\w+) \((\d+)/(\d+)\) - sees (\d+) existing chars
✅ Character created: ([^(]+) \(([^)]+)\)
```

Captures: char_name, index, total, existing_count, full_name, role

#### Bill Validation
```
[01:03:28] 🔬 Bill validating character: Harald Sigurdsson...
[01:03:49]    ✅ Harald Sigurdsson...: Historical figure, timeline VALID
```

**Regex:**
```
🔬 Bill validating character: (.+)
(✅|⚠️|❌) (.+): (.+)
```

---

### 5. **Narrative Method Selection**

```
[01:02:47] ℹ️ 📖 Narrative method selected: linear_single_pov
[01:02:47] ℹ️    POV: ['Harald']
[01:02:47] ℹ️    Hook strategy: Each chapter ends with Harald facing...
```

**Regex:**
```
Narrative method selected: (\w+)
POV: \['([^']+)'\]
Hook strategy: (.+)
```

---

### 6. **Structure V2 Refinement**

```
[HH:MM:SS] ℹ️ 🔄 STRUCTURE V2: Refining story structure with character context...
[HH:MM:SS] ℹ️ ✅ Structure V2 refinement complete
```

**Regex:**
```
🔄 STRUCTURE V2: Refining
✅ Structure V2 refinement complete
```

Duration: Time between these two markers

---

### 7. **Chapter Generation**

#### Chapter Start
```
[HH:MM:SS] ℹ️ 📖 CHAPTER 1: Writing with full narrative context...
```

#### NarrativeAgent Draft
```
[HH:MM:SS] ℹ️ 📤 NarrativeAgent raw output: XXXXX chars
```

#### Round Table Review
```
[HH:MM:SS] ℹ️ 📋 [Round Table] Convening to review Chapter 1...
[HH:MM:SS] ℹ️    📝 Guillermo (structure): [APPROVED/CONCERN/BLOCK]
[HH:MM:SS] ℹ️    📝 Bill (factcheck): [APPROVED]
[HH:MM:SS] ℹ️    📝 Clarissa (character): [APPROVED]
[HH:MM:SS] ℹ️    📝 Benjamin (line_editor): [CONCERN]
[HH:MM:SS] ℹ️    📝 Continuity (continuity): [APPROVED]
[HH:MM:SS] ℹ️    📝 Stephen (tension): [APPROVED]
[HH:MM:SS] ℹ️ 🎯 [Round Table] DECISION: approved_with_notes
```

**Regex:**
```
📋 \[Round Table\] Convening to review Chapter (\d+)
📝 (\w+) \((\w+)\): \[(\w+)\]
🎯 \[Round Table\] DECISION: (\w+)
```

Captures: chapter_num, agent_display, agent_id, verdict, decision

#### Polish Pass
```
[HH:MM:SS] ℹ️ ✨ POLISH PASS - Applying X reviewer suggestions...
[HH:MM:SS] ℹ️    ✅ Polish complete: 1200 → 1250 words (+50)
```

**Regex:**
```
✨ POLISH PASS - Applying (\d+) reviewer suggestions
✅ Polish complete: (\d+) → (\d+) words \(([+-]\d+)\)
```

#### Chapter Complete
```
[HH:MM:SS] ℹ️ ✅ Chapter 1 COMPLETE: "The Hall of the Black King" (1389 words)
```

**Regex:**
```
✅ Chapter (\d+) COMPLETE: "([^"]+)" \((\d+) words\)
```

---

### 8. **Timing Markers**

All log lines start with timestamps:
```
[HH:MM:SS]
```

**Calculate durations:**
- StructureAgent: First Guillermo call to "Structure saved"
- Character Creation: "Creating ALL X characters" to last "Character created"
- Structure V2: "STRUCTURE V2: Refining" to "V2 refinement complete"
- Chapter N: "CHAPTER N: Writing" to "Chapter N COMPLETE"
- Round Table: "Convening to review" to "DECISION"

---

## Analytics Extraction Script (Example)

```python
import re
from datetime import datetime, timedelta

def parse_log(log_path):
    with open(log_path) as f:
        content = f.read()

    analytics = {
        "story_id": None,
        "child_age": None,
        "language": None,
        "models": {},
        "characters": {
            "total": 0,
            "major": 0,
            "supporting": 0,
            "minor": 0,
            "details": []
        },
        "chapters": [],
        "timings": {}
    }

    # Extract story ID
    match = re.search(r'Story initialized: ([a-f0-9-]{36})', content)
    if match:
        analytics["story_id"] = match.group(1)

    # Extract child age
    match = re.search(r'is (\d+) years old', content)
    if match:
        analytics["child_age"] = int(match.group(1))

    # Extract models
    for match in re.finditer(r'• (\w+) \((\w+)\): ([^\[]+) \[([^\]]+)\]', content):
        display, agent_id, model, source = match.groups()
        analytics["models"][agent_id] = {
            "display_name": display,
            "model": model.strip(),
            "source": source
        }

    # Extract character counts
    match = re.search(r'Creating ALL (\d+) unique characters', content)
    if match:
        analytics["characters"]["total"] = int(match.group(1))

    match = re.search(r'📌 Major: (\d+)', content)
    if match:
        analytics["characters"]["major"] = int(match.group(1))

    match = re.search(r'📌 Supporting: (\d+)', content)
    if match:
        analytics["characters"]["supporting"] = int(match.group(1))

    match = re.search(r'📌 Minor: (\d+)', content)
    if match:
        analytics["characters"]["minor"] = int(match.group(1))

    # Extract character details
    for match in re.finditer(r'✅ Character created: ([^(]+) \(([^)]+)\)', content):
        analytics["characters"]["details"].append({
            "name": match.group(1).strip(),
            "role": match.group(2)
        })

    # Extract Round Table reviews
    for match in re.finditer(
        r'📋 \[Round Table\] Convening to review Chapter (\d+).*?'
        r'🎯 \[Round Table\] DECISION: (\w+)',
        content, re.DOTALL
    ):
        chapter_num = int(match.group(1))
        decision = match.group(2)

        # Get individual verdicts
        verdicts = {}
        review_section = match.group(0)
        for v in re.finditer(r'📝 (\w+) \((\w+)\): \[(\w+)\]', review_section):
            verdicts[v.group(2)] = v.group(3)

        analytics["chapters"].append({
            "number": chapter_num,
            "decision": decision,
            "verdicts": verdicts
        })

    return analytics
```

---

## Comparing Multiple Log Files

### Key Metrics to Compare

| Metric | Description | How to Extract |
|--------|-------------|----------------|
| **Total Duration** | End-to-end time | Last timestamp - first timestamp |
| **Structure Time** | Time for initial planning | "Structure saved" timestamp - story init |
| **Character Time** | Time for all characters | Last "Character created" - first "Creating" |
| **Ch1 Generation** | Chapter 1 total time | "Chapter 1 COMPLETE" - "CHAPTER 1: Writing" |
| **Round Table Time** | Review duration | "DECISION" - "Convening to review" |
| **Word Count** | Chapter length | "Chapter N COMPLETE: ... (X words)" |
| **Characters Created** | Total count | "Creating ALL X unique characters" |
| **Round Table Decision** | Pass rate | Count APPROVED vs CONCERN vs BLOCK |
| **Models Used** | Provider comparison | Parse "Agent models:" section |

### Sample Comparison Output

```
| Metric               | Anthropic Test | OpenAI Test | Google Test |
|---------------------|----------------|-------------|-------------|
| Total Duration      | 30.6 min       | 9.9 min     | 18.0 min    |
| Structure Time      | 105s           | 45s         | 72s         |
| Character Time      | 280s           | 120s        | 180s        |
| Ch1 Word Count      | 1389           | 3115        | 1285        |
| Characters Created  | 9              | 12          | 12          |
| Round Table Pass    | 6/6            | 5/6         | 6/6         |
| Chapters Completed  | 1              | 2           | 2           |
```

---

## Event Types Summary

| Event | Marker | Data Available |
|-------|--------|----------------|
| Story Init | `Story initialized:` | UUID, age, language |
| Structure Start | `StructureAgent raw output` | Output length |
| Structure Done | `Structure saved to Firebase` | Timestamp |
| Char Batch Info | `📌 Major/Supporting/Minor` | Counts |
| Char Created | `✅ Character created:` | Name, role |
| Bill Validation | `🔬 Bill validating` | Character, result |
| Structure V2 Start | `STRUCTURE V2: Refining` | Timestamp |
| Structure V2 Done | `V2 refinement complete` | Timestamp |
| Chapter Start | `CHAPTER N: Writing` | Chapter number |
| Round Table Start | `Convening to review` | Chapter number |
| Agent Verdict | `📝 Agent (id): [VERDICT]` | Agent, verdict |
| RT Decision | `DECISION:` | approved/revise |
| Polish Pass | `POLISH PASS` | Suggestion count |
| Chapter Done | `Chapter N COMPLETE` | Title, word count |

---

## Notes

1. **Line numbers shift** between runs - always use pattern matching, not line numbers
2. **Timestamps are relative** - use `[HH:MM:SS]` format, parse with strptime
3. **LiteLLM debug logs** contain full prompts - useful for debugging but very verbose
4. **Color codes** like `[92m` are ANSI escapes - strip them for parsing
5. **Multi-line outputs** (like JSON) need special handling
