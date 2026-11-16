SECTION_SUMMARY_PROMPT = """Summarize what this Wikipedia section covers in NO MORE than 200 words.

Requirements:
- Include all specific dates (YYYY-MM-DD or "Month Day, Year") mentioned in the content
- Include all event names, decisions, actions, or outcomes with their dates
- Classify the section type: overview, chronological timeline, list of events, biographical, statistical, or other
- Use clear, neutral language
- WORD COUNT MUST NOT EXCEED 200 WORDS

Section title: {section_title}

Content:
{content}

Output a JSON object with exactly these fields:
{{"summary": "your summary here (max 200 words)", "has_accurate_dates": true/false, "timeline_extractable": true/false}}

- "has_accurate_dates": true if the content contains specific, verifiable dates (not vague references like "in recent years")
- "timeline_extractable": true if the content describes a sequence of events that can be arranged chronologically"""


CHAPTER_SELECTION_PROMPT = """You are evaluating Wikipedia chapters for a scenario history timeline.

Event: {event_title}

Evaluate each chapter for:
1. Is it relevant to the event (discusses it, actors involved, or related developments)?
2. Is it suitable for timeline extraction (contains dated events, actions, decisions, or chronological developments)?

Chapters and their summaries:
{chapter_summaries}

Answer each chapter with YES or NO on two separate lines:
- "Relevant: YES/NO"
- "Timeline-suitable: YES/NO"

Only mark Timeline-suitable as YES if the chapter contains actual dated events (not just definitions, opinions, or static descriptions).

Return a JSON object mapping chapter names to their evaluation:
{{"chapter_name": {{"relevant": true/false, "timeline_suitable": true/false}}, ...}}"""


CHAPTER_SECTION_SELECTION_PROMPT = """You are evaluating individual sections within a Wikipedia chapter.

Event: {event_title}

Evaluate each section for:
1. Is it relevant to the event (discusses it, actors involved, or related developments)?
2. Is it suitable for timeline extraction (contains dated events, actions, decisions, or chronological developments)?

Chapter: {chapter_name}

Sections:
{section_list}

Answer each section with YES or NO:
- "Relevant: YES/NO"
- "Timeline-suitable: YES/NO"

Only mark Timeline-suitable as YES if the section contains actual dated events.

Return a JSON object:
{{"selected": ["path1", "path2", ...]}}"""


TIMELINE_EXTRACTION_PROMPT = """Extract dated events from the following text as a structured historical timeline.

Section: {section_title}

Text:
{content}

Rules:
- Keep an event only when its year AND month are explicitly available or can be confidently inferred from the surrounding chronology and source text. If the month cannot be resolved, omit the event.
- Use an explicit date in the section heading or its parent chronology when it governs the entry. Copy the actual date wording or relative phrase from the source. Treat different documented actions on the same day as separate events.
- Do not invent a month merely to satisfy the format. A year alone is not a valid timestamp. Use "YYYY-MM" when the month is known; use "YYYY-MM-DD" only when the exact day is known.
- Distinguish the date an action occurred from a report date, measurement coverage period, and future effective date. Quarter/year coverage is not an occurrence month: omit it unless a separate occurrence month is explicit.
- A past announcement of a future plan is an announcement event at the announcement date; never treat its future effective date as an already-realized outcome.
- For an actual event date range with an explicit start month, retain the source range; never replace a quarter or year with its first month.
- Preserve the original date phrase in "date_original".
- Copy evidence_text verbatim from the source, preserving the date or a short immediately adjacent date anchor. Do not paraphrase the quotation.
- Set time_role to occurrence or announcement for retained events and modality to occurred or announced. Measurement coverage and forecast horizons are not historical occurrences.
- Classify "event_type" as one of: policy, diplomatic, economic, military, legal, election, protest, communication, agreement, crisis, technology, other.
- "importance_score" should reflect how central the event is to the overall scenario, from 0.0 to 1.0.
- "confidence" should reflect confidence in the date and event extraction, from 0.0 to 1.0.

Return ONLY valid JSON with this exact shape:
{{
  "events": [
    {{
      "timestamp": "YYYY-MM-DD or YYYY-MM or YYYY",
      "date_precision": "day|month|year|range|approximate",
      "date_original": "date phrase copied or summarized from the text",
      "date_range": {{"start": "YYYY-MM-DD or YYYY-MM or YYYY", "end": "YYYY-MM-DD or YYYY-MM or YYYY"}},
      "time_role": "occurrence|announcement",
      "modality": "occurred|announced",
      "description": "concise description of what happened",
      "event_type": "economic",
      "actors": ["actor name"],
      "evidence_text": "short source evidence",
      "confidence": 0.8,
      "importance_score": 0.7
    }}
  ]
}}"""


TIMELINE_DEDUP_PROMPT = """You are reviewing structured timeline events for deduplication.

Time span: {time_span}

Events JSON:
{events_text}

Each input event includes source_event_ids. When you merge events, preserve and combine those source_event_ids.
For near-duplicates, keep the most complete description and preserve useful metadata:
- timestamp
- date_precision
- date_original
- date_range
- description
- event_type
- actors
- evidence_text
- confidence
- importance_score
- source_sections
- source_event_ids

Return ONLY valid JSON:
{{"deduplicated": [
  {{
    "timestamp": "YYYY-MM-DD or YYYY-MM (year and month required)",
    "date_precision": "day|month|range|approximate",
    "date_original": "...",
    "date_range": {{"start": "...", "end": "..."}},
    "description": "...",
    "event_type": "economic",
    "actors": ["..."],
    "evidence_text": "...",
    "confidence": 0.8,
    "importance_score": 0.7,
    "source_sections": ["..."],
    "source_event_ids": [1, 2]
  }}
]}}

Only include unique events. Sort by timestamp ascending."""


TIMELINE_AGENT_PROMPT = """Build the event timeline for "{event_title}" from the evidence sections below.

{event_title} is the overarching event. Every supplied section belongs to it.

Sections:
{sections}

Return one result:

- "historical": the events that actually happened, in chronological order. Every event must
  have a resolvable year and month. Use YYYY-MM, or YYYY-MM-DD only when the day is known.
  You may infer a month only when the surrounding source chronology makes it clear; omit an
  event when its month cannot be established. Never return year-only dates. Cite the section
  "path" values exactly as given in "source_sections".

Set "analysis" to a short note on the record, or leave it empty."""
