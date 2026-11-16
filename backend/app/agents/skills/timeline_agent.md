Timeline construction skill: build one event timeline from the selected evidence sections.

Goal:
Produce the whole recorded timeline in a single pass. This timeline is the shared factual
base for figure extraction, relationship extraction, and every later forecast step.

Method:
1. Read every provided section before writing anything.
2. Extract the dated events that actually happened, in chronological order.
3. Record only what the sections support. The forecast produces its own forward events; this
   pass does not speculate.

Output list:

historical — events recorded in the provided sections.
- Extract only events with an identifiable date, month, year, or date range.
- Year inference: the overarching event is the one named in the request. When a date gives
  only month and day, infer the year from the surrounding text, from the event title, and
  above all from the section title — if a section title carries a year, that year anchors
  the dates inside that section.
- Never invent a missing month or day. Use "YYYY" for year-only, "YYYY-MM" for year-month,
  "YYYY-MM-DD" for a full date.
- For a range, set "timestamp" to the normalized start and "date_precision" to "range".
- Keep the original date phrase in "date_original".
- "evidence_text" is one short clause copied or closely paraphrased from the section it
  came from.
- "source_sections" lists the section paths this event was drawn from. Use the paths
  exactly as given, and never cite a path that was not supplied.
- "actors" may only contain people and organizations named in the source text.
- "event_type" is one of: policy, diplomatic, economic, military, legal, election, protest,
  communication, agreement, crisis, technology, other.
- "date_precision" is one of: day, month, year, range, approximate.
- "importance_score" reflects how central the event is to the overall scenario,
  0.0 to 1.0.
- "confidence" reflects confidence in the date and the extraction itself, 0.0 to 1.0.

Output discipline:
- At most 60 historical events.
- Do not fill slots just to reach the maximum. A short, well-grounded timeline beats a
  padded one.

Language:
Write descriptions in the language of the source sections. Keep actor names as they appear
in the source.
