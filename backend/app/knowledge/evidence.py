"""Source-grounded date validation. Model-normalized dates are proposals only."""

from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta

VERSION = "date-evidence-6"
MONTHS = {
    name.casefold(): i
    for i, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}
MONTHS.update({name[:3]: i for name, i in list(MONTHS.items())})
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
YEAR = r"[12]\d{3}"
COARSE = re.compile(
    r"\b(?:Q[1-4]|(?:first|second|third|fourth|[1-4](?:st|nd|rd|th))\s+quarter|quarter|spring|summer|autumn|winter|half[- ]year)\b|季度|半年|春季|夏季|秋季|冬季",
    re.I,
)


def _normalized(value: str):
    # Keep offsets into the original text, even after whitespace normalization.
    chars, positions = [], []
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }
    for i, character in enumerate(value):
        c = replacements.get(character, character)
        if c.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                positions.append(i)
        else:
            chars.append(c)
            positions.append(i)
    while chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    if not quote.strip():
        return None
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    normalized, offsets = _normalized(text)
    needle, _ = _normalized(quote)
    start = normalized.find(needle)
    if start < 0 and re.search(r"\.{3}|…", quote):
        parts = [_normalized(p.strip())[0] for p in re.split(r"\.{3}|…+", quote)]
        if (
            not 2 <= len(parts) <= 6
            or any(len(p) < 8 for p in parts)
            or sum(map(len, parts)) < 40
        ):
            return None
        matches = []
        first = normalized.find(parts[0])
        attempts = 0
        while first >= 0:
            attempts += 1
            if attempts > 32:
                return None
            end = first + len(parts[0])
            valid = True
            for part in parts[1:]:
                boundary = min(len(normalized), first + 1800, end + 800 + len(part))
                position = normalized.find(part, end, boundary)
                if position < 0:
                    valid = False
                    break
                # Repeated possible alignments are uncertain, not a reason to
                # search an exponential number of wildcard combinations.
                if normalized.find(part, position + 1, boundary) >= 0:
                    return None
                end = position + len(part)
            if valid:
                a, b = offsets[first], offsets[end - 1] + 1
                if b - a <= 1800 and "\n" not in text[a:b]:
                    matches.append((a, b))
                if len(matches) > 1:
                    return None
            first = normalized.find(parts[0], first + 1)
        return matches[0] if len(matches) == 1 else None
    if start < 0 or not needle:
        return None
    return offsets[start], offsets[start + len(needle) - 1] + 1


def date_parts(text: str, anchor_year: int | None = None) -> set[str]:
    """Parse supported expressions, validating days with the calendar."""
    output, explicit_spans = set(), []

    def add(year, month, day=None):
        try:
            value = date(int(year), int(month), int(day or 1))
            output.add(value.isoformat() if day else value.strftime("%Y-%m"))
        except (ValueError, TypeError):
            pass

    for m in re.finditer(
        rf"(?<!\d)({YEAR})\s*[-/.年]\s*(0?[1-9]|1[0-2])(?!\d)(?:\s*[-/.月]\s*(0?[1-9]|[12]\d|3[01])(?!\d)\s*日?)?",
        text,
    ):
        add(*m.groups())
    for m in re.finditer(
        rf"\b({MONTH_RE})\.?\s+(?:(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+)?({YEAR})\b",
        text,
        re.I,
    ):
        explicit_spans.append(m.span())
        add(m[3], MONTHS[m[1].lower()], m[2])
    for m in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\.?[,]?\s+({YEAR})\b", text, re.I
    ):
        explicit_spans.append(m.span())
        add(m[3], MONTHS[m[2].lower()], m[1])
    if anchor_year:
        for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\b", text, re.I):
            if any(a <= m.start() < b for a, b in explicit_spans):
                continue
            explicit_spans.append(m.span())
            add(anchor_year, MONTHS[m[2].lower()], m[1])
        for m in re.finditer(rf"\b({MONTH_RE})\b(?:\s+(\d{{1,2}})(?!\d))?", text, re.I):
            if any(a <= m.start() < b for a, b in explicit_spans):
                continue
            # A standalone "may" is also a modal verb.
            if (
                m[1].lower() == "may"
                and not m[2]
                and not re.search(
                    r"(?:in|during|on|by)\s*$",
                    text[max(0, m.start() - 10) : m.start()],
                    re.I,
                )
            ):
                continue
            add(anchor_year, MONTHS[m[1].lower()], m[2])
        for m in re.finditer(r"(?<!\d)([1-9]|1[0-2])月(?:(\d{1,2})日)?", text):
            add(anchor_year, m[1], m[2])
    return output


# Keep bounds/comparisons separate from the occurrence they accompany. A sentence
# may contain an exact occurrence date AND "for the first time since 2005".
DATE_EXPRESSION = (
    rf"(?:{YEAR}[-/.年]\s*\d{{1,2}}(?:[-/.月]\s*\d{{1,2}}日?)?"
    rf"|\d{{1,2}}\s+(?:{MONTH_RE})(?:,?\s+{YEAR})?"
    rf"|(?:{MONTH_RE})(?:\s+\d{{1,2}}(?!\d),?)?(?:\s+{YEAR})?"
    rf"|{YEAR}(?:年)?(?:\d{{1,2}}月(?:\d{{1,2}}日)?)?)"
)
BOUND_DATE = re.compile(
    rf"\b(?:before|after|by|until|through|since|earlier than|later than)\s+(?:the\s+)?{DATE_EXPRESSION}\b"
    rf"|(?:截至|不晚于){DATE_EXPRESSION}|{DATE_EXPRESSION}(?:之前|之后|以前|以后)",
    re.I,
)
_ANNOUNCEMENT = re.compile(r"announc|declared|said|stated|reported|issued|宣布|表示|公布|发布|通报", re.I)


def _date_only_context(value: str, anchor_year: int | None = None) -> bool:
    """Allow a date prefix/suffix of the SAME clause, never a neighboring event."""
    stripped = re.sub(DATE_EXPRESSION, "", value, flags=re.I)
    stripped = re.sub(r"\b(?:on|in|during|at)\b|于|在", "", stripped, flags=re.I)
    return bool(date_parts(value, anchor_year)) and not re.search(r"[\w\u4e00-\u9fff]", stripped)


def _section_chronology(section: dict) -> tuple[int | None, str, set[str]]:
    parts = str(section.get("path") or "").split("/")[1:]
    heading = re.sub(r" \(part \d+\)$", "", parts[-1] if parts else str(section.get("title") or "")).strip()
    heading = heading if re.fullmatch(DATE_EXPRESSION, heading, re.I) else ""
    year_contexts = [*reversed(parts), str(section.get("title") or ""), str(section.get("source_title") or "")]
    year = None
    for context in year_contexts:
        years = set(re.findall(rf"(?<!\d){YEAR}(?!\d)", context))
        if years:
            year = int(next(iter(years))) if len(years) == 1 else None
            break
    return year, heading, date_parts(heading, year)


# Bind a date-prefixed assertion to the event's predicate. This is deliberately
# conservative: unsupported syntax still needs narrower evidence, not a guess.
_ACTION_FAMILIES = {
    'sign': r'signed|signs|sign|签署|签订',
    'approve': r'approved|approves|approve|批准',
    'pass': r'passed|passes|pass|通过',
    'issue': r'issued|issues|issue|颁布|发布',
    'announce': r'announced|announces|announce|宣布|公布',
    'submit': r'submitted|submits|submit|提交',
    'meet': r'met|meets|meet|会见|会晤',
    'arrive': r'arrived|arrives|arrive|抵达',
    'open': r'opened|opens|open|开业|开放',
    'close': r'closed|closes|close|关闭',
    'start': r'began|begins|begin|started|starts|start|开始',
    'end': r'ended|ends|end|finished|finishes|finish|结束',
    'release': r'released|releases|release|发放',
    'deliver': r'delivered|delivers|deliver|交付',
    'resign': r'resigned|resigns|resign|辞职',
    'file': r'filed|files|file|提起',
}


def _first_action(text):
    found = []
    for family, pattern in _ACTION_FAMILIES.items():
        match = re.search(rf'(?<!\w)(?:{pattern})(?!\w)', text, re.I)
        # Chinese predicates do not have whitespace word boundaries.
        if not match:
            match = re.search('|'.join(re.escape(p) for p in pattern.split('|') if not p.isascii()) or r'(?!)', text)
        if match:
            found.append((match.start(), family))
    return min(found)[1] if found else None


def _dated_assertion(quote: str, description: str) -> str | None:
    prefix = re.match(rf'^\s*(?:(?:on|in|during)\s+|于|在)?({DATE_EXPRESSION})\s*[,，:：]\s*', quote, re.I)
    if not prefix or not date_parts(prefix[1]):
        return None
    # Commas inside the initial date have already been consumed. Stop before
    # consequences, relative clauses or another finite coordinated assertion.
    tail = quote[prefix.end():]
    split = re.search(r',\s*(?:\w+ing\b|which\b|whereas\b|while\b)|\b(?:and|but)\s+(?:on|in)\s+|[;；。]|[，](?:并|将|有效|延长|避免)', tail, re.I)
    clause = quote[:prefix.end() + split.start()] if split else quote
    if re.search(r'\b(?:will|would|plans?|scheduled|expected|intends?|proposed)\b|将于|计划|拟于|预计', clause, re.I):
        return None
    predicate = _first_action(description)
    return clause if predicate and predicate == _first_action(clause[prefix.end():]) else None


def validate_event(raw: dict, section: dict) -> tuple[dict | None, str]:
    text = str(section.get("content") or "")
    quote = str(raw.get("evidence_text") or "")
    span = locate_quote(text, quote)
    if span is None:
        return None, "evidence_not_located"
    original = str(raw.get("date_original") or "")
    role = str(raw.get("time_role") or "occurrence")
    if role not in {"occurrence", "announcement"} or raw.get(
        "modality", "occurred"
    ) not in {"occurred", "announced"}:
        return None, "nonhistorical_or_unsupported_temporal_role"
    if COARSE.search(original) or role in {
        "measurement_coverage",
        "forecast_horizon",
        "effective_period",
    }:
        return None, "coverage_period_is_not_occurrence_month"
    a, b = span
    section_year, heading, heading_dates = _section_chronology(section)
    # Prefer the evidence sentence; use only its local line as immediate context.
    line_start, line_end = text.rfind("\n", 0, a) + 1, text.find("\n", b)
    if line_end < 0:
        line_end = len(text)
    # Chinese punctuation does not require following whitespace.
    local_start, local_end = max(line_start, a - 180), min(line_end, b + 180)
    sentence_end = r"[。！？]|[.!?](?=\s|$)"
    boundaries = list(re.finditer(sentence_end, text[local_start:a]))
    if boundaries:
        local_start += boundaries[-1].end()
    actual_quote = text[a:b]
    dated_assertion = _dated_assertion(actual_quote, str(raw.get('description', '')))
    assertion_quote = dated_assertion or actual_quote
    if role == "occurrence" and re.search(
        r"\b(?:will|would|plans? to|scheduled to|expected to|set to)\s+(?:\w+\s+){0,3}(?:begin|start|happen|occur|take effect|launch|open|impose|implement|introduce|raise|cut|meet|negotiate|sign|announce|vote|strike|deliver|ship|close|stop|end|release|participate|hold)|(?:将于|计划于|拟于|预计于|将会|将实施|将采取|将举行|将开始|将结束|将提高|将降低|将生效|将签署|将推出|将交付)",
        assertion_quote,
        re.I,
    ):
        return None, "planned_event_is_not_observed_occurrence"
    if re.search(r"[.!?。！？]\s*$", actual_quote):
        local_end = b
    else:
        tail = re.search(sentence_end, text[b:local_end])
        if tail:
            local_end = b + tail.end()
    local = text[local_start:local_end]
    if role == "announcement" and not _ANNOUNCEMENT.search(local):
        return None, "announcement_act_not_supported"
    occurrence_quote = BOUND_DATE.sub("", assertion_quote)
    if (BOUND_DATE.search(original) and not date_parts(BOUND_DATE.sub('', original))) or (
        BOUND_DATE.search(actual_quote) and not date_parts(occurrence_quote)
    ):
        return None, "temporal_bound_is_not_occurrence_month"
    if COARSE.search(actual_quote) and not date_parts(actual_quote):
        return None, "coverage_period_is_not_occurrence_month"
    prefix, suffix = text[local_start:a], text[b:local_end]
    bound_context = actual_quote
    if _date_only_context(prefix, section_year):
        bound_context = prefix + bound_context
    if _date_only_context(suffix, section_year):
        bound_context += suffix
    bound_context = BOUND_DATE.sub("", bound_context)
    direct = date_parts(bound_context)
    years = {int(x[:4]) for x in direct}
    years.update(
        int(m[1])
        for m in re.finditer(
            rf"(?:\b(?:in|during|since|year|before|after|late|early|by)\s+({YEAR})\b|({YEAR})年)",
            bound_context,
            re.I,
        )
        if m[1]
    )
    years.update(
        int(m[2])
        for m in re.finditer(
            rf"(?:\b(?:in|during|since|year|before|after|late|early|by)\s+({YEAR})\b|({YEAR})年)",
            bound_context,
            re.I,
        )
        if m[2]
    )
    anchor = next(iter(years)) if len(years) == 1 else section_year if not years else None
    supported = date_parts(occurrence_quote, anchor)
    temporal_quote = actual_quote
    if role == "announcement":
        # Bind the announcement to its own clause, not a future effective date.
        clause = re.split(
            r"\bthat\b|\b(?:would|will)\s+(?:take effect|begin|start)|将于|并将|生效",
            actual_quote,
            maxsplit=1,
            flags=re.I,
        )[0]
        if _ANNOUNCEMENT.search(clause):
            supported = date_parts(clause, anchor)
            temporal_quote = clause
    date_resolution = "dated_predicate_clause" if dated_assertion else "unique_expression"
    if len({v[:7] for v in supported}) > 1:
        # A source's start/end range can support two atomic events. Bind only
        # when a verb in the event matches an explicitly quoted date clause.
        clauses = list(
            re.finditer(
                rf"\b(began|started|ended|finished)\s+(?:on\s+)?((?:{MONTH_RE})\s+\d{{1,2}}(?:,?\s+{YEAR})?)",
                actual_quote,
                re.I,
            )
        )
        selected = [
            m
            for m in clauses
            if original.strip()
            and _normalized(m[2])[0].casefold() == _normalized(original)[0].casefold()
        ]
        description = str(raw.get("description", ""))
        families = {
            "start" if m[1].lower() in {"began", "started"} else "end" for m in clauses
        }
        if len(clauses) == 2 and families == {"start", "end"} and len(selected) == 1:
            chosen = selected[0]
            verb = (
                r"\b(?:began|begin|started|start)\b|开始|爆发"
                if chosen[1].lower() in {"began", "started"}
                else r"\b(?:ended|end|finished|finish)\b|结束"
            )
            if re.search(verb, description, re.I):
                supported = date_parts(chosen[2], anchor)
                date_resolution = "explicit_start_end_clause"
        if len({v[:7] for v in supported}) > 1:
            return None, "multiple_event_months_require_narrower_evidence"
    relative = re.search(
        r"\b(?:the\s+)?(?:next|following)\s+(day|month)\b|次日|翌日|次月",
        temporal_quote,
        re.I,
    )
    prior_day = re.search(
        r"\b(?:the\s+)?(?:day before(?=\s*(?:[.!?,;:]|$))|prior day\b|previous day\b)|前一日|前一天",
        temporal_quote, re.I,
    )
    if relative:
        previous = date_parts(text[max(0, a - 250):a].rsplit("\n\n", 1)[-1])
        if len(previous) != 1:
            return None, "relative_date_anchor_ambiguous"
        previous = next(iter(previous))
        day = relative[1] == "day" or relative[0] in {"次日", "翌日"}
        if day:
            if len(previous) != 10:
                return None, "relative_day_needs_exact_anchor"
            supported = {(date.fromisoformat(previous) + timedelta(days=1)).isoformat()}
        else:
            y, m = map(int, previous[:7].split("-"))
            supported = {f"{y+(m==12):04d}-{m%12+1:02d}"}
    elif prior_day:
        report_dates = supported or heading_dates
        if len(report_dates) != 1 or len(report_date := next(iter(report_dates))) != 10:
            return None, "relative_day_needs_exact_anchor"
        supported = {(date.fromisoformat(report_date) - timedelta(days=1)).isoformat()}
        date_resolution = "relative_to_source_day"
    elif re.search(
        r"\b(?:next|following|previous)\s+year\b|翌年|次年|去年", actual_quote, re.I
    ):
        return None, "relative_year_requires_explicit_resolution"
    if not supported:
        # Date immediately outside a short quote is allowed only when unique.
        context_dates = date_parts(bound_context, anchor)
        if len({d[:7] for d in context_dates}) == 1:
            supported = context_dates
        elif not context_dates and not years and heading_dates and not re.search(
            rf"(?<!\d){YEAR}(?!\d)", occurrence_quote
        ):
            supported = heading_dates
            date_resolution = "section_date_heading"
    proposed = str(raw.get("timestamp") or "").strip()
    if proposed in supported:
        timestamp = proposed
    elif date_resolution in {"section_date_heading", "relative_to_source_day"} and len(supported) == 1:
        timestamp = next(iter(supported))
    elif len(proposed) >= 7 and proposed[:7] in {v[:7] for v in supported}:
        # Never retain an unsupported exact day.
        timestamp = proposed[:7]
    elif len({d[:7] for d in supported}) == 1:
        timestamp = next(iter(supported))[:7]
    else:
        return None, "no_unambiguous_source_month"
    if not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", timestamp):
        return None, "invalid_normalized_date"
    original_years = set(re.findall(YEAR, original))
    quote_years = set(re.findall(rf"\b(?:in|during|year)\s+({YEAR})\b|({YEAR})年", occurrence_quote, re.I))
    explicit_quote_years = {y for pair in quote_years for y in pair if y}
    if (original_years and timestamp[:4] not in original_years) or (
        len(explicit_quote_years) == 1 and not date_parts(occurrence_quote)
        and timestamp[:4] not in explicit_quote_years
    ):
        return None, "explicit_event_year_conflict"
    # A date_original claim must itself occur in the source, not be model prose.
    proof = {
        "validator": VERSION,
        "status": "supported",
        "time_role": role,
        "method": (
            "source_expression" if date_parts(actual_quote) else "anchored_context"
        ),
        "section_path": section.get("path"),
        "quote_start": a,
        "quote_end": b,
        "date_context": bound_context,
        "section_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "anchor_year": anchor,
        "supported_dates": sorted(supported),
        "date_resolution": date_resolution,
        **({"source_date_heading": heading} if heading and date_resolution in {
            "section_date_heading", "relative_to_source_day"} else {}),
        **({"source_title": section["source_title"]} if section_year and not years and section.get("source_title") else {}),
        **({'occurrence_clause': {'quote': dated_assertion, 'start': a, 'end': a + len(dated_assertion)}}
           if dated_assertion else {}),
        "quote_resolution": (
            "ordered_ellipsis_restored"
            if re.search(r"\.{3}|…", quote) and actual_quote != quote
            else "literal_or_normalized"
        ),
    }
    return {
        **raw,
        "timestamp": timestamp,
        "date_precision": "day" if len(timestamp) == 10 else "month",
        "evidence_text": actual_quote,
        "date_evidence": proof,
    }, "accepted"


def retained_background(rejected: list[dict]) -> list[dict]:
    """Keep candidate quotes with coarse timing for source revalidation at export."""
    retained = []
    for row in rejected:
        candidate = row.get("candidate") or {}
        quote = str(candidate.get("evidence_text") or "")
        reason = row.get("reason")
        if not quote or len(quote) > 1600 or not candidate.get("source_sections"):
            continue
        if reason in {"no_unambiguous_source_month", "multiple_event_months_require_narrower_evidence",
                      "coverage_period_is_not_occurrence_month"}:
            status = "source_report"
        elif reason in {"announcement_act_not_supported", "planned_event_is_not_observed_occurrence",
                        "nonhistorical_or_unsupported_temporal_role"} and re.search(
            r"\b(?:plans? to|planned to|scheduled to|expects? to|intends? to)\b|计划|拟于|拟将|预计",
            quote, re.I,
        ):
            status = "reported_plan"
        else:
            continue
        original = str(candidate.get("date_original") or "")
        # Use only date wording found in the same quote. The model's proposed
        # timestamp and summary are intentionally excluded from this layer.
        span = locate_quote(quote, original) if original else None
        date_text = quote[span[0]:span[1]] if span else ""
        years = sorted(set(re.findall(rf"(?<!\d){YEAR}(?!\d)", date_text)))
        dates = sorted(date_parts(date_text, int(years[0]) if len(years) == 1 else None))
        if len(dates) > 1 and re.search(r"\b(?:between|to|through|from)\b|至|到|[–—]", date_text, re.I):
            precision, bounds = "range", [dates[0], dates[-1]]
        elif len(years) == 1 and not dates:
            precision, bounds = "year", years
        elif status == "reported_plan" and len(dates) == 1:
            precision, bounds = ("day" if len(dates[0]) == 10 else "month"), dates
        else:
            precision, bounds = "unknown", []
        retained.append({"quote": quote, "source_sections": candidate["source_sections"],
                         "fact_status": status, "date_original": date_text,
                         "date_precision": precision, "reported_bounds": bounds, "reason": reason})
    return retained


def validate_history(history: list[dict], sections: list[dict]):
    """Revalidate existing artifacts without a model call; never mutate input."""
    accepted, rejected = [], []
    for raw in history:
        candidates = [
            s for s in sections if s.get("path") in raw.get("source_sections", [])
        ]
        result, reason = None, "source_section_missing"
        for section in candidates:
            result, reason = validate_event(raw, section)
            if result:
                break
        if result:
            accepted.append(result)
        else:
            rejected.append(
                {
                    "timeline_id": raw.get("timeline_id"),
                    "timestamp": raw.get("timestamp"),
                    "description": raw.get("description"),
                    "reason": reason,
                }
            )
    return accepted, rejected
