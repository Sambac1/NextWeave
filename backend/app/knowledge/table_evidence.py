"""Conservative, source-preserving evidence from MediaWiki tables.

Cell spans are semantic structure, not styling. Unsupported or ambiguous tables
remain quotable raw evidence; this module never fills in missing column names.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import html
import re


_TABLE = re.compile(r"(?ms)^[ \t]*\{\|[^\n]*(?:\n|$).*?(?:^[ \t]*\|\}[^\n]*(?:\n|$)|\Z)")
_ATTR = re.compile(r'''([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s]+))''')
_MAX_COLUMNS = 128


def _id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:20]


@dataclass
class _Cell:
    value: str
    header: bool
    rowspan: int = 1
    colspan: int = 1
    scope: str = ""
    problems: list[str] = field(default_factory=list)
    row: int = 0
    column: int = 0


@dataclass
class _Row:
    start: int
    end: int
    prefix_start: int
    cells: list[_Cell]
    problems: list[str] = field(default_factory=list)


def _inline_cells(value: str, header: bool) -> list[str]:
    """Split inline cells without splitting pipes inside links or templates."""
    result, start, index, links, templates = [], 0, 0, 0, 0
    while index < len(value):
        pair = value[index:index + 2]
        if pair == "[[":
            links += 1
        elif pair == "]]" and links:
            links -= 1
        elif pair == "{{":
            templates += 1
        elif pair == "}}" and templates:
            templates -= 1
        elif not links and not templates and (pair == "||" or header and pair == "!!"):
            result.append(value[start:index])
            start = index + 2
        else:
            index += 1
            continue
        index += 2
    result.append(value[start:])
    return result


def _cell(value: str, header: bool) -> _Cell:
    attrs = {}
    problems = []
    prefix, separator, rest = value.partition("|")
    if separator and "=" in prefix:
        matches = list(_ATTR.finditer(prefix))
        residue = _ATTR.sub("", prefix).strip()
        if matches and not residue:
            for match in matches:
                key = match[1].lower()
                if key in attrs:
                    problems.append("duplicate_cell_attribute")
                attrs[key] = next(part for part in match.groups()[1:] if part is not None)
            value = rest
        elif re.search(r"\b(?:rowspan|colspan|scope)\s*=", prefix, re.I):
            problems.append("malformed_cell_attributes")
    spans = {}
    for name in ("rowspan", "colspan"):
        raw = attrs.get(name, "1")
        if not raw.isdecimal() or not 1 <= int(raw) <= _MAX_COLUMNS:
            problems.append("invalid_" + name)
            spans[name] = 1
        else:
            spans[name] = int(raw)
    scope = attrs.get("scope", "").lower()
    if scope not in ("", "row", "col", "rowgroup", "colgroup"):
        problems.append("unsupported_scope")
    # Do not attempt to expand templates or reinterpret nested/complex markup.
    if re.search(r"\{\{|\}\}|<\s*(?:table|nowiki|math|script)\b", value, re.I):
        problems.append("unsupported_cell_markup")
    value = re.sub(r"\[\[([^\[\]]+)\]\]", lambda match: match[1].split("|", 1)[-1], value)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"</?(?:span|small|b|i|strong|em|sup|sub)\b[^>]*>", "", value, flags=re.I)
    if "<" in value or ">" in value:
        # Comparisons such as '> 10' are data, not an unknown HTML element.
        if re.search(r"</?[A-Za-z][^>]*>", value):
            problems.append("unsupported_cell_markup")
    value = " ".join(html.unescape(value.replace("'''", "").replace("''", "")).split())
    return _Cell(value, header, scope=scope, problems=problems, **spans)


def _read_rows(table: str) -> tuple[list[_Row], list[str]]:
    rows, problems = [], []
    lines = list(re.finditer(r"[^\n]*(?:\n|$)", table))
    if not lines or not lines[0][0].lstrip().startswith("{|"):
        return [], ["missing_table_start"]
    if len(re.findall(r"(?m)^[ \t]*\{\|", table)) != 1:
        problems.append("nested_table")
    if not re.search(r"(?m)^[ \t]*\|\}", table):
        problems.append("unclosed_table")
    cells, row_problems = [], []
    start = prefix_start = lines[0].end()

    def finish(end):
        if cells:
            rows.append(_Row(start, end, prefix_start, cells.copy(), row_problems.copy()))

    for line in lines[1:]:
        stripped = line[0].lstrip()
        if stripped.startswith(("|-", "|}")):
            finish(line.start())
            cells, row_problems = [], []
            start, prefix_start = line.end(), line.start()
            if stripped.startswith("|}"):
                break
        elif stripped.startswith("|+"):
            if cells:
                row_problems.append("caption_inside_row")
            else:
                start = prefix_start = line.end()
        elif stripped.startswith(("!", "|")):
            header = stripped[0] == "!"
            cells.extend(_cell(value, header) for value in _inline_cells(stripped[1:].rstrip("\r\n"), header))
        elif stripped.strip():
            if cells:
                continuation = _cell(cells[-1].value + " " + stripped, cells[-1].header)
                cells[-1].value = continuation.value
                cells[-1].problems.extend(continuation.problems)
            else:
                row_problems.append("unsupported_row_content")
    else:
        finish(len(table))
    return rows, problems


def _analyze(table: str) -> tuple[dict, list[_Row], int]:
    rows, structural = _read_rows(table)
    table_id = _id("table_", table)
    result = {"table_id": table_id, "rows": [], "ambiguity": [], "original_quote": table}
    header_count = 0
    for row in rows:
        if all(cell.header and cell.scope not in ("row", "rowgroup") for cell in row.cells):
            header_count += 1
        else:
            break
    if not header_count:
        structural.append("missing_column_headers")
    if header_count == len(rows):
        structural.append("missing_data_rows")
    grid = {}
    for r, row in enumerate(rows):
        column = 0
        structural.extend(row.problems)
        for cell in row.cells:
            structural.extend(cell.problems)
            while (r, column) in grid:
                column += 1
            cell.row, cell.column = r, column
            if column + cell.colspan > _MAX_COLUMNS:
                structural.append("too_many_columns")
                break
            if r + cell.rowspan > len(rows):
                structural.append("rowspan_outside_table")
            if r < header_count < r + cell.rowspan:
                structural.append("header_spans_into_data")
            for rr in range(r, min(r + cell.rowspan, len(rows))):
                for cc in range(column, column + cell.colspan):
                    if (rr, cc) in grid:
                        structural.append("overlapping_spans")
                    grid[rr, cc] = cell
            column += cell.colspan
    width = max((column + 1 for row, column in grid if row < header_count), default=0)
    if any({column for rr, column in grid if rr == r} != set(range(width)) for r in range(header_count)):
        structural.append("inconsistent_header_width")
    paths = []
    for column in range(width):
        path, seen = [], set()
        for r in range(header_count):
            cell = grid.get((r, column))
            if cell is not None and id(cell) not in seen:
                seen.add(id(cell))
                if cell.value:
                    path.append(cell.value)
        paths.append(tuple(path))
    repeated = Counter(paths)
    column_issues = {}
    for column, path in enumerate(paths):
        if not path:
            column_issues[column] = "unlabeled_column:" + str(column + 1)
        elif repeated[path] > 1:
            column_issues[column] = "duplicate_column_path:" + str(column + 1)
    structural = list(dict.fromkeys(structural))
    result["ambiguity"] = structural + list(column_issues.values())
    for r, row in enumerate(rows[header_count:], header_count):
        issues = structural.copy()
        occupied = {column for rr, column in grid if rr == r}
        if occupied != set(range(width)):
            issues.append("row_width_mismatch")
        if all(cell.header and cell.scope not in ("row", "rowgroup") for cell in row.cells):
            issues.append("header_row_inside_data")
        inherited_start = min((rows[cell.row].start for (rr, _), cell in grid.items() if rr == r and cell.row >= header_count), default=row.start)
        quote = table[inherited_start:row.end].rstrip("\r\n")
        row_id = _id("row_", table_id + ":" + str(r) + ":" + quote)
        fields = []
        if not issues:
            for column in range(width):
                cell = grid[r, column]
                if column in column_issues:
                    issues.append(column_issues[column])
                elif cell.colspan > 1:
                    issues.append("merged_data_cell:" + str(column + 1))
                else:
                    path = list(paths[column])
                    fields.append({"field_id": _id("field_", row_id + ":" + repr(path)),
                                   "column_path": path, "value": cell.value})
        result["rows"].append({"row_id": row_id, "fields": fields, "original_quote": quote,
                               "ambiguity": list(dict.fromkeys(issues))})
    return result, rows, header_count


def parse_table_evidence(text: str) -> list[dict]:
    """Return labeled rows, exact source quotes and explicit ambiguity reasons.

Labels and values remain strings, including units. Unique, known paths may be
returned alongside ambiguous columns. Structural errors suppress all fields.
    """
    return [_analyze(match[0])[0] for match in _TABLE.finditer(text)]


def prose_spans(text: str) -> list[str]:
    """Return the original text outside raw Wiki tables."""
    spans, start = [], 0
    for match in _TABLE.finditer(text):
        spans.append(text[start:match.start()].strip())
        start = match.end()
    spans.append(text[start:].strip())
    return [span for span in spans if span]


def table_spans(text: str) -> list[str]:
    """Return raw Wiki tables with their captions, headers and data intact."""
    return [match[0].strip() for match in _TABLE.finditer(text)]


def table_passages(table: str, max_chars: int = 2300) -> list[str]:
    """Repeat the complete raw header; never split a row or an active rowspan.

The bound is soft when one indivisible header/row group exceeds it. Returning
that evidence intact is preferable to silently deleting it or its schema.
    """
    if len(table) <= max_chars:
        return [table]
    parsed, rows, header_count = _analyze(table)
    if (not rows or not header_count or header_count == len(rows)
            or any(not issue.startswith(("unlabeled_column:", "duplicate_column_path:"))
                   for issue in parsed["ambiguity"])
            or any("row_width_mismatch" in row["ambiguity"] for row in parsed["rows"])):
        return [table]
    header = table[:rows[header_count].prefix_start]
    closing_match = re.search(r"(?m)^[ \t]*\|\}", table)
    if closing_match is None:
        return [table]
    closing = table[closing_match.start():]
    groups, start, through = [], header_count, header_count
    for r in range(header_count, len(rows)):
        through = max(through, *(r + cell.rowspan - 1 for cell in rows[r].cells))
        if r == through:
            groups.append((start, r))
            start = r + 1
    passages, chunk_start, chunk_end = [], None, None
    for first, last in groups:
        candidate_start = first if chunk_start is None else chunk_start
        candidate = header + table[rows[candidate_start].prefix_start:rows[last].end] + closing
        if chunk_start is not None and len(candidate) > max_chars:
            passages.append(header + table[rows[chunk_start].prefix_start:rows[chunk_end].end] + closing)
            chunk_start = first
        elif chunk_start is None:
            chunk_start = first
        chunk_end = last
    if chunk_start is not None:
        passages.append(header + table[rows[chunk_start].prefix_start:rows[chunk_end].end] + closing)
    return passages or [table]
