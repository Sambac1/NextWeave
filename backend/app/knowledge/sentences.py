"""Complete, source-located sentences; never manufacture a clipped ending."""
import re


def sentence_records(text: str) -> list[dict]:
    records, start = [], 0
    endings = re.finditer(r'[。！？!?][”’"\']*|\.[”’"\']*(?=\s|$)', text)
    for match in endings:
        prefix = text[start:match.end()]
        # Initials, common titles and decimals are not sentence endings.
        if match.group().startswith('.') and re.search(
            r'(?:\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc)|\b[A-Z]|(?:[A-Za-z]\.)+[A-Za-z])\.$',
            prefix,
        ):
            continue
        a = start + len(text[start:match.end()]) - len(prefix.lstrip())
        quote = text[a:match.end()].strip()
        if quote:
            records.append({'id': f's{a}_{match.end()}', 'quote': quote,
                            'start': a, 'end': match.end()})
        start = match.end()
    return records


def complete_excerpt(text: str, limit: int) -> str:
    selected, size = [], 0
    for row in sentence_records(text):
        quote = row['quote']
        if size + len(quote) + bool(selected) > limit:
            break
        selected.append(quote)
        size += len(quote) + (1 if len(selected) > 1 else 0)
    return ' '.join(selected)
