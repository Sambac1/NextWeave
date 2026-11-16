"""Search signals from the qualifying action in a binary contract."""
import re

from app.knowledge.retrieval import terms
from app.knowledge.sentences import sentence_records

_YES = re.compile(r"""\byes\b["”']?\s+if\s+""", re.I)
_TIME_LIMIT = re.compile(
    r"\s+(?:by|before|through|until|no later than)\s+(?=(?:the\s+)?(?:\d|[A-Z][a-z]+\s+\d))", re.I
)
_CONTROL = re.compile(r"\b(?:territorial control|control over|occup(?:y|ies|ation)|inva(?:de|des|sion))\b", re.I)
_MILITARY = re.compile(r"\b(?:military|armed|ground)\s+(?:offensive|operation|invasion)\w*\b", re.I)
_INITIATES = re.compile(r"\b(?:commenc\w*|launch\w*|order\w*|authori[sz]\w*|initiat\w*|inva(?:de|des))\b", re.I)
_AUTHORITY = re.compile(r"\b(?:commander[ -]in[ -]chief|chain of command|command structure|"
                        r"operational command|command authority|war powers)\b", re.I)
_DIRECT = re.compile(r"\b(?:ground (?:troops|forces|operations?|offensives?)|territorial control|"
                     r"military offensive|control over|inva(?:de|des|sion)|occup(?:y|ation))\b", re.I)
_CURRENT = re.compile(r"\b(?:war|strikes?|attacks?|ceasefire|blockade|deploy\w*|negotiat\w*|"
                      r"diploma\w*|talks|troops|operations?)\b", re.I)
_DIPLOMACY = re.compile(r"\b(?:ceasefire|negotiat\w*|peace talks|diplomatic talks|mediat\w*)\b", re.I)
_ADVICE = re.compile(r"\b(?:advis\w*|consult\w*|recommend\w*|discuss\w*|report\w*)\b", re.I)
_US = re.compile(r"\b(?:U\.S\.?|USA|US|American)(?!\w)")
_YEAR = re.compile(r"\b(?:19|20)\d{2}(?!\d)")
_MONTH = re.compile(r"\b(?:July|August|September|October|November|December)\b", re.I)


def last_named_year(text: str) -> int:
    return max((int(value) for value in _YEAR.findall(text)), default=0)


def positive_condition(rules: str) -> str:
    for statement in sentence_records(rules):
        if match := _YES.search(statement["quote"]):
            return _TIME_LIMIT.split(statement["quote"][match.end():], maxsplit=1)[0].strip().rstrip(".")
    return ""


def territorial_offensive(rules: str) -> bool:
    condition = positive_condition(rules)
    return bool(_MILITARY.search(condition) and _CONTROL.search(condition))


def initiates_offensive(description: str) -> bool:
    action = _INITIATES.search(description)
    return bool(action and _CONTROL.search(description) and not _ADVICE.search(description[:action.start()]))


class TerritorialOffensiveFocus:
    def __init__(self, question: str, rules: str, year: int):
        self.condition = positive_condition(rules)
        self.year = year
        action = _INITIATES.search(self.condition)
        self.subject = re.sub(r"^the\s+", "", self.condition[:action.start()], flags=re.I).strip() if action else ""
        self.subject_words = set(terms(self.subject))
        self.subject_pattern = re.compile(r"(?<!\w)" + re.escape(self.subject) + r"(?!\w)", re.I)
        condition_words = set(terms(self.condition))
        self.topic = [word for word in terms(question, limit=20)
                      if word in condition_words and word not in self.subject_words
                      and word.isalpha() and word not in {"us", "before", "invade", "invasion", "military", "offensive"}]
        self.enabled = bool(self.subject_words and self.topic and _MILITARY.search(self.condition) and _CONTROL.search(self.condition))

    def matches_topic(self, text: str) -> bool:
        words = set(terms(text))
        return any(word in words or word + "ian" in words or word + "an" in words for word in self.topic)

    def matches_subject(self, text: str) -> bool:
        if self.subject.casefold() == "united states":
            text = _US.sub("United States", text)
        return bool(self.subject_pattern.search(text))

    def authority_title(self, title: str) -> bool:
        return bool(self.matches_subject(title) and
                    (_AUTHORITY.search(title) or re.search(r"\b(?:president|armed forces|department of defense)\b", title, re.I)))

    def passage_score(self, title: str, heading: str, text: str, *, section_year=0) -> int:
        scope = f"{title} {heading}"
        year = last_named_year(f"{heading} {text}") or section_year or last_named_year(title)
        if year and year < self.year - 1:
            return 0
        if self.authority_passage(title, heading, text):
            return 9
        if not self.matches_topic(text) and not self.matches_topic(scope):
            return 0
        direct = _DIRECT.search(text) and any(
            _DIRECT.search(row["quote"]) and self.matches_subject(row["quote"]) for row in sentence_records(text)
        )
        score = 6 * bool(direct) + 3 * bool(_CURRENT.search(text))
        if not score:
            return 0
        if re.search(r"\b(?:rescue|hostage|casualties|claims tribunal)\b", title, re.I):
            return 1 if year >= self.year - 1 else 0
        return score + 2 * (year == self.year) + 2 * bool(year == self.year and _MONTH.search(heading))

    def authority_passage(self, title: str, heading: str, text: str) -> bool:
        if not _AUTHORITY.search(text):
            return False
        if self.authority_title(f"{title} {heading}"):
            return True
        return any(_AUTHORITY.search(row["quote"]) and self.matches_subject(row["quote"])
                   and re.search(r"\b(?:president|prime minister|government|head of state|defen[cs]e minister)\b", row["quote"], re.I)
                   for row in sentence_records(text))

    def diplomacy_passage(self, title: str, heading: str, text: str) -> bool:
        return bool(_DIPLOMACY.search(f"{heading} {text}") and
                    self.matches_topic(f"{title} {text}"))
