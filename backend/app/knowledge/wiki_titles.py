"""Build Wikipedia timeline queries and judge search results from titles alone."""

import re

from app.knowledge.retrieval import terms

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DEADLINE = re.compile(
    r"\s+(?:before|by|until|through|no later than)\s+(?=(?:the\s+)?(?:\d|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|end|next|year[- ]end)\b))",
    re.I,
)
_TARGET_VERB = re.compile(r"\b(?:invade|visit|travel to|attack|strike)\b", re.I)
_DETAIL = re.compile(r"\b(?:casualties|reactions|rescue|fatalities|involvement|economic impact|environmental impact|opinion polling|endorsements)\b", re.I)
_QUALIFIERS = set("united general federal presidential parliamentary national local municipal final interim".split())
_TITLE_FORMAT = {"timeline", "chronology"}
_NOISE = set(
    "after before during through until which who what when where does did can could may whether "
    "would next first most least how happen occur held take close closes win wins result results "
    "end announce announces announced confirm confirmed latest party parties january february march april may june "
    "july august september october november december".split()
)
_EVENT_WORDS = (
    (("war", "conflict"), "war conflict invasion invade invades invaded hostilities"),
    (("election", "campaign"), "election elections elect elected ballot campaign primary primaries nomination"),
    (("visit", "trip"), "visit visits visited trip trips travel travelled traveled"),
    (("negotiations", "talks"), "negotiation negotiations talks summit meeting ceasefire"),
    (("acquisition", "deal"), "acquisition acquisitions merger mergers takeover"),
    (("agreement", "deal"), "agreement agreements deal treaty treaties accord accords pact"),
)
_ELECTED_OFFICES = {"senate", "mayor", "governor", "congress", "parliament"}


def _normalize(text: str) -> str:
    text = re.sub(r"(?<!\w)(?:U\.S\.?|USA|US)(?!\w)", "United States", text)
    return re.sub(r"(?<!\w)(?:U\.K\.?|UK)(?!\w)", "United Kingdom", text)


def _keywords(text: str) -> list[str]:
    return [word for word in terms(_normalize(text), limit=24)
            if word not in _NOISE and not word.isdigit()]


class TimelineTitles:
    def __init__(self, question: str, cutoff_year: int):
        self.topic = _normalize(_DEADLINE.split(question.strip(), maxsplit=1)[0])
        self.years = set(_YEAR.findall(self.topic))
        self.year = max(self.years, default=str(cutoff_year))
        self.words = _keywords(self.topic)
        self.labels, self.event_words = next(
            ((labels, set(aliases.split())) for labels, aliases in _EVENT_WORDS
             if set(self.words) & set(aliases.split())), ((), set())
        )
        if not self.labels and self.words and _ELECTED_OFFICES.intersection(self.words) and re.search(
            r"\b(?:win|wins|be elected)\b", self.topic, re.I
        ):
            self.labels, aliases = _EVENT_WORDS[1]
            self.event_words = set(aliases.split())
        if self.labels == ("acquisition", "deal"):
            self.event_words.add("deal")
        self.subject_words = set(self.words) - self.event_words
        self.anchors = self.subject_words - _QUALIFIERS or self.subject_words
        target = _TARGET_VERB.split(self.topic, maxsplit=1)
        self.target = (set(_keywords(target[1])) - self.event_words) if len(target) == 2 else set()
        self.details = {match[0].casefold() for match in _DETAIL.finditer(self.topic)}

    def queries(self) -> list[str]:
        base = " ".join([*self.words[:10], *sorted(self.years)])
        if not base:
            return []
        anchors = " ".join(word for word in self.words if word in self.subject_words)
        subject = " ".join(word for word in self.words if word in self.target) or anchors
        if self.labels:
            canonical, alternate = self.labels
            if canonical == "election" and self.years:
                _, _, descriptor = self.topic.partition(self.year)
                event_subject = " ".join(word for word in _keywords(descriptor) if word not in self.event_words)
                subject = event_subject or subject
            variants = [f"{self.year} {subject} {canonical}", f"{anchors} {alternate} {self.year}",
                        f"timeline {self.year} {subject} {canonical}"]
        else:
            variants = [f"{base} {self.year}" if not self.years else f"timeline {base}", f"history {base}"]
        return list(dict.fromkeys(" ".join(query.split()) for query in [base, *variants]))[:4]

    def assess(self, title: str) -> tuple[int, str]:
        words = set(_keywords(title))
        years = set(_YEAR.findall(title))
        if years and (self.years and not self.years & years or not self.years and int(max(years)) < int(self.year) - 1):
            return 0, "Title refers to a different period"
        if any(match[0].casefold() not in self.details for match in _DETAIL.finditer(title)):
            return 0, "Title covers a narrower subtopic"
        overlap = self.anchors & words
        if not overlap or self.target and not self.target <= words:
            return 0, "Title does not identify the event participants or subject"
        if self.event_words and not self.event_words & words:
            return 0, "Title does not identify the same type of event"
        if not self.event_words and len(overlap) < min(2, len(self.anchors)):
            return 0, "Title has insufficient overlap with the event"
        extra_scope = words - self.subject_words - self.event_words - _QUALIFIERS - _TITLE_FORMAT
        score = 10 + 2 * len(overlap) + 6 * (self.year in years) + 2 * bool(self.event_words & words)
        score -= min(4, len(extra_scope))
        return score, "Title matches the event and its subject"
