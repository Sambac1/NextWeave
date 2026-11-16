import asyncio

import pytest

from app.knowledge.coverage import extract_coverage, MergeMembership
from app.knowledge.timeline import TimelineBatch


def sections(count=20):
    return [{"section_id": f"s{i}", "path": f"source_01/2020/section{i}", "title": str(i),
             "content": f"2020-01-{i+1:02d}: unique_item_{i} occurred."} for i in range(count)]


class Client:
    model = "test"
    calls = 0
    fail = False
    async def json_call(self, **kw):
        if kw["schema"] is MergeMembership:
            raise AssertionError("Unique events need no semantic model pass")
        self.calls += 1
        if self.fail and "s2_" in kw["call_name"]:
            raise RuntimeError("Chunk failed")
        import re
        number = int(re.search(r"section_s(\d+)", kw["call_name"])[1])
        return TimelineBatch(events=[{"timestamp": f"2020-01-{number+1:02d}",
            "description": f"unique_item_{number}", "evidence_text": f"unique_item_{number} occurred."}])


def test_all_eligible_sections_are_processed_and_reused(tmp_path):
    client = Client()
    data = sections() + [{"section_id": "refs", "path": "source_01/References", "content": "References"}]
    outcome = asyncio.run(extract_coverage(data, client, tmp_path, lambda *_: None, planning_mode="rules"))
    assert len(outcome.historical) == 20 and client.calls == 20
    again = asyncio.run(extract_coverage(data, client, tmp_path, lambda *_: None, planning_mode="rules"))
    assert len(again.historical) == 20 and client.calls == 20
    assert again.stats["chunk_cache_hits"] == 20


def test_failed_chunk_does_not_become_a_false_empty_result(tmp_path):
    client = Client(); client.fail = True
    with pytest.raises(ValueError, match="chunks failed"):
        asyncio.run(extract_coverage(sections(4), client, tmp_path, lambda *_: None, planning_mode="rules"))
    assert client.calls == 4
    client.fail = False
    result = asyncio.run(extract_coverage(sections(4), client, tmp_path, lambda *_: None, planning_mode="rules"))
    assert len(result.historical) == 4 and client.calls == 5
