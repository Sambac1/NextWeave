from app.layers.pipeline.actor_cache import ActorBatchCache


def test_valid_empty_actor_batch_is_cached_but_provider_failure_is_not(tmp_path):
    cache = ActorBatchCache(tmp_path, "model", "https://example.org")
    entries = [{"timeline_id": "h1", "description": "A price observation"}]
    assert cache.load(entries) is None
    cache.save(entries, [(entries[0], [], [], {"status": "reasoning_only"})])
    assert cache.load(entries) is None
    cache.save(entries, [(entries[0], [], [], {"status": "ok"})])
    new_entries = [dict(entries[0])]
    hit = cache.load(new_entries)
    assert hit[0][0] is new_entries[0] and hit[0][1:3] == ([], [])
    assert ActorBatchCache(tmp_path, "different-model", "https://example.org").load(entries) is None
    assert cache.load([{"timeline_id": "h1", "description": "Changed evidence"}]) is None


def test_cache_accepts_the_real_stage_path_adapter(tmp_path):
    from app.layers.pipeline.stage import ArtifactDir
    from app.storage.json_store import EventRepository
    events = EventRepository(tmp_path)
    case = events.create("Test", ["https://example.org"], "")
    cache = ActorBatchCache(ArtifactDir(events, case["event_id"]), "model", None)
    entries = [{"timeline_id": "h", "description": "Evidence"}]
    cache.save(entries, [(entries[0], [], [], {"status": "ok"})])
    assert cache.load(entries) is not None
