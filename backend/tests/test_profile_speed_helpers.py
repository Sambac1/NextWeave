from app.layers.pipeline.llm_figure_service import EXTRACT_BATCH_SIZE
from app.layers.pipeline.profile_service import (
    _kind_from_entity_type,
    _role_from_generation,
    _role_from_profile,
)


def test_extract_batch_size_fits_deepseek_output():
    assert EXTRACT_BATCH_SIZE <= 4


def test_kind_from_entity_type():
    assert _kind_from_entity_type("person") == "individual"
    assert _kind_from_entity_type("government") == "institution"
    assert _kind_from_entity_type("group") == "collective"
    assert _kind_from_entity_type("unknown-label") == "institution"


def test_role_from_profile_prefers_explicit_label():
    assert _role_from_profile({"role": "initiator"}) == "initiator"
    assert _role_from_profile({"role_in_event": "弹劾总统"}) == "opponent"
    assert _role_from_profile({"summary": "no clear part"}) == "neutral"


def test_role_from_generation_overrides_fallback():
    assert _role_from_generation({"role_label": "mediator"}, "neutral") == "mediator"
    assert _role_from_generation({"semantic": {}}, "supporter") == "supporter"
