"""Prepare a forecast from a question or market rule record."""
from datetime import date, datetime, time, timezone
import re

import httpx

from app.benchmark.dynamics import contract_state, scheduled_round_dates, simulation_case
from app.knowledge.wikipedia import discover_wikipedia
from app.benchmark.markets import resolve_contract
from app.knowledge.forecast_evidence import frozen_sources
from app.knowledge.public_evidence import public_sources
from app.knowledge.sentences import sentence_records
from app.domain.forecast_task import ForecastConfirmation, ForecastSpec
from app.knowledge.brief import forecast_brief_preview
from app.settings import get_settings
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository, GraphRepository

_ABSOLUTE_DATE = re.compile(r"\b(by|before|through)\s+(\d{4}-\d{2}-\d{2})\b", re.I)
_BINARY_QUESTION = re.compile(
    r"^(?:will|can|could|is|are|does|do|did|has|have|would|was|were|may|had)\b"
    r"|\bwhether\b|是否|能否|可否|会不会|能不能|有没有|是不是", re.I,
)
_ABSENCE_EVIDENCE = " When the answer depends on nonoccurrence, the evidence must cover the full forecast window."


def preview_store() -> EnsembleStore:
    return EnsembleStore(get_settings().data_dir / "forecast_previews")


async def preview_question(query: str, client: httpx.AsyncClient) -> ForecastSpec:
    query = query.strip()
    if re.match(r"^(?:[a-z]+://|(?:www\.)?polymarket\.com/)", query, re.I):
        contract = await resolve_contract(client, query)
        return ForecastSpec.model_validate(simulation_case(contract) | {
            "kind": "market", "market_rules_at": contract["market_rules_at"],
        })
    start = datetime.now(timezone.utc)
    deadlines = set(_ABSOLUTE_DATE.findall(query))
    end = None
    if len(deadlines) == 1:
        try:
            qualifier, value = deadlines.pop()
            end = datetime.combine(date.fromisoformat(value),
                                   time.min if qualifier.casefold() == "before" else time(23, 59, 59),
                                   timezone.utc)
        except ValueError:
            pass
    if end and end <= start:
        end = None
    yes, no = question_criteria(query) if _BINARY_QUESTION.search(query) else ("", "")
    return ForecastSpec(
        kind="question", question=query, as_of=start, outcome_deadline=end,
        deadline_basis="question_date_utc" if end else "",
        yes_criterion=yes, no_criterion=no,
    )


def question_criteria(question: str, yes: str | None = None, no: str | None = None) -> tuple[str, str]:
    yes, no = (yes or "").strip(), (no or "").strip()
    if yes and no:
        if " ".join(yes.casefold().split()) == " ".join(no.casefold().split()):
            raise ValueError("Enter distinct conditions for Yes and No.")
        return yes, no
    if yes:
        return yes, "the evidence establishes that the stated Yes condition was false within the forecast window." + _ABSENCE_EVIDENCE
    if no:
        return "the evidence establishes that the stated No condition was false within the forecast window." + _ABSENCE_EVIDENCE, no
    if not _BINARY_QUESTION.search(question):
        raise ValueError("Use a yes/no question to generate both conditions, or enter at least one observable condition.")
    question = " ".join(question.split())
    return (
        f"the evidence establishes a Yes answer to “{question}” within the forecast window." + _ABSENCE_EVIDENCE,
        f"the evidence establishes a No answer to “{question}” within the forecast window." + _ABSENCE_EVIDENCE,
    )


def source_milestones(spec: ForecastSpec, sources: list[dict]) -> list[dict]:
    milestones = {}
    for source in sources:
        for sentence in sentence_records(source["content"]):
            quote = sentence["quote"]
            if not re.search(r"\b(?:scheduled|planned|set to|set for)\b", quote, re.I):
                continue
            for day in scheduled_round_dates(quote):
                if spec.as_of.date() < day < spec.outcome_deadline.date():
                    milestones.setdefault(day, {
                        "date": day.isoformat(), "quote": quote[:600],
                        "url": source.get("revision_url", source["url"]),
                    })
    return [milestones[day] for day in sorted(milestones)[:5]]


async def create_forecast_case(request: ForecastConfirmation, client: httpx.AsyncClient, *,
                               previews: EnsembleStore, events: EventRepository) -> dict:
    stored = previews.load(request.preview_id, "preview")
    if not stored:
        raise FileNotFoundError("Forecast preview not found. Review the question again.")
    previews.acquire(request.preview_id)
    try:
        spec = ForecastSpec.model_validate(stored)
        deadline = request.outcome_deadline or spec.outcome_deadline
        if not deadline or deadline <= spec.as_of:
            raise ValueError("Choose a forecast deadline after the evidence cutoff, with a time zone.")
        if spec.kind == "market":
            if spec.deadline_basis == "explicit_contract_deadline" and deadline != spec.outcome_deadline:
                raise ValueError("The confirmed deadline must match the explicit contract rule.")
        else:
            yes, no = question_criteria(spec.question, request.yes_criterion, request.no_criterion)
            spec = spec.model_copy(update={
                "yes_criterion": yes, "no_criterion": no, "outcome_window_start": spec.as_of,
                "outcome_window_basis": "forecast_starts_at_evidence_cutoff",
            })
        if request.outcome_deadline and deadline != spec.outcome_deadline:
            spec = spec.model_copy(update={"deadline_basis": "user_confirmed_deadline"})
        spec = spec.model_copy(update={"outcome_deadline": deadline.astimezone(timezone.utc)})
        confirmed = spec.model_dump(mode="json", exclude_none=True)
        previous = previews.load(request.preview_id, "confirmed")
        if previous:
            if previous["spec"] != confirmed or previous["seeds"] != request.seed_wiki_urls:
                raise ValueError("This preview already created an analysis. Review the question again to change it.")
            events.directory(previous["result"]["event_id"])
            return previous["result"]

        case = spec.contract_case()
        pages, discovery = await discover_wikipedia(
            client, question=spec.question, rules=case["rules"], as_of=spec.as_of,
            seed_urls=request.seed_wiki_urls, max_pages=8, title_discovery=True, max_link_attempts=8,
            include_search_with_seeds=True,
        )
        sources = frozen_sources(case | {"evidence": pages}, "expanded", require_relevant=True)
        if not sources:
            raise ValueError("No relevant Wikipedia passages were available. Add a relevant Wikipedia seed page.")
        by_url = {row["url"]: row for row in pages}
        for row in sources:
            page = by_url[row["url"]]
            row.update({key: page[key] for key in ("revision_url", "retrieved_at", "revision_sha256", "discovered_from") if key in page})
        sources = [source | packet for source, packet in zip(sources, public_sources(sources))]
        event = events.create(spec.question, [row["url"] for row in sources], None)
        event_id = event["event_id"]
        events.save_sources(event_id, sources)
        brief = forecast_brief_preview(confirmed, sources)
        background = brief["text"]
        events.save_artifact(event_id, "background_brief", brief)
        events.save_artifact(event_id, "background", background)
        events.save_artifact(event_id, "forecast_spec", confirmed)
        events.save_artifact(event_id, "scheduled_milestones", source_milestones(spec, sources))
        events.save_artifact(event_id, "decision_context", {
            "question": spec.question, "rules": case["rules"], "as_of": case["as_of"], "deadline": case["outcome_deadline"],
        })
        events.save_artifact(event_id, "ingestion_manifest", {"source_count": len(sources), "discovery": discovery})
        GraphRepository(events).add_node(event_id, {"id": event_id, "label": spec.question, "type": "root_event"})
        result = {
            **event, "forecast_spec": confirmed, "background": background,
            "scraped_articles": [{key: value for key, value in row.items() if key != "content"} for row in sources],
        }
        previews.save(request.preview_id, "confirmed", {"spec": confirmed, "seeds": request.seed_wiki_urls, "result": result})
        return result
    finally:
        previews.release(request.preview_id)


def question_contract(spec: ForecastSpec, sources: list[dict]) -> dict:
    contract = contract_state(spec.contract_case(), sources, include_projection=False)
    if spec.market_url:
        contract["market_url"] = spec.market_url
        contract["resolution_source"] = spec.resolution_source
    contract["forecast_criteria"] = {
        "yes": spec.yes_criterion or next((row['quote'] for row in sentence_records(contract['rules'])
            if re.search(r'resolve\w*\s+to\s+[\"“]?Yes[\"”]?\s+if\b', row['quote'], re.I)), contract['rules']),
        "no": spec.no_criterion or next((row['quote'] for row in sentence_records(contract['rules'])
            if re.search(r'resolve\w*\s+to\s+[\"“]?No\b', row['quote'], re.I)),
            "The exact Yes condition is not met by the stated deadline under the frozen rules."),
    }
    return contract


def prepare_question_experiment(request, events: EventRepository):
    from app.simulation.question_schedule import simulation_round_plan
    from app.domain.ensemble import EnsembleRequest, MainQuestionSimulation, OutcomeDefinition
    from app.knowledge.decision_seats import actor_coverage

    event_id = request.event_id
    raw = events.load_artifact(event_id, "forecast_spec", None)
    if not raw:
        raise ValueError("This analysis has no confirmed forecast question.")
    spec = ForecastSpec.model_validate(raw)
    case = spec.contract_case()
    sources = events.load_sources(event_id)
    if not sources:
        raise ValueError("No frozen Wikipedia sources are available for this question.")
    contract = question_contract(spec, sources)
    coverage = actor_coverage(
        events.load_artifact(event_id, "decision_seat_plan", {}),
        events.load_artifact(event_id, "agents", []), case["as_of"], require_bdi=True, rules=case["rules"],
    )
    if coverage.get("status") == "observation_only":
        raise ValueError("This question requires a supported observation or environment mechanism.")
    if missing := coverage.get("missing_essential"):
        raise ValueError(f"Essential decision lacks a verified actor: {missing[0]} Add Wikipedia evidence about who can make that decision.")
    if dependencies := coverage.get("decision_dependencies"):
        contract["decision_dependencies"] = dependencies
    contract["source_milestones"] = events.load_artifact(event_id, "scheduled_milestones", [])
    plan = simulation_round_plan(case, "trajectory_sampling", request.steps, contract, keep_milestones=True)
    return EnsembleRequest(
        **request.model_dump(exclude_unset=True, exclude={"event_id", "steps"}),
        event_id=event_id, event_title=spec.question,
        start_date=plan["boundaries"][0], steps=plan["steps"], step_unit="day", planning_mode="rules",
        scenario_description=spec.question, decision_policy="weighted",
        initial_world_state={"benchmark_contract": contract},
        outcome_definitions=[OutcomeDefinition(id=key, label=key.title(), criterion=value)
                             for key, value in contract["forecast_criteria"].items()],
        question_simulation=MainQuestionSimulation(
            algorithm="conditional-stratified-v2",
            round_boundaries=plan["boundaries"], resolution_rules=case["rules"],
            as_of=case["as_of"], outcome_deadline=case["outcome_deadline"],
            sequential_outcome=True, prompt_profile="compact",
        ),
    )
