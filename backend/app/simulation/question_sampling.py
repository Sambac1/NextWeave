"""Public replay and probability aggregation for weighted question forecasts."""
import math

from app.knowledge.okf import digest
from app.simulation.outlook import SEQUENTIAL_PROTOCOL
from app.simulation.terminal_outcomes import public_measurement_events


def public_replay(record):
    from app.simulation.joint_settlement import public_resolution
    events_by_step = {}
    registries = {}
    public_registry = {}
    rejected_triggers = {issue.get("round_id") for issue in (record.get("snapshot") or {}).get("validation_issues", [])
                         if issue.get("code") == "pending_annotation_rejected"}

    def public_choices(frame):
        waited = {action["agent_id"] for action in frame.get("agent_actions", [])
                  if action.get("agent_id") and action.get("decision") in {"wait", "monitor", "no_action"}}
        return [choice | ({"waited": True} if choice.get("actor_id") in waited else {})
                for choice in frame.get("sampled_choices", [])]

    for frame in record.get("rounds", []):
        for action in [*frame.get("agent_actions", []), *frame.get("intervention_actions", [])]:
            if action.get("action_id") and action.get("visibility", "public") == "public":
                public_registry[action["action_id"]] = action.get("agent_name", "")
        registries[frame["step"]] = dict(public_registry)
    for frame, event in public_measurement_events(record).values():
        events_by_step.setdefault(frame["step"], []).append({
            "id": event["timeline_id"], "text": event["description"], "date": event["timestamp"],
        })
    return [
        {"step": frame["step"], "interval": frame.get("simulation_interval"),
         **public_resolution(frame, (record.get("snapshot") or {}).get("world_state", {}).get("question_conditions", []),
                             registries[frame["step"]]),
         "events": events_by_step.get(frame["step"], []),
         **({"sampled_choices": public_choices(frame)} if frame.get("sampled_choices") else {}),
         **({"deliberation_count": len(frame.get("active_agent_ids", []))}
            if frame.get("activation_decisions") else {}),
         **({"pending_trigger_rejected": True} if frame["step"] in rejected_triggers else {}),
         **({"contract_outlook": frame["contract_outlook"]} if frame.get("contract_outlook") else {}),
         **({"outlook_issue": frame["outlook_issue"]} if frame.get("outlook_issue") else {}),
         **({"state_changes": [{"path": change["path"], "reason": change["reason"]}
                               for change in frame["state_changes"]]} if frame.get("state_changes")
            and not (frame.get("simulation_interval") or {}).get("sequential_harness") else {}),
         **({"contract_observation": frame["contract_observation"]} if frame.get("contract_observation") else {})}
        for frame in record.get("rounds", [])
    ]


def _valid_probability(value, *, positive=False):
    return type(value) in (int, float) and 0 <= value <= 1 and (not positive or value > 0)


def _recorded_draws(frames, expected_steps):
    """Verify the actual sampled decisions, including deliberate wait actions."""
    if not frames or (expected_steps is not None and len(frames) != expected_steps):
        return None
    draws = []
    for step, frame in enumerate(frames, 1):
        choices = frame.get("sampled_choices") or []
        if frame.get("step") != step or not all(
            _valid_probability(choice.get("p_selected"), positive=True) for choice in choices
        ):
            return None
        if expected_steps is not None:
            actor_ids = [choice.get("actor_id") for choice in choices]
            actions = [action.get("agent_id") for action in frame.get("agent_actions", [])]
            active = frame.get("active_agent_ids", [])
            if (any(not actor for actor in actor_ids) or any(not choice.get("candidate_id") for choice in choices)
                    or len(actor_ids) != len(set(actor_ids)) or len(actions) != len(actor_ids)
                    or len(active) != len(actor_ids) or set(actor_ids) != set(actions) or set(actor_ids) != set(active)):
                return None
            if not choices and frame.get("decision_sampling", {}).get("audited_silent_round") != 1:
                # Older recordings allowed a quiet round within an otherwise
                # valid trajectory. Only new audited silence is a full trace.
                if (frame.get("simulation_interval") or {}).get("sequential_harness"):
                    return None
        draws.extend(choice["p_selected"] for choice in choices)
    return draws if draws or all(frame.get("decision_sampling", {}).get("audited_silent_round") == 1 for frame in frames) else None


def _sequential_diagnostics(records, trajectories):
    """Describe recorded simulation evidence without judging or changing estimates.

    Coverage here concerns simulator records, never real-world outcome verification.
    Silent rounds, identical estimates and narrow actor casts may all be legitimate.
    """
    samples, active_actors, available_actors, public_paths = [], set(), set(), set()
    for record, trajectory in zip(records, trajectories):
        frames = record.get("rounds", [])
        available_actors.update(actor["agent_id"] for actor in (record.get("snapshot") or {}).get("agents", [])
                                if actor.get("agent_id"))
        actors = {actor for frame in frames for actor in frame.get("active_agent_ids", [])}
        actors.update(choice["actor_id"] for frame in frames for choice in frame.get("sampled_choices", [])
                      if choice.get("actor_id"))
        active_actors.update(actors)
        public_events = {ref: (frame, event) for ref, (frame, event) in public_measurement_events(record).items()
                         if event.get("origin") != "historical"}
        substantive = {ref: (frame, event) for ref, (frame, event) in public_events.items()
                       if event.get("event_type") != "scheduled_window"
                       and event.get("change_kind") != "continuation"}
        observed = False
        for frame in frames:
            observation = frame.get("contract_observation") or {}
            outlook = frame.get("contract_outlook") or {}
            if observation.get("matched_ref") in substantive or (
                outlook.get("estimate_kind") == "verified_simulation" and outlook.get("evidence_ref") in substantive
            ):
                observed = True
            if observation.get("answer") in {"yes", "no"} and (
                observation.get("event_ref") in substantive or (
                    observation.get("kind") == "programmatic_count" and observation.get("seats")
                    and any(event_frame is frame for event_frame, _ in substantive.values())
                )
            ):
                observed = True
        valid = trajectory["p_yes"] is not None
        row = {
            "sample_id": trajectory["sample_id"], "recorded_rounds": len(frames),
            "active_rounds": sum(bool(frame.get("active_agent_ids")) for frame in frames),
            "draw_rounds": sum(bool(frame.get("sampled_choices")) for frame in frames),
            "audited_silent_rounds": sum(frame.get("decision_sampling", {}).get("audited_silent_round") == 1
                                         for frame in frames),
            "sampled_decisions": sum(len(frame.get("sampled_choices", [])) for frame in frames),
            "active_actor_count": len(actors), "public_event_count": len(public_events),
            "substantive_public_event_count": len(substantive),
            "state_change_count": sum(change.get("before") != change.get("after")
                                      for frame in frames for change in frame.get("state_changes", [])),
            "condition_change_count": sum(change.get("before") != change.get("after")
                                          for frame in frames for change in frame.get("question_updates", [])),
            "valid_estimate": valid, "outcome_observed": observed,
            "conditional_only": valid and not observed,
        }
        samples.append(row)
        if valid:
            # Exact public text comparison only; do not imply semantic equivalence
            # or make private decisions distinguish otherwise identical public paths.
            public_paths.add(digest([
                {"step": frame.get("step"), "events": sorted(
                    (event.get("timestamp", ""), " ".join(event.get("description", "").casefold().split()))
                    for event_frame, event in public_events.values() if event_frame is frame
                )} for frame in frames
            ]))
    counters = {key: sum(row[key] for row in samples) for key in (
        "recorded_rounds", "active_rounds", "draw_rounds", "audited_silent_rounds", "sampled_decisions",
        "public_event_count", "substantive_public_event_count", "state_change_count", "condition_change_count",
    )}
    count = len(samples)
    valid_count = sum(row["valid_estimate"] for row in samples)
    observed_count = sum(row["outcome_observed"] for row in samples)
    warnings = []
    if counters["recorded_rounds"] and not (
        counters["state_change_count"] or counters["condition_change_count"] or observed_count
    ):
        warnings.append({"code": "no_future_state_change", "message":
                         "No explicit world-state or tracked-condition changes were recorded. Public activity alone does not establish the outcome."})
    if len(active_actors) < 2 and len(available_actors) > len(active_actors):
        warnings.append({"code": "limited_actor_coverage", "message":
                         "Fewer than two of the supplied actors participated in recorded decisions. This does not require every actor to act."})
    if counters["draw_rounds"] * 2 < counters["recorded_rounds"]:
        warnings.append({"code": "sparse_deliberation", "message":
                         "Most recorded rounds contained no sampled actor choice. Audited silence may be appropriate."})
    probabilities = [row["p_yes"] for row in trajectories if row["p_yes"] is not None]
    if len(probabilities) > 1 and len(set(probabilities)) == 1:
        warnings.append({"code": "identical_terminal_estimates", "message":
                         "All available terminal estimates are equal. This alone does not establish an error or independent corroboration."})
    return {
        "version": "sequential-diagnostics-1", **counters,
        "active_actor_count": len(active_actors), "available_actor_count": len(available_actors),
        "valid_estimate_samples": valid_count, "valid_estimate_coverage": valid_count / count if count else 0,
        "outcome_observed_samples": observed_count, "outcome_observation_coverage": observed_count / count if count else 0,
        "conditional_only_samples": sum(row["conditional_only"] for row in samples),
        "observation_basis": "Recorded public contract evidence inside the simulation; not observed real-world outcomes.",
        "unique_public_trajectory_count": len(public_paths), "public_trajectory_samples": valid_count,
        "public_trajectory_comparison": "Exact normalized public event text and timing, grouped by round.",
        "samples": samples, "warnings": warnings,
    }


def summarize_sequential(records, *, expected_steps=None):
    trajectories = []
    for number, record in enumerate(records, 1):
        rounds = public_replay(record)
        status = record.get("status", "not_started")
        error = record.get("error")
        draws = _recorded_draws(record.get("rounds", []), expected_steps if status == "completed" else None)
        probability = None
        if status == "completed":
            if draws is None:
                status, error = "invalid_trace", "No valid sampled actor decision trace was recorded for this trajectory."
            elif _valid_probability(value := (rounds[-1].get("contract_outlook") or {}).get("p_yes")):
                probability = value
            else:
                status, error = "unresolved", "The final settled state did not yield a supported conditional probability."
        trajectories.append({
            "sample_id": number, "rounds": rounds, "branches": [], "status": status,
            "p_yes": probability, "error": error,
            "action_path_probability": math.prod(draws) if draws else None,
        })
    count = len(trajectories)
    known = [row["p_yes"] for row in trajectories if row["p_yes"] is not None]
    total = math.fsum(known)
    complete = bool(count) and len(known) == count
    statuses = [row["status"] for row in trajectories]
    invalid = statuses.count("invalid_trace")
    unresolved = statuses.count("unresolved")
    failed = statuses.count("failed")
    interrupted = statuses.count("interrupted")
    readout_retry = any(row["status"] == "unresolved" and row["rounds"]
                        and row["rounds"][-1].get("outlook_issue") in {
                            "outlook_call_unavailable", "unsupported_quote", "unverified_certainty",
                            "missing_observation_mechanism", "no_terminal_observation",
                            "non_substantive_quote", "unsupported_interpretation", "basis_too_long",
                            "missing_change_driver", "missing_outlook",
                        } for row in trajectories)
    return {
        "probability_method": SEQUENTIAL_PROTOCOL, "trajectories": trajectories,
        "simulation_diagnostics": _sequential_diagnostics(records, trajectories),
        "p_yes": total / count if complete else None,
        "covered_p_yes": total / len(known) if known else None,
        "coverage": len(known) / count if count else 0,
        "unknown_bounds": [total / count, (total + count - len(known)) / count] if count else None,
        "unknown_samples": unresolved,
        "invalid_trace_samples": invalid,
        "failed_samples": failed,
        "interrupted_samples": interrupted,
        "unfinished_samples": count - len(known) - unresolved - invalid - failed - interrupted,
        "resume_available": bool(readout_retry or failed or interrupted or count > len(known) + unresolved + invalid),
        "status": "completed" if complete else "partial" if known or unresolved else "failed",
        "error": None if complete else "A full estimate requires a valid action trace and final conditional probability for every planned trajectory.",
    }
