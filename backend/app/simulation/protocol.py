"""Program-owned clock, schedules, numeric rates and explicit action effects."""

from __future__ import annotations

import copy
import math
import operator

from app.domain.protocol import ScenarioProtocol
from app.simulation.measurements import at_path, _number

RESERVED = {
    "simulation_clock",
    "measurement_paths",
    "scenario_protocol",
    "program_observations",
    "decision_queue",
    "state_invariants",
    "scenario",
}


def set_path(state, path, value):
    parts = path.split(".")
    if parts[0] in RESERVED:
        raise ValueError(f"Mechanism cannot overwrite engine field: {path}")
    target = state
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"Missing mechanism path: {path}")
        target = target[part]
    if parts[-1] not in target:
        raise ValueError(f"Missing mechanism path: {path}")
    target[parts[-1]] = value


def protocol_for(snapshot):
    return ScenarioProtocol.model_validate(
        snapshot.world_state.get("scenario_protocol") or {}
    )


def conditions_hold(state, conditions):
    ops = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }
    try:
        return all(ops[c.operator](at_path(state, c.path), c.value) for c in conditions)
    except (TypeError, ValueError):
        return False


def prepare_step(snapshot, protocol, step):
    working = snapshot.model_copy(deep=True)
    for rate in protocol.rates:
        set_path(
            working.world_state,
            rate.path,
            _number(at_path(working.world_state, rate.path)) + rate.delta_per_step,
        )
    observations = list(working.world_state.get("program_observations", []))
    for i, rate in enumerate(protocol.rates):
        if rate.delta_per_step:
            observations.append(
                {
                    "id": f"rate_{step}_{i}",
                    "step": step,
                    "path": rate.path,
                    "value": at_path(working.world_state, rate.path),
                    "origin": "configured_scenario_assumption",
                }
            )
    for event in protocol.scheduled_observations:
        if event.step == step:
            for path, value in event.updates.items():
                set_path(working.world_state, path, copy.deepcopy(value))
            observations.append(
                {
                    "id": f"scheduled_{step}_{len(observations)}",
                    "step": step,
                    "description": event.description,
                    "origin": "configured_scenario_assumption",
                }
            )
    working.world_state["program_observations"] = observations
    working.world_state["decision_queue"] = [
        a for a, steps in protocol.decision_due_steps.items() if step in steps
    ]
    working.world_state["scenario_protocol"] = protocol.model_dump()
    return working


def validate_action(snapshot, proposal):
    protocol = protocol_for(snapshot)
    if not protocol.action_mechanisms or proposal.decision != "act":
        return []
    rule = next(
        (
            r
            for r in protocol.action_mechanisms
            if r.action_type == proposal.action_type
        ),
        None,
    )
    if rule is None:
        return ["ACTION_OUTSIDE_CONFIGURED_MECHANISMS"]
    if rule.allowed_actor_ids and proposal.agent_id not in rule.allowed_actor_ids:
        return ["ACTOR_NOT_ALLOWED_BY_MECHANISM"]
    if not conditions_hold(snapshot.world_state, rule.preconditions):
        return ["MECHANISM_PRECONDITION_FAILED"]
    return []


def program_projection(snapshot, actions):
    protocol = protocol_for(snapshot)
    expected = {
        rate.path: at_path(snapshot.world_state, rate.path) for rate in protocol.rates
    }
    for rule in protocol.action_mechanisms:
        expected.update(
            {
                effect.path: at_path(snapshot.world_state, effect.path)
                for effect in rule.effects
            }
        )
    for event in protocol.scheduled_observations:
        expected.update({p: at_path(snapshot.world_state, p) for p in event.updates})
    for action in actions:
        violations = validate_action(snapshot, action)
        if violations:
            raise ValueError(str(violations))
        if action.decision != "act":
            continue
        rule = next(
            (
                r
                for r in protocol.action_mechanisms
                if r.action_type == action.action_type
            ),
            None,
        )
        if rule:
            for effect in rule.effects:
                current = expected.get(
                    effect.path, at_path(snapshot.world_state, effect.path)
                )
                expected[effect.path] = _number(current) + effect.delta_per_step
    projected = copy.deepcopy(snapshot.world_state)
    for path, value in expected.items():
        set_path(projected, path, value)
    from app.domain.ensemble import StateInvariant
    from app.simulation.measurements import validate_invariants

    validate_invariants(
        snapshot.world_state,
        projected,
        [
            StateInvariant.model_validate(r)
            for r in snapshot.world_state.get("state_invariants", [])
        ],
    )
    return expected


def validate_program_projection(expected, after):
    for path, value in expected.items():
        actual = at_path(after, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9)
            ):
                raise ValueError(
                    f"Program-owned state differs from configured mechanism: {path}"
                )
        elif actual != value:
            raise ValueError(
                f"Program-owned state differs from configured mechanism: {path}"
            )


def public_digest(snapshot):
    from app.knowledge.okf import digest

    state = {
        k: v
        for k, v in snapshot.world_state.items()
        if k not in {"simulation_clock", "program_observations", "decision_queue"}
    }
    return digest(
        {
            "state": state,
            "summary": snapshot.state_summary,
            "process_issues": [i.model_dump() for i in snapshot.process_issues],
            "capabilities": {
                a.agent_id: [c.model_dump() for c in a.capabilities]
                for a in snapshot.agents
            },
            "relationships": [r.model_dump() for r in snapshot.relationships],
        }
    )


def can_advance_quiet(snapshot, record, protocol):
    return (
        protocol.idle_policy == "advance_if_unchanged"
        and record.get("quiet_candidate", False)
        and record.get("post_round_public_digest") == public_digest(snapshot)
        and not any(
            o.get("step") == snapshot.round_id + 1
            for o in snapshot.world_state.get("program_observations", [])
        )
        and not snapshot.world_state.get("decision_queue")
        and not any(i.pending_actor_ids for i in snapshot.process_issues)
        and not any(
            a.activation.last_deliberation_round is None
            or snapshot.round_id + 1 - a.activation.last_deliberation_round >= 3
            for a in snapshot.agents
        )
        and not any(
            not o.processed for a in snapshot.agents for o in a.private_observations
        )
    )


def advance_quiet(snapshot):
    from app.domain import ForecastRound

    after = snapshot.model_copy(deep=True)
    after.round_id += 1
    after.snapshot_version += 1
    for a in after.agents:
        a.activation.status = "inactive"
        a.last_processed_world_round = snapshot.round_id
    return after, ForecastRound(
        step=after.round_id,
        input_snapshot_version=snapshot.snapshot_version,
        output_snapshot_version=after.snapshot_version,
        state_summary=snapshot.state_summary,
        analysis="Configured idle policy advanced one reporting interval: no new facts, due decisions or pending private observations.",
    )
