"""Domain-neutral deterministic checks and held-out scoring helpers."""
from __future__ import annotations

import math
import operator

from app.domain.ensemble import OutcomeDefinition, StateInvariant


def at_path(state: dict, path: str):
    value = state
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Required state field is missing: {path}")
        value = value[part]
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("State constraint requires a finite numeric value")
    return value


def condition_value(state, conditions):
    """Three-valued conjunction with strict boolean/numeric boundaries."""
    ops = {'eq': operator.eq, 'ne': operator.ne, 'lt': operator.lt, 'le': operator.le, 'gt': operator.gt, 'ge': operator.ge}
    values = []
    for c in conditions:
        try:
            value = at_path(state, c.path)
            if value is None or isinstance(value, bool) != isinstance(c.value, bool) or (isinstance(value, float) and not math.isfinite(value)):
                values.append(None)
            else:
                values.append(ops[c.operator](value, c.value))
        except (ValueError, TypeError):
            values.append(None)
    return None if not values or None in values else all(values)


def validate_invariants(before: dict, after: dict, rules: list[StateInvariant]):
    for rule in rules:
        old = [at_path(before, p) for p in rule.paths]
        new = [at_path(after, p) for p in rule.paths]
        if rule.rule == "fixed" and old != new:
            raise ValueError(f"Fixed state changed: {rule.paths}")
        if rule.rule == "range":
            for value in new:
                number = _number(value)
                if (rule.minimum is not None and number < rule.minimum) or (rule.maximum is not None and number > rule.maximum):
                    raise ValueError(f"State range violated: {rule.paths}")
        if rule.rule == "nondecreasing" and any(_number(b) < _number(a) for a, b in zip(old, new)):
            raise ValueError(f"Monotonic state decreased: {rule.paths}")
        if rule.rule == "conserved_sum" and not math.isclose(sum(map(_number, old)), sum(map(_number, new)), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Conserved state total changed: {rule.paths}")


def classify_by_rules(state: dict, categories: list[OutcomeDefinition]) -> str | None:
    if not all(c.conditions for c in categories):
        return None
    checks = [condition_value(state, c.conditions) for c in categories]
    if any(v is None for v in checks):
        return 'other'
    matched = [c.id for c, applies in zip(categories, checks) if applies]
    return matched[0] if len(matched) == 1 else "other"
