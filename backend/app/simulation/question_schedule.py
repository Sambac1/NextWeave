"""Freeze an adaptive round schedule around observable forecast milestones."""
from datetime import date, timedelta

from app.benchmark.dynamics import scheduled_round_dates
from app.knowledge.wikipedia import utc_timestamp


SINGLE_ROUND_METHODS = {"one_step", "one_step_ensemble"}
MAX_ADAPTIVE_ROUNDS = 8


def round_boundaries(start: date, end: date, steps: int, *, milestone: date | None = None) -> list[str]:
    days = (end - start).days
    if days < steps:
        raise ValueError("Simulation needs at least one UTC calendar day per step")
    boundaries = [start + timedelta(days=i * days // steps) for i in range(steps + 1)]
    if milestone and start < milestone < end and steps > 1 and milestone not in boundaries:
        nearest = min(range(1, steps), key=lambda i: abs((boundaries[i] - milestone).days))
        boundaries[nearest] = milestone
        boundaries.sort()
    return [boundary.isoformat() for boundary in boundaries]


def simulation_round_plan(case, method, steps, contract, *, keep_milestones: bool = False) -> dict:
    """Freeze a calendar plan from cutoff-safe inputs shared by every trajectory."""
    as_of = utc_timestamp(case["as_of"], "as_of")
    deadline = utc_timestamp(case["outcome_deadline"], "outcome_deadline")
    if deadline <= as_of:
        raise ValueError("The simulation deadline must follow the forecast cutoff")
    start, end = as_of.date(), deadline.date()
    milestone = contract.get("scheduled_milestone", {})
    vote = date.fromisoformat(milestone["date"]) if milestone.get("date") else None
    mode = "single_step" if method in SINGLE_ROUND_METHODS else "adaptive" if steps is None else "fixed"
    count = 1 if mode == "single_step" else steps
    if start == end and count in (None, 1):
        return {"mode": mode, "steps": 1, "boundaries": [start.isoformat(), end.isoformat()], "milestones": []}
    if count is not None and (not keep_milestones or mode == "single_step"):
        dates = round_boundaries(start, end, count, milestone=vote)
        return {"mode": mode, "steps": count, "boundaries": dates,
                "milestones": [vote.isoformat()] if vote and vote.isoformat() in dates[1:-1] else []}

    days = (end - start).days
    if count is not None and count > days:
        raise ValueError(f"This forecast window supports at most {max(1, days)} time steps with UTC day boundaries.")
    target = next((rounds for limit, rounds in ((3, 2), (14, 3), (45, 4), (120, 5), (240, 6))
                   if days <= limit), MAX_ADAPTIVE_ROUNDS)
    counting_from = milestone.get("counting_from")
    registered = [vote, date.fromisoformat(counting_from) if counting_from else None]
    source_days = [date.fromisoformat(row["date"]) for row in contract.get("source_milestones", [])]
    ordered = dict.fromkeys([*registered, *scheduled_round_dates(case["rules"]), *source_days])
    anchor_limit = (count if count is not None else MAX_ADAPTIVE_ROUNDS) - 1
    anchors = sorted([day for day in ordered if day is not None and start < day < end][:anchor_limit])
    if count is None:
        count = min(days, max(target, len(anchors) + 1))
    if not anchors:
        dates = round_boundaries(start, end, count)
    else:
        points = [start, *anchors, end]
        while len(points) < count + 1:
            index = max(range(len(points) - 1), key=lambda i: points[i + 1] - points[i])
            left, right = points[index:index + 2]
            points.insert(index + 1, left + (right - left) // 2)
        dates = [point.isoformat() for point in points]
    return {"mode": mode, "steps": count, "boundaries": dates, "milestones": [day.isoformat() for day in anchors]}


