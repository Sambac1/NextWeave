"""Calendar intervals belong to the engine, not to model-generated dates."""
from calendar import monthrange
from datetime import date, datetime, time, timedelta, timezone


def resolve_start(request, history):
    bounds = []
    for event in history:
        raw = event.get('timestamp', '')
        try:
            if len(raw) == 10:
                bounds.append(date.fromisoformat(raw))
            elif len(raw) == 7:
                year, month = map(int, raw.split('-'))
                bounds.append(date(year, month, monthrange(year, month)[1]))
        except (ValueError, TypeError):
            continue
    if request.as_of_month:
        year, month = map(int, request.as_of_month.split('-'))
        bounds.append(date(year, month, monthrange(year, month)[1]))
    boundary = max(bounds) if bounds else None
    if request.start_date:
        chosen = date.fromisoformat(request.start_date)
        if boundary and chosen < boundary:
            raise ValueError(f'The simulation cannot start before the latest historical date, {boundary}. Events dated only to a month use the end of that month.')
        return chosen.isoformat(), 'explicit_caller_date'
    return (boundary or date.today()).isoformat(), 'historical_upper_bound' if boundary else 'creation_date_assumption'


def _advance(start, count, unit):
    if unit in {'day', 'week'}:
        return start + timedelta(days=count * (7 if unit == 'week' else 1))
    months = count * (12 if unit == 'year' else 1)
    year, month = divmod(start.year * 12 + start.month - 1 + months, 12)
    month += 1
    last = monthrange(year, month)[1]
    day = last if start.day == monthrange(start.year, start.month)[1] else min(start.day, last)
    return date(year, month, day)


def step_clock(start, step, unit, horizon, cutoff=None, *, step_size=1):
    start = date.fromisoformat(start)
    left = _advance(start, (step - 1) * step_size, unit)
    right = _advance(start, step * step_size, unit)
    return {'step': step, 'unit': unit, **({'step_size': step_size} if step_size != 1 else {}),
            'historical_cutoff': cutoff, 'horizon_steps': horizon,
            'interval_start': left.isoformat(), 'interval_end': right.isoformat(),
            'horizon_end': _advance(start, horizon * step_size, unit).isoformat(),
            'elapsed_days': (right - start).days, 'interval_days': (right - left).days}


def explicit_step_clock(boundaries, step, *, as_of=None, outcome_deadline=None):
    left, right = boundaries[step - 1:step + 1]
    clock = {
        "step": step, "unit": "day", "horizon_steps": len(boundaries) - 1,
        "interval_start": left, "interval_end": right, "horizon_end": boundaries[-1],
        "elapsed_days": (date.fromisoformat(right) - date.fromisoformat(boundaries[0])).days,
        "interval_days": (date.fromisoformat(right) - date.fromisoformat(left)).days,
    }
    if as_of and outcome_deadline:
        def midnight(day):
            return datetime.combine(date.fromisoformat(day), time.min, timezone.utc).isoformat()

        clock.update(
            interval_start_utc=as_of if step == 1 else midnight(left),
            interval_end_utc=outcome_deadline if step == len(boundaries) - 1 else midnight(right),
            horizon_end_utc=outcome_deadline,
        )
    return clock


def align_events(events, clock):
    if not clock.get('interval_start'):
        return
    left, right = clock['interval_start'], clock['interval_end']
    for event in events:
        inside = event_in_window(event.timestamp, clock)
        if inside is False:
            raise ValueError(f'The simulated event date {event.timestamp} falls outside the interval ({left}, {right}]')
        if len(event.timestamp) > 10:
            event.date_original = event.date_original or event.timestamp
            event.timestamp = _instant(event.timestamp).isoformat()
            event.date_precision = 'datetime'
        elif len(event.timestamp) == 10:
            event.date_precision = 'day' if inside else 'interval'
        else:
            event.date_original = event.date_original or event.timestamp
            event.timestamp = right[:7]
            event.date_precision = 'interval'
        event.date_range = {'start': clock.get('interval_start_utc', left), 'end': clock.get('interval_end_utc', right)}
        event.date_evidence = {'kind': 'simulation_step_interval', 'exact_occurrence_known': inside is True}


def _instant(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError('Simulated timestamps require a timezone')
    return parsed.astimezone(timezone.utc)


def on_or_after(value, lower_bound):
    """Compare occurrence dates to an inclusive contract start at their available precision."""
    if not lower_bound:
        return True
    try:
        if len(value) == 7:
            first = date.fromisoformat(value + "-01")
            last = first.replace(day=monthrange(first.year, first.month)[1])
        elif len(value) == 10:
            first = last = date.fromisoformat(value)
        else:
            instant = _instant(value)
            if len(lower_bound) > 10:
                return instant >= _instant(lower_bound)
            first = last = instant.date()
        start = date.fromisoformat(lower_bound[:10])
    except (TypeError, ValueError):
        return None
    if last < start:
        return False
    if first > start or (first == start and len(lower_bound) == 10):
        return True
    return None


def event_in_window(value, clock):
    """True for contained times, False for disjoint, None for partial or unknown."""
    left, right = clock['interval_start'], clock['interval_end']
    exact = 'interval_start_utc' in clock and 'interval_end_utc' in clock
    if len(value) > 10:
        instant = _instant(value)
        if exact:
            return _instant(clock['interval_start_utc']) < instant <= _instant(clock['interval_end_utc'])
        return date.fromisoformat(left) < instant.date() <= date.fromisoformat(right)
    if len(value) != 10:
        return None
    day = date.fromisoformat(value)
    if not exact:
        return date.fromisoformat(left) < day <= date.fromisoformat(right)
    start, end = _instant(clock['interval_start_utc']), _instant(clock['interval_end_utc'])
    day_start = datetime.combine(day, time.min, timezone.utc)
    next_day = day_start + timedelta(days=1)
    if next_day <= start or day_start > end:
        return False
    return True if start < day_start and next_day <= end else None
