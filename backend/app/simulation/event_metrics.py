"""Evidence-aware, three-valued event measurements; missing is never false."""
from copy import deepcopy
import re

from app.domain.ensemble import EventQuestion, QuestionAnswer
from app.simulation.measurements import condition_value
from app.simulation.activities import observations as activity_observations
from app.simulation.terminal_outcomes import public_measurement_events

VERSION = 'event-measurement-3'


def observe(state, plan, clock, *, step=0):
    return {'step': step, 'interval': deepcopy(clock),
            'questions': {q.id: condition_value(state, q.conditions) for q in plan.event_questions if q.conditions}}


def needs_duration(question):
    return bool(question.required_duration_days or re.search(
        r'(?:连续|持续|至少).{0,16}(?:小时|天|日)|\b(?:for|last|at least)\s+\d+\s+(?:hours?|days?)\b|\b\d+-(?:hours?|days?)\b|\b(?:continuous|consecutive|sustained)\b',
        question.criterion, re.I))


def resolve_answer(question, answer, record, *, strict_occurrence=False):
    """Normalize model claims against actual coverage and persisted evidence."""
    q = EventQuestion.model_validate(question) if isinstance(question, dict) else question
    a = QuestionAnswer.model_validate(answer) if isinstance(answer, dict) else answer.model_copy(deep=True)
    events = {e['timeline_id']: e for r in record.get('rounds', []) for e in r.get('timeline', [])}
    if strict_occurrence:
        events = {ref: event for ref, (_, event) in public_measurement_events(record).items()}
    known = {v['agent_id'] for v in record.get('snapshot', {}).get('agents', [])}

    # Log matching is a separate estimand: absence in a generated trajectory,
    # rather than a claim of real-world nonoccurrence. Preserve it before the
    # stricter evidence validator can downgrade a judgment.
    if a.trace_answer is None:
        valid_refs = bool(a.event_refs) and set(a.event_refs) <= events.keys()
        if a.answer == 'yes' and valid_refs and a.evidence_kind != 'not_observed' and not needs_duration(q) and not q.conditions:
            a.trace_answer = 'matched'
        elif (a.answer == 'no' or (a.answer == 'unclear' and a.evidence_kind == 'not_observed')) and (not a.event_refs or set(a.event_refs) <= events.keys()):
            a.trace_answer = 'not_matched'
        else:
            a.trace_answer = 'unclear'
        a.trace_reason = a.reason or ('The generated records contain no match for the full event criteria.' if a.trace_answer == 'not_matched' else 'No trajectory match assessment is available.')

    def unknown(reason):
        return QuestionAnswer(question_id=q.id, answer='unclear', reason=reason,
                              evidence_kind='unknown', event_refs=list(dict.fromkeys(r for r in a.event_refs if r in events)),
                              trace_answer=a.trace_answer, trace_reason=a.trace_reason)

    if q.coverage_status == 'limited' or not set(q.required_actor_ids) <= known:
        return unknown('The question depends on an actor outside the simulation.')
    if q.conditions:
        history = record.get('measurement_history', [])
        observations = [h.get('questions', {}).get(q.id) for h in history if h.get('step', 0) > 0]
        if q.required_duration_days:
            if not q.assume_continuity:
                return unknown('Only boundary states are available. A duration requires continuous observation or an explicit persistence rule.')
            duration, maximum, transition = 0, 0, False
            previous = history[0].get('questions', {}).get(q.id) if history else None
            for frame in history[1:]:
                current = frame.get('questions', {}).get(q.id)
                transition = transition or (previous is not None and current is not None and previous != current)
                duration = duration + frame.get('interval', {}).get('interval_days', 0) if previous is True and current is True else 0
                maximum = max(maximum, duration)
                previous = current
            value = (duration if q.scope == 'terminal' else maximum) >= q.required_duration_days
            if not value and transition and not (q.scope == 'terminal' and previous is False):
                return unknown('The state changed between observations. Its exact timing and duration are unknown.')
            if not value and (not observations or len(observations) != len(record.get('rounds', [])) or any(v is None for v in observations)):
                return unknown('The duration observations have gaps, so the threshold cannot be ruled out.')
        elif q.scope == 'terminal':
            value = condition_value(record.get('snapshot', {}).get('world_state', {}), q.conditions)
        else:
            initial = history[0].get('questions', {}).get(q.id) if history else None
            if q.condition_coverage == 'cumulative_horizon' and initial is not False:
                return unknown('The cumulative flag is unknown at the start of the forecast window. The event may have occurred earlier.')
            if any(v is True for v in observations):
                value = True
            elif not observations or len(observations) != len(record.get('rounds', [])) or any(v is None for v in observations):
                value = None
            elif q.condition_coverage != 'cumulative_horizon':
                return unknown('The boundary states did not meet the criterion. Observations between those boundaries are unavailable.')
            else:
                value = False
        if value is None:
            return unknown('A required state field or observation interval is missing.')
        return QuestionAnswer(question_id=q.id, answer='yes' if value else 'no',
                              evidence_kind='state_rule', reason='Assessed from recorded structured states and the explicit timing rules.',
                              trace_answer='matched' if value else 'not_matched',
                              trace_reason='Assessed from saved structured states.')
    if needs_duration(q):
        return unknown('This question requires a duration. Verifiable continuous state records are unavailable.')
    if q.scope == 'terminal' and record.get('rounds'):
        terminal_ids = {e['timeline_id'] for e in record['rounds'][-1].get('timeline', [])}
        if not a.event_refs or not set(a.event_refs) <= terminal_ids:
            return unknown('No direct evidence from the final interval supports a terminal assessment.')
    if strict_occurrence and a.answer == 'yes' and not q.conditions:
        quote = a.evidence_quote.strip()
        cited = [events[ref] for ref in a.event_refs if ref in events]
        supporting = [event for event in cited if len(quote) >= 8 and quote in event.get('description', '')]
        actors = {actor for event in supporting for actor in event.get('actors', [])}
        stage_ok = not q.milestone or any(
            item["stage"] == q.milestone and (quote in item["quote"] or item["quote"] in quote)
            for event in supporting for item in activity_observations(event)
        )
        if not supporting or not set(a.event_refs) <= events.keys() or not set(q.required_actor_ids) <= actors or not stage_ok:
            a.trace_answer = 'unclear'
            a.trace_reason = 'The citation could not verify the actor, quoted passage, or required stage.'
            return unknown(a.trace_reason)
    if a.answer == 'yes' and a.event_refs and set(a.event_refs) <= events.keys() and a.evidence_kind not in {'not_observed'}:
        a.evidence_kind = 'occurrence'
        return a
    if strict_occurrence and a.answer == 'no' and q.scope == 'during_horizon':
        a.trace_answer = 'unclear'
        a.trace_reason = 'No event record covers the entire forecast window to verify non-occurrence.'
        return unknown(a.trace_reason)
    if a.answer == 'no' and a.evidence_kind == 'explicit_negative' and a.event_refs and a.evidence_quote:
        if q.scope == 'during_horizon' and a.evidence_scope != 'whole_horizon':
            return unknown('Evidence of non-occurrence does not cover the full forecast interval.')
        last_ids = {e['timeline_id'] for e in record.get('rounds', [{}])[-1].get('timeline', [])} if record.get('rounds') else set()
        if not set(a.event_refs) & last_ids:
            return unknown('Earlier evidence does not rule out a later occurrence. The final interval lacks direct evidence.')
        if all(r in events for r in a.event_refs) and any(a.evidence_quote in events[r].get('description', '') for r in a.event_refs):
            return a
    if a.answer == 'unclear' and a.evidence_kind == 'unknown':
        return a
    return unknown('No qualifying event was recorded, and the available evidence cannot establish occurrence or non-occurrence. Original assessment · ' + a.reason[:400])


def event_metrics(question, records, planned, interval, *, strict_occurrence=False):
    answers = []
    for record in records:
        if record.get('status') != 'completed':
            answers.append({'sample_id': record['sample_id'], 'answer': 'unfinished', 'reason': record.get('error') or 'The trajectory is unfinished.', 'event_refs': [], 'evidence': []})
            continue
        raw = next((a for a in (record.get('classification') or {}).get('questions', []) if a['question_id'] == question.id), None)
        raw = raw or QuestionAnswer(question_id=question.id, answer='unclear', reason='No assessment was saved for this question.')
        value = resolve_answer(question, raw, record, strict_occurrence=strict_occurrence)
        events = ({ref: event for ref, (_, event) in public_measurement_events(record).items()}
                  if strict_occurrence else
                  {e['timeline_id']: e for r in record.get('rounds', []) for e in r.get('timeline', [])})
        answers.append(value.model_dump() | {'sample_id': record['sample_id'], 'evidence': [
            {'timeline_id': r, 'description': events[r]['description']} for r in dict.fromkeys(value.event_refs) if r in events]})
    yes = [a['sample_id'] for a in answers if a['answer'] == 'yes']
    no = [a['sample_id'] for a in answers if a['answer'] == 'no']
    unknown = [a['sample_id'] for a in answers if a['answer'] == 'unclear']
    unfinished = planned - len(yes) - len(no) - len(unknown)
    complete = not unknown and unfinished == 0 and planned > 0
    matched = [a['sample_id'] for a in answers if a.get('trace_answer') == 'matched']
    not_matched = [a['sample_id'] for a in answers if a.get('trace_answer') == 'not_matched']
    unreviewed = [a['sample_id'] for a in answers if a['answer'] != 'unfinished' and a.get('trace_answer') not in {'matched', 'not_matched'}]
    reviewed = len(matched) + len(not_matched)
    trace = {'basis':'generated_trajectory_match', 'matched_count':len(matched),
             'not_matched_count':len(not_matched), 'unclear_count':len(unreviewed),
             'reviewed_count':reviewed, 'unfinished_count':unfinished, 'planned_count':planned,
             'match_frequency':len(matched)/reviewed if reviewed else None,
             'matched_sample_ids':matched, 'not_matched_sample_ids':not_matched}
    return question.model_dump() | {
        'count': len(yes), 'no_count': len(no), 'unclear_count': len(unknown), 'unfinished_count': unfinished,
        'sample_ids': yes, 'no_sample_ids': no, 'unknown_sample_ids': unknown, 'planned_samples': planned,
        'assessable_samples': len(yes) + len(no), 'completed_samples': planned - unfinished,
        'frequency': len(yes) / planned if complete else None,
        'known_sample_frequency': len(yes) / (len(yes) + len(no)) if yes or no else None,
        'interpretation_bounds': [len(yes) / planned, (len(yes) + len(unknown) + unfinished) / planned] if planned else None,
        'sampling_interval_95': interval(len(yes), planned) if complete else None,
        'measurement_status': 'measured' if complete else 'unassessable' if not yes and not no else 'partial',
        'measurement_version': VERSION, 'answers_by_sample': answers, 'trace_metrics':trace,
    }
