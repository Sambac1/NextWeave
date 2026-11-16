"""Read-only process comparison. No model calls, private states, or target rewriting."""
from collections import Counter
from itertools import combinations
import re
from app.simulation.process import key

VERSION = 'process-summary-1'


def reliability_counts(records):
    completed = [r for r in records if r.get('status') == 'completed']
    codes = Counter(i.get('code') for r in completed for i in r.get('snapshot', {}).get('validation_issues', []))
    return {'bdi_updates_rejected': codes['bdi_patch_rejected'],
            'issue_updates_rejected': codes['issue_update_rejected'],
            'goal_updates_rejected': codes['goal_update_rejected'],
            'goal_ids_repaired': codes['goal_id_repaired'],
            'degraded_samples': sum(any(i.get('code') != 'goal_id_repaired' for i in r.get('snapshot', {}).get('validation_issues', [])) for r in completed)}


def public_issue_reconvergence(records):
    """Count completed pairs with an observed public issue split and later match.

    Compare only matching, nonempty sets of tracked issues with explicit status
    codes. A different public choice and a structured issue difference must both
    precede convergence. The result says nothing about private or untracked state.
    """
    histories = []
    for record in records:
        if record.get('status') != 'completed':
            continue
        history = {}
        issues = {}
        for frame in record.get('rounds', []):
            for update in frame.get('issue_updates', []):
                issues[update['issue_id']] = update
            choices = {
                action['agent_id']: (action.get('decision'), key(action.get('action_type', '')), action.get('target', ''))
                for action in frame.get('agent_actions', []) if action.get('visibility') == 'public'
            }
            known = issues and all(
                update.get('state_code') not in (None, '', 'unknown') and update.get('after', '').strip()
                for update in issues.values()
            )
            if known:
                issue_ids = tuple(sorted(issues))
                structure = tuple(
                    (issues[issue_id]['state_code'], tuple(sorted(set(issues[issue_id].get('actor_ids', [])))),
                     tuple(sorted(set(issues[issue_id].get('pending_actor_ids', [])))))
                    for issue_id in issue_ids
                )
                states = tuple(' '.join(issues[issue_id]['after'].split()) for issue_id in issue_ids)
                issue_frame = (issue_ids, structure, states)
            else:
                issue_frame = None
            history[frame['step']] = (choices, issue_frame)
        histories.append(history)

    divergence_pairs = reconvergence_pairs = 0
    for first, second in combinations(histories, 2):
        different_choice = False
        diverged_issue_sets = set()
        for step in sorted(first.keys() & second.keys()):
            choices_a, issues_a = first[step]
            choices_b, issues_b = second[step]
            different_choice |= any(choices_a[actor] != choices_b[actor] for actor in choices_a.keys() & choices_b.keys())
            if not different_choice or issues_a is None or issues_b is None or issues_a[0] != issues_b[0]:
                continue
            if issues_a[1] != issues_b[1]:
                diverged_issue_sets.add(issues_a[0])
            elif issues_a == issues_b and issues_a[0] in diverged_issue_sets:
                reconvergence_pairs += 1
                break
        divergence_pairs += bool(diverged_issue_sets)
    return {'public_issue_divergence_pairs': divergence_pairs,
            'public_issue_reconvergence_pairs': reconvergence_pairs}


def process_summary(records, planned):
    completed = sorted((r for r in records if r.get('status') == 'completed'), key=lambda r: r['sample_id'])
    groups = {}
    topic_counts = {}
    all_structured = bool(completed)
    for record in completed:
        steps = []
        signature = []
        seen = set()
        for frame in record.get('rounds', []):
            actions = [a for a in frame.get('agent_actions', []) if a.get('visibility', 'public') == 'public']
            changes = [u for u in frame.get('issue_updates', []) if u.get('kind') != 'unchanged']
            structured = bool(frame.get('issue_updates')) or record.get('process_tracking_version') == 'issue-process-1'
            all_structured &= structured
            events = []
            for event in frame.get('timeline', []):
                text = key(event.get('description', ''))
                if not text or text in seen:
                    continue
                seen.add(text)
                if event.get('change_kind') != 'continuation':
                    events.append({'timeline_id': event.get('timeline_id'), 'description': event.get('description', '')})
            for change in changes:
                topic = topic_counts.setdefault(change['issue_id'], {'id': change['issue_id'], 'title': change['title'], '_samples': set()})
                topic['_samples'].add(record['sample_id'])
            if structured:
                marker = tuple(sorted((u['issue_id'], u.get('state_code', 'unknown'), u['kind'],
                                       key(u['after']) if u.get('state_code', 'unknown') == 'unknown' else '',
                                       tuple(sorted(u.get('actor_ids', []))), tuple(re.findall(r'\d+(?:\.\d+)?%?', u['after'])),
                                       tuple(sorted(u.get('pending_actor_ids', [])))) for u in changes))
                if not changes and events:
                    marker = ('unmapped_public_events', tuple(key(e['description']) for e in events))
            else:
                # Exact public records only; do not declare differently worded
                # old narratives semantically equivalent using a verb regex.
                marker = tuple((e['timeline_id'] and key(e['description'])) for e in events)
            signature.append(marker)
            steps.append({'step': frame.get('step'), 'interval': frame.get('simulation_interval'),
                          'changes': changes, 'events': events[:2], 'public_event_count': len(events),
                          'action_count': sum(a.get('decision') == 'act' for a in actions),
                          'wait_count': sum(a.get('decision') in {'wait', 'monitor', 'no_action'} for a in actions),
                          'structured': structured, 'continuing_count': len(frame.get('continuing_state_notes', []))})
        terminal = record.get('classification', {}).get('category_id', 'other')
        marker = (terminal, tuple(signature))
        group = groups.setdefault(marker, {'representative_sample_id': record['sample_id'], 'sample_ids': [], 'steps': steps,
                                          'terminal_category_id': terminal})
        group['sample_ids'].append(record['sample_id'])
    paths = []
    for index, group in enumerate(sorted(groups.values(), key=lambda g: (-len(g['sample_ids']), g['representative_sample_id'])), 1):
        paths.append({'id': f'path_{index}', **group, 'count': len(group['sample_ids']),
                      'frequency': len(group['sample_ids']) / len(completed)})
    topics = [{**{k:v for k,v in value.items() if k != '_samples'}, 'count': len(value['_samples'])}
              for value in topic_counts.values()]
    return {'version': VERSION, 'completed_samples': len(completed), 'planned_samples': planned,
            'mode': 'structured_public_changes' if all_structured else 'recorded_public_events',
            'comparison_basis': 'Grouping compares issues, participants, state types, values, and sequence. Older records are grouped by their saved public text.',
            'path_count': len(paths), 'paths': paths, 'topics': topics[:4]}
