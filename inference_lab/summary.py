"""Read-only extraction: search-stage maxima are never final scores."""
import csv
import hashlib
import io
import json
from pathlib import Path

from .core import fingerprint


def summarize_runs(inputs):
    files = set()
    finals = set()
    for source in inputs:
        source = Path(source).resolve()
        if source.is_dir():
            files.update(source.rglob('run.json'))
            finals.update(source.rglob('final.json'))
        elif source.name == 'run.json':
            files.add(source)
            finals.update(source.parent.parent.rglob('final.json'))
        else:
            raise ValueError('Summary input must be a directory or run.json')
    final_records = [(p, json.loads(p.read_text(encoding='utf-8'))) for p in sorted(finals)]
    rows = []
    for path in sorted(files):
        run = json.loads(path.read_text(encoding='utf-8'))
        trials = run.get('trials', [])
        costs = run.get('costs', {})
        new = run.get('record_version') == 'proposal-ledger-v1'
        calls = costs.get('llm_calls') if new else None
        executed = 0
        first_success_slot = first_success_experiment = None
        for trial in trials:
            consumed = trial.get('candidate_executed')
            if consumed is None:
                consumed = 'proposal' in trial and trial.get('verdict', {}).get('reason') not in (
                    'proposal_failed', 'unchanged_candidate', 'search_space_exhausted')
            executed += bool(consumed)
            if trial.get('verdict', {}).get('accepted') and first_success_slot is None:
                first_success_slot = trial['id']
                first_success_experiment = executed
        matches = []
        best_path = path.parent / 'best.json'
        best_hash = hashlib.sha256(best_path.read_bytes()).hexdigest() if best_path.exists() else None
        for final_path, final in final_records:
            provenance = final.get('provenance', {})
            if (provenance.get('source_run') and Path(provenance['source_run']).resolve() == path.parent
                    and provenance.get('source_best_sha256') == best_hash
                    and final.get('settings') == run.get('settings')
                    and final.get('selected_config') == (run.get('state') or {}).get('config')):
                matches.append((final_path, final))
        # Multiple final scores are ambiguous: do not choose the nicest or latest.
        final_path, final = matches[0] if len(matches) == 1 else (None, {})
        valid_final = final.get('status') == 'completed' and final.get('valid') is True
        gain = final.get('throughput_gain') if valid_final else None
        rows.append({'source_run': str(path.parent), 'status': run.get('status'),
                     'task_id': fingerprint({'settings': run.get('settings'), 'backend': run.get('backend'),
                                              'hardware': run.get('hardware'),
                                              'evaluator_sources': run.get('evaluator_sources')}),
                     'method': run.get('planner'), 'planner_version': run.get('planner_version'),
                     'context_mode': run.get('context_mode'),
                     'method_label': ('local:' + (run.get('context_mode') or 'unknown'))
                                     if run.get('planner') == 'local' else run.get('planner'),
                     'proposal_slot_limit': run.get('proposal_slot_limit'),
                     'seed': run.get('seed'), 'environment_fingerprint': run.get('environment_fingerprint'),
                     'final_verified_improvement': gain,
                     'final_valid': final.get('valid'), 'final_record_count': len(matches),
                     'final_source': str(final_path) if final_path else None,
                     'selected_config': (run.get('state') or {}).get('config'),
                     'accepted_candidates': sum(t.get('verdict', {}).get('accepted', False) for t in trials),
                     'search_success': any(t.get('verdict', {}).get('accepted', False) for t in trials),
                     'budget_to_success_slots': first_success_slot,
                     'budget_to_success_experiments': first_success_experiment,
                     'proposal_slots': len(trials), 'experiments_consumed': executed,
                     'llm_calls': calls, 'retry_calls': costs.get('retry_calls') if new else None,
                     'known_total_tokens': costs.get('known_total_tokens') if new else None,
                     'total_tokens': costs.get('total_tokens') if new else None,
                     'inference_cost': costs.get('cost') if new else None,
                     'usage_missing_calls': costs.get('usage_missing_calls') if new else None,
                     'invalid_proposal_rate': costs.get('invalid_calls', 0) / calls if calls else None,
                     'duplicate_proposal_rate': costs.get('duplicate_calls', 0) / calls if calls else None,
                     'rollback_count': costs.get('rollback_count') if new else sum(
                         t.get('rollback', '').startswith('incumbent_restart') for t in trials),
                     'proposal_seconds': costs.get('proposal_seconds') if new else None,
                     'planner_setup_seconds': costs.get('planner_setup_seconds') if new else None,
                     'wall_clock_seconds': costs.get('wall_clock_seconds') if new else None,
                     'telemetry_complete': new,
                     'note': 'Legacy costs unknown' if not new else
                             ('Multiple final records; no score selected' if len(matches) > 1 else '')})
    return rows


def paired_differences(rows):
    """Descriptive task pairs only; repetitions/seeds never become extra tasks."""
    result = []
    for task_id in sorted({r['task_id'] for r in rows}):
        group = [r for r in rows if r['task_id'] == task_id]
        reference = [r for r in group if r.get('method_label') == 'local:full']
        if len(reference) != 1:
            continue
        reference = reference[0]
        for other in group:
            if other is reference:
                continue
            left, right = reference['final_verified_improvement'], other['final_verified_improvement']
            same_budget = reference.get('proposal_slot_limit') == other.get('proposal_slot_limit')
            result.append({'task_id': task_id, 'reference_run': reference['source_run'],
                           'comparator_run': other['source_run'], 'comparator': other.get('method_label'),
                           'comparator_seed': other.get('seed'), 'same_slot_ceiling': same_budget,
                           'paired_gain_difference': left - right if left is not None and right is not None
                           and same_budget else None,
                           'note': 'Descriptive within-task pair; not an independent statistical sample'})
    return result


def serialize(rows, format='json'):
    if format == 'json':
        return json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output = io.StringIO(newline='')
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict)
                         else ('unknown' if v is None else v) for k, v in row.items()} for row in rows)
    return output.getvalue()


def export_summary(inputs, format='json', output=None):
    rows = summarize_runs(inputs)
    text = serialize(rows, format)
    if output is not None:
        target = Path(output).resolve()
        # Exclusive create: cannot overwrite a raw record or a previous summary.
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='') as stream:
            stream.write(text)
    return text
