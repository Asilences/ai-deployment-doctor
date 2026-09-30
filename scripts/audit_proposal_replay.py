"""Read-only replay integrity checks. Does not validate semantic claims or performance."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.candidate_ids import candidate_catalog, parse_id
from inference_lab.core import fingerprint
from inference_lab.proposal_replay import (extract_snapshot, file_sha, input_variant, payload_for,
    read, require, validate_snapshot)
from inference_lab.proposal_trace import trace_totals

SPEC = importlib.util.spec_from_file_location('replay_audit_runner', ROOT / 'scripts/run_proposal_replay.py')
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def audit_result(job, result, snapshot, repo_root):
    require(result['representation'] == job['representation'] and result['history_mode'] == job['history_mode'],
            'Variant mismatch')
    require(result['decision_sha256'] == snapshot['decision_sha256'], 'Decision mismatch')
    require(result['candidate_executed'] is False, 'Replay executed a target candidate')
    brief = input_variant(snapshot, job['representation'], job['history_mode'])
    require(fingerprint(brief) == job['input_sha256'], 'Input mismatch')
    calls = result['proposal_calls']
    require(len(calls) <= 2, 'Call ceiling exceeded')
    expected = payload_for(brief, snapshot['decision']['settings'], repo_root)
    parsed = None
    for i, call in enumerate(calls):
        require(call['call_index'] == i + 1, 'Call index mismatch')
        require(call['request']['payload'] == expected and call['request']['payload_sha256'] == fingerprint(expected),
                'Payload or sampling changed')
        require(call['request']['evidence'] == brief and call['request']['available_candidates'] == brief['available_candidates'],
                'Recorded evidence changed')
        check = result['preflight'][i]
        require(check['status'] == 'passed' and check['prompt_tokens'] + expected['max_tokens'] <= check['context_limit'],
                'Context capacity mismatch')
        require(check['context_limit'] == snapshot['decision']['settings']['max_model_len'], 'Changed context limit')
        require(call['elapsed_seconds'] >= 0 and call['ended_utc'] >= call['started_utc'], 'Call timing mismatch')
        if call['usage']['prompt_tokens'] is not None:
            matches = call['usage']['prompt_tokens'] == check['prompt_tokens']
            require(call['prompt_count_matches_preflight'] == matches, 'Prompt guard mismatch')
        if call['transport_status'] == 'received' and call['failure_category'] != 'prompt_token_count_mismatch':
            try:
                parsed = parse_id({'choices': [{'message': {'content': call['response_content']}}]},
                                  candidate_catalog(snapshot['decision']['settings']), brief['available_candidates'], 96)
            except ValueError as error:
                parsed = None
                category = 'duplicate_candidate' if str(error).startswith('Duplicate') else 'invalid_output'
                require(call['parse_result'] == 'rejected' and call['failure_category'] == category, 'Parse classification mismatch')
            else:
                require(call['parse_result'] == 'valid', 'Valid response marked invalid')
        if i + 1 < len(calls):
            require(call['failure_category'] in ('invalid_output', 'duplicate_candidate'), 'Unapproved retry')
            expected['messages'].append({'role': 'user', 'content':
                'Previous response was rejected: ' + call['failure_category'] +
                '. Select one available candidate ID and return exactly the requested JSON fields. '
                'Available candidates: ' + json.dumps(brief['available_candidates'])})
    require((result['proposal'] == parsed and parsed is not None) if result['status'] == 'valid'
            else result['proposal'] is None, 'Proposal/status mismatch')
    for k, value in trace_totals(calls).items():
        require(result['costs'][k] == value, 'Cost mismatch: ' + k)
    require(result['costs']['preflight_seconds'] == sum(c['elapsed_seconds'] for c in result['preflight']),
            'Inspection cost mismatch')
    return {'job_id': job['job_id'], 'candidate_id': (result['proposal'] or {}).get('candidate_id'),
            'llm_calls': len(calls), 'total_tokens': result['costs']['total_tokens']}


def audit_batch(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    plan = manifest['plan']
    require(manifest['status'] == 'completed', 'Require a completed replay')
    require(fingerprint(plan) == manifest['plan_sha256'], 'Plan hash mismatch')
    require(REPLAY.code_identity() == plan['code_identity'], 'Use the recorded replay implementation')
    require(all(s['runtime_provenance']['git_commit'] == plan['implementation_commit']
                and s['runtime_provenance']['git_dirty'] is False and s['original_files_unchanged']
                for s in manifest['sessions']), 'Runtime provenance mismatch')
    for path, sha in plan['readonly_guard'].items():
        require(file_sha(path) == sha, 'Original file changed')
    snapshots = {}
    for item in plan['snapshots']:
        path = batch / item['file']
        require(file_sha(path) == item['file_sha256'], 'Frozen snapshot changed')
        snapshot = read(path)
        validate_snapshot(snapshot)
        require(snapshot['decision_sha256'] == item['decision_sha256'], 'Frozen decision changed')
        rebuilt = extract_snapshot(snapshot['source']['run_directory'], snapshot['source']['trial_number'], ROOT)
        require(rebuilt == snapshot, 'Snapshot no longer matches original pre-proposal state')
        snapshots[item['snapshot_id']] = snapshot
    require(len(manifest['jobs']) == len(plan['jobs']), 'Job count mismatch')
    rows = []
    for job, frozen in zip(manifest['jobs'], plan['jobs']):
        require(all(job[k] == v for k, v in frozen.items()), 'Job plan changed')
        require(job['status'] in ('valid', 'failed'), 'Unfinished job')
        result = read(batch / (job['job_id'] + '.json'))
        require(result['status'] == job['status'], 'Job result status mismatch')
        rows.append(audit_result(job, result, snapshots[job['snapshot_id']], ROOT))
    require(len(rows) == plan['proposal_slot_ceiling'] and sum(r['llm_calls'] for r in rows) <= plan['llm_call_ceiling'],
            'Budget mismatch')
    require(read(batch / 'summary.json') == REPLAY.summarize_batch(batch, manifest), 'Summary mismatch')
    return {'integrity_verified': True, 'jobs': rows, 'original_files_unchanged': True,
            'note': 'Integrity is not semantic grounding, history usefulness or performance superiority'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    print(json.dumps(audit_batch(parser.parse_args().batch), ensure_ascii=False, indent=2))
