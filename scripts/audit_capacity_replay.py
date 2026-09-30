"""Read-only capacity integrity audit, including actual independent planner selectors."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.core import fingerprint
from inference_lab.proposal_replay import extract_snapshot, file_sha, read, require, validate_snapshot
from audit_proposal_replay import audit_result as legacy_audit_result
from planning_service import service_identity, validate_profile
import run_capacity_replay as runner


def audit_result(job, result, snapshot, profile, root):
    identity = service_identity(profile, root)
    require(result['version'] == 'capacity-proposal-v1', 'Wrong proposal version')
    require(result['planner_service'] == identity, 'Planner identity changed')
    require(job['planner_id'] == profile['service_id'], 'Wrong planning service')
    require(profile['max_model_len'] == snapshot['decision']['settings']['max_model_len'], 'Protocol context differs')
    normalized = copy.deepcopy(result)
    for original, call in zip(result['proposal_calls'], normalized['proposal_calls']):
        request = original['request']
        require(request['planner_service'] == identity, 'Call endpoint or planner identity changed')
        require(request['payload']['model'] == identity['model_selector'], 'Call uses the target instead of planner')
        require(fingerprint(request['payload']) == request['payload_sha256'], 'Actual payload hash mismatch')
        # Reuse unchanged schema/evidence/cost audit on a copy, only after checking the actual selector.
        call['request']['payload']['model'] = str((root / snapshot['decision']['settings']['model_path']).resolve())
        call['request']['payload_sha256'] = fingerprint(call['request']['payload'])
    checked = legacy_audit_result(job, normalized, snapshot, root)
    return {**checked, 'planner_id': profile['service_id'], 'actual_planner_identity_verified': True}


def audit_batch(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    plan = manifest['plan']
    require(manifest['status'] == 'completed', 'Require a completed capacity batch')
    require(plan['protocol'] == 'development-planner-capacity-v1', 'Wrong protocol')
    require(fingerprint(plan) == manifest['plan_sha256'], 'Frozen plan changed')
    require(runner.code_identity() == plan['code_identity'], 'Use recorded capacity implementation')
    require(manifest['original_files_unchanged'], 'Original evidence changed')
    for path, sha in plan['readonly_guard'].items():
        require(file_sha(path) == sha, 'Source or profile changed')
    profiles = {p['service_id']: validate_profile(p, ROOT) for p in plan['planner_profiles']}
    require(set(profiles) == set(plan['planner_order']) and len(profiles) == 2, 'Planner order mismatch')
    require([s['planner_id'] for s in manifest['sessions']] == plan['planner_order'], 'Missing or repeated service block')
    for session in manifest['sessions']:
        profile = profiles[session['planner_id']]
        require(session['planner_service'] == service_identity(profile, ROOT), 'Session identity mismatch')
        require(session['runtime_provenance']['git_commit'] == plan['implementation_commit']
                and session['runtime_provenance']['git_dirty'] is False and session['original_files_unchanged'], 'Runtime provenance mismatch')
        require(not session.get('error_type'), 'Service failed')
        require(session['props']['default_generation_settings']['n_ctx'] == profile['max_model_len'], 'Service context mismatch')
        backend = runner.legacy.ReplayBackend(profile, ROOT)
        expected = backend.serve_command(plan['service_config'])
        require(read(batch / ('service-' + session['planner_id']) / 'server_command.json') == expected, 'Actual service command mismatch')
        model = ROOT / profile['model_path']
        require(model.stat().st_size == profile['model_size_bytes'] and file_sha(model) == profile['model_sha256'], 'Actual planner bytes mismatch')
    snapshots = {}
    for item in plan['snapshots']:
        path = batch / item['file']
        require(file_sha(path) == item['file_sha256'], 'Snapshot bytes changed')
        snapshot = read(path)
        validate_snapshot(snapshot)
        require(snapshot['decision_sha256'] == item['decision_sha256'], 'Decision hash mismatch')
        require(extract_snapshot(snapshot['source']['run_directory'], snapshot['source']['trial_number'], ROOT) == snapshot,
                'Snapshot no longer reconstructs from original evidence')
        snapshots[item['snapshot_id']] = snapshot
    require(len(manifest['jobs']) == len(plan['jobs']), 'Job count mismatch')
    rows = []
    for job, frozen in zip(manifest['jobs'], plan['jobs']):
        require(all(job[k] == v for k, v in frozen.items()), 'Frozen job changed')
        result = read(batch / (job['job_id'] + '.json'))
        require(job['status'] == result['status'] and job['status'] in ('valid', 'failed'), 'Slot state mismatch')
        rows.append(audit_result(job, result, snapshots[job['snapshot_id']], profiles[job['planner_id']], ROOT))
    require(len(rows) == plan['proposal_slot_ceiling'] and sum(r['llm_calls'] for r in rows) <= plan['llm_call_ceiling'], 'Budget exceeded')
    require(read(batch / 'summary.json') == runner.summarize(batch, manifest), 'Summary mismatch')
    return {'integrity_verified': True, 'jobs_verified': len(rows), 'jobs': rows,
            'original_files_unchanged': True, 'target_experiments': 0,
            'note': 'Integrity does not establish semantic correctness, model-size causality or performance gain'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    print(json.dumps(audit_batch(parser.parse_args().batch), indent=2))
