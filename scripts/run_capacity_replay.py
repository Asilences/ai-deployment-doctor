"""Compare independent planning services on identical, read-only target snapshots."""
import argparse
import csv
import json
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.core import dump, fingerprint
from inference_lab.proposal_replay import file_sha, input_variant, read, require, validate_snapshot
from inference_lab.proposal_trace import trace_totals, utc_now
from inference_lab.runner import provenance
import run_proposal_replay as legacy
from audit_proposal_replay import audit_batch as audit_legacy
from capacity_proposal import capacity_proposal
from planning_service import service_identity, validate_profile

IMPLEMENTATION_FILES = ('planning_service.py', 'capacity_proposal.py', 'run_capacity_replay.py',
                        'audit_capacity_replay.py')


def code_identity():
    return fingerprint({'legacy_identity': legacy.code_identity(),
                        'scripts': {name: file_sha(ROOT / 'scripts' / name) for name in IMPLEMENTATION_FILES}})


def prepare(selection, output):
    spec = read(selection)
    require(spec['protocol'] == 'development-planner-capacity-v1', 'Unsupported capacity selection')
    require(0 < spec['max_seconds'] <= 900, 'Invalid time ceiling')
    source = (ROOT / spec['source_batch']).resolve()
    require(source.is_relative_to(ROOT / 'runs'), 'Source outside local runs')
    audit_legacy(source)
    profiles = [validate_profile(read(ROOT / name), ROOT) for name in spec['planner_profiles']]
    require(len(profiles) == 2 and len({p['service_id'] for p in profiles}) == 2, 'Require two distinct planners')
    require(len({p['port'] for p in profiles}) == 2, 'Planner ports must differ')
    runtime_keys = ('llama_cpp_version', 'llama_cpp_binary', 'max_model_len', 'batch_size', 'gpu_layers',
                    'cpu_threads', 'startup_timeout_s')
    require(all(p[k] == profiles[0][k] for p in profiles for k in runtime_keys), 'Mixed runtime settings')
    selection_path = (ROOT / spec['source_selection']).resolve()
    original = read(source / 'manifest.json')['plan']
    require(file_sha(selection_path) == original['selection_sha256'], 'Source selection changed')
    output = legacy.prepare(selection_path, output)
    manifest = read(output / 'manifest.json')
    plan = manifest['plan']
    for frozen, previous in zip(plan['snapshots'], original['snapshots']):
        require(frozen == previous, 'Source snapshot differs from previous frozen state')
        path = source / previous['file']
        plan['readonly_guard'][str(path)] = file_sha(path)
    require(len(plan['snapshots']) == len(original['snapshots']), 'State count changed')
    require(all(p['max_model_len'] == read(output / s['file'])['decision']['settings']['max_model_len']
                for p in profiles for s in plan['snapshots']), 'Planning context must match the frozen protocol')
    for name in ('manifest.json', 'summary.json'):
        path = source / name
        plan['readonly_guard'][str(path)] = file_sha(path)
    target_ports = {read(output / s['file'])['decision']['settings']['port'] for s in plan['snapshots']}
    require(not target_ports.intersection(p['port'] for p in profiles), 'Planner port overlaps target')
    for name in spec['planner_profiles']:
        path = (ROOT / name).resolve()
        plan['readonly_guard'][str(path)] = file_sha(path)
    order = [p['service_id'] for p in profiles]
    random.Random(spec['order_seed']).shuffle(order)
    old_jobs = plan['jobs']
    jobs = []
    # Identical variant order for each model, keeping each state within a model block.
    for service_id in order:
        for item in old_jobs:
            jobs.append({**item, 'job_id': f'J{len(jobs):02d}', 'planner_id': service_id})
    plan.update(protocol=spec['protocol'], selection=spec, selection_sha256=file_sha(selection),
                code_identity=code_identity(), planner_profiles=profiles, planner_order=order,
                jobs=jobs, proposal_slot_ceiling=len(jobs), llm_call_ceiling=len(jobs) * 2,
                note='Planning model identity comparison only; target evidence and verifier unchanged; no performance scores')
    manifest.update(plan=plan, plan_sha256=fingerprint(plan),
                    jobs=[{**j, 'status': 'pending'} for j in jobs])
    dump(output / 'manifest.json', manifest)
    return output


def summarize(batch, manifest):
    rows, all_calls, by_planner = [], [], {}
    for job in manifest['jobs']:
        path = Path(batch) / (job['job_id'] + '.json')
        if not path.exists():
            continue
        result = read(path)
        all_calls.extend(result['proposal_calls'])
        rows.append({**{k: job[k] for k in ('job_id', 'snapshot_id', 'representation', 'history_mode', 'planner_id')},
                     'status': result['status'], 'candidate_id': (result['proposal'] or {}).get('candidate_id'),
                     'costs': result['costs'], 'candidate_executed': False})
    for service_id in manifest['plan']['planner_order']:
        selected = [r for r in rows if r['planner_id'] == service_id]
        calls = [c for r in selected for c in read(Path(batch) / (r['job_id'] + '.json'))['proposal_calls']]
        by_planner[service_id] = {'slots_recorded': len(selected), 'valid_slots': sum(r['status'] == 'valid' for r in selected),
                                 'costs': trace_totals(calls)}
    costs = trace_totals(all_calls)
    costs.update(preflight_seconds=sum(r['costs']['preflight_seconds'] for r in rows),
                 service_setup_seconds=sum(s['service_setup_seconds'] or 0 for s in manifest['sessions']),
                 execution_wall_clock_seconds=manifest.get('execution_elapsed_seconds'))
    return {'rows': rows, 'by_planner': by_planner, 'costs': costs,
            'candidate_experiments_consumed': 0, 'final_scores_generated': 0,
            'semantic_labels': 'require human review; legal JSON alone is not grounding',
            'note': 'Single serial model block per identity; timing and model-size causality are not established'}


def write_review(batch, manifest):
    with (batch / 'review_template.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        keys = ['job_id', 'snapshot_id', 'representation', 'history_mode', 'planner_id']
        writer = csv.DictWriter(stream, fieldnames=keys + ['field', 'text', 'label', 'source_config_id', 'review_note'])
        writer.writeheader()
        for job in manifest['jobs']:
            path = batch / (job['job_id'] + '.json')
            if path.exists() and read(path)['proposal']:
                proposal = read(path)['proposal']
                for field in ('hypothesis', 'expected_effect'):
                    writer.writerow({k: job[k] for k in keys} | {'field': field, 'text': proposal[field]})


def execute(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    plan = manifest['plan']
    require(manifest['status'] == 'prepared', 'Cannot repeat this capacity session; prepare a new batch')
    require(fingerprint(plan) == manifest['plan_sha256'] and plan['code_identity'] == code_identity(), 'Frozen implementation changed')
    actual = provenance(ROOT / 'inference_lab', 'llama_cpp')
    require(actual['git_dirty'] is False and actual['git_commit'] == plan['implementation_commit'], 'Require clean prepared commit')
    require(manifest['jobs'] == [{**j, 'status': 'pending'} for j in plan['jobs']], 'Prepared jobs changed')
    require(not any((batch / (j['job_id'] + '.json')).exists() for j in plan['jobs']), 'Existing result cannot be overwritten')
    snapshots = {}
    for item in plan['snapshots']:
        path = batch / item['file']
        require(file_sha(path) == item['file_sha256'], 'Snapshot changed')
        snapshots[item['snapshot_id']] = read(path)
        validate_snapshot(snapshots[item['snapshot_id']])
    profiles = {p['service_id']: validate_profile(p, ROOT) for p in plan['planner_profiles']}
    def guarded():
        return all(Path(p).is_file() and file_sha(p) == sha for p, sha in plan['readonly_guard'].items())
    require(guarded(), 'Original evidence changed')
    for job in plan['jobs']:
        brief = input_variant(snapshots[job['snapshot_id']], job['representation'], job['history_mode'])
        require(fingerprint(brief) == job['input_sha256'], 'Input changed')
    # Verify all assets before consuming any generation calls.
    for p in profiles.values():
        model = ROOT / p['model_path']
        require(model.stat().st_size == p['model_size_bytes'] and file_sha(model) == p['model_sha256'], 'Planner asset changed')
    started = time.monotonic()
    manifest.update(status='running', execution_started_utc=utc_now())
    dump(batch / 'manifest.json', manifest)
    try:
        for service_id in plan['planner_order']:
            if time.monotonic() - started >= plan['selection']['max_seconds']:
                manifest['stop_reason'] = 'time_budget_reached'
                break
            profile = profiles[service_id]
            session = {'planner_id': service_id, 'planner_service': service_identity(profile, ROOT),
                       'started_utc': utc_now(), 'runtime_provenance': actual, 'service_setup_seconds': None}
            manifest['sessions'].append(session)
            backend = None
            block_start = time.monotonic()
            try:
                backend = legacy.ReplayBackend(profile, ROOT)
                setup = time.monotonic()
                try:
                    backend.start(plan['service_config'], batch / ('service-' + service_id))
                finally:
                    session['service_setup_seconds'] = time.monotonic() - setup
                props = backend._json('/props')
                require(props['default_generation_settings']['n_ctx'] == profile['max_model_len'], 'Wrong planning context')
                session['props'] = props
                def preflight(payload):
                    prompt = backend._json('/apply-template', payload, timeout=30)['prompt']
                    tokens = backend._json('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True}, timeout=30)['tokens']
                    return {'method': 'apply-template/tokenize', 'prompt_tokens': len(tokens),
                            'prompt_sha256': fingerprint(prompt), 'context_limit': profile['max_model_len']}
                for job in manifest['jobs']:
                    if job['planner_id'] != service_id:
                        continue
                    if time.monotonic() - started >= plan['selection']['max_seconds']:
                        manifest['stop_reason'] = 'time_budget_reached'
                        break
                    job['status'] = 'running'
                    dump(batch / 'manifest.json', manifest)
                    result = capacity_proposal(snapshots[job['snapshot_id']], job['representation'], job['history_mode'],
                        profile, ROOT, lambda payload: backend._json('/v1/chat/completions', payload, timeout=90), preflight)
                    dump(batch / (job['job_id'] + '.json'), result)
                    job.update(status=result['status'], result_file=job['job_id'] + '.json')
                    dump(batch / 'manifest.json', manifest)
                    print(job['job_id'], service_id, job['snapshot_id'], job['representation'], job['history_mode'],
                          result['status'], (result['proposal'] or {}).get('candidate_id'), flush=True)
                    if result['status'] == 'interrupted' or any(c['failure_category'] == 'prompt_token_count_mismatch' for c in result['proposal_calls']):
                        manifest['stop_reason'] = 'interruption_or_prompt_guard'
                        break
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                session.update(error_type=type(error).__name__, error_message=str(error))
                manifest['stop_reason'] = 'service_failure'
            finally:
                if backend is not None:
                    backend.stop()
                session.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - block_start,
                               original_files_unchanged=guarded())
                if not session['original_files_unchanged']:
                    manifest['stop_reason'] = 'source_guard_failed'
                dump(batch / 'manifest.json', manifest)
            if manifest.get('stop_reason'):
                break
    finally:
        manifest.update(status='completed' if all(j['status'] in ('valid', 'failed') for j in manifest['jobs']) else 'partial',
                        execution_ended_utc=utc_now(), execution_elapsed_seconds=time.monotonic() - started,
                        original_files_unchanged=guarded())
        if not manifest['original_files_unchanged']:
            manifest['status'] = 'source_guard_failed'
        dump(batch / 'manifest.json', manifest)
        dump(batch / 'summary.json', summarize(batch, manifest))
        write_review(batch, manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path)
    action.add_argument('--execute', type=Path)
    action.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        output = args.output or ROOT / 'runs' / ('capacity-replay-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        print(prepare(args.prepare, output))
    elif args.execute:
        print('Batch status:', execute(args.execute)['status'])
    else:
        print(json.dumps(summarize(args.summary, read(args.summary / 'manifest.json')), indent=2))
