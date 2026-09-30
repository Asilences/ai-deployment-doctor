"""Frozen two-task performance pilot with independent planning and fresh final scores."""
import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.core import dump, fingerprint, validate_settings
from inference_lab.final_eval import evaluate_final
from inference_lab.planner import Planner
from inference_lab.proposal_replay import file_sha, read, require
from inference_lab.proposal_trace import utc_now
from inference_lab.runner import execute, provenance, render_report
from inference_lab.summary import serialize, summarize_runs
from independent_search import IndependentPlanner, PlanningIntegrityError, TimedTargetBackend, timing_totals
from planning_service import validate_profile
import run_capacity_replay as capacity
from setup_windows import sha256 as asset_sha

SCRIPT_NAMES = ('independent_search.py', 'run_planner_performance.py', 'audit_planner_performance.py', 'setup_windows.py')


def code_identity():
    return fingerprint({'capacity_dependencies': capacity.code_identity(),
                        'scripts': {name: file_sha(ROOT / 'scripts' / name) for name in SCRIPT_NAMES}})


def build_jobs(spec):
    require(spec['iterations'] == 2 and spec['random_seeds'] == [100, 101, 102], 'Frozen pilot budgets changed')
    require(set(spec['tasks']) == {'serial', 'prefill'}, 'Require both development tasks')
    rng = random.Random(spec['order_seed'])
    tasks = list(spec['tasks'])
    rng.shuffle(tasks)
    jobs = []
    for task in tasks:
        block = [{'task': task, 'method': model, 'seed': 42, 'iterations': 2} for model in ('local-1.5b', 'local-3b')]
        block += [{'task': task, 'method': 'random', 'seed': seed, 'iterations': 2} for seed in spec['random_seeds']]
        block += [{'task': task, 'method': 'fixed', 'seed': 731, 'iterations': 1}]
        rng.shuffle(block)
        jobs.extend(block)
    return [{**j, 'job_id': f'J{i:02d}', 'slot_ceiling': 2} for i, j in enumerate(jobs)]


def prepare(selection, batch):
    spec = read(selection)
    require(spec['protocol'] == 'development-independent-planner-performance-v1', 'Wrong protocol')
    require(spec['representation'] == 'original' and spec['history_mode'] == 'full', 'Input intervention changed')
    jobs = build_jobs(spec)
    settings = {k: read(ROOT / v) for k, v in spec['tasks'].items()}
    for s in settings.values():
        validate_settings(s)
    profiles = {k: validate_profile(read(ROOT / v), ROOT) for k, v in spec['planner_profiles'].items()}
    require(set(profiles) == {'local-1.5b', 'local-3b'}, 'Missing planning profile')
    require(all(s['port'] != p['port'] and s['max_model_len'] == p['max_model_len']
                for s in settings.values() for p in profiles.values()), 'Overlapping endpoint or context mismatch')
    assets = {p['model_path']: p for p in profiles.values()}
    for s in settings.values():
        require(s['model_path'] in assets and s['model_sha256'] == assets[s['model_path']]['model_sha256'], 'Target identity changed')
    for path, p in assets.items():
        require((ROOT / path).stat().st_size == p['model_size_bytes'] and asset_sha(ROOT / path) == p['model_sha256'], 'Model asset mismatch')
    guard = {str((ROOT / path).resolve()): file_sha(ROOT / path)
             for path in list(spec['tasks'].values()) + list(spec['planner_profiles'].values())}
    # Protect historical principal records; they never enter this pilot's planning inputs.
    for pattern in ('run.json', 'best.json', 'final.json', 'manifest.json'):
        for path in (ROOT / 'runs').rglob(pattern):
            guard[str(path)] = file_sha(path)
    actual = provenance(ROOT / 'inference_lab', 'llama_cpp')
    require(actual['git_dirty'] is False, 'Prepare requires a clean implementation commit')
    identity = code_identity()
    # Include controller identity in the persisted target-system identity, shared by ALL methods.
    # Original settings cannot inherit these states; final uses this same augmented identity.
    for s in settings.values():
        s.update(experiment_controller_sha256=identity, experiment_protocol=spec['protocol'])
    plan = {'selection': spec, 'selection_sha256': file_sha(selection), 'protocol': spec['protocol'],
            'code_identity': identity, 'implementation_commit': actual['git_commit'], 'jobs': jobs,
            'settings': settings, 'planner_profiles': profiles, 'readonly_guard': guard,
            'candidate_execution_ceiling': sum(j['iterations'] for j in jobs), 'llm_call_ceiling': 16}
    batch = Path(batch).resolve()
    require(batch.is_relative_to(ROOT / 'runs'), 'Output outside runs')
    batch.mkdir(parents=True, exist_ok=False)
    dump(batch / 'manifest.json', {'plan': plan, 'plan_sha256': fingerprint(plan), 'status': 'prepared',
                                 'jobs': [{**j, 'status': 'pending'} for j in jobs], 'sessions': [], 'created_utc': utc_now()})
    return batch


def annotate_search(directory, target, plan, method):
    run = read(directory / 'run.json')
    records = [read(p) for p in sorted(directory.glob('proposal-*/planning.json'))]
    run['independent_controller'] = {'code_identity': plan['code_identity'], 'method': method,
                                    'implementation_commit': plan['implementation_commit'],
                                    'planning_profile': plan['planner_profiles'].get(method),
                                    'representation': 'original', 'history_mode': 'full'}
    run['costs'].update(planner_setup_seconds=sum(r['setup_seconds'] or 0 for r in records),
                        planning_wall_seconds=sum(r['elapsed_seconds'] for r in records),
                        planning_shutdown_seconds=sum(r['shutdown_seconds'] or 0 for r in records),
                        preflight_seconds=sum((r['result'] or {}).get('costs', {}).get('preflight_seconds', 0) for r in records),
                        target_operations_seconds=timing_totals(target.events))
    dump(directory / 'target_timing.json', {'events': target.events, 'totals': timing_totals(target.events),
                                         'note': 'Top-level operation durations only; nested stop included in start, never added twice'})
    dump(directory / 'run.json', run)
    render_report(directory, run)


def summary(batch, manifest):
    rows = summarize_runs([batch])
    jobs = {str((Path(batch) / (j['job_id'] + '-search')).resolve()): j for j in manifest['jobs']}
    for row in rows:
        job = jobs[row['source_run']]
        row.update(job_id=job['job_id'], task=job['task'], method_label=job['method'],
                   proposal_slot_limit=job['slot_ceiling'], job_status=job['status'],
                   job_wall_seconds=job.get('elapsed_seconds'), final_wall_seconds=job.get('final_elapsed_seconds'))
        if row['final_source']:
            final = read(row['final_source'])
            row.update(final_baseline=final['baseline_median'], final_selected=final['selected_median'])
    return rows


def execute_block(batch, max_seconds=900):
    require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 900, 'Invalid block budget')
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    plan = manifest['plan']
    require(fingerprint(plan) == manifest['plan_sha256'] and code_identity() == plan['code_identity'], 'Frozen plan or source changed')
    actual = provenance(ROOT / 'inference_lab', 'llama_cpp')
    require(actual['git_dirty'] is False and actual['git_commit'] == plan['implementation_commit'], 'Use clean prepared commit')
    require(manifest['status'] in ('prepared', 'partial'), 'Batch already finished or unsafe')
    require(len(manifest['jobs']) == len(plan['jobs']), 'Job count changed')
    for job, frozen in zip(manifest['jobs'], plan['jobs']):
        require(all(job[k] == v for k, v in frozen.items()), 'Job plan changed')
        if job['status'] in ('searching', 'final_running'):
            job['status'] = 'interrupted_not_retried'
    def guarded():
        return all(Path(path).is_file() and file_sha(path) == sha for path, sha in plan['readonly_guard'].items())
    require(guarded(), 'Historical evidence or config changed')
    for profile in plan['planner_profiles'].values():
        require(asset_sha(ROOT / profile['model_path']) == profile['model_sha256'], 'Planning bytes changed')
    started = time.monotonic()
    session = {'started_utc': utc_now(), 'max_seconds': max_seconds, 'runtime_provenance': actual}
    manifest['sessions'].append(session)
    manifest['status'] = 'running'
    def save():
        dump(batch / 'manifest.json', manifest)
    save()
    try:
        for job in manifest['jobs']:
            if job['status'] not in ('pending', 'search_completed'):
                continue
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0 or (job['status'] == 'pending' and remaining < plan['selection']['minimum_search_start_seconds']):
                session['stop_reason'] = 'time_block_reached'
                break
            require(guarded(), 'Historical evidence changed during execution')
            s = plan['settings'][job['task']]
            search = batch / (job['job_id'] + '-search')
            final_dir = batch / (job['job_id'] + '-final')
            job.update(search=str(search), final=str(final_dir))
            job_start = time.monotonic()
            carried = job.get('elapsed_seconds', 0)
            target = None
            try:
                if job['status'] == 'pending':
                    require(not search.exists() and not final_dir.exists(), 'Refuse to overwrite spent job')
                    target = TimedTargetBackend(s, ROOT)
                    local = job['method'].startswith('local-')
                    planner = IndependentPlanner(s, plan['planner_profiles'][job['method']], ROOT, target, search) if local else Planner(
                        job['method'], s, ROOT, seed=job['seed'])
                    job['status'] = 'searching'
                    save()
                    try:
                        execute(s, target, planner, search, job['iterations'], max_seconds=remaining)
                    finally:
                        if (search / 'run.json').exists():
                            annotate_search(search, target, plan, job['method'])
                    job['status'] = 'search_completed'
                    save()
                if time.monotonic() - started >= max_seconds:
                    break
                before = file_sha(search / 'best.json')
                job['status'] = 'final_running'
                save()
                target_final = TimedTargetBackend(s, ROOT)
                final_start = time.monotonic()
                try:
                    final = evaluate_final(s, target_final, search / 'best.json', final_dir)
                finally:
                    job['final_elapsed_seconds'] = time.monotonic() - final_start
                    dump(final_dir / 'target_timing.json', {'events': target_final.events, 'totals': timing_totals(target_final.events)})
                require(before == file_sha(search / 'best.json'), 'Final changed selected state')
                job.update(status='completed', final_valid=final['valid'], final_gain=final['throughput_gain'])
                print(job['job_id'], job['task'], job['method'], job['seed'], 'final:', job['final_gain'], flush=True)
            except (OSError, RuntimeError, ValueError, KeyError, PlanningIntegrityError) as error:
                job.update(status='failed', error_type=type(error).__name__, error_message=str(error))
                if isinstance(error, PlanningIntegrityError):
                    session['stop_reason'] = 'planning_integrity_failure'
                    break
            finally:
                job['elapsed_seconds'] = carried + time.monotonic() - job_start
                save()
    finally:
        session.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - started, original_files_unchanged=guarded())
        manifest['status'] = 'completed' if all(j['status'] in ('completed', 'failed', 'interrupted_not_retried') for j in manifest['jobs']) else 'partial'
        if not session['original_files_unchanged']:
            manifest['status'] = 'source_guard_failed'
        save()
        rows = summary(batch, manifest)
        dump(batch / 'summary.json', rows)
        (batch / 'summary.csv').write_text(serialize(rows, 'csv'), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path)
    action.add_argument('--execute', type=Path)
    action.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=float, default=900)
    args = parser.parse_args()
    if args.prepare:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        print(prepare(args.prepare, args.output or ROOT / 'runs' / ('planner-performance-' + stamp)))
    elif args.execute:
        print('Batch status:', execute_block(args.execute, args.max_seconds)['status'])
    else:
        print(json.dumps(summary(args.summary, read(args.summary / 'manifest.json')), indent=2))
