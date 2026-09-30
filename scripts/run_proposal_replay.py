"""Freeze real pre-proposal snapshots, then replay inputs without performance searches."""
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
from inference_lab.llama_backend import LlamaCppBackend
from inference_lab.proposal_replay import (HISTORY_MODES, REPRESENTATIONS, extract_snapshot, file_sha,
    input_variant, read, replay_proposal, require, validate_snapshot)
from inference_lab.proposal_trace import trace_totals, utc_now
from inference_lab.runner import provenance


def code_identity():
    paths = sorted((ROOT / 'inference_lab').glob('*.py')) + [Path(__file__)]
    return fingerprint({str(p.relative_to(ROOT)): file_sha(p) for p in paths})


def prepare(selection, output):
    spec = read(selection)
    require(spec['protocol'] == 'development-same-state-replay-v1', 'Unsupported selection')
    require(1 <= len(spec['snapshots']) <= 6, 'Require one to six source states')
    require(0 < spec['max_seconds'] <= 900, 'Require at most 900 seconds per session')
    sources, guard = [], {}
    for item in spec['snapshots']:
        source = (ROOT / item['run_directory']).resolve()
        require(source.is_relative_to(ROOT / 'runs'), 'Source must be a local run')
        snap = extract_snapshot(source, item['trial_number'], ROOT)
        guard.update(snap['source']['files'])
        best = source / 'best.json'
        if best.exists():
            # Only a byte hash for the write guard; contents never enter decision inputs.
            guard[str(best)] = file_sha(best)
        sources.append(snap)
    require(len({s['decision_sha256'] for s in sources}) == len(sources), 'Duplicate decision states')
    service_keys = ('backend', 'model_path', 'model_revision', 'model_sha256', 'llama_cpp_version',
                    'llama_cpp_binary', 'port', 'max_model_len', 'gpu_layers', 'cpu_threads', 'batch_size')
    base = sources[0]['decision']['settings']
    require(all(all(s['decision']['settings'][k] == base[k] for k in service_keys) for s in sources),
            'Mixed planning service identities are unsupported')
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / 'runs'), 'Output must be a new local runs directory')
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(spec['order_seed'])
    frozen, jobs = [], []
    for i, (snapshot, item) in enumerate(zip(sources, spec['snapshots'])):
        label = f'S{i:02d}'
        target = output / (label + '.json')
        dump(target, snapshot)
        frozen.append({'snapshot_id': label, 'label': item['label'], 'file': target.name,
                       'file_sha256': file_sha(target), 'decision_sha256': snapshot['decision_sha256']})
        variants = [(r, h) for r in REPRESENTATIONS for h in HISTORY_MODES]
        rng.shuffle(variants)
        for representation, mode in variants:
            # Freeze exact inputs before any outputs; four variants share one decision hash.
            brief = input_variant(snapshot, representation, mode)
            jobs.append({'job_id': f'J{len(jobs):02d}', 'snapshot_id': label,
                         'representation': representation, 'history_mode': mode,
                         'input_sha256': fingerprint(brief)})
    plan = {'protocol': spec['protocol'], 'selection': spec, 'selection_sha256': file_sha(selection),
            'code_identity': code_identity(), 'implementation_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
            'snapshots': frozen, 'jobs': jobs, 'readonly_guard': guard,
            'proposal_slot_ceiling': len(jobs), 'llm_call_ceiling': 2 * len(jobs),
            'service_config': {'parallel': 1, 'ubatch_size': 128},
            'note': 'Development decisions only; no new performance scores or promotion'}
    manifest = {'plan': plan, 'plan_sha256': fingerprint(plan), 'status': 'prepared',
                'created_utc': utc_now(), 'jobs': [{**j, 'status': 'pending'} for j in jobs], 'sessions': []}
    dump(output / 'manifest.json', manifest)
    return output


def summarize_batch(batch, manifest):
    batch = Path(batch)
    rows, calls, pairs = [], [], []
    for job in manifest['jobs']:
        result_path = batch / (job['job_id'] + '.json')
        if not result_path.exists():
            continue
        result = read(result_path)
        calls.extend(result['proposal_calls'])
        rows.append({'job_id': job['job_id'], 'snapshot_id': job['snapshot_id'],
                     'representation': job['representation'], 'history_mode': job['history_mode'],
                     'status': result['status'], 'candidate_id': (result['proposal'] or {}).get('candidate_id'),
                     'elapsed_seconds': result['elapsed_seconds'], 'costs': result['costs'],
                     'error_type': result.get('error_type'), 'candidate_executed': False})
    for snap in manifest['plan']['snapshots']:
        subset = [r for r in rows if r['snapshot_id'] == snap['snapshot_id']]
        for representation in REPRESENTATIONS:
            a = next((r for r in subset if r['representation'] == representation and r['history_mode'] == 'full'), None)
            b = next((r for r in subset if r['representation'] == representation and r['history_mode'] == 'no_history'), None)
            valid = a is not None and b is not None and a['status'] == b['status'] == 'valid'
            pairs.append({'snapshot_id': snap['snapshot_id'], 'representation': representation,
                          'both_valid': valid, 'history_changes_id': a['candidate_id'] != b['candidate_id'] if valid else None})
    costs = trace_totals(calls)
    costs.update(preflight_seconds=sum(r['costs']['preflight_seconds'] for r in rows),
                 session_wall_clock_seconds=sum(s['elapsed_seconds'] for s in manifest['sessions']
                                                if 'elapsed_seconds' in s),
                 service_setup_seconds=sum(s['service_setup_seconds'] for s in manifest['sessions']
                                           if s['service_setup_seconds'] is not None))
    return {'rows': rows, 'history_pairs': pairs, 'costs': costs,
            'valid_slots': sum(r['status'] == 'valid' for r in rows), 'slots_recorded': len(rows),
            'candidate_experiments_consumed': 0, 'semantic_labels': 'require explicit human review',
            'note': 'Changed IDs and valid JSON are not evidence of performance improvement'}


class ReplayBackend(LlamaCppBackend):
    def serve_command(self, candidate):
        return super().serve_command(candidate) + ['--no-context-shift']


def execute(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    plan = manifest['plan']
    require(fingerprint(plan) == manifest['plan_sha256'], 'Frozen plan was modified')
    require(plan['code_identity'] == code_identity(), 'Replay implementation changed since preparation')
    require(manifest['status'] in ('prepared', 'partial'), 'Batch cannot be re-executed')
    actual = provenance(ROOT / 'inference_lab', 'llama_cpp')
    require(actual['git_dirty'] is False and actual['git_commit'] == plan['implementation_commit'],
            'Execution requires the clean prepared implementation commit')
    require(len(manifest['jobs']) == len(plan['jobs']), 'Job count changed')
    for job, frozen in zip(manifest['jobs'], plan['jobs']):
        require(all(job[k] == v for k, v in frozen.items()), 'Frozen job changed')
        require(job['status'] in ('pending', 'valid', 'failed', 'interrupted'), 'Unfinished attempt cannot be resumed')
        result_path = batch / (job['job_id'] + '.json')
        require(result_path.exists() == (job['status'] != 'pending'), 'Attempt evidence/status mismatch')
    snapshots = {}
    for item in plan['snapshots']:
        path = batch / item['file']
        require(file_sha(path) == item['file_sha256'], 'Frozen snapshot file changed')
        snapshots[item['snapshot_id']] = read(path)
        validate_snapshot(snapshots[item['snapshot_id']])
    def guarded():
        return all(Path(p).is_file() and file_sha(p) == sha for p, sha in plan['readonly_guard'].items())
    require(guarded(), 'Original evidence changed since preparation')
    for job in manifest['jobs']:
        brief = input_variant(snapshots[job['snapshot_id']], job['representation'], job['history_mode'])
        require(fingerprint(brief) == job['input_sha256'], 'Frozen input mismatch')
    s = next(iter(snapshots.values()))['decision']['settings']
    require(file_sha(ROOT / s['model_path']) == s['model_sha256'], 'Planning model bytes changed')
    backend = ReplayBackend(s, ROOT)
    started = time.monotonic()
    session = {'started_utc': utc_now(), 'max_seconds': plan['selection']['max_seconds'],
               'runtime_provenance': actual, 'service_setup_seconds': None}
    manifest['sessions'].append(session)
    manifest['status'] = 'running'
    dump(batch / 'manifest.json', manifest)
    try:
        setup = time.monotonic()
        try:
            backend.start(plan['service_config'], batch / f'service-{len(manifest["sessions"])}')
        finally:
            session['service_setup_seconds'] = time.monotonic() - setup
        props = backend._json('/props')
        require(props['default_generation_settings']['n_ctx'] == s['max_model_len'], 'Wrong service context')
        def preflight(payload):
            prompt = backend._json('/apply-template', payload, timeout=30)['prompt']
            tokens = backend._json('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True}, timeout=30)['tokens']
            return {'method': 'apply-template/tokenize', 'prompt_tokens': len(tokens),
                    'prompt_sha256': fingerprint(prompt), 'context_limit': s['max_model_len']}
        for job in manifest['jobs']:
            if job['status'] != 'pending':
                continue
            if time.monotonic() - started >= session['max_seconds']:
                session['stop_reason'] = 'time_budget_reached'
                break
            job['status'] = 'running'
            dump(batch / 'manifest.json', manifest)
            result = replay_proposal(snapshots[job['snapshot_id']], job['representation'], job['history_mode'],
                                     ROOT, lambda payload: backend._json('/v1/chat/completions', payload, timeout=90), preflight)
            dump(batch / (job['job_id'] + '.json'), result)
            job.update(status=result['status'], result_file=job['job_id'] + '.json')
            dump(batch / 'manifest.json', manifest)
            print(job['job_id'], job['snapshot_id'], job['representation'], job['history_mode'],
                  result['status'], (result['proposal'] or {}).get('candidate_id'), flush=True)
            if result['status'] == 'interrupted' or any(c['failure_category'] == 'prompt_token_count_mismatch'
                                                       for c in result['proposal_calls']):
                session['stop_reason'] = 'interruption_or_prompt_guard'
                break
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        session.update(error_type=type(error).__name__, error_message=str(error))
    finally:
        backend.stop()
        session.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                       original_files_unchanged=guarded())
        manifest['status'] = 'completed' if all(j['status'] in ('valid', 'failed') for j in manifest['jobs']) else 'partial'
        if not session['original_files_unchanged']:
            manifest['status'] = 'source_guard_failed'
        dump(batch / 'manifest.json', manifest)
        dump(batch / 'summary.json', summarize_batch(batch, manifest))
        # A blank review sheet is not a model-generated correctness label.
        review = batch / 'review_template.csv'
        if not review.exists():
            with review.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=['job_id', 'snapshot_id', 'representation', 'history_mode',
                    'field', 'text', 'label', 'source_config_id', 'review_note'])
                writer.writeheader()
                for job in manifest['jobs']:
                    path = batch / (job['job_id'] + '.json')
                    if path.exists():
                        proposal = read(path)['proposal']
                        if proposal:
                            for field in ('hypothesis', 'expected_effect'):
                                writer.writerow({k: job[k] for k in ('job_id', 'snapshot_id', 'representation', 'history_mode')} |
                                                {'field': field, 'text': proposal[field]})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path)
    action.add_argument('--execute', type=Path)
    action.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        output = args.output or ROOT / 'runs' / ('proposal-replay-' + stamp)
        print(prepare(args.prepare, output))
    elif args.execute:
        result = execute(args.execute)
        print('Batch status:', result['status'])
    else:
        print(json.dumps(summarize_batch(args.summary, read(args.summary / 'manifest.json')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
