"""Frozen development schedule: sequential GPU use, independent final, resumable pending jobs."""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.context_ablation import MODES
from inference_lab.core import dump, fingerprint, validate_settings
from inference_lab.final_eval import evaluate_final
from inference_lab.llama_backend import LlamaCppBackend
from inference_lab.planner import Planner
from inference_lab.runner import execute
from inference_lab.summary import paired_differences, serialize, summarize_runs


def code_identity():
    paths = sorted((ROOT / 'inference_lab').glob('*.py')) + [Path(__file__)]
    return fingerprint({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def build_schedule(tasks, methods, random_seeds, order_seed, iterations):
    if len(set(tasks)) != len(tasks) or len(set(methods)) != len(methods) or len(set(random_seeds)) != len(random_seeds):
        raise ValueError('Duplicate tasks, methods or seeds are not allowed')
    if not methods or not tasks or iterations not in (1, 2):
        raise ValueError('Require tasks, methods and one or two proposal slots')
    if not set(methods) <= set(MODES) | {'random', 'fixed'}:
        raise ValueError('Unknown method')
    if type(order_seed) is not int or order_seed < 0 or any(type(s) is not int or s < 0 for s in random_seeds):
        raise ValueError('Seeds must be nonnegative integers')
    if 'random' in methods and not random_seeds:
        raise ValueError('Random search requires explicit seeds')
    rng = random.Random(order_seed)
    blocks = list(tasks)
    rng.shuffle(blocks)
    jobs = []
    for task in blocks:
        block = [{'task': task, 'method': method, 'seed': seed,
                  'iterations': 1 if method == 'fixed' else iterations,
                  'slot_ceiling': iterations, 'status': 'pending'}
                 for method in methods for seed in (random_seeds if method == 'random' else [731])]
        rng.shuffle(block)
        jobs.extend(block)
    for i, job in enumerate(jobs):
        job['job_id'] = f'J{i:02d}'
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=['serial', 'prefill'], default=['serial', 'prefill'])
    parser.add_argument('--methods', nargs='+', choices=list(MODES) + ['random', 'fixed'],
                        default=['full', 'no_context', 'no_feedback', 'random', 'fixed'])
    parser.add_argument('--random-seeds', type=int, nargs='+', default=[100, 101, 102])
    parser.add_argument('--order-seed', type=int, default=20260930)
    parser.add_argument('--iterations', type=int, choices=[1, 2], default=1)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Time budget must be finite and positive')
    files = {'serial': ROOT / 'configs/pilot-serial-windows-1.5b.json',
             'prefill': ROOT / 'configs/pilot-prefill-windows-1.5b.json'}
    identity = code_identity()
    if args.resume:
        batch = args.resume.resolve()
        manifest = json.loads((batch / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['code_identity'] != identity:
            raise ValueError('Cannot resume with changed source code')
        for name, value in manifest['config_hashes'].items():
            if hashlib.sha256(files[name].read_bytes()).hexdigest() != value:
                raise ValueError('Cannot resume with changed workload')
        for job in manifest['jobs']:
            if job['status'] == 'searching':
                # Never repeat a partially spent search budget invisibly.
                job['status'] = 'interrupted_search_not_retried'
    else:
        jobs = build_schedule(args.tasks, args.methods, args.random_seeds, args.order_seed, args.iterations)
        manifest = {'protocol': 'development-context-pilot-v1', 'code_identity': identity,
                    'config_hashes': {k: hashlib.sha256(files[k].read_bytes()).hexdigest() for k in args.tasks},
                    'order_seed': args.order_seed, 'jobs': jobs, 'sessions': [],
                    'hypothesis': 'Explicit context or feedback changes accepted configuration selection',
                    'controls': 'Same model, start, candidate space, bounded output interface and fixed verifier',
                    'falsification': 'Same choices/scores across ablations, or failure to outperform simple controls',
                    'claim_scope': 'Development diagnostic only; no significance or superiority claim'}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        batch = ROOT / 'runs' / ('context-pilot-' + stamp)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if not args.resume:
        batch.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    session = {'started_utc': datetime.now(timezone.utc).isoformat(), 'max_seconds': args.max_seconds}
    manifest['sessions'].append(session)
    print('Batch: ' + str(batch), flush=True)

    def save():
        dump(batch / 'manifest.json', manifest)

    save()
    try:
        for job in manifest['jobs']:
            if job['status'] not in ('pending', 'search_completed'):
                continue
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            s = json.loads(files[job['task']].read_text(encoding='utf-8'))
            validate_settings(s)
            search_dir, final_dir = batch / (job['job_id'] + '-search'), batch / (job['job_id'] + '-final')
            job.update(search=str(search_dir), final=str(final_dir))
            try:
                if job['status'] == 'pending':
                    job['status'] = 'searching'
                    save()
                    local = job['method'] in MODES
                    planner = Planner('local' if local else job['method'], s, ROOT, seed=job['seed'],
                                      version='v3.1' if local else 'v2',
                                      context_mode=job['method'] if local else 'full')
                    execute(s, LlamaCppBackend(s, ROOT), planner, search_dir, job['iterations'], max_seconds=remaining)
                    job['status'] = 'search_completed'
                    save()
                if time.monotonic() - started >= args.max_seconds:
                    break
                # Final evaluation has no history feedback to the planner.
                final = evaluate_final(s, LlamaCppBackend(s, ROOT), search_dir / 'best.json', final_dir)
                job.update(status='completed', final_valid=final['valid'], final_gain=final['throughput_gain'])
                print(job['job_id'] + ' ' + job['method'] + ': ' + str(job['final_gain']), flush=True)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                job.update(status='failed', error_type=type(error).__name__)
            finally:
                save()
    finally:
        session.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        manifest['status'] = 'completed' if not any(j['status'] in ('pending', 'searching', 'search_completed')
                                                   for j in manifest['jobs']) else 'partial'
        save()
        rows = summarize_runs([batch])
        slots = {str(batch / (j['job_id'] + '-search')): j['slot_ceiling'] for j in manifest['jobs']}
        for row in rows:
            # A ceiling is not equal cost; fixed rule may use fewer actual slots.
            row['proposal_slot_limit'] = slots[row['source_run']]
        dump(batch / 'summary.json', rows)
        (batch / 'summary.csv').write_text(serialize(rows, 'csv'), encoding='utf-8')
        dump(batch / 'paired.json', paired_differences(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
