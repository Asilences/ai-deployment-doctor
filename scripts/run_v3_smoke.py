"""Two existing development workloads; bounded wall time, no forced termination."""
import argparse
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_lab.core import dump, validate_settings
from inference_lab.final_eval import evaluate_final
from inference_lab.llama_backend import LlamaCppBackend
from inference_lab.planner import Planner
from inference_lab.runner import execute
from inference_lab.summary import export_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--planner-version', choices=['v3', 'v3.1'], default='v3')
    parser.add_argument('--iterations', type=int, choices=[1, 2], default=2)
    parser.add_argument('--tasks', nargs='+', choices=['serial', 'prefill'], default=['serial', 'prefill'])
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error('--max-seconds must be positive')
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    batch = ROOT / 'runs' / (args.planner_version.replace('.', '_') + '-smoke-' + stamp)
    batch.mkdir(parents=True, exist_ok=False)
    manifest = {'protocol': 'development-v3-smoke-v1' if args.planner_version == 'v3' else 'development-v31-bounded-output-v1',
                'planner_version': args.planner_version,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'hypothesis': ('ID selection is bounded and all proposal costs are retained, including failures'
                               if args.planner_version == 'v3' else
                               'A 96-character limit on both explanation fields permits a complete valid JSON proposal'),
                'independent_variable': 'candidate output schema and explanation constraints',
                'controls': 'Existing model, action space, workloads and fixed verifier',
                'budget': {'seconds': args.max_seconds, 'slots_per_task': args.iterations, 'calls_per_slot_max': 2},
                'falsification': 'No complete valid proposal under the bounded call budget',
                'claim_scope': 'Development validation only; not a superiority or ablation experiment',
                'status': 'running', 'tasks': []}
    dump(batch / 'manifest.json', manifest)
    print('Batch: ' + str(batch), flush=True)
    try:
        for label, config in [('serial', 'pilot-serial-windows-1.5b.json'),
                              ('prefill', 'pilot-prefill-windows-1.5b.json')]:
            if label not in args.tasks:
                continue
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                manifest['stop_reason'] = 'time_budget_reached'
                break
            source = ROOT / 'configs' / config
            s = json.loads(source.read_text(encoding='utf-8'))
            validate_settings(s)
            task = {'label': label, 'config': str(source),
                    'config_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'search': str(batch / (label + '-search')), 'final': None}
            manifest['tasks'].append(task)
            dump(batch / 'manifest.json', manifest)
            try:
                run = execute(s, LlamaCppBackend(s, ROOT), Planner('local', s, ROOT, version=args.planner_version),
                              task['search'], args.iterations, max_seconds=remaining)
                task['search_status'] = run['status']
                if time.monotonic() - started < args.max_seconds:
                    final_dir = batch / (label + '-final')
                    task['final'] = str(final_dir)
                    final = evaluate_final(s, LlamaCppBackend(s, ROOT),
                                           Path(task['search']) / 'best.json', final_dir)
                    task['final_valid'] = final['valid']
                    task['final_gain'] = final['throughput_gain']
                    print(label + ' final: ' + str(task['final_gain']), flush=True)
                else:
                    task['final_skipped'] = 'time_budget_reached'
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                task['error_type'] = type(error).__name__
                print(label + ' failed: ' + type(error).__name__, flush=True)
            finally:
                dump(batch / 'manifest.json', manifest)
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'interrupted'
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['elapsed_seconds'] = time.monotonic() - started
        dump(batch / 'manifest.json', manifest)
        export_summary([batch], output=batch / 'summary.json')
        export_summary([batch], format='csv', output=batch / 'summary.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
