import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .backend import MockBackend, VllmBackend
from .audit import audit_holdout
from .final_eval import evaluate_final
from .llama_backend import LlamaCppBackend
from .core import validate_settings
from .planner import Planner
from .runner import execute
from .summary import export_summary
from .context_ablation import MODES

ROOT = Path(__file__).resolve().parents[1]


def doctor():
    def capture(command):
        try:
            p = subprocess.run(command, capture_output=True, timeout=15)
            data = p.stdout or p.stderr
            encoding = 'utf-16-le' if b'\x00' in data else 'utf-8'
            return {'exit_code': p.returncode, 'output': data.decode(encoding, errors='replace').strip()[:1500]}
        except (OSError, subprocess.TimeoutExpired):
            return {'available': False}
    installed = {}
    for package in ('vllm', 'torch'):
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
    # Read only this project's credential configuration; print presence, never values.
    sys.path.insert(0, str(ROOT / 'tools'))
    from check_openrouter import settings
    values = settings()
    data = {'os': platform.platform(), 'python': sys.version.split()[0], 'packages': installed,
            'gpu': capture(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader']),
            'openrouter_key_configured': bool(values.get('OPENROUTER_API_KEY')),
            'openrouter_model_configured': bool(values.get('OPENROUTER_MODEL')),
            'project_disk_free_gib': round(shutil.disk_usage(ROOT).free / 2**30, 1)}
    if sys.platform == 'win32':
        native = json.loads((ROOT / 'configs/windows-1.5b.json').read_text(encoding='utf-8'))
        binary = ROOT / native['llama_cpp_binary']
        model = ROOT / native['model_path']
        data['native_windows_inference'] = {
            'configured_model': native['model'],
            'server_binary_present': binary.is_file(),
            'model_file_present': model.is_file(),
            'ready_for_baseline': binary.is_file() and model.is_file()
        }
        data['wsl'] = capture(['wsl', '--list', '--verbose'])
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Bounded L2 inference optimization lab')
    parser.add_argument('command', choices=['doctor', 'demo', 'baseline', 'run', 'audit', 'final', 'summary'])
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--planner', choices=['random', 'local', 'openrouter', 'fixed'], default=None)
    parser.add_argument('--planner-version', choices=['v2', 'v3', 'v3.1'], default='v2')
    parser.add_argument('--context-mode', choices=MODES, default='full')
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='Stop starting new proposal slots after this elapsed time; finish recovery safely')
    parser.add_argument('--input', type=Path, nargs='+', help='Summary directories or run.json files')
    parser.add_argument('--format', choices=['json', 'csv'], default='json')
    parser.add_argument('--output', type=Path, help='Create a new summary file; never overwrite')
    parser.add_argument('--seed', type=int, default=731, help='Random-search seed, saved with each random proposal')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--inherit', type=Path, help='A previous best.json with identical experiment identity')
    parser.add_argument('--candidate', type=Path, help='A verified best.json to evaluate on a holdout workload')
    args = parser.parse_args()
    if args.command == 'doctor':
        doctor()
        return 0
    try:
        if args.command == 'summary':
            text = export_summary(args.input or [ROOT / 'runs'], args.format, args.output)
            print('Summary: ' + str(args.output) if args.output else text)
            return 0
        if not 0 <= args.iterations <= 20:
            raise ValueError('Iterations must be between 0 and 20')
        config_path = args.config or ROOT / 'configs' / ('windows-1.5b.json' if sys.platform == 'win32' else 'local.json')
        s = json.loads(config_path.read_text(encoding='utf-8-sig'))
        validate_settings(s)
        if args.command == 'audit':
            if args.config is None or args.candidate is None:
                raise ValueError('Audit requires --config for the holdout and --candidate for a saved best.json')
            backend = LlamaCppBackend(s, ROOT) if s.get('backend') == 'llama_cpp' else VllmBackend(s, ROOT)
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
            directory = ROOT / 'runs' / (backend.name + '-audit-' + stamp)
            result = audit_holdout(s, backend, args.candidate, directory)
            print('Audit: ' + str(directory / 'audit.json'))
            print('Holdout throughput gain: ' + f"{result['throughput_gain']:+.1%}")
            print('Holdout verdict: ' + result['holdout_verdict']['reason'])
            return 0
        if args.command == 'final':
            if args.config is None or args.candidate is None:
                raise ValueError('Final evaluation requires --config and --candidate best.json')
            backend = LlamaCppBackend(s, ROOT) if s.get('backend') == 'llama_cpp' else VllmBackend(s, ROOT)
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
            directory = ROOT / 'runs' / (backend.name + '-final-' + stamp)
            result = evaluate_final(s, backend, args.candidate, directory)
            print('Final evaluation: ' + str(directory / 'final.json'))
            print('Fresh throughput gain: ' + f"{result['throughput_gain']:+.1%}")
            print('Valid: ' + str(result['valid']))
            return 0
        mock = args.command == 'demo'
        kind = 'scripted' if mock else ('random' if args.command == 'baseline' else
               (args.planner or ('local' if s.get('backend') == 'llama_cpp' else 'openrouter')))
        planner = Planner(kind, s, ROOT, seed=args.seed, version=args.planner_version,
                          context_mode=args.context_mode)
        backend = MockBackend(s, ROOT) if mock else (LlamaCppBackend(s, ROOT)
                  if s.get('backend') == 'llama_cpp' else VllmBackend(s, ROOT))
        initial = json.loads(args.inherit.read_text(encoding='utf-8')) if args.inherit else None
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        directory = ROOT / 'runs' / (backend.name + '-' + kind + '-' + stamp)
        execute(s, backend, planner, directory, 0 if args.command == 'baseline' else args.iterations,
                initial, max_seconds=args.max_seconds)
        print('Report: ' + str(directory / 'report.html'))
        print('Retained state: ' + str(directory / 'best.json'))
        if mock:
            print('MOCK ONLY: no model inference, no measured speedup, no L2 claim.')
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        # Do not dump remote response bodies, environment, credentials or tracebacks.
        print('Experiment stopped: ' + type(error).__name__ + '. Check prerequisites and local run artifacts.', file=sys.stderr)
        if isinstance(error, (ValueError, RuntimeError)) and not isinstance(error, json.JSONDecodeError):
            print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
