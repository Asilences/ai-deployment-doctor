import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .core import dump

PROMPTS = [
    'Complete this sentence: The capital of France is',
    'Continue the sequence: one, two, three,',
    'A computer stores information in'
]


class MockBackend:
    """Synthetic fixture only: no inference, performance evidence, or AI decisions."""
    name = 'mock'

    def __init__(self, s, root):
        self.s = s
        self.active = None

    def start(self, config, directory):
        self.active = dict(config)

    def stop(self):
        self.active = None

    def outputs(self):
        return ['Paris', 'four, five', 'bits']

    def measure(self, directory):
        w = self.s['workload']
        n = self.active['max_num_seqs']
        rate = 100 + 15 * min(n, 8) + self.active['max_num_batched_tokens'] / 64
        row = {'duration': w['requests'] * w['output_tokens'] / rate,
               'completed': w['requests'], 'failed': 0,
               'total_output_tokens': w['requests'] * w['output_tokens'],
               'output_lens': [w['output_tokens']] * w['requests'],
               'input_lens': [w['input_tokens']] * w['requests'],
               'output_throughput': rate, 'p95_ttft_ms': 180 if n <= 16 else 5000,
               'p95_e2el_ms': 800, 'errors': []}
        dump(Path(directory) / 'benchmark.json', row)
        return row


def child_env():
    # The benchmark and model server never need the controller's remote API keys.
    return {k: v for k, v in os.environ.items()
            if not any(x in k.upper() for x in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}


def terminate(process):
    if process is None:
        return
    # Each Linux child is created in its own session, including worker descendants.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


class VllmBackend:
    name = 'vllm'

    def __init__(self, s, root):
        if sys.platform != 'linux':
            raise ValueError('The vLLM backend requires Linux/WSL2. See docs/SETUP.md')
        if importlib.metadata.version('vllm') != s['vllm_version']:
            raise ValueError('vLLM version differs from the pinned experiment version')
        self.s, self.root = s, root
        self.process = self.log = None
        self.model = (root / s['model_path']).resolve()
        manifest = json.loads((self.model / 'download_manifest.json').read_text())
        if manifest != {'model': s['model'], 'revision': s['model_revision']}:
            raise ValueError('Model manifest mismatch; run scripts/download_model.py')
        self.base = 'http://127.0.0.1:' + str(s['port'])
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _json(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, headers={'Content-Type': 'application/json'})
        with self.http.open(request, timeout=60) as response:
            return json.load(response)

    def serve_command(self, config):
        return [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(self.model),
                '--served-model-name', 'inference-lab', '--host', '127.0.0.1', '--port', str(self.s['port']),
                '--dtype', self.s['dtype'], '--max-model-len', str(self.s['max_model_len']),
                '--gpu-memory-utilization', str(self.s['gpu_memory_utilization']),
                '--max-num-seqs', str(config['max_num_seqs']),
                '--max-num-batched-tokens', str(config['max_num_batched_tokens']),
                '--enable-chunked-prefill', '--no-enable-prefix-caching', '--enforce-eager', '--seed', '42']

    def start(self, config, directory):
        self.stop()
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', self.s['port']))
            except OSError:
                raise RuntimeError('Experiment port is occupied; no existing service was modified') from None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        command = self.serve_command(config)
        dump(directory / 'server_command.json', command)
        self.log = (directory / 'server.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=child_env(), start_new_session=True)
        deadline = time.monotonic() + self.s['startup_timeout_s']
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Model server exited; see the local server.log')
                try:
                    result = self._json('/v1/models')
                    if any(m['id'] == 'inference-lab' for m in result.get('data', [])):
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            raise RuntimeError('Model startup timed out; see the local server.log')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        try:
            terminate(self.process)
        finally:
            self.process = None
            if self.log:
                self.log.close()
                self.log = None

    def outputs(self):
        values = []
        for prompt in PROMPTS:
            response = self._json('/v1/completions', {'model': 'inference-lab', 'prompt': prompt,
                                   'max_tokens': 16, 'temperature': 0, 'seed': 42})
            value = response['choices'][0]['text']
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError('Functional probe returned empty output')
            values.append(value)
        return values

    def bench_command(self, directory):
        w = self.s['workload']
        return [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'bench', 'serve',
                '--backend', 'vllm', '--base-url', self.base, '--endpoint', '/v1/completions',
                '--model', str(self.model), '--served-model-name', 'inference-lab',
                '--dataset-name', 'random', '--random-input-len', str(w['input_tokens']),
                '--random-output-len', str(w['output_tokens']), '--random-range-ratio', '0',
                '--num-prompts', str(w['requests']), '--max-concurrency', str(w['concurrency']),
                '--request-rate', 'inf', '--seed', str(w['seed']), '--temperature', '0', '--ignore-eos',
                '--num-warmups', '2', '--percentile-metrics', 'ttft,e2el', '--metric-percentiles', '95',
                '--save-result', '--save-detailed', '--result-dir', str(directory),
                '--result-filename', 'benchmark.json']

    def measure(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        command = self.bench_command(directory)
        dump(directory / 'benchmark_command.json', command)
        with (directory / 'benchmark.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=child_env(), start_new_session=True)
            try:
                code = process.wait(timeout=self.s['benchmark_timeout_s'])
                if code:
                    raise RuntimeError('Benchmark failed; see benchmark.log')
            finally:
                terminate(process)
        raw = json.loads((directory / 'benchmark.json').read_text())
        # Upstream names verified against v0.30.0; absent metrics fail closed.
        keys = ('duration', 'completed', 'failed', 'total_output_tokens', 'output_lens', 'input_lens',
                'output_throughput', 'p95_ttft_ms', 'p95_e2el_ms', 'errors')
        return {k: raw[k] for k in keys if k in raw}
