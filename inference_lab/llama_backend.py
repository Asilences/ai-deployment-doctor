"""Native Windows Vulkan inference, measured through llama.cpp's HTTP API."""
import concurrent.futures
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .backend import PROMPTS, child_env
from .core import dump


def p95(values):
    if not values:
        raise ValueError('No completed requests')
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


class LlamaCppBackend:
    name = 'llama_cpp'

    def __init__(self, settings, root):
        if sys.platform != 'win32':
            raise ValueError('The native llama.cpp adapter requires Windows')
        self.s, self.root = settings, root
        self.exe = (root / settings['llama_cpp_binary']).resolve()
        self.model = (root / settings['model_path']).resolve()
        if not self.exe.is_file() or not self.model.is_file():
            raise ValueError('Run python scripts/setup_windows.py before a real experiment')
        build_manifest = json.loads((self.exe.parent / 'download_manifest.json').read_text(encoding='utf-8'))
        model_manifest = json.loads(self.model.with_suffix(self.model.suffix + '.manifest.json').read_text(encoding='utf-8'))
        if build_manifest['build_tag'] != settings['llama_cpp_version']:
            raise ValueError('Pinned binary revision mismatch')
        if model_manifest['model_revision'] != settings['model_revision'] or model_manifest['model'] != settings['model']:
            raise ValueError('Pinned model revision mismatch')
        if model_manifest['model_sha256'] != settings['model_sha256']:
            raise ValueError('Model checksum manifest mismatch')
        self.base = 'http://127.0.0.1:' + str(settings['port'])
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.process = self.log = None
        self.prompts = None

    def _json(self, path, body=None, timeout=60):
        data = None if body is None else json.dumps(body).encode('utf-8')
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with self.http.open(req, timeout=timeout) as response:
            return json.load(response)

    def serve_command(self, candidate):
        return [str(self.exe), '--model', str(self.model), '--host', '127.0.0.1',
                '--port', str(self.s['port']), '--ctx-size', str(self.s['max_model_len']),
                '--gpu-layers', str(self.s['gpu_layers']), '--threads', str(self.s['cpu_threads']),
                '--parallel', str(candidate['parallel']), '--batch-size', str(self.s['batch_size']),
                '--ubatch-size', str(candidate['ubatch_size']), '--cont-batching',
                '--no-cache-prompt']

    @staticmethod
    def gpu_free_mib():
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=15, check=True)
        return int(result.stdout.strip().splitlines()[0])

    def start(self, candidate, directory):
        self.stop()
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', self.s['port']))
            except OSError:
                raise RuntimeError('Experiment port occupied; no existing service was modified') from None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        before = self.gpu_free_mib()
        cmd = self.serve_command(candidate)
        dump(directory / 'server_command.json', cmd)
        self.log = (directory / 'server.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen(cmd, cwd=self.exe.parent, stdout=self.log,
                                        stderr=subprocess.STDOUT, env=child_env(),
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + self.s['startup_timeout_s']
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('llama-server exited; inspect server.log')
                try:
                    if self._json('/health', timeout=2).get('status') == 'ok':
                        after = self.gpu_free_mib()
                        gpu_delta = before - after
                        dump(directory / 'gpu_memory.json', {'free_mib_before': before,
                                                              'free_mib_after': after,
                                                              'model_server_delta_mib': gpu_delta})
                        if gpu_delta < 100:
                            raise RuntimeError('GPU memory did not rise enough to verify model offload')
                        props = self._json('/props')
                        if props.get('total_slots') != candidate['parallel']:
                            raise RuntimeError('Server did not apply the requested parallel-slot setting')
                        self.prompts = None
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            raise RuntimeError('llama-server startup timed out; inspect server.log')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def outputs(self):
        values = []
        for prompt in PROMPTS:
            r = self._json('/completion', {'prompt': prompt, 'n_predict': 16,
                                           'temperature': 0, 'seed': 42,
                                           'cache_prompt': False}, timeout=60)
            value = r.get('content')
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError('Functional probe returned no text')
            values.append(value)
        return values

    def _prepare_prompts(self):
        count = self.s['workload']['requests']
        length = self.s['workload']['input_tokens']
        prompts = []
        for i in range(count):
            raw = f'Request identifier {i}. Explain inference batching and caching. ' * (length + 8)
            tokenized = self._json('/tokenize', {'content': raw, 'add_special': False})['tokens']
            if len(tokenized) < length:
                raise RuntimeError('Tokenizer returned fewer tokens than requested')
            prompts.append(tokenized[:length])
        self.prompts = prompts

    def _one_request(self, token_ids):
        start = time.perf_counter()
        data = json.dumps({'prompt': token_ids, 'n_predict': self.s['workload']['output_tokens'],
                           'temperature': 0, 'seed': self.s['workload']['seed'],
                           'ignore_eos': True, 'cache_prompt': False,
                           'stream': True, 'return_tokens': True}).encode('utf-8')
        req = urllib.request.Request(self.base + '/completion', data=data,
                                     headers={'Content-Type': 'application/json'})
        first = None
        token_count = 0
        final = None
        with self.http.open(req, timeout=self.s['benchmark_timeout_s']) as response:
            for raw in response:
                if not raw.startswith(b'data: '):
                    continue
                part = raw[6:].strip()
                if part == b'[DONE]':
                    break
                event = json.loads(part)
                if 'error' in event:
                    raise RuntimeError('Server returned an error event')
                tokens = event.get('tokens') or []
                if tokens:
                    if first is None:
                        first = time.perf_counter()
                    token_count += len(tokens)
                if event.get('stop') is True:
                    final = event
                    break
        end = time.perf_counter()
        if final is None or first is None:
            raise RuntimeError('Incomplete streaming response')
        if final.get('truncated') or final.get('stop_type') != 'limit':
            raise RuntimeError('Request truncated or stopped before fixed output length')
        output_count = final.get('tokens_predicted', token_count)
        input_count = final.get('tokens_evaluated')
        return {'input_tokens': input_count, 'output_tokens': output_count,
                'ttft_ms': (first - start) * 1000, 'e2el_ms': (end - start) * 1000}

    def measure(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.prompts is None:
            self._prepare_prompts()
        samples, errors = [], []
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.s['workload']['concurrency']) as pool:
            futures = [pool.submit(self._one_request, ids) for ids in self.prompts]
            for future in futures:
                try:
                    samples.append(future.result(timeout=self.s['benchmark_timeout_s']))
                except (OSError, ValueError, RuntimeError, concurrent.futures.TimeoutError) as error:
                    errors.append(type(error).__name__)
        duration = time.perf_counter() - start
        output_total = sum(row['output_tokens'] for row in samples if type(row['output_tokens']) is int)
        row = {'duration': duration, 'completed': len(samples), 'failed': len(errors),
               'input_lens': [x['input_tokens'] for x in samples],
               'output_lens': [x['output_tokens'] for x in samples],
               'total_output_tokens': output_total, 'output_throughput': output_total / duration,
               'p95_ttft_ms': p95([x['ttft_ms'] for x in samples]) if samples else None,
               'p95_e2el_ms': p95([x['e2el_ms'] for x in samples]) if samples else None,
               'errors': errors}
        dump(directory / 'benchmark.json', row)
        return row

    def warmup(self, directory):
        # The first real request can compile Vulkan shaders. Apply the same
        # fixed warmup to every restarted configuration before measurement.
        self.measure(Path(directory) / 'warmup')
