import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_lab.llama_backend import LlamaCppBackend, p95

ROOT = Path(__file__).resolve().parents[1]


class LlamaBackendTests(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT / 'configs/windows.json').read_text())

    def test_server_command_only_uses_allowlisted_settings(self):
        b = object.__new__(LlamaCppBackend)
        b.s, b.exe, b.model = self.s, Path('llama-server.exe'), Path('model.gguf')
        command = b.serve_command({'parallel': 4, 'ubatch_size': 256})
        self.assertEqual(command[command.index('--parallel') + 1], '4')
        self.assertEqual(command[command.index('--ubatch-size') + 1], '256')
        self.assertIn('--no-cache-prompt', command)
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')

    def test_percentile_uses_completed_request_distribution(self):
        self.assertEqual(p95(list(range(1, 21))), 19)
        self.assertEqual(p95([12]), 12)

    def test_stream_requires_full_length_and_server_token_counts(self):
        b = object.__new__(LlamaCppBackend)
        b.s, b.base = self.s, 'http://127.0.0.1:8765'
        final = {'stop': True, 'stop_type': 'limit', 'tokens_predicted': 32,
                 'tokens_evaluated': 128, 'truncated': False}
        lines = [b'data: ' + json.dumps({'stop': False, 'tokens': [i]}).encode() + b'\n'
                 for i in range(32)]
        lines.append(b'data: ' + json.dumps(final).encode() + b'\n')

        class FakeResponse:
            def __enter__(self):
                return iter(lines)
            def __exit__(self, *_):
                return False
        b.http = SimpleNamespace(open=lambda *_, **__: FakeResponse())
        row = b._one_request([1] * 128)
        self.assertEqual((row['input_tokens'], row['output_tokens']), (128, 32))


if __name__ == '__main__':
    unittest.main()
