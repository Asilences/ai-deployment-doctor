import copy
import json
import tempfile
import unittest
from pathlib import Path

from inference_lab.backend import MockBackend, VllmBackend
from inference_lab.core import check_samples, validate_candidate, verdict
from inference_lab.planner import Planner
from inference_lab.runner import execute

ROOT = Path(__file__).resolve().parents[1]


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT / 'configs/local.json').read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def sample(self, config=None):
        b = MockBackend(self.s, ROOT)
        b.start(config or self.s['baseline'], self.root)
        return b.measure(self.root)

    def test_gate_rejects_cheating_or_invalid_measurements(self):
        good = self.sample()
        edits = [{'completed': 31}, {'output_lens': [63] * 32}, {'input_lens': [127] * 32},
                 {'total_output_tokens': 100}, {'errors': ['timeout']}, {'duration': 0},
                 {'output_throughput': float('nan')}, {'output_throughput': float('inf')},
                 {'output_throughput': good['output_throughput'] * 2}, {'p95_ttft_ms': 5000}]
        self.assertIsNone(check_samples([good] * 3, self.s))
        for edit in edits:
            with self.subTest(edit=edit):
                self.assertIsNotNone(check_samples([{**good, **edit}] * 3, self.s))

    def test_agent_cannot_edit_workload_or_model(self):
        for config in [{**self.s['baseline'], 'model': 'other'}, {'max_num_seqs': True, 'max_num_batched_tokens': 512},
                       {'max_num_seqs': '8; reboot', 'max_num_batched_tokens': 512}]:
            with self.assertRaises(ValueError):
                validate_candidate(config, self.s)

    def test_functional_regression_rejected_despite_speedup(self):
        old = self.sample()
        new = self.sample({'max_num_seqs': 8, 'max_num_batched_tokens': 1024})
        self.assertFalse(verdict([new] * 3, [old] * 3, ['wrong'], ['right'], self.s)['accepted'])

    def test_noisy_gain_rejected(self):
        old = self.sample()
        new = self.sample({'max_num_seqs': 8, 'max_num_batched_tokens': 1024})
        self.assertFalse(verdict([new, new, old], [old] * 3, ['ok'], ['ok'], self.s)['accepted'])

    def test_success_regression_rollback_and_restart_inheritance(self):
        b = MockBackend(self.s, ROOT)
        result = execute(self.s, b, Planner('scripted', self.s, ROOT), self.root / 'first', 3)
        self.assertEqual([t['verdict']['accepted'] for t in result['trials']], [True, False, False])
        self.assertEqual(result['state']['config']['max_num_seqs'], 8)
        self.assertEqual(result['state']['generation'], 1)
        self.assertIsNone(b.active)
        retained = json.loads((self.root / 'first/best.json').read_text())
        again = execute(self.s, b, Planner('scripted', self.s, ROOT), self.root / 'second', 0, retained)
        self.assertEqual(again['state']['config'], retained['config'])

    def test_cross_workload_inheritance_rejected(self):
        result = execute(self.s, MockBackend(self.s, ROOT), Planner('scripted', self.s, ROOT), self.root / 'first', 0)
        changed = copy.deepcopy(self.s)
        changed['workload']['requests'] = 64
        with self.assertRaises(ValueError):
            execute(changed, MockBackend(changed, ROOT), Planner('scripted', changed, ROOT),
                    self.root / 'second', 0, result['state'])

    def test_candidate_startup_failure_recovers_incumbent(self):
        class Broken(MockBackend):
            def start(self, config, directory):
                if config['max_num_seqs'] == 8:
                    raise RuntimeError('simulated server crash')
                super().start(config, directory)
        result = execute(self.s, Broken(self.s, ROOT), Planner('scripted', self.s, ROOT), self.root / 'run', 1)
        self.assertEqual(result['state']['config'], self.s['baseline'])
        self.assertEqual(result['trials'][0]['rollback'], 'incumbent_restart_verified')

    def test_failed_promotion_restarts_old_service(self):
        class Broken(MockBackend):
            def start(self, config, directory):
                if 'restore-check' in str(directory):
                    raise RuntimeError('simulated failed promotion')
                super().start(config, directory)
        result = execute(self.s, Broken(self.s, ROOT), Planner('scripted', self.s, ROOT), self.root / 'run', 1)
        self.assertEqual(result['state']['config'], self.s['baseline'])
        self.assertEqual(result['trials'][0]['verdict']['reason'], 'promotion_restart_failed')

    def test_failed_rollback_halts_without_committing_candidate(self):
        class Broken(MockBackend):
            def start(self, config, directory):
                if 'restore-check' in str(directory) or 'fallback-check' in str(directory):
                    raise RuntimeError('simulated restart failure')
                super().start(config, directory)
        b = Broken(self.s, ROOT)
        with self.assertRaises(RuntimeError):
            execute(self.s, b, Planner('scripted', self.s, ROOT), self.root / 'run', 1)
        retained = json.loads((self.root / 'run/best.json').read_text())
        self.assertEqual(retained['config'], self.s['baseline'])
        self.assertEqual(json.loads((self.root / 'run/run.json').read_text())['status'], 'failed')
        self.assertIsNone(b.active)

    def test_real_commands_keep_model_workload_and_output_fixed(self):
        b = object.__new__(VllmBackend)
        b.s, b.model, b.base = self.s, Path('/model'), 'http://127.0.0.1:8765'
        command = b.bench_command(Path('/results'))
        self.assertIn('--ignore-eos', command)
        self.assertIn('--save-detailed', command)
        self.assertEqual(command[command.index('--random-range-ratio') + 1], '0')
        config = {'max_num_seqs': 8, 'max_num_batched_tokens': 1024}
        server = b.serve_command(config)
        self.assertEqual(server[server.index('--max-model-len') + 1], '2048')
        self.assertEqual(server[server.index('--host') + 1], '127.0.0.1')


if __name__ == '__main__':
    unittest.main()
