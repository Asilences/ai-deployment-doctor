import json
import tempfile
import unittest
from pathlib import Path

from inference_lab.audit import audit_holdout, evaluator_hashes, load_audit_candidate
from inference_lab.final_eval import evaluate_final, load_final_candidate
from inference_lab.backend import MockBackend
from inference_lab.core import fingerprint


ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.train = json.loads((ROOT / 'configs/local.json').read_text())
        self.holdout = json.loads(json.dumps(self.train))
        self.holdout['workload']['requests'] = 48
        self.candidate = {'max_num_seqs': 8, 'max_num_batched_tokens': 1024}
        hashes = evaluator_hashes('mock')
        state = {'identity': fingerprint({'settings': self.train, 'backend': 'mock',
                                          'evaluator_sources': hashes}),
                 'config': self.candidate, 'generation': 1,
                 'history': [{'proposal': {'config': self.candidate},
                              'verdict': {'accepted': True}}]}
        self.source = self.base / 'source'
        self.source.mkdir()
        self.best = self.source / 'best.json'
        self.best.write_text(json.dumps(state))
        self.run = self.source / 'run.json'
        self.run.write_text(json.dumps({'status': 'completed', 'backend': 'mock', 'planner': 'scripted',
                                        'trials': [{'verdict': {'accepted': True}}],
                                        'state': state, 'settings': self.train,
                                        'evaluator_sources': hashes}))

    def test_holdout_alternates_and_keeps_saved_candidate_untouched(self):
        original = self.best.read_bytes()
        backend = MockBackend(self.holdout, ROOT)
        directory = self.base / 'audit'
        result = audit_holdout(self.holdout, backend, self.best, directory)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['functional_match'])
        self.assertTrue(result['holdout_verdict']['accepted'])
        self.assertEqual(len(result['samples']['baseline']), 3)
        self.assertEqual(len(result['samples']['candidate']), 3)
        self.assertTrue((directory / '1-candidate' / 'benchmark.json').is_file())
        self.assertEqual(self.best.read_bytes(), original)
        self.assertEqual(json.loads(self.run.read_text())['state']['generation'], 1)

    def test_rejects_in_sample_and_changed_model(self):
        with self.assertRaisesRegex(ValueError, 'must differ'):
            load_audit_candidate(self.best, self.train, 'mock')
        changed = json.loads(json.dumps(self.holdout))
        changed['model'] = 'another-model'
        with self.assertRaisesRegex(ValueError, 'only change workload'):
            load_audit_candidate(self.best, changed, 'mock')

    def test_fresh_final_score_uses_original_workload_and_does_not_edit_source(self):
        original = self.best.read_bytes()
        result = evaluate_final(self.train, MockBackend(self.train, ROOT), self.best, self.base / 'final')
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['samples']['baseline']), 3)
        self.assertEqual(len(result['samples']['selected']), 3)
        self.assertGreater(result['throughput_gain'], 0)
        self.assertEqual(self.best.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'original workload'):
            load_final_candidate(self.best, self.holdout, 'mock')

    def test_final_score_handles_unchanged_baseline(self):
        state = json.loads(self.best.read_text())
        state.update(config=self.train['baseline'], generation=0, history=[])
        run = json.loads(self.run.read_text())
        run['state'] = state
        self.best.write_text(json.dumps(state))
        self.run.write_text(json.dumps(run))
        result = evaluate_final(self.train, MockBackend(self.train, ROOT), self.best, self.base / 'baseline-final')
        self.assertTrue(result['valid'])
        self.assertTrue(result['selected_is_baseline'])
        self.assertEqual(result['throughput_gain'], 0)


if __name__ == '__main__':
    unittest.main()
