import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from inference_lab.summary import export_summary, summarize_runs


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.run = {'planner': 'local', 'status': 'completed', 'settings': {'workload': 'fixture'},
                    'state': {'config': {'x': 1}},
                    'trials': [{'id': 1, 'proposal': {'config': {'x': 1}},
                                'verdict': {'accepted': True, 'gain': 9}}]}
        self.write(self.source / 'run.json', self.run)
        self.write(self.source / 'best.json', self.run['state'])

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def final(self):
        return {'status': 'completed', 'valid': True, 'throughput_gain': 0.07,
                'settings': self.run['settings'], 'selected_config': self.run['state']['config'],
                'provenance': {'source_run': str(self.source),
                               'source_best_sha256': hashlib.sha256((self.source / 'best.json').read_bytes()).hexdigest()}}

    def test_search_winner_never_becomes_final_score_and_legacy_costs_unknown(self):
        row = summarize_runs([self.root])[0]
        self.assertIsNone(row['final_verified_improvement'])
        self.assertIsNone(row['llm_calls'])
        self.assertIsNone(row['total_tokens'])

    def test_valid_final_is_joined_by_source_and_fingerprint(self):
        self.write(self.root / 'final/final.json', self.final())
        row = summarize_runs([self.root])[0]
        self.assertEqual(row['final_verified_improvement'], 0.07)
        self.assertEqual(row['budget_to_success_slots'], 1)

    def test_invalid_final_and_changed_source_never_score(self):
        final = self.final()
        final['valid'] = False
        self.write(self.root / 'final/final.json', final)
        self.assertIsNone(summarize_runs([self.root])[0]['final_verified_improvement'])
        final.update(valid=True)
        final['provenance']['source_best_sha256'] = 'wrong'
        self.write(self.root / 'final/final.json', final)
        self.assertEqual(summarize_runs([self.root])[0]['final_record_count'], 0)

    def test_multiple_final_scores_are_not_cherry_picked(self):
        self.write(self.root / 'a/final.json', self.final())
        self.write(self.root / 'b/final.json', {**self.final(), 'throughput_gain': 0.9})
        row = summarize_runs([self.root])[0]
        self.assertEqual(row['final_record_count'], 2)
        self.assertIsNone(row['final_verified_improvement'])

    def test_export_never_overwrites_input_and_preserves_source_bytes(self):
        source = self.source / 'run.json'
        before = source.read_bytes()
        with self.assertRaises(FileExistsError):
            export_summary([self.root], output=source)
        output = self.root / 'summary.csv'
        export_summary([self.root], format='csv', output=output)
        self.assertEqual(before, source.read_bytes())
        self.assertIn('unknown', output.read_text())
        with self.assertRaises(FileExistsError):
            export_summary([self.root], output=output)
