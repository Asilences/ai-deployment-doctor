import importlib.util
import io
import json
import unittest
from pathlib import Path

from inference_lab.context_ablation import filter_context
from inference_lab.planner import Planner
from inference_lab.summary import paired_differences

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('context_pilot_test', ROOT / 'scripts/run_context_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ContextAblationTests(unittest.TestCase):
    def test_filters_remove_only_declared_evidence_without_mutating_input(self):
        brief = {'environment': {'model': 'ENV_MARKER'}, 'hardware': {'gpu': 'HW_MARKER'},
                 'workload': {'input': 12345}, 'current_metrics': {'rate': 67890},
                 'prior_trials': [{'reason': 'FEEDBACK_MARKER'}], 'available_candidates': ['C00'],
                 'current_config': {'parallel': 2}, 'objective': 'fixed', 'latency_limits_ms': [1, 2]}
        expected = {'no_environment': {'environment', 'hardware'}, 'no_workload': {'workload'},
                    'no_context': {'environment', 'hardware', 'workload'},
                    'no_feedback': {'current_metrics', 'prior_trials'}, 'full': set()}
        for mode, removed in expected.items():
            with self.subTest(mode=mode):
                result = filter_context(brief, mode)
                self.assertEqual(set(result), set(brief) - removed)
                self.assertEqual(result['available_candidates'], ['C00'])
                self.assertEqual(result['latency_limits_ms'], [1, 2])
        self.assertEqual(brief['environment']['model'], 'ENV_MARKER')

    def test_actual_request_and_recorded_evidence_both_obey_no_feedback(self):
        s = json.loads((ROOT / 'configs/windows-1.5b.json').read_text())
        planner = Planner('local', s, ROOT, version='v3.1', context_mode='no_feedback')
        requests = []
        response = {'choices': [{'message': {'content': json.dumps({
            'candidate_id': 'C11', 'hypothesis': 'Try batching', 'expected_effect': 'Unknown'})}}]}
        def open_fake(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())
        planner.local_http = type('Client', (), {'open': staticmethod(open_fake)})()
        planner.propose({'config': s['baseline'], 'metrics': {'rate': 'METRIC_MARKER'}},
                        [{'verdict': {'reason': 'HISTORY_MARKER'}, 'candidate_executed': False}])
        self.assertNotIn('METRIC_MARKER', json.dumps(requests))
        self.assertNotIn('HISTORY_MARKER', json.dumps(requests))
        evidence = planner.last_trace[0]['request']['evidence']
        self.assertNotIn('prior_trials', evidence)
        self.assertNotIn('current_metrics', evidence)

    def test_invalid_ablation_version_combination_fails_explicitly(self):
        s = json.loads((ROOT / 'configs/windows-1.5b.json').read_text())
        with self.assertRaises(ValueError):
            Planner('local', s, ROOT, version='v3', context_mode='no_context')
        with self.assertRaises(ValueError):
            Planner('random', s, ROOT, context_mode='no_feedback')

    def test_schedule_seed_reproducibility_unique_jobs_and_no_free_fixed_slots(self):
        first = pilot.build_schedule(['serial', 'prefill'], ['full', 'random', 'fixed'], [100, 101, 102], 17, 2)
        second = pilot.build_schedule(['serial', 'prefill'], ['full', 'random', 'fixed'], [100, 101, 102], 17, 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len({j['job_id'] for j in first}), 10)
        for job in first:
            self.assertEqual(job['slot_ceiling'], 2)
            self.assertEqual(job['iterations'], 1 if job['method'] == 'fixed' else 2)
        with self.assertRaises(ValueError):
            pilot.build_schedule(['prefill'], ['random'], [100, 100], 17, 1)

    def test_pairs_use_tasks_not_measurement_repeats_or_best_seed(self):
        def row(method, gain, seed):
            return {'task_id': 'one-task', 'method_label': method, 'source_run': method + str(seed),
                    'final_verified_improvement': gain, 'proposal_slot_limit': 1, 'seed': seed}
        rows = [row('local:full', .1, 731), row('random', .05, 100), row('random', .2, 101)]
        pairs = paired_differences(rows)
        self.assertEqual(len(pairs), 2)
        self.assertAlmostEqual(pairs[0]['paired_gain_difference'], .05)
        self.assertAlmostEqual(pairs[1]['paired_gain_difference'], -.1)
        self.assertEqual({p['task_id'] for p in pairs}, {'one-task'})
        rows[-1]['proposal_slot_limit'] = 2
        self.assertIsNone(paired_differences(rows)[-1]['paired_gain_difference'])
        rows.append(row('local:full', .3, 731))
        self.assertEqual(paired_differences(rows), [])
