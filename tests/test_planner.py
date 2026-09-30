import json
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from inference_lab.planner import Planner

ROOT = Path(__file__).resolve().parents[1]


class PlannerTests(unittest.TestCase):
    def test_fixed_heuristic_ignores_metrics_and_returns_preset_config(self):
        settings = json.loads((ROOT / 'configs/windows-1.5b.json').read_text())
        proposal = Planner('fixed', settings, ROOT).propose(
            {'config': settings['baseline'], 'metrics': {'output_throughput': 1}}, [])
        self.assertEqual(proposal['config'], {'parallel': 4, 'ubatch_size': 512})

    def test_random_search_seed_is_reproducible_and_recorded(self):
        settings = json.loads((ROOT / 'configs/windows.json').read_text())
        state = {'config': settings['baseline']}
        first = Planner('random', settings, ROOT, seed=732).propose(state, [])
        second = Planner('random', settings, ROOT, seed=732).propose(state, [])
        self.assertEqual(first, second)
        self.assertEqual(first['search_seed'], 732)
        with self.assertRaises(ValueError):
            Planner('random', settings, ROOT, seed=-1)

    def make(self, content):
        p = object.__new__(Planner)
        p.kind = 'openrouter'
        p.s = json.loads((ROOT / 'configs/local.json').read_text())
        p.credentials = {'OPENROUTER_API_KEY': 'test-secret-never-real', 'OPENROUTER_MODEL': 'test/model'}
        p.api = SimpleNamespace(request=Mock(return_value={
            'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': 100}}))
        return p

    def test_valid_response_is_parsed_and_secret_not_retained(self):
        text = json.dumps({'hypothesis': 'test-secret-never-real', 'expected_effect': 'Higher throughput',
                           'config': {'max_num_seqs': 8, 'max_num_batched_tokens': 1024}})
        p = self.make(text)
        result = p.propose({'config': p.s['baseline']}, [])
        self.assertNotIn('test-secret-never-real', json.dumps(result))
        self.assertEqual(result['usage']['total_tokens'], 100)
        payload = p.api.request.call_args.args[2]
        self.assertNotIn('test-secret-never-real', json.dumps(payload))
        self.assertEqual(payload['max_tokens'], 1200)

    def test_out_of_scope_and_non_json_responses_fail_closed(self):
        for text in ['not JSON', '{}', '{"config":{"model":"other"}}',
                     json.dumps({'hypothesis': 'x', 'expected_effect': 'y',
                                 'config': {'max_num_seqs': 999, 'max_num_batched_tokens': 512}})]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.make(text).propose({}, [])

    def test_transport_failure_never_silently_uses_scripted_planner(self):
        p = self.make('{}')
        p.api.request.side_effect = RuntimeError('API returned HTTP 401')
        with self.assertRaises(RuntimeError):
            p.propose({}, [])

    def test_local_planner_uses_running_model_and_measured_evidence(self):
        settings = json.loads((ROOT / 'configs/windows.json').read_text())
        p = Planner('local', settings, ROOT)
        proposal = {'hypothesis': 'more parallel slots may improve throughput',
                    'expected_effect': 'higher output tokens per second',
                    'config': {'parallel': 4, 'ubatch_size': 256}}
        response = {'choices': [{'message': {'content': json.dumps(proposal)}}]}
        seen = []

        def fake_open(request, timeout):
            seen.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())
        p.local_http = type('Client', (), {'open': staticmethod(fake_open)})()
        state = {'config': settings['baseline'], 'metrics': {'output_throughput': 301}}
        self.assertEqual(p.propose(state, [])['config'], proposal['config'])
        self.assertEqual(seen[0]['messages'][1]['content'].count('301'), 1)
        self.assertEqual(seen[0]['response_format']['type'], 'json_schema')

    def test_local_planner_retries_one_no_op(self):
        settings = json.loads((ROOT / 'configs/windows.json').read_text())
        p = Planner('local', settings, ROOT)
        current = {'hypothesis': 'no change', 'expected_effect': 'none',
                   'config': settings['baseline']}
        changed = {'hypothesis': 'test more slots', 'expected_effect': 'higher throughput',
                   'config': {'parallel': 4, 'ubatch_size': 128}}
        replies = [current, changed]
        calls = []

        def fake_open(request, timeout):
            calls.append(json.loads(request.data))
            answer = {'choices': [{'message': {'content': json.dumps(replies.pop(0))}}]}
            return io.BytesIO(json.dumps(answer).encode())
        p.local_http = type('Client', (), {'open': staticmethod(fake_open)})()
        result = p.propose({'config': settings['baseline'], 'metrics': {'output_throughput': 300}}, [])
        self.assertEqual(result['config'], changed['config'])
        self.assertEqual(len(calls), 2)

    def test_local_planner_retries_malformed_response_once(self):
        settings = json.loads((ROOT / 'configs/windows.json').read_text())
        p = Planner('local', settings, ROOT)
        changed = {'hypothesis': 'test larger batch', 'expected_effect': 'higher throughput',
                   'config': {'parallel': 4, 'ubatch_size': 256}}
        replies = ['{"hypothesis":', json.dumps(changed)]
        calls = []

        def fake_open(request, timeout):
            calls.append(json.loads(request.data))
            answer = {'choices': [{'message': {'content': replies.pop(0)}}]}
            return io.BytesIO(json.dumps(answer).encode())
        p.local_http = type('Client', (), {'open': staticmethod(fake_open)})()
        result = p.propose({'config': settings['baseline'], 'metrics': {'output_throughput': 300}}, [])
        self.assertEqual(result['config'], changed['config'])
        self.assertEqual(len(calls), 2)
        self.assertIn('invalid', calls[1]['messages'][-1]['content'])

    def test_local_planner_uses_failure_history_to_avoid_retesting(self):
        settings = json.loads((ROOT / 'configs/windows.json').read_text())
        p = Planner('local', settings, ROOT)
        rejected = {'hypothesis': 'old', 'expected_effect': 'unknown',
                    'config': {'parallel': 4, 'ubatch_size': 256}}
        fresh = {'hypothesis': 'try fewer slots', 'expected_effect': 'lower latency',
                 'config': {'parallel': 1, 'ubatch_size': 128}}
        replies = [rejected, fresh]
        calls = []

        def fake_open(request, timeout):
            calls.append(json.loads(request.data))
            answer = {'choices': [{'message': {'content': json.dumps(replies.pop(0))}}]}
            return io.BytesIO(json.dumps(answer).encode())
        p.local_http = type('Client', (), {'open': staticmethod(fake_open)})()
        history = [{'proposal': rejected, 'verdict': {'accepted': False}}]
        result = p.propose({'config': settings['baseline'], 'metrics': {}}, history)
        self.assertEqual(result['config'], fresh['config'])
        self.assertEqual(len(calls), 2)
        self.assertIn('Forbidden configurations', calls[1]['messages'][-1]['content'])

    def test_local_planner_refuses_repeated_candidate_after_bounded_retry(self):
        settings = json.loads((ROOT / 'configs/windows.json').read_text())
        p = Planner('local', settings, ROOT)
        old = {'hypothesis': 'old', 'expected_effect': 'unknown',
               'config': {'parallel': 4, 'ubatch_size': 256}}
        calls = []

        def fake_open(request, timeout):
            calls.append(1)
            answer = {'choices': [{'message': {'content': json.dumps(old)}}]}
            return io.BytesIO(json.dumps(answer).encode())
        p.local_http = type('Client', (), {'open': staticmethod(fake_open)})()
        with self.assertRaisesRegex(ValueError, 'previously tested'):
            p.propose({'config': settings['baseline'], 'metrics': {}},
                      [{'proposal': old, 'verdict': {'accepted': False}}])
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
