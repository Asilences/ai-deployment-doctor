import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from inference_lab.backend import MockBackend
from inference_lab.candidate_ids import SearchSpaceExhausted, available_candidates, candidate_catalog
from inference_lab.planner import Planner
from inference_lab.proposal_trace import trace_totals
from inference_lab.runner import execute

ROOT = Path(__file__).resolve().parents[1]


class LlamaFixture(MockBackend):
    """Synthetic control-flow fixture only; not a GPU measurement."""
    def measure(self, directory):
        previous = self.active
        self.active = {'max_num_seqs': previous['parallel'],
                       'max_num_batched_tokens': previous['ubatch_size']}
        try:
            return super().measure(directory)
        finally:
            self.active = previous


class CandidateIdTests(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT / 'configs/windows-1.5b.json').read_text())
        self.state = {'config': self.s['baseline'], 'metrics': {'output_throughput': 100}}

    def reply(self, candidate='C11', usage=10):
        response = {'choices': [{'message': {'content': json.dumps({
            'candidate_id': candidate, 'hypothesis': 'test batching', 'expected_effect': 'unknown'})}}]}
        if usage is not None:
            response['usage'] = {'total_tokens': usage}
        return response

    def planner(self, replies, version='v3'):
        p = Planner('local', self.s, ROOT, version=version)
        calls = []
        def open_fake(request, timeout):
            calls.append(json.loads(request.data))
            response = replies.pop(0)
            if isinstance(response, BaseException):
                raise response
            return io.BytesIO(json.dumps(response).encode())
        p.local_http = type('Client', (), {'open': staticmethod(open_fake)})()
        return p, calls

    def test_catalog_stable_across_mapping_order(self):
        changed = copy.deepcopy(self.s)
        changed['search_space'] = dict(reversed(list(changed['search_space'].items())))
        self.assertEqual(candidate_catalog(self.s), candidate_catalog(changed))
        self.assertEqual(len(candidate_catalog(self.s)), 16)
        self.assertEqual(candidate_catalog(self.s)[11],
                         {'candidate_id': 'C11', 'config': {'parallel': 4, 'ubatch_size': 512}})

    def test_available_excludes_current_and_all_executed_history(self):
        catalog = candidate_catalog(self.s)
        history = [{'proposal': {'config': c['config']}, 'candidate_executed': True} for c in catalog[:4]]
        history += [{'verdict': {'reason': 'proposal_failed'}, 'candidate_executed': False}]
        ids = [c['candidate_id'] for c in available_candidates(self.s, self.state, history)]
        self.assertNotIn('C05', ids)
        self.assertFalse(set(ids) & {'C00', 'C01', 'C02', 'C03'})

    def test_selected_id_maps_exactly_and_context_is_supplied(self):
        p, calls = self.planner([self.reply()])
        result = p.propose(self.state, [])
        self.assertEqual(result['config'], {'parallel': 4, 'ubatch_size': 512})
        payload = json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(payload['workload'], self.s['workload'])
        self.assertNotIn('C05', [c['candidate_id'] for c in payload['available_candidates']])
        self.assertEqual(p.last_trace[0]['parse_result'], 'valid')

    def test_repair_cost_includes_both_calls(self):
        p, calls = self.planner([self.reply('not-an-id', 7), self.reply('C11', 13)])
        result = p.propose(self.state, [])
        self.assertEqual(result['usage']['total_tokens'], 20)
        self.assertEqual(len(calls), 2)
        totals = trace_totals(p.last_trace)
        self.assertEqual(totals['invalid_calls'], 1)
        self.assertEqual(totals['retry_calls'], 1)
        for c in p.last_trace:
            self.assertGreaterEqual(c['elapsed_seconds'], 0)
            self.assertIn('ended_utc', c)

    def test_duplicate_retry_then_no_fallback(self):
        p, calls = self.planner([self.reply('C05'), self.reply('C05')])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.propose(self.state, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(trace_totals(p.last_trace)['duplicate_calls'], 2)

    def test_malformed_output_is_not_silently_corrected(self):
        malformed = {'choices': [{'message': {'content': 'not JSON'}}], 'usage': {'total_tokens': 4}}
        p, calls = self.planner([malformed, malformed])
        with self.assertRaises(ValueError):
            p.propose(self.state, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(trace_totals(p.last_trace)['total_tokens'], 8)

    def test_missing_usage_remains_unknown_with_known_subtotal(self):
        p, _ = self.planner([self.reply('wrong', 7), self.reply('C11', None)])
        result = p.propose(self.state, [])
        self.assertIsNone(result['usage']['total_tokens'])
        totals = trace_totals(p.last_trace)
        self.assertEqual(totals['known_total_tokens'], 7)
        self.assertEqual(totals['usage_missing_calls'], 1)
        self.assertIsNone(totals['cost'])

    def test_timeout_logged_and_never_replaced(self):
        p, calls = self.planner([TimeoutError('fixture timeout')])
        with self.assertRaises(TimeoutError):
            p.propose(self.state, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(p.last_trace[0]['failure_category'], 'timeout')
        self.assertIsNone(p.last_trace[0]['usage']['total_tokens'])

    def test_exhaustion_stops_before_model_call(self):
        p, calls = self.planner([])
        history = [{'proposal': {'config': c['config']}, 'candidate_executed': True}
                   for c in candidate_catalog(self.s)]
        with self.assertRaises(SearchSpaceExhausted):
            p.propose(self.state, history)
        self.assertEqual(calls, [])

    def test_v2_default_and_retry_usage_remain_compatible(self):
        self.assertEqual(Planner('local', self.s, ROOT).version, 'v2')
        def response(config, usage):
            return {'choices': [{'message': {'content': json.dumps({
                'config': config, 'hypothesis': 'test', 'expected_effect': 'unknown'})}}],
                    'usage': {'total_tokens': usage}}
        p, _ = self.planner([response(self.s['baseline'], 7),
                             response({'parallel': 4, 'ubatch_size': 512}, 13)], version='v2')
        self.assertEqual(p.propose(self.state, [])['usage']['total_tokens'], 20)
        with self.assertRaises(ValueError):
            Planner('random', self.s, ROOT, version='v3')

    def test_runner_retains_failed_slots_cost_and_full_history(self):
        p, _ = self.planner([self.reply('wrong', 3), self.reply('wrong', 5), self.reply('C11', 11)])
        with tempfile.TemporaryDirectory() as temp:
            result = execute(self.s, LlamaFixture(self.s, ROOT), p, Path(temp) / 'run', 2)
            self.assertEqual(result['costs']['llm_calls'], 3)
            self.assertEqual(result['costs']['known_total_tokens'], 19)
            self.assertEqual(result['costs']['experiments_consumed'], 1)
            self.assertEqual(len(result['state']['history']), 2)
            self.assertEqual(result['state']['history'][0]['verdict']['reason'], 'proposal_failed')
            self.assertFalse(result['trials'][0]['candidate_executed'])
            self.assertEqual(result['planner_service_config'], {'parallel': 1, 'ubatch_size': 128})

    def test_time_budget_stops_before_new_slot(self):
        p, calls = self.planner([])
        with tempfile.TemporaryDirectory() as temp:
            result = execute(self.s, LlamaFixture(self.s, ROOT), p, Path(temp) / 'run', 2, max_seconds=1e-12)
            self.assertEqual(calls, [])
            self.assertEqual(result['stop_reason'], 'time_budget_reached')
            self.assertEqual(result['status'], 'completed')

    def test_v31_bounds_both_fields_in_schema_and_parser(self):
        p, calls = self.planner([self.reply()], version='v3.1')
        result = p.propose(self.state, [])
        self.assertEqual(result['candidate_id'], 'C11')
        schema = calls[0]['response_format']['json_schema']['schema']
        for key in ('hypothesis', 'expected_effect'):
            self.assertEqual(schema['properties'][key]['maxLength'], 96)
            self.assertEqual(schema['properties'][key]['minLength'], 1)
        self.assertEqual(calls[0]['max_tokens'], 300)
        self.assertEqual(calls[0]['seed'], 42)

    def test_v31_rejects_overlong_fields_without_truncating(self):
        for key in ('hypothesis', 'expected_effect'):
            with self.subTest(key=key):
                response = self.reply()
                content = json.loads(response['choices'][0]['message']['content'])
                content[key] = 'x' * 97
                response['choices'][0]['message']['content'] = json.dumps(content)
                p, calls = self.planner([response, response], version='v3.1')
                with self.assertRaises(ValueError):
                    p.propose(self.state, [])
                self.assertEqual(len(calls), 2)
                self.assertIn('x' * 97, p.last_trace[0]['response_content'])

    def test_v3_schema_remains_unbounded_and_finish_reason_is_recorded(self):
        response = self.reply()
        response['choices'][0]['finish_reason'] = 'length'
        p, calls = self.planner([response])
        p.propose(self.state, [])
        schema = calls[0]['response_format']['json_schema']['schema']
        self.assertNotIn('maxLength', schema['properties']['hypothesis'])
        self.assertEqual(p.last_trace[0]['finish_reason'], 'length')
        self.assertTrue(p.last_trace[0]['generation_limit_reached'])
