import copy
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_lab.candidate_ids import available_candidates, candidate_catalog
from inference_lab.core import fingerprint
from inference_lab.planner import Planner
from inference_lab.proposal_replay import (input_variant, legacy_brief, payload_for,
    reconstruct, replay_proposal, validate_snapshot)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('proposal_replay_script_test', ROOT / 'scripts/run_proposal_replay.py')
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class ProposalReplayTests(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT / 'configs/windows-1.5b.json').read_text())
        w = self.s['workload']
        self.samples = [{'duration': w['requests'] * w['output_tokens'] / 100,
                         'output_throughput': 100, 'p95_ttft_ms': 50, 'p95_e2el_ms': 100,
                         'completed': w['requests'], 'failed': 0,
                         'input_lens': [w['input_tokens']] * w['requests'],
                         'output_lens': [w['output_tokens']] * w['requests'],
                         'total_output_tokens': w['requests'] * w['output_tokens']} for _ in range(self.s['repeats'])]
        self.state = {'config': self.s['baseline'], 'metrics': {'output_throughput': 100,
                       'p95_ttft_ms': 50, 'p95_e2el_ms': 100}}
        self.past = {'id': 1, 'candidate_executed': True,
                     'proposal': {'candidate_id': 'C02', 'config': candidate_catalog(self.s)[2]['config']},
                     'verdict': {'accepted': False, 'gain': -.1, 'reason': 'insufficient_or_noisy_gain'}}
        self.run = {'status': 'completed', 'git_dirty': False, 'planner': 'local', 'planner_version': 'v3.1',
                    'settings': self.s, 'candidate_catalog': candidate_catalog(self.s), 'hardware': {},
                    'environment_fingerprint': 'environment', 'git_commit': 'a' * 40,
                    'trials': [self.past], 'state': {'future': 'must not leak'}}
        self.set_target([self.past], 'full')

    def set_target(self, history, mode, state=None):
        state = state or self.state
        brief = legacy_brief(self.s, {}, state, history)
        if mode == 'no_history':
            brief['prior_trials'] = []
        self.run['context_mode'] = mode
        self.run['trials'] = history + [{'id': len(history) + 1, 'proposal': {'future': 999999},
            'candidate_executed': True, 'verdict': {'accepted': True, 'gain': 999999},
            'proposal_calls': [{'request': {'available_candidates': available_candidates(self.s, state, history),
                'evidence': brief, 'payload_sha256': fingerprint(payload_for(brief, self.s, ROOT))}}]}]

    def snapshot(self):
        return reconstruct(self.run, self.samples, len(self.run['trials']), ROOT)

    def response(self, candidate='C11', usage=10):
        r = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'candidate_id': candidate, 'hypothesis': 'Batching may help.', 'expected_effect': 'Unknown until measured.'})}}]}
        if usage is not None:
            r['usage'] = {'total_tokens': usage}
        return r

    def replay(self, replies, preflight=None):
        sent = []
        def send(payload):
            sent.append(payload)
            item = replies.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        check = preflight or (lambda _: {'prompt_tokens': 1000, 'context_limit': 2048})
        return replay_proposal(self.snapshot(), 'original', 'full', ROOT, send, check), sent

    def test_future_trial_outputs_and_final_state_do_not_enter_snapshot(self):
        a = self.snapshot()
        self.run['state'] = {'config': {'future': 123}, 'metrics': {'rate': 999999}}
        self.run['trials'][-1]['proposal'] = {'hypothesis': 'future result'}
        self.run['trials'][-1]['verdict']['gain'] = -999999
        self.assertEqual(a, self.snapshot())
        self.assertNotIn('999999', json.dumps(a['decision']))

    def test_historical_payload_and_unchanged_source_are_required(self):
        before = copy.deepcopy(self.run)
        self.snapshot()
        self.assertEqual(self.run, before)
        self.run['trials'][-1]['proposal_calls'][0]['request']['payload_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'historical request'):
            self.snapshot()

    def test_hidden_original_history_is_reconstructed_only_from_past(self):
        self.set_target([self.past], 'no_history')
        snapshot = self.snapshot()
        self.assertEqual(len(input_variant(snapshot, 'original', 'full')['prior_trials']), 1)
        self.assertEqual(input_variant(snapshot, 'original', 'no_history')['prior_trials'], [])

    def test_history_removal_changes_no_other_input_or_feasibility(self):
        snapshot = self.snapshot()
        for representation in ('original', 'owned-v1'):
            full = input_variant(snapshot, representation, 'full')
            hidden = input_variant(snapshot, representation, 'no_history')
            full['prior_trials'] = []
            self.assertEqual(full, hidden)
            self.assertNotIn('C02', [x['candidate_id'] for x in full['available_candidates']])

    def test_owned_input_binds_metrics_without_adding_future_measurements(self):
        snapshot = self.snapshot()
        before = copy.deepcopy(snapshot)
        owned = input_variant(snapshot, 'owned-v1', 'full')
        self.assertEqual(owned['current_metrics']['measured_config_id'], 'C05')
        self.assertEqual(owned['current_metrics']['metrics'], self.state['metrics'])
        self.assertEqual(owned['prior_trials'][0]['reference_config_id'], 'C05')
        self.assertEqual(owned['prior_trials'][0]['gain'], -.1)
        self.assertTrue(all(x['measurement_status'] == 'unknown' for x in owned['available_candidates']))
        self.assertEqual(before, snapshot)

    def test_accepted_past_changes_incumbent_and_reference_owner(self):
        accepted = copy.deepcopy(self.past)
        accepted['verdict'] = {'accepted': True, 'gain': .5, 'reason': 'verified_gain'}
        accepted['candidate'] = {'output_throughput': 150, 'p95_ttft_ms': 50, 'p95_e2el_ms': 100}
        state = {'config': accepted['proposal']['config'], 'metrics': accepted['candidate']}
        self.set_target([accepted], 'full', state)
        snapshot = self.snapshot()
        validate_snapshot(snapshot)
        owned = input_variant(snapshot, 'owned-v1', 'full')
        self.assertEqual(owned['current_metrics']['measured_config_id'], 'C02')
        self.assertEqual(owned['prior_trials'][0]['reference_config_id'], 'C05')

    def test_snapshot_hash_and_recomputed_semantic_tampering_rejected(self):
        snapshot = self.snapshot()
        snapshot['decision']['state']['metrics']['output_throughput'] = 999
        with self.assertRaisesRegex(ValueError, 'hash'):
            validate_snapshot(snapshot)
        snapshot['decision_sha256'] = fingerprint(snapshot['decision'])
        with self.assertRaisesRegex(ValueError, 'evidence'):
            validate_snapshot(snapshot)

    def test_original_payload_matches_existing_v31_request_exactly(self):
        snapshot = self.snapshot()
        captured = []
        planner = Planner('local', self.s, ROOT, version='v3.1')
        planner.environment = {}
        def open_fake(request, timeout):
            captured.append(json.loads(request.data))
            return io.BytesIO(json.dumps(self.response()).encode())
        planner.local_http = type('Client', (), {'open': staticmethod(open_fake)})()
        planner.propose(self.state, [self.past])
        self.assertEqual(captured[0], payload_for(input_variant(snapshot, 'original', 'full'), self.s, ROOT))

    def test_output_schema_system_and_sampling_unchanged_between_representations(self):
        snapshot = self.snapshot()
        a = payload_for(input_variant(snapshot, 'original', 'full'), self.s, ROOT)
        b = payload_for(input_variant(snapshot, 'owned-v1', 'full'), self.s, ROOT)
        a['messages'][1] = b['messages'][1]
        self.assertEqual(a, b)

    def test_valid_proposal_never_executes_or_updates_snapshot(self):
        before = self.snapshot()
        result, sent = self.replay([self.response()])
        self.assertEqual(result['status'], 'valid')
        self.assertFalse(result['candidate_executed'])
        self.assertEqual(len(sent), 1)
        self.assertEqual(before, self.snapshot())

    def test_two_failed_calls_and_costs_retained_without_fallback(self):
        result, sent = self.replay([self.response('C05', 7), self.response('C05', 13)])
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['proposal'])
        self.assertEqual(len(sent), 2)
        self.assertEqual(result['costs']['total_tokens'], 20)
        self.assertEqual(result['costs']['duplicate_calls'], 2)

    def test_missing_usage_timeout_and_interruption_remain_recorded(self):
        result, _ = self.replay([self.response('wrong', 7), self.response('C11', None)])
        self.assertIsNone(result['costs']['total_tokens'])
        self.assertEqual(result['costs']['known_total_tokens'], 7)
        for error, status in [(TimeoutError('fixture'), 'failed'), (KeyboardInterrupt(), 'interrupted')]:
            result, _ = self.replay([error])
            self.assertEqual(result['status'], status)
            self.assertEqual(result['costs']['llm_calls'], 1)
            self.assertIsNone(result['costs']['total_tokens'])

    def test_context_overflow_and_inspection_failure_do_not_generate(self):
        result, sent = self.replay([], lambda _: {'prompt_tokens': 1900, 'context_limit': 2048})
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(sent, [])
        self.assertEqual(len(result['preflight']), 1)
        self.assertEqual(result['costs']['llm_calls'], 0)
        def failed(_):
            raise TimeoutError('fixture tokenize')
        result, sent = self.replay([], failed)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(sent, [])

    def test_prompt_count_mismatch_stops_without_retry(self):
        response = self.response()
        response['usage']['prompt_tokens'] = 500
        result, sent = self.replay([response])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(sent), 1)
        self.assertEqual(result['proposal_calls'][0]['failure_category'], 'prompt_token_count_mismatch')

    def fixture_batch(self, temp):
        root = Path(temp)
        source = root / 'runs/source'
        source.mkdir(parents=True)
        (source / 'run.json').write_text('{}')
        (source / 'best.json').write_text('{"not_an_input": 1}')
        snapshot = self.snapshot()
        model = root / self.s['model_path']
        model.parent.mkdir(parents=True)
        model.write_bytes(b'fixture-not-a-model')
        snapshot['decision']['settings']['model_sha256'] = SCRIPT.file_sha(model)
        snapshot['decision_sha256'] = fingerprint(snapshot['decision'])
        snapshot['source']['files'] = {str(source / 'run.json'): SCRIPT.file_sha(source / 'run.json')}
        selection = root / 'selection.json'
        selection.write_text(json.dumps({'protocol': 'development-same-state-replay-v1',
            'order_seed': 7, 'max_seconds': 900, 'snapshots': [
                {'run_directory': 'runs/source', 'trial_number': 2, 'label': 'fixture'}]}))
        with patch.object(SCRIPT, 'ROOT', root), patch.object(SCRIPT, 'extract_snapshot', return_value=snapshot), \
             patch.object(SCRIPT, 'code_identity', return_value='fixture-code'), \
             patch.object(SCRIPT.subprocess, 'check_output', return_value=('a' * 40).encode()):
            batch = SCRIPT.prepare(selection, root / 'runs/replay')
        return root, source, batch

    def test_frozen_runner_calls_only_planner_and_preserves_sources(self):
        calls = []
        response = self.response()
        response['usage']['prompt_tokens'] = 1000
        class Backend:
            def __init__(self, s, root):
                pass
            def start(self, config, directory):
                calls.append(('start', config))
            def stop(self):
                calls.append(('stop', None))
            def _json(self, path, body=None, timeout=None):
                calls.append((path, None))
                if path == '/props':
                    return {'default_generation_settings': {'n_ctx': 2048}}
                if path == '/apply-template':
                    return {'prompt': 'fixture'}
                if path == '/tokenize':
                    return {'tokens': list(range(1000))}
                if path == '/v1/chat/completions':
                    return copy.deepcopy(response)
                raise AssertionError('Unexpected target operation: ' + path)
        with tempfile.TemporaryDirectory() as temp:
            root, source, batch = self.fixture_batch(temp)
            with patch.object(SCRIPT, 'ROOT', root), patch.object(SCRIPT, 'code_identity', return_value='fixture-code'), \
                 patch.object(SCRIPT, 'provenance', return_value={'git_dirty': False, 'git_commit': 'a' * 40}), \
                 patch.object(SCRIPT, 'ReplayBackend', Backend):
                result = SCRIPT.execute(batch)
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['sessions'][0]['original_files_unchanged'])
            summary = SCRIPT.read(batch / 'summary.json')
            self.assertEqual(summary['slots_recorded'], 4)
            self.assertEqual(summary['costs']['llm_calls'], 4)
            self.assertEqual(summary['costs']['total_tokens'], 40)
            self.assertEqual(summary['candidate_experiments_consumed'], 0)
            self.assertEqual(sum(c[0] == 'start' for c in calls), 1)
            self.assertEqual((source / 'best.json').read_text(), '{"not_an_input": 1}')
            self.assertTrue((batch / 'review_template.csv').exists())

    def test_runner_rejects_changed_original_before_starting_service(self):
        with tempfile.TemporaryDirectory() as temp:
            root, source, batch = self.fixture_batch(temp)
            (source / 'best.json').write_text('{"changed": true}')
            with patch.object(SCRIPT, 'ROOT', root), patch.object(SCRIPT, 'code_identity', return_value='fixture-code'), \
                 patch.object(SCRIPT, 'provenance', return_value={'git_dirty': False, 'git_commit': 'a' * 40}), \
                 patch.object(SCRIPT, 'ReplayBackend') as backend:
                with self.assertRaisesRegex(ValueError, 'Original evidence changed'):
                    SCRIPT.execute(batch)
                backend.assert_not_called()
