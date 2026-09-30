import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from inference_lab.core import dump, fingerprint
from inference_lab.proposal_replay import input_variant, payload_for, read
from inference_lab.proposal_trace import trace_totals
from planning_service import planner_payload, service_identity, validate_profile
from capacity_proposal import capacity_proposal
import run_capacity_replay as runner
import audit_capacity_replay as audit

# Reuse the public synthetic baseline fixture, never local experimental outcomes.
sys.path.insert(0, str(ROOT / 'tests'))
import test_proposal_replay as fixtures


class CapacityReplayTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ProposalReplayTests()
        fixture.setUp()
        self.snapshot = fixture.snapshot()
        self.response = fixture.response
        self.small = read(ROOT / 'configs/planner-local-1.5b.json')
        self.large = read(ROOT / 'configs/planner-local-3b.json')

    def replay(self, responses, profile=None, preflight=None):
        sent = []
        def send(payload):
            sent.append(payload)
            reply = responses.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply
        result = capacity_proposal(self.snapshot, 'original', 'full', profile or self.large, ROOT, send,
                                   preflight or (lambda _: {'prompt_tokens': 1000, 'context_limit': 2048}))
        return result, sent

    def job(self):
        brief = input_variant(self.snapshot, 'original', 'full')
        return {'job_id': 'J00', 'snapshot_id': 'S00', 'representation': 'original', 'history_mode': 'full',
                'planner_id': self.large['service_id'], 'input_sha256': fingerprint(brief)}

    def test_control_payload_is_exactly_legacy(self):
        brief = input_variant(self.snapshot, 'original', 'full')
        self.assertEqual(planner_payload(brief, self.snapshot['decision']['settings'], self.small, ROOT),
                         payload_for(brief, self.snapshot['decision']['settings'], ROOT))

    def test_large_planner_changes_only_model_selector(self):
        before = copy.deepcopy(self.snapshot)
        brief = input_variant(self.snapshot, 'owned-v1', 'full')
        original = payload_for(brief, self.snapshot['decision']['settings'], ROOT)
        actual = planner_payload(brief, self.snapshot['decision']['settings'], self.large, ROOT)
        self.assertNotEqual(original['model'], actual['model'])
        actual['model'] = original['model']
        self.assertEqual(actual, original)
        self.assertEqual(before, self.snapshot)
        self.assertIn('1.5B', brief['environment']['model'])

    def test_target_fields_cannot_enter_service_profile(self):
        for key in ('baseline', 'workload', 'search_space', 'min_gain'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile(self.large | {key: {}}, ROOT)

    def test_model_and_runtime_paths_cannot_escape(self):
        for key, value in [('model_path', '../bad.gguf'), ('model_path', 'configs/bad.gguf'),
                           ('llama_cpp_binary', '../llama-server.exe')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_profile(self.large | {key: value}, ROOT)

    def test_model_revision_and_checksums_must_be_pinned(self):
        for key, value in [('model_revision', 'main'), ('model_sha256', 'unknown'), ('port', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile(self.large | {key: value}, ROOT)

    def test_actual_planner_and_endpoint_are_in_each_call(self):
        result, sent = self.replay([self.response()])
        identity = service_identity(self.large, ROOT)
        self.assertEqual(result['planner_service'], identity)
        self.assertEqual(result['proposal_calls'][0]['request']['planner_service'], identity)
        self.assertEqual(result['proposal_calls'][0]['request']['payload'], sent[0])
        self.assertFalse(result['candidate_executed'])
        self.assertTrue(audit.audit_result(self.job(), result, self.snapshot, self.large, ROOT)['actual_planner_identity_verified'])

    def test_retry_accumulates_both_calls_without_fallback(self):
        result, sent = self.replay([self.response('C02', 11), self.response('C11', 13)])
        self.assertEqual(result['costs']['total_tokens'], 24)
        self.assertEqual(result['costs']['retry_calls'], 1)
        self.assertEqual(len(sent), 2)
        self.assertEqual(result['proposal']['candidate_id'], 'C11')
        audit.audit_result(self.job(), result, self.snapshot, self.large, ROOT)

    def test_two_invalid_calls_end_without_random_candidate(self):
        result, sent = self.replay([self.response('C02', 11), self.response('C02', 13)])
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['proposal'])
        self.assertEqual(result['costs']['total_tokens'], 24)
        self.assertEqual(len(sent), 2)
        audit.audit_result(self.job(), result, self.snapshot, self.large, ROOT)

    def test_timeout_preserves_unknown_cost_and_does_not_retry(self):
        result, sent = self.replay([TimeoutError('test')])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['costs']['llm_calls'], 1)
        self.assertIsNone(result['costs']['total_tokens'])
        self.assertEqual(result['proposal_calls'][0]['failure_category'], 'timeout')
        self.assertEqual(len(sent), 1)

    def test_capacity_failure_consumes_no_generation_calls(self):
        result, sent = self.replay([], preflight=lambda _: {'prompt_tokens': 1900, 'context_limit': 2048})
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(sent, [])
        self.assertEqual(result['costs']['llm_calls'], 0)
        self.assertEqual(result['preflight'][0]['status'], 'failed')

    def test_usage_guard_failure_is_preserved(self):
        response = self.response()
        response['usage']['prompt_tokens'] = 999
        result, sent = self.replay([response])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['proposal_calls'][0]['failure_category'], 'prompt_token_count_mismatch')
        self.assertEqual(len(sent), 1)

    def test_audit_rejects_actual_selector_and_endpoint_tampering(self):
        result, _ = self.replay([self.response()])
        for variant in ('selector', 'endpoint', 'evidence'):
            changed = copy.deepcopy(result)
            request = changed['proposal_calls'][0]['request']
            if variant == 'selector':
                request['payload']['model'] = str(ROOT / self.small['model_path'])
                request['payload_sha256'] = fingerprint(request['payload'])
            elif variant == 'endpoint':
                request['planner_service']['endpoint'] = 'http://127.0.0.1:8765/v1/chat/completions'
            else:
                request['evidence']['environment']['model'] = 'planner mistaken for target'
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                audit.audit_result(self.job(), changed, self.snapshot, self.large, ROOT)
        self.assertEqual(result['planner_service'], service_identity(self.large, ROOT))

    def fixture(self, root):
        import hashlib
        profiles = []
        for p in (self.small, self.large):
            data = p['service_id'].encode()
            model = root / p['model_path']
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(data)
            profiles.append(p | {'model_size_bytes': len(data), 'model_sha256': hashlib.sha256(data).hexdigest()})
        batch = root / 'runs/new'
        batch.mkdir(parents=True)
        dump(batch / 'S00.json', self.snapshot)
        protected = root / 'best.json'
        dump(protected, {'must_not_change': 123})
        jobs = [{**self.job(), 'job_id': f'J{i:02d}', 'planner_id': p['service_id']} for i, p in enumerate(profiles)]
        plan = {'protocol': 'development-planner-capacity-v1', 'code_identity': 'test',
                'implementation_commit': 'a' * 40, 'selection': {'max_seconds': 900},
                'snapshots': [{'snapshot_id': 'S00', 'file': 'S00.json', 'file_sha256': runner.file_sha(batch / 'S00.json')}],
                'planner_profiles': profiles, 'planner_order': [p['service_id'] for p in profiles],
                'readonly_guard': {str(protected): runner.file_sha(protected)}, 'jobs': jobs,
                'service_config': {'parallel': 1, 'ubatch_size': 128}}
        dump(batch / 'manifest.json', {'plan': plan, 'plan_sha256': fingerprint(plan), 'status': 'prepared',
                                     'jobs': [{**j, 'status': 'pending'} for j in jobs], 'sessions': []})
        return batch, protected

    def fake_backend(self, fail=False):
        response = self.response()
        seen = []
        class Fake:
            def __init__(self, profile, root):
                self.profile = profile
                seen.append(('construct', profile['service_id']))
            def start(self, config, directory):
                if fail:
                    raise RuntimeError('controlled startup failure')
                seen.append(('start', self.profile['service_id']))
            def stop(self):
                seen.append(('stop', self.profile['service_id']))
            def _json(self, path, body=None, **kwargs):
                if path == '/props':
                    return {'default_generation_settings': {'n_ctx': 2048}}
                if path == '/apply-template':
                    return {'prompt': 'fixed prompt'}
                if path == '/tokenize':
                    return {'tokens': list(range(1000))}
                if path == '/v1/chat/completions':
                    seen.append(('request', body['model']))
                    return response
                raise AssertionError('Target operation was invoked: ' + path)
        return Fake, seen

    def execute_fixture(self, batch, root, backend):
        with patch.object(runner, 'ROOT', root), patch.object(runner, 'code_identity', return_value='test'), \
             patch.object(runner, 'provenance', return_value={'git_dirty': False, 'git_commit': 'a' * 40}), \
             patch.object(runner.legacy, 'ReplayBackend', backend):
            return runner.execute(batch)

    def test_two_services_use_identical_target_evidence_and_leave_best_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, protected = self.fixture(root)
            before = protected.read_bytes()
            fake, seen = self.fake_backend()
            result = self.execute_fixture(batch, root, fake)
            self.assertEqual(result['status'], 'completed')
            a, b = (read(batch / (j['job_id'] + '.json')) for j in result['jobs'])
            self.assertEqual(a['proposal_calls'][0]['request']['evidence'], b['proposal_calls'][0]['request']['evidence'])
            self.assertNotEqual(a['proposal_calls'][0]['request']['payload']['model'], b['proposal_calls'][0]['request']['payload']['model'])
            self.assertEqual(protected.read_bytes(), before)
            self.assertEqual([x[0] for x in seen], ['construct', 'start', 'request', 'stop'] * 2)
            self.assertEqual(read(batch / 'summary.json')['candidate_experiments_consumed'], 0)
            with self.assertRaises(ValueError):
                self.execute_fixture(batch, root, fake)

    def test_failed_service_is_recorded_and_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, _ = self.fixture(root)
            fake, seen = self.fake_backend(fail=True)
            result = self.execute_fixture(batch, root, fake)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['stop_reason'], 'service_failure')
            self.assertEqual(len(result['sessions']), 1)
            self.assertEqual(read(batch / 'summary.json')['costs']['llm_calls'], 0)
            self.assertEqual([x[0] for x in seen], ['construct', 'stop'])

    def test_changed_original_evidence_stops_before_service_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, protected = self.fixture(root)
            protected.write_text('tampered')
            fake, seen = self.fake_backend()
            with self.assertRaises(ValueError):
                self.execute_fixture(batch, root, fake)
            self.assertEqual(seen, [])

    def test_changed_asset_stops_before_consuming_any_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, _ = self.fixture(root)
            (root / self.large['model_path']).write_bytes(b'tampered')
            fake, seen = self.fake_backend()
            with self.assertRaises(ValueError):
                self.execute_fixture(batch, root, fake)
            self.assertEqual(seen, [])


if __name__ == '__main__':
    unittest.main()
