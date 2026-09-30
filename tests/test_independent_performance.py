import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from inference_lab.core import dump, fingerprint
from inference_lab.final_eval import load_final_candidate
from inference_lab.proposal_replay import read
import inference_lab.runner as old_runner
import independent_search as independent
import run_planner_performance as pilot
import test_capacity_replay as fixtures


class IndependentPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.s = read(ROOT / 'configs/pilot-serial-windows-1.5b.json')
        self.profile = read(ROOT / 'configs/planner-local-3b.json')
        self.spec = read(ROOT / 'configs/planner-performance-20260930.json')
        self.state = {'config': self.s['baseline'], 'metrics': {'output_throughput': 100, 'p95_ttft_ms': 50, 'p95_e2el_ms': 100}}

    def test_frozen_schedule_covers_all_methods_seeds_and_tasks(self):
        jobs = pilot.build_jobs(self.spec)
        self.assertEqual(jobs, pilot.build_jobs(copy.deepcopy(self.spec)))
        self.assertEqual(len(jobs), 12)
        self.assertEqual(sum(j['iterations'] for j in jobs), 22)
        for task in ('serial', 'prefill'):
            block = [j for j in jobs if j['task'] == task]
            self.assertEqual({j['seed'] for j in block if j['method'] == 'random'}, {100, 101, 102})
            self.assertEqual(len([j for j in block if j['method'].startswith('local-')]), 2)
            self.assertEqual([j['iterations'] for j in block if j['method'] == 'fixed'], [1])

    def test_live_snapshot_has_exact_past_metrics_and_no_mutation(self):
        before = copy.deepcopy(self.state)
        snap = independent.live_snapshot(self.s, {'gpu': 'fixture'}, self.state, [])
        self.assertEqual(snap['decision']['brief']['current_metrics'], before['metrics'])
        self.assertEqual(snap['decision']['brief']['prior_trials'], [])
        self.assertEqual(self.state, before)
        snap['decision']['state']['metrics']['output_throughput'] = 999
        self.assertEqual(self.state, before)

    def test_live_accepted_history_binds_reference_owner_and_incumbent(self):
        trial = {'candidate_executed': True, 'proposal': {'candidate_id': 'C11', 'config': {'parallel': 4, 'ubatch_size': 512}},
                 'verdict': {'accepted': True, 'gain': .3, 'reason': 'verified_gain'}}
        state = {'config': trial['proposal']['config'], 'metrics': self.state['metrics']}
        snap = independent.live_snapshot(self.s, {}, state, [trial])
        self.assertEqual(snap['decision']['history_reference_configs'], [self.s['baseline']])
        with self.assertRaises(ValueError):
            independent.live_snapshot(self.s, {}, self.state, [trial])

    def test_target_and_planner_are_never_active_together(self):
        events = []
        class Target:
            active = True
            def stop(self):
                self.active = False
                events.append('target_stop')
        target = Target()
        class Service:
            def __init__(self, p, root): pass
            def start(self, config, directory):
                assert not target.active
                events.append('planner_start')
            def stop(self): events.append('planner_stop')
            def _json(self, path, body=None, **kwargs):
                if path == '/props': return {'default_generation_settings': {'n_ctx': 2048}}
                if path == '/apply-template': return {'prompt': 'x'}
                if path == '/tokenize': return {'tokens': list(range(1000))}
                if path == '/v1/chat/completions':
                    events.append('generation')
                    return {'choices': [{'message': {'content': json.dumps({'candidate_id': 'C11',
                            'hypothesis': 'May improve.', 'expected_effect': 'Unknown.'})}}], 'usage': {'total_tokens': 11}}
                raise AssertionError(path)
        with tempfile.TemporaryDirectory() as d:
            planner = independent.IndependentPlanner(self.s, self.profile, ROOT, target, Path(d), Service)
            result = planner.propose(self.state, [])
            record = read(Path(d) / 'proposal-1/planning.json')
            self.assertEqual(result['config'], {'parallel': 4, 'ubatch_size': 512})
            self.assertEqual(events, ['target_stop', 'planner_start', 'generation', 'planner_stop'])
            self.assertEqual(record['result']['proposal_calls'], planner.last_trace)
            self.assertIsNotNone(record['setup_seconds'])
            self.assertIsNotNone(record['shutdown_seconds'])

    def test_planner_start_failure_closes_service_and_keeps_cost_record(self):
        stopped = []
        class Target:
            def stop(self): stopped.append('target')
        class Service:
            def __init__(self, p, r): pass
            def start(self, c, d): raise RuntimeError('controlled startup failure')
            def stop(self): stopped.append('planner')
        with tempfile.TemporaryDirectory() as d:
            planner = independent.IndependentPlanner(self.s, self.profile, ROOT, Target(), Path(d), Service)
            with self.assertRaises(RuntimeError): planner.propose(self.state, [])
            record = read(Path(d) / 'proposal-1/planning.json')
            self.assertEqual(record['status'], 'failed')
            self.assertIsNone(record['result'])
            self.assertEqual(planner.last_trace, [])
            self.assertEqual(stopped, ['target', 'planner'])

    def test_nested_target_stop_is_not_added_twice(self):
        target = independent.TimedTargetBackend.__new__(independent.TimedTargetBackend)
        target.events, target.depth = [], 0
        with patch.object(independent.time, 'monotonic', side_effect=[0, 1, 3, 5]):
            target.timed('start', lambda: target.timed('stop', lambda: None))
        totals = independent.timing_totals(target.events)
        self.assertEqual(totals['start'], 5)
        self.assertEqual(totals['stop'], 0)

    def test_timing_adapter_does_not_change_target_command(self):
        from inference_lab.llama_backend import LlamaCppBackend
        a = LlamaCppBackend.__new__(LlamaCppBackend)
        b = independent.TimedTargetBackend.__new__(independent.TimedTargetBackend)
        for obj in (a, b):
            obj.s, obj.exe, obj.model = self.s, ROOT / 'server.exe', ROOT / self.s['model_path']
        self.assertEqual(a.serve_command(self.s['baseline']), b.serve_command(self.s['baseline']))

    def mock_assets(self, root):
        spec = copy.deepcopy(self.spec)
        spec['minimum_search_start_seconds'] = 1
        for name in ('configs', 'models', 'runs'): (root / name).mkdir()
        profiles = {}
        for method, config in spec['planner_profiles'].items():
            profile = read(ROOT / config)
            data = profile['service_id'].encode()
            path = root / profile['model_path']
            path.write_bytes(data)
            profile.update(model_sha256=hashlib.sha256(data).hexdigest(), model_size_bytes=len(data))
            dump(root / config, profile)
            profiles[method] = profile
        for task, name in spec['tasks'].items():
            s = read(ROOT / name)
            s['model_sha256'] = profiles['local-1.5b']['model_sha256']
            dump(root / name, s)
        dump(root / 'selection.json', spec)
        return spec

    def test_complete_pilot_runs_real_control_flow_and_fresh_finals_on_mock_measurements(self):
        runtime = {'git_dirty': False, 'git_commit': 'a' * 40, 'hardware': {}, 'code_sources': {}}
        targets = []
        class Target:
            name = 'llama_cpp'
            def __init__(self, s, root):
                self.s, self.events, self.active = s, [], False
                targets.append(self)
            def start(self, config, directory):
                self.config, self.active = config, True
                Path(directory).mkdir(parents=True, exist_ok=True)
            def stop(self): self.active = False
            def outputs(self): return ['one', 'two', 'three']
            def warmup(self, directory): pass
            def measure(self, directory):
                w = self.s['workload']
                rate = 100 if self.config == self.s['baseline'] else 130
                row = {'duration': w['requests'] * w['output_tokens'] / rate, 'output_throughput': rate,
                       'p95_ttft_ms': 50, 'p95_e2el_ms': 100, 'completed': w['requests'], 'failed': 0,
                       'input_lens': [w['input_tokens']] * w['requests'], 'output_lens': [w['output_tokens']] * w['requests'],
                       'total_output_tokens': w['requests'] * w['output_tokens']}
                dump(Path(directory) / 'benchmark.json', row)
                return row
        class Service:
            def __init__(self, profile, root): pass
            def start(self, config, directory):
                assert not any(t.active for t in targets)
            def stop(self): pass
            def _json(self, path, body=None, **kwargs):
                if path == '/props': return {'default_generation_settings': {'n_ctx': 2048}}
                if path == '/apply-template': return {'prompt': 'fixed'}
                if path == '/tokenize': return {'tokens': list(range(1000))}
                if path == '/v1/chat/completions':
                    candidate = body['response_format']['json_schema']['schema']['properties']['candidate_id']['enum'][0]
                    return {'choices': [{'message': {'content': json.dumps({'candidate_id': candidate,
                            'hypothesis': 'May help.', 'expected_effect': 'Unknown.'})}}],
                            'usage': {'prompt_tokens': 1000, 'completion_tokens': 7, 'total_tokens': 1007}}
                raise AssertionError(path)
        def planner_factory(s, p, root, target, directory):
            return independent.IndependentPlanner(s, p, root, target, directory, Service)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.mock_assets(root)
            with patch.object(pilot, 'ROOT', root), patch.object(pilot, 'code_identity', return_value='test'), \
                 patch.object(pilot, 'provenance', return_value=runtime), patch.object(old_runner, 'provenance', return_value=runtime), \
                 patch.object(pilot, 'TimedTargetBackend', Target), patch.object(pilot, 'IndependentPlanner', planner_factory):
                batch = pilot.prepare(root / 'selection.json', root / 'runs/pilot')
                result = pilot.execute_block(batch)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(len(result['jobs']), 12)
                self.assertTrue(all(j['final_valid'] for j in result['jobs']))
                rows = read(batch / 'summary.json')
                self.assertEqual({r['method_label'] for r in rows}, {'local-1.5b', 'local-3b', 'random', 'fixed'})
                self.assertEqual(sum(r['llm_calls'] for r in rows), 8)
                self.assertTrue(all(r['final_record_count'] == 1 for r in rows))
                source = batch / (result['jobs'][0]['job_id'] + '-search/best.json')
                augmented = result['plan']['settings'][result['jobs'][0]['task']]
                load_final_candidate(source, augmented, 'llama_cpp')
                original = {k: v for k, v in augmented.items() if not k.startswith('experiment_')}
                with self.assertRaises(ValueError): load_final_candidate(source, original, 'llama_cpp')
                with self.assertRaises(ValueError): pilot.execute_block(batch)
                self.assertFalse(any(t.active for t in targets))


if __name__ == '__main__':
    unittest.main()
