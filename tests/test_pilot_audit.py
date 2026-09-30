import copy
import importlib.util
import json
import unittest
from pathlib import Path

from inference_lab.candidate_ids import available_candidates
from inference_lab.proposal_trace import trace_totals

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_audit_test', ROOT / 'scripts/audit_context_pilot.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PilotAuditTests(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT / 'configs/windows-1.5b.json').read_text())
        self.state = {'config': self.s['baseline'], 'metrics': {'rate': 123}}
        self.history = [{'candidate_executed': True,
                         'proposal': {'candidate_id': 'C02', 'config': {'parallel': 1, 'ubatch_size': 256}},
                         'verdict': {'accepted': False, 'gain': -.1, 'reason': 'failed'}}]
        self.evidence = {'environment': {}, 'hardware': {}, 'workload': self.s['workload'],
                         'current_config': self.state['config'], 'current_metrics': self.state['metrics'],
                         'available_candidates': available_candidates(self.s, self.state, self.history),
                         'latency_limits_ms': [self.s['max_p95_ttft_ms'], self.s['max_p95_e2el_ms']],
                         'prior_trials': []}

    def test_audit_detects_semantic_history_leak_and_changed_metrics(self):
        audit.audit_evidence(self.evidence, 'no_history', self.s, self.state, self.history)
        leaked = copy.deepcopy(self.evidence)
        leaked['prior_trials'] = self.history
        with self.assertRaisesRegex(ValueError, 'Semantic-history'):
            audit.audit_evidence(leaked, 'no_history', self.s, self.state, self.history)
        leaked = copy.deepcopy(self.evidence)
        leaked['current_metrics']['rate'] = 124
        with self.assertRaisesRegex(ValueError, 'Measured-metric'):
            audit.audit_evidence(leaked, 'no_history', self.s, self.state, self.history)

    def test_audit_detects_reintroduced_visited_id(self):
        self.evidence['available_candidates'] = available_candidates(self.s, self.state, [])
        with self.assertRaisesRegex(ValueError, 'Feasibility'):
            audit.audit_evidence(self.evidence, 'no_history', self.s, self.state, self.history)

    def test_audit_retains_unknown_usage_and_detects_cost_tampering(self):
        call = {'call_index': 1, 'usage': {'prompt_tokens': None, 'completion_tokens': None,
                                          'total_tokens': None, 'cost': None},
                'elapsed_seconds': 1.0, 'failure_category': 'invalid_output'}
        run = {'trials': [{'proposal_calls': [call], 'candidate_executed': False}],
               'costs': {**trace_totals([call]), 'experiments_consumed': 0}}
        audit.audit_costs(run)
        run['costs']['total_tokens'] = 0
        with self.assertRaisesRegex(ValueError, 'Cost mismatch: total_tokens'):
            audit.audit_costs(run)
