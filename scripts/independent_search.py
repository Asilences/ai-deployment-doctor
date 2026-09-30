"""Live bounded ID planning, with exclusive GPU ownership and separate planner identity."""
import copy
import platform
import time
from pathlib import Path

from inference_lab.candidate_ids import SearchSpaceExhausted, candidate_catalog
from inference_lab.core import dump, fingerprint
from inference_lab.llama_backend import LlamaCppBackend
from inference_lab.proposal_replay import legacy_brief, require, validate_snapshot
from inference_lab.proposal_trace import utc_now
from capacity_proposal import capacity_proposal
from planning_service import service_identity, validate_profile
from run_proposal_replay import ReplayBackend


class PlanningIntegrityError(Exception):
    """Stop the search, rather than spending another slot after an integrity failure."""


def live_snapshot(settings, hardware, state, history):
    incumbent = copy.deepcopy(settings['baseline'])
    owners = []
    for trial in history:
        owners.append(copy.deepcopy(incumbent))
        if trial.get('verdict', {}).get('accepted'):
            incumbent = copy.deepcopy(trial['proposal']['config'])
    require(incumbent == state['config'], 'Live incumbent/history mismatch')
    current = {k: copy.deepcopy(state[k]) for k in ('config', 'metrics')}
    decision = {'settings': copy.deepcopy(settings), 'hardware': copy.deepcopy(hardware),
                'state': current, 'history': copy.deepcopy(history), 'history_reference_configs': owners,
                'brief': legacy_brief(settings, hardware, current, history),
                'environment_fingerprint': fingerprint({'settings': settings, 'platform': platform.platform(),
                                                       'hardware': hardware, 'backend': 'llama_cpp'})}
    snapshot = {'version': 'same-state-snapshot-v1', 'decision': decision,
                'decision_sha256': fingerprint(decision), 'source': {'kind': 'live', 'prior_trials': len(history)}}
    validate_snapshot(snapshot)
    return snapshot


class IndependentPlanner:
    kind = 'local'
    version = 'v3.1-independent-v1'
    context_mode = 'full'
    seed = 42

    def __init__(self, settings, profile, root, target, directory, backend_factory=ReplayBackend):
        self.s, self.profile, self.root = settings, validate_profile(profile, root), root
        require(settings['port'] != profile['port'], 'Planning endpoint overlaps target')
        require(settings['max_model_len'] == profile['max_model_len'], 'Frozen context differs')
        self.target, self.directory, self.backend_factory = target, Path(directory), backend_factory
        self.catalog = candidate_catalog(settings)
        self.environment, self.last_trace, self.last_available = {}, [], []
        self.n = 0

    def propose(self, state, history):
        self.n += 1
        self.last_trace, self.last_available = [], []
        snapshot = live_snapshot(self.s, self.environment, state, history)
        self.last_available = snapshot['decision']['brief']['available_candidates']
        if not self.last_available:
            raise SearchSpaceExhausted('No untried candidate remains; no service or generation call')
        directory = self.directory / f'proposal-{self.n}'
        dump(directory / 'decision.json', snapshot)
        record = {'started_utc': utc_now(), 'planner_service': service_identity(self.profile, self.root),
                  'decision_sha256': snapshot['decision_sha256'], 'status': 'running', 'result': None,
                  'setup_seconds': None, 'shutdown_seconds': None, 'target_release_seconds': None}
        started = time.monotonic()
        backend = None
        try:
            released = time.monotonic()
            self.target.stop()
            record['target_release_seconds'] = time.monotonic() - released
            backend = self.backend_factory(self.profile, self.root)
            setup = time.monotonic()
            try:
                backend.start({'parallel': 1, 'ubatch_size': 128}, directory / 'service')
            finally:
                record['setup_seconds'] = time.monotonic() - setup
            props = backend._json('/props')
            require(props['default_generation_settings']['n_ctx'] == self.profile['max_model_len'], 'Wrong planning context')
            record['props'] = props
            def preflight(payload):
                prompt = backend._json('/apply-template', payload, timeout=30)['prompt']
                tokens = backend._json('/tokenize', {'content': prompt, 'add_special': True,
                                                   'parse_special': True}, timeout=30)['tokens']
                return {'method': 'apply-template/tokenize', 'prompt_tokens': len(tokens),
                        'prompt_sha256': fingerprint(prompt), 'context_limit': self.profile['max_model_len']}
            result = capacity_proposal(snapshot, 'original', 'full', self.profile, self.root,
                lambda payload: backend._json('/v1/chat/completions', payload, timeout=90), preflight)
            record['result'] = result
            self.last_trace = result['proposal_calls']
            if result['status'] == 'interrupted':
                raise KeyboardInterrupt('Planning interrupted; ledger preserved')
            if any(c['failure_category'] == 'prompt_token_count_mismatch' for c in self.last_trace):
                raise PlanningIntegrityError('Prompt-token integrity failure; no next slot')
            if result['status'] != 'valid':
                raise ValueError('Independent proposal failed; no candidate substituted')
            record['status'] = 'valid'
            return result['proposal']
        except BaseException as error:
            record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error_type=type(error).__name__)
            raise
        finally:
            if backend is not None:
                stopped = time.monotonic()
                backend.stop()
                record['shutdown_seconds'] = time.monotonic() - stopped
            record.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
            dump(directory / 'planning.json', record)


class TimedTargetBackend(LlamaCppBackend):
    """Wall-clock telemetry only; serving commands, measurements and gates stay unchanged."""
    def __init__(self, settings, root):
        super().__init__(settings, root)
        self.events, self.depth = [], 0

    def timed(self, operation, action, directory=None):
        level = self.depth
        self.depth += 1
        started = time.monotonic()
        row = {'operation': operation, 'depth': level, 'directory': str(directory) if directory else None,
               'started_utc': utc_now(), 'status': 'running'}
        try:
            value = action()
            row['status'] = 'completed'
            return value
        except BaseException as error:
            row.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            self.depth -= 1
            row.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
            self.events.append(row)

    def start(self, candidate, directory):
        return self.timed('start', lambda: super(TimedTargetBackend, self).start(candidate, directory), directory)

    def outputs(self):
        return self.timed('functional', lambda: super(TimedTargetBackend, self).outputs())

    def warmup(self, directory):
        return self.timed('warmup', lambda: super(TimedTargetBackend, self).warmup(directory), directory)

    def measure(self, directory):
        return self.timed('measure', lambda: super(TimedTargetBackend, self).measure(directory), directory)

    def stop(self):
        return self.timed('stop', lambda: super(TimedTargetBackend, self).stop())


def timing_totals(events):
    # start includes its nested stop; never add both to a total.
    top = [e for e in events if e['depth'] == 0]
    return {operation: sum(e['elapsed_seconds'] for e in top if e['operation'] == operation)
            for operation in ('start', 'functional', 'warmup', 'measure', 'stop')}
