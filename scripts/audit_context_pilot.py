"""Read-only integrity audit for completed local context pilots; no re-scoring or writes."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.candidate_ids import available_candidates, candidate_catalog
from inference_lab.core import check_samples, fingerprint, summarize
from inference_lab.proposal_trace import trace_totals


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_costs(run):
    calls = [c for t in run['trials'] for c in t['proposal_calls']]
    for key, value in trace_totals(calls).items():
        require(run['costs'][key] == value, 'Cost mismatch: ' + key)
    require(run['costs']['experiments_consumed'] == sum(t['candidate_executed'] for t in run['trials']),
            'Executed-candidate count mismatch')
    return calls


def audit_evidence(evidence, mode, settings, state, history):
    require(evidence['available_candidates'] == available_candidates(settings, state, history),
            'Feasibility list mismatch')
    require(evidence['current_config'] == state['config'], 'Incumbent mismatch')
    require(evidence['latency_limits_ms'] == [settings['max_p95_ttft_ms'], settings['max_p95_e2el_ms']],
            'Constraint mismatch')
    for field, hidden in (('environment', mode in ('no_environment', 'no_context')),
                          ('hardware', mode in ('no_environment', 'no_context')),
                          ('workload', mode in ('no_workload', 'no_context')),
                          ('current_metrics', mode == 'no_feedback')):
        require((field not in evidence) if hidden else (field in evidence), 'Field visibility mismatch: ' + field)
    if mode != 'no_feedback':
        require(evidence['current_metrics'] == state['metrics'], 'Measured-metric mismatch')
    if mode == 'no_feedback':
        require('prior_trials' not in evidence, 'History leaked into no_feedback')
    else:
        expected = [] if mode == 'no_history' else [
            {'candidate_id': t.get('proposal', {}).get('candidate_id'),
             'config': t.get('proposal', {}).get('config'),
             'accepted': t.get('verdict', {}).get('accepted'), 'gain': t.get('verdict', {}).get('gain'),
             'reason': t.get('verdict', {}).get('reason')} for t in history[-12:]]
        require(evidence['prior_trials'] == expected, 'Semantic-history mismatch')


def audit_batch(batch, repo_root=ROOT):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    require(manifest['protocol'] == 'development-context-pilot-v1', 'Unsupported protocol')
    require(manifest['status'] == 'completed', 'Require a completed batch, not partial evidence')
    rows = []
    for job in manifest['jobs']:
        require(re.fullmatch(r'J[0-9]{2,}', job['job_id']) is not None, 'Invalid job ID')
        require(job['status'] == 'completed', 'Job not completed: ' + job['job_id'])
        directory = batch / (job['job_id'] + '-search')
        final_directory = batch / (job['job_id'] + '-final')
        run = read(directory / 'run.json')
        require(run['status'] == 'completed' and run['git_dirty'] is False, 'Unclean or incomplete run')
        for filename in ('core.py', 'candidate_ids.py', 'proposal_trace.py'):
            require(fingerprint((ROOT / 'inference_lab' / filename).read_text(encoding='utf-8'))
                    == run['code_sources'].get(filename),
                    'Audit helper version mismatch; use the recorded checkout: ' + filename)
        require(re.fullmatch(r'[0-9a-f]{40}', run['git_commit']) is not None, 'Missing source commit')
        # Resolve the recorded committed source, not today's working tree.
        for filename, value in run['code_sources'].items():
            require(re.fullmatch(r'[a-z_][a-z0-9_]*\.py', filename) is not None, 'Invalid source filename')
            text = subprocess.check_output(['git', 'show', run['git_commit'] + ':inference_lab/' + filename],
                                           cwd=repo_root).decode('utf-8').replace('\r\n', '\n')
            require(fingerprint(text) == value, 'Recorded source hash mismatch: ' + filename)
        settings = run['settings']
        require(run['candidate_catalog'] == candidate_catalog(settings), 'Catalogue mismatch')
        require(run['proposal_slot_limit'] == job['iterations'], 'Slot budget mismatch')
        samples = [read(directory / f'baseline-{n}' / 'benchmark.json') for n in range(settings['repeats'])]
        require(check_samples(samples, settings) is None, 'Invalid initial measurements')
        state = {'config': settings['baseline'], 'metrics': summarize(samples)}
        history = []
        calls = audit_costs(run)
        ids = []
        for trial in run['trials']:
            for call in trial['proposal_calls']:
                audit_evidence(call['request']['evidence'], job['method'], settings, state, history)
            proposal = trial.get('proposal', {})
            ids.append(next((c['candidate_id'] for c in run['candidate_catalog']
                             if c['config'] == proposal.get('config')), None))
            if trial['candidate_executed']:
                require(proposal['config'] in [c['config'] for c in available_candidates(settings, state, history)],
                        'Executed an unavailable candidate')
            if trial['verdict']['accepted']:
                state = {'config': proposal['config'], 'metrics': trial['candidate']}
            history.append(trial)
        require(run['state']['config'] == state['config'], 'Final incumbent mismatch')
        final = read(final_directory / 'final.json')
        require(final['status'] == 'completed' and final['valid'] is True, 'Invalid final')
        require(Path(final['provenance']['source_run']).resolve() == directory, 'Wrong source run')
        require(final['provenance']['source_best_sha256'] == hashlib.sha256((directory / 'best.json').read_bytes()).hexdigest(),
                'Best-file hash mismatch')
        require(final['settings'] == settings and final['selected_config'] == state['config'], 'Wrong final selection')
        require(all(check_samples(s, settings) is None for s in final['samples'].values()), 'Invalid final samples')
        gain = summarize(final['samples']['selected'])['output_throughput'] / summarize(final['samples']['baseline'])['output_throughput'] - 1
        require(final['throughput_gain'] == gain == job['final_gain'], 'Final score mismatch')
        golden = read(directory / 'best.json')['golden']
        outputs = list(final_directory.glob('*/functional_outputs.json'))
        expected_output_files = settings['repeats'] * (1 if final['selected_is_baseline'] else 2)
        require(len(outputs) == expected_output_files and all(read(p) == golden for p in outputs), 'Functional outputs mismatch')
        rows.append({'job': job['job_id'], 'method': job['method'], 'seed': job['seed'],
                     'candidate_ids': ids, 'final_gain': gain, 'llm_calls': len(calls),
                     'total_tokens': run['costs']['total_tokens'], 'source_commit': run['git_commit']})
    return {'integrity_verified': True, 'jobs': rows, 'note': 'Integrity checks are not strategy superiority evidence'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(audit_batch(args.batch, args.repo_root), ensure_ascii=False, indent=2))
