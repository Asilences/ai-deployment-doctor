"""Read-only independent-planner pilot audit: proposals, raw gates, final and cost evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_lab.candidate_ids import available_candidates, candidate_catalog
from inference_lab.core import check_samples, fingerprint, summarize, verdict
from inference_lab.final_eval import load_final_candidate
from inference_lab.proposal_replay import file_sha, read, require
from inference_lab.proposal_trace import trace_totals
from audit_capacity_replay import audit_result as audit_proposal
from audit_context_pilot import audit_costs
from independent_search import live_snapshot, timing_totals
from planning_service import service_identity
from run_proposal_replay import ReplayBackend
import run_planner_performance as runner


def audit_job(job, plan, batch, root):
    directory = Path(batch) / (job['job_id'] + '-search')
    run = read(directory / 'run.json')
    s = plan['settings'][job['task']]
    require(run['settings'] == s and s['experiment_controller_sha256'] == plan['code_identity'], 'Target/controller identity changed')
    require(run['git_dirty'] is False and run['git_commit'] == plan['implementation_commit'], 'Runtime provenance changed')
    require(run['independent_controller']['code_identity'] == plan['code_identity']
            and run['independent_controller']['method'] == job['method'], 'Independent controller missing')
    require(run['candidate_catalog'] == candidate_catalog(s), 'Candidate directory changed')
    require(run['proposal_slot_limit'] == job['iterations'] and len(run['trials']) <= job['iterations'], 'Slot ceiling changed')
    for filename, value in run['code_sources'].items():
        text = subprocess.check_output(['git', 'show', run['git_commit'] + ':inference_lab/' + filename], cwd=root).decode().replace('\r\n', '\n')
        require(fingerprint(text) == value, 'Recorded source mismatch: ' + filename)
    calls = audit_costs(run)
    require(len(calls) <= 2 * job['iterations'], 'Generation ceiling exceeded')
    if job['status'] != 'completed':
        return {'job_id': job['job_id'], 'status': job['status'], 'llm_calls': len(calls),
                'final_gain': None, 'note': 'Failure evidence retained; no valid final score claimed'}
    require(run['status'] == 'completed', 'Completed job has failed search')
    baseline = [read(directory / f'baseline-{n}/benchmark.json') for n in range(s['repeats'])]
    require(check_samples(baseline, s) is None, 'Invalid baseline')
    state = {'config': s['baseline'], 'metrics': summarize(baseline)}
    history, planning_records, ids = [], [], []
    for trial in run['trials']:
        available = available_candidates(s, state, history)
        proposal = trial.get('proposal')
        local = job['method'].startswith('local-')
        if local:
            location = directory / f"proposal-{trial['id']}"
            snapshot = read(location / 'decision.json')
            require(snapshot == live_snapshot(s, run['hardware'], state, history), 'Live decision includes wrong state or future data')
            planning = read(location / 'planning.json')
            planning_records.append(planning)
            profile = plan['planner_profiles'][job['method']]
            require(planning['planner_service'] == service_identity(profile, root), 'Wrong independent planner')
            result = planning['result']
            if result:
                brief = snapshot['decision']['brief']
                item = {'job_id': job['job_id'], 'planner_id': profile['service_id'], 'representation': 'original',
                        'history_mode': 'full', 'input_sha256': fingerprint(brief)}
                audit_proposal(item, result, snapshot, profile, root)
                require(result['proposal_calls'] == trial['proposal_calls'] and result['proposal'] == proposal, 'Search lost planning ledger')
                require(read(location / 'service/server_command.json') == ReplayBackend(profile, root).serve_command(
                    {'parallel': 1, 'ubatch_size': 128}), 'Actual planning service command changed')
            else:
                require(not trial['candidate_executed'] and not trial['proposal_calls'], 'Startup failure fabricated a decision')
        else:
            require(not trial['proposal_calls'], 'Control unexpectedly called a model')
        ids.append(next((c['candidate_id'] for c in candidate_catalog(s) if c['config'] == (proposal or {}).get('config')), None))
        if trial['candidate_executed']:
            require(proposal['config'] in [c['config'] for c in available], 'Executed unavailable candidate')
            a = [read(directory / f"trial-{trial['id']}/{n}-reference/benchmark.json") for n in range(s['repeats'])]
            b = [read(directory / f"trial-{trial['id']}/{n}-candidate/benchmark.json") for n in range(s['repeats'])]
            paths = list((directory / f"trial-{trial['id']}").glob('[0-9]*-*/functional_outputs.json'))
            functional = len(paths) == 2 * s['repeats'] and all(read(p) == run['state']['golden'] for p in paths)
            expected = verdict(b, a, run['state']['golden'] if functional else [], run['state']['golden'], s)
            actual = trial['verdict']
            require(actual == expected or (expected['accepted'] and actual == {'accepted': False, 'reason': 'promotion_restart_failed'}),
                    'Acceptance gate differs from raw evidence')
            restore = directory / f"trial-{trial['id']}" / ('fallback-check' if actual['reason'] == 'promotion_restart_failed' else 'restore-check')
            sample = read(restore / 'benchmark.json')
            require(read(restore / 'functional_outputs.json') == run['state']['golden']
                    and check_samples([sample], {**s, 'repeats': 1}) is None, 'Recovery not verified')
            if actual['accepted']:
                require(sample['output_throughput'] >= summarize(a)['output_throughput'] * (1 + s['min_gain']), 'Promotion restart gain failed')
                state = {'config': proposal['config'], 'metrics': summarize(b)}
        else:
            require(not trial['verdict']['accepted'], 'Nonexecuted proposal accepted')
        history.append(trial)
    require(state['config'] == run['state']['config'] and state['metrics'] == run['state']['metrics'], 'Retained incumbent mismatch')
    require(run['costs']['planner_setup_seconds'] == sum(r['setup_seconds'] or 0 for r in planning_records), 'Lost planner setup cost')
    timing = read(directory / 'target_timing.json')
    require(timing['totals'] == timing_totals(timing['events']) == run['costs']['target_operations_seconds'], 'Target timing mismatch')
    selected, _ = load_final_candidate(directory / 'best.json', s, 'llama_cpp')
    final = read(Path(batch) / (job['job_id'] + '-final/final.json'))
    require(final['status'] == 'completed' and final['valid'] == job['final_valid'], 'Final status mismatch')
    require(final['settings'] == s and final['selected_config'] == selected
            and Path(final['provenance']['source_run']).resolve() == directory.resolve()
            and final['provenance']['source_best_sha256'] == file_sha(directory / 'best.json'), 'Final/source identity mismatch')
    require(all(check_samples(values, s) is None for values in final['samples'].values()), 'Invalid final samples')
    gain = summarize(final['samples']['selected'])['output_throughput'] / summarize(final['samples']['baseline'])['output_throughput'] - 1
    require(gain == final['throughput_gain'] == job['final_gain'], 'Final score mismatch')
    outputs = list((Path(batch) / (job['job_id'] + '-final')).glob('*/functional_outputs.json'))
    require(len(outputs) == s['repeats'] * (1 if final['selected_is_baseline'] else 2)
            and all(read(p) == run['state']['golden'] for p in outputs), 'Final functionality mismatch')
    final_timing = read(Path(batch) / (job['job_id'] + '-final/target_timing.json'))
    require(final_timing['totals'] == timing_totals(final_timing['events']), 'Final timing mismatch')
    return {'job_id': job['job_id'], 'task': job['task'], 'method': job['method'], 'seed': job['seed'],
            'candidate_ids': ids, 'final_valid': final['valid'], 'final_gain': gain, 'llm_calls': len(calls),
            'total_tokens': run['costs']['total_tokens'], 'executed': run['costs']['experiments_consumed']}


def audit_batch(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    plan = manifest['plan']
    require(manifest['status'] == 'completed', 'Batch incomplete; pending jobs cannot receive zero scores')
    require(fingerprint(plan) == manifest['plan_sha256'] and runner.code_identity() == plan['code_identity'], 'Plan or implementation changed')
    for path, sha in plan['readonly_guard'].items():
        require(file_sha(path) == sha, 'Historical evidence changed')
    require(all(s['original_files_unchanged'] and s['runtime_provenance']['git_commit'] == plan['implementation_commit']
                and s['runtime_provenance']['git_dirty'] is False for s in manifest['sessions']), 'Block provenance mismatch')
    require(len(manifest['jobs']) == len(plan['jobs']), 'Job count mismatch')
    rows = []
    for job, frozen in zip(manifest['jobs'], plan['jobs']):
        require(all(job[k] == v for k, v in frozen.items()), 'Job plan changed')
        require(job['status'] in ('completed', 'failed', 'interrupted_not_retried'), 'Job incomplete')
        require((batch / (job['job_id'] + '-search/run.json')).exists(), 'No search evidence for spent job')
        rows.append(audit_job(job, plan, batch, ROOT))
    require(sum(r['llm_calls'] for r in rows) <= plan['llm_call_ceiling'], 'Batch generation budget exceeded')
    require(read(batch / 'summary.json') == runner.summary(batch, manifest), 'Summary mismatch')
    return {'integrity_verified': True, 'jobs_verified': len(rows), 'jobs': rows,
            'original_files_unchanged': True, 'note': 'Two development workloads on one GPU; not general strategy superiority'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    print(json.dumps(audit_batch(parser.parse_args().batch), indent=2))
