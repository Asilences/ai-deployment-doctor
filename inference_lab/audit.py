"""Read-only evaluation of a saved configuration on an unseen workload."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .core import check_samples, dump, fingerprint, summarize, validate_candidate, verdict


def evaluator_hashes(backend_name):
    source = Path(__file__).parent
    names = ['core.py', 'backend.py', 'runner.py']
    if backend_name == 'llama_cpp':
        names.append('llama_backend.py')
    return {name: fingerprint((source / name).read_text(encoding='utf-8')) for name in names}


def load_audit_candidate(path, holdout_settings, backend_name):
    path = Path(path).resolve()
    state = json.loads(path.read_text(encoding='utf-8'))
    run = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'completed' or run.get('backend') != backend_name or run.get('state') != state:
        raise ValueError('Candidate must be the final retained state of a completed run on this backend')
    sources = evaluator_hashes(backend_name)
    if run.get('evaluator_sources') != sources:
        raise ValueError('Evaluator source changed since the candidate was selected')
    train = run['settings']
    expected = fingerprint({'settings': train, 'backend': backend_name, 'evaluator_sources': sources})
    if state.get('identity') != expected:
        raise ValueError('Candidate provenance fingerprint mismatch')
    without_workload = lambda settings: {k: v for k, v in settings.items() if k != 'workload'}
    if without_workload(train) != without_workload(holdout_settings):
        raise ValueError('Holdout may only change workload; keep model, runtime and acceptance settings fixed')
    if train['workload'] == holdout_settings['workload']:
        raise ValueError('Holdout workload must differ from the selection workload')
    candidate = validate_candidate(state['config'], holdout_settings)
    if candidate == holdout_settings['baseline']:
        raise ValueError('Saved candidate is identical to the baseline')
    if not state.get('generation') or not any(
            item.get('verdict', {}).get('accepted') and item.get('proposal', {}).get('config') == candidate
            for item in state.get('history', [])):
        raise ValueError('Candidate has no recorded accepted experiment')
    return candidate, {'source_run': str(path.parent), 'source_best_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def audit_holdout(settings, backend, candidate_path, directory):
    """Measure fixed baseline/candidate; never write to the source run or promote."""
    directory = Path(directory).resolve()
    candidate, provenance = load_audit_candidate(candidate_path, settings, backend.name)
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'backend': backend.name, 'settings': settings, 'provenance': provenance,
              'baseline_config': settings['baseline'], 'candidate_config': candidate,
              'samples': {'baseline': [], 'candidate': []}, 'functional_match': None,
              'holdout_verdict': None, 'note': 'Audit only; never updates the saved candidate'}

    def save():
        dump(directory / 'audit.json', record)

    def evaluate(role, config, label):
        target = directory / label
        backend.start(config, target)
        outputs = backend.outputs()
        dump(target / 'functional_outputs.json', outputs)
        if hasattr(backend, 'warmup'):
            backend.warmup(target)
        sample = backend.measure(target)
        record['samples'][role].append(sample)
        save()
        return outputs

    save()
    try:
        golden = None
        matching = True
        for index in range(settings['repeats']):
            order = [('baseline', settings['baseline']), ('candidate', candidate)]
            if index % 2:
                order.reverse()
            for role, config in order:
                outputs = evaluate(role, config, f'{index}-{role}')
                if golden is None:
                    golden = outputs
                matching = matching and outputs == golden
        record['functional_match'] = matching
        baseline = record['samples']['baseline']
        contender = record['samples']['candidate']
        record['baseline_median'] = summarize(baseline)
        record['candidate_median'] = summarize(contender)
        record['throughput_gain'] = record['candidate_median']['output_throughput'] / record['baseline_median']['output_throughput'] - 1
        record['sample_errors'] = {'baseline': check_samples(baseline, settings),
                                   'candidate': check_samples(contender, settings)}
        record['holdout_verdict'] = verdict(contender, baseline, golden if matching else [], golden, settings)
        record['status'] = 'completed'
        return record
    except BaseException as error:
        record['status'] = 'failed'
        record['error_type'] = type(error).__name__
        raise
    finally:
        backend.stop()
        save()
