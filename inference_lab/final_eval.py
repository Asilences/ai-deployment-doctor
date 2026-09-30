"""Fresh paired score for a completed search run on its original workload."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .audit import evaluator_hashes
from .core import check_samples, dump, fingerprint, summarize, validate_candidate


def load_final_candidate(path, settings, backend_name):
    path = Path(path).resolve()
    state = json.loads(path.read_text(encoding='utf-8'))
    run = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'completed' or run.get('backend') != backend_name or run.get('state') != state:
        raise ValueError('Final score requires the retained best.json from a completed run')
    sources = evaluator_hashes(backend_name)
    if run.get('evaluator_sources') != sources or run.get('settings') != settings:
        raise ValueError('Final score requires the original workload and unchanged evaluator')
    expected = fingerprint({'settings': settings, 'backend': backend_name, 'evaluator_sources': sources})
    if state.get('identity') != expected:
        raise ValueError('Saved state identity does not match the completed search run')
    config = validate_candidate(state['config'], settings)
    return config, {'source_run': str(path.parent),
                    'source_best_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'search_planner': run['planner'], 'search_trials': len(run['trials'])}


def evaluate_final(settings, backend, candidate_path, directory):
    """Never modifies source state; all measurements occur after selection."""
    config, provenance = load_final_candidate(candidate_path, settings, backend.name)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    same = config == settings['baseline']
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'settings': settings, 'backend': backend.name, 'provenance': provenance,
              'baseline_config': settings['baseline'], 'selected_config': config,
              'selected_is_baseline': same, 'samples': {'baseline': [], 'selected': []},
              'functional_match': None, 'valid': None,
              'note': 'Independent post-selection score; never updates best.json'}

    def save():
        dump(directory / 'final.json', record)

    def measure(role, candidate, label):
        target = directory / label
        backend.start(candidate, target)
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
            order = [('baseline', settings['baseline'])]
            if not same:
                order.append(('selected', config))
                if index % 2:
                    order.reverse()
            for role, candidate in order:
                outputs = measure(role, candidate, f'{index}-{role}')
                if golden is None:
                    golden = outputs
                matching = matching and outputs == golden
        if same:
            record['samples']['selected'] = list(record['samples']['baseline'])
        baseline = record['samples']['baseline']
        selected = record['samples']['selected']
        record['baseline_median'] = summarize(baseline)
        record['selected_median'] = summarize(selected)
        record['throughput_gain'] = record['selected_median']['output_throughput'] / record['baseline_median']['output_throughput'] - 1
        record['functional_match'] = matching
        record['sample_errors'] = {'baseline': check_samples(baseline, settings),
                                   'selected': check_samples(selected, settings)}
        record['valid'] = matching and not any(record['sample_errors'].values())
        record['status'] = 'completed'
        return record
    except BaseException as error:
        record['status'] = 'failed'
        record['error_type'] = type(error).__name__
        raise
    finally:
        backend.stop()
        save()
