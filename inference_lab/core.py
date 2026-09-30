import hashlib
import json
import math
import os
import statistics
from pathlib import Path


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_candidate(value, settings):
    space = settings['search_space']
    if not isinstance(value, dict) or set(value) != set(space):
        raise ValueError('Candidate must contain exactly the two permitted configuration keys')
    for key, allowed in space.items():
        if type(value[key]) is not int or value[key] not in allowed:
            raise ValueError('Candidate value is outside the permitted search space: ' + key)
    return dict(value)


def validate_settings(s):
    validate_candidate(s['baseline'], s)
    for k in ('requests', 'input_tokens', 'output_tokens', 'concurrency', 'seed'):
        if type(s['workload'][k]) is not int or s['workload'][k] < (0 if k == 'seed' else 1):
            raise ValueError('Invalid workload: ' + k)
    if s['repeats'] < 2 or s['min_gain'] <= 0:
        raise ValueError('Use at least two repeats and a positive minimum gain')
    if s['workload']['input_tokens'] + s['workload']['output_tokens'] > s['max_model_len']:
        raise ValueError('Workload exceeds the fixed context limit')
    if s.get('backend') == 'llama_cpp':
        if max(s['search_space']['ubatch_size']) > s['batch_size']:
            raise ValueError('Physical batch cannot exceed the fixed logical batch')
    elif not 0 < s['gpu_memory_utilization'] < 1:
        raise ValueError('Invalid GPU memory fraction')


def summarize(samples):
    return {key: statistics.median(x[key] for x in samples) for key in
            ('output_throughput', 'p95_ttft_ms', 'p95_e2el_ms')}


def check_samples(samples, s):
    """Fail closed on missing, non-finite, short-output or incomplete measurements."""
    if len(samples) != s['repeats']:
        return 'incorrect_repeat_count'
    w = s['workload']
    for row in samples:
        for key in ('duration', 'output_throughput', 'p95_ttft_ms', 'p95_e2el_ms'):
            v = row.get(key)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                return 'invalid_metric:' + key
        if row.get('completed') != w['requests'] or row.get('failed', 0) != 0:
            return 'incomplete_requests'
        if any(row.get('errors', [])):
            return 'request_errors'
        if row.get('output_lens') != [w['output_tokens']] * w['requests']:
            return 'output_token_mismatch'
        if row.get('input_lens') != [w['input_tokens']] * w['requests']:
            return 'input_token_mismatch'
        expected = w['requests'] * w['output_tokens']
        if row.get('total_output_tokens') != expected:
            return 'total_output_mismatch'
        if not math.isclose(row['output_throughput'], expected / row['duration'], rel_tol=0.01):
            return 'inconsistent_throughput'
        if row['p95_ttft_ms'] > s['max_p95_ttft_ms'] or row['p95_e2el_ms'] > s['max_p95_e2el_ms']:
            return 'latency_limit'
    return None


def verdict(candidate, reference, outputs, golden, s):
    reason = check_samples(candidate, s) or check_samples(reference, s)
    if reason:
        return {'accepted': False, 'reason': reason}
    if outputs != golden or not golden or not all(x.strip() for x in golden):
        return {'accepted': False, 'reason': 'functional_regression'}
    # Re-measure the incumbent around each candidate; require separation of ranges.
    # Conservative engineering gate, not a statistical significance claim.
    gain = summarize(candidate)['output_throughput'] / summarize(reference)['output_throughput'] - 1
    separated = min(x['output_throughput'] for x in candidate) > max(x['output_throughput'] for x in reference)
    ok = gain >= s['min_gain'] and separated
    return {'accepted': ok, 'reason': 'verified_gain' if ok else 'insufficient_or_noisy_gain', 'gain': gain}
