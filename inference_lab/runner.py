import html
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import check_samples, dump, fingerprint, summarize, validate_candidate, verdict
from .candidate_ids import SearchSpaceExhausted
from .proposal_trace import trace_totals


def provenance(source_dir, backend_name):
    code = {p.name: fingerprint(p.read_text(encoding='utf-8')) for p in sorted(source_dir.glob('*.py'))}
    commit = None
    dirty = None
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source_dir.parent,
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            commit = result.stdout.strip()
            result = subprocess.run(['git', 'status', '--porcelain'], cwd=source_dir.parent,
                                    capture_output=True, text=True, timeout=5)
            dirty = bool(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    hardware = {}
    if backend_name != 'mock':
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                     '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                hardware['nvidia_smi'] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {'git_commit': commit, 'git_dirty': dirty, 'code_sources': code, 'hardware': hardware}


def render_report(directory, run):
    rows = []
    for t in run['trials']:
        cells = [t['id'], t.get('proposal', {}).get('hypothesis', ''),
                 t.get('proposal', {}).get('config', {}), t.get('verdict', {}), t.get('rollback', '')]
        rows.append('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in cells) + '</tr>')
    title = '模拟闭环测试（无真实推理、无 L2 达成证据）' if run['backend'] == 'mock' else '真实推理实验记录'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Inference Lab</title>
<style>body{font:16px/1.7 system-ui;margin:40px auto;max-width:1100px;background:#f6f7fa;color:#192b40}
table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6dce5;padding:12px;text-align:left}
pre{white-space:pre-wrap;background:white;padding:20px}h1{font-size:27px}</style>'''
    page += '<h1>' + title + '</h1><p>目标：固定模型与请求要求，在延迟约束下提高输出吞吐。</p>'
    page += '<p>规划器：' + html.escape(run['planner']) + ' · 状态：' + html.escape(run['status']) + '</p>'
    page += '<table><tr><th>实验</th><th>假设</th><th>候选配置</th><th>独立验收</th><th>恢复检查</th></tr>' + ''.join(rows) + '</table>'
    page += '<h2>最终保留状态</h2><pre>' + html.escape(json.dumps(run.get('state'), ensure_ascii=False, indent=2)) + '</pre>'
    page += '<p>原始记录：run.json；每轮 benchmark.json、服务日志、命令和输出保存在相邻目录。</p></html>'
    (directory / 'report.html').write_text(page, encoding='utf-8')


def execute(s, backend, planner, directory, iterations, initial=None, max_seconds=None):
    started = time.monotonic()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('Time budget must be positive')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source_dir = Path(__file__).parent
    source_names = ['core.py', 'backend.py', 'runner.py']
    if backend.name == 'llama_cpp':
        source_names.append('llama_backend.py')
    source_hashes = {name: fingerprint((source_dir / name).read_text(encoding='utf-8'))
                     for name in source_names}
    identity = fingerprint({'settings': s, 'backend': backend.name, 'evaluator_sources': source_hashes})
    if initial and initial['identity'] != identity:
        raise ValueError('Cannot inherit state from another model, backend, workload or evaluation protocol')
    config = validate_candidate(initial['config'] if initial else s['baseline'], s)
    run = {'backend': backend.name, 'planner': planner.kind, 'status': 'running',
           'started_utc': datetime.now(timezone.utc).isoformat(), 'settings': s,
           'platform': platform.platform(), 'evaluator_sources': source_hashes, 'trials': [], 'state': None}
    run.update(provenance(source_dir, backend.name))
    run.update(record_version='proposal-ledger-v1', protocol_version='paired-ab-v1',
               verifier_version='bounded-gate-v1', planner_version=getattr(planner, 'version', 'unknown'),
               planner_sources={k: v for k, v in run['code_sources'].items()
                                if k in ('planner.py', 'candidate_ids.py', 'proposal_trace.py', 'context_ablation.py')},
               seed=getattr(planner, 'seed', None), proposal_slot_limit=iterations,
               context_mode=getattr(planner, 'context_mode', None),
               max_seconds=max_seconds, candidate_catalog=getattr(planner, 'catalog', None))
    run['environment_fingerprint'] = fingerprint({'settings': s, 'platform': run['platform'],
                                                  'hardware': run['hardware'], 'backend': backend.name})
    planner.environment = run['hardware']
    run['planner_service_config'] = (validate_candidate({'parallel': 1, 'ubatch_size': 128}, s)
                                     if getattr(planner, 'version', None) in ('v3', 'v3.1') else None)

    def save():
        calls = [c for trial in run['trials'] for c in trial.get('proposal_calls', [])]
        run['costs'] = {**trace_totals(calls),
                        'proposal_slots': len(run['trials']),
                        'experiments_consumed': sum(t.get('candidate_executed', False) for t in run['trials']),
                        'rollback_count': sum(t.get('rollback', '').startswith('incumbent_restart') for t in run['trials']),
                        'planner_setup_seconds': sum(t.get('planner_setup_seconds', 0) for t in run['trials']),
                        'wall_clock_seconds': time.monotonic() - started}
        dump(directory / 'run.json', run)
        render_report(directory, run)

    def remember(trial):
        history.append(trial.copy())
        state['history'] = list(history)
        run['state'] = state
        dump(directory / 'best.json', state)
        save()

    def evaluate(candidate, label):
        target = directory / label
        backend.start(candidate, target)
        outputs = backend.outputs()
        dump(target / 'functional_outputs.json', outputs)
        if hasattr(backend, 'warmup'):
            backend.warmup(target)
        return backend.measure(target), outputs

    try:
        print('Measuring initial configuration...', flush=True)
        baseline, golden = [], None
        for n in range(s['repeats']):
            sample, outputs = evaluate(config, f'baseline-{n}')
            baseline.append(sample)
            if golden is not None and golden != outputs:
                raise RuntimeError('Initial functional outputs are unstable across restarts')
            golden = outputs
        reason = check_samples(baseline, s)
        if reason:
            raise RuntimeError('Initial configuration does not pass the frozen protocol: ' + reason)
        if initial and golden != initial['golden']:
            raise RuntimeError('Inherited configuration failed the persisted functional probes')
        state = {'identity': identity, 'config': config, 'golden': golden, 'metrics': summarize(baseline),
                 'generation': initial['generation'] if initial else 0,
                 'history': list(initial.get('history', [])) if initial else []}
        run['state'] = state
        dump(directory / 'best.json', state)
        save()
        history = list(initial.get('history', [])) if initial else []
        for i in range(iterations):
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                run['stop_reason'] = 'time_budget_reached'
                break
            trial = {'id': i + 1, 'candidate_executed': False, 'proposal_calls': []}
            run['trials'].append(trial)
            save()
            print(f'Experiment {i + 1}/{iterations}: proposing candidate...', flush=True)
            planner.last_trace = []
            try:
                if run['planner_service_config'] is not None:
                    setup_started = time.monotonic()
                    try:
                        backend.start(run['planner_service_config'], directory / f'proposal-{i + 1}/service')
                    finally:
                        trial['planner_setup_seconds'] = time.monotonic() - setup_started
                p = planner.propose(state, history)
                trial['proposal'] = p
                candidate = validate_candidate(p['config'], s)
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                exhausted = isinstance(error, SearchSpaceExhausted)
                trial['verdict'] = {'accepted': False, 'reason': 'search_space_exhausted' if exhausted else 'proposal_failed',
                                    'error_type': type(error).__name__}
                trial['proposal_calls'] = list(getattr(planner, 'last_trace', []))
                trial['available_candidates'] = list(getattr(planner, 'last_available', []))
                remember(trial)
                if exhausted:
                    run['stop_reason'] = 'search_space_exhausted'
                    break
                continue
            trial['proposal_calls'] = list(getattr(planner, 'last_trace', []))
            trial['available_candidates'] = list(getattr(planner, 'last_available', []))
            if candidate == state['config']:
                trial['verdict'] = {'accepted': False, 'reason': 'unchanged_candidate'}
                remember(trial)
                continue
            trial['candidate_executed'] = True
            save()
            reference, measurements, functional = [], [], True
            try:
                # Alternating A/B order reduces drift; each evaluation uses a fresh server.
                for n in range(s['repeats']):
                    order = [('reference', state['config']), ('candidate', candidate)]
                    if n % 2:
                        order.reverse()
                    for role, cfg in order:
                        sample, outputs = evaluate(cfg, f'trial-{i + 1}/{n}-{role}')
                        (reference if role == 'reference' else measurements).append(sample)
                        functional = functional and outputs == golden
                decision = verdict(measurements, reference, golden if functional else [], golden, s)
                if not check_samples(reference, s):
                    trial['reference'] = summarize(reference)
                if not check_samples(measurements, s):
                    trial['candidate'] = summarize(measurements)
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                decision = {'accepted': False, 'reason': 'execution_failed', 'error_type': type(error).__name__}
            trial['verdict'] = decision
            next_config = candidate if decision['accepted'] else state['config']
            # Promotion and rollback both require a fresh restart plus a full benchmark.
            try:
                sample, outputs = evaluate(next_config, f'trial-{i + 1}/restore-check')
                reason = check_samples([sample], {**s, 'repeats': 1})
                if outputs != golden or reason:
                    raise RuntimeError('Restored service failed verification')
                if decision['accepted'] and sample['output_throughput'] < summarize(reference)['output_throughput'] * (1 + s['min_gain']):
                    raise RuntimeError('Promoted service did not reproduce the required gain')
                trial['rollback'] = 'promotion_restart_verified' if decision['accepted'] else 'incumbent_restart_verified'
            except (RuntimeError, OSError, ValueError, KeyError):
                if not decision['accepted']:
                    trial['rollback'] = 'failed; run halted'
                    raise RuntimeError('Could not verify the incumbent after restart') from None
                decision = {'accepted': False, 'reason': 'promotion_restart_failed'}
                trial['verdict'] = decision
                trial['rollback'] = 'failed; run halted'
                sample, outputs = evaluate(state['config'], f'trial-{i + 1}/fallback-check')
                if outputs != golden or check_samples([sample], {**s, 'repeats': 1}):
                    raise RuntimeError('Could not recover incumbent after failed promotion')
                trial['rollback'] = 'incumbent_restart_verified_after_failed_promotion'
            if decision['accepted']:
                state.update(config=candidate, generation=state['generation'] + 1,
                             metrics=summarize(measurements))
            remember(trial)
            print(f"  {decision['reason']}; {trial['rollback']}", flush=True)
        run['status'] = 'completed'
        return run
    except BaseException as error:
        run['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        run['error_type'] = type(error).__name__
        if run['trials'] and 'verdict' not in run['trials'][-1]:
            run['trials'][-1]['proposal_calls'] = list(getattr(planner, 'last_trace', []))
        raise
    finally:
        try:
            backend.stop()
        finally:
            run['ended_utc'] = datetime.now(timezone.utc).isoformat()
            save()
