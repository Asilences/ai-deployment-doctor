"""Version-isolated capacity replay; legacy replay fingerprints remain unchanged."""
import copy
import json
import time
from inference_lab.candidate_ids import candidate_catalog, parse_id
from inference_lab.core import fingerprint
from inference_lab.proposal_replay import input_variant, require, ContextBudgetError
from inference_lab.proposal_trace import tracked_call, trace_totals, utc_now
from planning_service import planner_payload, service_identity


def capacity_proposal(snapshot, representation, history_mode, profile, repo_root, send, preflight):
    """Inject transports for tests; no target backend or verifier is invoked here."""
    started = time.monotonic()
    result = {'version': 'capacity-proposal-v1', 'planner_service': service_identity(profile, repo_root), 'representation': representation,
              'history_mode': history_mode, 'decision_sha256': snapshot['decision_sha256'],
              'started_utc': utc_now(), 'status': 'running', 'proposal': None,
              'candidate_executed': False, 'preflight': []}
    tracker = type('LocalTracker', (), {'kind': 'local'})()
    tracker.last_trace = []
    try:
        brief = input_variant(snapshot, representation, history_mode)
        payload = planner_payload(brief, snapshot['decision']['settings'], profile, repo_root)
        for attempt in range(2):
            # Failure here consumes no generation call; retain separate inspection cost.
            check = {'attempt': attempt + 1, 'started_utc': utc_now()}
            result['preflight'].append(check)
            inspected = time.monotonic()
            try:
                check.update(preflight(copy.deepcopy(payload)))
                require(check['prompt_tokens'] + payload['max_tokens'] <= check['context_limit'],
                        'Prompt plus generation exceeds frozen context capacity')
                check['status'] = 'passed'
            except BaseException as error:
                check.update(status='failed', error_type=type(error).__name__)
                raise
            finally:
                check.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - inspected)
            response = tracked_call(tracker, lambda: send(copy.deepcopy(payload)),
                                    {'planner_service': service_identity(profile, repo_root), 'payload': copy.deepcopy(payload), 'payload_sha256': fingerprint(payload),
                                     'evidence': brief, 'available_candidates': brief['available_candidates']})
            trace = tracker.last_trace[-1]
            reported_prompt = trace['usage']['prompt_tokens']
            trace['prompt_count_matches_preflight'] = (reported_prompt == check['prompt_tokens']
                                                       if reported_prompt is not None else None)
            if trace['prompt_count_matches_preflight'] is False:
                trace.update(parse_result='rejected', failure_category='prompt_token_count_mismatch')
                raise ContextBudgetError('Server prompt count differs from preflight; no candidate executed')
            try:
                result['proposal'] = parse_id(response, candidate_catalog(snapshot['decision']['settings']),
                                              brief['available_candidates'], 96)
                trace.update(parse_result='valid', candidate_id=result['proposal']['candidate_id'])
                result['status'] = 'valid'
                break
            except ValueError as error:
                category = 'duplicate_candidate' if str(error).startswith('Duplicate') else 'invalid_output'
                trace.update(parse_result='rejected', failure_category=category)
                if attempt == 1:
                    raise
                payload['messages'].append({'role': 'user', 'content':
                    'Previous response was rejected: ' + category +
                    '. Select one available candidate ID and return exactly the requested JSON fields. '
                    'Available candidates: ' + json.dumps(brief['available_candidates'])})
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        result.update(status='failed', error_type=type(error).__name__, error_message=str(error))
    except BaseException as error:
        result.update(status='interrupted', error_type=type(error).__name__)
        # Return the ledger first; the orchestrator saves it and stops the batch.
    finally:
        result.update(proposal_calls=tracker.last_trace, costs=trace_totals(tracker.last_trace),
                      ended_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        result['costs']['preflight_seconds'] = sum(c['elapsed_seconds'] for c in result['preflight'])
    return result
