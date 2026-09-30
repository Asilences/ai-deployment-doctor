"""Proposal telemetry; missing usage is unknown, never silently zero."""
import math
import time
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_usage(response):
    usage = response.get('usage') if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return {key: usage[key] if type(usage.get(key)) in (int, float)
            and math.isfinite(usage[key]) and usage[key] >= 0 else None
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost')}


def tracked_call(planner, send, request_metadata=None):
    row = {'call_index': len(planner.last_trace) + 1, 'started_utc': utc_now(),
           'request': request_metadata, 'usage': clean_usage({}),
           'parse_result': 'not_parsed', 'failure_category': None}
    planner.last_trace.append(row)
    started = time.monotonic()
    try:
        response = send()
        row['usage'] = clean_usage(response)
        row['transport_status'] = 'received'
        if planner.kind == 'local':
            # Local generation only; remote bodies/headers/credentials are never retained.
            try:
                row['response_content'] = response['choices'][0]['message']['content']
            except (KeyError, TypeError, IndexError):
                pass
        return response
    except BaseException as error:
        row['transport_status'] = 'failed'
        row['failure_category'] = 'timeout' if isinstance(error, TimeoutError) else 'transport_or_response_error'
        row['error_type'] = type(error).__name__
        raise
    finally:
        row['ended_utc'] = utc_now()
        row['elapsed_seconds'] = time.monotonic() - started


def trace_totals(calls):
    totals = {'llm_calls': len(calls), 'retry_calls': sum(c['call_index'] > 1 for c in calls),
              'usage_missing_calls': sum(c['usage']['total_tokens'] is None for c in calls),
              'known_total_tokens': sum(c['usage']['total_tokens'] or 0 for c in calls),
              'proposal_seconds': sum(c['elapsed_seconds'] for c in calls),
              'invalid_calls': sum(c['failure_category'] == 'invalid_output' for c in calls),
              'duplicate_calls': sum(c['failure_category'] == 'duplicate_candidate' for c in calls)}
    totals['total_tokens'] = None if totals['usage_missing_calls'] else totals['known_total_tokens']
    for key in ('prompt_tokens', 'completion_tokens', 'cost'):
        known = [c['usage'][key] for c in calls if c['usage'][key] is not None]
        totals['known_' + key] = sum(known)
        totals[key] = sum(known) if len(known) == len(calls) else None
    return totals
