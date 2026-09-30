"""Remove explicit evidence, not the controller's common feasibility constraints."""
MODES = ('full', 'no_environment', 'no_workload', 'no_context', 'no_feedback', 'no_history')


def filter_context(brief, mode):
    if mode not in MODES:
        raise ValueError('Unknown context mode')
    result = dict(brief)
    if mode in ('no_environment', 'no_context'):
        result.pop('environment', None)
        result.pop('hardware', None)
    if mode in ('no_workload', 'no_context'):
        result.pop('workload', None)
    if mode == 'no_feedback':
        result.pop('current_metrics', None)
        result.pop('prior_trials', None)
    if mode == 'no_history':
        # Match the empty-history first-slot interface; keep live incumbent metrics.
        result['prior_trials'] = []
    return result
