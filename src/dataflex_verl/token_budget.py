"""Pure helper for the matched-update-token stopping rule (revision_plan_3 P1.2).

Import-light (no verl) so it can be unit tested directly; ``trainer.py`` calls
``round_up_to_boundary`` once the cumulative nonzero-update-token budget is crossed.
"""


def round_up_to_boundary(global_step: int, save_freq: int, ceiling: int) -> int:
    """Smallest checkpoint-aligned step >= ``global_step``, capped at ``ceiling``.

    verl's training loop computes ``is_last_step = global_steps >= total_training_steps``
    BEFORE running the step body, so shrinking ``total_training_steps`` mid-step to the
    exact crossing step would miss both the ``is_last_step`` checkpoint trigger and (if the
    crossing step isn't a ``save_freq`` multiple) the periodic one — silently losing the
    final checkpoint. Instead we round the new ``total_training_steps`` UP to the next
    ``save_freq`` boundary at or after the crossing step, so the existing periodic
    checkpoint logic saves it with no special-casing, at the cost of a small, deliberate
    overshoot of the token budget (never a shortfall).
    """
    save_freq = max(1, int(save_freq))
    global_step = int(global_step)
    if global_step % save_freq == 0:
        stop_at = global_step
    else:
        stop_at = ((global_step // save_freq) + 1) * save_freq
    return min(stop_at, int(ceiling))
