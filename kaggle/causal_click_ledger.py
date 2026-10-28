"""Deduplicated click contexts for an exploratory tied-item factor model.

Inputs must contain fitting events only. This module never reads validation
labels. Times are integer UTC seconds; identical user/item/time triples identify
one click even when repeated in several supplied histories.
"""
from bisect import bisect_left
from collections import defaultdict


def build_ledger(events, cutoff):
    """events: iterable of (user, event_time, [(item,time)], clicked_items).

    Reject post-cutoff labeled events rather than silently consuming their
    histories. A click observed at the cutoff is excluded from the fitting log.
    """
    clicks = defaultdict(set)
    for user, event_time, history, clicked_items in events:
        if event_time >= cutoff:
            raise ValueError('Labeled fitting event must precede cutoff')
        if len(clicked_items) > 1:
            raise ValueError('At most one click per event')
        for item, timestamp in history:
            if timestamp >= event_time:
                raise ValueError('History click must precede its event')
            clicks[user].add((int(timestamp), item))
        for item in clicked_items:
            clicks[user].add((int(event_time), item))
    return {user: tuple(sorted(values)) for user, values in clicks.items()}


def context_before(ledger, user, timestamp, limit=100):
    """All known clicks strictly before an impression, most recent first."""
    if limit < 1:
        raise ValueError('limit must be positive')
    values = ledger.get(user, ())
    stop = bisect_left([value[0] for value in values], timestamp)
    return tuple(reversed(values[max(0, stop-limit):stop]))


def other_click_context(ledger, user, target, fit_cutoff):
    """Training-only leave-one-click-out context, never impression features.

    Later clicks are permitted here only when already inside the fitting prefix.
    Excludes the exact target occurrence, retaining legitimate repeat clicks.
    """
    if target[0] >= fit_cutoff:
        raise ValueError('Auxiliary target must precede fitting cutoff')
    return tuple(value for value in ledger.get(user, ())
                 if value[0] < fit_cutoff and value != target)
