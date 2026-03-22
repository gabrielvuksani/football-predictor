"""Walk-forward validation for football prediction models.

Implements temporal cross-validation strategies that respect the sequential
nature of football data, preventing future information from leaking into
training sets.

Standard time-series CV is insufficient for football because:
1. Features (form, ELO, etc.) are computed from recent matches — an embargo
   period is needed so that features near the split boundary don't encode
   information about test-set outcomes.
2. A single matchday round can span Friday–Monday, so a naive date split
   may place matches from the same round on both sides of the boundary.
   Group purging removes all matches from any day that falls inside the
   embargo window.
3. The training window must expand (not slide) to maximise sample size
   while preserving temporal ordering.

References:
    de Prado (2018) "Advances in Financial Machine Learning", Ch. 7
    Cerqueira et al. (2020) "Evaluating time series forecasting models"
"""
from __future__ import annotations

import numpy as np
import logging

log = logging.getLogger(__name__)


def purged_group_time_series_split(
    dates: np.ndarray,
    n_splits: int = 5,
    embargo_days: int = 3,
    test_size_frac: float = 0.15,
    min_train_size: int = 1000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged Group Time-Series Split for football prediction.

    Prevents data leakage by:
    1. Sorting by date (temporal ordering)
    2. Expanding train window (never look ahead)
    3. Embargo gap between train and test (removes features contamination)
    4. Group purging (removes entire matchday if it spans boundary)

    Args:
        dates: Array of match dates (datetime-like, castable to
            ``numpy.datetime64``).
        n_splits: Number of CV folds.
        embargo_days: Days to embargo between train and test.  Any training
            sample whose date falls within this many days *before* the first
            test date is removed, along with every other sample that shares
            the same calendar day (group purging).
        test_size_frac: Fraction of the total dataset used for each test
            fold.
        min_train_size: Minimum number of training samples required.  Folds
            that would have fewer training samples after purging are
            silently skipped.

    Returns:
        List of ``(train_indices, test_indices)`` tuples where the indices
        refer to positions in the *original* (unsorted) ``dates`` array.

    Raises:
        ValueError: If the input array is empty, ``n_splits < 1``, or
            parameters are otherwise invalid.

    Example::

        from footy.models.walkforward import purged_group_time_series_split
        import numpy as np

        dates = np.array(["2023-01-01", "2023-01-02", ...], dtype="datetime64")
        folds = purged_group_time_series_split(dates, n_splits=5, embargo_days=3)
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            ...
    """
    # ── Validate inputs ────────────────────────────────────────────────
    if len(dates) == 0:
        raise ValueError("dates array must not be empty")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < test_size_frac < 1.0:
        raise ValueError(
            f"test_size_frac must be in (0, 1), got {test_size_frac}"
        )
    if embargo_days < 0:
        raise ValueError(
            f"embargo_days must be >= 0, got {embargo_days}"
        )

    # ── Cast to datetime64 and sort ────────────────────────────────────
    dt_dates = np.asarray(dates, dtype="datetime64[D]")
    sorted_order = np.argsort(dt_dates, kind="mergesort")  # stable sort
    sorted_dates = dt_dates[sorted_order]

    n_samples = len(sorted_dates)
    test_size = max(1, int(n_samples * test_size_frac))
    embargo_td = np.timedelta64(embargo_days, "D")

    # ── Generate folds ─────────────────────────────────────────────────
    # Test blocks are placed at evenly-spaced positions in the latter
    # portion of the data so that the training window expands with each
    # fold.  The earliest possible test-start leaves room for at least
    # ``min_train_size`` training samples (before purging).
    earliest_test_start = max(min_train_size, test_size)
    available = n_samples - earliest_test_start
    if available < test_size:
        log.warning(
            "Dataset too small for requested split parameters "
            "(n=%d, min_train=%d, test_size=%d).  Returning empty folds.",
            n_samples,
            min_train_size,
            test_size,
        )
        return []

    # Space test-start positions evenly across the available range.
    if n_splits == 1:
        test_starts = [n_samples - test_size]
    else:
        step = max(1, (n_samples - earliest_test_start - test_size) // (n_splits - 1))
        test_starts = [
            earliest_test_start + i * step
            for i in range(n_splits)
        ]
        # Ensure last fold uses the very end of the dataset.
        test_starts[-1] = min(test_starts[-1], n_samples - test_size)

    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx, t_start in enumerate(test_starts):
        t_end = min(t_start + test_size, n_samples)
        test_indices_sorted = np.arange(t_start, t_end)

        # ── Determine embargo boundary ─────────────────────────────────
        test_start_date = sorted_dates[t_start]
        embargo_boundary = test_start_date - embargo_td

        # ── Build candidate training set (everything before test) ──────
        candidate_train = np.arange(0, t_start)
        if len(candidate_train) == 0:
            continue

        candidate_dates = sorted_dates[candidate_train]

        # ── Identify samples inside the embargo window ─────────────────
        in_embargo = candidate_dates > embargo_boundary  # strictly after

        # ── Group purging: find calendar days that overlap the embargo ──
        # Any day that has at least one match inside the embargo window
        # is entirely removed from the training set.
        if np.any(in_embargo):
            embargo_sample_dates = candidate_dates[in_embargo]
            purged_days = np.unique(embargo_sample_dates)

            # Build a boolean mask: True = keep this training sample
            keep_mask = np.ones(len(candidate_train), dtype=bool)
            for day in purged_days:
                keep_mask &= candidate_dates != day

            # Also remove any remaining samples that fall exactly on a
            # day shared with the embargo zone (catches edge cases where
            # a day straddles the embargo boundary).
            train_indices_sorted = candidate_train[keep_mask]
        else:
            train_indices_sorted = candidate_train

        # ── Check minimum training size ────────────────────────────────
        if len(train_indices_sorted) < min_train_size:
            log.debug(
                "Fold %d skipped: only %d training samples after purging "
                "(need %d).",
                fold_idx,
                len(train_indices_sorted),
                min_train_size,
            )
            continue

        # ── Map back to original (unsorted) indices ────────────────────
        train_original = sorted_order[train_indices_sorted]
        test_original = sorted_order[test_indices_sorted]

        folds.append((train_original, test_original))

        log.info(
            "Fold %d: train=%d [%s → %s], test=%d [%s → %s], "
            "purged=%d embargo samples",
            fold_idx,
            len(train_original),
            sorted_dates[train_indices_sorted[0]],
            sorted_dates[train_indices_sorted[-1]],
            len(test_original),
            sorted_dates[test_indices_sorted[0]],
            sorted_dates[test_indices_sorted[-1]],
            len(candidate_train) - len(train_indices_sorted),
        )

    if not folds:
        log.warning(
            "No valid folds produced.  Consider lowering min_train_size "
            "(%d) or embargo_days (%d).",
            min_train_size,
            embargo_days,
        )

    return folds
