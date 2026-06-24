"""Greedy subset selection for the three-objective integral R2 indicator.

This module builds on ``r2_exact3d.py``.  For positive loss vectors
p=(p1,p2,p3), it greedily selects k points that minimize

    R2(S) = int_{Delta_2} min_{p in S} max_i w_i p_i dw.

Equivalently, with a fixed dominated anchor ``pbar`` and
R0 = R2({pbar}), it maximizes the normalized gain

    Q_pbar(S) = R0 - R2(S).

The implementation is intentionally transparent and small.  It is intended as
reference code for the greedy theorem in the report, not as a high-performance
large-scale implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple
import math
import time

import numpy as np

from r2_exact3d import EvaluationResult, exact_r2_3d, monte_carlo_r2_3d, random_nondominated_points


@dataclass
class GreedyStep:
    """One step of greedy R2 subset selection."""

    step: int
    chosen_index: int
    r2_value: float
    marginal_gain: float
    gain: float
    evaluated_candidates: int


@dataclass
class GreedyResult:
    """Result of greedy R2 subset selection."""

    selected_indices: List[int]
    selected_points: np.ndarray
    r0: float
    r2_value: float
    gain: float
    steps: List[GreedyStep]
    anchor: np.ndarray


@dataclass
class BruteForceResult:
    """Exact cardinality-k optimum for small instances."""

    selected_indices: Tuple[int, ...]
    r2_value: float
    gain: float
    combinations_evaluated: int


def _validate_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an array of shape (n, 3)")
    if len(pts) == 0:
        raise ValueError("at least one point is required")
    if not np.all(np.isfinite(pts)):
        raise ValueError("all coordinates must be finite")
    if np.any(pts <= 0):
        raise ValueError("all coordinates must be positive")
    return pts


def default_dominated_anchor(points: np.ndarray, scale: float = 1.05, offset: float = 1e-9) -> np.ndarray:
    """Return a coordinatewise dominated anchor for minimization-style losses.

    Since all candidate losses are positive and smaller is better, a point with
    coordinates at least as large as every candidate coordinate satisfies
    g_anchor(w) >= g_p(w) for every simplex weight vector w and every candidate
    p.  This is enough for the gap-normalized greedy guarantee.
    """
    pts = _validate_points(points)
    if scale < 1.0:
        raise ValueError("scale must be at least 1")
    return scale * pts.max(axis=0) + offset


def _subset_key(indices: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(i) for i in indices))


def make_exact_oracle(points: np.ndarray):
    """Create a cached exact R2 oracle for subsets of candidate indices."""
    pts = _validate_points(points)
    cache: Dict[Tuple[int, ...], float] = {}

    def value(indices: Iterable[int]) -> float:
        key = _subset_key(indices)
        if not key:
            raise ValueError("R2 of the empty set is not finite; use an anchor-normalized gain instead")
        if key not in cache:
            cache[key] = exact_r2_3d(pts[list(key)]).value
        return cache[key]

    return value, cache


def greedy_r2_subset(points: np.ndarray, k: int, anchor: np.ndarray | None = None) -> GreedyResult:
    """Greedily select k points by exact R2 marginal improvement.

    The first point is the one with the smallest singleton R2 value; subsequent
    points maximize the exact decrease R2(S)-R2(S union {p}).  The same choices
    maximize Q_anchor(S)=R0-R2(S) for any fixed dominated anchor.
    """
    pts = _validate_points(points)
    n = len(pts)
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= number of candidate points")
    anchor_vec = default_dominated_anchor(pts) if anchor is None else np.asarray(anchor, dtype=float)
    if anchor_vec.shape != (3,):
        raise ValueError("anchor must be a 3-vector")
    if np.any(anchor_vec <= 0):
        raise ValueError("anchor must be positive")
    r0 = exact_r2_3d(anchor_vec.reshape(1, 3)).value

    r2_value, cache = make_exact_oracle(pts)
    selected: List[int] = []
    remaining = set(range(n))
    steps: List[GreedyStep] = []
    current_r2: float | None = None
    current_gain = 0.0

    for step in range(1, k + 1):
        best_idx = None
        best_r2 = None
        best_marginal = -math.inf
        evaluated = 0
        for idx in sorted(remaining):
            trial = selected + [idx]
            trial_r2 = r2_value(trial)
            if current_r2 is None:
                marginal = r0 - trial_r2
            else:
                marginal = current_r2 - trial_r2
            evaluated += 1
            # tie-break by smaller resulting R2, then smaller index for reproducibility
            better = marginal > best_marginal + 1e-15
            if not better and abs(marginal - best_marginal) <= 1e-15:
                if best_r2 is None or trial_r2 < best_r2 - 1e-15 or (abs(trial_r2 - best_r2) <= 1e-15 and idx < int(best_idx)):
                    better = True
            if better:
                best_idx = idx
                best_r2 = trial_r2
                best_marginal = marginal
        assert best_idx is not None and best_r2 is not None
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
        current_r2 = float(best_r2)
        current_gain = r0 - current_r2
        steps.append(GreedyStep(step, int(best_idx), current_r2, float(best_marginal), current_gain, evaluated))

    return GreedyResult(selected, pts[selected], float(r0), float(current_r2), float(current_gain), steps, anchor_vec)


def brute_force_optimum(points: np.ndarray, k: int, anchor: np.ndarray | None = None) -> BruteForceResult:
    """Compute the exact cardinality-k optimum by enumeration.

    This is only for validation on small instances.
    """
    pts = _validate_points(points)
    n = len(pts)
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= number of candidate points")
    anchor_vec = default_dominated_anchor(pts) if anchor is None else np.asarray(anchor, dtype=float)
    r0 = exact_r2_3d(anchor_vec.reshape(1, 3)).value
    best_combo: Tuple[int, ...] | None = None
    best_r2 = math.inf
    count = 0
    for combo in combinations(range(n), k):
        count += 1
        val = exact_r2_3d(pts[list(combo)]).value
        if val < best_r2 - 1e-15:
            best_r2 = float(val)
            best_combo = tuple(combo)
    assert best_combo is not None
    return BruteForceResult(best_combo, best_r2, float(r0 - best_r2), count)


def greedy_ratio_to_optimal_gain(greedy: GreedyResult, optimum: BruteForceResult) -> float:
    """Return Q(G_k)/Q(S*_k), with inf/nan handling."""
    if abs(optimum.gain) < 1e-15:
        return float("nan")
    return greedy.gain / optimum.gain


if __name__ == "__main__":
    pts = np.array([
        [0.30, 1.20, 1.00],
        [0.65, 0.70, 0.85],
        [1.10, 0.40, 0.55],
        [0.95, 0.95, 0.25],
    ])
    k = 2
    greedy = greedy_r2_subset(pts, k)
    optimum = brute_force_optimum(pts, k)
    ratio = greedy_ratio_to_optimal_gain(greedy, optimum)
    print(f"selected indices: {greedy.selected_indices}")
    print(f"greedy R2       : {greedy.r2_value:.10f}")
    print(f"optimal indices : {list(optimum.selected_indices)}")
    print(f"optimal R2      : {optimum.r2_value:.10f}")
    print(f"gain ratio      : {ratio:.6f}  (guarantee threshold {1-1/math.e:.6f})")
