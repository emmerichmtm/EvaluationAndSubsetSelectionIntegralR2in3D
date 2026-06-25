r"""Verification experiment for the perspective mapping used in the 3D integral R2 report.

The report uses the change of variables

    x_i = w_i / t,  i=1,2,3,
    t = 1/(x_1+x_2+x_3),
    w_i = x_i/(x_1+x_2+x_3),

with w in the simplex Delta_2 and t > 0.  For a positive objective
vector p, the inequality t <= max_i w_i p_i is equivalent to
x not in [0,1/p_1] x [0,1/p_2] x [0,1/p_3].

This script verifies three consequences numerically:
  1. pointwise membership equivalence for random (w,t),
  2. the Jacobian determinant |d x / d(w_1,w_2,t)| = t^{-4},
  3. equality between scalarization-space improvement and weighted
     reciprocal-box gain:

        R2(B) - R2(B union S)
        = int_{U(b(B union S)) \ U(b(B))} (x_1+x_2+x_3)^(-4) dx.

The scalarization-side value is computed with the arrangement evaluator
r2_exact3d.py.  The reciprocal-space integral is estimated by Monte Carlo
sampling in a bounding box.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from r2_exact3d import exact_r2_3d, monte_carlo_r2_3d


def reciprocal_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if np.any(points <= 0):
        raise ValueError("all objective coordinates must be positive")
    return 1.0 / points


def in_union_of_anchored_boxes(x: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Vectorized membership in union_a [0,a_1] x [0,a_2] x [0,a_3]."""
    x = np.asarray(x, dtype=float)
    corners = np.asarray(corners, dtype=float)
    # shape: samples x boxes x coordinates
    inside_each = np.all(x[:, None, :] <= corners[None, :, :] + 1e-15, axis=2)
    return np.any(inside_each, axis=1)


def tau(points: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Tchebycheff lower envelope tau_S(w)=min_p max_i w_i p_i."""
    points = np.asarray(points, dtype=float)
    w = np.asarray(w, dtype=float)
    vals = np.max(w[:, None, :] * points[None, :, :], axis=2)
    return np.min(vals, axis=1)


def sample_simplex(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform samples from Delta_2 by reflection."""
    u = rng.random((n, 2))
    mask = u.sum(axis=1) > 1.0
    u[mask] = 1.0 - u[mask]
    return np.column_stack([u[:, 0], u[:, 1], 1.0 - u[:, 0] - u[:, 1]])


def pointwise_membership_test(
    points: np.ndarray,
    samples: int,
    seed: int,
    t_max: float,
) -> int:
    """Count failures of t <= tau_S(w) iff x not in U(b(S))."""
    rng = np.random.default_rng(seed)
    w = sample_simplex(rng, samples)
    # Avoid exactly t=0 for the perspective map.
    t = rng.uniform(1e-9, t_max, size=samples)
    x = w / t[:, None]
    left = t <= tau(points, w) + 1e-12
    right = ~in_union_of_anchored_boxes(x, reciprocal_corners(points))
    return int(np.count_nonzero(left != right))


def jacobian_test(samples: int, seed: int) -> float:
    """Return max relative error in analytic Jacobian determinant check."""
    rng = np.random.default_rng(seed)
    w = sample_simplex(rng, samples)
    # Keep t in a numerically comfortable range.
    t = rng.uniform(0.05, 2.0, size=samples)
    max_rel = 0.0
    for (w1, w2, _w3), tt in zip(w, t):
        J = np.array([
            [1.0 / tt, 0.0, -w1 / (tt * tt)],
            [0.0, 1.0 / tt, -w2 / (tt * tt)],
            [-1.0 / tt, -1.0 / tt, -(1.0 - w1 - w2) / (tt * tt)],
        ])
        det_abs = abs(float(np.linalg.det(J)))
        expected = tt ** -4
        rel = abs(det_abs - expected) / expected
        max_rel = max(max_rel, rel)
    return max_rel


def weighted_box_mc(
    baseline: np.ndarray,
    candidates: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    r"""Monte Carlo estimate of weighted measure of U(B union S) \ U(B)."""
    all_points = np.vstack([baseline, candidates])
    b_base = reciprocal_corners(baseline)
    b_all = reciprocal_corners(all_points)
    upper = np.max(b_all, axis=0)
    volume = float(np.prod(upper))

    rng = np.random.default_rng(seed)
    x = rng.random((samples, 3)) * upper[None, :]
    in_new = in_union_of_anchored_boxes(x, b_all)
    in_old = in_union_of_anchored_boxes(x, b_base)
    diff = in_new & (~in_old)
    denom = np.sum(x, axis=1)
    y = np.zeros(samples, dtype=float)
    y[diff] = denom[diff] ** -4
    estimate = volume * float(np.mean(y))
    stderr = volume * float(np.std(y, ddof=1) / math.sqrt(samples))
    return estimate, stderr


def run_experiment(
    scalar_mc_samples: int = 300_000,
    box_mc_samples: int = 800_000,
    pointwise_samples: int = 200_000,
    jacobian_samples: int = 10_000,
    seed: int = 20260625,
) -> dict[str, float | int]:
    # A small positive 3-objective minimization instance.  The candidates improve
    # different parts of the baseline envelope, so the improvement region is nontrivial.
    baseline = np.array([
        [1.20, 1.30, 1.10],
        [1.05, 1.75, 1.45],
    ], dtype=float)
    candidates = np.array([
        [0.78, 1.55, 1.35],
        [1.42, 0.82, 1.22],
        [1.30, 1.25, 0.74],
    ], dtype=float)
    all_points = np.vstack([baseline, candidates])

    exact_base = exact_r2_3d(baseline)
    exact_all = exact_r2_3d(all_points)
    exact_improvement = exact_base.value - exact_all.value

    scalar_base_mc, scalar_base_se = monte_carlo_r2_3d(baseline, scalar_mc_samples, seed + 1)
    scalar_all_mc, scalar_all_se = monte_carlo_r2_3d(all_points, scalar_mc_samples, seed + 2)
    scalar_mc_improvement = scalar_base_mc - scalar_all_mc
    scalar_mc_se = math.sqrt(scalar_base_se ** 2 + scalar_all_se ** 2)

    box_mc_est, box_mc_se = weighted_box_mc(baseline, candidates, box_mc_samples, seed + 3)

    t_max = float(np.max(all_points))
    fail_base = pointwise_membership_test(baseline, pointwise_samples, seed + 4, t_max)
    fail_all = pointwise_membership_test(all_points, pointwise_samples, seed + 5, t_max)
    jac_rel = jacobian_test(jacobian_samples, seed + 6)

    return {
        "baseline_size": len(baseline),
        "candidate_size": len(candidates),
        "total_size": len(all_points),
        "exact_r2_baseline": exact_base.value,
        "exact_r2_union": exact_all.value,
        "exact_improvement": exact_improvement,
        "scalar_mc_improvement": scalar_mc_improvement,
        "scalar_mc_se": scalar_mc_se,
        "scalar_mc_z": abs(scalar_mc_improvement - exact_improvement) / scalar_mc_se,
        "box_mc_improvement": box_mc_est,
        "box_mc_se": box_mc_se,
        "box_mc_z": abs(box_mc_est - exact_improvement) / box_mc_se,
        "pointwise_failures_baseline": fail_base,
        "pointwise_failures_union": fail_all,
        "pointwise_samples_each": pointwise_samples,
        "jacobian_max_relative_error": jac_rel,
        "jacobian_samples": jacobian_samples,
        "arrangement_cells_baseline": exact_base.num_cells,
        "arrangement_cells_union": exact_all.num_cells,
        "arrangement_lines_baseline": exact_base.num_lines,
        "arrangement_lines_union": exact_all.num_lines,
        "scalar_mc_samples": scalar_mc_samples,
        "box_mc_samples": box_mc_samples,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    result = run_experiment()

    csv_path = out_dir / "perspective_mapping_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print("Perspective mapping verification")
    print("--------------------------------")
    print(f"exact improvement       : {result['exact_improvement']:.12f}")
    print(f"scalarization MC        : {result['scalar_mc_improvement']:.12f} +/- {1.96*result['scalar_mc_se']:.12f} (95% CI)")
    print(f"weighted reciprocal MC  : {result['box_mc_improvement']:.12f} +/- {1.96*result['box_mc_se']:.12f} (95% CI)")
    print(f"pointwise failures B    : {result['pointwise_failures_baseline']} / {result['pointwise_samples_each']}")
    print(f"pointwise failures B+S  : {result['pointwise_failures_union']} / {result['pointwise_samples_each']}")
    print(f"max Jacobian rel. error : {result['jacobian_max_relative_error']:.3e}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
