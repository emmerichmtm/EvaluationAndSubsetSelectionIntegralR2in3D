"""Validation script for greedy_r2_3d.py.

The tests compare exact greedy selection against exact brute-force optima on
small instances and also check the selected set with Monte Carlo integration.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

from greedy_r2_3d import brute_force_optimum, greedy_r2_subset, greedy_ratio_to_optimal_gain
from r2_exact3d import monte_carlo_r2_3d, random_nondominated_points


def run() -> None:
    out = Path(__file__).resolve().parent / "greedy_experiment_results.csv"
    cases = [
        ("hand4_k2", 2, np.array([
            [0.30, 1.20, 1.00],
            [0.65, 0.70, 0.85],
            [1.10, 0.40, 0.55],
            [0.95, 0.95, 0.25],
        ])),
        ("hand4_k3", 3, np.array([
            [0.30, 1.20, 1.00],
            [0.65, 0.70, 0.85],
            [1.10, 0.40, 0.55],
            [0.95, 0.95, 0.25],
        ])),
        ("random4_k2", 2, random_nondominated_points(4, seed=31)),
    ]
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", "n", "k", "greedy_indices", "optimal_indices", "r0",
            "greedy_r2", "optimal_r2", "greedy_gain", "optimal_gain",
            "gain_ratio", "guarantee", "bruteforce_combinations", "greedy_seconds",
            "optimal_seconds", "mc_greedy", "mc_stderr", "mc_abs_error"
        ])
        for name, k, pts in cases:
            t0 = time.perf_counter()
            greedy = greedy_r2_subset(pts, k)
            tg = time.perf_counter() - t0
            t0 = time.perf_counter()
            optimum = brute_force_optimum(pts, k, anchor=greedy.anchor)
            to = time.perf_counter() - t0
            ratio = greedy_ratio_to_optimal_gain(greedy, optimum)
            mc, se = monte_carlo_r2_3d(greedy.selected_points, samples=200_000, seed=54321)
            mc_abs_error = abs(mc - greedy.r2_value)
            writer.writerow([
                name, len(pts), k, " ".join(map(str, greedy.selected_indices)),
                " ".join(map(str, optimum.selected_indices)), f"{greedy.r0:.12g}",
                f"{greedy.r2_value:.12g}", f"{optimum.r2_value:.12g}",
                f"{greedy.gain:.12g}", f"{optimum.gain:.12g}",
                f"{ratio:.6g}", f"{1-1/math.e:.6g}", optimum.combinations_evaluated,
                f"{tg:.4f}", f"{to:.4f}", f"{mc:.12g}", f"{se:.4g}", f"{mc_abs_error:.4g}"
            ])
            print(
                f"{name:10s} n={len(pts):2d} k={k} "
                f"greedy={greedy.selected_indices} R2={greedy.r2_value:.8f} "
                f"opt={list(optimum.selected_indices)} R2*={optimum.r2_value:.8f} "
                f"gain_ratio={ratio:.4f}"
            )
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
