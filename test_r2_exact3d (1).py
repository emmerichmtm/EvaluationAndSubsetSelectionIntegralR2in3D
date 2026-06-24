"""Small verification script for r2_exact3d.py.

The arrangement implementation is deliberately simple and intended as a
reference implementation.  The number of arrangement cells grows quickly, so
this script uses small 3D examples.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from r2_exact3d import exact_r2_3d, monte_carlo_r2_3d, random_nondominated_points


def run() -> None:
    out = Path(__file__).resolve().parent / "experiment_results.csv"
    cases = [
        ("hand4", np.array([[0.30, 1.20, 1.00], [0.65, 0.70, 0.85], [1.10, 0.40, 0.55], [0.95, 0.95, 0.25]])),
        ("random3", random_nondominated_points(3, seed=10)),
        ("random4", random_nondominated_points(4, seed=11)),
    ]
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "n", "exact", "monte_carlo", "stderr", "abs_error", "z_score", "cells", "lines", "exact_seconds"])
        for name, pts in cases:
            t0 = time.perf_counter()
            exact = exact_r2_3d(pts)
            seconds = time.perf_counter() - t0
            mc, se = monte_carlo_r2_3d(pts, samples=200_000, seed=12345)
            abs_err = abs(mc - exact.value)
            z = abs_err / se if se > 0 else float("nan")
            writer.writerow([name, len(pts), f"{exact.value:.12g}", f"{mc:.12g}", f"{se:.4g}", f"{abs_err:.4g}", f"{z:.3f}", exact.num_cells, exact.num_lines, f"{seconds:.4f}"])
            print(f"{name:8s} n={len(pts):2d} exact={exact.value:.8f} MC={mc:.8f} +/- {1.96*se:.6f} cells={exact.num_cells:5d} lines={exact.num_lines:4d}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
