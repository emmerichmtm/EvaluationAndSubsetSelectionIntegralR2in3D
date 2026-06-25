# Exact 3D integral R2 evaluator and greedy selector

This folder contains small reference implementations for the algorithms described in the report:

- exact arrangement-based evaluation of the three-objective integral R2 indicator;
- greedy cardinality-k subset selection using exact R2 marginal improvements.

The code is intended to be transparent and reproducible. It is not optimized for large instances.

## Files

- `r2_exact3d.py` - reference implementation for three-objective integral R2 evaluation.
- `test_r2_exact3d.py` - verification script comparing the arrangement result to Monte Carlo integration.
- `experiment_results.csv` - output of `test_r2_exact3d.py`.
- `greedy_r2_3d.py` - greedy subset selection using the exact evaluator as an oracle.
- `test_greedy_r2_3d.py` - validation script comparing greedy selections to brute-force optima on small instances.
- `greedy_experiment_results.csv` - output of `test_greedy_r2_3d.py`.
- `integral_r2_3d_report_v17.tex` - patched report source with implementation appendices.
- `integral_r2_3d_report_v17.pdf` - compiled report.
- `verify_perspective_mapping.py` - a verification script for the perspective mapping

## Mathematical convention

For positive objective/loss vectors `p=(p1,p2,p3)` and simplex weights
`w=(w1,w2,w3)`, `w_i >= 0`, `w1+w2+w3=1`, the exact evaluator computes

```math
R_2(S) = \int_{\Delta_2} \min_{p\in S} \max_i w_i p_i\,dw.
```

Here `Delta_2` is represented as the right triangle
`{(w1,w2): w1>=0, w2>=0, w1+w2<=1}` with area `1/2`. Therefore the code
returns the **unnormalized integral**. If a probability average over the
simplex is desired, multiply the returned value by `2`.

The greedy code uses a fixed dominated anchor `pbar` and

```math
Q_{pbar}(S) = R_2({pbar}) - R_2(S).
```

By default, `pbar` is chosen coordinatewise above all candidate points:
`pbar = 1.05 * max(points, axis=0) + 1e-9`.

## Exact evaluation method

The evaluator constructs all equality lines between affine pieces

```math
w_1p_1, \quad w_2p_2, \quad (1-w_1-w_2)p_3.
```

These lines induce a planar subdivision of the simplex. On each cell the lower
Tchebycheff envelope is one affine function. The code integrates that affine
function over the polygonal cell by using the polygon area and centroid.

## Greedy method

Run greedy selection by repeatedly adding the point with the largest exact R2
decrease:

```math
p_t \in \arg\max_{p\in P\setminus S_{t-1}}
      \{ R_2(S_{t-1}) - R_2(S_{t-1}\cup\{p\}) \}.
```

At the first step, this is equivalent to choosing the singleton with the
smallest R2 value, or equivalently the largest gain from the dominated anchor.

The validation script also enumerates all cardinality-k subsets for small test
instances. This checks the actual greedy result against the exact optimum, but
brute force is only feasible for tiny examples.

## Run

```bash
python3 r2_exact3d.py
python3 test_r2_exact3d.py
python3 greedy_r2_3d.py
python3 test_greedy_r2_3d.py
python3 verify_perspective_mapping.py
```

The two test scripts write `experiment_results.csv` and
`greedy_experiment_results.csv`.

## Dependencies

```bash
pip install numpy shapely
```

The experiments also use only the Python standard library plus NumPy and Shapely.

## Notes and limitations

- The implementation is a clear reference implementation, not an optimized QR2-style code.
- The geometric arrangement is computed using double-precision polygon clipping through Shapely. Thus it is exact at the level of the symbolic subdivision idea, but numerically it is subject to floating-point and clipping tolerance.
- The number of cells grows quickly with the number of points. The included tests use small instances, as expected for a direct arrangement implementation.
- Greedy selection repeatedly calls the exact evaluator. For larger candidate sets, one would use lazy greedy, incremental updates, Monte Carlo estimates, or a specialized exact evaluator.
