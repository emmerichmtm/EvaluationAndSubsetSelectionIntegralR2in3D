"""Exact arrangement-based evaluation of the integral R2 indicator in 3 objectives.

The implementation follows the arrangement idea in Section 7 of the report:
for a finite set S of positive objective vectors p=(p1,p2,p3), evaluate

    R2(S) = int_{Delta_2} min_{p in S} max_i w_i p_i dw,

where Delta_2 = {w1,w2 >= 0, w1+w2 <= 1}, w3=1-w1-w2.

The algorithm constructs the planar arrangement induced by all equality
lines between affine pieces w_i p_i.  On each resulting polygonal cell the
lower envelope is one affine function, and integration is exact up to ordinary
floating-point and polygon-clipping roundoff.

Dependencies: numpy, shapely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import math

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import split

Affine = Tuple[int, int, float, float, float]  # point index, coordinate index, a*x+b*y+c
LineCoeff = Tuple[float, float, float]


@dataclass
class EvaluationResult:
    value: float
    num_cells: int
    num_lines: int
    active_counts: Dict[Tuple[int, int], int]


def _validate_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an array of shape (n, 3)")
    if len(pts) == 0:
        raise ValueError("at least one point is required")
    if not np.all(np.isfinite(pts)):
        raise ValueError("all coordinates must be finite")
    if np.any(pts <= 0):
        raise ValueError("all coordinates must be positive in this implementation")
    return pts


def affine_forms(points: np.ndarray) -> List[Affine]:
    """Return affine forms for w1*p1, w2*p2, w3*p3 with w3=1-w1-w2."""
    pts = _validate_points(points)
    forms: List[Affine] = []
    for r, (p1, p2, p3) in enumerate(pts):
        forms.append((r, 0, float(p1), 0.0, 0.0))       # p1*w1
        forms.append((r, 1, 0.0, float(p2), 0.0))       # p2*w2
        forms.append((r, 2, -float(p3), -float(p3), float(p3)))  # p3*(1-w1-w2)
    return forms


def _normalize_line(a: float, b: float, c: float, tol: float = 1e-14) -> LineCoeff | None:
    norm = math.hypot(a, b)
    if norm < tol:
        return None
    a, b, c = a / norm, b / norm, c / norm
    # Fix sign for duplicate detection.
    if a < -tol or (abs(a) <= tol and b < -tol):
        a, b, c = -a, -b, -c
    # Normalize near-zero values.
    def clean(x: float) -> float:
        return 0.0 if abs(x) < 1e-13 else x
    return clean(a), clean(b), clean(c)


def arrangement_lines(forms: Sequence[Affine], digits: int = 12) -> List[LineCoeff]:
    """All unique equality lines between affine pieces."""
    seen = set()
    lines: List[LineCoeff] = []
    for i in range(len(forms)):
        _, _, a1, b1, c1 = forms[i]
        for j in range(i + 1, len(forms)):
            _, _, a2, b2, c2 = forms[j]
            lc = _normalize_line(a1 - a2, b1 - b2, c1 - c2)
            if lc is None:
                continue
            key = tuple(round(x, digits) for x in lc)
            if key not in seen:
                seen.add(key)
                lines.append(lc)
    return lines


def _line_string_for_box(a: float, b: float, c: float, lo: float = -1.0, hi: float = 2.0) -> LineString:
    """Construct a long segment of a*x+b*y+c=0 through a box containing the simplex."""
    pts: List[Tuple[float, float]] = []
    if abs(b) > 1e-14:
        for x in (lo, hi):
            y = -(a * x + c) / b
            if lo - 1e-9 <= y <= hi + 1e-9:
                pts.append((x, y))
    if abs(a) > 1e-14:
        for y in (lo, hi):
            x = -(b * y + c) / a
            if lo - 1e-9 <= x <= hi + 1e-9:
                pts.append((x, y))
    # In rare cases the intersection points are outside the small box due to rounding.
    if len(pts) < 2:
        # Use a point on the line and a perpendicular direction.
        if abs(a) >= abs(b):
            x0, y0 = -c / a, 0.0
        else:
            x0, y0 = 0.0, -c / b
        dx, dy = -b, a
        scale = 10.0
        pts = [(x0 - scale * dx, y0 - scale * dy), (x0 + scale * dx, y0 + scale * dy)]
    # Remove duplicate points.
    unique = []
    for p in pts:
        if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 1e-24 for q in unique):
            unique.append(p)
    if len(unique) == 1:
        x0, y0 = unique[0]
        dx, dy = -b, a
        unique = [(x0 - 10 * dx, y0 - 10 * dy), (x0 + 10 * dx, y0 + 10 * dy)]
    return LineString(unique[:2])


def _iter_polygons(geom) -> Iterable[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []


def arrangement_cells(lines: Sequence[LineCoeff], area_tol: float = 1e-13) -> List[Polygon]:
    """Split the simplex triangle by all arrangement lines."""
    simplex = Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    cells: List[Polygon] = [simplex]
    for a, b, c in lines:
        cut = _line_string_for_box(a, b, c)
        new_cells: List[Polygon] = []
        for poly in cells:
            if poly.area <= area_tol:
                continue
            try:
                pieces = list(split(poly, cut).geoms)
            except Exception:
                pieces = [poly]
            if len(pieces) <= 1:
                new_cells.append(poly)
            else:
                for piece in pieces:
                    if isinstance(piece, Polygon) and piece.area > area_tol:
                        new_cells.append(piece)
        cells = new_cells
    return cells


def _form_value(form: Affine, x: float, y: float) -> float:
    _, _, a, b, c = form
    return a * x + b * y + c


def active_form(forms: Sequence[Affine], n_points: int, x: float, y: float) -> Affine:
    """Return the affine form that equals tau at a representative point."""
    vals = np.array([_form_value(f, x, y) for f in forms], dtype=float).reshape(n_points, 3)
    point_max = vals.max(axis=1)
    p = int(point_max.argmin())
    coord = int(vals[p].argmax())
    return forms[3 * p + coord]


def exact_r2_3d(points: np.ndarray, return_cells: bool = False) -> EvaluationResult | Tuple[EvaluationResult, List[Polygon]]:
    """Evaluate R2 exactly by arrangement subdivision in three objectives.

    The measure is the unnormalized Lebesgue measure on Delta_2, so the area
    of Delta_2 is 1/2.  For a probability average over the simplex, multiply
    the returned value by 2.
    """
    pts = _validate_points(points)
    forms = affine_forms(pts)
    lines = arrangement_lines(forms)
    cells = arrangement_cells(lines)
    total = 0.0
    counts: Dict[Tuple[int, int], int] = {}
    for poly in cells:
        # representative point avoids boundary degeneracy better than centroid
        rp = poly.representative_point()
        form = active_form(forms, len(pts), rp.x, rp.y)
        pidx, coord, a, b, c = form
        centroid = poly.centroid
        total += poly.area * (a * centroid.x + b * centroid.y + c)
        counts[(pidx, coord)] = counts.get((pidx, coord), 0) + 1
    result = EvaluationResult(float(total), len(cells), len(lines), counts)
    return (result, cells) if return_cells else result


def monte_carlo_r2_3d(points: np.ndarray, samples: int = 100_000, seed: int = 0) -> Tuple[float, float]:
    """Monte Carlo estimate of the same unnormalized integral and its standard error."""
    pts = _validate_points(points)
    rng = np.random.default_rng(seed)
    u = rng.random((samples, 2))
    mask = u.sum(axis=1) > 1.0
    u[mask] = 1.0 - u[mask]
    w1 = u[:, 0]
    w2 = u[:, 1]
    w3 = 1.0 - w1 - w2
    vals = np.minimum.reduce([
        np.maximum.reduce([w1 * p[0], w2 * p[1], w3 * p[2]]) for p in pts
    ])
    area = 0.5
    estimate = area * float(vals.mean())
    stderr = area * float(vals.std(ddof=1) / math.sqrt(samples))
    return estimate, stderr


def random_nondominated_points(n: int, seed: int = 0) -> np.ndarray:
    """Generate a simple positive three-objective minimization test set."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.15, 1.5, size=(5 * n, 3))
    # Keep nondominated points in minimization sense.
    keep = []
    for i, p in enumerate(raw):
        dominated = False
        for j, q in enumerate(raw):
            if j != i and np.all(q <= p) and np.any(q < p):
                dominated = True
                break
        if not dominated:
            keep.append(p)
        if len(keep) >= n:
            break
    if len(keep) < n:
        keep.extend(rng.uniform(0.2, 1.2, size=(n - len(keep), 3)))
    return np.array(keep[:n], dtype=float)


if __name__ == "__main__":
    pts = np.array([
        [0.30, 1.20, 1.00],
        [0.65, 0.70, 0.85],
        [1.10, 0.40, 0.55],
        [0.95, 0.95, 0.25],
    ])
    exact = exact_r2_3d(pts)
    mc, se = monte_carlo_r2_3d(pts, samples=200_000, seed=1)
    print(f"exact R2 integral: {exact.value:.10f}")
    print(f"MC estimate       : {mc:.10f} +/- {1.96*se:.10f} (95% CI)")
    print(f"cells={exact.num_cells}, arrangement lines={exact.num_lines}")
