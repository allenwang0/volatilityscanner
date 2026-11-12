"""Raw SVI smiles (Gatheral 2004): fit, evaluate, and check for static arbitrage.

Total implied variance at log-forward-moneyness k = log(K/F):
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

DENSITY_GRID_POINTS = 201
SEARCH_GRID_M = 9
SEARCH_GRID_S = 8
# Short-dated wings have almost no vega; without a floor the fit ignores them and the
# smile there can miss the quotes by several vol points.
MIN_RELATIVE_WEIGHT = 0.2


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        d = k - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d * d + self.s * self.s))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    def density_g(self, k: np.ndarray) -> np.ndarray:
        """Gatheral-Jacquier g(k); the smile is free of butterfly arbitrage where g(k) >= 0."""
        d = k - self.m
        root = np.sqrt(d * d + self.s * self.s)
        w = self.total_variance(k)
        w1 = self.b * (self.rho + d / root)
        w2 = self.b * self.s * self.s / root**3
        return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2

    @classmethod
    def from_atm(
        cls, T: float, atm_vol: float, b: float, rho: float, m: float, s: float
    ) -> "SVIParams":
        """Choose a so that w(0) = atm_vol^2 * T."""
        a = atm_vol**2 * T - b * (rho * -m + math.sqrt(m * m + s * s))
        return cls(a, b, rho, m, s)


@dataclass(frozen=True)
class SVIFit:
    params: SVIParams
    rmse_vol_points: float
    min_density_g: float
    success: bool


def _unpack(x: np.ndarray) -> SVIParams:
    # x = (c, b, rho, m, s) with c = a + b s sqrt(1 - rho^2) >= 0, the minimum of w(k).
    c, b, rho, m, s = (float(v) for v in x)
    return SVIParams(c - b * s * math.sqrt(1 - rho * rho), b, rho, m, s)


def _grid_m(k: np.ndarray) -> np.ndarray:
    return np.linspace(k.min(), k.max(), SEARCH_GRID_M)


def _grid_s(k: np.ndarray) -> np.ndarray:
    return np.geomspace(0.005, max(float(k.max() - k.min()), 0.01), SEARCH_GRID_S)


def _linear_start(k: np.ndarray, w: np.ndarray, wts: np.ndarray, m: float, s: float) -> np.ndarray:
    """For fixed (m, s), w is linear in (a, b*rho, b): solve it, then project into the bounds."""
    d = k - m
    design = np.column_stack([np.ones_like(k), d, np.sqrt(d * d + s * s)]) * wts[:, None]
    a, b_rho, b = np.linalg.lstsq(design, w * wts, rcond=None)[0]
    b = max(float(b), 1e-8)
    rho = float(np.clip(b_rho / b, -0.99, 0.99))
    c = max(float(a) + b * s * math.sqrt(1 - rho * rho), 0.0)
    return np.array([c, b, rho, m, s])


def fit_svi(k: np.ndarray, iv: np.ndarray, T: float, weights: np.ndarray | None = None) -> SVIFit:
    """Bounded least squares on total variance, weighted (for example by vega).

    Weights are scaled to a maximum of 1 and floored at MIN_RELATIVE_WEIGHT.

    The start comes from a grid over (m, s) with a linear solve for the other three
    parameters (the quasi-explicit idea of Zeliade 2009). The constraint
    a + b s sqrt(1 - rho^2) >= 0 (non-negative minimum variance) is enforced by
    fitting that minimum directly with a lower bound of zero.
    """
    ok = np.isfinite(iv)
    k, iv = k[ok], iv[ok]
    if weights is None:
        wts = np.ones_like(k)
    else:
        wts = np.maximum(weights[ok] / np.max(weights[ok]), MIN_RELATIVE_WEIGHT)
    w = iv**2 * T

    def residuals(x: np.ndarray) -> np.ndarray:
        return wts * (_unpack(x).total_variance(k) - w)

    lower = [0.0, 0.0, -0.999, 2 * k.min() - k.max(), 1e-4]
    upper = [np.inf, np.inf, 0.999, 2 * k.max() - k.min(), 1.0]
    x0 = min(
        (_linear_start(k, w, wts, m, sig) for m in _grid_m(k) for sig in _grid_s(k)),
        key=lambda x: float(np.sum(residuals(x) ** 2)),
    )
    x0 = np.clip(x0, lower, upper)
    result = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac", max_nfev=500)
    params = _unpack(result.x)
    rmse = float(np.sqrt(np.mean((params.implied_vol(k, T) - iv) ** 2)) * 100)
    grid = np.linspace(k.min(), k.max(), DENSITY_GRID_POINTS)
    return SVIFit(params, rmse, float(params.density_g(grid).min()), bool(result.success))


def calendar_violations(
    slices: list[tuple[float, SVIParams]], k_grid: np.ndarray, tol: float = 1e-10
) -> list[tuple[float, float]]:
    """Return (earlier_T, later_T) pairs whose fitted total variance crosses on k_grid."""
    ordered = sorted(slices, key=lambda item: item[0])
    return [
        (t1, t2)
        for (t1, p1), (t2, p2) in zip(ordered, ordered[1:], strict=False)
        if np.any(p2.total_variance(k_grid) < p1.total_variance(k_grid) - tol)
    ]
