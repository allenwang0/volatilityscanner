"""Raw SVI smiles (Gatheral 2004): fit, no-arbitrage checks, ATM vol and 25-delta skew."""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

DENSITY_GRID = np.linspace(-1.0, 1.0, 201)  # k grid for the butterfly (density) check
RHO_MAX = 0.999  # keeps sqrt(1 - rho^2) away from zero
# Roger Lee's moment formula caps the wing slope of total variance,
# b (1 + |rho|), at 2; b <= 1 keeps every fit inside it.
B_MAX = 1.0
DELTA_K_GRID = np.linspace(-1.0, 1.0, 401)
# Vega weights are floored at 5% of the largest so short-dated wing quotes
# (near-zero vega) still shape the fit; without the floor the 7-day fit
# misses its wings by over a vol point.
MIN_WEIGHT = 0.05


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def w(self, k):
        """Total implied variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))."""
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.s * self.s))

    def dw(self, k):
        x = np.asarray(k, dtype=float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.s * self.s))

    def d2w(self, k):
        x = np.asarray(k, dtype=float) - self.m
        return self.b * self.s * self.s / (x * x + self.s * self.s) ** 1.5

    def iv(self, k, T: float):
        return np.sqrt(np.maximum(self.w(k), 0.0) / T)

    def as_dict(self) -> dict:
        return asdict(self)


def ssvi_slice(theta: float, rho: float, eta: float, gamma: float) -> SVIParams:
    """One expiry of an SSVI surface (Gatheral and Jacquier 2014) written as raw SVI.

    theta is ATM total variance; phi(theta) = eta theta^-gamma (1 + theta)^(gamma - 1).
    The surface has no calendar arbitrage when theta rises with T and gamma is
    in (0, 1/2], and no butterfly arbitrage when eta (1 + |rho|) <= 2.
    """
    phi = eta * theta**-gamma * (1 + theta) ** (gamma - 1)
    return SVIParams(
        a=theta / 2 * (1 - rho * rho),
        b=theta * phi / 2,
        rho=rho,
        m=-rho / phi,
        s=math.sqrt(1 - rho * rho) / phi,
    )


def density_g(p: SVIParams, k):
    """Gatheral-Jacquier g(k); the risk-neutral density is non-negative iff g >= 0."""
    w, w1, w2 = p.w(k), p.dw(k), p.d2w(k)
    return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


@dataclass
class SVIFit:
    params: SVIParams
    rmse_vol_pts: float  # fitted vs market IV, in vol points (1 vol point = 0.01)
    n_points: int
    butterfly_k: list[float]  # grid k where g(k) < 0


def _to_params(x) -> SVIParams:
    c, b, rho, m, s = x
    # c = a + b s sqrt(1 - rho^2) is the smile's minimum total variance; c >= 0
    # keeps w >= 0 everywhere and is a simple box bound for least_squares.
    return SVIParams(c - b * s * math.sqrt(1 - rho * rho), b, rho, m, s)


def fit_svi(k, iv, T: float, weights=None, start: SVIParams | None = None) -> SVIFit:
    """Fit raw SVI to market IVs at log-forward moneyness k by weighted least squares
    on total variance, with b >= 0, |rho| < 1, s > 0 and a + b s sqrt(1 - rho^2) >= 0.

    Weights are typically vega, floored at MIN_WEIGHT of the largest. Tries a
    few starting points (plus `start`, such as the previous tick's fit) and
    keeps the best.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    w_mkt = iv * iv * T
    wt = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    wt = np.sqrt(np.maximum(wt / wt.max(), MIN_WEIGHT))
    span = max(float(k.max() - k.min()), 0.05)

    lo = [0.0, 0.0, -RHO_MAX, float(k.min()) - span, 1e-4]
    hi = [2 * float(w_mkt.max()) + 1e-6, B_MAX, RHO_MAX, float(k.max()) + span, 2 * span]

    def residuals(x):
        return wt * (_to_params(x).w(k) - w_mkt)

    starts = []
    if start is not None:
        c0 = start.a + start.b * start.s * math.sqrt(1 - start.rho**2)
        starts.append([c0, start.b, start.rho, start.m, start.s])
    w_min = float(w_mkt.min())
    for rho0 in (-0.5, 0.0, -0.9):
        starts.append([w_min, 0.1 * float(w_mkt.max()) / span, rho0, float(k[np.argmin(w_mkt)]), 0.1 * span])

    best = None
    for x0 in starts:
        x0 = np.clip(x0, lo, hi)
        res = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=2000)
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    p = _to_params(best.x)
    rmse = float(np.sqrt(np.mean((p.iv(k, T) - iv) ** 2))) * 100
    g = density_g(p, DENSITY_GRID)
    return SVIFit(p, rmse, int(k.size), [float(x) for x in DENSITY_GRID[g < 0]])


def calendar_crossings(slices: list[tuple[float, SVIParams]], k_grid=DENSITY_GRID) -> list[dict]:
    """Grid points where a later expiry's fitted total variance is below an earlier one's."""
    out = []
    ordered = sorted(slices, key=lambda x: x[0])
    for (t1, p1), (t2, p2) in zip(ordered, ordered[1:], strict=False):
        bad = k_grid[p2.w(k_grid) < p1.w(k_grid)]
        if bad.size:
            out.append({"T1": t1, "T2": t2, "k_min": float(bad.min()), "k_max": float(bad.max())})
    return out


def atm_vol(p: SVIParams, T: float) -> float:
    return float(p.iv(0.0, T))


def _k_at_delta(p: SVIParams, T: float, call_delta: float) -> float:
    """Log-forward moneyness where the forward call delta N(d1) equals call_delta.

    Takes the root closest to the money; NaN if the smile never reaches that delta.
    """

    def f(k):
        sig = np.sqrt(np.maximum(p.w(k), 1e-12))
        return ndtr(-k / sig + sig / 2) - call_delta

    vals = f(DELTA_K_GRID)
    crossings = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
    if crossings.size == 0:
        return float("nan")
    i = crossings[np.argmin(np.abs(DELTA_K_GRID[crossings]))]
    return brentq(lambda x: float(f(x)), DELTA_K_GRID[i], DELTA_K_GRID[i + 1])


def skew_25d(p: SVIParams, T: float) -> float:
    """25-delta put IV minus 25-delta call IV, in vol points (NaN if out of reach)."""
    k_put = _k_at_delta(p, T, 0.75)  # a 25-delta put has call delta 0.75
    k_call = _k_at_delta(p, T, 0.25)
    return float(p.iv(k_put, T) - p.iv(k_call, T)) * 100
