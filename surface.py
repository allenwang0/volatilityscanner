"""Per-expiry SVI fits and the summary numbers the dashboard and report show."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from engine import CALL, PUT, Chain, vega_vec
from svi import SVIFit, SVIParams, calendar_violations, fit_svi

CURVE_POINTS = 101
SURFACE_K_POINTS = 41
SKEW_DELTA = 0.25


@dataclass(frozen=True)
class SliceFit:
    days: float
    T: float
    forward: float
    fit: SVIFit
    atm_iv: float
    skew_25d: float
    k_min: float
    k_max: float


def otm_mask(chain: Chain) -> np.ndarray:
    """Puts below the forward and calls at or above it: the liquid side of each strike."""
    below = chain.strike < chain.forward
    return np.where(below, chain.flag == PUT, chain.flag == CALL)


def _k_at_forward_delta(
    params: SVIParams, T: float, delta: float, flag: int, k_lo: float, k_hi: float
) -> float:
    """Log-moneyness in [k_lo, k_hi] where the undiscounted Black delta is delta (calls) or
    -delta (puts); NaN when the fitted smile does not reach that delta inside the range."""

    def gap(k: float) -> float:
        w = float(params.total_variance(np.array([k]))[0])
        d1 = (-k + w / 2) / math.sqrt(w)
        return float(ndtr(d1)) - delta if flag == CALL else float(ndtr(-d1)) - delta

    if gap(k_lo) * gap(k_hi) > 0:
        return math.nan
    return brentq(gap, k_lo, k_hi)


def skew_25_delta(params: SVIParams, T: float, k_lo: float, k_hi: float) -> float:
    """25-delta put IV minus 25-delta call IV in vol points, searched only over quoted strikes.

    Outside the quotes an SVI fit is an extrapolation and can be far off (at IV noise sd
    0.002 a 90-day fit reached rho = 0.88), so a skew that needs it is NaN instead.
    """
    k_put = _k_at_forward_delta(params, T, SKEW_DELTA, PUT, k_lo, k_hi)
    k_call = _k_at_forward_delta(params, T, SKEW_DELTA, CALL, k_lo, k_hi)
    if math.isnan(k_put) or math.isnan(k_call):
        return math.nan
    iv = params.implied_vol(np.array([k_put, k_call]), T)
    return float(iv[0] - iv[1]) * 100


def fit_surface(chain: Chain, iv: np.ndarray) -> list[SliceFit]:
    """Fit one SVI slice per expiry to the out-of-the-money quotes with a solved IV."""
    otm = otm_mask(chain)
    vega = vega_vec(chain.spot, chain.strike, chain.T, chain.r, np.where(np.isfinite(iv), iv, 0.2), chain.q)
    slices = []
    for days in np.unique(chain.expiry_days):
        use = (chain.expiry_days == days) & otm & np.isfinite(iv)
        if use.sum() < 5:
            continue
        T = float(chain.T[use][0])
        forward = float(chain.forward[use][0])
        k = np.log(chain.strike[use] / forward)
        fit = fit_svi(k, iv[use], T, vega[use])
        slices.append(
            SliceFit(
                days=float(days),
                T=T,
                forward=forward,
                fit=fit,
                atm_iv=float(fit.params.implied_vol(np.array([0.0]), T)[0]),
                skew_25d=skew_25_delta(fit.params, T, float(k.min()), float(k.max())),
                k_min=float(k.min()),
                k_max=float(k.max()),
            )
        )
    return slices


def slice_curve(s: SliceFit, points: int = CURVE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    k = np.linspace(s.k_min, s.k_max, points)
    return k, s.fit.params.implied_vol(k, s.T)


def surface_grid(slices: list[SliceFit]) -> dict:
    """Fitted IV on a common k grid (the range every slice covers) by days to expiry."""
    k = np.linspace(max(s.k_min for s in slices), min(s.k_max for s in slices), SURFACE_K_POINTS)
    return {
        "k": k.tolist(),
        "days": [s.days for s in slices],
        "iv": [(s.fit.params.implied_vol(k, s.T) * 100).tolist() for s in slices],
    }


def fitted_calendar_crossings(slices: list[SliceFit]) -> list[tuple[float, float]]:
    k = np.linspace(max(s.k_min for s in slices), min(s.k_max for s in slices), 201)
    return calendar_violations([(s.days, s.fit.params) for s in slices], k)
