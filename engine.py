import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

CALL = 1
PUT = -1


def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))


def standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type_flag: int, q: float = 0.0
) -> float:
    """European option price; option_type_flag is CALL (1) or PUT (-1), q a dividend yield."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) * option_type_flag)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc_spot = S * math.exp(-q * T)
    disc_strike = K * math.exp(-r * T)
    if option_type_flag == CALL:
        return disc_spot * standard_normal_cdf(d1) - disc_strike * standard_normal_cdf(d2)
    return disc_strike * standard_normal_cdf(-d2) - disc_spot * standard_normal_cdf(-d1)


def calculate_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * standard_normal_pdf(d1)


SIGMA_LO, SIGMA_HI = 1e-4, 5.0  # search bracket for sigma
MAX_NEWTON_ITER = 50
SIGMA_TOL = 1e-12  # stop when a Newton step moves sigma less than this
MIN_VEGA_PER_SPOT = 1e-8  # below this Newton steps are unreliable; use Brent instead


@dataclass
class Chain:
    """One snapshot of an option chain as parallel arrays, one entry per quote.

    price is the mid used for IV; bid and ask are NaN when the source has no quotes.
    forward is the forward price for each quote's expiry.
    """

    spot: float
    r: float
    q: float
    expiry_days: np.ndarray
    T: np.ndarray
    strike: np.ndarray
    flag: np.ndarray
    price: np.ndarray
    forward: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    source: str = "Simulated data"

    def __len__(self) -> int:
        return len(self.price)


@dataclass
class IVResult:
    sigma: float  # NaN when not converged
    converged: bool
    iterations: int
    reason: str  # "ok" or why no volatility was found


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type_flag: int,
    q: float = 0.0,
) -> IVResult:
    """Black-Scholes implied volatility with bounds checks, Newton and a Brent fallback.

    Failed solves return sigma NaN with a reason code; an unconverged iterate is never
    returned.
    """
    nan = float("nan")
    if not all(math.isfinite(x) for x in (market_price, S, K, T, r, q)):
        return IVResult(nan, False, 0, "invalid_input")
    if T <= 0 or S <= 0 or K <= 0:
        return IVResult(nan, False, 0, "invalid_input")
    if market_price <= 0:
        return IVResult(nan, False, 0, "zero_price")

    # Static no-arbitrage bounds on a European option price
    fwd_spot = S * math.exp(-q * T)
    disc_strike = K * math.exp(-r * T)
    if option_type_flag == CALL:
        lower, upper = max(0.0, fwd_spot - disc_strike), fwd_spot
    else:
        lower, upper = max(0.0, disc_strike - fwd_spot), disc_strike
    if market_price <= lower:
        return IVResult(nan, False, 0, "at_or_below_intrinsic")
    if market_price >= upper:
        return IVResult(nan, False, 0, "above_upper_bound")

    def objective(sig: float) -> float:
        return black_scholes_price(S, K, T, r, sig, option_type_flag, q) - market_price

    # Brenner-Subrahmanyam initial guess, clamped into the bracket
    sigma = market_price / fwd_spot * math.sqrt(2 * math.pi / T)
    sigma = min(max(sigma, SIGMA_LO), SIGMA_HI)

    for i in range(1, MAX_NEWTON_ITER + 1):
        vega = calculate_vega(S, K, T, r, sigma, q)
        if vega < MIN_VEGA_PER_SPOT * S:
            break
        step = objective(sigma) / vega
        new_sigma = sigma - step
        if not SIGMA_LO <= new_sigma <= SIGMA_HI:
            break
        sigma = new_sigma
        if abs(step) < SIGMA_TOL:
            return IVResult(sigma, True, i, "ok")

    # Newton left the bracket, stalled on a flat vega, or ran out of steps
    if objective(SIGMA_LO) * objective(SIGMA_HI) > 0:
        return IVResult(nan, False, 0, "outside_bracket")
    # brentq raises if it fails to converge
    sigma, info = brentq(objective, SIGMA_LO, SIGMA_HI, xtol=SIGMA_TOL, full_output=True)
    return IVResult(float(sigma), True, info.iterations, "ok")


@dataclass
class IVBatch:
    """Vectorized IVResult: one entry per option."""

    sigma: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    reason: np.ndarray
    brent_fallback: np.ndarray


def black_scholes_price_vec(S, K, T, r, sigma, flag, q=0.0):
    """Array version of black_scholes_price for T > 0 and sigma > 0."""
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return flag * (S * np.exp(-q * T) * ndtr(flag * d1) - K * np.exp(-r * T) * ndtr(flag * d2))


def vega_vec(S, K, T, r, sigma, q=0.0):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return S * np.exp(-q * T) * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)


def implied_volatility_vec(price, S, K, T, r, flag, q=0.0) -> IVBatch:
    """implied_volatility for a whole chain at once.

    Runs the same bounds checks and Newton iteration as the scalar solver on every
    option in parallel, masking out options as they converge. Options whose Newton
    step leaves the bracket, whose vega is too small, or that run out of steps go to
    the scalar solver, which finishes them with Brent's method.
    """
    price, S, K, T, r, flag, q = np.broadcast_arrays(
        *(np.asarray(x, float) for x in (price, S, K, T, r, flag, q))
    )
    n = price.shape
    sigma = np.full(n, np.nan)
    converged = np.zeros(n, bool)
    iterations = np.zeros(n, int)
    reason = np.full(n, "ok", dtype=object)

    with np.errstate(invalid="ignore", divide="ignore"):
        finite = np.isfinite(price) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T)
        finite &= np.isfinite(r) & np.isfinite(q)
        invalid = ~finite | (T <= 0) | (S <= 0) | (K <= 0)
        fwd_spot = S * np.exp(-q * T)
        disc_strike = K * np.exp(-r * T)
        lower = np.maximum(0.0, flag * (fwd_spot - disc_strike))
        upper = np.where(flag == CALL, fwd_spot, disc_strike)
    checks = (
        (invalid, "invalid_input"),
        (price <= 0, "zero_price"),
        (price <= lower, "at_or_below_intrinsic"),
        (price >= upper, "above_upper_bound"),
    )
    pending = np.ones(n, bool)
    for failed, code in checks:
        hit = pending & failed
        reason[hit] = code
        pending &= ~hit

    guess = (
        price / np.where(pending, fwd_spot, 1.0) * np.sqrt(2 * np.pi / np.where(pending, T, 1.0))
    )
    sig = np.clip(guess, SIGMA_LO, SIGMA_HI)
    active = pending.copy()
    fallback = np.zeros(n, bool)
    for i in range(1, MAX_NEWTON_ITER + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        a_S, a_K, a_T, a_r, a_q, a_f = S[idx], K[idx], T[idx], r[idx], q[idx], flag[idx]
        vega = vega_vec(a_S, a_K, a_T, a_r, sig[idx], a_q)
        flat = vega < MIN_VEGA_PER_SPOT * a_S
        model = black_scholes_price_vec(a_S, a_K, a_T, a_r, sig[idx], a_f, a_q)
        step = np.where(flat, 0.0, (model - price[idx]) / np.where(flat, 1.0, vega))
        new = sig[idx] - step
        out = flat | (new < SIGMA_LO) | (new > SIGMA_HI)
        fallback[idx[out]] = True
        moved = idx[~out]
        sig[moved] = new[~out]
        done = moved[np.abs(step[~out]) < SIGMA_TOL]
        sigma[done] = sig[done]
        converged[done] = True
        iterations[done] = i
        active[idx[out]] = False
        active[done] = False
    fallback |= active

    for j in np.flatnonzero(fallback):
        res = implied_volatility(price[j], S[j], K[j], T[j], r[j], int(flag[j]), q=q[j])
        sigma[j], converged[j], iterations[j], reason[j] = (
            res.sigma,
            res.converged,
            res.iterations,
            res.reason,
        )
    return IVBatch(sigma, converged, iterations, reason, fallback)
