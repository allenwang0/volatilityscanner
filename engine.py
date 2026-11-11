import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------
# 1. Standard Black-Scholes (No Numba)
# ---------------------------------------------------------
# REMOVED: @jit(nopython=True)
def standard_normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))


# REMOVED: @jit(nopython=True)
def standard_normal_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# REMOVED: @jit(nopython=True)
def black_scholes_price(S, K, T, r, sigma, option_type_flag, q=0.0):
    # option_type_flag: 1 for Call, -1 for Put; q: continuous dividend yield
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) * option_type_flag)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type_flag == 1:  # Call
        price = S * math.exp(-q * T) * standard_normal_cdf(d1) - K * math.exp(-r * T) * standard_normal_cdf(d2)
    else:  # Put
        price = K * math.exp(-r * T) * standard_normal_cdf(-d2) - S * math.exp(-q * T) * standard_normal_cdf(-d1)
    return price


# REMOVED: @jit(nopython=True)
def calculate_vega(S, K, T, r, sigma, q=0.0):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * standard_normal_pdf(d1)


# ---------------------------------------------------------
# 2. Implied Volatility Solver (bracketed Newton with Brent fallback)
# ---------------------------------------------------------
SIGMA_LO, SIGMA_HI = 1e-4, 5.0  # search bracket for sigma
MAX_NEWTON_ITER = 50
SIGMA_TOL = 1e-12  # stop when a Newton step moves sigma less than this


@dataclass
class IVResult:
    sigma: float  # NaN when not converged
    converged: bool
    iterations: int
    reason: str  # "ok" or why no volatility was found


def implied_volatility(market_price, S, K, T, r, option_type_flag, q=0.0) -> IVResult:
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
    if option_type_flag == 1:
        lower, upper = max(0.0, fwd_spot - disc_strike), fwd_spot
    else:
        lower, upper = max(0.0, disc_strike - fwd_spot), disc_strike
    if market_price <= lower:
        return IVResult(nan, False, 0, "at_or_below_intrinsic")
    if market_price >= upper:
        return IVResult(nan, False, 0, "above_upper_bound")

    def objective(sig):
        return black_scholes_price(S, K, T, r, sig, option_type_flag, q) - market_price

    # Brenner-Subrahmanyam initial guess, clamped into the bracket
    sigma = market_price / fwd_spot * math.sqrt(2 * math.pi / T)
    sigma = min(max(sigma, SIGMA_LO), SIGMA_HI)

    for i in range(1, MAX_NEWTON_ITER + 1):
        vega = calculate_vega(S, K, T, r, sigma, q)
        if vega < 1e-8 * S:
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
    return IVResult(sigma, True, info.iterations, "ok")


# ---------------------------------------------------------
# 3. Arbitrage Scanner
# ---------------------------------------------------------
ARB_TOL = 1e-6  # violations smaller than this are float noise, not arbitrage


@dataclass
class Violation:
    type: str  # "monotonicity", "slope", "butterfly" or "calendar"
    expiry: float  # days
    strikes: list
    size: float  # price units


def scan_arbitrage(chain, tol=ARB_TOL) -> list:
    """
    Static no-arbitrage checks on a call chain, one expiry at a time:
    monotonicity and slope bounds in strike, butterfly convexity, and
    calendar (total variance non-decreasing in T at fixed log-moneyness).
    """
    violations = []
    by_expiry = {}
    for opt in chain:
        by_expiry.setdefault(opt["time_to_expiry"], []).append(opt)

    slices = []  # (T, days, strikes, log-forward-moneyness, total variance, quotes)
    for T in sorted(by_expiry):
        calls = sorted(by_expiry[T], key=lambda x: x["strike"])
        days = calls[0]["expiry"]
        r = calls[0]["interest_rate"]
        K = [c["strike"] for c in calls]
        C = [c["price"] for c in calls]

        for i in range(1, len(calls)):
            dK = K[i] - K[i - 1]
            dC = C[i] - C[i - 1]
            # Call price must not increase with strike
            if dC > tol:
                violations.append(Violation("monotonicity", days, [K[i - 1], K[i]], dC))
            # ...and must not fall faster than the discounted strike step
            excess = -dC - math.exp(-r * T) * dK
            if excess > tol:
                violations.append(Violation("slope", days, [K[i - 1], K[i]], excess))

        for i in range(1, len(calls) - 1):
            # Convexity with non-uniform spacing:
            # (K3-K2) C1 - (K3-K1) C2 + (K2-K1) C3 >= 0, scaled to one unit of K2
            w1, w3 = K[i + 1] - K[i], K[i] - K[i - 1]
            fly = (w1 * C[i - 1] + w3 * C[i + 1]) / (w1 + w3) - C[i]
            if -fly > tol:
                violations.append(Violation("butterfly", days, [K[i - 1], K[i], K[i + 1]], -fly))

        k, w, strikes, quotes = [], [], [], []
        for c in calls:
            iv = implied_volatility(c["price"], c["spot"], c["strike"], T, r, 1).sigma
            if not math.isnan(iv):
                k.append(math.log(c["strike"] / (c["spot"] * math.exp(r * T))))
                w.append(iv * iv * T)
                strikes.append(c["strike"])
                quotes.append(c)
        slices.append((T, days, strikes, k, w, quotes))

    # Calendar: on each longer expiry's points, compare with the shorter
    # expiry's total variance interpolated linearly in k (no extrapolation).
    # Size is in price units: the longer call's price at the shorter expiry's
    # total variance, less its quoted price.
    for (_, _, _, k1, w1, _), (T2, days2, strikes2, k2, w2, quotes2) in zip(slices, slices[1:]):
        if len(k1) < 2:
            continue
        for strike, k, w, c in zip(strikes2, k2, w2, quotes2):
            if not k1[0] <= k <= k1[-1]:
                continue
            w_min = float(np.interp(k, k1, w1))
            if w_min <= w:
                continue
            floor = black_scholes_price(c["spot"], strike, T2, c["interest_rate"], math.sqrt(w_min / T2), 1)
            if floor - c["price"] > tol:
                violations.append(Violation("calendar", days2, [strike], floor - c["price"]))

    return violations
