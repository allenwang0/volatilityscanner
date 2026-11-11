"""Black-Scholes pricing, implied volatility, and static-arbitrage checks."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

# Implied volatility search bracket. 1e-4 is below any listed equity option's
# vol; 5.0 (500%) is above it.
SIGMA_LO, SIGMA_HI = 1e-4, 5.0
MAX_NEWTON_ITER = 50
SIGMA_TOL = 1e-12  # Newton stops when a step moves sigma less than this; brentq xtol
FLAT_VEGA = 1e-8  # vega below FLAT_VEGA * S makes a Newton step meaningless; use Brent
ARB_TOL = 1e-6  # simulated mode: violations smaller than this (price units) are float noise


def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))


def standard_normal_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type_flag: int, q: float = 0.0
) -> float:
    """European option price. option_type_flag: 1 call, -1 put; q: continuous dividend yield."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) * option_type_flag)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    fwd_spot = S * math.exp(-q * T)
    disc_strike = K * math.exp(-r * T)
    if option_type_flag == 1:
        return fwd_spot * standard_normal_cdf(d1) - disc_strike * standard_normal_cdf(d2)
    return disc_strike * standard_normal_cdf(-d2) - fwd_spot * standard_normal_cdf(-d1)


def calculate_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * standard_normal_pdf(d1)


@dataclass
class IVResult:
    sigma: float  # NaN when not converged
    converged: bool
    iterations: int
    reason: str  # "ok" or why no volatility was found


def _price_bounds(S: float, K: float, T: float, r: float, flag: int, q: float) -> tuple[float, float]:
    """Static no-arbitrage bounds (lower, upper) on a European option price."""
    fwd_spot = S * math.exp(-q * T)
    disc_strike = K * math.exp(-r * T)
    if flag == 1:
        return max(0.0, fwd_spot - disc_strike), fwd_spot
    return max(0.0, disc_strike - fwd_spot), disc_strike


def _brent(market_price: float, S: float, K: float, T: float, r: float, flag: int, q: float) -> IVResult:
    def objective(sig: float) -> float:
        return black_scholes_price(S, K, T, r, sig, flag, q) - market_price

    if objective(SIGMA_LO) * objective(SIGMA_HI) > 0:
        return IVResult(float("nan"), False, 0, "outside_bracket")
    # brentq raises if it fails to converge; on a valid bracket it does not
    sigma, info = brentq(objective, SIGMA_LO, SIGMA_HI, xtol=SIGMA_TOL, full_output=True)
    return IVResult(sigma, True, info.iterations, "ok")


def _screen(market_price: float, S: float, K: float, T: float, r: float, flag: int, q: float) -> str:
    """Reason a quote has no implied volatility, or "" if the solver should run."""
    if not all(math.isfinite(x) for x in (market_price, S, K, T, r, q)):
        return "invalid_input"
    if T <= 0 or S <= 0 or K <= 0:
        return "invalid_input"
    if market_price <= 0:
        return "zero_price"
    lower, upper = _price_bounds(S, K, T, r, flag, q)
    if market_price <= lower:
        return "at_or_below_intrinsic"
    if market_price >= upper:
        return "above_upper_bound"
    return ""


def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type_flag: int, q: float = 0.0
) -> IVResult:
    """Bounds check, Brenner-Subrahmanyam guess, Newton in a bracket, Brent fallback."""
    reason = _screen(market_price, S, K, T, r, option_type_flag, q)
    if reason:
        return IVResult(float("nan"), False, 0, reason)

    sigma = market_price / (S * math.exp(-q * T)) * math.sqrt(2 * math.pi / T)
    sigma = min(max(sigma, SIGMA_LO), SIGMA_HI)

    for i in range(1, MAX_NEWTON_ITER + 1):
        vega = calculate_vega(S, K, T, r, sigma, q)
        if vega < FLAT_VEGA * S:
            break
        step = (black_scholes_price(S, K, T, r, sigma, option_type_flag, q) - market_price) / vega
        new_sigma = sigma - step
        if not SIGMA_LO <= new_sigma <= SIGMA_HI:
            break
        sigma = new_sigma
        if abs(step) < SIGMA_TOL:
            return IVResult(sigma, True, i, "ok")

    # Newton left the bracket, stalled on a flat vega, or ran out of steps
    return _brent(market_price, S, K, T, r, option_type_flag, q)


@dataclass
class Violation:
    type: str  # "monotonicity", "slope", "butterfly", "calendar" or "parity"
    expiry: float  # days
    strikes: list[float]
    size: float  # price units
    exceeds_spread: bool | None = None  # live quotes only: size > the legs' summed half-spreads


def _half_spread(quotes: list[dict]) -> float | None:
    if not all("bid" in q and "ask" in q for q in quotes):
        return None
    return sum((q["ask"] - q["bid"]) / 2 for q in quotes)


def _violation(kind: str, days: float, legs: list[dict], size: float) -> Violation:
    half = _half_spread(legs)
    return Violation(kind, days, [leg["strike"] for leg in legs], size, None if half is None else size > half)


def scan_arbitrage(chain: list[dict], tol: float = ARB_TOL) -> list[Violation]:
    return scan_arbitrage_with_counts(chain, tol)[0]


def scan_arbitrage_with_counts(chain: list[dict], tol: float = ARB_TOL) -> tuple[list[Violation], int]:
    """Static no-arbitrage checks on a chain, one expiry at a time.

    Calls: price non-increasing in strike, slope no steeper than -e^{-rT},
    butterfly convexity, and calendar (total variance non-decreasing in T at
    fixed log-forward moneyness). Calls and puts at the same strike: put-call
    parity. Mid prices are used; a quote dict may carry bid and ask.
    Returns the violations and the number of conditions checked.
    """
    violations: list[Violation] = []
    checked = 0
    by_expiry: dict[float, list[dict]] = {}
    for opt in chain:
        by_expiry.setdefault(opt["time_to_expiry"], []).append(opt)

    slices = []  # (T, days, log-forward moneyness, total variance, calls), solved calls only
    for T in sorted(by_expiry):
        quotes = by_expiry[T]
        calls = sorted((o for o in quotes if o.get("type", "call") == "call"), key=lambda x: x["strike"])
        puts = {o["strike"]: o for o in quotes if o.get("type") == "put"}
        days = quotes[0]["expiry"]
        r = quotes[0]["interest_rate"]
        q = quotes[0].get("dividend_yield", 0.0)
        K = [c["strike"] for c in calls]
        C = [c["price"] for c in calls]

        for i in range(1, len(calls)):
            dK = K[i] - K[i - 1]
            dC = C[i] - C[i - 1]
            legs = calls[i - 1 : i + 1]
            checked += 2
            # Call price must not increase with strike
            if dC > tol:
                violations.append(_violation("monotonicity", days, legs, dC))
            # ...and must not fall faster than the discounted strike step
            excess = -dC - math.exp(-r * T) * dK
            if excess > tol:
                violations.append(_violation("slope", days, legs, excess))

        for i in range(1, len(calls) - 1):
            # Convexity with non-uniform spacing:
            # (K3-K2) C1 - (K3-K1) C2 + (K2-K1) C3 >= 0, scaled to one unit of K2
            w1, w3 = K[i + 1] - K[i], K[i] - K[i - 1]
            fly = (w1 * C[i - 1] + w3 * C[i + 1]) / (w1 + w3) - C[i]
            checked += 1
            if -fly > tol:
                violations.append(_violation("butterfly", days, calls[i - 1 : i + 2], -fly))

        for c in calls:
            p = puts.get(c["strike"])
            if p is None:
                continue
            # C - P = S e^{-qT} - K e^{-rT}
            gap = abs(
                c["price"] - p["price"] - (c["spot"] * math.exp(-q * T) - c["strike"] * math.exp(-r * T))
            )
            checked += 1
            if gap > tol:
                violations.append(_violation("parity", days, [c, p], gap))

        k, w, solved = [], [], []
        for c in calls:
            iv = implied_volatility(c["price"], c["spot"], c["strike"], T, r, 1, q).sigma
            if not math.isnan(iv):
                k.append(math.log(c["strike"] / (c["spot"] * math.exp((r - q) * T))))
                w.append(iv * iv * T)
                solved.append(c)
        slices.append((T, days, k, w, solved))

    # Calendar: on each longer expiry's points, compare with the shorter
    # expiry's total variance interpolated linearly in k (no extrapolation).
    # Size is in price units: the longer call's price at the shorter expiry's
    # total variance, less its quoted price.
    # An expiry with fewer than two solved quotes cannot be interpolated, so
    # later expiries are compared with the last one that can.
    base = None
    for T2, days2, k2, w2, calls2 in slices:
        if base is None:
            if len(k2) >= 2:
                base = (k2, w2)
            continue
        k1, w1 = base
        if len(k2) >= 2:
            base = (k2, w2)
        for k, w, c in zip(k2, w2, calls2, strict=True):
            if not k1[0] <= k <= k1[-1]:
                continue
            checked += 1
            w_min = float(np.interp(k, k1, w1))
            if w_min <= w:
                continue
            q = c.get("dividend_yield", 0.0)
            floor = black_scholes_price(
                c["spot"], c["strike"], T2, c["interest_rate"], math.sqrt(w_min / T2), 1, q
            )
            if floor - c["price"] > tol:
                violations.append(_violation("calendar", days2, [c], floor - c["price"]))

    return violations, checked
