import math
from dataclasses import dataclass

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
def scan_arbitrage(df):
    """
    Simple scanner to flag violations of monotonicity.
    Call prices should generally decrease as Strike increases.
    """
    anomalies = []
    # Sort by strike
    calls = sorted(df, key=lambda x: x['strike'])

    for i in range(1, len(calls)):
        prev = calls[i - 1]
        curr = calls[i]

        # Violation: Higher Strike Call costs MORE than Lower Strike Call
        if curr['price'] > prev['price']:
            anomalies.append({
                "type": "Monotonicity Violation",
                "strike": curr['strike'],
                "price": curr['price'],
                "details": f"Strike {curr['strike']} > Strike {prev['strike']}"
            })

    return anomalies