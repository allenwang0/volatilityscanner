import math


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
def black_scholes_price(S, K, T, r, sigma, option_type_flag):
    # option_type_flag: 1 for Call, -1 for Put
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) * option_type_flag)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type_flag == 1:  # Call
        price = S * standard_normal_cdf(d1) - K * math.exp(-r * T) * standard_normal_cdf(d2)
    else:  # Put
        price = K * math.exp(-r * T) * standard_normal_cdf(-d2) - S * standard_normal_cdf(-d1)
    return price


# REMOVED: @jit(nopython=True)
def calculate_vega(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * standard_normal_pdf(d1)


# ---------------------------------------------------------
# 2. Implied Volatility Solver (Newton-Raphson)
# ---------------------------------------------------------
# REMOVED: @jit(nopython=True)
def implied_volatility(market_price, S, K, T, r, option_type_flag):
    sigma = 0.5  # Initial guess
    tol = 1e-5
    max_iter = 100

    for i in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type_flag)
        diff = market_price - price

        if abs(diff) < tol:
            return sigma

        vega = calculate_vega(S, K, T, r, sigma)

        if vega == 0:
            return sigma  # Prevent divide by zero, return best guess

        sigma = sigma + diff / vega

        # Clamp sigma to realistic bounds to prevent divergence
        if sigma <= 0: sigma = 0.01
        if sigma > 5: sigma = 5.0

    return sigma


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