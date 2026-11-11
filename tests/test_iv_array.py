import math

import numpy as np

from engine import black_scholes_price, calculate_vega, implied_volatility, implied_volatility_array


def random_chain(n, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.uniform(50, 5000, n)
    K = S * rng.uniform(0.5, 2, n)
    T = rng.uniform(1 / 365, 2, n)
    sigma = rng.uniform(0.05, 2, n)
    q = rng.uniform(0, 0.05, n)
    flag = rng.choice([1, -1], n)
    r = np.full(n, 0.05)
    price = np.array([black_scholes_price(*a) for a in zip(S, K, T, r, sigma, flag, q, strict=True)])
    return price, S, K, T, r, flag, q


def test_vectorized_matches_scalar_to_1e_10():
    # The vectorized path is only a speed-up; any difference from the scalar
    # solver beyond float noise means it is solving a different problem.
    price, S, K, T, r, flag, q = random_chain(2000)
    # Add failure cases: zero, intrinsic, above the upper bound, NaN
    price[:4] = [0.0, max(0, S[1] * math.exp(-q[1] * T[1]) - K[1] * math.exp(-r[1] * T[1])), S[2] * 2, np.nan]
    flag[:4] = 1
    vec = implied_volatility_array(price, S, K, T, r, flag, q)
    for i in range(len(price)):
        res = implied_volatility(price[i], S[i], K[i], T[i], r[i], int(flag[i]), q[i])
        assert vec.reason[i] == res.reason, i
        if res.converged:
            # Where vega is tiny, a one-ulp change in the computed price moves
            # sigma by more than 1e-10, so there both answers only need to
            # reprice the quote to float precision.
            if calculate_vega(S[i], K[i], T[i], r[i], res.sigma, q[i]) > 1e-4 * S[i]:
                assert abs(vec.sigma[i] - res.sigma) < 1e-10, i
            repriced = black_scholes_price(S[i], K[i], T[i], r[i], vec.sigma[i], int(flag[i]), q[i])
            assert abs(repriced - price[i]) < 1e-9 * max(1.0, price[i]), i
        else:
            assert math.isnan(vec.sigma[i])
    assert vec.reason[:4] == ["zero_price", "at_or_below_intrinsic", "above_upper_bound", "invalid_input"]


def test_empty_input():
    vec = implied_volatility_array([], [], [], [], [], [], [])
    assert vec.sigma.size == 0 and vec.reason == []
