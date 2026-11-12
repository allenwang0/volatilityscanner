import math

import numpy as np

from engine import (
    CALL,
    PUT,
    black_scholes_price,
    calculate_vega,
    implied_volatility,
    implied_volatility_vec,
)


def random_contracts(n, seed):
    rng = np.random.default_rng(seed)
    S = np.full(n, 5000.0)
    K = S * rng.uniform(0.5, 2.0, n)
    T = rng.uniform(1 / 365, 2.0, n)
    sigma = rng.uniform(0.05, 2.0, n)
    flag = rng.choice([CALL, PUT], n)
    price = np.array(
        [
            black_scholes_price(5000.0, k, t, 0.05, s, int(f))
            for k, t, s, f in zip(K, T, sigma, flag)
        ]
    )
    return price, S, K, T, flag


def test_vectorized_matches_scalar_to_1e10_including_failures():
    # The vectorized path is only worth having if it gives the scalar answer. Where
    # vega < 1e-4 * S the root is ill-conditioned (a 1e-15 price change moves sigma by
    # 1e-9), so sigma is compared only where it is identifiable; reasons must always match.
    price, S, K, T, flag = random_contracts(3000, seed=0)
    price[:4] = [0.0, float("nan"), 1e9, 1e-300]  # zero, invalid, above bound, at intrinsic
    batch = implied_volatility_vec(price, S, K, T, 0.05, flag)
    for j in range(len(price)):
        res = implied_volatility(price[j], S[j], K[j], T[j], 0.05, int(flag[j]))
        assert batch.reason[j] == res.reason, j
        if math.isnan(res.sigma):
            assert math.isnan(batch.sigma[j])
        elif calculate_vega(S[j], K[j], T[j], 0.05, res.sigma) > 1e-4 * S[j]:
            assert abs(batch.sigma[j] - res.sigma) < 1e-10, j


def test_some_contracts_need_the_brent_fallback():
    price, S, K, T, flag = random_contracts(3000, seed=0)
    batch = implied_volatility_vec(price, S, K, T, 0.05, flag)
    assert 0 < batch.brent_fallback.sum() < len(price)
