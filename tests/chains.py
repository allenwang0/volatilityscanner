"""Synthetic test chains built from Black-Scholes with a known smile; no market data."""

from dataclasses import replace

import numpy as np

from engine import CALL, Chain, black_scholes_price, implied_volatility

SPOT = 5000.0
RATE = 0.05
STRIKES = np.round(np.linspace(4500, 5500, 20), 2)
EXPIRY_DAYS = (7, 30, 60, 90)
ATM = 9  # strike 4973.68; the grid has no 5000 strike


def smile(strike: float) -> float:
    return 0.15 + 1.5 * (strike / SPOT - 1) ** 2


def noiseless_call_chain() -> Chain:
    """Free of static arbitrage by construction."""
    days = np.repeat(np.array(EXPIRY_DAYS, dtype=float), len(STRIKES))
    strike = np.tile(STRIKES, len(EXPIRY_DAYS))
    T = days / 365
    price = np.array(
        [black_scholes_price(SPOT, k, t, RATE, smile(k), CALL) for k, t in zip(strike, T)]
    )
    n = len(price)
    return Chain(
        spot=SPOT,
        r=RATE,
        q=0.0,
        expiry_days=days,
        T=T,
        strike=strike,
        flag=np.full(n, CALL),
        price=price,
        forward=SPOT * np.exp(RATE * T),
        bid=np.full(n, np.nan),
        ask=np.full(n, np.nan),
    )


def subset(chain: Chain, keep: np.ndarray) -> Chain:
    arrays = ("expiry_days", "T", "strike", "flag", "price", "forward", "bid", "ask")
    return replace(chain, **{name: getattr(chain, name)[keep] for name in arrays})


def row(chain: Chain, days: int, i: int) -> int:
    """Index of the call at expiry `days` and strike STRIKES[i]."""
    return int(np.flatnonzero((chain.expiry_days == days) & (chain.strike == STRIKES[i]))[0])


def solve_ivs(chain: Chain) -> np.ndarray:
    return np.array(
        [
            implied_volatility(p, chain.spot, k, t, chain.r, int(f), q=chain.q).sigma
            for p, k, t, f in zip(chain.price, chain.strike, chain.T, chain.flag)
        ]
    )
