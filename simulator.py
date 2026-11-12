"""Seeded option chain whose smiles come from an SSVI surface with equity-style skew."""

import math

import numpy as np

from engine import black_scholes_price
from svi import SVIParams, ssvi_slice

EXPIRY_DAYS = (7, 30, 60, 90)
STRIKES = tuple(float(k) for k in range(4500, 5501, 50))
SPOT_STEP_SD = 0.0005  # per-tick log return of spot
# Default quote noise, as an IV standard deviation (0.002 = 0.2 vol points).
# Independent noise per strike breaks call-price convexity at short expiries,
# so the scanner flags real butterfly arbitrage in the quoted prices.
DEFAULT_JITTER = 0.002

# SSVI surface. ATM vol rises from about 13% at 7 days to about 15% at 90 days.
# eta (1 + |rho|) <= 2 for every rho in the clip range, and gamma = 0.5, so each
# tick's true surface is free of butterfly and calendar arbitrage.
ETA, GAMMA = 1.1, 0.5
RHO_MEAN, RHO_RANGE = -0.7, (-0.8, -0.5)
LEVEL_REVERSION, LEVEL_SD = 0.98, 0.01  # AR(1) on log ATM-variance level, per tick
RHO_REVERSION, RHO_SD = 0.95, 0.005


def atm_vol_curve(T: float) -> float:
    return 0.13 + 0.03 * (1 - math.exp(-T / 0.25))


class MarketSimulator:
    def __init__(self, seed=7, jitter=DEFAULT_JITTER, tick_size=None, rate=0.05, dividend_yield=0.0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spot_price = 5000.0  # SPX-like level
        self.r = rate
        self.q = dividend_yield
        self.strikes = STRIKES
        self.expiries = tuple(d / 365 for d in EXPIRY_DAYS)
        self.jitter = jitter
        # Optional quote increment (e.g. 0.05); None keeps full-precision prices
        self.tick_size = tick_size
        self.log_level = 0.0
        self.rho = RHO_MEAN

    def describe(self) -> dict:
        return {"kind": "simulated", "seed": self.seed, "jitter": self.jitter, "tick_size": self.tick_size}

    def true_params(self, T: float) -> SVIParams:
        theta = atm_vol_curve(T) ** 2 * T * math.exp(self.log_level)
        return ssvi_slice(theta, self.rho, ETA, GAMMA)

    def _evolve(self):
        self.spot_price *= math.exp(self.rng.normal(0, SPOT_STEP_SD))
        self.log_level = LEVEL_REVERSION * self.log_level + self.rng.normal(0, LEVEL_SD)
        rho = RHO_MEAN + RHO_REVERSION * (self.rho - RHO_MEAN) + self.rng.normal(0, RHO_SD)
        self.rho = min(max(rho, RHO_RANGE[0]), RHO_RANGE[1])

    def generate_tick(self) -> list[dict]:
        """Advance one tick and return calls and puts for every strike and expiry."""
        self._evolve()
        S = self.spot_price
        chain = []
        for T, days in zip(self.expiries, EXPIRY_DAYS, strict=True):
            params = self.true_params(T)
            forward = S * math.exp((self.r - self.q) * T)
            for K in self.strikes:
                true_iv = float(params.iv(math.log(K / forward), T))
                # The same noisy vol prices both legs, so put-call parity holds
                iv = max(true_iv + self.rng.normal(0, self.jitter), 1e-3) if self.jitter else true_iv
                for kind, flag in (("call", 1), ("put", -1)):
                    price = black_scholes_price(S, K, T, self.r, iv, flag, self.q)
                    if self.tick_size:
                        price = round(price / self.tick_size) * self.tick_size
                    chain.append(
                        {
                            "strike": K,
                            "expiry": days,
                            "time_to_expiry": T,
                            "price": price,
                            "type": kind,
                            "spot": S,
                            "interest_rate": self.r,
                            "dividend_yield": self.q,
                        }
                    )
        return chain
