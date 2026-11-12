"""Seeded synthetic option chain. Every price here is simulated, not market data."""

import math

import numpy as np

from engine import CALL, PUT, Chain, black_scholes_price
from svi import SVIParams

SIM_SPOT = 5000.0
SIM_RATE = 0.05
SIM_STRIKES = np.round(np.linspace(4500, 5500, 20), 2)
SPOT_STEP_SD = 0.0005
IV_NOISE_SD = 0.0002
LEVEL_MEAN_REVERSION = 0.98
LEVEL_STEP_SD = 0.003

# Raw SVI per expiry: ATM vol rises with maturity, rho < 0 gives equity-style put skew.
# Columns: days, ATM vol, b, rho, m, s.
SVI_TABLE = (
    (7, 0.130, 0.10 * math.sqrt(7 / 365), -0.70, 0.000, 0.02),
    (30, 0.145, 0.10 * math.sqrt(30 / 365), -0.65, 0.010, 0.04),
    (60, 0.155, 0.10 * math.sqrt(60 / 365), -0.60, 0.020, 0.06),
    (90, 0.160, 0.10 * math.sqrt(90 / 365), -0.55, 0.020, 0.08),
)


def base_slices() -> list[tuple[float, SVIParams]]:
    return [
        (days, SVIParams.from_atm(days / 365, atm, b, rho, m, s))
        for days, atm, b, rho, m, s in SVI_TABLE
    ]


class MarketSimulator:
    """Spot follows a random walk; the SVI surface scales with a slow mean-reverting level.

    The smile is sticky in log-forward-moneyness. Each quote's IV gets independent noise.
    """

    def __init__(
        self,
        seed: int | None = None,
        tick_size: float | None = None,
        noise_sd: float = IV_NOISE_SD,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.tick_size = tick_size
        self.noise_sd = noise_sd
        self.spot = SIM_SPOT
        self.r = SIM_RATE
        self.level = 0.0
        self.slices = base_slices()
        self.last_model_iv = np.empty(0)

    def current_slices(self) -> list[tuple[float, SVIParams]]:
        scale = math.exp(self.level)
        return [
            (days, SVIParams(p.a * scale, p.b * scale, p.rho, p.m, p.s))
            for days, p in self.slices
        ]

    def generate_tick(self) -> Chain:
        self.spot *= 1 + self.rng.normal(0, SPOT_STEP_SD)
        self.level = LEVEL_MEAN_REVERSION * self.level + self.rng.normal(0, LEVEL_STEP_SD)

        n_strikes = len(SIM_STRIKES)
        days, strike, flag, model_iv = [], [], [], []
        for d, params in self.current_slices():
            T = d / 365
            k = np.log(SIM_STRIKES / (self.spot * math.exp(self.r * T)))
            iv = params.implied_vol(k, T)
            for f in (CALL, PUT):
                days.append(np.full(n_strikes, float(d)))
                strike.append(SIM_STRIKES)
                flag.append(np.full(n_strikes, f))
                model_iv.append(iv)
        days_a = np.concatenate(days)
        strike_a = np.concatenate(strike)
        flag_a = np.concatenate(flag)
        self.last_model_iv = np.concatenate(model_iv)
        noise = self.rng.normal(0, self.noise_sd, n_strikes * len(self.slices))
        quoted_iv = self.last_model_iv + np.repeat(noise.reshape(len(self.slices), 1, -1), 2, 1).ravel()
        T_a = days_a / 365
        price = np.array(
            [
                black_scholes_price(self.spot, K, t, self.r, sigma, int(f))
                for K, t, sigma, f in zip(strike_a, T_a, quoted_iv, flag_a, strict=True)
            ]
        )
        if self.tick_size:
            price = np.round(price / self.tick_size) * self.tick_size
        n = len(price)
        return Chain(
            spot=self.spot,
            r=self.r,
            q=0.0,
            expiry_days=days_a,
            T=T_a,
            strike=strike_a,
            flag=flag_a,
            price=price,
            forward=self.spot * np.exp(self.r * T_a),
            bid=np.full(n, np.nan),
            ask=np.full(n, np.nan),
            source="Simulated data" if self.seed is None else f"Simulated data, seed {self.seed}",
        )
