import math

import numpy as np

import simulator
from engine import calculate_vega, implied_volatility
from simulator import MarketSimulator


def _noiseless(monkeypatch):
    # Zero out the spot random walk and the IV jitter so the true sigma is known.
    monkeypatch.setattr(simulator.np.random, "normal", lambda loc, scale: 0.0)
    return MarketSimulator()


def test_payload_prices_invert_to_the_simulated_smile(monkeypatch):
    # The solver only sees (strike, price) from the payload, so the price must
    # belong to that exact strike at full precision, or the surface is wrong.
    sim = _noiseless(monkeypatch)
    chain = sim.generate_tick()
    checked = 0
    for opt in chain:
        S, K, T = opt["spot"], opt["strike"], opt["time_to_expiry"]
        true_sigma = sim.base_vol + 1.5 * (K / S - 1) ** 2
        if calculate_vega(S, K, T, sim.r, true_sigma) <= 1e-4 * S:
            continue
        res = implied_volatility(opt["price"], S, K, T, sim.r, 1)
        assert res.converged, (K, T, res)
        assert abs(res.sigma - true_sigma) < 1e-6, (K, T)
        checked += 1
    assert checked >= 60


def test_tick_size_rounds_prices_and_failures_are_explained(monkeypatch):
    monkeypatch.setattr(simulator.np.random, "normal", lambda loc, scale: 0.0)
    sim = MarketSimulator(tick_size=0.05)
    chain = sim.generate_tick()
    for opt in chain:
        assert abs(opt["price"] / 0.05 - round(opt["price"] / 0.05)) < 1e-9
    reasons = {
        implied_volatility(o["price"], o["spot"], o["strike"], o["time_to_expiry"], sim.r, 1).reason
        for o in chain
    }
    assert "zero_price" in reasons  # deep OTM 7-day calls round to 0.00
    assert reasons <= {"ok", "zero_price", "at_or_below_intrinsic"}
