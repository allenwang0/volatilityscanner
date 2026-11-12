import math

import numpy as np
import pytest

from engine import CALL, PUT, black_scholes_price, implied_volatility
from simulator import MarketSimulator


def test_payload_prices_match_the_payload_strike_and_spot():
    # The solver inverts what the payload says; if the price was computed for a different
    # strike or rounded to cents, every recovered IV is off.
    sim = MarketSimulator(seed=1, noise_sd=0.0)
    chain = sim.generate_tick()
    for K, T, f, price, sigma in zip(chain.strike, chain.T, chain.flag, chain.price, sim.last_model_iv):
        expected = black_scholes_price(chain.spot, K, T, chain.r, sigma, int(f))
        assert price == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_solver_recovers_the_model_iv_of_every_quote():
    sim = MarketSimulator(seed=1, noise_sd=0.0)
    chain = sim.generate_tick()
    for p, K, T, f, sigma in zip(chain.price, chain.strike, chain.T, chain.flag, sim.last_model_iv):
        result = implied_volatility(p, chain.spot, K, T, chain.r, int(f))
        assert result.converged and abs(result.sigma - sigma) < 1e-6


def test_tick_size_rounds_prices_and_the_solver_explains_the_failures():
    chain = MarketSimulator(seed=1, tick_size=0.05, noise_sd=0.0).generate_tick()
    assert np.allclose(chain.price / 0.05, np.round(chain.price / 0.05))
    results = [
        implied_volatility(p, chain.spot, K, T, chain.r, int(f))
        for p, K, T, f in zip(chain.price, chain.strike, chain.T, chain.flag)
    ]
    failed = [r for r in results if not r.converged]
    assert failed, "tick rounding of 7-day wings should produce unsolvable quotes"
    assert all(math.isnan(r.sigma) for r in failed)
    assert {r.reason for r in failed} <= {"zero_price", "at_or_below_intrinsic", "outside_bracket"}
    assert "zero_price" in {r.reason for r in failed}  # deep OTM 7-day options round to 0.00


def test_same_seed_gives_the_same_chain():
    a = MarketSimulator(seed=7).generate_tick()
    b = MarketSimulator(seed=7).generate_tick()
    assert np.array_equal(a.price, b.price)


def test_surface_has_put_skew_and_rising_atm_term_structure():
    # The point of the realistic simulator: smiles that differ by expiry and lean to the put side.
    sim = MarketSimulator(seed=1, noise_sd=0.0)
    sim.generate_tick()
    atm = []
    for days, params in sim.current_slices():
        T = days / 365
        down, zero, up = params.implied_vol(np.array([-0.05, 0.0, 0.05]), T)
        assert down > up
        atm.append(zero)
    assert atm == sorted(atm) and atm[0] < atm[-1]


def test_emits_calls_and_puts_at_every_strike_and_expiry():
    chain = MarketSimulator(seed=1).generate_tick()
    for flag in (CALL, PUT):
        assert (chain.flag == flag).sum() == len(chain) // 2
