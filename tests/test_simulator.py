import math

from engine import calculate_vega, implied_volatility, scan_arbitrage
from simulator import EXPIRY_DAYS, MarketSimulator
from svi import atm_vol, skew_25d


def test_noiseless_quotes_invert_to_the_true_smile():
    # The solver only sees (strike, price) from the payload, so the price must
    # belong to that exact strike at full precision, or the smile is wrong.
    sim = MarketSimulator(seed=1, jitter=0)
    chain = sim.generate_tick()
    checked = 0
    for opt in chain:
        S, K, T = opt["spot"], opt["strike"], opt["time_to_expiry"]
        true_iv = float(sim.true_params(T).iv(math.log(K / (S * math.exp(sim.r * T))), T))
        if calculate_vega(S, K, T, sim.r, true_iv) <= 1e-4 * S:
            continue
        flag = 1 if opt["type"] == "call" else -1
        res = implied_volatility(opt["price"], S, K, T, sim.r, flag)
        assert res.converged, (K, T, res)
        assert abs(res.sigma - true_iv) < 1e-6, (K, T)
        checked += 1
    assert checked >= 120


def test_chain_has_calls_and_puts_for_every_strike_and_expiry():
    sim = MarketSimulator()
    chain = sim.generate_tick()
    assert len(chain) == 2 * len(sim.strikes) * len(EXPIRY_DAYS)
    assert {o["type"] for o in chain} == {"call", "put"}


def test_same_seed_same_ticks():
    # --seed must make a run reproducible, including the report's numbers.
    a, b = MarketSimulator(seed=3), MarketSimulator(seed=3)
    for _ in range(3):
        assert a.generate_tick() == b.generate_tick()
    assert MarketSimulator(seed=4).generate_tick() != MarketSimulator(seed=3).generate_tick()


def test_surface_has_equity_skew_and_rising_term_structure():
    sim = MarketSimulator()
    atm = [atm_vol(sim.true_params(T), T) for T in sim.expiries]
    assert atm == sorted(atm)
    assert all(skew_25d(sim.true_params(T), T) > 3.0 for T in sim.expiries)


def test_noiseless_chain_has_no_arbitrage_flags():
    sim = MarketSimulator(seed=7, jitter=0)
    for _ in range(5):
        assert scan_arbitrage(sim.generate_tick()) == []


def test_default_jitter_produces_real_butterfly_flags():
    # Independent IV noise per strike makes some quoted call prices
    # non-convex in strike. Those quotes are arbitrageable as quoted, so the
    # flags are correct, not false positives. Parity still holds because each
    # strike's call and put share the noisy vol.
    sim = MarketSimulator(seed=7)
    kinds = set()
    for _ in range(5):
        kinds |= {v.type for v in scan_arbitrage(sim.generate_tick())}
    assert "butterfly" in kinds
    assert "parity" not in kinds


def test_tick_size_rounds_prices_and_failures_are_explained():
    sim = MarketSimulator(jitter=0, tick_size=0.05)
    chain = sim.generate_tick()
    for opt in chain:
        assert abs(opt["price"] / 0.05 - round(opt["price"] / 0.05)) < 1e-9
    reasons = {
        implied_volatility(
            o["price"], o["spot"], o["strike"], o["time_to_expiry"], sim.r, 1 if o["type"] == "call" else -1
        ).reason
        for o in chain
    }
    assert "zero_price" in reasons  # deep OTM 7-day options round to 0.00
    assert reasons <= {"ok", "zero_price", "at_or_below_intrinsic", "outside_bracket"}
