import numpy as np
import pytest

from engine import implied_volatility_vec
from simulator import MarketSimulator
from surface import fit_surface, fitted_calendar_crossings, skew_25_delta, surface_grid


@pytest.fixture(scope="module")
def noiseless():
    sim = MarketSimulator(seed=3, noise_sd=0.0)
    chain = sim.generate_tick()
    iv = implied_volatility_vec(chain.price, chain.spot, chain.strike, chain.T, chain.r, chain.flag).sigma
    return sim, fit_surface(chain, iv)


def test_one_fit_per_expiry_that_matches_the_generating_surface(noiseless):
    sim, slices = noiseless
    truth = dict(sim.current_slices())
    assert [s.days for s in slices] == sorted(truth)
    for s in slices:
        assert s.fit.rmse_vol_points < 0.05
        true_atm = truth[s.days].implied_vol(np.array([0.0]), s.T)[0]
        assert s.atm_iv == pytest.approx(true_atm, abs=1e-3)
        assert s.skew_25d == pytest.approx(skew_25_delta(truth[s.days], s.T), abs=0.05)


def test_term_structure_rises_and_skew_favours_puts(noiseless):
    _, slices = noiseless
    atm = [s.atm_iv for s in slices]
    assert atm == sorted(atm)
    assert all(s.skew_25d > 0 for s in slices)
    assert fitted_calendar_crossings(slices) == []


def test_surface_grid_has_one_row_per_expiry(noiseless):
    _, slices = noiseless
    grid = surface_grid(slices)
    assert len(grid["iv"]) == len(slices) == len(grid["days"])
    assert all(len(row) == len(grid["k"]) for row in grid["iv"])
