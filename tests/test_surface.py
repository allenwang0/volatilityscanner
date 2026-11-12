import math

import numpy as np
import pytest

from engine import implied_volatility_vec
from simulator import MarketSimulator
from surface import fit_surface, fitted_calendar_crossings, skew_25_delta, surface_grid
from svi import SVIParams


@pytest.fixture(scope="module")
def noiseless():
    sim = MarketSimulator(seed=3, noise_sd=0.0)
    chain = sim.generate_tick()
    iv = implied_volatility_vec(
        chain.price, chain.spot, chain.strike, chain.T, chain.r, chain.flag
    ).sigma
    return sim, fit_surface(chain, iv)


def test_one_fit_per_expiry_that_matches_the_generating_surface(noiseless):
    sim, slices = noiseless
    truth = dict(sim.current_slices())
    assert [s.days for s in slices] == sorted(truth)
    for s in slices:
        assert s.fit.rmse_vol_points < 0.05
        true_atm = truth[s.days].implied_vol(np.array([0.0]), s.T)[0]
        assert s.atm_iv == pytest.approx(true_atm, abs=1e-3)
        expected = skew_25_delta(truth[s.days], s.T, s.k_min, s.k_max)
        assert s.skew_25d == pytest.approx(expected, abs=0.05)


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


def test_skew_survives_a_wild_fit_and_is_nan_when_quotes_miss_25_delta():
    # The 90-day fit seen at IV noise sd 0.002 (seed 7): its call delta never reaches 0.25
    # on [-3, 3], which raised inside brentq and dropped the whole tick. The search now
    # stays inside the quoted strikes.
    wild = SVIParams(a=-0.04159, b=0.90544, rho=0.88201, m=0.26819, s=0.10899)
    assert math.isfinite(skew_25_delta(wild, 90 / 365, -0.13, 0.08))
    calm = SVIParams.from_atm(30 / 365, 0.145, 0.0287, -0.65, 0.01, 0.04)
    assert math.isfinite(skew_25_delta(calm, 30 / 365, -0.12, 0.09))
    assert math.isnan(skew_25_delta(calm, 30 / 365, -0.01, 0.01))
