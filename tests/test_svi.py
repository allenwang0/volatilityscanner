import math

import numpy as np
import pytest

from engine import calculate_vega
from simulator import STRIKES, MarketSimulator
from svi import DENSITY_GRID, SVIParams, atm_vol, calendar_crossings, density_g, fit_svi, skew_25d

SIM = MarketSimulator(seed=7)
S, R = SIM.spot_price, SIM.r
T30 = 30 / 365
TRUE30 = SIM.true_params(T30)
K = np.array(STRIKES)
k = np.log(K / (S * math.exp(R * T30)))

# Tolerances for the noisy fit (IV noise sd 0.002, 21 strikes over +-10%).
# rho, m and s trade off against each other on a strike range this narrow,
# so their tolerances are wide; the fitted smile itself is tight.
PARAM_TOL = {"a": 2e-3, "b": 0.015, "rho": 0.5, "m": 0.02, "s": 0.04}


def noisy_fit(seed, mask=slice(None)):
    rng = np.random.default_rng(seed)
    iv = TRUE30.iv(k, T30) + rng.normal(0, 0.002, k.size)
    vega = np.array([calculate_vega(S, kk, T30, R, v) for kk, v in zip(K, iv, strict=True)])
    return fit_svi(k[mask], iv[mask], T30, weights=vega[mask]), iv


def test_noiseless_smile_recovers_parameters_exactly():
    fit = fit_svi(k, TRUE30.iv(k, T30), T30)
    for name, value in TRUE30.as_dict().items():
        assert getattr(fit.params, name) == pytest.approx(value, abs=1e-5), name
    assert fit.rmse_vol_pts < 1e-4


@pytest.mark.parametrize("seed", range(20))
def test_noisy_smile_fit(seed):
    fit, _ = noisy_fit(seed)
    # 0.2 vol points of noise: a good fit leaves about that much residual.
    assert fit.rmse_vol_pts < 0.5
    for name, tol in PARAM_TOL.items():
        assert abs(getattr(fit.params, name) - getattr(TRUE30, name)) < tol, name
    # What the dashboard shows: the smile, its ATM level and 25-delta skew.
    # Wing quotes carry little vega weight, so the far wings get more room.
    err = np.abs(fit.params.iv(k, T30) - TRUE30.iv(k, T30)) * 100
    assert err[np.abs(k) <= 0.05].max() < 0.4
    assert err.max() < 1.0
    assert abs(atm_vol(fit.params, T30) - atm_vol(TRUE30, T30)) * 100 < 0.3
    assert abs(skew_25d(fit.params, T30) - skew_25d(TRUE30, T30)) < 1.0


@pytest.mark.parametrize("seed", range(5))
def test_held_out_strikes(seed):
    # Fit on two strikes in three, score on the third: a fit that only
    # memorizes its own points would do worse here than in-sample.
    held = np.arange(k.size) % 3 == 0
    fit, iv = noisy_fit(seed, ~held)
    rmse = np.sqrt(np.mean((fit.params.iv(k[held], T30) - iv[held]) ** 2)) * 100
    assert rmse < 0.5


def test_simulated_surface_is_free_of_static_arbitrage():
    slices = [(T, SIM.true_params(T)) for T in SIM.expiries]
    for _, p in slices:
        assert density_g(p, DENSITY_GRID).min() > 0
    assert calendar_crossings(slices) == []


def test_density_check_flags_a_known_arbitrageable_smile():
    # Axel Vogt's raw SVI example (T = 1), the standard case of an SVI slice
    # with butterfly arbitrage (Gatheral and Jacquier 2014).
    vogt = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, s=0.4153)
    assert density_g(vogt, DENSITY_GRID).min() < 0
    fit = fit_svi(DENSITY_GRID, vogt.iv(DENSITY_GRID, 1.0), 1.0)
    assert fit.butterfly_k  # the fit reproduces the arbitrage and reports it


def test_calendar_crossing_is_reported_when_expiries_are_swapped():
    p30, p90 = SIM.true_params(30 / 365), SIM.true_params(90 / 365)
    crossings = calendar_crossings([(30 / 365, p90), (90 / 365, p30)])
    assert len(crossings) == 1


def test_negative_rho_gives_put_skew():
    assert skew_25d(TRUE30, T30) > 3.0  # equity-style: OTM puts richer than calls
    flipped = SVIParams(TRUE30.a, TRUE30.b, -TRUE30.rho, -TRUE30.m, TRUE30.s)
    assert skew_25d(flipped, T30) < 0
