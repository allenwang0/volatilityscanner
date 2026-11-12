import math

import numpy as np
import pytest

from engine import calculate_vega
from svi import SVIParams, calendar_violations, fit_svi

S, R, NOISE_SD = 5000.0, 0.05, 0.002

# Known equity-style slices: negative rho, ATM vol rising with maturity.
TRUE = [
    (d, SVIParams.from_atm(d / 365, atm, 0.1 * math.sqrt(d / 365), rho, m, s))
    for d, atm, rho, m, s in (
        (7, 0.130, -0.70, 0.00, 0.02),
        (30, 0.145, -0.65, 0.01, 0.04),
        (60, 0.155, -0.60, 0.02, 0.06),
        (90, 0.160, -0.55, 0.02, 0.08),
    )
]

# Largest error seen over 20 seeds was a 1e-3, b 3e-3, rho 0.10, m 8e-3, s 0.017;
# these tolerances leave about 1.5x headroom. rho trades off against b and m on a
# finite strike range, so it is the loosest.
TOLERANCE = {"a": 0.002, "b": 0.005, "rho": 0.15, "m": 0.015, "s": 0.025}


def noisy_slice(params, days, seed):
    """41 strikes from 5 standard deviations below the forward to 3 above, IV noise sd 0.002."""
    rng = np.random.default_rng(seed)
    T = days / 365
    sd = params.implied_vol(np.array([0.0]), T)[0] * math.sqrt(T)
    k = np.linspace(-5 * sd, 3 * sd, 41)
    iv = params.implied_vol(k, T) + rng.normal(0, NOISE_SD, len(k))
    strikes = S * math.exp(R * T) * np.exp(k)
    vega = np.array([calculate_vega(S, K, T, R, s) for K, s in zip(strikes, iv)])
    return k, iv, T, vega


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("days, true", TRUE)
def test_fit_recovers_known_parameters_and_fits_within_half_a_vol_point(days, true, seed):
    k, iv, T, vega = noisy_slice(true, days, seed)
    fit = fit_svi(k, iv, T, vega)
    for name, tol in TOLERANCE.items():
        assert abs(getattr(fit.params, name) - getattr(true, name)) < tol, name
    assert fit.rmse_vol_points < 0.5
    assert fit.min_density_g >= 0


@pytest.mark.parametrize("days, true", TRUE)
def test_held_out_strikes_are_predicted_within_half_a_vol_point(days, true):
    # Fit on two thirds of the strikes; the left-out third measures interpolation error.
    k, iv, T, vega = noisy_slice(true, days, seed=0)
    train = np.arange(len(k)) % 3 != 0
    fit = fit_svi(k[train], iv[train], T, vega[train])
    held_out = np.sqrt(np.mean((fit.params.implied_vol(k[~train], T) - iv[~train]) ** 2)) * 100
    assert held_out < 0.5


def test_failed_solves_are_left_out_of_the_fit():
    days, true = TRUE[1]
    k, iv, T, vega = noisy_slice(true, days, seed=0)
    iv[[0, 5]] = np.nan
    fit = fit_svi(k, iv, T, vega)
    assert math.isfinite(fit.rmse_vol_points) and fit.rmse_vol_points < 0.5


def test_density_condition_flags_a_smile_with_butterfly_arbitrage():
    # Axel Vogt's example from Gatheral and Jacquier (2014), "Arbitrage-free SVI
    # volatility surfaces": a valid-looking slice whose density goes negative.
    bad = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, s=0.4153)
    assert bad.density_g(np.linspace(-1.5, 1.5, 201)).min() < 0
    _, good = TRUE[1]
    assert good.density_g(np.linspace(-0.5, 0.5, 201)).min() >= 0


def test_calendar_check_flags_crossing_slices_only():
    grid = np.linspace(-0.3, 0.3, 201)
    assert calendar_violations(TRUE, grid) == []
    swapped = [(30, TRUE[2][1]), (60, TRUE[1][1])]
    assert calendar_violations(swapped, grid) == [(30, 60)]
