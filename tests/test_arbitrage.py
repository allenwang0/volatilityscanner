import math

import numpy as np

from arbitrage import check_static_arbitrage
from engine import CALL, PUT, Chain, black_scholes_price
from tests.chains import ATM, RATE, SPOT, STRIKES, noiseless_call_chain, row, solve_ivs, subset


def scan(chain, tol=1e-6):
    return check_static_arbitrage(chain, solve_ivs(chain), tol).violations


def of_type(violations, kind):
    return [v for v in violations if v.type == kind]


def test_clean_chain_has_no_violations():
    # The old scanner sorted all four expiries together by strike and flagged
    # 60 false violations here.
    chain = noiseless_call_chain()
    result = check_static_arbitrage(chain, solve_ivs(chain))
    assert result.violations == []
    assert result.conditions_checked > 0 and result.quotes_checked == len(chain)


def test_injected_butterfly_is_caught_once():
    chain = noiseless_call_chain()
    # The clean 30-day ATM butterfly has about 2.57 of convexity per unit of
    # K2 on this grid, so a 2.00 bump is not an arbitrage; 3.00 is.
    chain.price[row(chain, 30, ATM)] += 3.00
    v = scan(chain)
    assert len(v) == 1
    assert v[0].type == "butterfly"
    assert v[0].expiry_days == 30
    assert v[0].strikes == (STRIKES[ATM - 1], STRIKES[ATM], STRIKES[ATM + 1])
    assert 0 < v[0].size < 3.00  # the bump less the original convexity


def test_swapped_prices_give_one_monotonicity_violation():
    chain = noiseless_call_chain()
    a, b = row(chain, 60, ATM), row(chain, 60, ATM + 1)
    chain.price[[a, b]] = chain.price[[b, a]]
    mono = of_type(scan(chain), "monotonicity")
    assert len(mono) == 1
    assert mono[0].expiry_days == 60
    assert mono[0].strikes == (STRIKES[ATM], STRIKES[ATM + 1])


def test_slope_steeper_than_discounted_strike_step_is_flagged():
    chain = noiseless_call_chain()
    # Make C(K1) - C(K2) exceed e^{-rT}(K2 - K1) by raising the deep ITM leg.
    chain.price[row(chain, 90, 0)] += 60.0
    slope = of_type(scan(chain), "slope")
    assert len(slope) == 1
    assert slope[0].strikes == (STRIKES[0], STRIKES[1])


def _drop_60_day_atm_vol(chain, sigma):
    i = row(chain, 60, ATM)
    chain.price[i] = black_scholes_price(SPOT, STRIKES[ATM], 60 / 365, RATE, sigma, CALL)


def test_calendar_total_variance_decrease_is_caught_once():
    chain = noiseless_call_chain()
    # sigma 0.10 at 60 days: w = 0.01 * 60/365 < 30-day ATM w = 0.15^2 * 30/365
    _drop_60_day_atm_vol(chain, 0.10)
    cal = of_type(scan(chain), "calendar")
    assert len(cal) == 1
    assert cal[0].expiry_days == 60
    assert cal[0].strikes == (STRIKES[ATM],)


def test_calendar_size_is_in_price_units_so_a_price_tolerance_applies():
    # Every check shares one tolerance in dollars (half a tick in live mode).
    # A size in total-variance units (about 2e-4 here) would sit under 0.005
    # and the violation would vanish.
    chain = noiseless_call_chain()
    _drop_60_day_atm_vol(chain, 0.10)
    cal = of_type(scan(chain, tol=0.005), "calendar")
    assert len(cal) == 1
    assert 1.0 < cal[0].size < 10.0


def test_calendar_skips_over_an_expiry_with_too_few_quotes():
    # With one quote, the 30-day slice cannot be interpolated. The 60-day
    # slice must then be checked against the 7-day one, not left unchecked.
    full = noiseless_call_chain()
    chain = subset(full, (full.expiry_days != 30) | (full.strike == STRIKES[ATM]))
    # sigma 0.04 at 60 days: w = 0.0016 * 60/365 < 7-day ATM w = 0.15^2 * 7/365
    _drop_60_day_atm_vol(chain, 0.04)
    cal = of_type(scan(chain), "calendar")
    assert [(v.expiry_days, v.strikes) for v in cal] == [(60, (STRIKES[ATM],))]


def test_parity_break_is_sized_and_compared_with_the_spread():
    # Live chains carry both legs; a put 0.50 rich against parity, quoted 0.20
    # wide on each leg, is a violation that survives crossing the spread.
    calls = noiseless_call_chain()
    calls = subset(calls, calls.expiry_days == 30)
    puts = calls.price - math.exp(-RATE * 30 / 365) * (calls.forward - calls.strike)
    puts[5] += 0.50
    prices = np.concatenate([calls.price, puts])
    n = len(calls)
    both = Chain(
        spot=SPOT,
        r=RATE,
        q=0.0,
        expiry_days=np.tile(calls.expiry_days, 2),
        T=np.tile(calls.T, 2),
        strike=np.tile(calls.strike, 2),
        flag=np.concatenate([np.full(n, CALL), np.full(n, PUT)]),
        price=prices,
        forward=np.tile(calls.forward, 2),
        bid=prices - 0.10,
        ask=prices + 0.10,
    )
    parity = of_type(scan(both, tol=0.005), "put_call_parity")
    assert len(parity) == 1
    assert parity[0].strikes == (STRIKES[5],)
    assert math.isclose(parity[0].size, 0.50, abs_tol=1e-9)
    assert parity[0].exceeds_spread is True
