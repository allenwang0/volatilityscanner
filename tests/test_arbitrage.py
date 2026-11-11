import numpy as np

from engine import black_scholes_price, scan_arbitrage

S, R = 5000.0, 0.05
STRIKES = [round(k, 2) for k in np.linspace(4500, 5500, 20)]
DAYS = [7, 30, 60, 90]
ATM = 9  # strike 4973.68; the grid has no 5000 strike


def clean_chain():
    # Noiseless Black-Scholes chain: free of static arbitrage by construction.
    chain = []
    for days in DAYS:
        T = days / 365
        for K in STRIKES:
            sigma = 0.15 + 1.5 * (K / S - 1) ** 2
            chain.append(
                {
                    "strike": K,
                    "expiry": days,
                    "time_to_expiry": T,
                    "price": black_scholes_price(S, K, T, R, sigma, 1),
                    "type": "call",
                    "spot": S,
                    "interest_rate": R,
                }
            )
    return chain


def row(chain, days, i):
    return chain[DAYS.index(days) * len(STRIKES) + i]


def of_type(violations, kind):
    return [v for v in violations if v.type == kind]


def test_clean_chain_has_no_violations():
    # The old scanner mixed expiries and flagged 60 false violations here.
    assert scan_arbitrage(clean_chain()) == []


def test_injected_butterfly_is_caught_once():
    chain = clean_chain()
    # The clean 30-day ATM butterfly has about 2.57 of convexity per unit of
    # K2 on this grid, so a 2.00 bump is not an arbitrage; 3.00 is.
    row(chain, 30, ATM)["price"] += 3.00
    v = scan_arbitrage(chain)
    assert len(v) == 1
    assert v[0].type == "butterfly"
    assert v[0].expiry == 30
    assert v[0].strikes == [STRIKES[ATM - 1], STRIKES[ATM], STRIKES[ATM + 1]]
    assert 0 < v[0].size < 3.00  # the bump less the original convexity


def test_swapped_prices_give_one_monotonicity_violation():
    chain = clean_chain()
    a, b = row(chain, 60, ATM), row(chain, 60, ATM + 1)
    a["price"], b["price"] = b["price"], a["price"]
    mono = of_type(scan_arbitrage(chain), "monotonicity")
    assert len(mono) == 1
    assert mono[0].expiry == 60
    assert mono[0].strikes == [STRIKES[ATM], STRIKES[ATM + 1]]


def test_slope_steeper_than_discounted_strike_step_is_flagged():
    chain = clean_chain()
    # Make C(K1) - C(K2) exceed e^{-rT}(K2 - K1) by dropping the deep ITM leg.
    row(chain, 90, 0)["price"] += 60.0
    slope = of_type(scan_arbitrage(chain), "slope")
    assert len(slope) == 1
    assert slope[0].strikes == [STRIKES[0], STRIKES[1]]


def test_calendar_total_variance_decrease_is_caught_once():
    chain = clean_chain()
    r60 = row(chain, 60, ATM)
    # sigma 0.10 at 60 days: w = 0.01 * 60/365 < 30-day ATM w = 0.15^2 * 30/365
    r60["price"] = black_scholes_price(S, r60["strike"], 60 / 365, R, 0.10, 1)
    cal = of_type(scan_arbitrage(chain), "calendar")
    assert len(cal) == 1
    assert cal[0].expiry == 60
    assert cal[0].strikes == [STRIKES[ATM]]


def test_calendar_size_is_in_price_units_so_a_price_tolerance_applies():
    # Every check shares one tolerance in dollars (half a tick in live mode).
    # A calendar size in total-variance units (about 2e-4 here) would sit
    # under a 0.005 tolerance and the violation would vanish.
    chain = clean_chain()
    r60 = row(chain, 60, ATM)
    r60["price"] = black_scholes_price(S, r60["strike"], 60 / 365, R, 0.10, 1)
    cal = of_type(scan_arbitrage(chain, tol=0.005), "calendar")
    assert len(cal) == 1
    # Size: how far the 60-day call sits below the cheapest price that keeps
    # its total variance at the 30-day level at the same log-forward moneyness.
    assert 1.0 < cal[0].size < 10.0


def test_calendar_skips_over_an_expiry_with_too_few_quotes():
    # With one quote, the 30-day slice cannot be interpolated. The 60-day
    # slice must then be checked against the 7-day one, not left unchecked.
    chain = [o for o in clean_chain() if o["expiry"] != 30 or o["strike"] == STRIKES[ATM]]
    r60 = next(o for o in chain if o["expiry"] == 60 and o["strike"] == STRIKES[ATM])
    # sigma 0.04 at 60 days: w = 0.0016 * 60/365 < 7-day ATM w = 0.15^2 * 7/365
    r60["price"] = black_scholes_price(S, r60["strike"], 60 / 365, R, 0.04, 1)
    cal = of_type(scan_arbitrage(chain), "calendar")
    assert [(v.expiry, v.strikes) for v in cal] == [(60, [STRIKES[ATM]])]
