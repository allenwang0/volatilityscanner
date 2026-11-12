import math

from hypothesis import given, settings, strategies as st

from engine import black_scholes_price, calculate_vega, implied_volatility

S, R, T7 = 5000.0, 0.05, 7 / 365


def test_deep_itm_price_at_intrinsic_is_nan_not_clamp_floor():
    # 504.31 is the discounted intrinsic to the cent; no volatility prices it,
    # so returning the 0.01 clamp floor would plot a fake 1% vol point.
    res = implied_volatility(504.31, S, 4500.0, T7, R, 1)
    assert math.isnan(res.sigma)
    assert not res.converged
    assert res.reason == "at_or_below_intrinsic"


def test_zero_price_is_nan_not_an_unconverged_iterate():
    # A 0.00 quote carries no volatility information.
    res = implied_volatility(0.0, S, 5500.0, T7, R, 1)
    assert math.isnan(res.sigma)
    assert res.reason == "zero_price"


def test_price_above_upper_bound_is_nan():
    res = implied_volatility(S + 1.0, S, 5000.0, T7, R, 1)
    assert math.isnan(res.sigma)
    assert res.reason == "above_upper_bound"


def test_deep_itm_with_time_value_recovers_sigma():
    # The plan's failing case once the price carries full precision.
    sigma = 0.15 + 1.5 * (4500 / 5000 - 1) ** 2
    price = black_scholes_price(S, 4500.0, T7, R, sigma, 1)
    res = implied_volatility(price, S, 4500.0, T7, R, 1)
    assert res.converged
    assert abs(res.sigma - sigma) < 1e-6


@settings(max_examples=2000, deadline=None)
@given(
    s=st.floats(50, 5000),
    moneyness=st.floats(0.5, 2),
    t=st.floats(1 / 365, 2),
    sigma=st.floats(0.05, 2),
    q=st.floats(0, 0.05),
    flag=st.sampled_from([1, -1]),
)
def test_round_trip_recovers_sigma(s, moneyness, t, sigma, q, flag):
    k = s * moneyness
    vega = calculate_vega(s, k, t, R, sigma, q=q)
    if vega <= 1e-4 * s:
        return  # price is flat in sigma here; sigma is not identifiable
    price = black_scholes_price(s, k, t, R, sigma, flag, q=q)
    res = implied_volatility(price, s, k, t, R, flag, q=q)
    assert res.converged, res
    assert abs(res.sigma - sigma) < 1e-6


def test_put_call_parity_gives_same_iv():
    s, k, t, q, sigma = 5000.0, 5100.0, 0.25, 0.02, 0.2
    call = black_scholes_price(s, k, t, R, sigma, 1, q=q)
    put = call - s * math.exp(-q * t) + k * math.exp(-R * t)
    iv_call = implied_volatility(call, s, k, t, R, 1, q=q).sigma
    iv_put = implied_volatility(put, s, k, t, R, -1, q=q).sigma
    assert abs(iv_call - iv_put) < 1e-8


def test_non_finite_inputs_are_rejected_not_raised():
    # One NaN quote or spot (yfinance returns them) used to make brentq raise,
    # which threw away the whole tick for every client.
    nan = float("nan")
    for args in [(nan, S, 5000.0, T7), (100.0, nan, 5000.0, T7), (100.0, S, 5000.0, nan),
                 (float("inf"), S, 5000.0, T7)]:
        res = implied_volatility(*args, R, 1)
        assert math.isnan(res.sigma)
        assert res.reason == "invalid_input"


@settings(max_examples=1000, deadline=None)
@given(
    s=st.floats(50, 5000),
    moneyness=st.floats(0.5, 2),
    t=st.floats(1 / 365, 2),
    sigma=st.floats(0.05, 2),
    q=st.floats(0, 0.05),
)
def test_call_and_parity_implied_put_agree_across_contracts(s, moneyness, t, sigma, q):
    # Parity ties the two legs together, so any disagreement is solver error.
    k = s * moneyness
    if calculate_vega(s, k, t, R, sigma, q=q) <= 1e-4 * s:
        return
    call = black_scholes_price(s, k, t, R, sigma, 1, q=q)
    put = call - s * math.exp(-q * t) + k * math.exp(-R * t)
    iv_call = implied_volatility(call, s, k, t, R, 1, q=q)
    iv_put = implied_volatility(put, s, k, t, R, -1, q=q)
    assert iv_call.converged and iv_put.converged
    assert abs(iv_call.sigma - iv_put.sigma) < 1e-8
