import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from analysis import analyze_chain
from engine import black_scholes_price

NY = ZoneInfo("America/New_York")


class FakeColumn:
    def __init__(self, values):
        self.iloc = values


class FakeCalls:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return enumerate(self.rows)


def bs_row(strike, flag, spread=0.10):
    # Quotes priced from Black-Scholes with q = 1.3% so the implied carry is known.
    mid = black_scholes_price(500.0, strike, T_FAKE, 0.04, 0.2, flag, q=0.013)
    return {"strike": strike, "bid": mid - spread / 2, "ask": mid + spread / 2}


T_FAKE = None  # set by the fixture from the frozen expiry


class FakeTicker:
    """Offline stand-in for yfinance.Ticker with one expiry far in the future."""

    options = ["2099-01-16"]

    def __init__(self, symbol):
        pass

    def history(self, period):
        return {"Close": FakeColumn([500.0])}

    def option_chain(self, expiry):
        calls = [bs_row(k, 1) for k in (480.0, 490.0, 500.0, 510.0, 520.0)]
        calls += [
            {"strike": 495.0, "bid": 0.0, "ask": 6.0},  # no bid
            {"strike": 505.0, "bid": 3.0, "ask": 2.5},  # crossed
            {"strike": 515.0, "bid": float("nan"), "ask": 1.0},  # missing bid
            {"strike": 600.0, "bid": 1.0, "ask": 1.1},  # outside the 10% band
        ]
        puts = [bs_row(k, -1) for k in (480.0, 490.0, 500.0, 510.0, 520.0)]
        return types.SimpleNamespace(calls=FakeCalls(calls), puts=FakeCalls(puts))


@pytest.fixture
def live_feed(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    monkeypatch.delitem(sys.modules, "live_feed", raising=False)
    import live_feed

    global T_FAKE
    T_FAKE = live_feed.years_to_expiry("2099-01-16", datetime.now(NY))
    return live_feed


def test_next_day_expiry_is_30_hours_not_skipped(live_feed):
    # Monday 10:00 ET to Tuesday's 16:00 ET close. Whole-day flooring gave
    # 0 days here and dropped the expiry from the surface.
    now = datetime(2026, 11, 9, 10, 0, tzinfo=NY)
    assert live_feed.years_to_expiry("2026-11-10", now) == pytest.approx(30 / 24 / 365, abs=1e-12)


def test_same_day_expiry_counts_hours_to_the_close(live_feed):
    now = datetime(2026, 11, 9, 10, 0, tzinfo=NY)
    assert live_feed.years_to_expiry("2026-11-09", now) == pytest.approx(6 / 24 / 365, abs=1e-12)


def test_quotes_without_a_bid_or_crossed_are_dropped(live_feed):
    # A zero bid makes the mid half the ask, which is not a price anyone traded.
    chain = live_feed.LiveFeed(rate=0.04).generate_tick()
    calls = sorted(o["strike"] for o in chain if o["type"] == "call")
    assert calls == [480.0, 490.0, 500.0, 510.0, 520.0]


def test_carry_comes_from_put_call_parity_at_the_money(live_feed):
    # Using spot with q = 0 would misstate the forward by the dividend yield
    # and skew every IV; parity at the ATM strike recovers q.
    chain = live_feed.LiveFeed(rate=0.04).generate_tick()
    assert chain[0]["dividend_yield"] == pytest.approx(0.013, abs=1e-6)
    assert {o["type"] for o in chain} == {"call", "put"}
    assert all("bid" in o and "ask" in o for o in chain)


def test_live_chain_solves_and_checks_against_spreads(live_feed):
    chain = live_feed.LiveFeed(rate=0.04).generate_tick()
    snapshot, _ = analyze_chain(chain, tol=0.005)
    ivs = [q["iv"] for q in snapshot["quotes"]]
    assert all(iv == pytest.approx(0.2, abs=1e-3) for iv in ivs)
    assert snapshot["violations"] == []
    # Push one put off parity by more than the two legs' half-spreads (0.10).
    put = next(o for o in chain if o["type"] == "put" and o["strike"] == 510.0)
    put["price"] += 0.30
    snapshot, _ = analyze_chain(chain, tol=0.005)
    parity = [v for v in snapshot["violations"] if v["type"] == "parity"]
    assert len(parity) == 1 and parity[0]["exceeds_spread"] is True


def test_expiry_across_the_dst_change_counts_real_elapsed_hours(live_feed):
    # Clocks fall back on 2026-11-01. Friday 10:00 EDT to Monday 16:00 EST is
    # 79 hours of real time; subtracting two NY-zoned datetimes directly
    # gives the 78-hour wall-clock difference.
    now = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
    assert live_feed.years_to_expiry("2026-11-02", now) == pytest.approx(79 / 24 / 365, abs=1e-12)


def test_after_the_close_on_expiry_day_time_is_negative(live_feed):
    # generate_tick skips expiries with T <= 0, so a finished expiry is dropped.
    now = datetime(2026, 11, 9, 16, 30, tzinfo=NY)
    assert live_feed.years_to_expiry("2026-11-09", now) < 0
