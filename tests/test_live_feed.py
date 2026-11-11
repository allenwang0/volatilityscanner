import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

NY = ZoneInfo("America/New_York")


class FakeColumn:
    def __init__(self, values):
        self.iloc = values


class FakeCalls:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return enumerate(self.rows)


class FakeTicker:
    """Offline stand-in for yfinance.Ticker with one expiry a year out."""
    options = ["2099-01-16"]

    def __init__(self, symbol):
        pass

    def history(self, period):
        return {"Close": FakeColumn([500.0])}

    def option_chain(self, expiry):
        return types.SimpleNamespace(calls=FakeCalls([
            {"strike": 495.0, "bid": 10.0, "ask": 10.4},  # good quote
            {"strike": 500.0, "bid": 0.0, "ask": 6.0},    # no bid
            {"strike": 505.0, "bid": 3.0, "ask": 2.5},    # crossed
            {"strike": 510.0, "bid": float("nan"), "ask": 1.0},  # missing bid
        ]))


@pytest.fixture
def live_feed(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    monkeypatch.delitem(sys.modules, "live_feed", raising=False)
    import live_feed
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
    chain = live_feed.LiveFeed().generate_tick()
    assert [o["strike"] for o in chain] == [495.0]
    assert chain[0]["price"] == pytest.approx(10.2)
