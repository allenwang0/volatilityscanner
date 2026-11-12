"""Real option chains from Yahoo Finance via yfinance (opt-in live mode)."""

import math
from datetime import UTC, datetime
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import yfinance as yf

NY = ZoneInfo("America/New_York")
MARKET_CLOSE = clock_time(16, 0)  # options stop trading at the 16:00 ET close


def years_to_expiry(expiry_date, now):
    """Time in years from `now` to 16:00 New York time on `expiry_date` ('YYYY-MM-DD')."""
    expiry = datetime.combine(datetime.strptime(expiry_date, "%Y-%m-%d").date(), MARKET_CLOSE, tzinfo=NY)
    # Subtract in UTC: datetimes sharing a tzinfo subtract as wall-clock times,
    # which is an hour off across a DST change.
    elapsed = expiry.astimezone(UTC) - now.astimezone(UTC)
    return elapsed.total_seconds() / (365 * 24 * 3600)


MONEYNESS_BAND = 0.10  # keep strikes within 10% of spot; far wings are mostly stale quotes
MAX_EXPIRIES = 3  # each expiry is one HTTP request of about a second
DEFAULT_RATE = 0.05  # placeholder; pass --rate with the current 3-month T-bill yield


def _good(row) -> bool:
    # Drop quotes with no bid or a crossed market; their mid is not a price.
    # Written as "not ok" so missing (NaN) quotes are dropped too.
    return bool(row["bid"] > 0 and row["ask"] >= row["bid"])


def implied_carry(spot, rate, T, calls: dict, puts: dict) -> float | None:
    """Dividend yield q implied by put-call parity at the strike nearest spot.

    F = K + e^{rT} (C - P), and q = r - ln(F / S) / T.
    """
    common = [k for k in calls if k in puts]
    if not common:
        return None
    K = min(common, key=lambda k: abs(k - spot))
    forward = K + math.exp(rate * T) * (calls[K] - puts[K])
    if forward <= 0:
        return None
    return rate - math.log(forward / spot) / T


class LiveFeed:
    def __init__(self, symbol="SPY", rate=DEFAULT_RATE):
        self.symbol = symbol
        self.rate = rate
        self.ticker = yf.Ticker(symbol)
        self.fetched_at = None

    def describe(self) -> dict:
        return {"kind": "yfinance", "symbol": self.symbol, "rate": self.rate, "fetched_at": self.fetched_at}

    def get_spot_price(self):
        return float(self.ticker.history(period="1d")["Close"].iloc[-1])

    def generate_tick(self) -> list[dict]:
        """Download calls and puts for the nearest expiries (takes a few seconds)."""
        spot = self.get_spot_price()
        now = datetime.now(UTC)
        chain = []
        for expiry_date in self.ticker.options[:MAX_EXPIRIES]:
            T = years_to_expiry(expiry_date, now)
            if T <= 0:
                continue  # already expired
            tables = self.ticker.option_chain(expiry_date)
            legs = {}
            for kind, table in (("call", tables.calls), ("put", tables.puts)):
                legs[kind] = {
                    float(row["strike"]): row
                    for _, row in table.iterrows()
                    if abs(row["strike"] / spot - 1) < MONEYNESS_BAND and _good(row)
                }
            mids = {
                kind: {k: (r["bid"] + r["ask"]) / 2 for k, r in rows.items()} for kind, rows in legs.items()
            }
            q = implied_carry(spot, self.rate, T, mids["call"], mids["put"])
            if q is None:
                continue  # no forward without a call and put at a common strike
            for kind, rows in legs.items():
                for K, row in rows.items():
                    chain.append(
                        {
                            "strike": K,
                            "expiry": T * 365,  # days
                            "time_to_expiry": T,
                            "price": mids[kind][K],
                            "bid": float(row["bid"]),
                            "ask": float(row["ask"]),
                            "type": kind,
                            "spot": spot,
                            "interest_rate": self.rate,
                            "dividend_yield": q,
                        }
                    )
        self.fetched_at = now.isoformat(timespec="seconds")
        return chain
