"""Delayed option chains from Yahoo Finance via yfinance (pip install -e ".[live]").

Yahoo's data is for personal use under its terms; nothing fetched here is saved.
"""

import math
from datetime import datetime, timezone
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf

from engine import CALL, PUT, Chain

NY = ZoneInfo("America/New_York")
MARKET_CLOSE = clock_time(16, 0)  # options stop trading at the 16:00 ET close
# FRED series DTB3, 3-month Treasury bill secondary market rate, 4.06% on 2026-11-09.
DEFAULT_RATE = 0.0406
MONEYNESS_BAND = 0.10  # keep strikes within 10% of spot to limit download and fit size
EXPIRIES = 4


def years_to_expiry(expiry_date, now):
    """Time in years from `now` to 16:00 New York time on `expiry_date` ('YYYY-MM-DD')."""
    expiry = datetime.combine(
        datetime.strptime(expiry_date, "%Y-%m-%d").date(), MARKET_CLOSE, tzinfo=NY
    )
    # Subtract in UTC: datetimes sharing a tzinfo subtract as wall-clock times,
    # which is an hour off across a DST change.
    elapsed = expiry.astimezone(timezone.utc) - now.astimezone(timezone.utc)
    return elapsed.total_seconds() / (365 * 24 * 3600)


def usable_quotes(rows, spot):
    """(strike, bid, ask) for quotes near spot with a positive bid and an uncrossed market."""
    out = []
    for _, row in rows.iterrows():
        strike, bid, ask = float(row["strike"]), float(row["bid"]), float(row["ask"])
        if not abs(strike / spot - 1) < MONEYNESS_BAND:
            continue
        # Written as "not ok" so missing (NaN) quotes are dropped too.
        if not (bid > 0 and ask >= bid):
            continue
        out.append((strike, bid, ask))
    return out


def parity_forward(calls, puts, spot, r, T):
    """Forward implied by put-call parity at the strike nearest spot with both legs quoted."""
    put_mid = {k: (b + a) / 2 for k, b, a in puts}
    both = [(k, (b + a) / 2) for k, b, a in calls if k in put_mid]
    if not both:
        return spot * math.exp(r * T)
    k, call_mid = min(both, key=lambda item: abs(item[0] - spot))
    return k + math.exp(r * T) * (call_mid - put_mid[k])


class LiveFeed:
    def __init__(self, symbol="SPY", rate=DEFAULT_RATE):
        self.symbol = symbol
        self.r = rate
        self.ticker = yf.Ticker(symbol)

    def get_spot_price(self):
        data = self.ticker.history(period="1d")
        return float(data["Close"].iloc[-1])

    def generate_tick(self, now=None) -> Chain:
        """Download the nearest expiries (2 to 3 seconds each) and return calls and puts at mid."""
        now = now or datetime.now(NY)
        spot = self.get_spot_price()
        cols = {name: [] for name in ("days", "T", "strike", "flag", "bid", "ask", "forward")}
        for expiry_date in self.ticker.options[:EXPIRIES]:
            T = years_to_expiry(expiry_date, now)
            if T <= 0:
                continue  # already expired
            chain = self.ticker.option_chain(expiry_date)
            calls = usable_quotes(chain.calls, spot)
            puts = usable_quotes(chain.puts, spot)
            forward = parity_forward(calls, puts, spot, self.r, T)
            for flag, quotes in ((CALL, calls), (PUT, puts)):
                for strike, bid, ask in quotes:
                    cols["days"].append(T * 365)
                    cols["T"].append(T)
                    cols["strike"].append(strike)
                    cols["flag"].append(flag)
                    cols["bid"].append(bid)
                    cols["ask"].append(ask)
                    cols["forward"].append(forward)
        arr = {name: np.array(values, dtype=float) for name, values in cols.items()}
        return Chain(
            spot=spot,
            r=self.r,
            q=0.0,
            expiry_days=arr["days"],
            T=arr["T"],
            strike=arr["strike"],
            flag=arr["flag"].astype(int),
            price=(arr["bid"] + arr["ask"]) / 2,
            forward=arr["forward"],
            bid=arr["bid"],
            ask=arr["ask"],
            source=f"yfinance {self.symbol}, delayed, fetched {now:%Y-%m-%d %H:%M} ET",
        )
