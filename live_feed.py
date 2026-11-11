import yfinance as yf
import time
from datetime import datetime, time as clock_time, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
MARKET_CLOSE = clock_time(16, 0)  # options stop trading at the 16:00 ET close


def years_to_expiry(expiry_date, now):
    """Time in years from `now` to 16:00 New York time on `expiry_date` ('YYYY-MM-DD')."""
    expiry = datetime.combine(datetime.strptime(expiry_date, "%Y-%m-%d").date(), MARKET_CLOSE, tzinfo=NY)
    # Subtract in UTC: datetimes sharing a tzinfo subtract as wall-clock times,
    # which is an hour off across a DST change.
    elapsed = expiry.astimezone(timezone.utc) - now.astimezone(timezone.utc)
    return elapsed.total_seconds() / (365 * 24 * 3600)


class LiveFeed:
    def __init__(self):
        self.ticker_symbol = "SPY"
        self.ticker = yf.Ticker(self.ticker_symbol)

    def get_spot_price(self):
        # Fast fetch of current price
        data = self.ticker.history(period="1d")
        return data['Close'].iloc[-1]

    def generate_tick(self):
        """
        Fetches REAL option chain from Yahoo Finance.
        Note: This can take 2-3 seconds to download.
        """
        spot = self.get_spot_price()
        options_chain = []

        # Get expiration dates (Yahoo returns 'YYYY-MM-DD')
        # We only take the first 3 expirations to keep it fast
        expirations = self.ticker.options[:3]

        for expiry_date in expirations:
            # Fetch the chain for this specific date
            opt_chain = self.ticker.option_chain(expiry_date)
            calls = opt_chain.calls

            # Calculate time to expiry (T) in years
            T = years_to_expiry(expiry_date, datetime.now(NY))
            if T <= 0:
                continue  # Already expired

            # Parse calls
            for index, row in calls.iterrows():
                # Filter for reasonable moneyness to reduce data load
                # (Only keep strikes within 10% of spot price)
                if not (spot * 0.9 < row['strike'] < spot * 1.1):
                    continue

                # Drop quotes with no bid or a crossed market; their mid is not a price.
                # Written as "not ok" so missing (NaN) quotes are dropped too.
                if not (row['bid'] > 0 and row['ask'] >= row['bid']):
                    continue

                options_chain.append({
                    "strike": row['strike'],
                    "expiry": T * 365,  # Days
                    "time_to_expiry": T,
                    "price": (row['bid'] + row['ask']) / 2,  # Midpoint price
                    "type": "call",
                    "spot": spot,
                    "interest_rate": 0.05  # Hardcoded risk-free rate (approx 5%)
                })

        return options_chain