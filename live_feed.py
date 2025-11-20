import yfinance as yf
import pandas as pd
import time
from datetime import datetime


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
            exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
            days_to_expiry = (exp_dt - datetime.now()).days
            if days_to_expiry < 1:
                continue  # Skip expiring today

            T = days_to_expiry / 365.0

            # Parse calls
            for index, row in calls.iterrows():
                # Filter for reasonable moneyness to reduce data load
                # (Only keep strikes within 10% of spot price)
                if not (spot * 0.9 < row['strike'] < spot * 1.1):
                    continue

                options_chain.append({
                    "strike": row['strike'],
                    "expiry": days_to_expiry,
                    "time_to_expiry": T,
                    "price": (row['bid'] + row['ask']) / 2,  # Midpoint price
                    "type": "call",
                    "spot": spot,
                    "interest_rate": 0.05  # Hardcoded risk-free rate (approx 5%)
                })

        return options_chain