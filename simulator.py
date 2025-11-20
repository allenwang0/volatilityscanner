import numpy as np
import time
from engine import black_scholes_price


class MarketSimulator:
    def __init__(self):
        self.spot_price = 5000.0  # SPX-like level
        self.r = 0.05  # 5% Risk-free rate
        self.strikes = np.linspace(4500, 5500, 20)  # 20 Strike levels
        self.expiries = np.array([7 / 365, 30 / 365, 60 / 365, 90 / 365])  # Days to years

        # Base Volatility Surface (The "Smile" Shape)
        # Vol increases as we move away from ATM (At-The-Money)
        self.base_vol = 0.15

    def generate_tick(self):
        """
        Simulates a tick of data.
        Returns a list of option dicts with current Market Prices.
        """
        # Random walk for Spot Price
        self.spot_price *= (1 + np.random.normal(0, 0.0005))

        options_chain = []

        for T in self.expiries:
            for K in self.strikes:
                # Simulate Vol Smile: Curvature based on moneyness
                moneyness = K / self.spot_price
                # A parabola equation for vol: 0.15 + steepness * (moneyness - 1)^2
                simulated_iv = self.base_vol + 1.5 * (moneyness - 1) ** 2

                # Add random noise (jitter) to simulate market inefficiencies
                noise = np.random.normal(0, 0.002)
                simulated_iv += noise

                # Calculate the theoretical price using this messy IV
                # We only simulate CALLS for this demo to keep it simple
                price = black_scholes_price(self.spot_price, K, T, self.r, simulated_iv, 1)

                options_chain.append({
                    "strike": round(K, 2),
                    "expiry": round(T * 365, 1),  # Days
                    "time_to_expiry": T,
                    "price": round(price, 2),
                    "type": "call",
                    "spot": round(self.spot_price, 2),
                    "interest_rate": self.r
                })

        return options_chain