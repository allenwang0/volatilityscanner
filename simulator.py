import numpy as np

from engine import CALL, Chain, black_scholes_price


class MarketSimulator:
    def __init__(self, tick_size=None):
        self.spot_price = 5000.0  # SPX-like level
        self.r = 0.05  # 5% Risk-free rate
        self.strikes = np.linspace(4500, 5500, 20)  # 20 Strike levels
        self.expiries = np.array([7 / 365, 30 / 365, 60 / 365, 90 / 365])  # Days to years

        # Base Volatility Surface (The "Smile" Shape)
        # Vol increases as we move away from ATM (At-The-Money)
        self.base_vol = 0.15

        # Optional quote increment (e.g. 0.05); None keeps full-precision prices
        self.tick_size = tick_size

    def generate_tick(self) -> Chain:
        """Move spot one step and return a call chain priced from a noisy smile."""
        self.spot_price *= 1 + np.random.normal(0, 0.0005)

        days = np.repeat(np.round(self.expiries * 365, 1), len(self.strikes))
        T = np.repeat(self.expiries, len(self.strikes))
        # Price the same strike the payload reports
        strike = np.tile(np.round(self.strikes, 2), len(self.expiries))
        price = np.empty(len(strike))
        for i, (K, t) in enumerate(zip(strike, T, strict=True)):
            # Simulate Vol Smile: Curvature based on moneyness
            simulated_iv = self.base_vol + 1.5 * (K / self.spot_price - 1) ** 2
            # Add random noise (jitter) to simulate market inefficiencies
            simulated_iv += np.random.normal(0, 0.002)
            price[i] = black_scholes_price(self.spot_price, K, t, self.r, simulated_iv, CALL)
        if self.tick_size:
            price = np.round(price / self.tick_size) * self.tick_size

        n = len(price)
        return Chain(
            spot=self.spot_price,
            r=self.r,
            q=0.0,
            expiry_days=days,
            T=T,
            strike=strike,
            flag=np.full(n, CALL),
            price=price,
            forward=self.spot_price * np.exp(self.r * T),
            bid=np.full(n, np.nan),
            ask=np.full(n, np.nan),
        )
