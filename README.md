# Volatility Scanner

A live options scanner: it streams an option chain, works out the implied volatility (the volatility the market price implies) of every quote, fits a smooth curve per expiry and flags prices that would allow risk-free profit. It runs as a web dashboard that updates twice a second.

![Report: false alarms by quote noise, SVI residuals](docs/assets/report.png)

[Report](https://allenwang0.github.io/volatilityscanner/)

## Highlights

- **Robust implied-volatility solver:** price-bound checks, a smart starting guess, Newton steps with a Brent fallback. On 10,000 random contracts, every one with usable sensitivity solves, with max error 2.5e-12.
- **SVI curve fits** (a standard volatility-smile model) checked for arbitrage, with 0.017–0.032 vol points of error in-sample and 0.021–0.114 held out (simulated chain).
- **Five no-arbitrage checks,** each sized in dollars and compared with the cost of crossing the bid-ask spread. At 0.0005 IV noise per quote they fire on 85% of simulated ticks.
- **WebSocket server** where a slow client skips to the newest update instead of slowing everyone else.

## Run it

```bash
git clone https://github.com/allenwang0/volatilityscanner.git && cd volatilityscanner
make setup PYTHON=python3.12
.venv/bin/python main.py --seed 7
# open http://127.0.0.1:8000
```

## How it works

- By default a seeded simulator generates a realistic option chain; a real-data mode pulls SPY options from yfinance.
- Each quote's price is inverted through Black-Scholes to get implied volatility; failed solves are shown as hollow markers, never as wrong numbers.
- Each expiry's quotes are fit to an SVI smile, which gives at-the-money volatility and skew.
- The checks test that prices fall with strike, butterflies are non-negative, put-call parity holds and longer expiries never carry less total variance.

## Limitations

- The headline numbers come from the simulator, not market data. Its smiles are SVI-shaped, so SVI fits by design.
- Black-Scholes assumes European exercise; SPY options are American, so some prices are only approximately right.
- Yahoo quotes are delayed and not synchronized across strikes.
