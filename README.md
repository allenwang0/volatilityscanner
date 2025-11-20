# Volatility Scanner

**Volatility Scanner** is a comprehensive Python-based tool designed to analyze historical market data, calculate volatility metrics, and simulate future price movements. It features a modular backend (`engine` and `simulator`) and a web-based frontend for interactive analysis.

## 📂 Project Structure

* **`main.py`**: The web application entry point (likely Flask or similar). It routes user requests to the backend logic and renders the `templates`.
* **`engine.py`**: The core computational unit. This module is responsible for:
    * Fetching historical price data (OHLC).
    * Computing logarithmic returns.
    * Calculating rolling statistical volatility.
* **`simulator.py`**: A quantitative module that uses the data from `engine.py` to project future scenarios, likely employing Monte Carlo simulations or random walk models.
* **`templates/`**: HTML interfaces for the dashboard and data visualization.

## 🧮 Mathematical Core

This project relies on fundamental financial mathematics to assess risk. Below is the breakdown of the quantitative methods used in the **Volatility Engine**.

### 1. Logarithmic Returns
Instead of simple percentage changes, the scanner likely uses **Log Returns** for time-additive properties, which are essential for accurate volatility modeling over time.

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

* Where $P_t$ is the price at time $t$ and $P_{t-1}$ is the price at time $t-1$.

### 2. Historical Volatility (Standard Deviation)
The core metric of the scanner is the standard deviation ($\sigma$) of these log returns. This measures the dispersion of asset returns from the mean, representing "risk."

$$\sigma = \sqrt{\frac{1}{N-1} \sum_{t=1}^{N} (r_t - \bar{r})^2}$$

* $N$: Number of observations (e.g., trading days).
* $\bar{r}$: The average return over the period.

### 3. Annualized Volatility
To make the daily volatility comparable to annual metrics (like interest rates or implied volatility), the engine applies an annualization factor. Since there are typically **252 trading days** in a year, the formula is:

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

### 4. Market Simulation (Geometric Brownian Motion)
The `simulator.py` likely projects future price paths using the **Geometric Brownian Motion (GBM)** model, which assumes prices drift by expected return ($\mu$) and are shocked by random volatility ($\sigma$):

$$\frac{dS_t}{S_t} = \mu dt + \sigma dW_t$$

* $\mu$: Expected drift (average return).
* $\sigma$: Historical volatility calculated by the engine.
* $dW_t$: A stochastic process (Wiener process) representing random market noise.

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/allenwang0/volatilityscanner.git](https://github.com/allenwang0/volatilityscanner.git)
    cd volatilityscanner
    ```

2.  **Install Dependencies**
    Ensure you have `pandas`, `numpy`, and web framework dependencies installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python main.py
    ```
    Access the scanner at `http://localhost:5000` (default port).

## 🛠 Usage

1.  **Input Ticker**: Enter a stock symbol (e.g., `SPY`, `AAPL`) in the web interface.
2.  **Select Window**: Choose the lookback period for volatility calculation (e.g., 30-day, 90-day).
3.  **Run Simulation**: Trigger `simulator.py` to generate potential future price paths based on current volatility regimes.

## 📜 License
Open Source.