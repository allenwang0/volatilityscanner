"""Scalar loop vs vectorized IV solver on seeded synthetic contracts.

Run: python -m bench.bench_iv
"""

import platform
import statistics
import time
from dataclasses import dataclass

import numpy as np

from engine import (
    CALL,
    PUT,
    black_scholes_price_vec,
    implied_volatility,
    implied_volatility_vec,
    vega_vec,
)

SIZES = (1_000, 10_000)
REPEATS = 5
SPOT = 5000.0
RATE = 0.05


@dataclass(frozen=True)
class BenchRow:
    n: int
    scalar_ms: float
    vectorized_ms: float
    brent_share: float
    max_abs_diff: float


def contracts(n: int, seed: int):
    """SPX-like options: K/S in [0.7, 1.3], 1 day to 1 year, vol 8% to 60%, calls and puts."""
    rng = np.random.default_rng(seed)
    K = SPOT * rng.uniform(0.7, 1.3, n)
    T = rng.uniform(1 / 365, 1.0, n)
    sigma = rng.uniform(0.08, 0.6, n)
    flag = rng.choice([CALL, PUT], n).astype(float)
    price = black_scholes_price_vec(SPOT, K, T, RATE, sigma, flag)
    return price, K, T, flag, sigma


def _median_ms(fn) -> float:
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def run(seed: int = 7) -> list[BenchRow]:
    rows = []
    for n in SIZES:
        price, K, T, flag, sigma = contracts(n, seed)

        def scalar():
            return np.array(
                [
                    implied_volatility(p, SPOT, k, t, RATE, int(f)).sigma
                    for p, k, t, f in zip(price, K, T, flag)
                ]
            )

        def vectorized():
            return implied_volatility_vec(price, SPOT, K, T, RATE, flag)

        batch = vectorized()
        # Where vega < 1e-4 * S the root is ill-conditioned and both answers are noise.
        identifiable = vega_vec(SPOT, K, T, RATE, sigma) > 1e-4 * SPOT
        diff = np.abs(batch.sigma - scalar())[identifiable]
        rows.append(
            BenchRow(
                n=n,
                scalar_ms=_median_ms(scalar),
                vectorized_ms=_median_ms(vectorized),
                brent_share=float(batch.brent_fallback.mean()),
                max_abs_diff=float(np.nanmax(diff)),
            )
        )
    return rows


def machine() -> str:
    return f"{platform.machine()} {platform.system()}, Python {platform.python_version()}"


def main() -> None:
    print(f"{machine()}, median of {REPEATS} runs, seed 7")
    print(
        "| options | scalar loop (ms) | vectorized (ms) | speedup | Brent fallback "
        "| max diff, vega > 1e-4 S |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for row in run():
        print(
            f"| {row.n:,} | {row.scalar_ms:.1f} | {row.vectorized_ms:.1f} | "
            f"{row.scalar_ms / row.vectorized_ms:.0f}x | {row.brent_share:.1%} "
            f"| {row.max_abs_diff:.1e} |"
        )


if __name__ == "__main__":
    main()
