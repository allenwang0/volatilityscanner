"""Time the scalar IV loop against the vectorized solver on random chains.

Run: python bench/bench_iv.py
"""

import os
import platform
import statistics
import sys
import timeit

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import black_scholes_price_array, implied_volatility, implied_volatility_array  # noqa: E402

SEED = 7
REPEATS = 5


def random_chain(n: int, seed: int = SEED):
    """Equity-like chain: spot 5000, strikes within 30%, 1 day to 2 years, vol 10% to 80%."""
    rng = np.random.default_rng(seed)
    S = np.full(n, 5000.0)
    K = S * rng.uniform(0.7, 1.3, n)
    T = rng.uniform(1 / 365, 2, n)
    sigma = rng.uniform(0.1, 0.8, n)
    flag = rng.choice([1.0, -1.0], n)
    r = np.full(n, 0.05)
    q = np.zeros(n)
    price = black_scholes_price_array(S, K, T, r, sigma, flag, q)
    return price, S, K, T, r, flag, q


def run(sizes=(1_000, 10_000)) -> list[dict]:
    rows = []
    for n in sizes:
        chain = random_chain(n)
        args = [list(map(float, a)) for a in chain]

        def scalar(args=args, n=n):
            for i in range(n):
                implied_volatility(
                    args[0][i], args[1][i], args[2][i], args[3][i], args[4][i], int(args[5][i]), args[6][i]
                )

        def vector(chain=chain):
            return implied_volatility_array(*chain)

        t_scalar = statistics.median(timeit.repeat(scalar, number=1, repeat=REPEATS)) * 1000
        t_vector = statistics.median(timeit.repeat(vector, number=1, repeat=REPEATS)) * 1000
        result = vector()
        rows.append(
            {
                "n": n,
                "scalar_ms": t_scalar,
                "vector_ms": t_vector,
                "speedup": t_scalar / t_vector,
                "brent_share": float(result.brent.mean()),
            }
        )
    return rows


def machine() -> str:
    return (
        f"{platform.machine()} {platform.system()} {platform.release()}, Python {platform.python_version()}"
    )


if __name__ == "__main__":
    print(f"seed {SEED}, median of {REPEATS} runs, {machine()}")
    print("| options | scalar loop (ms) | vectorized (ms) | speed-up | Brent fallback |")
    print("|---:|---:|---:|---:|---:|")
    for row in run():
        print(
            f"| {row['n']:,} | {row['scalar_ms']:.1f} | {row['vector_ms']:.1f} "
            f"| {row['speedup']:.1f}x | {row['brent_share']:.2%} |"
        )
