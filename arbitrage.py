"""Static no-arbitrage checks on one chain snapshot, grouped by expiry.

Every violation size is in price units (dollars per option). A violation smaller
than the tolerance is not reported. When the chain has bid and ask quotes, each
violation also says whether it exceeds the cost of crossing half the spread on
every leg, which is what a trade at those quotes would pay.
"""

import math
from dataclasses import dataclass

import numpy as np

from engine import CALL, PUT, Chain, black_scholes_price

SIMULATED_TOLERANCE = 1e-6
LIVE_TOLERANCE = 0.005  # half of the $0.01 minimum tick on SPY options


@dataclass(frozen=True)
class Violation:
    type: str
    expiry_days: float
    strikes: tuple[float, ...]
    size: float
    exceeds_spread: bool | None


@dataclass(frozen=True)
class ArbitrageScan:
    violations: list[Violation]
    conditions_checked: int
    quotes_checked: int


def _half_spread(chain: Chain, idx: np.ndarray) -> np.ndarray:
    return (chain.ask[idx] - chain.bid[idx]) / 2.0


def _exceeds(size: float, cost: float) -> bool | None:
    return None if math.isnan(cost) else bool(size > cost)


def _strike_checks(
    chain: Chain, days: float, calls: np.ndarray, tol: float
) -> tuple[list[Violation], int]:
    K = chain.strike[calls]
    C = chain.price[calls]
    hs = _half_spread(chain, calls)
    discount = math.exp(-chain.r * chain.T[calls[0]])
    found: list[Violation] = []
    checked = 0
    for i in range(len(K) - 1):
        k1, k2 = float(K[i]), float(K[i + 1])
        cost = float(hs[i] + hs[i + 1])
        rise = float(C[i + 1] - C[i])
        too_steep = float(C[i] - C[i + 1]) - discount * (k2 - k1)
        checked += 2
        if rise > tol:
            found.append(Violation("monotonicity", days, (k1, k2), rise, _exceeds(rise, cost)))
        if too_steep > tol:
            found.append(Violation("slope", days, (k1, k2), too_steep, _exceeds(too_steep, cost)))
    for i in range(len(K) - 2):
        k1, k2, k3 = float(K[i]), float(K[i + 1]), float(K[i + 2])
        lam = (k3 - k2) / (k3 - k1)
        size = float(C[i + 1] - (lam * C[i] + (1 - lam) * C[i + 2]))
        cost = float(lam * hs[i] + hs[i + 1] + (1 - lam) * hs[i + 2])
        checked += 1
        if size > tol:
            found.append(Violation("butterfly", days, (k1, k2, k3), size, _exceeds(size, cost)))
    return found, checked


def _parity_checks(
    chain: Chain, days: float, calls: np.ndarray, puts: np.ndarray, tol: float
) -> tuple[list[Violation], int]:
    put_by_strike = {float(chain.strike[j]): j for j in puts}
    found: list[Violation] = []
    checked = 0
    for i in calls:
        j = put_by_strike.get(float(chain.strike[i]))
        if j is None:
            continue
        K = float(chain.strike[i])
        T = float(chain.T[i])
        parity = math.exp(-chain.r * T) * (float(chain.forward[i]) - K)
        size = abs(float(chain.price[i] - chain.price[j]) - parity)
        checked += 1
        if size > tol:
            cost = float(_half_spread(chain, np.array([i, j])).sum())
            found.append(Violation("put_call_parity", days, (K,), size, _exceeds(size, cost)))
    return found, checked


def _calendar_checks(
    chain: Chain, iv: np.ndarray, call_groups: list[tuple[float, np.ndarray]], tol: float
) -> tuple[list[Violation], int]:
    """Total variance w = sigma^2 T must not fall with T at fixed k = log(K/F).

    For each pair of adjacent expiries, the earlier slice's w is linearly
    interpolated at the later slice's k values (no extrapolation). An expiry with
    fewer than two solved quotes cannot be interpolated, so later expiries are
    compared with the last one that can. The size is the later call's
    Black-Scholes price at the earlier slice's total variance minus its market
    price.
    """
    found: list[Violation] = []
    checked = 0
    early: np.ndarray | None = None
    for days, group in call_groups:
        late = group[np.isfinite(iv[group])]
        if early is None:
            if len(late) >= 2:
                early = late
            continue
        base = early
        if len(late) >= 2:
            early = late
        k_early = np.log(chain.strike[base] / chain.forward[base])
        w_early = iv[base] ** 2 * chain.T[base]
        for i in late:
            k = math.log(chain.strike[i] / chain.forward[i])
            if not k_early[0] <= k <= k_early[-1]:
                continue
            checked += 1
            w_floor = float(np.interp(k, k_early, w_early))
            T = float(chain.T[i])
            if w_floor <= iv[i] ** 2 * T:
                continue
            floor_price = black_scholes_price(
                chain.spot,
                float(chain.strike[i]),
                T,
                chain.r,
                math.sqrt(w_floor / T),
                CALL,
                chain.q,
            )
            size = floor_price - float(chain.price[i])
            if size > tol:
                nearest = base[np.argsort(np.abs(k_early - k))[:2]]
                cost = float(_half_spread(chain, np.append(nearest, i)).sum())
                found.append(
                    Violation(
                        "calendar", days, (float(chain.strike[i]),), size, _exceeds(size, cost)
                    )
                )
    return found, checked


def check_static_arbitrage(
    chain: Chain, iv: np.ndarray, tol: float = SIMULATED_TOLERANCE
) -> ArbitrageScan:
    """Run every static check on the chain. iv holds one implied vol per quote (NaN if failed)."""
    violations: list[Violation] = []
    checked = 0
    call_groups: list[tuple[float, np.ndarray]] = []
    for days in np.unique(chain.expiry_days):
        in_expiry = chain.expiry_days == days
        calls = np.flatnonzero(in_expiry & (chain.flag == CALL))
        calls = calls[np.argsort(chain.strike[calls], kind="stable")]
        puts = np.flatnonzero(in_expiry & (chain.flag == PUT))
        days_f = float(days)
        if len(calls):
            call_groups.append((days_f, calls))
            found, n = _strike_checks(chain, days_f, calls, tol)
            violations += found
            checked += n
        found, n = _parity_checks(chain, days_f, calls, puts, tol)
        violations += found
        checked += n
    found, n = _calendar_checks(chain, iv, call_groups, tol)
    violations += found
    checked += n
    return ArbitrageScan(violations, checked, len(chain))
