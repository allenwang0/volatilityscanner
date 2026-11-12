"""Turn one option chain into the dashboard's snapshot: IVs, SVI fits, term structure, arbitrage."""

import math
from dataclasses import asdict

import numpy as np

from engine import ARB_TOL, implied_volatility_array, scan_arbitrage_with_counts, vega_array
from svi import SVIParams, atm_vol, calendar_crossings, fit_svi, skew_25d

MIN_FIT_POINTS = 5  # SVI has five parameters
CURVE_POINTS = 101


def _num(x):
    """JSON has no NaN; failed values become null."""
    return None if x is None or not math.isfinite(x) else float(x)


def forward_of(quote: dict) -> float:
    if "forward" in quote:
        return quote["forward"]
    T = quote["time_to_expiry"]
    return quote["spot"] * math.exp((quote["interest_rate"] - quote.get("dividend_yield", 0.0)) * T)


def analyze_chain(
    chain: list[dict], previous: dict[float, SVIParams] | None = None, tol: float = ARB_TOL
) -> tuple[dict, dict[float, SVIParams]]:
    """Return the snapshot dict and this tick's SVI fits by expiry (to warm-start the next tick).

    The smile uses out-of-the-money quotes: puts below the forward, calls at
    or above it. In-the-money quotes are still solved and listed.
    """
    previous = previous or {}
    flags = np.array([1.0 if o["type"] == "call" else -1.0 for o in chain])
    cols = {
        name: np.array([o[key] if key != "dividend_yield" else o.get(key, 0.0) for o in chain], dtype=float)
        for name, key in [
            ("price", "price"),
            ("S", "spot"),
            ("K", "strike"),
            ("T", "time_to_expiry"),
            ("r", "interest_rate"),
            ("q", "dividend_yield"),
        ]
    }
    ivs = implied_volatility_array(
        cols["price"], cols["S"], cols["K"], cols["T"], cols["r"], flags, cols["q"]
    )

    quotes = []
    for i, o in enumerate(chain):
        F = forward_of(o)
        k = math.log(o["strike"] / F)
        quote = {
            "type": o["type"],
            "strike": o["strike"],
            "expiry": o["expiry"],
            "T": o["time_to_expiry"],
            "price": o["price"],
            "k": k,
            "iv": _num(ivs.sigma[i]),
            "iterations": int(ivs.iterations[i]),
            "reason": ivs.reason[i],
            "smile": (o["type"] == "put") == (o["strike"] < F),
        }
        for side in ("bid", "ask"):
            if side in o:
                quote[side] = o[side]
        quotes.append(quote)

    fits: dict[float, SVIParams] = {}
    expiries = []
    for T in sorted({q["T"] for q in quotes}):
        rows = [q for q in quotes if q["T"] == T]
        pts = [q for q in rows if q["smile"] and q["iv"] is not None]
        first = next(o for o in chain if o["time_to_expiry"] == T)
        entry = {
            "days": rows[0]["expiry"],
            "T": T,
            "forward": forward_of(first),
            "n_quotes": len(rows),
            "n_failed": sum(q["iv"] is None for q in rows if q["smile"]),
            "svi": None,
            "rmse_vol_pts": None,
            "atm_iv": None,
            "skew_25d": None,
            "butterfly_k": [],
            "curve": None,
        }
        if len(pts) >= MIN_FIT_POINTS:
            k = np.array([q["k"] for q in pts])
            iv = np.array([q["iv"] for q in pts])
            vega = vega_array(
                first["spot"], np.array([q["strike"] for q in pts]), T, first["interest_rate"], iv
            )
            fit = fit_svi(k, iv, T, weights=vega, start=previous.get(T))
            fits[T] = fit.params
            grid = np.linspace(k.min(), k.max(), CURVE_POINTS)
            entry.update(
                svi=fit.params.as_dict(),
                rmse_vol_pts=fit.rmse_vol_pts,
                atm_iv=atm_vol(fit.params, T),
                skew_25d=skew_25d(fit.params, T),
                butterfly_k=fit.butterfly_k,
                curve={"k": grid.tolist(), "iv": fit.params.iv(grid, T).tolist()},
            )
        expiries.append(entry)

    violations, n_checked = scan_arbitrage_with_counts(chain, tol)
    snapshot = {
        "spot": chain[0]["spot"],
        "rate": chain[0]["interest_rate"],
        "quotes": quotes,
        "expiries": expiries,
        "svi_calendar_crossings": calendar_crossings(list(fits.items())),
        "violations": [asdict(v) for v in violations],
        "checks": {"conditions": n_checked, "quotes": len(chain)},
    }
    return snapshot, fits
