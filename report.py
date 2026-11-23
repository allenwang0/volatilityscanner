"""Compute the report's results (and optionally the portfolio card) from one seeded run.

python report.py --seed 7 --results results/report.json --card site/card.json

The page itself is rendered from the committed results by scripts/build_report.py.
Solver timings come from bench/bench_iv.py (results/bench_iv.json), not from here.
"""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bench.bench_iv import USABLE_VEGA
from engine import CALL, PUT, black_scholes_price_vec, implied_volatility_vec, vega_vec
from main import build_payload
from simulator import IV_NOISE_SD, SIM_STRIKES, MarketSimulator
from surface import otm_mask
from svi import fit_svi

HERE = Path(__file__).parent
ROUND_TRIP_CONTRACTS = 10_000
NOISE_LEVELS = (0.0002, 0.0005, 0.001, 0.002)
NOISE_TICKS = 40
# Round-trip contract ranges (uniform draws): spot, K/S, years, sigma, dividend yield; fixed rate.
ROUND_TRIP_RANGES = {
    "spot": (50.0, 5000.0),
    "moneyness": (0.5, 2.0),
    "years": (1 / 365, 2.0),
    "sigma": (0.05, 2.0),
    "dividend": (0.0, 0.05),
}
ROUND_TRIP_RATE = 0.03
VEGA_FLOOR_DECADE = -15  # vega/S below 1e-15 shares one bin (some draws reach 1e-245)
CARD_EXPIRY_DAYS = 30
CARD_FIT_POINTS = 101


def round_trip_errors(seed: int) -> dict[str, Any]:
    """Price random contracts exactly, solve them back, and bin |sigma_hat - sigma| by vega/S."""
    rng = np.random.default_rng(seed)
    n = ROUND_TRIP_CONTRACTS
    R = ROUND_TRIP_RANGES
    S = rng.uniform(*R["spot"], n)
    K = S * rng.uniform(*R["moneyness"], n)
    T = rng.uniform(*R["years"], n)
    sigma = rng.uniform(*R["sigma"], n)
    q = rng.uniform(*R["dividend"], n)
    flag = rng.choice([CALL, PUT], n).astype(float)
    r = ROUND_TRIP_RATE
    price = black_scholes_price_vec(S, K, T, r, sigma, flag, q)
    solved = implied_volatility_vec(price, S, K, T, r, flag, q)
    rel_vega = vega_vec(S, K, T, r, sigma, q) / S
    identifiable = rel_vega > USABLE_VEGA
    err = np.abs(solved.sigma - sigma)
    ok = identifiable & np.isfinite(err)
    finite = np.isfinite(err)
    bins = []
    top = int(np.ceil(np.log10(rel_vega.max())))
    for e in range(VEGA_FLOOR_DECADE - 1, top):
        lo = 0.0 if e < VEGA_FLOOR_DECADE else 10.0**e  # first bin: everything below the floor
        m = (rel_vega >= lo) & (rel_vega < 10.0 ** (e + 1))
        hit = m & finite
        bins.append(
            {
                "vega_over_spot_lo": lo,
                "vega_over_spot_hi": 10.0 ** (e + 1),
                "contracts": int(m.sum()),
                "solved": int(hit.sum()),
                "max_error": float(err[hit].max()) if hit.any() else None,
                "median_error": float(np.median(err[hit])) if hit.any() else None,
            }
        )
    return {
        "contracts": n,
        "ranges": {key: list(value) for key, value in R.items()},
        "rate": r,
        "usable_vega": USABLE_VEGA,
        "zero_vega": int((rel_vega == 0).sum()),
        "unidentifiable_solved": int((~identifiable & finite).sum()),
        "by_vega_decade": bins,
        "identifiable": int(identifiable.sum()),
        "unsolved_identifiable": int((identifiable & ~np.isfinite(err)).sum()),
        "max_error": float(err[ok].max()),
        "median_error": float(np.median(err[ok])),
        "brent_share": float(solved.brent_fallback.mean()),
    }


def held_out_rmse(chain, iv) -> dict[float, dict[str, float]]:
    """Per expiry: fit SVI on two thirds of the solved OTM quotes, RMSE (vol points) on the rest."""
    otm = otm_mask(chain)
    vega = vega_vec(chain.spot, chain.strike, chain.T, chain.r, np.where(np.isfinite(iv), iv, 0.2))
    out = {}
    for days in np.unique(chain.expiry_days):
        idx = np.flatnonzero((chain.expiry_days == days) & otm & np.isfinite(iv))
        idx = idx[np.argsort(chain.strike[idx])]
        k = np.log(chain.strike[idx] / chain.forward[idx])
        T = float(chain.T[idx[0]])
        train = np.arange(len(idx)) % 3 != 0
        fit = fit_svi(k[train], iv[idx][train], T, vega[idx][train])
        resid = fit.params.implied_vol(k[~train], T) - iv[idx][~train]
        out[float(days)] = {
            "rmse_vol_points": float(np.sqrt(np.mean(resid**2)) * 100),
            "train_quotes": int(train.sum()),
            "test_quotes": int((~train).sum()),
        }
    return out


def noise_sweep(seed: int) -> list[dict[str, float]]:
    """Static-arbitrage violations per tick as independent IV noise grows."""
    rows = []
    for sd in NOISE_LEVELS:
        sim = MarketSimulator(seed=seed, noise_sd=sd)
        counts = [
            len(build_payload(sim.generate_tick(), t)["violations"]) for t in range(NOISE_TICKS)
        ]
        rows.append(
            {
                "noise_sd": sd,
                "per_tick": float(np.mean(counts)),
                "share_of_ticks": float(np.mean([c > 0 for c in counts])),
            }
        )
    return rows


def build(seed: int) -> dict[str, Any]:
    sim = MarketSimulator(seed=seed)
    chain = sim.generate_tick()
    payload = build_payload(chain, 1)
    iv = np.array([np.nan if v is None else v for v in payload["iv"]])
    held_out = held_out_rmse(chain, iv)
    for s in payload["slices"]:
        h = held_out[s["days"]]
        s["held_out_rmse_vol_points"] = h["rmse_vol_points"]
        s["held_out_train_quotes"] = h["train_quotes"]
        s["held_out_test_quotes"] = h["test_quotes"]
    return {
        "seed": seed,
        "label": f"Simulated data, seed {seed}",
        "payload": payload,
        "round_trip": round_trip_errors(seed),
        "noise_sweep": noise_sweep(seed),
    }


def results(report: dict[str, Any]) -> dict[str, Any]:
    """The committed summary the page is rendered from (no per-quote arrays)."""
    p = report["payload"]
    keep = (
        "days",
        "forward",
        "atm_iv",
        "skew_25d",
        "rmse_vol_points",
        "held_out_rmse_vol_points",
        "held_out_train_quotes",
        "held_out_test_quotes",
        "min_density_g",
    )
    return {
        "seed": report["seed"],
        "label": report["label"],
        "simulator": {
            "spot": p["spot"],
            "strikes": len(SIM_STRIKES),
            "strike_range": [float(SIM_STRIKES[0]), float(SIM_STRIKES[-1])],
            "iv_noise_sd": IV_NOISE_SD,
        },
        "tick": {
            "quotes_checked": p["quotes_checked"],
            "conditions_checked": p["conditions_checked"],
            "violations": p["violations"],
            "unsolved": sum(v is None for v in p["iv"]),
        },
        "slices": [{key: s[key] for key in keep} for s in p["slices"]],
        "round_trip": report["round_trip"],
        "noise_sweep": report["noise_sweep"],
    }


def card_slice(report: dict[str, Any]) -> dict[str, Any]:
    slices = report["payload"]["slices"]
    return min(slices, key=lambda s: abs(s["days"] - CARD_EXPIRY_DAYS))


def build_card(report: dict[str, Any], command: str) -> dict[str, Any]:
    p = report["payload"]
    s = card_slice(report)
    spot = p["spot"]
    quotes = []
    for i in range(len(p["strike"])):
        if p["expiry_days"][i] != s["days"] or p["iv"][i] is None:
            continue
        below = p["k"][i] < 0
        if (below and p["type"][i] == "put") or (not below and p["type"][i] == "call"):
            quotes.append([round(p["strike"][i] / spot, 6), round(p["iv"][i] * 100, 4)])
    k = np.linspace(s["curve_k"][0], s["curve_k"][-1], CARD_FIT_POINTS)
    svi = s["svi"]
    d = k - svi["m"]
    w = svi["a"] + svi["b"] * (svi["rho"] * d + np.sqrt(d * d + svi["s"] ** 2))
    fit_iv = np.sqrt(np.maximum(w, 0) / (s["days"] / 365)) * 100
    fit_x = s["forward"] * np.exp(k) / spot
    fit = [[round(float(x), 6), round(float(y), 4)] for x, y in zip(fit_x, fit_iv, strict=True)]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
    ).stdout.strip()
    days = round(s["days"])
    takeaway = (
        f"SVI recovers the {days}-day simulated smile to {s['rmse_vol_points']:.2f} vol points RMSE, "
        f"the level of the {IV_NOISE_SD * 100:.2f}-point noise added to it."
    )
    return {
        "schema_version": 1,
        "repo": "volatilityscanner",
        "sha": sha,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "data_note": f"Seeded simulated chain, seed {report['seed']}, SVI parameters in simulator.py",
        "metric": {
            "label": "SVI fit RMSE, nearest monthly expiry",
            "value": round(s["rmse_vol_points"], 6),
            "unit": "vol points",
            "higher_is_better": False,
        },
        "takeaway": takeaway,
        "chart": {
            "type": "line",
            "x_label": "Moneyness K/S",
            "x_unit": "",
            "x_scale": "linear",
            "y_label": "Implied volatility",
            "y_unit": "%",
            "y_scale": "linear",
            "series": [
                {"name": "Market quotes", "style": "dots", "points": quotes},
                {"name": "SVI fit", "style": "solid", "points": fit},
            ],
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results", type=Path, default=Path("results/report.json"))
    parser.add_argument(
        "--card", type=Path, default=None, help="also write the portfolio card JSON"
    )
    args = parser.parse_args(argv)
    report = build(args.seed)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results(report), indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.results}")
    if args.card:
        command = shlex.join(["python", "report.py", *(sys.argv[1:] if argv is None else argv)])
        card = build_card(report, command)
        args.card.parent.mkdir(parents=True, exist_ok=True)
        args.card.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
        print(f"wrote {args.card}")
    for s in report["payload"]["slices"]:
        print(
            f"{s['days']:g} days: RMSE {s['rmse_vol_points']:.3f}, held-out "
            f"{s['held_out_rmse_vol_points']:.3f} vol pts, skew {s['skew_25d']:.2f}"
        )
    print(f"round trip max error {report['round_trip']['max_error']:.1e}")


if __name__ == "__main__":
    main()
