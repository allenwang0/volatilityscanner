"""Build the static report (and optionally the portfolio card) from one seeded run.

python report.py --seed 7 --out docs/index.html --card site/card.json
"""

import argparse
import html
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from bench.bench_iv import REPEATS, BenchRow, machine
from bench.bench_iv import run as run_benchmark
from engine import CALL, PUT, black_scholes_price_vec, implied_volatility_vec, vega_vec
from main import build_payload
from simulator import IV_NOISE_SD, MarketSimulator
from surface import otm_mask
from svi import fit_svi

HERE = Path(__file__).parent
ROUND_TRIP_CONTRACTS = 10_000
NOISE_LEVELS = (0.0002, 0.0005, 0.001, 0.002)
NOISE_TICKS = 40
CARD_EXPIRY_DAYS = 30
CARD_FIT_POINTS = 101


def round_trip_errors(seed: int) -> dict[str, Any]:
    """Price random contracts exactly, solve them back, and summarize |sigma_hat - sigma|."""
    rng = np.random.default_rng(seed)
    n = ROUND_TRIP_CONTRACTS
    S = rng.uniform(50, 5000, n)
    K = S * rng.uniform(0.5, 2.0, n)
    T = rng.uniform(1 / 365, 2.0, n)
    sigma = rng.uniform(0.05, 2.0, n)
    q = rng.uniform(0.0, 0.05, n)
    flag = rng.choice([CALL, PUT], n).astype(float)
    r = 0.03
    price = black_scholes_price_vec(S, K, T, r, sigma, flag, q)
    solved = implied_volatility_vec(price, S, K, T, r, flag, q)
    identifiable = vega_vec(S, K, T, r, sigma, q) > 1e-4 * S
    err = np.abs(solved.sigma - sigma)
    ok = identifiable & np.isfinite(err)
    return {
        "contracts": n,
        "identifiable": int(identifiable.sum()),
        "exact": int((err[ok] == 0).sum()),
        "unsolved_identifiable": int((identifiable & ~np.isfinite(err)).sum()),
        "max_error": float(err[ok].max()),
        "median_error": float(np.median(err[ok])),
        "brent_share": float(solved.brent_fallback.mean()),
    }


def held_out_fit(chain, iv) -> dict[float, dict[str, Any]]:
    """Per expiry: fit SVI on two thirds of the solved OTM quotes, RMSE (vol points) on the rest.

    Also returns every quote's residual (quote IV - fit, vol points) against that same fit,
    flagged 1 if the strike was held out, so the report can plot what the RMSE summarizes.
    """
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
        resid = (iv[idx] - fit.params.implied_vol(k, T)) * 100
        out[float(days)] = {
            "rmse": float(np.sqrt(np.mean(resid[~train] ** 2))),
            "train_rmse": float(np.sqrt(np.mean(resid[train] ** 2))),
            "points": [
                [round(float(a), 5), round(float(b), 5), int(not t)]
                for a, b, t in zip(k, resid, train, strict=True)
            ],
        }
    return out


def noise_sweep(seed: int) -> list[dict[str, float]]:
    """Static-arbitrage violations per tick as independent IV noise grows."""
    rows = []
    for sd in NOISE_LEVELS:
        sim = MarketSimulator(seed=seed, noise_sd=sd)
        found = [build_payload(sim.generate_tick(), t)["violations"] for t in range(NOISE_TICKS)]
        counts = [len(v) for v in found]
        flat = [v for tick in found for v in tick]
        rows.append(
            {
                "noise_sd": sd,
                "ticks": NOISE_TICKS,
                "per_tick": float(np.mean(counts)),
                "share_of_ticks": float(np.mean([c > 0 for c in counts])),
                "ticks_flagged": sum(c > 0 for c in counts),
                "violations": len(flat),
                "butterfly": sum(v["type"] == "butterfly" for v in flat),
                "spread_checked": sum(v["exceeds_spread"] is not None for v in flat),
            }
        )
    return rows


def build(seed: int) -> dict[str, Any]:
    sim = MarketSimulator(seed=seed)
    chain = sim.generate_tick()
    payload = build_payload(chain, 1)
    iv = np.array([np.nan if v is None else v for v in payload["iv"]])
    held_out = held_out_fit(chain, iv)
    for s in payload["slices"]:
        s["held_out_rmse_vol_points"] = held_out[s["days"]]["rmse"]
        s["held_out_train_rmse_vol_points"] = held_out[s["days"]]["train_rmse"]
        s["residuals"] = held_out[s["days"]]["points"]
    bench: list[BenchRow] = run_benchmark(seed)
    return {
        "seed": seed,
        "sha": git_sha(),
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "default_noise_sd": IV_NOISE_SD,
        "repeats": REPEATS,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "label": f"Simulated data, seed {seed}",
        "payload": payload,
        "benchmark": [row.__dict__ for row in bench],
        "machine": machine(),
        "round_trip": round_trip_errors(seed),
        "noise_sweep": noise_sweep(seed),
    }


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
    ).stdout.strip()


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
    days = round(s["days"])
    takeaway = (
        f"SVI fits the {days}-day simulated smile to {s['held_out_rmse_vol_points']:.3f} vol "
        f"points on held-out strikes ({s['rmse_vol_points']:.3f} in-sample)."
    )
    return {
        "schema_version": 1,
        "repo": "volatilityscanner",
        "sha": report["sha"],
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "data_note": f"Seeded simulated chain, seed {report['seed']}, SVI parameters in simulator.py",
        "metric": {
            "label": "SVI held-out RMSE, nearest monthly expiry",
            "value": round(s["held_out_rmse_vol_points"], 6),
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
                {"name": "Simulated quotes", "style": "dots", "points": quotes},
                {"name": "SVI fit", "style": "solid", "points": fit},
            ],
        },
    }


NBSP = " "
REPO = "https://github.com/allenwang0/volatilityscanner"


def _table(head: list[str], rows: list[list[str]], caption: str) -> str:
    # First column is a row header; the rest are numbers, right-aligned by the kit.
    th = "".join(f"<th scope='col'>{html.escape(h)}</th>" for h in head)
    body = "".join(
        f"<tr><th scope='row'>{row[0]}</th>" + "".join(f"<td>{c}</td>" for c in row[1:]) + "</tr>"
        for row in rows
    )
    return (
        f"<div class='tablewrap' tabindex='0' role='region' aria-label='{html.escape(caption)}'>"
        f"<table><caption>{html.escape(caption)}</caption><thead><tr>{th}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _sci(x: float) -> str:
    """2.5e-12 -> 2.5 × 10<sup>−12</sup> (HTML)."""
    mant, exp = f"{x:.1e}".split("e")
    if int(exp) >= 0:
        return f"{x:.1f}"
    return f"{mant}{NBSP}×{NBSP}10<sup>−{abs(int(exp))}</sup>"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}{NBSP}%" if x == 0 or x >= 0.1 else f"{x * 100:.1f}{NBSP}%"


def _range(values: list[float]) -> str:
    return f"{min(values):.3f}–{max(values):.3f}"


def _pts(sd: float) -> str:
    """IV units -> vol points."""
    return f"{sd * 100:g}"


def render_html(report: dict[str, Any]) -> str:
    p = report["payload"]
    rt = report["round_trip"]
    slices = p["slices"]
    sweep = report["noise_sweep"]
    default_sd = report["default_noise_sd"]
    seed = report["seed"]
    sha = report["sha"][:7]
    # The first tested noise level above the simulator default that produced any flag.
    alarm = next(n for n in sweep if n["noise_sd"] > default_sd and n["share_of_ticks"] > 0)
    quiet = next(n for n in sweep if n["noise_sd"] == default_sd)
    in_sample = [s["rmse_vol_points"] for s in slices]
    held = [s["held_out_rmse_vol_points"] for s in slices]
    worst = max(slices, key=lambda s: s["held_out_rmse_vol_points"])
    worst_ratio = worst["held_out_rmse_vol_points"] / worst["rmse_vol_points"]
    worst_train_n = sum(1 for q in worst["residuals"] if not q[2])
    others_max = max(s["held_out_rmse_vol_points"] for s in slices if s is not worst)
    big = max(report["benchmark"], key=lambda b: b["n"])
    speedup = big["scalar_ms"] / big["vectorized_ms"]
    expiries = ", ".join(f"{s['days']:g}" for s in slices)
    machine_txt = html.escape(report["machine"])

    noise_table = _table(
        ["IV noise sd", "Vol points", "Ticks flagged", "Share", "Violations per tick", "Butterfly"],
        [
            [
                f"{n['noise_sd']:.4f}" + (" (default)" if n["noise_sd"] == default_sd else ""),
                _pts(n["noise_sd"]),
                f"{n['ticks_flagged']} of {n['ticks']}",
                _pct(n["share_of_ticks"]),
                f"{n['per_tick']:.2f}",
                f"{n['butterfly']} of {n['violations']}",
            ]
            for n in sweep
        ],
        f"Static-arbitrage violations by quote noise, {quiet['ticks']} ticks per level, seed {seed}",
    )
    fit_table = _table(
        [
            "Expiry",
            "In-sample RMSE",
            "⅔-fit train RMSE",
            "Held-out RMSE",
            "Held-out ÷ in-sample",
            "ATM IV",
            "25-delta skew",
            "min g(k)",
        ],
        [
            [
                f"{s['days']:g} days",
                f"{s['rmse_vol_points']:.3f}",
                f"{s['held_out_train_rmse_vol_points']:.3f}",
                f"{s['held_out_rmse_vol_points']:.3f}",
                f"{s['held_out_rmse_vol_points'] / s['rmse_vol_points']:.2f}×",
                f"{s['atm_iv'] * 100:.1f}{NBSP}%",
                f"{s['skew_25d']:.2f}",
                f"{s['min_density_g']:.3f}",
            ]
            for s in slices
        ],
        "SVI fit quality per expiry; RMSE and skew in vol points",
    )
    bench_table = _table(
        ["Options", "Scalar loop (ms)", "Vectorized (ms)", "Speedup", "Brent fallback", "Max diff"],
        [
            [
                f"{b['n']:,}",
                f"{b['scalar_ms']:.1f}",
                f"{b['vectorized_ms']:.1f}",
                f"{b['scalar_ms'] / b['vectorized_ms']:.2f}×",
                _pct(b["brent_share"]),
                _sci(b["max_abs_diff"]),
            ]
            for b in report["benchmark"]
        ],
        f"Solver timing, median of {report['repeats']} runs, {report['machine']}",
    )
    solver_table = _table(
        ["Round trip", "Value"],
        [
            ["Random contracts", f"{rt['contracts']:,}"],
            ["Identifiable (vega above 10<sup>−4</sup> × spot)", f"{rt['identifiable']:,}"],
            ["Identifiable but unsolved", f"{rt['unsolved_identifiable']:,}"],
            ["Recovered exactly (error 0)", f"{rt['exact']:,}"],
            ["Median absolute error", _sci(rt["median_error"])],
            ["Max absolute error", _sci(rt["max_error"])],
            ["Solved by the Brent fallback", _pct(rt["brent_share"])],
        ],
        f"IV solver round trip, seed {seed}; errors in volatility units",
    )
    env_table = _table(
        ["Field", "Value"],
        [
            ["Machine", machine_txt],
            ["NumPy", report["versions"]["numpy"]],
            ["SciPy", report["versions"]["scipy"]],
            ["Seed", str(seed)],
            ["Commit", f"<span class='mono'>{sha}</span>"],
            ["Built", report["date"]],
        ],
        "Build environment",
    )
    chart_data = {
        "noise": [
            {
                "label": f"{n['noise_sd']:.4f}"
                + (" default" if n["noise_sd"] == default_sd else ""),
                "share": round(n["share_of_ticks"] * 100, 1),
            }
            for n in sweep
        ],
        "residuals": [
            {
                "days": s["days"],
                "train": round(s["held_out_train_rmse_vol_points"], 3),
                "held_out": round(s["held_out_rmse_vol_points"], 3),
                "points": s["residuals"],
            }
            for s in slices
        ],
    }
    data = json.dumps(report, allow_nan=False).replace("</", "<\\/")
    charts = json.dumps(chart_data, allow_nan=False).replace("</", "<\\/")
    alarm_pct = _pct(alarm["share_of_ticks"])
    spread_checked = sum(n["spread_checked"] for n in sweep)
    flags_total = sum(n["violations"] for n in sweep)
    held_label = ", ".join(
        f"{s['days']:g} days {s['held_out_rmse_vol_points']:.3f}" for s in slices
    )
    title = "Volatility Scanner"
    desc = (
        "An implied-volatility solver, SVI smile fits and static-arbitrage checks, "
        f"run on a seeded simulated option chain (seed {seed})."
    )
    toggle = (
        '<div class="seg" role="radiogroup" aria-label="View" data-view-toggle>'
        '<button role="radio" aria-checked="true" data-value="chart">Chart</button>'
        '<button role="radio" aria-checked="false" data-value="table">Table</button></div>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title} · Allen Wang</title>
<meta name="description" content="{html.escape(desc)}">
<meta name="theme-color" content="#f5f5f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0b0b0b" media="(prefers-color-scheme: dark)">
<script>try{{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="kit/report.css">
<link rel="stylesheet" href="report.local.css">
<script src="kit/report.js" defer></script>
<script src="kit/charts.js" defer></script>
<script src="report.local.js" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="bar">
  <div class="wrap">
    <a class="back" href="https://allenwang0.github.io/">← Allen Wang</a>
    <span class="title" aria-hidden="true">{title}</span>
    <a class="pill" href="{REPO}">Code ↗</a>
  </div>
</header>
<main id="main">
  <section class="hero wrap" aria-labelledby="h1">
    <p class="eyebrow">Options analytics · Python</p>
    <h1 id="h1">Arbitrage checks fire on {alarm_pct} of ticks once quotes carry {alarm["noise_sd"]:g} IV noise</h1>
    <p class="lede">An implied-volatility solver, SVI smile fits and static-arbitrage checks, run
    on a seeded simulated option chain. The solver and fits hold up. The arbitrage checks need a
    noise threshold before a live flag means anything.</p>
    <p class="conditions" id="data-label"><span class="key">Synthetic data: simulated option chain, seed {seed}.</span><span>{p["quotes_checked"]} quotes, expiries {expiries} days</span><span>One seed</span></p>
    <div class="ratio"><span class="fig">{alarm_pct}</span><span class="cap">of {alarm["ticks"]} ticks
    had at least one false butterfly flag at noise sd {alarm["noise_sd"]:g}
    ({_pts(alarm["noise_sd"])} vol points). At the simulator's default {default_sd:g},
    {quiet["ticks_flagged"]} of {quiet["ticks"]} did.</span></div>
  </section>

  <section class="wrap" aria-label="Key results">
    <div class="bento">
      <div class="card span-6"><span class="label">SVI fit error, held-out strikes</span><span class="metric">{_range(held)} <span class="unit">vol pts</span></span><span class="cmp">vs {_range(in_sample)} in-sample. The {worst["days"]:g}-day fit is {worst_ratio:.2f}× worse held out.</span><span class="foot">RMSE per expiry · every third strike held out · one tick</span></div>
      <div class="card span-6"><span class="label">IV solver round trip, max error</span><span class="metric">{_sci(rt["max_error"])}</span><span class="cmp">{rt["identifiable"]:,} of {rt["contracts"]:,} contracts identifiable; {rt["unsolved_identifiable"]} unsolved</span><span class="foot">Random contracts priced exactly, then solved back</span></div>
      <div class="card span-6"><span class="label">Arbitrage checks at default noise</span><span class="metric">{len(p["violations"])} <span class="unit">of {p["conditions_checked"]}</span></span><span class="cmp">checks violated on this tick, IV noise sd {default_sd:g}</span><span class="foot">{p["quotes_checked"]} quotes on one simulated tick</span></div>
      <div class="card span-6"><span class="label">Vectorized solver speedup</span><span class="metric">{speedup:.2f}×</span><span class="cmp">{big["vectorized_ms"]:.1f} vs {big["scalar_ms"]:.1f} ms for {big["n"]:,} options in a scalar loop</span><span class="foot">{machine_txt} · median of {report["repeats"]} runs</span></div>
    </div>
  </section>

  <section class="s wrap" id="noise" aria-labelledby="h-noise">
    <p class="eyebrow">How noise-sensitive are the checks?</p>
    <h2 id="h-noise">Quote noise of {_pts(alarm["noise_sd"])} vol points trips the butterfly check on {alarm_pct} of ticks</h2>
    <p class="lede">Each quote's IV gets independent noise. Once that noise exceeds a butterfly's
    value, the convexity check fires even though the underlying surface has no arbitrage.</p>
    <figure class="figure" aria-labelledby="f-noise">
      <div class="fig-head"><div><h3 id="f-noise">False alarms rise from none to every tick as noise grows from {sweep[0]["noise_sd"]:g} to {sweep[-1]["noise_sd"]:g}</h3><p class="sub">Share of ticks with at least one violation · {alarm["ticks"]} ticks per level · seed {seed}</p></div>
        {toggle}</div>
      <div id="c-noise"></div><div id="t-noise" class="rk-table">{noise_table}</div>
      <figcaption>Each bar is the share of {alarm["ticks"]} simulated ticks flagged at one IV noise
      sd (volatility units; ×100 for vol points). Single seed, so no interval.</figcaption>
    </figure>
    <p class="caveat"><b>Every flag here is a false alarm.</b> The simulated surface passes every
    check at the default noise ({quiet["ticks_flagged"]} of {quiet["ticks"]} ticks flagged). The
    chain has no bid-ask spread, so {spread_checked} of {flags_total} flags could be compared with
    the cost of trading them.</p>
  </section>

  <section class="s wrap" id="fits" aria-labelledby="h-fits">
    <p class="eyebrow">How well does SVI fit?</p>
    <h2 id="h-fits">SVI misses held-out strikes by {_range(held)} vol points; the {worst["days"]:g}-day fit is {worst_ratio:.2f}× worse than in-sample</h2>
    <p class="lede">In-sample error is the fit on all out-of-the-money strikes. Held-out error
    refits on two thirds of them and scores every third strike. Beyond {worst["days"]:g} days,
    held-out error stays at or below {others_max:.3f} vol points. The {worst["days"]:g}-day gap is
    not overfitting: the two-thirds fit already misses its own {worst_train_n} training strikes by
    {worst["held_out_train_rmse_vol_points"]:.3f}, so with fewer strikes the short-dated smile is
    fit poorly everywhere.</p>
    <figure class="figure" aria-labelledby="f-fits">
      <div class="fig-head"><div><h3 id="f-fits">Held-out residuals are widest on the {worst["days"]:g}-day smile</h3><p class="sub">Quote IV minus SVI fit · vol points · one panel per expiry</p></div>
        {toggle}</div>
      <div id="c-fits" role="img" aria-label="Residuals per expiry. Held-out RMSE in vol points: {held_label}."></div><div id="t-fits" class="rk-table">{fit_table}</div>
      <figcaption>Residuals against the two-thirds fit. Filled dots are its training strikes; open
      dots are held out. All panels share both scales. The table adds ATM IV, 25-delta skew and the minimum
      Gatheral-Jacquier density g(k) on 201 points (a negative value would mean butterfly
      arbitrage).</figcaption>
    </figure>
    <p class="caveat"><b>SVI is expected to fit here.</b> The simulator generates SVI-shaped smiles
    and adds noise of sd {_pts(default_sd)} vol points, so in-sample error mostly measures that
    noise. This tests the fitting code, not whether SVI suits real markets.</p>
  </section>

  <section class="s wrap" id="solver" aria-labelledby="h-solver">
    <p class="eyebrow">Is the IV solver exact?</p>
    <h2 id="h-solver">The solver recovers every identifiable contract to within {_sci(rt["max_error"])}</h2>
    <p class="lede">{rt["contracts"]:,} random contracts are priced with Black-Scholes and solved
    back. {rt["contracts"] - rt["identifiable"]:,} have too little vega for their volatility to be
    identifiable, so they are excluded rather than reported as solved.</p>
    {solver_table}
  </section>

  <section class="s wrap" id="speed" aria-labelledby="h-speed">
    <p class="eyebrow">Does vectorizing pay?</p>
    <h2 id="h-speed">The vectorized solver is {speedup:.2f}× faster than a scalar loop on {big["n"]:,} options</h2>
    <p class="lede">Both solvers agree to within {_sci(big["max_abs_diff"])}. The gain is modest, and
    {_pct(big["brent_share"])} of these options need the Brent fallback.</p>
    {bench_table}
    <p class="caveat"><b>Timing depends on the machine.</b> This build ran on {machine_txt}. The
    Pages build re-times on every deploy, so these numbers change with the runner.</p>
  </section>

  <section class="s wrap" id="method" aria-labelledby="h-method">
    <p class="eyebrow">Methodology</p>
    <h2 id="h-method">How this was measured</h2>
    <div class="prose" style="margin-top:24px">
      <p>The simulator draws one tick of {p["quotes_checked"]} calls and puts across {len(slices)}
      expiries around spot {p["spot"]:,.0f}. Each expiry's true smile is a raw-SVI curve from
      <span class="mono">simulator.py</span>, and each quote's IV gets independent normal noise.</p>
      <p>Implied volatilities are solved with Newton steps and a Brent fallback, fit per expiry
      with SVI, and checked for static arbitrage. The noise sweep reruns the same seed at
      {len(sweep)} noise levels for {alarm["ticks"]} ticks each.</p>
    </div>
    <div class="two">
      <div><h3>What this shows</h3><ul><li>The solver is exact to {_sci(rt["max_error"])} wherever vega is identifiable.</li><li>SVI fits a simulated smile to {_range(held)} vol points on held-out strikes.</li><li>Butterfly checks need a noise threshold: {_pts(alarm["noise_sd"])} vol points of quote noise flags {alarm_pct} of ticks.</li></ul></div>
      <div><h3>What it doesn't show</h3><ul><li>Behavior on market quotes. Every number here is simulated.</li><li>Whether SVI is the right model, since the simulator generates SVI smiles.</li><li>Extrapolation: held-out strikes sit between training strikes.</li><li>Variation across seeds. This is one seed and one tick.</li></ul></div>
    </div>
  </section>

  <section class="s wrap" id="reproduce" aria-labelledby="h-repro">
    <p class="eyebrow">Reproduce</p>
    <h2 id="h-repro">Run it yourself</h2>
    <div class="codewrap"><pre><code>git clone {REPO}.git && cd volatilityscanner
make setup PYTHON=python3.12
.venv/bin/python report.py --seed {seed} --out docs/index.html   # this page
.venv/bin/python main.py --seed {seed}                           # live dashboard</code></pre></div>
    <details>
      <summary>Environment for this build · 6 rows</summary>
      {env_table}
    </details>
  </section>
</main>
<footer><div class="wrap"><span><a href="https://allenwang0.github.io/">← Allen Wang</a><a href="{REPO}">Code ↗</a></span><span>Data from commit <span class="mono">{sha}</span> · {report["date"]}</span></div></footer>
<script type="application/json" id="report-data">{data}</script>
<script type="application/json" id="chart-data">{charts}</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("docs/index.html"))
    parser.add_argument(
        "--card", type=Path, default=None, help="also write the portfolio card JSON"
    )
    args = parser.parse_args(argv)
    report = build(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(report), encoding="utf-8")
    print(f"wrote {args.out}")
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
