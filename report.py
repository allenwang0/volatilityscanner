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

from bench.bench_iv import BenchRow, machine
from bench.bench_iv import run as run_benchmark
from engine import CALL, PUT, black_scholes_price_vec, implied_volatility_vec, vega_vec
from main import build_payload
from simulator import MarketSimulator
from surface import otm_mask
from svi import fit_svi

HERE = Path(__file__).parent
PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
ROUND_TRIP_CONTRACTS = 10_000
NOISE_LEVELS = (0.0002, 0.0005, 0.001, 0.002)
NOISE_TICKS = 40
CARD_EXPIRY_DAYS = 30
CARD_FIT_POINTS = 101


def round_trip_errors(seed: int) -> dict[str, Any]:
    """Price random contracts exactly, solve them back, and bin log10 |sigma_hat - sigma|."""
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
    log_err = np.log10(np.maximum(err[ok], 1e-17))
    return {
        "log10_error": np.round(log_err, 3).tolist(),
        "identifiable": int(identifiable.sum()),
        "unsolved_identifiable": int((identifiable & ~np.isfinite(err)).sum()),
        "max_error": float(err[ok].max()),
        "median_error": float(np.median(err[ok])),
        "brent_share": float(solved.brent_fallback.mean()),
    }


def held_out_rmse(chain, iv) -> dict[float, float]:
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
        out[float(days)] = float(np.sqrt(np.mean(resid**2)) * 100)
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
        s["held_out_rmse_vol_points"] = held_out[s["days"]]
    bench: list[BenchRow] = run_benchmark(seed)
    return {
        "seed": seed,
        "label": f"Simulated data, seed {seed}",
        "payload": payload,
        "benchmark": [row.__dict__ for row in bench],
        "machine": machine(),
        "round_trip": round_trip_errors(seed),
        "noise_sweep": noise_sweep(seed),
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
        f"SVI fits the {days}-day simulated smile to {s['rmse_vol_points']:.2f} vol points RMSE; "
        f"25-delta skew is {s['skew_25d']:.1f} vol points."
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


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th scope='col'>{html.escape(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report: dict[str, Any]) -> str:
    p = report["payload"]
    rt = report["round_trip"]
    bench = _table(
        [
            "Options",
            "Scalar loop (ms)",
            "Vectorized (ms)",
            "Brent fallback",
            "Max diff, vega > 1e-4 S",
        ],
        [
            [
                f"{b['n']:,}",
                f"{b['scalar_ms']:.1f}",
                f"{b['vectorized_ms']:.1f}",
                f"{b['brent_share']:.1%}",
                f"{b['max_abs_diff']:.1e}",
            ]
            for b in report["benchmark"]
        ],
    )
    svi_rows = _table(
        ["Expiry", "ATM IV", "25-delta skew", "SVI RMSE", "Held-out RMSE", "min g(k)"],
        [
            [
                f"{s['days']:g} days",
                f"{s['atm_iv']:.2%}",
                f"{s['skew_25d']:.2f} vol pts",
                f"{s['rmse_vol_points']:.3f} vol pts",
                f"{s['held_out_rmse_vol_points']:.3f} vol pts",
                f"{s['min_density_g']:.3f}",
            ]
            for s in p["slices"]
        ],
    )
    noise = _table(
        ["IV noise sd", "Violations per tick", "Ticks with a violation"],
        [
            [f"{n['noise_sd']:.4f}", f"{n['per_tick']:.2f}", f"{n['share_of_ticks']:.0%}"]
            for n in report["noise_sweep"]
        ],
    )
    style = (HERE / "static" / "style.css").read_text()
    charts = (HERE / "static" / "charts.js").read_text()
    data = json.dumps(report, allow_nan=False).replace("</", "<\\/")
    label = html.escape(report["label"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volatility Scanner Report</title>
<script src="{PLOTLY}"></script>
<style>{style}
.prose {{ max-width: 760px; color: var(--muted); font-size: 0.92rem; }}
.label {{ font-weight: 600; color: var(--warn-ink); background: var(--warn-bg); padding: 2px 8px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <h1>Volatility Scanner report</h1>
  <span class="label" id="data-label">{label}</span>
  <div class="controls"><button type="button" id="theme">Theme</button></div>
</header>
<p class="prose">One tick of the seeded simulator: {p["quotes_checked"]} quotes (calls and puts, 20 strikes,
expiries {", ".join(f"{s['days']:g}" for s in p["slices"])} days), spot {p["spot"]:.2f}. Every number on this
page comes from <code>python report.py --seed {report["seed"]}</code>; the live app streams the same
analysis twice a second.</p>
<main class="grid">
  <section class="panel">
    <h2>Smile per expiry</h2>
    <p class="takeaway" id="smile-takeaway"></p>
    <div id="smile" class="chart"></div>
    <ul id="failed" class="failed"></ul>
  </section>
  <section class="panel">
    <h2>ATM term structure</h2>
    <p class="takeaway" id="term-takeaway"></p>
    <div id="term" class="chart small"></div>
    <h2>25-delta skew</h2>
    <p class="takeaway" id="skew-takeaway"></p>
    <div id="skew" class="chart small"></div>
  </section>
  <section class="panel wide">
    <h2>Static arbitrage</h2>
    <div id="arb" class="table-wrap"></div>
  </section>
  <section class="panel wide">
    <h2>SVI fit per expiry</h2>
    <p class="takeaway">Held-out RMSE: fit on two thirds of the strikes, scored on the rest.
    g(k) is the Gatheral-Jacquier density on 201 points; negative means butterfly arbitrage.</p>
    <div class="table-wrap">{svi_rows}</div>
  </section>
  <section class="panel">
    <h2>IV solver round trip</h2>
    <p class="takeaway" id="rt-takeaway">{rt["identifiable"]:,} of {ROUND_TRIP_CONTRACTS:,} random
    contracts have vega above 1e-4 S; max error {rt["max_error"]:.1e}, median {rt["median_error"]:.1e},
    {rt["unsolved_identifiable"]} unsolved. Exact recoveries (error 0) are drawn at -17.</p>
    <div id="hist" class="chart small"></div>
  </section>
  <section class="panel">
    <h2>Vectorized solver benchmark</h2>
    <p class="takeaway">{html.escape(report["machine"])}, median of 5 runs.</p>
    <div class="table-wrap">{bench}</div>
    <h2 style="margin-top:16px">Quote noise and false alarms</h2>
    <p class="takeaway">Independent IV noise on each quote breaks butterfly convexity once it
    exceeds the butterfly's value; {NOISE_TICKS} ticks per level.</p>
    <div class="table-wrap">{noise}</div>
  </section>
</main>
<script type="application/json" id="report-data">{data}</script>
<script>{charts}</script>
<script>
(function () {{
  var report = JSON.parse(document.getElementById("report-data").textContent);
  var p = report.payload, $ = function (id) {{ return document.getElementById(id); }};
  var state = {{ sortDesc: true }}, highlight = null;
  function draw() {{
    var t = VS.takeaways(p);
    $("smile-takeaway").textContent = t.smile;
    $("term-takeaway").textContent = t.term;
    $("skew-takeaway").textContent = t.skew;
    VS.renderSmile($("smile"), p, highlight);
    VS.renderFailed($("failed"), p);
    VS.renderTerm($("term"), p);
    VS.renderSkew($("skew"), p);
    VS.renderArbTable($("arb"), p, state, function (v) {{
      highlight = VS.highlightFor(p, v); VS.renderSmile($("smile"), p, highlight);
    }});
    var ink = getComputedStyle(document.documentElement).getPropertyValue("--ink");
    var grid = getComputedStyle(document.documentElement).getPropertyValue("--grid");
    Plotly.react($("hist"), [{{ type: "histogram", x: report.round_trip.log10_error, marker: {{ color: "#0072B2" }},
      hovertemplate: "log10 error %{{x}}<br>%{{y}} contracts<extra></extra>" }}], {{
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {{ color: ink }},
      margin: {{ l: 56, r: 12, t: 8, b: 48 }}, bargap: 0.05,
      xaxis: {{ title: {{ text: "log10 |sigma_hat - sigma|" }}, gridcolor: grid }},
      yaxis: {{ title: {{ text: "Contracts" }}, gridcolor: grid }} }}, {{ responsive: true, displaylogo: false }});
  }}
  $("theme").addEventListener("click", function () {{
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    draw();
  }});
  draw();
}})();
</script>
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
    args.out.write_text(render_html(report))
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
