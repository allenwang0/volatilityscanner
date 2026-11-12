"""Build a self-contained HTML report from one seeded simulated tick.

Run: python report.py --seed 7 --out docs/index.html
Writes the HTML file and, next to it, report.json with every number shown.
"""

import argparse
import html
import json
import math
import os
import platform
from datetime import UTC, datetime

import numpy as np
from jinja2 import Environment, FileSystemLoader

from analysis import analyze_chain
from bench.bench_iv import machine
from bench.bench_iv import run as run_benchmark
from engine import black_scholes_price_array, implied_volatility_array, vega_array
from simulator import DEFAULT_JITTER, MarketSimulator

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND_TRIP_N = 10_000
EXACT_FLOOR = -17.0  # log10 error plotted for an exact (zero-error) round trip


def round_trip(seed: int) -> dict:
    """Price 10,000 random contracts with Black-Scholes and solve them back."""
    rng = np.random.default_rng(seed)
    n = ROUND_TRIP_N
    S = rng.uniform(50, 5000, n)
    K = S * rng.uniform(0.5, 2, n)
    T = rng.uniform(1 / 365, 2, n)
    sigma = rng.uniform(0.05, 2, n)
    q = rng.uniform(0, 0.05, n)
    flag = rng.choice([1.0, -1.0], n)
    r = np.full(n, 0.05)
    price = black_scholes_price_array(S, K, T, r, sigma, flag, q)
    res = implied_volatility_array(price, S, K, T, r, flag, q)
    identifiable = vega_array(S, K, T, r, sigma, q) > 1e-4 * S
    err = np.abs(res.sigma - sigma)
    ok = identifiable & np.isfinite(err)
    log_err = np.log10(np.maximum(err[ok], 10**EXACT_FLOOR))
    return {
        "n": n,
        "identifiable": int(identifiable.sum()),
        "solved_identifiable": int(ok.sum()),
        "max_error": float(err[ok].max()),
        "median_error": float(np.median(err[ok])),
        "log10_errors": log_err.round(3).tolist(),
        "brent_share": float(res.brent.mean()),
    }


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build(seed: int, jitter: float) -> tuple[str, dict]:
    sim = MarketSimulator(seed=seed, jitter=jitter)
    snapshot, _ = analyze_chain(sim.generate_tick())
    payload = {"kind": "tick", "tick": 1, "source": sim.describe(), **snapshot}
    bench = run_benchmark()
    rt = round_trip(seed)
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    results = {
        "seed": seed,
        "jitter": jitter,
        "generated_at": generated,
        "machine": machine(),
        "expiries": [
            {k: e[k] for k in ("days", "rmse_vol_pts", "atm_iv", "skew_25d", "svi", "butterfly_k")}
            for e in snapshot["expiries"]
        ],
        "violations": len(snapshot["violations"]),
        "violations_by_type": {
            t: sum(v["type"] == t for v in snapshot["violations"])
            for t in {v["type"] for v in snapshot["violations"]}
        },
        "checks": snapshot["checks"],
        "benchmark": bench,
        "round_trip": {k: v for k, v in rt.items() if k != "log10_errors"},
    }

    svi_rows = [
        [
            f"{e['days']:.0f}",
            f"{e['rmse_vol_pts']:.2f}",
            f"{e['atm_iv'] * 100:.2f}%",
            f"{e['skew_25d']:.2f}",
            ", ".join(f"{k}={v:.4g}" for k, v in e["svi"].items()),
        ]
        for e in snapshot["expiries"]
        if e["svi"]
    ]
    bench_rows = [
        [
            f"{b['n']:,}",
            f"{b['scalar_ms']:.1f}",
            f"{b['vector_ms']:.1f}",
            f"{b['speedup']:.1f}x",
            f"{b['brent_share']:.1%}",
        ]
        for b in bench
    ]
    hist = {
        "type": "histogram",
        "x": rt["log10_errors"],
        "xbins": {"size": 0.25},
        "marker": {"color": "#0072B2"},
        "hovertemplate": "log10 error %{x}<br>%{y} contracts<extra></extra>",
    }
    after = f"""
    <section class="panel section">
      <h2>SVI fit per expiry</h2>
      {table(["Days", "RMSE (vol pts)", "ATM IV", "25-delta skew (vol pts)", "Raw SVI parameters"], svi_rows)}
    </section>
    <section class="panel section">
      <h2>IV solver round trip</h2>
      <div id="roundtrip" class="chart small"></div>
      <p class="takeaway">{rt["n"]:,} random contracts (S 50 to 5,000, K/S 0.5 to 2, T 1 day to 2 years,
      sigma 5% to 200%, q 0 to 5%, calls and puts), priced with Black-Scholes and solved back.
      {rt["identifiable"]:,} have vega above 1e-4 &times; S, where sigma is identifiable; all
      {rt["solved_identifiable"]:,} of those solve, with a largest error of {rt["max_error"]:.1e}.
      Exact round trips are plotted at {EXACT_FLOOR:.0f}.</p>
    </section>
    <section class="panel section">
      <h2>Benchmark: scalar loop vs vectorized solver</h2>
      {table(["Options", "Scalar loop (ms)", "Vectorized (ms)", "Speed-up", "Brent fallback"], bench_rows)}
      <p class="takeaway">Median of 5 runs, seed 7, {html.escape(machine())}.</p>
    </section>
    <script>
    document.addEventListener("DOMContentLoaded", function () {{
      var root = getComputedStyle(document.documentElement);
      var css = function (n) {{ return root.getPropertyValue(n).trim(); }};
      Plotly.newPlot("roundtrip", [{json.dumps(hist)}], {{
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {{ color: css("--text") }},
        margin: {{ l: 56, r: 12, t: 8, b: 44 }},
        xaxis: {{ title: {{ text: "log10 |sigma_hat - sigma|" }}, gridcolor: css("--grid") }},
        yaxis: {{ title: {{ text: "Contracts" }}, gridcolor: css("--grid") }}
      }}, {{ responsive: true, displaylogo: false }});
    }});
    </script>
    """
    before = (
        f'<div class="banner warn"><b>Simulated data, seed {seed}</b>&nbsp;(IV jitter {jitter}). '
        f"Generated {generated} by <code>python report.py --seed {seed}</code>.</div>"
    )
    env = Environment(loader=FileSystemLoader(os.path.join(HERE, "templates")), autoescape=True)
    with open(os.path.join(HERE, "static", "dashboard.css")) as f:
        css = f.read()
    with open(os.path.join(HERE, "static", "dashboard.js")) as f:
        js = f.read()
    page = env.get_template("index.html").render(
        static=True,
        title="Volatility Scanner report",
        heading=f"Volatility Scanner report: simulated data, seed {seed}",
        inline_css=css,
        inline_js=js,
        payload_json=json.dumps(payload, allow_nan=False).replace("</", "<\\/"),
        before_content=before,
        after_content=after,
    )
    return page, results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    parser.add_argument("--out", default=os.path.join("docs", "index.html"))
    args = parser.parse_args(argv)
    page, results = build(args.seed, args.jitter)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(page)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "report.json"), "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"wrote {args.out} (Python {platform.python_version()})")
    for e in results["expiries"]:
        if e["svi"]:
            print(
                f"{e['days']:>3.0f}d  RMSE {e['rmse_vol_pts']:.2f} vol pts  ATM {e['atm_iv']:.2%}"
                f"  skew {e['skew_25d']:.2f}"
            )
    print(f"violations: {results['violations']} {results['violations_by_type']}")
    if not math.isfinite(results["round_trip"]["max_error"]):
        raise SystemExit("round trip produced a non-finite error")


if __name__ == "__main__":
    main()
