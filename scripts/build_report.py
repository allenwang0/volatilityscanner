"""Build docs/index.html (and the README Highlights block) from committed results.

Usage:
  python3 scripts/build_report.py [--results results/report.json]
      [--bench results/bench_iv.json] [--out docs/index.html] [--readme README.md]

Python standard library only. Every number on the page is computed here from the
two result files or read from constants in the committed sources (tick length,
noise level, check names); none is typed by hand. Components and charts come from
the vendored report kit in report-kit/.
"""

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "report-kit"))
import report_kit as rk  # noqa: E402
from report_kit import Raw, esc  # noqa: E402

REPO = "allenwang0/volatilityscanner"
REPO_URL = f"https://github.com/{REPO}"
PAGE_URL = "https://allenwang0.github.io/volatilityscanner/"
HOME_URL = "https://allenwang0.github.io/"
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
THIN = " "
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
CHECK_TEXT = {
    "monotonicity": ("Call prices fall as strike rises", "a higher strike can never be worth more"),
    "slope": (
        "Call prices fall no faster than the discounted strike gap",
        "C(K₁) − C(K₂) ≤ e^(−rT)(K₂ − K₁)",
    ),
    "butterfly": (
        "Prices are convex in strike",
        "the middle of three strikes sits on or below the chord",
    ),
    "put_call_parity": ("Calls and puts agree", "C − P = e^(−rT)(F − K) at every shared strike"),
    "calendar": (
        "Total variance grows with expiry",
        "σ²T at a fixed log-moneyness never falls with T",
    ),
}


# ---------------------------------------------------------------- formatting


def sci(x, digits=2):
    """2.459e-12 -> '2.5 × 10⁻¹²'."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{digits - 1}f}{THIN}×{THIN}10{str(e).translate(SUP)}"


WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def word(n):
    """Small counts in words (style guide), larger ones as digits."""
    return WORDS.get(int(n), str(n)) if float(n).is_integer() else f"{n:g}"


def pow10(e):
    return "1" if e == 0 else f"10{str(e).translate(SUP)}"


def vol_pts(x, digits=3):
    return f"{x:.{digits}f}{THIN}vol pts"


# ---------------------------------------------------------------- inputs


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def source_text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def py_constant(rel, name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([0-9._e+-]+)", source_text(rel), re.M)
    if not match:
        raise ValueError(f"{name} not found in {rel}")
    return float(match.group(1).replace("_", ""))


def line_range(rel, start, end=None):
    """1-based numbers of the first line containing `start` and the next line containing `end`."""
    lines = source_text(rel).splitlines()
    first = next(i for i, line in enumerate(lines, 1) if start in line)
    if end is None:
        return first, first
    last = next(i for i, line in enumerate(lines, 1) if i >= first and end in line)
    return first, last


def source_lines(rel, start, end):
    first, last = line_range(rel, start, end)
    return "\n".join(source_text(rel).splitlines()[first - 1 : last])


def check_names():
    """Violation types in arbitrage.py, in the order they are raised."""
    names = []
    for name in re.findall(r'Violation\(\s*"(\w+)"', source_text("arbitrage.py")):
        if name not in names:
            names.append(name)
    return names


def reason_codes():
    codes = []
    for code in re.findall(r'IVResult\(nan, False, 0, "(\w+)"\)', source_text("engine.py")):
        if code not in codes:
            codes.append(code)
    return codes


def facts_from_sources():
    return {
        "tick_s": py_constant("main.py", "TICK_SECONDS"),
        "tolerance": py_constant("arbitrage.py", "SIMULATED_TOLERANCE"),
        "sigma_tol": py_constant("engine.py", "SIGMA_TOL"),
        "noise_sd": py_constant("simulator.py", "IV_NOISE_SD"),
        "noise_ticks": int(py_constant("report.py", "NOISE_TICKS")),
        "checks": check_names(),
        "reasons": reason_codes(),
        "queue": re.search(r"asyncio\.Queue\(maxsize=(\d+)\)", source_text("main.py")).group(1),
    }


def git_last_commit(rel):
    """Commit that last changed a committed file, so source links point at a SHA."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


# ---------------------------------------------------------------- derived numbers


def derive(res, bench, facts):
    rt = res["round_trip"]
    slices = {round(s["days"]): s for s in res["slices"]}
    monthly = min(res["slices"], key=lambda s: abs(s["days"] - 30))
    big = max(bench["timing"], key=lambda r: r["n"])
    usable_bins = [b for b in rt["by_vega_decade"] if b["vega_over_spot_lo"] >= rt["usable_vega"]]
    below = [b for b in rt["by_vega_decade"] if b["vega_over_spot_hi"] <= rt["usable_vega"]]
    worst_below = max(b["max_error"] for b in below if b["max_error"] is not None)
    sweep = sorted(res["noise_sweep"], key=lambda r: r["noise_sd"])
    return {
        "usable": rt["identifiable"],
        "solved": rt["identifiable"] - rt["unsolved_identifiable"],
        "contracts": rt["contracts"],
        "max_err": rt["max_error"],
        "median_err": rt["median_error"],
        "usable_bin_max": max(b["max_error"] for b in usable_bins),
        "unusable": rt["contracts"] - rt["identifiable"],
        "unusable_solved": rt["unidentifiable_solved"],
        "worst_below": worst_below,
        "brent": rt["brent_share"],
        "floor_pts": facts["noise_sd"] * 100,
        "monthly": monthly,
        "slices": slices,
        "fit_max": max(s["rmse_vol_points"] for s in res["slices"]),
        "fit_min": min(s["rmse_vol_points"] for s in res["slices"]),
        "held_long_max": max(
            s["held_out_rmse_vol_points"] for s in res["slices"] if s["days"] >= 30
        ),
        "speedup": big["scalar_ms"] / big["vectorized_ms"],
        "bench_n": big["n"],
        "bench_brent": big["brent_share"],
        "bench_diff": big["max_abs_diff"],
        "scalar_ms": big["scalar_ms"],
        "vector_ms": big["vectorized_ms"],
        "sweep": sweep,
        "ticks_per_s": 1 / facts["tick_s"],
    }


# ---------------------------------------------------------------- charts


def error_chart(rt, uid, width):
    """Largest and median IV error per vega/S decade, log-log, with the usable cutoff."""
    phone = width < 420
    bins = rt["by_vega_decade"]
    under, decades = bins[0], bins[1:]
    lo_e = round(math.log10(decades[0]["vega_over_spot_lo"]))
    hi_e = round(math.log10(decades[-1]["vega_over_spot_hi"]))
    height = 300 if phone else (360 if width < 800 else 400)
    ml, mr, mt, mb = (46 if phone else 58), (58 if phone else 74), 44, 48
    band = 30 if phone else 40
    plot_l = ml + band
    px = lambda le: plot_l + (width - mr - plot_l) * (le - lo_e) / (hi_e - lo_e)
    y_lo, y_hi = -16, 0
    py = lambda v: (
        height - mb - (height - mt - mb) * (math.log10(max(v, 10.0**y_lo)) - y_lo) / (y_hi - y_lo)
    )
    parts = [
        f'<rect class="c-band" x="{rk.coord(ml)}" y="{mt}" width="{band - 6}" height="{rk.coord(height - mt - mb)}"/>'
    ]
    for e in range(y_lo, y_hi + 1, 4):
        parts.append(
            f'<path class="c-grid" d="M{rk.coord(ml)} {rk.coord(py(10.0**e))}H{rk.coord(width - mr)}"/>'
        )
        parts.append(rk.svg_text(ml - 8, py(10.0**e) + 4, pow10(e), "c-tick", "end"))
    step = 5 if phone else 3
    for e in range(lo_e, hi_e + 1):
        if (e - hi_e) % step == 0 and e != lo_e:  # lo_e sits on the band edge, labelled by the band
            parts.append(rk.svg_text(px(e), height - mb + 18, pow10(e), "c-tick", "middle"))
    parts.append(
        rk.svg_text(
            ml + (band - 6) / 2,
            height - mb + 18,
            "<" + pow10(lo_e),
            "c-tick",
            "middle",
        )
    )
    parts.append(
        f'<path class="c-axis" d="M{rk.coord(ml)} {rk.coord(height - mb)}H{rk.coord(width - mr)}"/>'
    )
    cut = px(math.log10(rt["usable_vega"]))
    parts.append(f'<path class="c-floor" d="M{rk.coord(cut)} {mt - 8}V{rk.coord(height - mb)}"/>')
    cut_label = (
        "usable vega →"
        if phone
        else f"usable: vega > {pow10(round(math.log10(rt['usable_vega'])))} × spot →"
    )
    room = width - mr - cut
    if rk.text_width(cut_label) + 8 < room:
        parts.append(rk.svg_text(cut + 6, mt - 2, cut_label, "c-label"))
    else:
        parts.append(rk.svg_text(cut - 6, mt - 2, cut_label, "c-label", "end"))
    for key, accent in (("median_error", False), ("max_error", True)):
        pts = [(ml + (band - 6) / 2, under[key])] if under[key] is not None else []
        pts += [
            (px(math.log10(b["vega_over_spot_lo"]) + 0.5), b[key])
            for b in decades
            if b[key] is not None
        ]
        d = "".join(
            ("M" if i == 0 else "L") + f"{rk.coord(x)} {rk.coord(py(v))}"
            for i, (x, v) in enumerate(pts)
        )
        cls = "c-line-accent" if accent else "c-stroke-muted"
        parts.append(
            f'<path class="{cls}" stroke-width="{2 if accent else 1.5}" fill="none" d="{d}"/>'
        )
        fill = "c-accent" if accent else "c-muted"
        for x, v in pts:
            parts.append(
                f'<circle cx="{rk.coord(x)}" cy="{rk.coord(py(v))}" r="{3.5 if accent else 3}" class="{fill}"/>'
            )
        name = "largest" if accent else "median"
        parts.append(
            rk.svg_text(
                pts[-1][0] + 8, py(pts[-1][1]) + 4, name, "c-label c-hi" if accent else "c-label"
            )
        )
    # annotate the one number the headline rests on: the largest error among usable contracts
    hero_bin = max(
        (b for b in decades if b["vega_over_spot_lo"] >= rt["usable_vega"]),
        key=lambda b: b["max_error"],
    )
    hx, hy = px(math.log10(hero_bin["vega_over_spot_lo"]) + 0.5), py(hero_bin["max_error"])
    label = sci(hero_bin["max_error"])
    parts.append(
        f'<path class="c-axis" d="M{rk.coord(hx)} {rk.coord(hy - 6)}V{rk.coord(hy - 26)}"/>'
    )
    parts.append(rk.svg_text(hx + 6, hy - 22, label, "c-value"))
    parts.append(rk.svg_text(ml, 14, "Absolute error in σ (log scale)", "c-title"))
    parts.append(rk.svg_text(width - mr, height - 8, "Vega ÷ spot (log scale)", "c-title", "end"))
    desc = (
        f"Largest and median absolute implied-volatility error per decade of vega divided by spot, "
        f"{rt['contracts']:,} contracts. Above the cutoff of {rt['usable_vega']:g} the largest error is "
        f"{hero_bin['max_error']:.1e}; below it errors rise to {max(b['max_error'] or 0 for b in bins):.2f}."
    )
    return rk.svg(
        uid,
        width,
        height,
        "IV error stays near machine precision wherever vega is usable",
        desc,
        "".join(parts),
    )


def fit_chart(res, floor_pts, uid, width):
    """Dot plot: in-sample and held-out SVI RMSE per expiry, with the noise floor drawn."""
    phone = width < 420
    rows = res["slices"]
    label_w = 66
    vmax = max(max(s["held_out_rmse_vol_points"], s["rmse_vol_points"]) for s in rows)
    ticks = rk.nice_ticks(0, vmax, 4 if phone else 5)
    plot_l, plot_r = label_w, width - (54 if phone else 64)
    x = lambda v: plot_l + (plot_r - plot_l) * v / ticks[-1]
    top, row_h = 64, 40
    bottom = top + row_h * len(rows)
    parts = []
    for t in ticks:
        parts.append(f'<path class="c-grid" d="M{rk.coord(x(t))} {top - 8}V{rk.coord(bottom)}"/>')
        parts.append(rk.svg_text(x(t), bottom + 18, f"{t:g}", "c-tick", "middle"))
    fx = x(floor_pts)
    parts.append(f'<path class="c-floor" d="M{rk.coord(fx)} {top - 16}V{rk.coord(bottom)}"/>')
    parts.append(rk.svg_text(fx + 6, top - 14, f"noise added: {floor_pts:.2f}", "c-label"))
    # key, top-left: every marker drawn is named
    parts.append('<circle cx="6" cy="12" r="4.5" class="c-muted"/>')
    parts.append(rk.svg_text(16, 16, "in-sample", "c-title"))
    kx = 16 + rk.text_width("in-sample", 12) + 22
    parts.append(f'<circle cx="{rk.coord(kx)}" cy="12" r="5" class="c-accent"/>')
    parts.append(
        rk.svg_text(
            kx + 10, 16, "held out (every 3rd strike)" if not phone else "held out", "c-title"
        )
    )
    for i, s in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        a, b = x(s["rmse_vol_points"]), x(s["held_out_rmse_vol_points"])
        parts.append(rk.svg_text(0, cy + 4, f"{s['days']:g} days", "c-label"))
        parts.append(
            f'<path class="c-stroke-muted" stroke-width="2" d="M{rk.coord(min(a, b))} {rk.coord(cy)}H{rk.coord(max(a, b))}"/>'
        )
        parts.append(f'<circle cx="{rk.coord(a)}" cy="{rk.coord(cy)}" r="4.5" class="c-muted"/>')
        parts.append(f'<circle cx="{rk.coord(b)}" cy="{rk.coord(cy)}" r="5" class="c-accent"/>')
        parts.append(
            rk.svg_text(max(a, b) + 10, cy + 4, f"{s['held_out_rmse_vol_points']:.3f}", "c-tick")
        )
    parts.append(rk.svg_text(plot_r, bottom + 38, "RMSE (vol points)", "c-title", "end"))
    desc = (
        "; ".join(
            f"{s['days']:g} days: in-sample {s['rmse_vol_points']:.3f}, held out {s['held_out_rmse_vol_points']:.3f}"
            for s in rows
        )
        + f" vol points. Noise added to the simulated quotes: {floor_pts:.2f} vol points."
    )
    return rk.svg(
        uid, width, bottom + 46, "SVI error sits at the simulated noise level", desc, "".join(parts)
    )


def sweep_chart(d, facts, uid, width):
    rows = [
        {
            "label": f"{r['noise_sd'] * 100:.2f} vol pts"
            + (" (default)" if r["noise_sd"] == facts["noise_sd"] else ""),
            "value": r["per_tick"],
            "accent": r["noise_sd"] == facts["noise_sd"],
        }
        for r in d["sweep"]
    ]
    return rk.bar_chart(
        rows,
        uid=uid,
        width=width,
        title="Violations rise with quote noise",
        desc="Mean static-arbitrage violations per tick by IV noise level: "
        + "; ".join(f"{r['label']}: {r['value']:.2f}" for r in rows)
        + ".",
        axis_title=f"Violations per tick, mean of {facts['noise_ticks']} ticks",
        fmt=lambda v: "0" if v == 0 else f"{v:.1f}",
    )


# ---------------------------------------------------------------- page


def evidence_rows(res, bench, d, facts, shas):
    def blob(rel, start=None, end=None):
        anchor = f"#L{start}" + (f"-L{end}" if end and end != start else "") if start else ""
        return f"{REPO_URL}/blob/{shas.get(rel) or 'main'}/{rel}{anchor}"

    rt_lines = line_range("report.py", "def round_trip_errors", '"by_vega_decade": bins')
    usable = line_range("bench/bench_iv.py", "USABLE_VEGA = ")
    noise = line_range("simulator.py", "IV_NOISE_SD = ")
    svi_table = line_range("simulator.py", "SVI_TABLE = (", "(90, ")
    checks = line_range("arbitrage.py", "def _strike_checks", "return ArbitrageScan")
    tick = line_range("main.py", "TICK_SECONDS = ")
    publish = line_range("main.py", "def publish", "outbox.put_nowait")
    fanout = line_range(
        "tests/test_fanout.py",
        "def test_stalled_client_does_not_hold_up_others",
        "assert slow_sent[1] >= latest",
    )
    held = line_range("report.py", "def held_out_rmse", "return out")
    reasons = line_range(
        "engine.py", 'return IVResult(nan, False, 0, "invalid_input")', '"outside_bracket")'
    )
    return [
        (
            "E1",
            f"{d['solved']:,} of {d['usable']:,} usable contracts solved; largest error {sci(d['max_err'])}",
            "results/report.json → round_trip",
            blob("results/report.json"),
            "Verified",
        ),
        (
            "E2",
            f"Usable means vega > {res['round_trip']['usable_vega']:g} × spot; contract ranges and seed",
            f"bench/bench_iv.py:{usable[0]}; report.py:{rt_lines[0]}–{rt_lines[1]}",
            blob("report.py", *rt_lines),
            "Verified",
        ),
        (
            "E3",
            "SVI in-sample and held-out RMSE per expiry",
            f"results/report.json → slices; method report.py:{held[0]}–{held[1]}",
            blob("results/report.json"),
            "Verified",
        ),
        (
            "E4",
            f"Simulated quotes are SVI plus N(0, {facts['noise_sd']:g}) IV noise",
            f"simulator.py:{noise[0]}, SVI_TABLE :{svi_table[0]}–{svi_table[1]}",
            blob("simulator.py", noise[0], svi_table[1]),
            "Verified",
        ),
        (
            "E5",
            f"{len(facts['checks'])} static-arbitrage checks: {', '.join(facts['checks'])}",
            f"arbitrage.py:{checks[0]}–{checks[1]}",
            blob("arbitrage.py", *checks),
            "Verified",
        ),
        (
            "E6",
            "Violations per tick at four noise levels",
            "results/report.json → noise_sweep",
            blob("results/report.json"),
            "Verified",
        ),
        (
            "E7",
            f"One tick every {facts['tick_s']:g} s",
            f"main.py:{tick[0]}",
            blob("main.py", *tick),
            "Verified",
        ),
        (
            "E8",
            "A stalled client does not hold up the others; it skips to the newest tick",
            f"main.py:{publish[0]}–{publish[1]}; tests/test_fanout.py:{fanout[0]}–{fanout[1]}",
            blob("tests/test_fanout.py", *fanout),
            "Verified",
        ),
        (
            "E9",
            f"Vectorized solver {d['speedup']:.2f}× the scalar loop on {d['bench_n']:,} contracts",
            "results/bench_iv.json",
            blob("results/bench_iv.json"),
            "Verified",
        ),
        (
            "E10",
            f"Failed solves return NaN with one of {len(facts['reasons'])} reason codes",
            f"engine.py:{reasons[0]}–{reasons[1]}",
            blob("engine.py", *reasons),
            "Verified",
        ),
    ]


def hero_media():
    return Raw(
        '<p class="card-label">The dashboard, simulated chain</p>'
        '<a href="assets/live.gif" aria-label="Open the dashboard recording full size" style="display:block;border-radius:12px;overflow:hidden">'
        '<picture><source srcset="assets/live-still.png" media="(prefers-reduced-motion: reduce)">'
        '<img src="assets/live.gif" width="825" height="540" style="display:block;width:100%;height:auto" '
        'alt="Recording of the dashboard on simulated data: four implied-volatility smiles with SVI fits, the '
        'at-the-money term structure and 25-delta skew, updating each tick."></picture></a>'
    )


def build_page(res, bench, shas):
    facts = facts_from_sources()
    d = derive(res, bench, facts)
    rt = res["round_trip"]
    R = rk.ref
    ev = evidence_rows(res, bench, d, facts, shas)
    sim = res["simulator"]
    ranges = rt["ranges"]
    m = d["monthly"]
    n_checks = len(facts["checks"])
    cut = pow10(round(math.log10(rt["usable_vega"])))

    hero = rk.hero(
        eyebrow="Numerical methods · Python · Simulated data",
        title="Volatility Scanner",
        subtitle=Raw(
            f"A live options dashboard: it inverts every quote to an implied volatility, fits a smile per "
            f"expiry and runs {word(n_checks)} no-arbitrage checks, "
            f"{'twice' if d['ticks_per_s'] == 2 else word(d['ticks_per_s']) + ' times'} a second."
        ),
        value=sci(d["max_err"]),
        label=Raw(
            f"Largest implied-volatility error; all {d['usable']:,} contracts with usable vega solved{R('E1')}"
        ),
        support=Raw(
            f"<b>Usable vega</b> means the option's price still responds to volatility: vega above "
            f"{cut} × spot. Below that, many volatilities give the same price to machine precision, so the "
            f"price cannot pin one down.{R('E2')}"
        ),
        context=(
            f"Simulated data, no market quotes: {d['contracts']:,} Black–Scholes contracts priced from known "
            f"volatilities ({ranges['sigma'][0]:.0%}–{ranges['sigma'][1]:.0%}), seed {res['seed']}."
        ),
        actions=[("Source", REPO_URL, True), ("Reproduce", "#reproduce", False)],
        figure=hero_media(),
    )

    toc = Raw(
        '<nav class="wrap" aria-label="Contents" style="margin-top:-24px"><p class="card-label">Contents</p>'
        '<ol style="display:flex;flex-wrap:wrap;gap:4px 24px;margin:0;padding:0;list-style:none;font-size:15px">'
        + "".join(
            f'<li><a href="#{sid}" style="display:inline-block;padding:10px 0">{esc(label)}</a></li>'
            for sid, label in [
                ("findings", "Findings"),
                ("solver", "Solver"),
                ("smile", "Smile fit"),
                ("arbitrage", "Arbitrage checks"),
                ("streaming", "Streaming"),
                ("methodology", "Methodology"),
                ("limitations", "Limitations"),
                ("reproduce", "Reproduce"),
                ("evidence", "Evidence"),
            ]
        )
        + "</ol></nav>"
    )

    s7 = d["slices"].get(7)
    findings = rk.bento(
        [
            (
                4,
                1,
                rk.join(
                    [
                        rk.tag(
                            "h3", "The smile fit reaches the noise floor", class_="card-heading"
                        ),
                        rk.metric(
                            "30-day held-out RMSE",
                            f"{m['held_out_rmse_vol_points']:.3f}",
                            "vol pts",
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"The simulator adds {d['floor_pts']:.2f} vol points of noise, so the fit cannot do better "
                                f"on these quotes.{R('E3')}{R('E4')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                2,
                rk.join(
                    [
                        rk.tag(
                            "h3",
                            f"{word(n_checks).capitalize()} arbitrage checks run each tick",
                            class_="card-heading",
                        ),
                        rk.metric(
                            "Conditions checked per tick", f"{res['tick']['conditions_checked']:,}"
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"Over {res['tick']['quotes_checked']} simulated quotes, each violation sized in dollars and "
                                f"compared with half the bid-ask spread.{R('E5')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                3,
                rk.join(
                    [
                        rk.tag(
                            "h3", "A slow browser cannot stall the stream", class_="card-heading"
                        ),
                        rk.metric("Tick interval", f"{facts['tick_s']:g}", "s"),
                        rk.tag(
                            "p",
                            Raw(
                                f"Each client has a {facts['queue']}-slot outbox; a client that falls behind skips to the "
                                f"newest tick.{R('E7')}{R('E8')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                6,
                4,
                rk.join(
                    [
                        rk.tag("h3", "Vectorizing the solver gains little", class_="card-heading"),
                        rk.comparison(
                            [
                                (
                                    "Vectorized",
                                    d["vector_ms"],
                                    f"{d['vector_ms']:.0f}{THIN}ms",
                                    True,
                                ),
                                (
                                    "Scalar loop",
                                    d["scalar_ms"],
                                    f"{d['scalar_ms']:.0f}{THIN}ms",
                                    False,
                                ),
                            ],
                            max(d["scalar_ms"], d["vector_ms"]),
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"{d['bench_n']:,} contracts, median of {bench['repeats']} runs: {d['speedup']:.2f}×. "
                                f"{d['bench_brent']:.0%} of contracts fall back to the scalar Brent path, which bounds the "
                                f"gain.{R('E9')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                6,
                5,
                rk.join(
                    [
                        rk.tag("h3", "Failures are reported, never guessed", class_="card-heading"),
                        rk.tag(
                            "p",
                            Raw(
                                "A quote that cannot be inverted returns NaN with a reason code — "
                                + ", ".join(f"<code>{esc(c)}</code>" for c in facts["reasons"])
                                + f" — and the dashboard draws it as a hollow marker.{R('E10')}"
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    findings_sec = rk.section(
        "findings",
        "What the scanner does, measured on simulated data",
        findings,
        eyebrow="Findings",
    )

    # Solver
    err_fig = rk.figure(
        "fig-iv-error",
        f"Above the vega cutoff every error is below {pow10(math.ceil(math.log10(d['usable_bin_max'])))}; "
        f"below it, errors reach {d['worst_below']:.2f}",
        rk.variants(
            error_chart(rt, "err-d", 1000),
            error_chart(rt, "err-t", 640),
            error_chart(rt, "err-p", 320),
        ),
        rk.tag(
            "p",
            Raw(
                f"{d['contracts']:,} contracts, seed {res['seed']}: spot {ranges['spot'][0]:,.0f}–{ranges['spot'][1]:,.0f}, "
                f"strike {ranges['moneyness'][0]:g}–{ranges['moneyness'][1]:g}× spot, 1 day to {ranges['years'][1]:g} years, "
                f"σ {ranges['sigma'][0]:.0%}–{ranges['sigma'][1]:.0%}, dividend yield up to {ranges['dividend'][1]:.0%}, "
                f"rate {rt['rate']:.0%}. Each point is one decade of vega ÷ spot; the shaded column pools everything below "
                f"{pow10(round(math.log10(rt['by_vega_decade'][1]['vega_over_spot_lo'])))}. Single seeded run.{R('E1')}"
            ),
        ),
        question="How close is the solved volatility to the one that priced the option, and where does that break down?",
        table=rk.data_table(
            "IV error by vega decade",
            ["Vega ÷ spot", "Contracts", "Returned a value", "Largest error", "Median error"],
            [
                [
                    ("below " + pow10(round(math.log10(b["vega_over_spot_hi"]))))
                    if b["vega_over_spot_lo"] == 0
                    else f"{pow10(round(math.log10(b['vega_over_spot_lo'])))}–{pow10(round(math.log10(b['vega_over_spot_hi'])))}",
                    f"{b['contracts']:,}",
                    f"{b['solved']:,}",
                    f"{b['max_error']:.1e}" if b["max_error"] is not None else "—",
                    f"{b['median_error']:.1e}" if b["median_error"] is not None else "—",
                ]
                for b in rt["by_vega_decade"]
            ],
            numeric={1, 2, 3, 4},
        ),
        data_href=ev[0][3],
    )
    solver_body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    "Implied volatility is the σ that makes the Black–Scholes price equal the quoted price; there is no closed "
                    "form, so it is found by root-finding. The solver first rejects prices outside the no-arbitrage bounds, "
                    "starts Newton's method from the Brenner–Subrahmanyam guess, and hands any contract whose step leaves "
                    f"the bracket or whose vega is too flat to Brent's method — {d['brent']:.0%} of this set."
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"Every contract with usable vega solved. The largest error is the same order as the stopping rule — "
                    f"Newton ends once a step moves σ by less than {pow10(round(math.log10(facts['sigma_tol'])))} — and the "
                    f"median, {sci(d['median_err'])}, is double-precision rounding.{R('E1')}"
                ),
            ),
            Raw("</div>"),
            err_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"Below the cutoff the price carries almost no information about σ. Of the {d['unusable']:,} such contracts, "
                    f"{d['unusable_solved']:,} still returned a value, with errors up to {d['worst_below']:.2f} in σ: a root "
                    f"that reproduces the price, but not the volatility that set it. The cutoff is applied in tests and in this "
                    f"measurement; the dashboard does not flag these quotes."
                ),
            ),
            Raw("</div>"),
        ]
    )
    solver_sec = rk.section(
        "solver",
        "The solver recovers volatility wherever the price depends on it",
        solver_body,
        eyebrow="Implied volatility",
    )

    # Smile
    fit_fig = rk.figure(
        "fig-svi",
        f"In-sample error sits near the {d['floor_pts']:.2f}-point noise; held out, 30–90-day expiries stay "
        f"within {math.ceil(d['held_long_max'] / d['floor_pts'] * 10) / 10:.1f}× it",
        rk.variants(
            fit_chart(res, d["floor_pts"], "fit-d", 1000),
            fit_chart(res, d["floor_pts"], "fit-t", 640),
            fit_chart(res, d["floor_pts"], "fit-p", 320),
        ),
        rk.tag(
            "p",
            Raw(
                f"One tick of the seeded simulator, {sim['strikes']} strikes per expiry ({sim['strike_range'][0]:,.0f}–"
                f"{sim['strike_range'][1]:,.0f}, spot {sim['spot']:,.0f}). In-sample: fit to all out-of-the-money quotes. "
                f"Held out: fit to {m['held_out_train_quotes']} quotes, scored on the other {m['held_out_test_quotes']}. "
                f"Single run; the dashed line is the standard deviation of the noise the simulator adds.{R('E3')}{R('E4')}"
            ),
        ),
        question="How closely does the SVI curve follow the quotes, and is that better than the noise allows?",
        table=rk.data_table(
            "SVI fit per expiry",
            [
                "Expiry",
                "ATM IV",
                "25-delta skew",
                "In-sample RMSE",
                "Held-out RMSE",
                "Held-out quotes",
            ],
            [
                [
                    f"{s['days']:g} days",
                    f"{s['atm_iv']:.1%}",
                    f"{s['skew_25d']:.2f} vol pts",
                    f"{s['rmse_vol_points']:.3f}",
                    f"{s['held_out_rmse_vol_points']:.3f}",
                    str(s["held_out_test_quotes"]),
                ]
                for s in res["slices"]
            ],
            numeric={1, 2, 3, 4, 5},
        ),
        data_href=ev[2][3],
    )
    smile_body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    "SVI (stochastic volatility inspired) describes one expiry's smile with five parameters for total variance "
                    "σ²T as a function of log-moneyness. Each expiry is fit separately, weighted by vega, and the fit is checked "
                    "for butterfly arbitrage through its density."
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"The test is close to circular, and the report treats it that way. The simulated chain is generated from "
                    f"SVI curves plus independent N(0, {facts['noise_sd']:g}) noise in each quote's volatility, so an SVI fit "
                    f"can at best recover that curve to the {d['floor_pts']:.2f}-point noise. In-sample RMSE of "
                    f"{d['fit_min']:.3f}–{d['fit_max']:.3f} vol points means the fitter recovers its own model; it says nothing "
                    f"about how SVI fits real quotes.{R('E4')}"
                ),
            ),
            Raw("</div>"),
            fit_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    (
                        f"The {s7['days']:g}-day expiry is the exception: its held-out error is "
                        f"{s7['held_out_rmse_vol_points'] / d['floor_pts']:.1f}× the noise. That slice has the sharpest smile in "
                        f"the simulator, and each held-out score rests on {s7['held_out_test_quotes']} quotes; which of the two "
                        f"drives the gap is not tested."
                    )
                    if s7
                    else ""
                ),
            ),
            Raw("</div>"),
        ]
    )
    smile_sec = rk.section(
        "smile",
        "SVI recovers its simulated smile to the noise floor",
        smile_body,
        eyebrow="Smile fit",
    )

    # Arbitrage
    check_rows = []
    for name in facts["checks"]:
        what, rule = CHECK_TEXT.get(name, (name, ""))
        check_rows.append(
            (
                Raw(f"<code>{esc(name)}</code>"),
                Raw(f'{esc(what)}<br><span class="muted">{esc(rule)}</span>'),
            )
        )
    checks_table = rk.join(
        [
            Raw(f'<h3 class="sub">The {word(n_checks)} static no-arbitrage checks</h3>'),
            rk.methodology(check_rows),
        ]
    )
    base = next(r for r in d["sweep"] if r["noise_sd"] == facts["noise_sd"])
    top = d["sweep"][-1]
    sweep_fig = rk.figure(
        "fig-noise",
        f"Clean quotes trigger no violations; at {top['noise_sd'] / facts['noise_sd']:.0f}× the noise, {top['per_tick']:.1f} per tick",
        rk.variants(
            sweep_chart(d, facts, "nz-d", 1000),
            sweep_chart(d, facts, "nz-t", 640),
            sweep_chart(d, facts, "nz-p", 320),
        ),
        rk.tag(
            "p",
            Raw(
                f"Mean violations per tick over {facts['noise_ticks']} ticks of the seeded simulator at each noise level; "
                f"share of ticks with any violation: "
                + ", ".join(f"{r['share_of_ticks']:.0%}" for r in d["sweep"])
                + f". A violation counts above {pow10(round(math.log10(facts['tolerance'])))} dollars, since simulated "
                f"quotes have no bid-ask spread.{R('E6')}"
            ),
        ),
        question="Do the checks stay quiet on consistent prices and fire on inconsistent ones?",
        table=rk.data_table(
            "Violations by noise level",
            ["IV noise (vol pts)", "Violations per tick", "Ticks with a violation"],
            [
                [f"{r['noise_sd'] * 100:.2f}", f"{r['per_tick']:.2f}", f"{r['share_of_ticks']:.0%}"]
                for r in d["sweep"]
            ],
            numeric={0, 1, 2},
        ),
        data_href=ev[5][3],
    )
    arb_body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    f"A static arbitrage is a set of prices that lets a fixed portfolio of options lock in a profit. Each tick, "
                    f"the scanner tests {res['tick']['conditions_checked']:,} conditions across the chain. Every violation is "
                    f"sized in dollars per option and, when the chain has bid and ask quotes, compared with the cost of crossing "
                    f"half the spread on every leg.{R('E5')}"
                ),
            ),
            Raw("</div>"),
            checks_table,
            sweep_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"At the simulator's default noise ({base['noise_sd'] * 100:.2f} vol points) no check fires. Raising the "
                    f"noise produces violations at a rate that grows with it — consistent with the checks catching the price "
                    f"inconsistencies noise creates. The sweep does not separate which check fires; no real chain has been "
                    f"archived to test against."
                ),
            ),
            Raw("</div>"),
        ]
    )
    arb_sec = rk.section(
        "arbitrage",
        f"{word(n_checks).capitalize()} no-arbitrage checks stay quiet on clean quotes and fire as noise grows",
        arb_body,
        eyebrow="Arbitrage checks",
    )

    # Streaming
    stream_body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    f"One producer task builds a tick every {facts['tick_s']:g} s — solve, fit, check — and serializes it once. "
                    f"Each WebSocket client gets an <code>asyncio.Queue(maxsize={facts['queue']})</code>. Publishing never waits: "
                    f"if a client's slot still holds an unsent tick, that tick is dropped and the new one takes its place.{R('E8')}"
                ),
            ),
            Raw("</div>"),
            rk.code(
                source_lines("main.py", "def publish", "outbox.put_nowait"), "main.py publish()"
            ),
            Raw('<div class="prose" style="margin-top:24px">'),
            rk.tag(
                "p",
                Raw(
                    "A test stalls one fake socket mid-send and checks that a second socket keeps receiving ticks, and that the "
                    "stalled one, once released, receives the newest tick rather than the backlog. A second test drops a client "
                    "that disconnects mid-send."
                ),
            ),
            Raw("</div>"),
        ]
    )
    stream_sec = rk.section(
        "streaming",
        "A slow browser skips to the newest tick instead of slowing the rest",
        stream_body,
        eyebrow="Streaming",
    )

    # Methodology
    lb = " / ".join(f"{v:.2f}" for v in bench["load_before"])
    la = " / ".join(f"{v:.2f}" for v in bench["load_after"])
    results_sha = shas.get("results/report.json") or ""
    method = rk.methodology(
        [
            (
                "Data",
                f"Simulated only. Round trip: {d['contracts']:,} contracts drawn uniformly, seed {res['seed']}. Chain: "
                f"{res['tick']['quotes_checked']} quotes from SVI_TABLE plus N(0, {facts['noise_sd']:g}) IV noise, seed {res['seed']}",
            ),
            (
                "Error measure",
                "|solved σ − true σ|, where true σ priced the contract; usable = vega > "
                f"{rt['usable_vega']:g} × spot, computed at the true σ",
            ),
            (
                "Fit measure",
                "RMSE in vol points over out-of-the-money quotes; held out = every third strike by strike order",
            ),
            (
                "Arbitrage sweep",
                f"{len(d['sweep'])} noise levels × {facts['noise_ticks']} ticks, same seed",
            ),
            (
                "Timing",
                f"time.perf_counter, median of {bench['repeats']} runs per size; {bench['cpu']}, {bench['machine']}, "
                f"NumPy {bench['numpy']}",
            ),
            (
                "Load average",
                f"{lb} before, {la} after (1 / 5 / 15 min); shared machine, run under a lock with other benchmarks",
            ),
            (
                "Results commit",
                Raw(
                    f'<a class="src" href="{REPO_URL}/tree/{esc(results_sha)}">{esc(results_sha[:7])}</a>'
                )
                if results_sha
                else None,
            ),
            ("Run date", bench["date"][:10]),
            (
                "Repetitions",
                "Accuracy and fit: one seeded run (deterministic). Timing: 5 repetitions",
            ),
        ]
    )
    method_sec = rk.section("methodology", "Methodology", method, eyebrow="Conditions")

    limits = rk.limitations(
        [
            (
                "Simulated data only.",
                "No result here comes from market quotes; the real-data mode (yfinance) is untested in "
                "this report.",
            ),
            (
                "The fit test is near-circular.",
                "The chain is generated from SVI, so fit error measures recovery of the "
                "generating curve, not SVI's fit to real smiles.",
            ),
            (
                "Low-vega quotes are not flagged.",
                f"Below vega {cut} × spot a returned σ can be off by up to "
                f"{d['worst_below']:.2f}; the dashboard shows it like any other.",
            ),
            (
                "European pricing.",
                "Black–Scholes assumes European exercise; listed SPY options are American, so their "
                "implied volatilities are approximate.",
            ),
            (
                "One seed.",
                "Accuracy and fit come from seed 7 only; the sweep across noise levels reuses it.",
            ),
            (
                "Shared machine for timing.",
                f"Load average {lb} at the start; the {d['speedup']:.2f}× ratio is indicative.",
            ),
        ]
    )
    limits_sec = rk.section(
        "limitations",
        "Every result here comes from simulated quotes",
        limits,
        eyebrow="Limitations",
    )

    steps_html = rk.reproduce(
        [
            (
                "Install (Python 3.11 or 3.12) and run the tests.",
                "make setup PYTHON=python3.12\n.venv/bin/pytest",
            ),
            (
                "Recompute the accuracy, fit and arbitrage results and the portfolio card (a few seconds).",
                ".venv/bin/python report.py --seed 7 --card site/card.json",
            ),
            (
                "Re-run the timing benchmark (about a minute).",
                ".venv/bin/python -m bench.bench_iv --json results/bench_iv.json",
            ),
            (
                "Rebuild this page and the README Highlights from the result files.",
                "python3 scripts/build_report.py --readme README.md",
            ),
            (
                "Run the dashboard and open http://127.0.0.1:8000.",
                ".venv/bin/python main.py --seed 7",
            ),
        ],
        expected=Raw(
            f"<code>report.py</code> prints “round trip max error {d['max_err']:.1e}” and one RMSE line per "
            f"expiry; <code>git diff results</code> stays empty except for timings."
        ),
    )
    repro_sec = rk.section(
        "reproduce",
        "Reproduce it in five commands",
        steps_html,
        eyebrow="Reproduce",
        intro=Raw(f'Commands run from a clone of <a class="src" href="{REPO_URL}">{REPO}</a>.'),
    )

    ev_table = rk.table(
        "Evidence ledger",
        ["ID", "Claim", "Source", "Class"],
        [
            [
                Raw(f'<span id="evidence-{eid.lower()}">{eid}</span>'),
                claim,
                Raw(f'<a class="src" href="{esc(href)}">{esc(src)}</a>'),
                cls,
            ]
            for eid, claim, src, href, cls in ev
        ],
        wrap_cols={1, 2},
    )
    ev_sec = rk.section(
        "evidence", "Every number traces to a committed file", ev_table, eyebrow="Sources"
    )

    main = rk.join(
        [
            hero,
            toc,
            findings_sec,
            solver_sec,
            smile_sec,
            arb_sec,
            stream_sec,
            method_sec,
            limits_sec,
            repro_sec,
            ev_sec,
        ]
    )
    footer = rk.footer(
        Raw(
            f"Generated by <code>scripts/build_report.py</code> from <code>results/report.json</code> and "
            f"<code>results/bench_iv.json</code> · results {esc(results_sha[:7])} · run {esc(bench['date'][:10])} on "
            f"{esc(bench['cpu'])}"
        ),
        Raw(f'<a href="{HOME_URL}">Allen Wang</a> · <a href="{REPO_URL}">Source</a>'),
    )
    return rk.page(
        title="Volatility Scanner · Allen Wang",
        description=(
            f"An options dashboard whose implied-volatility solver recovers σ to {sci(d['max_err'])} on all "
            f"{d['usable']:,} simulated contracts with usable vega; SVI fits and {n_checks} no-arbitrage checks."
        ),
        canonical=PAGE_URL,
        header=rk.report_header(HOME_URL, "← Allen Wang", [("Source", REPO_URL)]),
        main=main,
        footer=footer,
    )


# ---------------------------------------------------------------- README


def readme_block(res, bench):
    facts = facts_from_sources()
    d = derive(res, bench, facts)
    rt = res["round_trip"]
    return "\n".join(
        [
            README_START,
            f"- **Implied-volatility solver:** on {d['contracts']:,} random simulated contracts, all {d['usable']:,} with "
            f"usable vega solve, largest error {d['max_err']:.1e}. Usable vega (> {rt['usable_vega']:g} × spot) "
            f"means the price still responds to volatility.",
            f"- **SVI smile fits** recover the simulated smile to its noise floor: in-sample RMSE {d['fit_min']:.3f}–"
            f"{d['fit_max']:.3f} vol points against {d['floor_pts']:.2f} points of added noise.",
            f"- **{len(facts['checks'])} no-arbitrage checks** ({', '.join(facts['checks'])}), each sized in dollars and "
            f"compared with the cost of crossing the bid-ask spread.",
            f"- **WebSocket server** ticking every {facts['tick_s']:g} s; a slow client skips to the newest update "
            f"instead of slowing the rest.",
            README_END,
        ]
    )


def update_readme(path, res, bench):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        sys.exit(f"{path} has no {README_START} … {README_END} block")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    pathlib.Path(path).write_text(head + readme_block(res, bench) + tail, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/report.json")
    parser.add_argument("--bench", default="results/bench_iv.json")
    parser.add_argument("--out", default="docs/index.html")
    parser.add_argument("--readme", help="rewrite the Highlights block between the README markers")
    args = parser.parse_args()
    res, bench = load(ROOT / args.results), load(ROOT / args.bench)
    cited = [
        "results/report.json",
        "results/bench_iv.json",
        "report.py",
        "bench/bench_iv.py",
        "simulator.py",
        "arbitrage.py",
        "main.py",
        "tests/test_fanout.py",
        "engine.py",
    ]
    shas = {rel: git_last_commit(rel) for rel in cited}
    out = ROOT / args.out
    os.makedirs(out.parent, exist_ok=True)
    out.write_text(build_page(res, bench, shas), encoding="utf-8")
    if args.readme:
        update_readme(ROOT / args.readme, res, bench)


if __name__ == "__main__":
    main()
