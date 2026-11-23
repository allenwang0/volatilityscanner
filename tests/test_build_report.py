"""The published report and README say what the committed results say.

Readers take the hero and the Highlights at face value, so the expected values are
recomputed here from results/report.json and the sources, independently of
scripts/build_report.py's helpers: a generator bug that shifts a number (wrong key,
rounding before comparing, a stale page) fails here instead of reaching Pages.
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "results" / "report.json").read_text())
BENCH = json.loads((ROOT / "results" / "bench_iv.json").read_text())


def build(tmp_path):
    out = tmp_path / "index.html"
    subprocess.run(
        [sys.executable, "scripts/build_report.py", "--out", str(out)], cwd=ROOT, check=True
    )
    return out.read_text()


def text_of(page):
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def superscript(n):
    return str(n).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def test_hero_is_the_round_trip_max_error_over_usable_contracts(tmp_path):
    # The hero is the one number a 5-second reader keeps: it must be the largest
    # error among usable-vega contracts, and it is only true if every one of them solved.
    rt = RESULTS["round_trip"]
    assert rt["unsolved_identifiable"] == 0, "hero claims every usable contract solved"
    page = build(tmp_path)
    value = re.search(r'<p class="hero-metric"><span class="value">([^<]+)</span>', page).group(1)
    exp = int(f"{rt['max_error']:e}".split("e")[1])
    mantissa = rt["max_error"] / 10**exp
    assert value == f"{mantissa:.1f} × 10{superscript(exp)}"
    text = text_of(page)
    assert f"all {rt['identifiable']:,} contracts with usable vega solved" in text
    assert "Simulated data, no market quotes" in text, "hero context must say the data is simulated"
    assert f"{rt['contracts']:,} Black–Scholes contracts" in text


def test_usable_vega_is_defined_in_plain_words_next_to_the_hero(tmp_path):
    hero = text_of(build(tmp_path).split('aria-label="Contents"')[0])
    assert "Usable vega" in hero and "price still responds to volatility" in hero


def test_page_and_readme_list_every_arbitrage_check_in_the_code(tmp_path):
    # The old README listed four of the five checks; both must now name all of them.
    source = (ROOT / "arbitrage.py").read_text()
    names = list(dict.fromkeys(re.findall(r'Violation\(\s*"(\w+)"', source)))
    assert len(names) == 5
    text = text_of(build(tmp_path))
    for name in names:
        assert name in text
    assert ", ".join(names) in (ROOT / "README.md").read_text()


def test_fit_claim_is_framed_against_the_noise_floor(tmp_path):
    # The simulator adds IV noise, so fit error only means something next to that floor.
    sim = (ROOT / "simulator.py").read_text()
    noise = float(re.search(r"^IV_NOISE_SD = ([\d.]+)", sim, re.M).group(1))
    text = text_of(build(tmp_path))
    assert "SVI recovers its simulated smile to the noise floor" in text
    assert f"noise added: {noise * 100:.2f}" in text
    monthly = min(RESULTS["slices"], key=lambda s: abs(s["days"] - 30))
    assert f"{monthly['held_out_rmse_vol_points']:.3f}" in text


def test_speedup_and_load_match_the_committed_benchmark(tmp_path):
    big = max(BENCH["timing"], key=lambda r: r["n"])
    text = text_of(build(tmp_path))
    assert f"{big['scalar_ms'] / big['vectorized_ms']:.2f}×" in text
    assert " / ".join(f"{v:.2f}" for v in BENCH["load_before"]) in text, "state the load average"


def test_committed_page_and_readme_are_current(tmp_path):
    # Pages publishes docs/ as committed; a stale page would show numbers no longer in results/.
    assert build(tmp_path) == (ROOT / "docs" / "index.html").read_text(), (
        "docs/index.html is stale: run python3 scripts/build_report.py --readme README.md"
    )
    readme = (ROOT / "README.md").read_text()
    rt = RESULTS["round_trip"]
    assert (
        f"all {rt['identifiable']:,} with usable vega solve, largest error {rt['max_error']:.1e}"
        in readme
    )
    assert "allenwang0.github.io/volatilityscanner" in readme


def test_unsupported_claims_stay_out(tmp_path):
    # "every one of 10,000" overstated the count (9,600 are usable); banned words stay out.
    pages = (("page", text_of(build(tmp_path))), ("README", (ROOT / "README.md").read_text()))
    for name, text in pages:
        for phrase in ("every one of 10,000", "real-time", "Robust", "production-ready"):
            assert phrase not in text, f"{name} contains {phrase!r}"
