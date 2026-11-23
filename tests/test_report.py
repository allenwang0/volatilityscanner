import json
import math

import report

RESULTS = report.HERE / "results"
PLATFORM_FILES = ("report.json", "report.linux-x86_64.json")


def test_results_carry_the_seed_label_and_no_nan(tmp_path):
    # The page is rendered from this file; it must say the data is simulated, and every
    # number in it must be finite (json.dumps(allow_nan=False) raises otherwise).
    out = tmp_path / "report.json"
    report.main(["--seed", "7", "--results", str(out)])
    data = json.loads(out.read_text())
    assert data["label"] == "Simulated data, seed 7"
    assert data["seed"] == 7 and len(data["slices"]) == 4
    assert data["round_trip"]["unsolved_identifiable"] == 0


def _sig1(x):
    return None if x is None else f"{x:.0e}"


def _assert_same_as_published(fresh, committed, alt):
    """Fresh results must match what the page shows, at the precision it shows them.

    Counts, the hero error and the arbitrage sweep are deterministic across platforms
    and compared exactly. SVI values are compared at their displayed precision. Errors
    below the usable-vega cutoff depend on the platform's floating point (sigma is not
    identifiable there), so a fresh value only has to land within one decade of the
    committed platform range; usable-vega bins must agree to one significant figure.
    """
    a, c = fresh["round_trip"], committed["round_trip"]
    for key in (
        "contracts",
        "identifiable",
        "unsolved_identifiable",
        "unidentifiable_solved",
        "zero_vega",
        "max_error",
        "median_error",
    ):
        assert a[key] == c[key], f"round_trip.{key}: {a[key]} vs committed {c[key]}"
    assert f"{a['brent_share']:.0%}" == f"{c['brent_share']:.0%}"
    assert fresh["tick"] == committed["tick"] and fresh["noise_sweep"] == committed["noise_sweep"]
    shown = (
        ("rmse_vol_points", ".3f"),
        ("held_out_rmse_vol_points", ".3f"),
        ("atm_iv", ".1%"),
        ("skew_25d", ".2f"),
    )
    for x, y in zip(fresh["slices"], committed["slices"], strict=True):
        for key, f in shown:
            assert format(x[key], f) == format(y[key], f), f"{x['days']}-day {key}"
    bins = zip(
        a["by_vega_decade"], c["by_vega_decade"], alt["round_trip"]["by_vega_decade"], strict=True
    )
    for x, y, z in bins:
        assert (x["contracts"], x["solved"]) == (y["contracts"], y["solved"])
        for key in ("max_error", "median_error"):
            if x[key] is None:
                assert y[key] is None
                continue
            if x["vega_over_spot_lo"] >= a["usable_vega"]:
                assert _sig1(x[key]) in (_sig1(y[key]), _sig1(z[key])), (
                    x["vega_over_spot_lo"],
                    key,
                )
            else:
                lo, hi = min(y[key], z[key]), max(y[key], z[key])
                assert lo / 10 <= x[key] <= hi * 10, (x["vega_over_spot_lo"], key, x[key], lo, hi)


def test_committed_results_match_a_fresh_run(tmp_path):
    # results/ is what Pages publishes; a fresh run on any platform must agree with it
    # wherever the page states a number, or the page is stale.
    out = tmp_path / "report.json"
    report.main(["--seed", "7", "--results", str(out)])
    committed, alt = (json.loads((RESULTS / f).read_text()) for f in PLATFORM_FILES)
    _assert_same_as_published(json.loads(out.read_text()), committed, alt)


def test_hero_numbers_are_identical_on_both_committed_platforms():
    # The hero must not depend on the machine: both committed runs give the same bits.
    mac, linux = (json.loads((RESULTS / f).read_text())["round_trip"] for f in PLATFORM_FILES)
    assert mac["max_error"] == linux["max_error"] and math.isfinite(mac["max_error"])
    assert mac["identifiable"] == linux["identifiable"]
    assert mac["unsolved_identifiable"] == linux["unsolved_identifiable"] == 0
