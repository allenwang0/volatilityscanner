import json

import report


def test_report_is_labelled_and_has_no_nan(tmp_path):
    # Readers of the static report cannot run anything, so it must say where
    # its data came from, and a NaN anywhere means a failed number slipped in.
    out = tmp_path / "index.html"
    report.main(["--seed", "7", "--out", str(out)])
    page = out.read_text()
    assert "Simulated data, seed 7" in page
    assert "NaN" not in page
    results = json.loads((tmp_path / "report.json").read_text())
    assert results["seed"] == 7
    assert all(e["rmse_vol_pts"] < 0.5 for e in results["expiries"])
    assert results["round_trip"]["solved_identifiable"] == results["round_trip"]["identifiable"]
    assert results["round_trip"]["max_error"] < 1e-9
