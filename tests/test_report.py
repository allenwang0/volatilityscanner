import json

import report


def test_results_carry_the_seed_label_and_no_nan(tmp_path):
    # The page is rendered from this file; it must say the data is simulated, and every
    # number in it must be finite (json.dumps(allow_nan=False) raises otherwise).
    out = tmp_path / "report.json"
    report.main(["--seed", "7", "--results", str(out)])
    data = json.loads(out.read_text())
    assert data["label"] == "Simulated data, seed 7"
    assert data["seed"] == 7 and len(data["slices"]) == 4
    assert data["round_trip"]["unsolved_identifiable"] == 0


def test_committed_results_match_a_fresh_run(tmp_path):
    # results/report.json is what Pages publishes; it must be what report.py produces now.
    out = tmp_path / "report.json"
    report.main(["--seed", "7", "--results", str(out)])
    committed = report.HERE / "results" / "report.json"
    assert json.loads(out.read_text()) == json.loads(committed.read_text()), (
        "results/report.json is stale: run python report.py --seed 7 --card site/card.json"
    )
