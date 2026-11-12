import json
import re

import report


def test_report_builds_with_the_seed_label_and_no_nan(tmp_path):
    # The Pages report must say its data is simulated, and every number on it must be real.
    out = tmp_path / "index.html"
    report.main(["--seed", "7", "--out", str(out)])
    page = out.read_text()
    assert "Simulated data, seed 7" in page
    assert "NaN" not in page
    data = json.loads(
        re.search(
            r'<script type="application/json" id="report-data">(.*?)</script>', page, re.S
        ).group(1)
    )
    assert data["seed"] == 7 and len(data["payload"]["slices"]) == 4
    assert data["round_trip"]["unsolved_identifiable"] == 0
