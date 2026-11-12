import json
import re
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "site" / "card.json"
REPORT = ROOT / "docs" / "index.html"
SCHEMA = ROOT / "tests" / "fixtures" / "card.schema.json"


def _report_data():
    page = REPORT.read_text()
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>', page, re.S
    )
    return json.loads(match.group(1))


def test_card_validates_against_the_site_schema():
    jsonschema.validate(json.loads(CARD.read_text()), json.loads(SCHEMA.read_text()))


def test_card_has_no_nan_or_infinity():
    # The site renders the card at build time; a non-finite number would break its SVG.
    def reject(const):
        raise ValueError(const)

    json.loads(CARD.read_text(), parse_constant=reject)


def test_card_rmse_is_the_reports_nearest_monthly_value():
    # The site shows the card's number; it must be the one the committed report shows.
    card = json.loads(CARD.read_text())
    slices = _report_data()["payload"]["slices"]
    monthly = min(slices, key=lambda s: abs(s["days"] - 30))
    assert card["metric"]["value"] == round(monthly["rmse_vol_points"], 6)
    assert "simulated chain" in card["data_note"]
    assert len(card["takeaway"]) <= 140
