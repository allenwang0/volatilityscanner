import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

import main
from simulator import MarketSimulator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SVI(Strict):
    a: float
    b: float
    rho: float
    m: float
    s: float


class Slice(Strict):
    days: float
    forward: float
    svi: SVI
    rmse_vol_points: float
    min_density_g: float
    atm_iv: float
    skew_25d: float
    curve_k: list[float]
    curve_iv: list[float]


class Violation(Strict):
    type: str
    expiry_days: float
    strikes: list[float]
    size: float
    exceeds_spread: bool | None


class Surface(Strict):
    k: list[float]
    days: list[float]
    iv: list[list[float]]


class Payload(Strict):
    tick: int
    source: str
    spot: float
    k: list[float]
    strike: list[float]
    expiry_days: list[float]
    type: list[str]
    price: list[float | None]
    iv: list[float | None]
    reason: list[str]
    iterations: list[int]
    violations: list[Violation]
    conditions_checked: int
    quotes_checked: int
    slices: list[Slice]
    surface: Surface | None
    fitted_calendar_crossings: list[list[float]]


@pytest.fixture
def seeded(monkeypatch):
    monkeypatch.setattr(main, "sim", MarketSimulator(seed=7, tick_size=0.05))
    monkeypatch.setattr(main, "TICK_SECONDS", 0.01)


def test_page_and_assets_are_served():
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        assert "plotly-2.35.2.min.js" in client.get("/").text
        assert client.get("/static/charts.js").status_code == 200


def test_websocket_payload_matches_the_schema_and_allows_null_ivs(seeded):
    # Tick-size rounding zeroes some 7-day wings, so this payload has null IVs.
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        payload = Payload.model_validate_json(ws.receive_text())
    assert None in payload.iv
    assert len(payload.slices) == 4
    lengths = {len(payload.strike), len(payload.iv), len(payload.reason), len(payload.k)}
    assert lengths == {payload.quotes_checked}


def test_snapshot_returns_the_latest_tick(seeded):
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        tick = json.loads(ws.receive_text())["tick"]
        snap = client.get("/snapshot.json")
    assert snap.status_code == 200
    body = Payload.model_validate_json(snap.text)
    assert body.tick >= tick and body.slices[0].svi.b >= 0


class FakeLive:
    """Repeats one chain (no new quotes), and fails once after a client connects (an outage)."""

    def __init__(self):
        self.failed = False
        self.chain = MarketSimulator(seed=1).generate_tick()

    def generate_tick(self):
        if main.clients and not self.failed:
            self.failed = True
            raise ConnectionError("yahoo timed out")
        return self.chain


def test_live_mode_skips_unchanged_chains_and_reports_errors(monkeypatch):
    monkeypatch.setattr(main, "sim", FakeLive())
    monkeypatch.setattr(main, "live", True)
    monkeypatch.setattr(main, "LIVE_REFRESH_SECONDS", 0.01)
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        first = json.loads(ws.receive_text())
        second = json.loads(ws.receive_text())
    # The new client gets the last good tick at once; unchanged chains are not resent,
    # so the next message is the outage, and the client keeps the tick it has.
    assert first["tick"] == 1
    assert second == {"error": "ConnectionError: yahoo timed out"}


def test_parse_args_defaults_to_the_offline_simulator():
    args = main.parse_args([])
    assert args.source == "simulated" and args.rate is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_finite_or_none_drops_non_finite_values(value):
    assert main.finite_or_none(np.array([1.0, value])) == [1.0, None]
