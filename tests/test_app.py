import json
from typing import Literal

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

import main


def test_index_page_renders():
    # The dashboard is the app's front door; Starlette 1.x rejects the old
    # TemplateResponse(name, {"request": ...}) call and the page returned 500.
    with TestClient(main.app) as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "<html" in res.text.lower()


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["call", "put"]
    strike: float
    expiry: float
    T: float
    price: float
    k: float
    iv: float | None  # null when the solver found no volatility
    iterations: int
    reason: str
    smile: bool
    bid: float | None = None
    ask: float | None = None


class Expiry(BaseModel):
    days: float
    T: float
    forward: float
    n_quotes: int
    n_failed: int
    svi: dict[str, float] | None
    rmse_vol_pts: float | None
    atm_iv: float | None
    skew_25d: float | None
    butterfly_k: list[float]
    curve: dict[str, list[float]] | None


class ViolationModel(BaseModel):
    type: Literal["monotonicity", "slope", "butterfly", "calendar", "parity"]
    expiry: float
    strikes: list[float]
    size: float
    exceeds_spread: bool | None


class Tick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tick"]
    tick: int
    source: dict
    generated_at: str
    spot: float
    rate: float
    quotes: list[Quote]
    expiries: list[Expiry]
    svi_calendar_crossings: list[dict]
    violations: list[ViolationModel]
    checks: dict[str, int]


def test_websocket_payload_matches_the_schema_the_page_reads(monkeypatch):
    # The front end indexes these fields directly; a renamed or missing field
    # breaks the page silently, so the schema is pinned here.
    monkeypatch.setattr(main, "TICK_SECONDS", 0.01)
    original = main.source.generate_tick

    def with_failed_quote():
        chain = original()
        chain[0]["price"] = 0.0
        return chain

    monkeypatch.setattr(main.source, "generate_tick", with_failed_quote)
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as ws:
            tick = Tick.model_validate(json.loads(ws.receive_text()))
    assert tick.quotes[0].iv is None
    assert len(tick.expiries) == 4
