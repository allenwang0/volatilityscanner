import json
import logging

import pytest
from fastapi.testclient import TestClient

import main


def strict_loads(text):
    # Browsers' JSON.parse rejects NaN/Infinity; so must we.
    def reject(const):
        raise ValueError(f"non-standard JSON constant {const}")

    return json.loads(text, parse_constant=reject)


@pytest.fixture
def fast_ticks(monkeypatch):
    monkeypatch.setattr(main, "TICK_SECONDS", 0.01, raising=False)


@pytest.fixture
def produced(monkeypatch):
    """Record the spot of every tick the shared simulator produces."""
    spots = []
    original = main.sim.generate_tick

    def recording_tick():
        chain = original()
        spots.append(chain.spot)
        return chain

    monkeypatch.setattr(main.sim, "generate_tick", recording_tick)
    return spots


def read_ticks(ws, n):
    payloads = [strict_loads(ws.receive_text()) for _ in range(n)]
    return {p["tick"]: p["spot"] for p in payloads}


def test_two_clients_see_one_tick_sequence(fast_ticks, produced):
    # Each connection used to tick the shared simulator itself, so two tabs
    # moved spot twice as fast and each saw only every other tick.
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as a:
            a.receive_text()
            with client.websocket_connect("/ws") as b:
                ticks_a = read_ticks(a, 8)
                ticks_b = read_ticks(b, 8)
    shared = set(ticks_a) & set(ticks_b)
    assert shared  # both clients saw some of the same ticks
    for tick in shared:
        assert ticks_a[tick] == ticks_b[tick]
    # Tick n carries the simulator's n-th spot: one step per tick, not per client.
    for tick, spot in {**ticks_a, **ticks_b}.items():
        assert produced[tick - 1] == spot


def test_tick_failure_is_logged_and_other_clients_keep_receiving(fast_ticks, monkeypatch, caplog):
    original = main.sim.generate_tick
    calls = {"n": 0}

    def flaky_tick():
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return original()

    monkeypatch.setattr(main.sim, "generate_tick", flaky_tick)
    with caplog.at_level(logging.ERROR):
        with TestClient(main.app) as client:
            with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
                ticks_a = read_ticks(a, 6)
                ticks_b = read_ticks(b, 6)
    assert any(r.exc_info and "boom" in str(r.exc_info[1]) for r in caplog.records)
    # The producer survived the failed tick 3 and kept serving both clients.
    assert max(ticks_a) > 3 and max(ticks_b) > 3
    assert 3 not in ticks_a and 3 not in ticks_b


def test_failed_iv_is_sent_as_null_with_a_reason(fast_ticks, monkeypatch):
    original = main.sim.generate_tick

    def tick_with_zero_quote():
        chain = original()
        chain.price[-1] = 0.0  # deep OTM 90-day call quoted at zero
        return chain

    monkeypatch.setattr(main.sim, "generate_tick", tick_with_zero_quote)
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as ws:
            payload = strict_loads(ws.receive_text())
    assert payload["iv"][-1] is None
    assert payload["reason"][-1] == "zero_price"
    assert len(payload["reason"]) == len(payload["iv"])
