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
        spots.append(chain[0]["spot"])
        return chain

    monkeypatch.setattr(main.sim, "generate_tick", recording_tick)
    return spots


def read_spots(ws, n):
    return [strict_loads(ws.receive_text())["spot"] for _ in range(n)]


def is_contiguous_run(seq, of):
    return any(of[i:i + len(seq)] == seq for i in range(len(of) - len(seq) + 1))


def test_two_clients_see_one_tick_sequence(fast_ticks, produced):
    # Each connection used to tick the shared simulator itself, so two tabs
    # moved spot twice as fast and each saw only every other tick.
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as a:
            a.receive_text()
            with client.websocket_connect("/ws") as b:
                spots_a = read_spots(a, 5)
                spots_b = read_spots(b, 5)
    assert is_contiguous_run(spots_a, produced)
    assert is_contiguous_run(spots_b, produced)
    shared = set(spots_a) & set(spots_b)
    assert shared  # both clients saw some of the same ticks


def test_tick_failure_is_logged_and_other_clients_keep_receiving(
    fast_ticks, monkeypatch, caplog
):
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
                read_spots(a, 5)
                read_spots(b, 5)
    assert calls["n"] >= 3
    assert any(r.exc_info and "boom" in str(r.exc_info[1]) for r in caplog.records)


def test_failed_iv_is_sent_as_null_with_a_reason(fast_ticks, monkeypatch):
    original = main.sim.generate_tick

    def tick_with_zero_quote():
        chain = original()
        chain[-1]["price"] = 0.0  # deep OTM 90-day call quoted at zero
        return chain

    monkeypatch.setattr(main.sim, "generate_tick", tick_with_zero_quote)
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as ws:
            payload = strict_loads(ws.receive_text())
    assert payload["z"][-1] is None
    assert payload["reasons"][-1] == "zero_price"
    assert len(payload["reasons"]) == len(payload["z"])
