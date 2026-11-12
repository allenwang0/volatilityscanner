"""Fan-out behaviour with fake sockets, so a stalled client can be simulated."""

import asyncio
import json

from fastapi import WebSocketDisconnect

import main


class FakeSocket:
    def __init__(self, stall_first_send=False, fail_send=False):
        self.sent = []
        self.release = asyncio.Event()
        self.stall_first_send = stall_first_send
        self.fail_send = fail_send
        self.closed = asyncio.Event()

    async def accept(self):
        pass

    async def receive_text(self):
        await self.closed.wait()  # a client that never sends anything
        raise WebSocketDisconnect()

    async def send_text(self, text):
        if self.fail_send:
            raise WebSocketDisconnect()  # client vanished mid-send
        if self.stall_first_send and not self.sent and not self.release.is_set():
            await self.release.wait()  # network buffer full
        self.sent.append(json.loads(text)["tick"])


def tiny_payload(chain, tick):
    # These tests are about delivery; a full surface fit per tick would only slow them.
    return {"tick": tick}


async def run_with(sockets, seconds):
    producer = asyncio.create_task(main.produce_ticks())
    endpoints = [asyncio.create_task(main.websocket_endpoint(s)) for s in sockets]
    await asyncio.sleep(seconds)
    return producer, endpoints


async def stop(producer, endpoints, sockets):
    for s in sockets:
        s.closed.set()
        s.release.set()
    await asyncio.gather(*endpoints, return_exceptions=True)
    producer.cancel()
    await asyncio.gather(producer, return_exceptions=True)


def test_stalled_client_does_not_hold_up_others(monkeypatch):
    # The producer used to await each send in turn, so one client with a full
    # network buffer froze every other client's feed.
    monkeypatch.setattr(main, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(main, "build_payload", tiny_payload)

    async def scenario():
        slow, fast = FakeSocket(stall_first_send=True), FakeSocket()
        producer, endpoints = await run_with([slow, fast], 0.3)
        fast_count = len(fast.sent)
        latest = fast.sent[-1]
        slow.release.set()
        await asyncio.sleep(0.05)
        await stop(producer, endpoints, [slow, fast])
        return fast_count, latest, slow.sent

    fast_count, latest, slow_sent = asyncio.run(scenario())
    assert fast_count >= 10
    # The stalled client gets the newest tick next, not a backlog of stale
    # ones: at most one tick waits per client, so memory stays bounded.
    assert len(slow_sent) >= 2
    assert slow_sent[1] >= latest


def test_client_that_disconnects_mid_send_is_dropped(monkeypatch):
    monkeypatch.setattr(main, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(main, "build_payload", tiny_payload)

    async def scenario():
        gone, ok = FakeSocket(fail_send=True), FakeSocket()
        producer, endpoints = await run_with([gone, ok], 0.2)
        gone_finished = endpoints[0].done()
        registered = gone in main.clients
        await stop(producer, endpoints, [gone, ok])
        return gone_finished, registered, ok.sent

    gone_finished, registered, ok_sent = asyncio.run(scenario())
    assert gone_finished and not registered
    assert len(ok_sent) >= 5
    assert not main.clients
