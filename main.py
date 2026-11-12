"""FastAPI app: one producer ticks the option source and fans each snapshot out to every client."""

import argparse
import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from analysis import analyze_chain
from engine import ARB_TOL
from simulator import DEFAULT_JITTER, MarketSimulator

logger = logging.getLogger(__name__)

TICK_SECONDS = 0.5  # simulated mode: 2 ticks per second
HERE = os.path.dirname(os.path.abspath(__file__))

source = MarketSimulator()
arb_tol = ARB_TOL
# Connected WebSockets, each with a one-slot outbox. All share one tick
# sequence; a client that falls behind skips to the newest tick.
clients: dict = {}
state: dict = {"latest": None, "fits": {}, "last_chain": None}


def build_payload(raw_chain, tick=0):
    snapshot, state["fits"] = analyze_chain(raw_chain, state["fits"], arb_tol)
    return {
        "kind": "tick",
        "tick": tick,
        "source": source.describe(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **snapshot,
    }


def publish(text):
    """Queue text for every client without waiting on any of them."""
    for outbox in clients.values():
        if outbox.full():
            outbox.get_nowait()  # drop the stale tick the client has not sent yet
        outbox.put_nowait(text)


def make_tick(tick):
    """Fetch a chain and build its payload text; None when the chain has not changed."""
    chain = source.generate_tick()
    if chain == state["last_chain"]:
        return None  # live quotes refresh slower than we poll
    state["last_chain"] = chain
    payload = build_payload(chain, tick)
    text = json.dumps(payload, allow_nan=False)
    state["latest"] = payload
    return text


async def produce_ticks():
    """Tick the source once per interval and send the same payload to every client."""
    tick = 0
    while True:
        tick += 1
        try:
            # In a thread: a live chain download takes seconds, and SVI fits
            # take tens of milliseconds; neither should stall the sockets.
            text = await asyncio.to_thread(make_tick, tick)
        except Exception as exc:
            logger.exception("Failed to build tick payload")
            publish(json.dumps({"kind": "error", "tick": tick, "message": f"{type(exc).__name__}: {exc}"}))
        else:
            if text is not None:
                publish(text)

        await asyncio.sleep(TICK_SECONDS)


@asynccontextmanager
async def lifespan(app):
    producer = asyncio.create_task(produce_ticks())
    yield
    producer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await producer


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/snapshot.json")
async def snapshot():
    """The latest tick's chain, IVs, reason codes, SVI fits and violations."""
    if state["latest"] is None:
        return JSONResponse({"detail": "Waiting for first tick"}, status_code=503)
    return JSONResponse(
        state["latest"],
        headers={"Content-Disposition": f'attachment; filename="snapshot-{state["latest"]["tick"]}.json"'},
    )


async def send_from(websocket, outbox):
    while True:
        await websocket.send_text(await outbox.get())


async def wait_for_close(websocket):
    # The client sends nothing; receive_text returns or raises when it leaves
    while True:
        await websocket.receive_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients[websocket] = asyncio.Queue(maxsize=1)
    tasks = [
        asyncio.create_task(send_from(websocket, clients[websocket])),
        asyncio.create_task(wait_for_close(websocket)),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                logger.error("WebSocket client failed; dropping it", exc_info=exc)
    finally:
        clients.pop(websocket, None)
        # Not awaited: under Starlette's anyio cancel scope, awaiting here
        # during shutdown cancels the whole session.
        for task in tasks:
            task.cancel()


def main(argv=None):
    import uvicorn

    global source, TICK_SECONDS, arb_tol
    parser = argparse.ArgumentParser(description="Serve the volatility dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7, help="simulator seed")
    parser.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_JITTER,
        help="IV noise sd per simulated quote; 0 for clean prices",
    )
    parser.add_argument(
        "--tick-size", type=float, default=None, help="round simulated prices to this increment"
    )
    args = parser.parse_args(argv)
    source = MarketSimulator(seed=args.seed, jitter=args.jitter, tick_size=args.tick_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
