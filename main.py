from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import contextlib
from contextlib import asynccontextmanager
import asyncio
import json
import logging
import math
from dataclasses import asdict
import numpy as np

from simulator import MarketSimulator
from arbitrage import check_static_arbitrage
from engine import CALL, implied_volatility

logger = logging.getLogger(__name__)

TICK_SECONDS = 0.5  # 2 ticks per second

sim = MarketSimulator()
# Connected WebSockets, each with a one-slot outbox. All share one tick
# sequence; a client that falls behind skips to the newest tick.
clients = {}


def finite_or_none(values):
    """JSON has no NaN and the browser's JSON.parse rejects it, so failures become null."""
    return [float(v) if math.isfinite(v) else None for v in values]


def build_payload(chain, tick=0):
    results = [
        implied_volatility(p, chain.spot, K, T, chain.r, int(f), q=chain.q)
        for p, K, T, f in zip(chain.price, chain.strike, chain.T, chain.flag, strict=True)
    ]
    iv = np.array([res.sigma for res in results])
    scan = check_static_arbitrage(chain, iv)
    return {
        "tick": tick,
        "spot": chain.spot,
        "strike": chain.strike.tolist(),
        "expiry_days": chain.expiry_days.tolist(),
        "type": ["call" if f == CALL else "put" for f in chain.flag],
        "price": finite_or_none(chain.price),
        "iv": finite_or_none(iv),
        "reason": [res.reason for res in results],
        "iterations": [res.iterations for res in results],
        "violations": [asdict(v) for v in scan.violations],
        "conditions_checked": scan.conditions_checked,
        "quotes_checked": scan.quotes_checked,
    }


def publish(text):
    """Queue text for every client without waiting on any of them."""
    for outbox in clients.values():
        if outbox.full():
            outbox.get_nowait()  # drop the stale tick the client has not sent yet
        outbox.put_nowait(text)


async def produce_ticks():
    """Tick the simulator once per interval and send the same payload to every client."""
    tick = 0
    while True:
        tick += 1
        try:
            text = json.dumps(build_payload(sim.generate_tick(), tick), allow_nan=False)
        except Exception:
            logger.exception("Failed to build tick payload")
        else:
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
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
