import asyncio
import contextlib
import json
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from engine import implied_volatility, scan_arbitrage
from simulator import MarketSimulator

logger = logging.getLogger(__name__)

TICK_SECONDS = 0.5  # 2 ticks per second

sim = MarketSimulator()
# Connected WebSockets, each with a one-slot outbox. All share one tick
# sequence; a client that falls behind skips to the newest tick.
clients = {}


def build_payload(raw_chain, tick=0):
    # 1. The "Quant" Job: Calculate IV and Surface Points
    x_strikes = []
    y_expiry = []
    z_iv = []
    reasons = []

    for opt in raw_chain:
        # Reverse engineer the IV from the price
        result = implied_volatility(
            opt["price"],
            opt["spot"],
            opt["strike"],
            opt["time_to_expiry"],
            opt["interest_rate"],
            1,  # Call
        )

        # Prepare data for Plotly 3D Mesh; JSON has no NaN, so failures are null
        x_strikes.append(opt["strike"])
        y_expiry.append(opt["expiry"])
        z_iv.append(None if math.isnan(result.sigma) else result.sigma)
        reasons.append(result.reason)

    # 2. Run Arb Scanner
    arbs = scan_arbitrage(raw_chain)

    return {
        "tick": tick,
        "spot": raw_chain[0]["spot"],
        "x": x_strikes,
        "y": y_expiry,
        "z": z_iv,
        "reasons": reasons,
        "arbs": [asdict(a) for a in arbs],
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
    return templates.TemplateResponse(request, "index.html")


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
