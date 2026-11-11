from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import asyncio
import json
import logging
import math
from dataclasses import asdict
import numpy as np

from simulator import MarketSimulator
from engine import implied_volatility, scan_arbitrage

logger = logging.getLogger(__name__)

TICK_SECONDS = 0.5  # 2 ticks per second

sim = MarketSimulator()
clients = set()  # connected WebSockets; all share one tick sequence


def build_payload(raw_chain):
    # 1. The "Quant" Job: Calculate IV and Surface Points
    x_strikes = []
    y_expiry = []
    z_iv = []
    reasons = []

    for opt in raw_chain:
        # Reverse engineer the IV from the price
        result = implied_volatility(
            opt['price'],
            opt['spot'],
            opt['strike'],
            opt['time_to_expiry'],
            opt['interest_rate'],
            1  # Call
        )

        # Prepare data for Plotly 3D Mesh; JSON has no NaN, so failures are null
        x_strikes.append(opt['strike'])
        y_expiry.append(opt['expiry'])
        z_iv.append(None if math.isnan(result.sigma) else result.sigma)
        reasons.append(result.reason)

    # 2. Run Arb Scanner
    arbs = scan_arbitrage(raw_chain)

    return {
        "spot": raw_chain[0]['spot'],
        "x": x_strikes,
        "y": y_expiry,
        "z": z_iv,
        "reasons": reasons,
        "arbs": [asdict(a) for a in arbs]
    }


async def produce_ticks():
    """Tick the simulator once per interval and send the same payload to every client."""
    while True:
        try:
            text = json.dumps(build_payload(sim.generate_tick()), allow_nan=False)
        except Exception:
            logger.exception("Failed to build tick payload")
        else:
            for ws in list(clients):
                try:
                    await ws.send_text(text)
                except Exception:
                    logger.exception("Failed to send to client; dropping it")
                    clients.discard(ws)

        await asyncio.sleep(TICK_SECONDS)


@asynccontextmanager
async def lifespan(app):
    producer = asyncio.create_task(produce_ticks())
    yield
    producer.cancel()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        # The producer does the sending; wait here until the client leaves
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
