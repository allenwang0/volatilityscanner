import argparse
import asyncio
import contextlib
import json
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from arbitrage import LIVE_TOLERANCE, SIMULATED_TOLERANCE, check_static_arbitrage
from engine import CALL, implied_volatility_vec
from simulator import IV_NOISE_SD, MarketSimulator
from surface import fit_surface, fitted_calendar_crossings, slice_curve, surface_grid

logger = logging.getLogger(__name__)

TICK_SECONDS = 0.5  # 2 ticks per second
LIVE_REFRESH_SECONDS = 30  # a yfinance chain takes 2 to 3 s per expiry to download
HERE = Path(__file__).parent

# The chain source. Simulated by default; --source yfinance swaps in a LiveFeed.
sim = MarketSimulator()
live = False
latest = {"text": None}
# Connected WebSockets, each with a one-slot outbox. All share one tick
# sequence; a client that falls behind skips to the newest tick.
clients: dict[WebSocket, asyncio.Queue[str]] = {}


def finite_or_none(values):
    """JSON has no NaN and the browser's JSON.parse rejects it, so failures become null."""
    return [float(v) if math.isfinite(v) else None for v in values]


def build_payload(chain, tick=0):
    solved = implied_volatility_vec(
        chain.price, chain.spot, chain.strike, chain.T, chain.r, chain.flag, chain.q
    )
    iv = solved.sigma
    has_quotes = bool(np.isfinite(chain.bid).any())
    scan = check_static_arbitrage(chain, iv, LIVE_TOLERANCE if has_quotes else SIMULATED_TOLERANCE)
    slices = fit_surface(chain, iv)
    return {
        "tick": tick,
        "source": chain.source,
        "spot": chain.spot,
        "k": np.log(chain.strike / chain.forward).tolist(),
        "strike": chain.strike.tolist(),
        "expiry_days": chain.expiry_days.tolist(),
        "type": ["call" if f == CALL else "put" for f in chain.flag],
        "price": finite_or_none(chain.price),
        "iv": finite_or_none(iv),
        "reason": solved.reason.tolist(),
        "iterations": solved.iterations.tolist(),
        "violations": [asdict(v) for v in scan.violations],
        "conditions_checked": scan.conditions_checked,
        "quotes_checked": scan.quotes_checked,
        "slices": [slice_payload(sl) for sl in slices],
        "surface": surface_grid(slices) if slices else None,
        "fitted_calendar_crossings": fitted_calendar_crossings(slices) if slices else [],
    }


def slice_payload(sl):
    k, iv = slice_curve(sl)
    return {
        "days": sl.days,
        "forward": sl.forward,
        "svi": asdict(sl.fit.params),
        "rmse_vol_points": sl.fit.rmse_vol_points,
        "min_density_g": sl.fit.min_density_g,
        "atm_iv": sl.atm_iv,
        "skew_25d": sl.skew_25d if math.isfinite(sl.skew_25d) else None,
        "curve_k": k.tolist(),
        "curve_iv": iv.tolist(),
    }


def publish(text):
    """Queue text for every client without waiting on any of them."""
    for outbox in clients.values():
        if outbox.full():
            outbox.get_nowait()  # drop the stale tick the client has not sent yet
        outbox.put_nowait(text)


def encode_tick(tick, previous_prices=None):
    """Return (json text, chain prices); text is None when a live chain has not changed."""
    chain = sim.generate_tick()
    if live and previous_prices is not None and np.array_equal(chain.price, previous_prices):
        return None, chain.price
    return json.dumps(build_payload(chain, tick), allow_nan=False), chain.price


async def produce_ticks():
    """Tick the source once per interval and send the same payload to every client."""
    tick = 0
    prices = None
    while True:
        tick += 1
        try:
            text, prices = await asyncio.to_thread(encode_tick, tick, prices)
        except Exception as exc:
            logger.exception("Failed to build tick payload")
            if live:
                # Clients keep the last good snapshot and show it as stale.
                publish(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        else:
            if text is not None:
                latest["text"] = text
                publish(text)

        await asyncio.sleep(LIVE_REFRESH_SECONDS if live else TICK_SECONDS)


@asynccontextmanager
async def lifespan(app):
    latest["text"] = None  # never serve a tick from a previous run
    producer = asyncio.create_task(produce_ticks())
    yield
    producer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await producer


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/snapshot.json")
async def snapshot():
    """The current tick: chain, IVs, reason codes, SVI parameters and violations."""
    if latest["text"] is None:
        return JSONResponse({"error": "no tick yet"}, status_code=503)
    return Response(
        latest["text"],
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=snapshot.json"},
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
    if latest["text"] is not None:
        # Live mode refreshes every 30 s; a new client should not wait that long.
        clients[websocket].put_nowait(latest["text"])
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve the volatility dashboard.")
    parser.add_argument("--source", choices=["simulated", "yfinance"], default="simulated")
    parser.add_argument("--seed", type=int, default=None, help="seed for the simulated chain")
    parser.add_argument("--tick-size", type=float, default=None, help="round simulated prices")
    parser.add_argument(
        "--noise-sd", type=float, default=IV_NOISE_SD, help="IV noise per simulated quote"
    )
    parser.add_argument("--symbol", default="SPY", help="yfinance symbol (live mode)")
    parser.add_argument(
        "--rate", type=float, default=None, help="risk-free rate for live mode (default 0.0406)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def configure(args):
    global sim, live
    if args.source == "yfinance":
        from live_feed import DEFAULT_RATE, LiveFeed

        rate = DEFAULT_RATE if args.rate is None else args.rate
        sim, live = LiveFeed(args.symbol, rate), True
    else:
        sim = MarketSimulator(seed=args.seed, tick_size=args.tick_size, noise_sd=args.noise_sd)
        live = False


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    cli = parse_args()
    configure(cli)
    uvicorn.run(app, host=cli.host, port=cli.port)
