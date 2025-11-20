from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import asyncio
import json
import numpy as np

from simulator import MarketSimulator
from engine import implied_volatility, scan_arbitrage

app = FastAPI()
templates = Jinja2Templates(directory="templates")
sim = MarketSimulator()


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Get Raw Data from Simulator
            raw_chain = sim.generate_tick()

            # 2. The "Quant" Job: Calculate IV and Surface Points
            x_strikes = []
            y_expiry = []
            z_iv = []

            for opt in raw_chain:
                # Reverse engineer the IV from the price
                iv = implied_volatility(
                    opt['price'],
                    opt['spot'],
                    opt['strike'],
                    opt['time_to_expiry'],
                    opt['interest_rate'],
                    1  # Call
                )

                # Prepare data for Plotly 3D Mesh
                x_strikes.append(opt['strike'])
                y_expiry.append(opt['expiry'])
                z_iv.append(iv)

            # 3. Run Arb Scanner
            arbs = scan_arbitrage(raw_chain)

            # 4. Send Payload
            payload = {
                "spot": raw_chain[0]['spot'],
                "x": x_strikes,
                "y": y_expiry,
                "z": z_iv,
                "arbs": arbs
            }

            await websocket.send_text(json.dumps(payload))

            # Throttle to 500ms (2 ticks per second)
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Connection closed: {e}")