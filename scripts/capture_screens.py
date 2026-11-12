"""Capture the README images: a screenshot of docs/index.html and a GIF of the live page.

Run after `python report.py --seed 7`: python scripts/capture_screens.py
Needs Playwright's Chromium and Pillow (both in the dev extra).
"""

import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
FRAMES = 24  # one per 0.5 s tick: 12 seconds of the live page
GIF_WIDTH = 900


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "main.py", "--seed", "7", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1000}, color_scheme="light")
            page.goto((ROOT / "docs" / "index.html").as_uri())
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / "report.png"), full_page=True)

            for width, scheme in [(1280, "dark"), (390, "light")]:
                page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
                page.goto(url)
                page.wait_for_selector("#content:not([hidden])", timeout=20_000)
                page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT / f"dashboard-{width}-{scheme}.png"), full_page=width < 700)

            page = browser.new_page(viewport={"width": 1280, "height": 760}, color_scheme="light")
            page.goto(url)
            page.wait_for_selector("#content:not([hidden])", timeout=20_000)
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(FRAMES):
                    page.screenshot(path=os.path.join(tmp, f"f{i:03d}.png"))
                    page.wait_for_timeout(500)
                frames = []
                for i in range(FRAMES):
                    img = Image.open(os.path.join(tmp, f"f{i:03d}.png")).convert("RGB")
                    img = img.resize((GIF_WIDTH, round(img.height * GIF_WIDTH / img.width)), Image.LANCZOS)
                    frames.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
                frames[0].save(
                    OUT / "dashboard.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,
                    loop=0,
                    optimize=True,
                )
            browser.close()
    finally:
        server.terminate()
        server.wait()
    for f in sorted(OUT.iterdir()):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
