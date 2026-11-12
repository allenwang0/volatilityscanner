"""Browser tests: start the real server and load the page in headless Chromium."""

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "test-results"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    # Noise 0 makes every tick free of static arbitrage, so the empty state is deterministic.
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--seed", "7", "--noise-sd", "0", "--port", str(port)],
        cwd=ROOT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while True:
        try:
            urllib.request.urlopen(url + "/snapshot.json", timeout=1)
            break
        except OSError:
            if time.time() > deadline:
                proc.kill()
                raise
            time.sleep(0.2)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("width", [390, 1280])
def test_dashboard_renders_without_horizontal_scroll(server, browser, scheme, width):
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server)
    assert page.inner_text("#waiting") == "Waiting for first tick" or page.is_hidden("#waiting")
    page.wait_for_selector("#views:not([hidden])", timeout=15000)
    page.wait_for_function("document.getElementById('smile').data !== undefined")

    fit_lines = page.evaluate(
        "document.getElementById('smile').data.filter(t => t.name.endsWith('SVI fit')).length"
    )
    assert fit_lines == 4
    assert page.inner_text("#arb").startswith("No static-arbitrage violations this tick")
    assert page.inner_text("#source") == "Simulated data, seed 7"
    assert "vol points" in page.inner_text("#smile-takeaway")

    SHOTS.mkdir(exist_ok=True)
    page.screenshot(path=str(SHOTS / f"dashboard-{scheme}-{width}.png"), full_page=True)
    scroll, inner = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    assert scroll <= inner, "page scrolls horizontally"
    assert errors == []
    page.close()


def test_surface_tab_and_pause(server, browser):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server)
    page.wait_for_selector("#views:not([hidden])", timeout=15000)
    page.click("#tab-surface")
    page.wait_for_function("document.getElementById('surface').data !== undefined")
    assert page.evaluate("document.getElementById('surface').data[0].type") == "surface"
    page.click("#pause")
    frozen = page.inner_text("#tick")
    page.wait_for_timeout(1500)
    assert page.inner_text("#tick") == frozen
    page.click("#pause")
    page.wait_for_function(f"document.getElementById('tick').textContent !== '{frozen}'")
    assert errors == []
    page.close()
