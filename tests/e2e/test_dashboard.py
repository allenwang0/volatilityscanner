"""Browser tests: start the server, load the page, check the views (needs Chromium and the Plotly CDN)."""

import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.e2e


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(*args):
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--port", str(port), *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    for _ in range(100):
        try:
            urllib.request.urlopen(url, timeout=1)
            return proc, url
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("server did not start")


@pytest.fixture(scope="module")
def clean_server():
    # Jitter 0: noiseless prices, so the page must show the empty state.
    proc, url = start_server("--jitter", "0")
    yield url
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def noisy_server():
    proc, url = start_server()
    yield url
    proc.terminate()
    proc.wait()


def wait_for_tick(page, url):
    page.goto(url)
    page.wait_for_selector("#content:not([hidden])", timeout=20_000)
    page.wait_for_function("document.querySelector('#smile .main-svg') !== null", timeout=20_000)


def expiry_groups(page):
    return page.evaluate(
        "new Set(document.getElementById('smile').data.map(t => t.legendgroup).filter(Boolean)).size"
    )


@pytest.mark.parametrize("width", [390, 1280])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_dashboard_first_tick(browser, clean_server, width, scheme, tmp_path):
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
    wait_for_tick(page, clean_server)
    assert expiry_groups(page) == 4  # one fit line and one marker set per expiry
    empty = page.locator("#arb-empty")
    assert empty.is_visible()
    assert empty.inner_text().startswith("No static-arbitrage violations this tick (checked: ")
    page.screenshot(path=str(tmp_path / f"dashboard-{width}-{scheme}.png"), full_page=True)
    # A phone-width page must not scroll sideways.
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.close()


def test_clicking_a_violation_rings_its_strikes(browser, noisy_server):
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    wait_for_tick(page, noisy_server)
    page.click("#pause")  # freeze the view so the row stays put
    page.wait_for_selector("#arb-body tr", timeout=20_000)
    page.click("#arb-body tr >> nth=0")
    names = page.evaluate("document.getElementById('smile').data.map(t => t.name)")
    assert "selected violation" in names
    page.close()


def test_surface_tab_draws_a_surface(browser, clean_server):
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    wait_for_tick(page, clean_server)
    page.click("[data-tab=surface]")
    page.wait_for_function("document.getElementById('surface').data !== undefined", timeout=10_000)
    types = page.evaluate("document.getElementById('surface').data.map(t => t.type)")
    assert types == ["surface", "scatter3d"]
    page.close()
