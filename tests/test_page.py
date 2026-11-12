import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "static", "dashboard.js")


def test_socket_scheme_follows_page_scheme():
    # Browsers block ws:// from an https:// page, so the scheme must follow it.
    js = open(SCRIPT).read()
    assert '"ws://" + location.host' not in js
    assert 'location.protocol === "https:" ? "wss://" : "ws://"' in js
