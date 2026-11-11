import os

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html")


def test_socket_scheme_follows_page_scheme():
    # Browsers block ws:// from an https:// page, so the scheme must follow it.
    html = open(PAGE).read()
    assert '"ws://" + location.host' not in html
    assert 'location.protocol === "https:" ? "wss://" : "ws://"' in html
