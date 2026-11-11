from fastapi.testclient import TestClient

import main


def test_index_page_renders():
    # The dashboard is the app's front door; Starlette 1.x rejects the old
    # TemplateResponse(name, {"request": ...}) call and the page returned 500.
    with TestClient(main.app) as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "<html" in res.text.lower()
