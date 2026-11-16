import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argon2 import PasswordHasher  # noqa: E402

ADMIN_PASSWORD = "correct horse battery staple"
_static = Path(tempfile.mkdtemp())
(_static / "assets").mkdir()
(_static / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
(_static / "assets" / "app.js").write_text("console.log('app')", encoding="utf-8")

os.environ.update({
    "PLAYGROUND_ACCESS_TOKEN": "",
    "GEMINI_API_KEY": "fake-key-for-tests",
    "AI_CORE_URL": "http://ai-core.test",
    "ADMIN_PASSWORD_HASH": PasswordHasher().hash(ADMIN_PASSWORD),
    "SESSION_SECRET": "test-session-secret",
    "PLAYGROUND_COOKIE_SECURE": "false",
    "PLAYGROUND_STATIC_DIR": str(_static),
    "PLAYGROUND_DAILY_REQUEST_LIMIT": "0",
    "PLAYGROUND_DAILY_COST_LIMIT_USD": "0",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import auth, storage  # noqa: E402
from app.main import app  # noqa: E402

CLIENT = {"X-Playground-Client": "client-aaaa1111"}
OTHER = {"X-Playground-Client": "client-bbbb2222"}


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    storage.reset_for_tests(tmp_path / "test.db")
    auth.reset_rate_limits()
    yield
    storage.close()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin(client):
    """A TestClient signed in as admin; `.csrf` holds the CSRF token."""
    res = client.post("/api/admin/login", json={"password": ADMIN_PASSWORD})
    assert res.status_code == 200, res.text
    client.csrf = res.json()["csrf_token"]
    return client
