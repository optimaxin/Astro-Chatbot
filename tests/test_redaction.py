"""Credentials never reach a log line, the usage ledger or an error message that feeds them.

Every secret below is SYNTHETIC (built from repeated filler), never a real key.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from app.observability.redact import REDACTED, redact_text, redact_value

FILL = "Zq7Lm2Xv9Kd4Rt8W" * 3
NVAPI = "nvapi-" + FILL
OPENROUTER = "sk-or-v1-" + FILL.lower()


@pytest.mark.parametrize("secret", [
    NVAPI, OPENROUTER, "sk-ant-api03-" + FILL, "sk-proj-" + FILL, "AIza" + FILL[:35], "pcsk_" + FILL[:24],
    "AKIA" + "ABCDEFGHIJKLMNOP", "ghp_" + FILL[:36], "xoxb-123456789012-" + FILL[:12],
    "eyJ" + "hbGciOiJIUzI1" + "." + "eyJzdWIiOiIxMjM0" + "." + "dBjftJeZ4CVPmB92",
])
def test_known_credential_shapes_are_removed(secret) -> None:
    out = redact_text(f"call failed: invalid key {secret} for request")
    assert secret not in out and REDACTED in out and out.startswith("call failed")


@pytest.mark.parametrize("text,leak", [
    ("Authorization: Bearer " + FILL[:22], FILL[:22]),
    ("postgresql://postgres:FakePassw0rdValue@db.example.com:5432/postgres", "FakePassw0rdValue"),
    ("rediss://default:FakeUpstashValue123@host:6379", "FakeUpstashValue123"),
    ("api_key=fake1234efgh5678", "fake1234efgh5678"),
    ('{"api_key": "fake1234efgh5678", "model": "x"}', "fake1234efgh5678"),
    ("password: fakehunter2value", "fakehunter2value"),
    ("client_secret=FakeSecret9876", "FakeSecret9876"),
])
def test_credentials_in_urls_headers_and_pairs_are_removed(text, leak) -> None:
    assert leak not in redact_text(text)


def test_a_secret_loaded_into_settings_is_removed_in_any_format() -> None:
    from app.config import settings as settings_module

    s = settings_module.settings
    original = s.nvidia
    object.__setattr__(s, "nvidia", dataclasses.replace(original, api_key="custom-format-key-0123456789"))
    try:
        out = redact_text("provider said: key custom-format-key-0123456789 is not valid")
    finally:
        object.__setattr__(s, "nvidia", original)
    assert "custom-format-key-0123456789" not in out


def test_ordinary_text_and_token_counts_are_left_alone() -> None:
    for text in ("Aapki Venus Mahadasha chal rahi hai.", "input_tokens=1200 max_tokens=700 total_tokens: 1900",
                 "rate_limited:429 after 3 attempts", "model nvidia/nemotron-3.5-bolt-e4b answered in 812 ms"):
        assert redact_text(text) == text


def test_redaction_walks_nested_structures_and_ignores_non_strings() -> None:
    value = {"a": [f"key {NVAPI}", {"b": (OPENROUTER, 3)}], "n": 5, "none": None}
    out = json.dumps(redact_value(value))
    assert NVAPI not in out and OPENROUTER not in out and '"n": 5' in out
    assert redact_text(None) is None and redact_text(12) == 12


def _format(msg: str, exc=None, **fields) -> str:
    from app.logging_setup import JsonFormatter

    record = logging.LogRecord("aicore.test", logging.WARNING, __file__, 1, msg, None, exc)
    for k, v in fields.items():
        setattr(record, k, v)
    return JsonFormatter().format(record)


def test_the_json_log_formatter_scrubs_message_fields_and_tracebacks() -> None:
    try:
        raise RuntimeError(f"401 Unauthorized: invalid api key {NVAPI}")
    except RuntimeError:
        line = _format(f"llm call failed with {OPENROUTER}", sys.exc_info(), error_message=f"body: {NVAPI}",
                       details={"headers": "Bearer " + FILL[:30]}, api_key=NVAPI)
    assert NVAPI not in line and OPENROUTER not in line and FILL[:30] not in line
    assert json.loads(line)["event"].startswith("llm call failed")           # still valid, readable JSON


def test_the_console_formatter_scrubs_too() -> None:
    from app.logging_setup import ConsoleFormatter

    record = logging.LogRecord("aicore.llm", logging.ERROR, __file__, 1, f"boom {NVAPI}", None, None)
    assert NVAPI not in ConsoleFormatter().format(record)


def test_a_credential_in_an_exception_never_reaches_the_usage_ledger(tmp_path, monkeypatch) -> None:
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)
    monkeypatch.setattr(usage_ledger, "_insert_postgres", lambda record: False)
    usage = SimpleNamespace(**{f: None for f in usage_ledger.LEDGER_FIELDS})
    usage.request_id, usage.role, usage.success = "r1", "job-main-turn", False
    usage.error = f"AuthenticationError: Incorrect API key provided: {NVAPI}"
    usage_ledger.write(usage)
    text = (tmp_path / "usage.jsonl").read_text(encoding="utf-8")
    assert NVAPI not in text and "AuthenticationError" in text


def test_settings_never_print_their_api_keys() -> None:
    from app.config.settings import settings

    shown = repr(settings)
    for group in (settings.nvidia, settings.openrouter, settings.pinecone):
        if group.api_key:
            assert group.api_key not in shown
