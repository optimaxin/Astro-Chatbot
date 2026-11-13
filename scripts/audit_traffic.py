"""Read-only integrity audit of ai_usage / ai_evaluations traffic labels.

    python -m scripts.audit_traffic

Every query runs in a READ ONLY transaction; nothing is written. Checks:
  * label distribution, and whether 'unknown' (migration 061) exists yet
  * rows with no user id labelled 'user'   (must be 0 once 061 is applied)
  * known script/test user ids labelled 'user'
  * test-suite rows written since the conftest fix (must be 0)
  * the 11 timing-probe rows (2026-11-12 22:26-22:29 UTC, user '-'): present, label
  * anonymous visitors (signed 'anon:' ids) labelled 'user', listed separately
  * ai_evaluations labels vs the ai_usage row of the same request
Exit code 1 if a check that must hold (independent of 061) fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import usage_store  # noqa: E402

SCRIPT_USERS = ("-", "nvidia-verify", "connection-test", "pipeline-verify-user", "sec-user-a", "sec-user-b",
                "test-user-001", "test-user-999")
CONFTEST_FIX = "2026-11-12 21:30+00"           # tests stopped writing evaluations
PROBE_WINDOW = ("2026-11-12 22:26+00", "2026-11-12 22:30+00")


def main() -> int:
    pool = usage_store._get_pool()
    if pool is None:
        print("No usage database configured.")
        return 2
    must_fail = []
    with pool.connection() as c:
        c.execute("SET TRANSACTION READ ONLY")
        q = lambda sql, *a: c.execute(sql, a).fetchall()  # noqa: E731

        applied = {r[0] for r in q("SELECT name FROM _migrations WHERE name LIKE '058_ai%%' OR name LIKE '061_%%'")}
        print("migrations applied:", sorted(applied) or "none")
        print("\nai_usage labels:", q("SELECT traffic_source, count(*) FROM ai_usage GROUP BY 1 ORDER BY 1"))
        print("ai_evaluations labels:", q("SELECT traffic_source, count(*) FROM ai_evaluations GROUP BY 1 ORDER BY 1"))

        no_user = q("SELECT count(*) FROM ai_usage WHERE traffic_source='user' AND COALESCE(user_id,'') IN ('', '-')")[0][0]
        print(f"\n[{'ok ' if no_user == 0 else 'PENDING-061'}] rows with no user labelled 'user': {no_user}")

        scripts = q("SELECT user_id, count(*) FROM ai_usage WHERE traffic_source='user' AND user_id = ANY(%s) "
                    "GROUP BY 1", list(SCRIPT_USERS))
        print(f"[{'ok ' if not scripts else 'PENDING-061'}] script/test user ids labelled 'user': {scripts}")

        tests_after = q("SELECT count(*) FROM ai_evaluations WHERE conversation_id LIKE 'pytest%%' AND created_at > %s",
                        CONFTEST_FIX)[0][0]
        tests_usage = q("SELECT count(*) FROM ai_usage WHERE conversation_id LIKE 'pytest%%' AND created_at > %s",
                        CONFTEST_FIX)[0][0]
        ok = tests_after == 0 and tests_usage == 0
        must_fail += [] if ok else ["test rows written after the conftest fix"]
        print(f"[{'ok ' if ok else 'FAIL'}] test-suite rows since the fix: evaluations={tests_after} usage={tests_usage}")

        probe = q("SELECT traffic_source, count(*) FROM ai_usage WHERE user_id='-' AND created_at BETWEEN %s AND %s "
                  "GROUP BY 1", *PROBE_WINDOW)
        n = sum(r[1] for r in probe)
        must_fail += [] if n == 11 else [f"expected 11 timing-probe rows preserved, found {n}"]
        print(f"[{'ok ' if n == 11 else 'FAIL'}] timing-probe rows preserved: {n} (labels {dict(probe)}; "
              "become 'debug' when 061 runs)")

        anon = q("SELECT count(*), count(DISTINCT user_id), min(created_at), max(created_at) FROM ai_usage "
                 "WHERE traffic_source='user' AND user_id LIKE 'anon:%%'")[0]
        print(f"[info] signed anonymous visitors labelled 'user': {anon[0]} rows, {anon[1]} ids, {anon[2]} .. {anon[3]}"
              "  (pre-2026-11-13 ones become 'unknown' with 061; later ones are web visitors)")

        mismatch = q("""SELECT e.traffic_source, u.src, count(*) FROM ai_evaluations e
                        JOIN (SELECT request_id, min(traffic_source) src FROM ai_usage GROUP BY 1) u
                          ON u.request_id = e.request_id
                        WHERE e.traffic_source <> u.src GROUP BY 1,2""")
        print(f"[{'ok ' if not mismatch else 'PENDING-061'}] evaluation label differs from its usage row: {mismatch}")

        labels = {r[0] for r in q("SELECT DISTINCT traffic_source FROM ai_usage")}
        print("[info] distinct labels:", sorted(labels))
    print("\nMUST-HOLD FAILURES:", must_fail or "none")
    return 1 if must_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
