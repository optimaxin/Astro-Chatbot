"""Back up or restore the playground database.

    python -m scripts.backup_db backup  [DEST_DIR]   # consistent online copy
    python -m scripts.backup_db restore BACKUP_FILE  # stop the service first

Backups use SQLite's online backup API, so they are safe while the service
runs. Restore replaces PLAYGROUND_DB_PATH and keeps the old file alongside.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import storage  # noqa: E402
from app.settings import settings  # noqa: E402


def backup(dest_dir: Path) -> Path:
    dest = dest_dir / f"playground-backup-{time.strftime('%Y%m%d-%H%M%S')}.db"
    storage.backup_to(dest)
    storage.close()
    return dest


def restore(source: Path) -> Path:
    with sqlite3.connect(source) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SystemExit(f"{source} failed the integrity check; not restoring.")
    target = settings.db_path
    if target.exists():
        keep = target.with_name(f"{target.name}.before-restore-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.move(target, keep)
        for suffix in ("-wal", "-shm"):
            Path(f"{target}{suffix}").unlink(missing_ok=True)
        print(f"previous database kept at {keep}")
    shutil.copy2(source, target)
    return target


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "backup":
        dest_dir = Path(argv[1]) if len(argv) > 1 else settings.db_path.parent / "backups"
        print(backup(dest_dir))
        return 0
    if len(argv) == 2 and argv[0] == "restore":
        print(f"restored into {restore(Path(argv[1]))}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
