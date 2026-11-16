"""Print an Argon2id hash for ADMIN_PASSWORD_HASH.

    python -m scripts.hash_password

The password is read without echo and is never written anywhere.
"""

from __future__ import annotations

import getpass
import sys

from argon2 import PasswordHasher


def main() -> int:
    password = getpass.getpass("Admin password: ")
    if len(password) < 12:
        print("Use at least 12 characters.", file=sys.stderr)
        return 1
    if getpass.getpass("Repeat: ") != password:
        print("Passwords do not match.", file=sys.stderr)
        return 1
    print(PasswordHasher().hash(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
