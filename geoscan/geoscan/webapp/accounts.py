"""User accounts and the activity log.

Accounts
    The administrator is the account from the environment (GEOSCAN_USER / GEOSCAN_PASSWORD). The
    administrator adds other users in the app; they are kept in <workspace>/_admin/users.json with
    salted PBKDF2 password hashes (plain passwords are never stored).
    Without GEOSCAN_PASSWORD (the local app) there is no login: everything runs as user "local",
    who is the administrator.

Activity log
    <workspace>/_admin/activity.jsonl, one JSON object per line:
        {"time": UTC ISO time, "user": ..., "action": ..., ...details}
    Actions: session_start, signed_out, upload_area, run_started, run_finished, run_failed, open_results,
    download, user_added, user_removed.

Both files are copied to permanent storage (see persist.py) a few seconds after they change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone

from . import catalog, persist

ADMIN_DIR = catalog.ROOT / "_admin"
USERS = ADMIN_DIR / "users.json"
LOG = ADMIN_DIR / "activity.jsonl"
ITERATIONS = 200_000
SESSION_GAP_S = 30 * 60          # a new "session_start" is logged after this much inactivity
_lock = threading.Lock()
_last_seen: dict[str, float] = {}
_dirty = threading.Event()


# ------------------------------------------------------------------ accounts
def login_required() -> bool:
    return bool(os.environ.get("GEOSCAN_PASSWORD"))


def admin_name() -> str:
    return os.environ.get("GEOSCAN_USER", "geoscan") if login_required() else "local"


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS).hex()


def _load() -> dict:
    try:
        return json.loads(USERS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _store(users: dict) -> None:
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    USERS.write_text(json.dumps(users, indent=2), encoding="utf-8")
    _dirty.set()


def check(user: str, password: str) -> bool:
    if secrets.compare_digest(user, admin_name()):
        return secrets.compare_digest(password, os.environ.get("GEOSCAN_PASSWORD", ""))
    rec = _load().get(user)
    return bool(rec) and secrets.compare_digest(_hash(password, rec["salt"]), rec["hash"])


def is_admin(user: str) -> bool:
    return user == admin_name()


def list_users() -> list[dict]:
    last = {}
    for e in read_log():
        last[e["user"]] = e["time"]
    out = [{"user": admin_name(), "role": "administrator", "created": "", "note": "", "last_active": last.get(admin_name(), "")}]
    for name, rec in sorted(_load().items()):
        out.append({"user": name, "role": "user", "created": rec.get("created", ""), "note": rec.get("note", ""),
                    "last_active": last.get(name, "")})
    return out


def add_user(name: str, password: str, note: str = "") -> None:
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", name):
        raise ValueError("User name: 3-32 characters, letters, digits, dot, dash or underscore.")
    if name in (admin_name(), "local"):
        raise ValueError("That user name is reserved.")
    if len(password) < 8:
        raise ValueError("Password: at least 8 characters.")
    with _lock:
        users = _load()
        salt = secrets.token_hex(16)
        users[name] = {"salt": salt, "hash": _hash(password, salt), "note": note[:200],
                       "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _store(users)


def remove_user(name: str) -> bool:
    with _lock:
        users = _load()
        if name not in users:
            return False
        del users[name]
        _store(users)
        return True


# ------------------------------------------------------------------ activity log
def record(user: str, action: str, **details) -> None:
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "user": user, "action": action,
             **{k: v for k, v in details.items() if v not in (None, "", [], {})}}
    with _lock:
        ADMIN_DIR.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    _dirty.set()


def seen(user: str, ip: str = "", agent: str = "") -> None:
    """Log the start of a session: the first request after SESSION_GAP_S of inactivity."""
    now = time.time()
    if now - _last_seen.get(user, 0) > SESSION_GAP_S:
        record(user, "session_start", ip=ip, browser=agent[:160])
    _last_seen[user] = now


def forget(user: str) -> None:
    """After a sign-out, the next request from this user starts a new session."""
    _last_seen.pop(user, None)


def read_log(limit: int | None = None) -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out[-limit:] if limit else out


def log_csv() -> str:
    import csv
    import io
    rows = read_log()
    keys = ["time", "user", "action"] + sorted({k for r in rows for k in r} - {"time", "user", "action"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    return buf.getvalue()


def _saver():
    """Copy users.json / activity.jsonl to permanent storage shortly after they change."""
    while True:
        _dirty.wait()
        time.sleep(15)                      # collect a burst of events into one upload
        _dirty.clear()
        try:
            persist.save_admin()
        except Exception:                   # storage problems must never break the app
            _dirty.set()
            time.sleep(60)


if persist.enabled():
    threading.Thread(target=_saver, daemon=True).start()
