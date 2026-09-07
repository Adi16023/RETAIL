"""
On-disk store for AryaChat conversations — one JSON file per chat, no database.

A chat belongs to one SCOPE (the book, identified by the data fingerprint of
every account's rows) and remembers which account it is currently about. If
the underlying rows change, every figure a past answer quoted may no longer
exist, so the old chats are left where they are (under the old fingerprint)
and the book starts with a clean list. A manager can open as many chats as
they like; each keeps its own context and its own account in focus.

A chat file holds:
  chat_id, scope, fingerprint, title, created_at, updated_at, model
  focus_account        the account the chat is currently about, or null
  summary              compacted text of the turns before `summarised_through`
  summarised_through   index into turns; everything before it is in the summary
  turns                [{ts, role, text, model?, tools_used?, tool_log?,
                         accounts_touched?, unsourced_figures?, usage?, truncated?}]

Tool logs are kept on the turn that used them. They make every answer
auditable after the fact ("how did it know that?") and cost a few hundred
bytes a turn. The directory is gitignored.
"""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, date
from pathlib import Path

CHAT_DIR = Path(__file__).resolve().parents[2] / ".cache" / "aryachat"
TITLE_CHARS = 48


def _folder(scope: str, fingerprint: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in f"{scope}_{fingerprint}")
    return CHAT_DIR / safe


def _path(scope: str, fingerprint: str, chat_id: str) -> Path:
    return _folder(scope, fingerprint) / f"{chat_id}.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_chat(scope: str, fingerprint: str, model: str, focus_account: str | None = None) -> dict:
    chat = {
        "chat_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}",
        "scope": scope,
        "fingerprint": fingerprint,
        "title": "New chat",
        "created_at": _now(),
        "updated_at": _now(),
        "model": model,
        "focus_account": focus_account,
        "summary": None,
        "summarised_through": 0,
        "turns": [],
    }
    save_chat(chat)
    return chat


def _scope_of(chat: dict) -> str:
    # Chats written by the earlier, per-account version carried the account
    # id as their scope under a different key. They still load and save.
    return chat.get("scope") or chat.get("account_id") or "book"


def save_chat(chat: dict) -> None:
    folder = _folder(_scope_of(chat), chat["fingerprint"])
    folder.mkdir(parents=True, exist_ok=True)
    chat["updated_at"] = _now()
    _path(_scope_of(chat), chat["fingerprint"], chat["chat_id"]).write_text(
        json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_chat(scope: str, fingerprint: str, chat_id: str) -> dict | None:
    path = _path(scope, fingerprint, chat_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt file must never take the page down; it reads as missing.
        return None


def delete_chat(scope: str, fingerprint: str, chat_id: str) -> None:
    _path(scope, fingerprint, chat_id).unlink(missing_ok=True)


GROUP_ORDER = ("Today", "Yesterday", "Previous 7 days", "Older")


def conversation_group_label(iso: str, today: date | None = None) -> str:
    """Same buckets as youkti-app's Arya chat sidebar."""
    today = today or date.today()
    try:
        day = datetime.fromisoformat(iso.replace("Z", "")).date()
    except (TypeError, ValueError):
        return "Older"
    delta = (today - day).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta < 7:
        return "Previous 7 days"
    return "Older"


def group_conversations(rows: list[dict], today: date | None = None) -> list[tuple[str, list[dict]]]:
    buckets = {label: [] for label in GROUP_ORDER}
    for row in rows:
        buckets[conversation_group_label(row.get("updated_at") or row.get("created_at") or "", today)].append(row)
    return [(label, buckets[label]) for label in GROUP_ORDER if buckets[label]]


def list_chats(scope: str, fingerprint: str) -> list[dict]:
    """Newest first: chat_id, title, updated_at, turn_count, focus_account."""
    folder = _folder(scope, fingerprint)
    if not folder.exists():
        return []
    rows = []
    for path in folder.glob("*.json"):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "chat_id": chat["chat_id"],
            "title": chat.get("title") or "New chat",
            "updated_at": chat.get("updated_at", ""),
            "turn_count": len(chat.get("turns") or []),
            "focus_account": chat.get("focus_account"),
        })
    return sorted(rows, key=lambda r: r["updated_at"], reverse=True)


def append_turn(chat: dict, role: str, text: str, **extra) -> dict:
    """Add a turn and title the chat from its first question."""
    turn = {"ts": _now(), "role": role, "text": text, **extra}
    chat.setdefault("turns", []).append(turn)
    if role == "user" and chat.get("title") in (None, "", "New chat"):
        chat["title"] = _title_from(text)
    return turn


def drop_last_turn(chat: dict) -> None:
    """Remove a question whose answer never arrived, so the history never
    carries an unanswered turn."""
    if chat.get("turns"):
        chat["turns"].pop()


def set_focus(chat: dict, account_id: str | None) -> None:
    """Record which account the chat is now about."""
    chat["focus_account"] = account_id or None


def _title_from(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"


def unsummarised_turns(chat: dict) -> list[dict]:
    return (chat.get("turns") or [])[chat.get("summarised_through", 0):]


def apply_summary(chat: dict, summary: str, through: int) -> None:
    """Record that turns[:through] are now represented by `summary`."""
    chat["summary"] = summary
    chat["summarised_through"] = max(0, min(through, len(chat.get("turns") or [])))
