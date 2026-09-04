"""
On-disk cache for completed investigations.

An investigation is one paid API call, and the same account analysed by the
same model over the same data gives the same answer — so re-running it on
every page interaction is pure waste. Results are keyed by the three things
that actually determine the answer:

  account  +  model  +  a fingerprint of that account's transaction rows

The fingerprint is the part that is easy to get wrong. Keying on account and
model alone would happily serve a verdict computed from a file the user has
since replaced, and the numbers in the verdict would silently disagree with
the charts above it. Hashing the rows means changed data misses the cache and
re-runs, while editing a DIFFERENT account leaves this one's verdict valid.

Stored on disk rather than in session state so a browser refresh does not
throw away work already paid for. The directory is gitignored; deleting it
costs nothing but a re-run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "investigations"


def account_fingerprint(df: pd.DataFrame, account_id: str) -> str:
    """Short, stable hash of one account's transaction rows."""
    rows = df[df["account_id"] == account_id]
    digest = int(pd.util.hash_pandas_object(rows, index=False).sum())
    return hashlib.sha1(str(digest).encode()).hexdigest()[:12]


def _path(fingerprint: str, account_id: str, model: str) -> Path:
    safe_model = "".join(c if c.isalnum() else "-" for c in model)
    return CACHE_DIR / f"{account_id}_{safe_model}_{fingerprint}.json"


def load(fingerprint: str, account_id: str, model: str) -> dict | None:
    path = _path(fingerprint, account_id, model)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt cache file must never break the page — treat it as a miss
        # and let the investigation re-run.
        return None


def save(fingerprint: str, account_id: str, model: str, report: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _path(fingerprint, account_id, model).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def clear(fingerprint: str, account_id: str, model: str) -> None:
    _path(fingerprint, account_id, model).unlink(missing_ok=True)


def count() -> int:
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
