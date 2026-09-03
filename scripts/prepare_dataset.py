"""
Extracts the official Quessathon workbook into the flat CSV/JSON files the
pipeline and tests consume.

`Quessathon_Revenue_Leakage_Dataset.xlsx` ships five data tabs plus a READ
ME. Only ONE of them is ever an agent input:

  Transactions     -> data/meridian/transactions.csv     THE agent input
  Accounts         -> data/meridian/accounts.csv         account dimension
  Products         -> data/meridian/products.csv         product dimension
  Monthly Summary  -> data/meridian/monthly_summary.csv  derived convenience
  Answer Key       -> data/meridian/answer_key.json      GROUND TRUTH — TESTS ONLY

The workbook's own READ ME is explicit on the last two: the monthly summary
is a "derived convenience, not an agent input", and the answer key must
never be fed to the agent. Both are extracted anyway because the test suite
needs them — `answer_key.json` drives tests/test_answer_key.py, and
`monthly_summary.csv` is used only to cross-check our own roll-ups against
the organisers' (see tests/test_meridian_dataset.py). Nothing under
src/pipeline/ ever reads either file.

Run from the repo root:  python scripts/prepare_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "Quessathon_Revenue_Leakage_Dataset.xlsx"
OUT_DIR = REPO_ROOT / "data" / "meridian"

# sheet name -> output file. Order matters only for the printed summary.
SHEET_FILES = {
    "Transactions": "transactions.csv",
    "Accounts": "accounts.csv",
    "Products": "products.csv",
    "Monthly Summary": "monthly_summary.csv",
}

ANSWER_KEY_SHEET = "Answer Key"
ANSWER_KEY_FILE = "answer_key.json"
README_SHEET = "READ ME"
README_FILE = "dataset_readme.txt"


def _read_sheet(sheet: str) -> pd.DataFrame:
    return pd.read_excel(WORKBOOK, sheet_name=sheet)


def extract_readme() -> str:
    """The READ ME tab is a two-column label/value sheet with blank spacer
    rows. Flattened to plain text so it is readable without Excel."""
    df = _read_sheet(README_SHEET)
    lines = []
    for label, value in df.itertuples(index=False):
        label = "" if pd.isna(label) else str(label).strip()
        value = "" if pd.isna(value) else str(value).strip()
        if not label and not value:
            lines.append("")
        elif label and value:
            lines.append(f"{label}: {value}")
        else:
            lines.append(label or value)
    return "\n".join(lines).strip() + "\n"


def extract_answer_key() -> list[dict]:
    df = _read_sheet(ANSWER_KEY_SHEET)
    return json.loads(df.to_json(orient="records"))


def main() -> None:
    if not WORKBOOK.exists():
        raise SystemExit(f"Workbook not found: {WORKBOOK}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sheet, filename in SHEET_FILES.items():
        df = _read_sheet(sheet)
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"Wrote {path.relative_to(REPO_ROOT)}  ({len(df)} rows, {len(df.columns)} cols)")

    answer_key = extract_answer_key()
    key_path = OUT_DIR / ANSWER_KEY_FILE
    key_path.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")
    print(f"Wrote {key_path.relative_to(REPO_ROOT)}  ({len(answer_key)} accounts) — TESTS ONLY, never an agent input")

    readme_path = OUT_DIR / README_FILE
    readme_path.write_text(extract_readme(), encoding="utf-8")
    print(f"Wrote {readme_path.relative_to(REPO_ROOT)}")

    transactions = pd.read_csv(OUT_DIR / SHEET_FILES["Transactions"])
    print(
        f"\nTransactions: {len(transactions)} lines, "
        f"{transactions['order_id'].nunique()} orders, "
        f"{transactions['account_id'].nunique()} accounts, "
        f"{transactions['order_date'].min()} .. {transactions['order_date'].max()}"
    )


if __name__ == "__main__":
    main()
