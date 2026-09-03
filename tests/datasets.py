"""
Test datasets.

Everything is derived from the official workbook extract in data/meridian/.
There is deliberately no second, home-grown dataset: the pipeline's
thresholds are calibrated against the workbook, so a test fixture that
disagrees with it would be testing a different product.

What the workbook cannot supply on its own is a THIN input file. Its
transactions tab is the richest input the pipeline will ever see (17
columns), and the event's Reality Test may well hand over something far
narrower. `thin_transactions` builds those cases by removing columns from
the real data, which keeps the degraded-path tests honest — same accounts,
same figures, same expected behaviour, just less to work with.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "meridian"
MERIDIAN_CSV = DATA_DIR / "transactions.csv"
ANSWER_KEY_JSON = DATA_DIR / "answer_key.json"

# Accounts referenced by name across the suite, chosen for what each proves.
FLAGSHIP_LEAK = "ACC-101"        # flat revenue, margin and tier mix collapse
DEFECTED_ACCOUNT = "ACC-106"     # Diagnostic Equipment stops dead at month 14
DEFECTED_CATEGORY = "Diagnostic Equipment"
HEALTHY_CONTROL = "ACC-102"      # nothing happening, must stay that way
THIN_HISTORY = "ACC-109"         # 5 months, must defer

# The columns a minimal transaction file is guaranteed to carry. Anything
# outside this list is enrichment that ingestion must treat as optional.
THIN_COLUMNS = [
    "order_id", "account_id", "order_date", "product",
    "category", "unit_net_price", "quantity", "line_revenue", "line_margin",
]


def meridian_transactions() -> pd.DataFrame:
    """The full 17-column workbook extract, as read off disk."""
    return pd.read_csv(MERIDIAN_CSV)


def thin_transactions(with_margin: bool = True) -> pd.DataFrame:
    """The same transactions with every enrichment column stripped.

    Drops tier, list_price, discount_pct, unit_cost, is_return and the
    account descriptors — so margin erosion, discount creep, tier mix and
    the returns handler all have nothing to work from. With
    `with_margin=False` even margin goes, leaving a revenue-only file.

    Column names are left in the workbook's own vocabulary (order_date,
    product, line_revenue) rather than the canonical ones, so these fixtures
    also keep exercising the synonym mapping rather than handing ingestion a
    pre-solved problem.
    """
    columns = list(THIN_COLUMNS)
    if not with_margin:
        columns.remove("line_margin")
    return meridian_transactions()[columns].copy()
