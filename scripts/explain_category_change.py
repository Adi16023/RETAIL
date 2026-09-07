"""
Rebuild one category's change-point figures by hand, step by step.

A plain-pandas walk-through of what the pipeline does for a sentence like
"Power Tools revenue dropped 68.9%, from Rs 88,810 to Rs 27,661 a month,
since April 2026, with no recovery in five months; Rs 61,149 a month at
risk". It reads the same transactions.csv, prints every intermediate table,
and ends by checking its answer against the real pipeline.

    venv/Scripts/python scripts/explain_category_change.py
    venv/Scripts/python scripts/explain_category_change.py --account ACC-106 --category Beverages
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The same constants the pipeline uses (src/pipeline/changepoint.py).
WINSOR_CAP_MULTIPLE = 1.5     # cap a month at 1.5x the median non-zero month
ROLLING_WINDOW = 3            # smooth with a trailing 3-month sum
MIN_SEGMENT_MONTHS = 5        # at least 5 months on each side of a split
DECLINE_THRESHOLD = 0.35      # a drop below 35% is not a change-point
RECOVERY_FRACTION = 0.75      # back to 75% of the old level = recovered


def show(title: str, series: pd.Series) -> None:
    print(f"\n{title}")
    print(series.round(0).astype(int).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="ACC-101")
    parser.add_argument("--category", default="Power Tools")
    parser.add_argument("--csv", default=str(REPO / "data" / "meridian" / "transactions.csv"))
    args = parser.parse_args()

    # Step 1 — the account's lines, one row per product per order.
    df = pd.read_csv(args.csv, parse_dates=["order_date"])
    account = df[df["account_id"] == args.account]
    lines = account[account["category"] == args.category]
    print(f"{args.account} / {args.category}: {len(lines)} lines across "
          f"{lines['order_id'].nunique()} orders")

    # Step 2 — sum line_revenue per calendar month; every month of the
    # account's history is kept, a month with no lines is zero.
    months = pd.period_range(account["order_date"].min(), account["order_date"].max(), freq="M")
    monthly = (lines.groupby(lines["order_date"].dt.to_period("M"))["line_revenue"]
               .sum().reindex(months, fill_value=0.0))
    show("STEP 2  monthly revenue (sum of line_revenue)", monthly)

    # Step 3 — cap outliers so one bulk order cannot fake a change.
    cap = monthly[monthly > 0].median() * WINSOR_CAP_MULTIPLE
    capped = monthly.clip(upper=cap)
    print(f"\nSTEP 3  cap = 1.5 x median non-zero month = {cap:,.0f}; "
          f"{int((capped != monthly).sum())} month(s) capped")

    # Step 4 — trailing 3-month sum.
    smoothed = capped.rolling(ROLLING_WINDOW, min_periods=1).sum()
    show("STEP 4  trailing 3-month sum", smoothed)

    # Step 5 — try every split with 5+ months each side; keep the biggest drop.
    n = len(smoothed)
    candidates = []
    for t in range(MIN_SEGMENT_MONTHS, n - MIN_SEGMENT_MONTHS + 1):
        before_med = smoothed.iloc[:t].median()
        after_med = smoothed.iloc[t:].median()
        if before_med > 0:
            candidates.append((str(smoothed.index[t]), before_med, after_med,
                               (before_med - after_med) / before_med))
    print("\nSTEP 5  drop for every candidate split month")
    for month, before_med, after_med, drop in candidates:
        print(f"  {month}: before {before_med:>10,.0f}  after {after_med:>10,.0f}  drop {drop:6.1%}")
    month, before_med, after_med, drop = max(candidates, key=lambda c: c[3])
    if drop < DECLINE_THRESHOLD:
        print(f"\nBiggest drop is {drop:.1%}, under the {DECLINE_THRESHOLD:.0%} threshold: no change-point.")
        return

    # Step 6 — the medians are 3-month sums: divide by 3 for a monthly rate.
    before_month = before_med / ROLLING_WINDOW
    after_month = after_med / ROLLING_WINDOW
    at_risk = before_month - after_month

    # Step 7 — recovered? sustained? (skip the first 2 post-split points,
    # which the trailing sum still blends with pre-change months)
    tail = smoothed[smoothed.index >= pd.Period(month, "M")]
    recovery_tail = tail.iloc[ROLLING_WINDOW - 1:] if len(tail) > ROLLING_WINDOW - 1 else tail
    recovered = bool(recovery_tail.max() >= before_med * RECOVERY_FRACTION)
    sustained = int((tail <= before_med * (1 - DECLINE_THRESHOLD / 2)).sum())

    print(f"""
RESULT
  change-point month        {month}
  before (3-month sum)      {before_med:,.0f}   -> per month {before_month:,.0f}
  after  (3-month sum)      {after_med:,.0f}   -> per month {after_month:,.0f}
  drop                      {drop:.1%}
  recovered                 {recovered}   (needs a later point >= {before_med * RECOVERY_FRACTION:,.0f})
  sustained months          {sustained}   (points <= {before_med * (1 - DECLINE_THRESHOLD / 2):,.0f})
  revenue at risk / month   {at_risk:,.0f}
  lost to date              {at_risk * sustained:,.0f}
  annualised                {at_risk * 12:,.0f}
""")

    # Cross-check against the real pipeline.
    from pipeline.evidence import build_evidence_pack
    from pipeline.ingest import ingest

    clean, _ = ingest(args.csv)
    pack = build_evidence_pack(clean, args.account)
    real = next((c for c in pack["category_changes"] if c["category"] == args.category), None)
    if real and real.get("change_point_detected"):
        print("PIPELINE says:", real["change_point_month"], f"{real['before_monthly_median']:,.0f}",
              "->", f"{real['after_monthly_median']:,.0f}", f"({real['pct_decline']:.1%})",
              "sustained", real["sustained_months_since_change_point"],
              "recovered", real["recovered"])
    else:
        print("PIPELINE says: no change-point for this category")


if __name__ == "__main__":
    main()
