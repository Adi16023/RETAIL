"""
Backtest for the lifetime value model: fit on the first part of every
account's history, predict the orders and value of the held-out tail, and
print actual against predicted.

This is the pass check for the Lifetime value page. A projection that cannot
reproduce the last six months on the book it was fitted to has no business
projecting the next twenty-four. The same numbers are quoted on the page
("how much to trust it"), from `pipeline.lifetime.backtest`.

    python scripts/check_lifetime_model.py                 # hold out the last 6 months
    python scripts/check_lifetime_model.py --holdout 9

Exit code is non-zero when the book-level error on expected orders exceeds
the tolerance, so it can gate a change the way check_real_data.py does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.ingest import ingest  # noqa: E402
from pipeline.lifetime import backtest  # noqa: E402

TOLERANCE_PCT = 25.0  # book-level error on held-out orders that still passes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", type=int, default=6, help="Months held out at the end of history.")
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "meridian" / "transactions.csv"))
    args = parser.parse_args()

    df, _ = ingest(args.data)
    result = backtest(df, holdout_months=args.holdout)
    if result.get("status") != "ok":
        print("Too few scorable accounts to fit.")
        return 1

    params = result["params"]
    print(f"Fitted on history up to {result['fitted_to']} ({params['accounts_fitted']} accounts); "
          f"holding out {result['holdout_months']} months\n")
    print(f"BG/NBD parameters: r={params['r']:.3f} alpha={params['alpha']:.3f} a={params['a']:.3f} b={params['b']:.3f}\n")

    basis = result["value_basis"]
    table = pd.DataFrame(result["rows"]).rename(columns={
        "account_id": "account", "predicted_orders": "predicted orders", "actual_orders": "actual orders",
        "predicted_value": f"predicted {basis}", "actual_value": f"actual {basis}",
    })
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    print()
    print(f"book-level orders : predicted {result['predicted_orders']:.1f} vs actual {result['actual_orders']}  "
          f"({result['order_error_pct']:.1f}% off)")
    print(f"book-level {basis:7s}: {result['value_error_pct']:.1f}% off (value per month frozen at the training "
          f"level, so this measures the activity projection, not the leak)")
    print(f"mean absolute error per account: {result['mean_abs_order_error']:.2f} orders over {result['holdout_months']} months")
    ok = result["order_error_pct"] <= TOLERANCE_PCT
    print(f"\n{'PASS' if ok else 'FAIL'}: book-level order error {result['order_error_pct']:.1f}% "
          f"({'within' if ok else 'over'} the {TOLERANCE_PCT:.0f}% tolerance)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
