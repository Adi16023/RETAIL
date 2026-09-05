"""
Measures the distributions the synthetic generator is calibrated to and writes
them to data/calibration/generator.json.

  behaviour   from data/external/online_retail_II.csv  (real B2B ordering)
  economics   from data/meridian/                       (workbook catalogue, pricing)

The JSON is small and committed, so generation is reproducible on a fresh
clone without the 45 MB download. Re-run this whenever either source changes.

    python scripts/calibrate_generator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.calibration import CALIBRATION_PATH, REAL_CSV, Calibration, measure_behaviour, measure_economics


def main() -> int:
    calibration = Calibration()

    if REAL_CSV.exists():
        print(f"Measuring behaviour from {REAL_CSV.relative_to(REPO_ROOT)} ...")
        calibration.behaviour = measure_behaviour(REAL_CSV)
    else:
        print(f"{REAL_CSV.relative_to(REPO_ROOT)} not found — keeping built-in behaviour defaults.")
        print("Run scripts/fetch_external_data.py to measure from real data.")

    print("Measuring economics from data/meridian ...")
    calibration.economics = measure_economics()

    calibration.save(CALIBRATION_PATH)
    print(f"Wrote {CALIBRATION_PATH.relative_to(REPO_ROOT)}")

    b, e = calibration.behaviour, calibration.economics
    print("\nbehaviour (real):")
    print(f"  active months share   median {b.active_month_share.q50:.2f}  (IQR {b.active_month_share.q25:.2f}-{b.active_month_share.q75:.2f})")
    print(f"  orders / active month median {b.orders_per_active_month.q50:.2f}")
    print(f"  lines / order         median {b.lines_per_order.q50:.1f}")
    print(f"  units / line          median {b.units_per_line.q50:.1f}")
    print(f"  history months        median {b.history_months.q50:.0f}")
    print(f"  return line rate      {b.return_line_rate*100:.1f}%")
    print(f"  monthly revenue CV    median {b.monthly_revenue_cv.q50:.2f}")
    print("economics (workbook):")
    print(f"  products {len(e.products)}  |  discount median {e.discount_pct.q50*100:.1f}%  |  "
          f"high-tier share median {e.high_tier_revenue_share.q50:.2f}  |  lines/order median {e.lines_per_order.q50:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
