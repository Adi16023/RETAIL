"""
Downloads the third-party datasets used to calibrate the synthetic generator
and to test the pipeline against real ordering behaviour.

Currently one source:

  UCI Online Retail II  (CC BY 4.0)
    1,067,371 line items, 5,942 customers, Dec 2009 - Dec 2011, mostly
    wholesalers. Real B2B ordering rhythm — irregular, bursty, half the
    months empty — which is the property the reference workbook lacks and
    the thing a synthetic generator cannot invent convincingly. It carries no
    margin, category, tier or discount, so it calibrates BEHAVIOUR only; the
    economics still come from the workbook.

Everything lands in data/external/, which is gitignored (45 MB). The
workbook is converted once to CSV because re-parsing two Excel sheets takes
longer than every downstream step combined.

    python scripts/fetch_external_data.py            # download if missing
    python scripts/fetch_external_data.py --force    # re-download
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = REPO_ROOT / "data" / "external"

ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ONLINE_RETAIL_ZIP = EXTERNAL / "online_retail_ii.zip"
ONLINE_RETAIL_XLSX = EXTERNAL / "online_retail_II.xlsx"
ONLINE_RETAIL_CSV = EXTERNAL / "online_retail_II.csv"
ONLINE_RETAIL_SHEETS = ("Year 2009-2010", "Year 2010-2011")


def _download(url: str, target: Path) -> None:
    def report(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0:
            print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, target, reporthook=report)
    print()


def fetch_online_retail(force: bool = False) -> Path:
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    if ONLINE_RETAIL_CSV.exists() and not force:
        print(f"Already present: {ONLINE_RETAIL_CSV.relative_to(REPO_ROOT)}")
        return ONLINE_RETAIL_CSV

    if not ONLINE_RETAIL_XLSX.exists() or force:
        print(f"Downloading Online Retail II from {ONLINE_RETAIL_URL}")
        _download(ONLINE_RETAIL_URL, ONLINE_RETAIL_ZIP)
        with zipfile.ZipFile(ONLINE_RETAIL_ZIP) as archive:
            archive.extractall(EXTERNAL)
        ONLINE_RETAIL_ZIP.unlink(missing_ok=True)

    print("Converting workbook to CSV (both year sheets)...")
    frames = [pd.read_excel(ONLINE_RETAIL_XLSX, sheet_name=sheet) for sheet in ONLINE_RETAIL_SHEETS]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(ONLINE_RETAIL_CSV, index=False)
    print(f"Wrote {ONLINE_RETAIL_CSV.relative_to(REPO_ROOT)}  ({len(combined):,} rows)")
    return ONLINE_RETAIL_CSV


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()
    try:
        fetch_online_retail(force=args.force)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("The file is optional: the pipeline and tests run without it; only "
              "scripts/check_real_data.py and generator calibration need it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
