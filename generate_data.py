"""
Synthetic B2B retail transaction generator.

Produces:
  <out-dir>/transactions.csv   - the only file the detection pipeline ever sees
  <out-dir>/ground_truth.json  - archetype labels for internal testing only, never fed to the pipeline

Spec: see synthetic_data_prompt.md and data.md.
"""

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

CATEGORY_CATALOG = {
    "Industrial Equipment": {"price_range": (6000, 16000), "margin_pct": 0.34, "tier": "Premium"},
    "Precision Tools":      {"price_range": (1500, 4500),  "margin_pct": 0.28, "tier": "Premium"},
    "Chemical Supplies":    {"price_range": (400, 1200),   "margin_pct": 0.22, "tier": "Standard"},
    "Safety Gear":          {"price_range": (150, 500),    "margin_pct": 0.18, "tier": "Standard"},
    "Office Supplies":      {"price_range": (40, 150),     "margin_pct": 0.10, "tier": "Commodity"},
    "Packaging Materials":  {"price_range": (20, 90),      "margin_pct": 0.08, "tier": "Commodity"},
    "Cleaning Supplies":    {"price_range": (15, 60),      "margin_pct": 0.09, "tier": "Commodity"},
}

# month (1-12) -> multiplier, for the categories with real seasonality
SEASONAL_CATEGORIES = {
    "Safety Gear": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
                    7: 1.6, 8: 1.6, 9: 1.2, 10: 1.0, 11: 1.0, 12: 1.0},  # summer compliance-audit peak
    "Chemical Supplies": {1: 0.5, 2: 0.5, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
                          7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},  # winter trough
}

ARCHETYPES = [
    "structural_mix_collapse",
    "seasonal_dip_temporary",
    "order_size_shrinkage",
    "one_off_bulk_baseline",
    "healthy",
    "sparse_short_history",
    "wildcard_substitution",
]


def build_product_catalog(rng):
    products = []
    for cat, spec in CATEGORY_CATALOG.items():
        for i in range(3):
            products.append({
                "product_id": f"{''.join(w[0] for w in cat.split())}-{i + 1:02d}",
                "category": cat,
                "base_price": float(rng.uniform(*spec["price_range"])),
                "margin_pct": spec["margin_pct"],
                "tier": spec["tier"],
            })
    return pd.DataFrame(products)


def seasonal_multiplier(category, month):
    return SEASONAL_CATEGORIES.get(category, {}).get(month, 1.0)


def make_order_rows(rng, order_id, account_id, order_date, product_pool, n_items,
                     qty_range=(1, 6), discount_range=(0.0, 0.08)):
    rows = []
    chosen = product_pool.sample(n=min(n_items, len(product_pool)),
                                  random_state=int(rng.integers(0, 1_000_000)))
    for _, p in chosen.iterrows():
        qty = int(rng.integers(*qty_range))
        if qty <= 0:
            qty = 1
        discount = rng.uniform(*discount_range)
        month_mult = seasonal_multiplier(p["category"], order_date.month)
        unit_price = round(p["base_price"] * month_mult * (1 - discount), 2)
        revenue = round(unit_price * qty, 2)
        margin = round(revenue * p["margin_pct"], 2)
        rows.append({
            "order_id": order_id,
            "account_id": account_id,
            "date": order_date.date().isoformat(),
            "product_id": p["product_id"],
            "category": p["category"],
            "unit_price": unit_price,
            "quantity": qty,
            "revenue": revenue,
            "margin": margin,
        })
    return rows


def make_backfill_order_rows(rng, order_id, account_id, order_date, low_pool, target_revenue):
    """Construct an order from low_pool sized to hit ~target_revenue.

    Used to compensate a collapsed high-value category with enough
    low-value volume that total account revenue stays roughly flat — a
    plain item-count bump can't do this given the ~40-100x price gap
    between Premium and Commodity categories, so quantity is solved for
    directly against the revenue target instead.
    """
    if len(low_pool) == 0 or target_revenue <= 0:
        return []
    rows = []
    n_line_items = min(len(low_pool), max(1, int(rng.integers(2, 4))))
    chosen = low_pool.sample(n=n_line_items, random_state=int(rng.integers(0, 1_000_000)))
    revenue_per_item = target_revenue / n_line_items
    for _, p in chosen.iterrows():
        month_mult = seasonal_multiplier(p["category"], order_date.month)
        unit_price = round(p["base_price"] * month_mult, 2)
        qty = max(1, int(round(revenue_per_item / max(unit_price, 1))))
        revenue = round(unit_price * qty, 2)
        margin = round(revenue * p["margin_pct"], 2)
        rows.append({
            "order_id": order_id, "account_id": account_id, "date": order_date.date().isoformat(),
            "product_id": p["product_id"], "category": p["category"],
            "unit_price": unit_price, "quantity": qty, "revenue": revenue, "margin": margin,
        })
    return rows


def generate_account(rng, account_id, archetype, products, current_date, months=20):
    start_date = current_date - pd.DateOffset(months=months)
    rows = []
    order_counter = 1
    ground_truth = {
        "account_id": account_id,
        "archetype": archetype,
        "change_point_date": None,
        "expected_verdict": None,
        "injected_high_value_categories": [],
        "injected_low_value_categories": [],
        "generation_seed": int(rng.bit_generator.state["state"]["state"] % (2**31)),
    }

    base_rate = rng.uniform(8.0, 15.0)  # orders/month
    items_mean = rng.uniform(2.0, 3.0)
    change_point_month = int(months * rng.uniform(0.55, 0.7))
    change_point_date = start_date + pd.DateOffset(months=change_point_month)

    high_val_cats = list(rng.choice(
        [c for c, s in CATEGORY_CATALOG.items() if s["tier"] in ("Premium",)],
        size=2, replace=False))
    low_val_cats = list(rng.choice(
        [c for c, s in CATEGORY_CATALOG.items() if s["tier"] == "Commodity"],
        size=2, replace=False))

    if archetype == "sparse_short_history":
        months = 4
        start_date = current_date - pd.DateOffset(months=months)
        base_rate = rng.uniform(0.5, 1.2)

    hv_revenue_history = []  # pre-change-point high-value revenue, per month — used to size the backfill

    for m in range(months):
        month_date = start_date + pd.DateOffset(months=m)

        rate = base_rate
        item_pool = products
        month_rows = []
        handled = False

        if archetype == "structural_mix_collapse":
            ground_truth["change_point_date"] = change_point_date.date().isoformat()
            ground_truth["expected_verdict"] = "structural"
            ground_truth["injected_high_value_categories"] = high_val_cats
            ground_truth["injected_low_value_categories"] = low_val_cats
            if m >= change_point_month:
                handled = True
                decay = max(0.0, 1 - 0.35 * (m - change_point_month + 1))
                high_pool = products[products["category"].isin(high_val_cats)]
                low_pool = products[products["category"].isin(low_val_cats)]
                other_pool = products[~products["category"].isin(high_val_cats + low_val_cats)]
                n_orders = max(1, int(rng.poisson(rate)))
                for _ in range(n_orders):
                    order_date = month_date + timedelta(days=int(rng.integers(0, 27)))
                    order_id = f"ORD-{account_id}-{order_counter:04d}"
                    order_counter += 1
                    pool = pd.concat([high_pool.sample(frac=decay, random_state=int(rng.integers(0, 1e6))) if decay > 0 and len(high_pool) else high_pool.iloc[0:0],
                                       low_pool, other_pool])
                    n_items = max(1, int(rng.poisson(items_mean)))
                    month_rows += make_order_rows(rng, order_id, account_id, order_date, pool, n_items)

                actual_hv = sum(r["revenue"] for r in month_rows if r["category"] in high_val_cats)
                target_hv = (sum(hv_revenue_history) / len(hv_revenue_history)) * rng.uniform(0.85, 1.05) if hv_revenue_history else 0.0
                deficit = max(0.0, target_hv - actual_hv)
                if deficit > 0 and len(low_pool):
                    n_backfill_orders = int(rng.integers(1, 3))
                    for _ in range(n_backfill_orders):
                        order_id = f"ORD-{account_id}-{order_counter:04d}"
                        order_counter += 1
                        order_date = month_date + timedelta(days=int(rng.integers(0, 27)))
                        month_rows += make_backfill_order_rows(
                            rng, order_id, account_id, order_date, low_pool, deficit / n_backfill_orders)

        elif archetype == "seasonal_dip_temporary":
            ground_truth["expected_verdict"] = "temporary"
            ground_truth["injected_high_value_categories"] = ["Chemical Supplies"]

        elif archetype == "order_size_shrinkage":
            ground_truth["change_point_date"] = change_point_date.date().isoformat()
            ground_truth["expected_verdict"] = "structural"
            if m >= change_point_month:
                items_mean = max(1.2, items_mean - 0.35 * (m - change_point_month + 1))

        elif archetype == "one_off_bulk_baseline":
            ground_truth["expected_verdict"] = "not_leakage"
            if m == 1:
                order_id = f"ORD-{account_id}-{order_counter:04d}"
                order_counter += 1
                order_date = month_date + timedelta(days=5)
                bulk_pool = products[products["category"].isin(high_val_cats)]
                rows += make_order_rows(rng, order_id, account_id, order_date, bulk_pool,
                                         n_items=len(bulk_pool), qty_range=(15, 30))

        elif archetype == "healthy":
            ground_truth["expected_verdict"] = "healthy"

        elif archetype == "wildcard_substitution":
            ground_truth["change_point_date"] = change_point_date.date().isoformat()
            ground_truth["expected_verdict"] = "structural_substitution"
            if m >= change_point_month:
                item_pool = products[~products["category"].isin(high_val_cats[:1])]

        if not handled:
            n_orders = max(0, int(rng.poisson(rate)))
            if archetype == "order_size_shrinkage":
                # This archetype's entire premise is an isolated "frequency
                # stays flat" variable — basket size shrinks, order count
                # doesn't. An unclipped Poisson draw can swing the monthly
                # count far enough by chance (as ACC-0011 did: +57% recent
                # frequency, which flipped total revenue positive and made
                # the account read as healthy instead of structural) to
                # invalidate the archetype for that account. Clip so
                # frequency reliably stays close to its target rate.
                lo, hi = round(rate * 0.75), round(rate * 1.25)
                n_orders = min(max(n_orders, lo), hi)
            for _ in range(n_orders):
                order_date = month_date + timedelta(days=int(rng.integers(0, 27)))
                order_id = f"ORD-{account_id}-{order_counter:04d}"
                order_counter += 1
                n_items = max(1, int(rng.poisson(items_mean)))
                month_rows += make_order_rows(rng, order_id, account_id, order_date, item_pool, n_items)

        if archetype == "structural_mix_collapse" and m < change_point_month:
            hv_revenue_history.append(sum(r["revenue"] for r in month_rows if r["category"] in high_val_cats))

        rows += month_rows

    return rows, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retail transaction data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="synthetic_data")
    parser.add_argument("--accounts-per-archetype", type=int, default=5)
    parser.add_argument("--current-date", type=str, default="2025-12-01")
    parser.add_argument("--no-margin", action="store_true", help="Drop the margin column entirely.")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    products = build_product_catalog(master_rng)
    current_date = pd.Timestamp(args.current_date)

    all_rows = []
    ground_truths = []
    acc_num = 1
    for archetype in ARCHETYPES:
        for _ in range(args.accounts_per_archetype):
            account_id = f"ACC-{acc_num:04d}"
            acc_num += 1
            account_seed = int(master_rng.integers(0, 2**31))
            account_rng = np.random.default_rng(account_seed)
            rows, gt = generate_account(account_rng, account_id, archetype, products, current_date)
            gt["generation_seed"] = account_seed
            all_rows.extend(rows)
            ground_truths.append(gt)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["account_id", "date"]).reset_index(drop=True)

    if args.no_margin:
        df = df.drop(columns=["margin"])

    transactions_path = f"{args.out_dir}/transactions.csv"
    ground_truth_path = f"{args.out_dir}/ground_truth.json"
    df.to_csv(transactions_path, index=False)
    with open(ground_truth_path, "w") as f:
        json.dump(ground_truths, f, indent=2)

    print(f"Wrote {len(df)} line items across {acc_num - 1} accounts to {transactions_path}")
    print(f"Wrote ground truth for {len(ground_truths)} accounts to {ground_truth_path}")
    print("\nCategory catalog:")
    print(products[["category", "tier"]].drop_duplicates().to_string(index=False))
    print("\nArchetypes generated (accounts each):")
    for a in ARCHETYPES:
        print(f"  {a}: {args.accounts_per_archetype}")


if __name__ == "__main__":
    main()
