Supplier: Meridian Supply Co — a B2B distributor selling professional / industrial supplies to retail & reseller accounts across India.
Currency: INR (rupees). All prices, revenue and margin are in Rs.
Period: 24 months, Sep 2024 through Aug 2026. Line-item level.
Scale: 18 accounts, 551 orders, 3,289 transaction lines.

WHY IT IS BUILT THIS WAY
The agent works on raw transaction lines and must roll them up itself (detect -> investigate -> attribute -> prioritise). Nothing is pre-diagnosed.
Only ~6 of 18 accounts are genuine structural leaks. "Always flag" is wrong ~66% of the time; "never flag" misses every real loss. The discrimination is the point.
Several accounts look like leaks but are not (seasonal, recovered, premiumisation, bulk-order, returns, missing month). Several look fine at revenue level but are leaking on margin/mix.
One account (ACC-109) has too little history to call — the agent should DEFER, not guess. That is the highest-weighted behaviour in the rubric (30%).
Realistic mess is intentional and defensible: returns/credits, a bulk-order outlier, a skipped month, a mid-history rep change, monthly noise.

THE DISCLOSED JURY SCENARIO: ACC-101 Northgate Traders is built to BE the scenario the challenge doc describes: total revenue broadly flat, high-value categories abandoned, low-value increased. Structural. Beat it cold; expect the jury to invert it live.

FILES: meridian_transactions.csv (core — the agent input) · meridian_accounts.csv · meridian_products.csv · meridian_monthly_summary.csv (derived convenience, not an agent input)
ANSWER KEY: The "Answer Key" tab is ground truth for YOU to validate and defend the agent. Do NOT feed it to the agent.
COLUMNS (transactions): order_id, order_date, account_id, account_name, region, account_manager, product, category, tier (High/Mid/Low value), quantity (negative = return), list_price, discount_pct, unit_net_price, unit_cost, line_revenue, line_margin, is_return
