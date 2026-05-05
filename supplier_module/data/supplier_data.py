"""
supplier_module/data/supplier_data.py
--------------------------------------
Supplier data layer:
  - Loads supplier / product / history data from CSV (or generates synthetic)
  - Validates schemas
  - Provides clean DataFrames to the scoring engine
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Column constants ──────────────────────────────────────────────────────────
SUPPLIER_ID         = "supplier_id"
SUPPLIER_NAME       = "supplier_name"
PRODUCT_ID          = "product_id"
PRODUCT_NAME        = "product_name"
CATEGORY            = "category"
UNIT_COST           = "unit_cost_usd"
MIN_ORDER_QTY       = "min_order_qty"
STATED_LEAD_TIME    = "stated_lead_time_days"
ACTUAL_LEAD_TIME    = "actual_lead_time_days"
LEAD_TIME_STD       = "lead_time_std_days"
LEAD_TIME_CV        = "lead_time_cv"
DEFECT_RATE         = "defect_rate"
COMPLIANCE_SCORE    = "compliance_score"
POTENCY_VARIANCE    = "potency_variance_pct"
MIN_SHELF_LIFE      = "min_shelf_life_at_delivery_days"
AVG_SHELF_LIFE      = "avg_shelf_life_at_delivery_days"
TOTAL_ORDERS        = "total_orders_placed"
DELAY_PROBABILITY   = "delay_probability"
AVG_DELAY_DAYS      = "avg_delay_days"
FULFILLMENT_RATE    = "fulfillment_rate"
RETURNS_RATE        = "returns_rate"
BUY_VS_USAGE        = "buy_vs_usage_efficiency"
ORDER_DATE          = "order_date"
PROMISED_DATE       = "promised_date"
DELIVERY_DATE       = "actual_delivery_date"
QTY_ORDERED         = "qty_ordered"
QTY_DELIVERED       = "qty_delivered"
WAS_DELAYED         = "was_delayed"
DELAY_DAYS          = "delay_days"

# ── Synthetic catalogue ───────────────────────────────────────────────────────
PRODUCTS = [
    ("SKU001", "Amoxicillin 500mg",    "Antibiotics",    4.50, 12.00),
    ("SKU002", "Metformin 500mg",      "Diabetes",       2.10,  6.00),
    ("SKU003", "Paracetamol 500mg",    "Analgesics",     1.20,  3.50),
    ("SKU004", "Atorvastatin 20mg",    "Cardiovascular", 3.80, 10.00),
    ("SKU005", "Insulin Glargine 100U","Diabetes",      25.00, 65.00),
    ("SKU006", "Omeprazole 20mg",      "Gastrology",     2.90,  8.50),
    ("SKU007", "Amlodipine 5mg",       "Cardiovascular", 2.50,  7.00),
]

SUPPLIERS = [
    ("SUP001", "MedSupply Pro",    "Pakistan",  94,  True),
    ("SUP002", "PharmaDist Ltd",   "UAE",       97,  True),
    ("SUP003", "GlobalMed Corp",   "India",     88,  True),
    ("SUP004", "QuickPharm",       "Pakistan",  91,  True),
    ("SUP005", "RegionalMeds",     "Pakistan",  85,  False),
]

# (supplier_idx, product_idx, unit_cost, stated_lt, lt_std, defect_rate,
#  min_shelf_life, fulfillment_rate, delay_prob, avg_delay, returns_rate)
SP_MATRIX = [
    # SUP001 × products
    (0, 0, 4.20, 14, 3.1, 0.008, 548, 0.971, 0.38, 4.2, 0.004),
    (0, 1, 2.00, 14, 2.8, 0.006, 600, 0.980, 0.32, 3.8, 0.003),
    (0, 2, 1.20, 14, 3.1, 0.008, 548, 0.971, 0.38, 4.2, 0.004),
    (0, 3, 3.60, 14, 2.9, 0.007, 580, 0.975, 0.35, 4.0, 0.003),
    (0, 5, 2.70, 14, 3.2, 0.009, 520, 0.968, 0.40, 4.5, 0.005),
    # SUP002 × products
    (1, 0, 4.80, 18, 5.2, 0.012, 620, 0.958, 0.29, 5.1, 0.008),
    (1, 1, 2.30, 18, 4.9, 0.010, 640, 0.962, 0.27, 4.9, 0.007),
    (1, 2, 1.50, 18, 5.2, 0.012, 620, 0.958, 0.29, 5.1, 0.008),
    (1, 4,24.00, 18, 4.1, 0.005, 420, 0.985, 0.22, 3.2, 0.002),
    (1, 6, 2.40, 18, 5.0, 0.011, 600, 0.960, 0.28, 5.0, 0.007),
    # SUP003 × products
    (2, 0, 5.20, 21, 7.8, 0.031, 480, 0.917, 0.58, 8.4, 0.021),
    (2, 2, 1.70, 21, 7.8, 0.031, 480, 0.917, 0.58, 8.4, 0.021),
    (2, 3, 4.10, 21, 7.2, 0.028, 490, 0.921, 0.55, 8.0, 0.019),
    (2, 5, 3.10, 21, 7.5, 0.029, 470, 0.915, 0.60, 8.6, 0.022),
    # SUP004 × products
    (3, 1, 2.20,  7, 2.0, 0.015, 560, 0.948, 0.35, 3.0, 0.010),
    (3, 4,26.00,  7, 1.8, 0.008, 400, 0.960, 0.28, 2.5, 0.005),
    (3, 6, 2.60,  7, 2.1, 0.016, 540, 0.945, 0.37, 3.2, 0.011),
    # SUP005 × products
    (4, 0, 4.50, 14, 4.0, 0.020, 500, 0.935, 0.48, 5.8, 0.015),
    (4, 2, 1.35, 14, 4.0, 0.020, 500, 0.935, 0.48, 5.8, 0.015),
]


def generate_synthetic_suppliers() -> pd.DataFrame:
    rows = []
    for sid, name, country, compliance, who in SUPPLIERS:
        rows.append({
            SUPPLIER_ID:      sid,
            SUPPLIER_NAME:    name,
            "country":        country,
            COMPLIANCE_SCORE: compliance,
            "who_certified":  who,
            "active":         True,
        })
    return pd.DataFrame(rows)


def generate_synthetic_products() -> pd.DataFrame:
    rows = []
    for pid, name, cat, cost, price in PRODUCTS:
        rows.append({
            PRODUCT_ID:   pid,
            PRODUCT_NAME: name,
            CATEGORY:     cat,
            "unit_cost":  cost,
            "sell_price": price,
        })
    return pd.DataFrame(rows)


def generate_synthetic_supplier_products() -> pd.DataFrame:
    """One row per supplier-product relationship — the core comparison table."""
    sup_df  = generate_synthetic_suppliers()
    prod_df = generate_synthetic_products()
    rows    = []

    for (si, pi, cost, lt, lt_std, defect, shelf,
         fulfill, delay_p, avg_delay, returns) in SP_MATRIX:

        sup  = sup_df.iloc[si]
        prod = prod_df.iloc[pi]

        eff_lt  = round(lt + 1.645 * lt_std, 1)
        lt_cv   = round(lt_std / max(lt, 1), 3)
        potency = round(1.0 + defect * 50, 1)          # rough proxy
        avg_shelf = round(shelf * 1.12, 0)

        rows.append({
            "sp_id":            f"{sup[SUPPLIER_ID]}_{prod[PRODUCT_ID]}",
            SUPPLIER_ID:        sup[SUPPLIER_ID],
            SUPPLIER_NAME:      sup[SUPPLIER_NAME],
            PRODUCT_ID:         prod[PRODUCT_ID],
            PRODUCT_NAME:       prod[PRODUCT_NAME],
            CATEGORY:           prod[CATEGORY],
            UNIT_COST:          cost,
            MIN_ORDER_QTY:      500 if lt <= 14 else (1000 if lt <= 18 else 2000),
            STATED_LEAD_TIME:   lt,
            LEAD_TIME_STD:      lt_std,
            LEAD_TIME_CV:       lt_cv,
            "effective_lead_time_days": eff_lt,
            DEFECT_RATE:        defect,
            COMPLIANCE_SCORE:   sup[COMPLIANCE_SCORE],
            POTENCY_VARIANCE:   potency,
            "cold_chain_certified": True,
            MIN_SHELF_LIFE:     shelf,
            AVG_SHELF_LIFE:     avg_shelf,
            TOTAL_ORDERS:       np.random.randint(12, 40),
            DELAY_PROBABILITY:  delay_p,
            AVG_DELAY_DAYS:     avg_delay,
            FULFILLMENT_RATE:   fulfill,
            RETURNS_RATE:       returns,
            BUY_VS_USAGE:       round(fulfill * (1 - defect * 5), 3),
        })

    return pd.DataFrame(rows)


def generate_delivery_history(n_records: int = 200,
                               seed: int = 42) -> pd.DataFrame:
    """Simulated historical delivery records."""
    rng  = np.random.default_rng(seed)
    sp_df = generate_synthetic_supplier_products()
    rows  = []
    today = datetime.today().date()

    for _ in range(n_records):
        sp = sp_df.sample(1, random_state=int(rng.integers(0, 9999))).iloc[0]
        lt = int(sp[STATED_LEAD_TIME])
        order_date    = today - timedelta(days=int(rng.integers(10, 730)))
        promised_date = order_date + timedelta(days=lt)
        actual_days   = max(1, int(rng.normal(lt, sp[LEAD_TIME_STD])))
        delivery_date = order_date + timedelta(days=actual_days)
        qty_ordered   = int(rng.integers(500, 2001))
        qty_delivered = qty_ordered if rng.random() < sp[FULFILLMENT_RATE] else \
                        int(qty_ordered * rng.uniform(0.85, 0.99))
        defects       = int(qty_delivered * rng.uniform(0, sp[DEFECT_RATE] * 2))

        rows.append({
            SUPPLIER_ID:     sp[SUPPLIER_ID],
            SUPPLIER_NAME:   sp[SUPPLIER_NAME],
            PRODUCT_ID:      sp[PRODUCT_ID],
            PRODUCT_NAME:    sp[PRODUCT_NAME],
            ORDER_DATE:      order_date,
            PROMISED_DATE:   promised_date,
            DELIVERY_DATE:   delivery_date,
            QTY_ORDERED:     qty_ordered,
            QTY_DELIVERED:   qty_delivered,
            WAS_DELAYED:     delivery_date > promised_date,
            DELAY_DAYS:      max(0, (delivery_date - promised_date).days),
            "defects":       defects,
            "defect_rate_order": round(defects / max(qty_delivered, 1), 4),
            "unit_cost_paid": sp[UNIT_COST],
        })

    return pd.DataFrame(rows)


def validate_sp_dataframe(df: pd.DataFrame) -> None:
    required = [SUPPLIER_ID, PRODUCT_ID, UNIT_COST, STATED_LEAD_TIME,
                DEFECT_RATE, FULFILLMENT_RATE, DELAY_PROBABILITY]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"supplier_products missing columns: {missing}")
    if df.empty:
        raise ValueError("supplier_products DataFrame is empty")


def save_all_csvs(output_dir: str = "sample_data") -> None:
    os.makedirs(output_dir, exist_ok=True)
    generate_synthetic_suppliers().to_csv(
        f"{output_dir}/suppliers.csv", index=False)
    generate_synthetic_products().to_csv(
        f"{output_dir}/products.csv", index=False)
    generate_synthetic_supplier_products().to_csv(
        f"{output_dir}/supplier_products.csv", index=False)
    generate_delivery_history().to_csv(
        f"{output_dir}/delivery_history.csv", index=False)
    print(f"Saved 4 CSV files to {output_dir}/")


if __name__ == "__main__":
    save_all_csvs("sample_data")
    sp = generate_synthetic_supplier_products()
    print(f"\nSupplier-product pairs: {len(sp)}")
    print(sp[[SUPPLIER_NAME, PRODUCT_NAME, UNIT_COST,
              STATED_LEAD_TIME, DEFECT_RATE, FULFILLMENT_RATE]].to_string(index=False))
