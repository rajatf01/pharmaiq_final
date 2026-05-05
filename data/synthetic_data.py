"""
data/synthetic_data.py

Realistic synthetic pharmacy dataset for PharmaIQ.
Designed for a large retail pharmacy / CVS-style demo.

Creates:
- 350 SKUs
- 3 years of daily sales history
- realistic categories, fast/slow movers, seasonal demand
- batch expiry pressure
- multiple suppliers per SKU
- supplier delays, disruptions, and lead-time variability
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.schema import (
    SKU_ID, PRODUCT_NAME, CATEGORY, DATE, QTY_SOLD, STOCK_LEVEL,
    BATCH_ID, EXPIRY_DATE, UNIT_COST, SELLING_PRICE,
    SUPPLIER_ID, SUPPLIER_NAME, LEAD_TIME_DAYS, LEAD_TIME_STD,
    REORDER_POINT, DAYS_TO_EXPIRY,
)

from config.settings import SYNTHETIC_SEED


BASE_PRODUCTS = [
    ("Amoxicillin 500mg", "Antibiotics", 4.50, 12.00, 18, 6, 365),
    ("Azithromycin 250mg", "Antibiotics", 8.50, 22.00, 10, 4, 730),
    ("Doxycycline 100mg", "Antibiotics", 5.60, 15.00, 11, 4, 730),
    ("Ciprofloxacin 500mg", "Antibiotics", 6.80, 18.50, 9, 3, 730),
    ("Metformin 500mg", "Diabetes", 2.10, 6.00, 34, 10, 730),
    ("Metformin ER 1000mg", "Diabetes", 3.20, 9.00, 20, 6, 730),
    ("Insulin Glargine Pen", "Diabetes", 34.00, 88.00, 7, 3, 180),
    ("Insulin Aspart Pen", "Diabetes", 38.00, 96.00, 6, 2, 180),
    ("Atorvastatin 20mg", "Cardiovascular", 3.80, 10.00, 32, 8, 730),
    ("Rosuvastatin 10mg", "Cardiovascular", 5.50, 14.00, 24, 7, 730),
    ("Amlodipine 5mg", "Cardiovascular", 2.50, 7.00, 30, 8, 730),
    ("Losartan 50mg", "Cardiovascular", 3.20, 9.00, 26, 7, 730),
    ("Lisinopril 10mg", "Cardiovascular", 2.80, 7.50, 25, 6, 730),
    ("Metoprolol 50mg", "Cardiovascular", 2.70, 7.20, 21, 6, 730),
    ("Omeprazole 20mg", "Gastrology", 2.90, 8.50, 28, 8, 365),
    ("Pantoprazole 40mg", "Gastrology", 3.40, 9.00, 20, 6, 365),
    ("Ondansetron 4mg", "Gastrology", 6.80, 18.00, 10, 4, 365),
    ("Paracetamol 500mg", "Analgesics", 1.20, 3.50, 70, 20, 365),
    ("Ibuprofen 400mg", "Analgesics", 1.80, 4.50, 55, 16, 365),
    ("Naproxen 500mg", "Analgesics", 2.40, 6.50, 22, 7, 365),
    ("Cetirizine 10mg", "Allergy", 1.90, 5.50, 34, 12, 730),
    ("Loratadine 10mg", "Allergy", 1.70, 4.80, 28, 10, 730),
    ("Fluticasone Nasal Spray", "Allergy", 8.90, 22.00, 12, 5, 365),
    ("Salbutamol Inhaler", "Respiratory", 12.00, 30.00, 15, 5, 365),
    ("Fluticasone Inhaler", "Respiratory", 18.00, 45.00, 9, 3, 365),
    ("Prednisone 5mg", "Immunology", 3.50, 9.50, 13, 4, 730),
    ("Levothyroxine 50mcg", "Thyroid", 3.60, 9.80, 20, 5, 730),
    ("Sertraline 50mg", "Psychiatry", 6.20, 16.00, 15, 5, 730),
    ("Fluoxetine 20mg", "Psychiatry", 4.90, 13.00, 14, 4, 730),
    ("Gabapentin 300mg", "Neurology", 5.80, 15.50, 18, 6, 730),
    ("Warfarin 5mg", "Anticoagulants", 4.20, 11.00, 9, 3, 730),
    ("Clopidogrel 75mg", "Anticoagulants", 5.10, 13.50, 12, 4, 730),
    ("Furosemide 40mg", "Diuretics", 2.30, 6.00, 15, 5, 730),
    ("Vitamin D3 2000 IU", "OTC Vitamins", 1.00, 6.99, 45, 15, 900),
    ("Multivitamin Adult", "OTC Vitamins", 3.50, 14.99, 38, 12, 900),
    ("Cough Syrup DM", "Cold & Flu", 4.40, 12.99, 22, 8, 365),
    ("Cold Relief Capsules", "Cold & Flu", 3.20, 9.99, 30, 10, 365),
    ("Electrolyte Solution", "Hydration", 2.00, 7.99, 24, 8, 540),
    ("Antacid Chewables", "Digestive OTC", 1.60, 6.49, 26, 9, 730),
    ("Hydrocortisone Cream", "Dermatology", 2.20, 8.99, 14, 5, 730),
]


SUPPLIER_POOL = {
    "SUP001": ("Cardinal Health", 10, 3, 0.12, 1.00),
    "SUP002": ("McKesson Direct", 12, 4, 0.16, 0.98),
    "SUP003": ("AmerisourceBergen", 14, 5, 0.20, 0.96),
    "SUP004": ("QuickPharm Regional", 6, 2, 0.10, 1.08),
    "SUP005": ("Global Generics Ltd", 24, 8, 0.42, 0.84),
    "SUP006": ("Apex Medical Imports", 28, 10, 0.55, 0.80),
    "SUP007": ("Metro Pharma Hub", 16, 6, 0.28, 0.92),
    "SUP008": ("CareLine Direct", 9, 3, 0.14, 1.04),
    "SUP009": ("ValueMed Wholesale", 20, 7, 0.35, 0.88),
    "SUP010": ("Prime Specialty Supply", 18, 5, 0.22, 1.12),
}


def _sku_name(base_name: str, idx: int) -> str:
    variants = [
        "Bottle",
        "Blister",
        "Generic",
        "Brand",
        "Clinic Pack",
        "Retail Pack",
        "90 Count",
    ]

    if idx < len(BASE_PRODUCTS):
        return base_name

    return f"{base_name} - {variants[idx % len(variants)]}"


def _generate_sales(
    sku: str,
    name: str,
    cat: str,
    demand: int,
    std: int,
    days: int,
    rng: np.random.Generator,
    risk_profile: str,
) -> pd.DataFrame:
    end = datetime.today().date()
    start = end - timedelta(days=days - 1)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(days)

    winter_categories = {"Antibiotics", "Respiratory", "Cold & Flu", "Analgesics"}
    spring_categories = {"Allergy"}
    chronic_categories = {"Diabetes", "Cardiovascular", "Thyroid", "Psychiatry"}

    if cat in winter_categories:
        seasonal = 1 + 0.38 * np.sin(2 * np.pi * (t - 285) / 365)
    elif cat in spring_categories:
        seasonal = 1 + 0.45 * np.sin(2 * np.pi * (t - 95) / 365)
    elif cat in chronic_categories:
        seasonal = 1 + 0.08 * np.sin(2 * np.pi * (t - 30) / 365)
    else:
        seasonal = 1 + 0.15 * np.sin(2 * np.pi * (t - 180) / 365)

    weekday_boost = np.array([
        1.10 if d.weekday() in [0, 1]
        else 0.92 if d.weekday() == 6
        else 1.0
        for d in dates
    ])

    trend = 1 + rng.uniform(-0.08, 0.22) * (t / max(1, days))
    promo_spikes = np.ones(days)

    for _ in range(int(rng.integers(3, 9))):
        center = int(rng.integers(0, days))
        width = int(rng.integers(4, 18))
        lift = float(rng.uniform(1.15, 1.75))
        promo_spikes += (lift - 1) * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    mean = demand * seasonal * weekday_boost * trend * promo_spikes
    raw = rng.normal(mean, std, size=days)
    qty = np.maximum(0, np.round(raw)).astype(int)

    zero_mask = rng.random(days) < float(rng.uniform(0.01, 0.05))
    qty[zero_mask] = 0

    stock = np.zeros(days, dtype=int)
    stock[0] = int(demand * rng.uniform(55, 130))

    for i in range(1, days):
        sold = min(qty[i], stock[i - 1])

        reorder_trigger = demand * rng.uniform(9, 18)
        reorder_amount = int(demand * rng.uniform(25, 55))

        if risk_profile == "stockout_prone":
            reorder_trigger *= 0.65
            reorder_amount = int(reorder_amount * 0.75)

        if risk_profile == "overstocked":
            reorder_trigger *= 1.5
            reorder_amount = int(reorder_amount * 1.5)

        disruption = rng.random() < 0.015
        restock = 0 if disruption else (reorder_amount if stock[i - 1] < reorder_trigger else 0)

        stock[i] = max(0, stock[i - 1] - sold + restock)

    return pd.DataFrame({
        SKU_ID: sku,
        PRODUCT_NAME: name,
        CATEGORY: cat,
        DATE: dates,
        QTY_SOLD: qty,
        STOCK_LEVEL: stock,
    })


def _generate_batches(
    sku: str,
    cost: float,
    price: float,
    demand: int,
    shelf: int,
    rng: np.random.Generator,
    risk_profile: str,
) -> pd.DataFrame:
    today = datetime.today().date()
    rows: list[dict[str, object]] = []

    n_batches = int(rng.integers(3, 9))

    if risk_profile == "expiry_prone":
        probs = [0.18, 0.34, 0.30, 0.18]
    elif risk_profile == "overstocked":
        probs = [0.10, 0.28, 0.42, 0.20]
    else:
        probs = [0.04, 0.16, 0.58, 0.22]

    for i in range(n_batches):
        bucket = rng.choice(["expired", "critical", "warning", "healthy"], p=probs)

        if bucket == "expired":
            d = int(rng.integers(-120, -1))
        elif bucket == "critical":
            d = int(rng.integers(1, 30))
        elif bucket == "warning":
            d = int(rng.integers(30, 90))
        else:
            d = int(rng.integers(90, max(120, shelf)))

        stock_multiplier = float(rng.uniform(8, 28))

        if risk_profile == "overstocked":
            stock_multiplier *= float(rng.uniform(1.5, 2.4))

        if risk_profile == "stockout_prone":
            stock_multiplier *= float(rng.uniform(0.4, 0.8))

        rows.append({
            SKU_ID: sku,
            BATCH_ID: f"{sku}_B{i+1:03d}",
            EXPIRY_DATE: pd.Timestamp(today + timedelta(days=d)),
            STOCK_LEVEL: max(0, int(rng.normal(demand * stock_multiplier, max(1, demand * 3)))),
            UNIT_COST: round(cost, 2),
            SELLING_PRICE: round(price, 2),
        })

    return pd.DataFrame(rows)


def _generate_supplier_history(
    sku: str,
    base_cost: float,
    rng: np.random.Generator,
    supplier_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for sid in supplier_ids:
        sname, lt, lstd, delay_prob, cost_mult = SUPPLIER_POOL[sid]
        n_orders = int(rng.integers(24, 72))
        supplier_cost = round(base_cost * cost_mult * float(rng.uniform(0.96, 1.07)), 2)

        for i in range(n_orders):
            seasonal_delay = int(rng.choice([0, 0, 0, 2, 5, 9], p=[0.55, 0.18, 0.10, 0.08, 0.06, 0.03]))
            is_delayed = rng.random() < delay_prob
            extra_delay = int(rng.integers(2, 14)) if is_delayed else 0
            actual = max(1, int(rng.normal(lt, lstd) + seasonal_delay + extra_delay))

            rows.append({
                SKU_ID: sku,
                SUPPLIER_ID: sid,
                SUPPLIER_NAME: sname,
                LEAD_TIME_DAYS: lt,
                LEAD_TIME_STD: lstd,
                "actual_lead_time": actual,
                "order_index": i + 1,
                UNIT_COST: supplier_cost,
            })

    return pd.DataFrame(rows)


def generate_synthetic_dataset(
    n_skus: int = 350,
    history_days: int = 1095,
    seed: int = SYNTHETIC_SEED,
) -> dict[str, pd.DataFrame]:
    print(f"USING NEW CVS-STYLE SYNTHETIC DATA: {n_skus} SKUs, {history_days} days")

    rng = np.random.default_rng(seed)

    sales_l: list[pd.DataFrame] = []
    batches_l: list[pd.DataFrame] = []
    suppliers_l: list[pd.DataFrame] = []
    products_l: list[dict[str, object]] = []

    supplier_keys = list(SUPPLIER_POOL.keys())

    for idx in range(n_skus):
        base_name, cat, cost, price, base_demand, base_std, shelf = BASE_PRODUCTS[idx % len(BASE_PRODUCTS)]

        sku = f"SKU{idx+1:04d}"
        name = _sku_name(base_name, idx)

        demand = max(1, int(base_demand * rng.lognormal(mean=0.0, sigma=0.45)))
        std = max(1, int(base_std * rng.uniform(0.8, 1.8)))

        cost = round(cost * float(rng.uniform(0.82, 1.35)), 2)
        margin = float(rng.uniform(1.8, 3.4))
        price = round(max(price, cost * margin), 2)

        risk_profile = str(rng.choice(
            ["normal", "expiry_prone", "stockout_prone", "overstocked"],
            p=[0.58, 0.17, 0.16, 0.09],
        ))

        if cat in {"Diabetes", "Cardiovascular", "Thyroid"}:
            demand = int(demand * rng.uniform(1.15, 1.75))

        if cat in {"Cold & Flu", "Allergy", "Respiratory"}:
            std = int(std * rng.uniform(1.3, 2.0))

        n_supplier_options = int(rng.integers(2, 6))
        chosen_suppliers = list(rng.choice(supplier_keys, size=n_supplier_options, replace=False))

        primary_supplier = chosen_suppliers[0]
        primary_name, primary_lt, primary_std, _, _ = SUPPLIER_POOL[primary_supplier]

        sales_l.append(_generate_sales(sku, name, cat, demand, std, history_days, rng, risk_profile))
        batches_l.append(_generate_batches(sku, cost, price, demand, shelf, rng, risk_profile))
        suppliers_l.append(_generate_supplier_history(sku, cost, rng, chosen_suppliers))

        products_l.append({
            SKU_ID: sku,
            PRODUCT_NAME: name,
            CATEGORY: cat,
            UNIT_COST: cost,
            SELLING_PRICE: price,
            REORDER_POINT: int(demand * primary_lt * rng.uniform(1.2, 2.4)),
            LEAD_TIME_DAYS: primary_lt,
            LEAD_TIME_STD: primary_std,
            SUPPLIER_ID: primary_supplier,
            SUPPLIER_NAME: primary_name,
        })

    sales_df = pd.concat(sales_l, ignore_index=True)
    batches_df = pd.concat(batches_l, ignore_index=True)
    suppliers_df = pd.concat(suppliers_l, ignore_index=True)
    products_df = pd.DataFrame(products_l)

    today = pd.Timestamp(datetime.today().date())

    batches_df[EXPIRY_DATE] = pd.to_datetime(batches_df[EXPIRY_DATE])
    batches_df[DAYS_TO_EXPIRY] = (batches_df[EXPIRY_DATE] - today).dt.days
    sales_df[DATE] = pd.to_datetime(sales_df[DATE])

    return {
        "sales": sales_df,
        "batches": batches_df,
        "suppliers": suppliers_df,
        "products": products_df,
    }


def save_sample_csvs(output_dir: str = "sample_data") -> None:
    os.makedirs(output_dir, exist_ok=True)

    data = generate_synthetic_dataset()

    for name, df in data.items():
        path = f"{output_dir}/{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path} ({len(df):,} rows)")


if __name__ == "__main__":
    save_sample_csvs()