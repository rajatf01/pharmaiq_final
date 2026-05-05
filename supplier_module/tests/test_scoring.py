"""
supplier_module/tests/test_scoring.py
--------------------------------------
Unit tests for the supplier scoring engine.
Run: python tests/test_scoring.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data.supplier_data import (
    generate_synthetic_supplier_products,
    generate_delivery_history,
    validate_sp_dataframe,
    SUPPLIER_ID, SUPPLIER_NAME, PRODUCT_ID, UNIT_COST, STATED_LEAD_TIME,
    DEFECT_RATE, FULFILLMENT_RATE, DELAY_PROBABILITY,
)
from models.scoring_engine import (
    compute_risk_score, compute_effective_lead_time,
    compute_mcdm_scores, compute_tradeoff,
    score_all_suppliers, score_product_suppliers,
    RISK_SCORE, MCDM_SCORE, COST_RANK, MCDM_RANK, IS_RECOMMENDED,
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"

errors = []

def check(name, condition, msg=""):
    if condition:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}  →  {msg}")
        errors.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Data generation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST GROUP 1 — Data Generation")
print("="*60)

sp_df = generate_synthetic_supplier_products()
check("sp_df is not empty",              not sp_df.empty,              f"len={len(sp_df)}")
check("Has required columns",            UNIT_COST in sp_df.columns,   "unit_cost_usd missing")
check("Has supplier_id column",          SUPPLIER_ID in sp_df.columns, "supplier_id missing")
check("Has product_id column",           PRODUCT_ID in sp_df.columns,  "product_id missing")
check("Multiple products present",       sp_df[PRODUCT_ID].nunique() > 1,
      f"only {sp_df[PRODUCT_ID].nunique()} products")
check("Multiple suppliers present",      sp_df[SUPPLIER_ID].nunique() > 1,
      f"only {sp_df[SUPPLIER_ID].nunique()} suppliers")
check("Unit costs are positive",         (sp_df[UNIT_COST] > 0).all(),
      "some costs ≤ 0")
check("Lead times are positive",         (sp_df[STATED_LEAD_TIME] > 0).all(),
      "some lead times ≤ 0")
check("Defect rates in [0,1]",           ((sp_df[DEFECT_RATE] >= 0) & (sp_df[DEFECT_RATE] <= 1)).all(),
      "defect rates out of range")
check("Fulfillment rates in [0,1]",      ((sp_df[FULFILLMENT_RATE] >= 0) & (sp_df[FULFILLMENT_RATE] <= 1)).all(),
      "fulfillment rates out of range")

hist_df = generate_delivery_history(n_records=100)
check("Delivery history generated",      len(hist_df) == 100, f"got {len(hist_df)}")
check("History has was_delayed column",  "was_delayed" in hist_df.columns)
check("Delay days non-negative",         (hist_df["delay_days"] >= 0).all())

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Risk scoring
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST GROUP 2 — Risk Scoring")
print("="*60)

# Low-risk supplier
low_risk = pd.Series({
    DELAY_PROBABILITY: 0.10,
    "lead_time_cv":    0.10,
    "avg_delay_days":  1.0,
    DEFECT_RATE:       0.002,
    "total_orders_placed": 30,
})
lr_score = compute_risk_score(low_risk)
check("Low-risk supplier scores < 0.35",  lr_score < 0.35,  f"got {lr_score}")

# High-risk supplier
high_risk = pd.Series({
    DELAY_PROBABILITY: 0.70,
    "lead_time_cv":    0.50,
    "avg_delay_days":  12.0,
    DEFECT_RATE:       0.04,
    "total_orders_placed": 30,
})
hr_score = compute_risk_score(high_risk)
check("High-risk supplier scores ≥ 0.60", hr_score >= 0.60, f"got {hr_score}")

# Sparse data confidence shrinkage
sparse = pd.Series({
    DELAY_PROBABILITY: 0.80,
    "lead_time_cv":    0.60,
    "avg_delay_days":  14.0,
    DEFECT_RATE:       0.05,
    "total_orders_placed": 3,    # very sparse
})
sparse_score = compute_risk_score(sparse)
dense  = sparse.copy(); dense["total_orders_placed"] = 30
dense_score  = compute_risk_score(dense)
check("Sparse data shrinks score toward 0.20",
      sparse_score < dense_score, f"sparse={sparse_score}, dense={dense_score}")

# Bounds
for row_data in [low_risk, high_risk, sparse]:
    s = compute_risk_score(row_data)
    check(f"Risk score in [0,1] for score={s:.2f}", 0.0 <= s <= 1.0, f"got {s}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Effective lead time
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST GROUP 3 — Effective Lead Time")
print("="*60)

row_elt = pd.Series({STATED_LEAD_TIME: 14, "lead_time_std_days": 4.0})
elt     = compute_effective_lead_time(row_elt)
expected = 14 + 1.645 * 4.0
check("ELT formula correct",         abs(elt - expected) < 0.1, f"got {elt}, expected {expected:.2f}")
check("ELT > stated lead time",      elt > 14,                   f"elt={elt}")
check("ELT is float",                isinstance(elt, float),     f"type={type(elt)}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: MCDM Scoring
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST GROUP 4 — MCDM Scoring")
print("="*60)

product_group = sp_df[sp_df[PRODUCT_ID] == "SKU003"].copy()
check("SKU003 has suppliers",         len(product_group) > 0, "no SKU003 suppliers")

if not product_group.empty:
    mcdm = compute_mcdm_scores(product_group)
    check("MCDM returns correct length", len(mcdm) == len(product_group), f"got {len(mcdm)}")
    check("MCDM scores in [0,100]",      ((mcdm >= 0) & (mcdm <= 100)).all(),
          f"min={mcdm.min()}, max={mcdm.max()}")
    check("MCDM scores vary",            mcdm.std() > 0.01, f"std={mcdm.std():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Trade-off Calculator
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST GROUP 5 — Trade-Off Calculator")
print("="*60)

sup_cheap = pd.Series({
    SUPPLIER_NAME:      "CheapSupplier",
    UNIT_COST:          1.00,
    DELAY_PROBABILITY:  0.50,
    "avg_delay_days":   8.0,
    DEFECT_RATE:        0.025,
    FULFILLMENT_RATE:   0.90,
})
sup_premium = pd.Series({
    SUPPLIER_NAME:      "PremiumSupplier",
    UNIT_COST:          1.50,
    DELAY_PROBABILITY:  0.15,
    "avg_delay_days":   2.0,
    DEFECT_RATE:        0.005,
    FULFILLMENT_RATE:   0.98,
})

tf = compute_tradeoff(sup_cheap, sup_premium, order_qty=1000)

check("Trade-off returns dict",              isinstance(tf, dict))
check("Has cost_saving key",                 "cost_saving" in tf)
check("Has net_benefit key",                 "net_benefit" in tf)
check("Has recommended key",                 "recommended" in tf)
check("Has text key",                        "text" in tf)
check("Cost saving is positive (cheap < premium)",
      tf["cost_saving"] > 0, f"got {tf['cost_saving']}")
check("Text is non-empty string",
      isinstance(tf["text"], str) and len(tf["text"]) > 50)
check("Recommended is one of the two suppliers",
      tf["recommended"] in ["CheapSupplier","PremiumSupplier"])

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Full pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST GROUP 6 — Full Scoring Pipeline")
print("="*60)

scored = score_all_suppliers(sp_df)
check("Scored df not empty",            not scored.empty,              f"len={len(scored)}")
check("Has risk_score column",          RISK_SCORE    in scored.columns)
check("Has mcdm_score column",          MCDM_SCORE    in scored.columns)
check("Has cost_rank column",           COST_RANK     in scored.columns)
check("Has mcdm_rank column",           MCDM_RANK     in scored.columns)
check("Has is_recommended column",      IS_RECOMMENDED in scored.columns)
check("Has headline column",            "headline"     in scored.columns)
check("Has positives column",           "positives"    in scored.columns)
check("Has risks column",               "risks"        in scored.columns)
check("Has action column",              "action"       in scored.columns)
check("Risk scores in [0,1]",
      ((scored[RISK_SCORE] >= 0) & (scored[RISK_SCORE] <= 1)).all())
check("MCDM scores in [0,100]",
      ((scored[MCDM_SCORE] >= 0) & (scored[MCDM_SCORE] <= 100)).all())
check("Each product has exactly one recommended",
      scored.groupby(PRODUCT_ID)[IS_RECOMMENDED].sum().eq(1).all(),
      f"counts: {scored.groupby(PRODUCT_ID)[IS_RECOMMENDED].sum().to_dict()}")
check("Cost rank 1 = cheapest per product",
      all(
          scored[scored[PRODUCT_ID] == pid][scored[scored[PRODUCT_ID]==pid][COST_RANK]==1][UNIT_COST].values[0] ==
          scored[scored[PRODUCT_ID] == pid][UNIT_COST].min()
          for pid in scored[PRODUCT_ID].unique()
          if not scored[scored[PRODUCT_ID] == pid].empty
      ))
check("All headlines are strings",
      scored["headline"].apply(lambda x: isinstance(x, str)).all())
check("All positives are lists",
      scored["positives"].apply(lambda x: isinstance(x, list)).all())

# Single product
para_scored = score_product_suppliers(sp_df, "SKU003")
check("Single product scoring works",   not para_scored.empty)
check("Only SKU003 in result",
      (para_scored[PRODUCT_ID] == "SKU003").all())

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"❌ {len(errors)} test(s) FAILED: {errors}")
else:
    print(f"✅ All tests passed!")
print("="*60)
