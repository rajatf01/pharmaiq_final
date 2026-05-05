"""
explainability/explainer.py
Plain-English AI explanations for every inventory decision.
"""
import os, sys, logging
import numpy as np
import pandas as pd
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (SKU_ID, PRODUCT_NAME, SALES_VELOCITY, STOCK_COVER,
    DEMAND_FORECAST, EXPIRY_RISK, EXPIRY_PRESSURE, STOCKOUT_RISK, SUPPLIER_RISK,
    RECOMMENDED_QTY, ACTION, ACTION_REASON, UNIT_COST, SELLING_PRICE)
from config.settings import (HIGH_EXPIRY_RISK, MEDIUM_EXPIRY_RISK,
    HIGH_STOCKOUT_RISK, HIGH_SUPPLIER_RISK, CRITICAL_EXPIRY_DAYS, WARNING_EXPIRY_DAYS)

logger = logging.getLogger(__name__)


def safe_val(row, col, default=0):
    try:
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except: return default


def explain_demand(row):
    velocity   = float(safe_val(row, SALES_VELOCITY, 0))
    avg_demand = float(safe_val(row, "avg_daily_demand", velocity))
    forecast   = float(safe_val(row, DEMAND_FORECAST, velocity*30))
    sku        = str(safe_val(row, PRODUCT_NAME, safe_val(row, SKU_ID, "This SKU")))
    lines = []
    if avg_demand > 0:
        pct = (velocity - avg_demand) / avg_demand * 100
        if pct > 20:
            lines.append(f"{sku} demand is trending UP by {pct:.0f}% vs historical average.")
        elif pct < -20:
            lines.append(f"{sku} demand has slowed DOWN by {abs(pct):.0f}% vs historical average. This increases expiry risk.")
        else:
            lines.append(f"{sku} demand is stable at ~{velocity:.1f} units/day ({pct:+.0f}% vs historical average).")
    else:
        lines.append(f"{sku} has no recent sales — check if product is active.")
    lines.append(f"Forecast for next 30 days: {forecast:.0f} units (~{velocity:.1f} units/day).")
    return " ".join(lines)


def explain_expiry_risk(row):
    risk         = float(safe_val(row, EXPIRY_RISK, 0))
    pressure     = str(safe_val(row, EXPIRY_PRESSURE, "Low"))
    dte          = float(safe_val(row, "nearest_expiry", 365))
    urgency      = float(safe_val(row, "max_urgency_ratio", 1))
    at_risk      = float(safe_val(row, "total_at_risk_units", 0))
    velocity     = max(float(safe_val(row, SALES_VELOCITY, 1)), 0.01)
    sku          = str(safe_val(row, PRODUCT_NAME, safe_val(row, SKU_ID, "This SKU")))
    lines = [f"Expiry risk: {pressure} ({risk:.0%})."]
    if dte < 0:
        lines.append("⚠️ At least one batch has ALREADY EXPIRED — remove immediately.")
        return " ".join(lines)
    if dte <= CRITICAL_EXPIRY_DAYS:
        lines.append(f"Nearest batch expires in {int(dte)} days — CRITICAL window.")
    elif dte <= WARNING_EXPIRY_DAYS:
        lines.append(f"Nearest batch expires in {int(dte)} days.")
    days_to_sell = at_risk / velocity if velocity > 0 else float("inf")
    if days_to_sell > dte and at_risk > 0:
        lines.append(f"At current velocity ({velocity:.1f}/day), it will take ~{days_to_sell:.0f} days to clear {at_risk:.0f} at-risk units — longer than the {int(dte)}-day expiry window.")
        lines.append("Consider promotions or returns to supplier to move stock faster.")
    elif at_risk > 0:
        lines.append(f"{at_risk:.0f} units are flagged at-risk, but current velocity should clear them before expiry.")
    else:
        lines.append("No immediate expiry exposure detected.")
    return " ".join(lines)


def explain_reorder(row):
    qty         = float(safe_val(row, RECOMMENDED_QTY, 0))
    expiry_risk = float(safe_val(row, EXPIRY_RISK, 0))
    stockout_r  = float(safe_val(row, STOCKOUT_RISK, 0))
    sup_risk    = float(safe_val(row, SUPPLIER_RISK, 0))
    cover       = float(safe_val(row, STOCK_COVER, 0))
    eff_lead    = float(safe_val(row, "effective_lead_time", 14))
    sku         = str(safe_val(row, PRODUCT_NAME, safe_val(row, SKU_ID, "This SKU")))
    lines = []
    if qty == 0:
        lines.append(f"No reorder recommended for {sku}.")
        if expiry_risk >= HIGH_EXPIRY_RISK:
            lines.append("High expiry risk means adding more stock would increase losses — focus on clearing existing inventory first.")
        elif cover > 60:
            lines.append(f"Current stock cover is {cover:.0f} days — no immediate reorder needed.")
        return " ".join(lines)
    lines.append(f"Recommended reorder: {int(qty)} units.")
    reasons = []
    if stockout_r >= HIGH_STOCKOUT_RISK:
        reasons.append(f"stockout risk is high ({stockout_r:.0%}) with only {cover:.0f} days of cover vs {eff_lead:.0f}-day lead time")
    if sup_risk >= HIGH_SUPPLIER_RISK:
        reasons.append(f"supplier reliability is low ({sup_risk:.0%}) — extra buffer added")
    if expiry_risk >= MEDIUM_EXPIRY_RISK:
        reasons.append(f"expiry risk ({expiry_risk:.0%}) reduced the order quantity to avoid adding to at-risk stock")
    if reasons:
        lines.append("Rationale: " + "; ".join(reasons) + ".")
    return " ".join(lines)


def explain_supplier_risk(row):
    sup_risk   = float(safe_val(row, SUPPLIER_RISK, 0))
    delay_prob = float(safe_val(row, "delay_probability", 0))
    lt_cv      = float(safe_val(row, "lead_time_cv", 0))
    mean_lt    = float(safe_val(row, "mean_lead_time", 14))
    avg_delay  = float(safe_val(row, "avg_delay_days", 0))
    sup_name   = str(safe_val(row, "supplier_name", "Supplier"))
    label = "High" if sup_risk >= HIGH_SUPPLIER_RISK else ("Medium" if sup_risk >= 0.35 else "Low")
    lines = [f"Supplier risk: {label} ({sup_risk:.0%}) for {sup_name}."]
    if delay_prob > 0.3:
        lines.append(f"Delays occur {delay_prob:.0%} of the time with avg delay of {avg_delay:.1f} days.")
    if lt_cv > 0.25:
        lines.append(f"Lead time is highly variable (CV={lt_cv:.2f}), making planning difficult.")
    if sup_risk < 0.35:
        lines.append(f"Lead time is consistent (~{mean_lt:.0f} days) — low risk.")
    return " ".join(lines)


def generate_full_explanation(row):
    return {
        "demand":   explain_demand(row),
        "expiry":   explain_expiry_risk(row),
        "reorder":  explain_reorder(row),
        "supplier": explain_supplier_risk(row),
    }


def add_explanations(master_df):
    df = master_df.copy()
    df["explain_demand"]   = df.apply(explain_demand,        axis=1)
    df["explain_expiry"]   = df.apply(explain_expiry_risk,   axis=1)
    df["explain_reorder"]  = df.apply(explain_reorder,       axis=1)
    df["explain_supplier"] = df.apply(explain_supplier_risk, axis=1)
    return df


def get_top_risk_insights(master_df, top_n=5):
    df = master_df.copy()
    insights = []
    if "nearest_expiry" in df.columns:
        n_exp = (df["nearest_expiry"] < 0).sum()
        if n_exp > 0:
            insights.append(f"🚨 {n_exp} SKU(s) have EXPIRED batches — immediate removal required.")
    if EXPIRY_RISK in df.columns:
        high = df[df[EXPIRY_RISK] >= HIGH_EXPIRY_RISK]
        if len(high) > 0:
            top  = high.nlargest(3, EXPIRY_RISK)
            nms  = ", ".join(top.get(PRODUCT_NAME, top[SKU_ID]).values.tolist())
            insights.append(f"⚠️ {len(high)} SKUs face critical expiry risk. Top concern: {nms}.")
    if STOCKOUT_RISK in df.columns:
        high_so = df[df[STOCKOUT_RISK] >= HIGH_STOCKOUT_RISK]
        if len(high_so) > 0:
            insights.append(f"📦 {len(high_so)} SKUs are at high stockout risk — reorder now to avoid lost sales.")
    if SUPPLIER_RISK in df.columns:
        high_sup = df[df[SUPPLIER_RISK] >= HIGH_SUPPLIER_RISK]
        if len(high_sup) > 0:
            insights.append(f"🚚 {len(high_sup)} SKUs are sourced from unreliable suppliers.")
    return insights[:top_n]
