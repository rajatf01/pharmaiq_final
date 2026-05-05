"""
business/financial_layer.py
Per-SKU cost calculation and AI vs baseline ROI comparison.
"""
import os, sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (SKU_ID, PRODUCT_NAME, UNIT_COST, SELLING_PRICE,
    STOCK_LEVEL, SALES_VELOCITY, EXPIRY_RISK, STOCKOUT_RISK,
    EXPIRY_LOSS, HOLDING_COST, STOCKOUT_PENALTY, TOTAL_COST,
    RECOMMENDED_QTY, DEMAND_FORECAST)
from config.settings import (HOLDING_COST_RATE, STOCKOUT_PENALTY_MULT,
    EXPIRY_SALVAGE_RATE, FORECAST_HORIZON_DAYS, HIGH_EXPIRY_RISK)

logger = logging.getLogger(__name__)


def calc_expiry_loss(row, horizon_days=FORECAST_HORIZON_DAYS):
    expiry_risk    = float(row.get(EXPIRY_RISK, 0))
    at_risk_units  = float(row.get("total_at_risk_units", 0))
    current_stock  = float(row.get("current_stock", 0))
    unit_cost      = max(float(row.get(UNIT_COST, 5.0)), 0.01)
    velocity       = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    nearest_expiry = float(row.get("nearest_expiry", 365))
    if at_risk_units <= 0:
        at_risk_units = expiry_risk * current_stock
    sellable      = velocity * max(nearest_expiry, 0)
    truly_at_risk = max(0, at_risk_units - sellable)
    return round(truly_at_risk * unit_cost * (1-EXPIRY_SALVAGE_RATE) * expiry_risk, 2)


def calc_holding_cost(row, horizon_days=FORECAST_HORIZON_DAYS):
    current_stock = max(float(row.get("current_stock", 0)), 0)
    unit_cost     = max(float(row.get(UNIT_COST, 5.0)), 0.01)
    return round(current_stock * unit_cost * (HOLDING_COST_RATE/365) * horizon_days, 2)


def calc_stockout_penalty(row, horizon_days=FORECAST_HORIZON_DAYS):
    stockout_risk = float(row.get(STOCKOUT_RISK, 0))
    velocity      = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    unit_cost     = max(float(row.get(UNIT_COST, 5.0)), 0.01)
    selling_price = max(float(row.get(SELLING_PRICE, unit_cost*2.5)), unit_cost)
    lead_time     = float(row.get("effective_lead_time", 14))
    gross_margin  = selling_price - unit_cost
    demand_at_risk = velocity * lead_time * stockout_risk
    return round(demand_at_risk * gross_margin * STOCKOUT_PENALTY_MULT, 2)


def calc_expiry_loss_with_risk(row, expiry_risk_override, horizon_days=FORECAST_HORIZON_DAYS):
    at_risk_units  = float(row.get("total_at_risk_units", 0))
    current_stock  = float(row.get("current_stock", 0))
    unit_cost      = max(float(row.get(UNIT_COST, 5.0)), 0.01)
    velocity       = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    nearest_expiry = float(row.get("nearest_expiry", 365))
    expiry_risk    = float(expiry_risk_override)
    if at_risk_units <= 0:
        at_risk_units = expiry_risk * current_stock
    sellable      = velocity * max(nearest_expiry, 0)
    truly_at_risk = max(0, at_risk_units - sellable)
    return round(truly_at_risk * unit_cost * (1-EXPIRY_SALVAGE_RATE) * expiry_risk, 2)


def calc_stockout_penalty_with_risk(row, stockout_risk_override, horizon_days=FORECAST_HORIZON_DAYS):
    stockout_risk = float(stockout_risk_override)
    velocity      = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    unit_cost     = max(float(row.get(UNIT_COST, 5.0)), 0.01)
    selling_price = max(float(row.get(SELLING_PRICE, unit_cost*2.5)), unit_cost)
    lead_time     = float(row.get("effective_lead_time", 14))
    gross_margin  = selling_price - unit_cost
    demand_at_risk = velocity * lead_time * stockout_risk
    return round(demand_at_risk * gross_margin * STOCKOUT_PENALTY_MULT, 2)


def simulate_baseline_costs(master_df, horizon_days=FORECAST_HORIZON_DAYS):
    df = master_df.copy()
    df["_base_expiry_risk"]   = (df[EXPIRY_RISK]  * 1.80).clip(0,1)
    df["_base_stockout_risk"] = (df[STOCKOUT_RISK] * 1.50).clip(0,1)
    df["baseline_expiry_loss"]       = df.apply(lambda r: calc_expiry_loss_with_risk(r, r["_base_expiry_risk"], horizon_days)*1.80, axis=1)
    df["baseline_holding_cost"]      = df.apply(lambda r: calc_holding_cost(r, horizon_days)*1.30, axis=1)
    df["baseline_stockout_penalty"]  = df.apply(lambda r: calc_stockout_penalty_with_risk(r, r["_base_stockout_risk"], horizon_days)*1.50, axis=1)
    df["baseline_total_cost"]        = df["baseline_expiry_loss"]+df["baseline_holding_cost"]+df["baseline_stockout_penalty"]
    df.drop(columns=[c for c in df.columns if c.startswith("_base_")], inplace=True, errors="ignore")
    return df


def compute_ai_costs(master_df, horizon_days=FORECAST_HORIZON_DAYS):
    df = master_df.copy()
    df[EXPIRY_LOSS]      = df.apply(lambda r: calc_expiry_loss(r, horizon_days), axis=1)
    df[HOLDING_COST]     = df.apply(lambda r: calc_holding_cost(r, horizon_days), axis=1)
    df[STOCKOUT_PENALTY] = df.apply(lambda r: calc_stockout_penalty(r, horizon_days), axis=1)
    df[TOTAL_COST]       = df[EXPIRY_LOSS]+df[HOLDING_COST]+df[STOCKOUT_PENALTY]
    return df


def build_financial_summary(master_df, implementation_cost=5000.0, horizon_days=FORECAST_HORIZON_DAYS):
    df = compute_ai_costs(master_df, horizon_days)
    df = simulate_baseline_costs(df, horizon_days)

    ai_total = df[TOTAL_COST].sum()
    bl_total = df["baseline_total_cost"].sum()
    savings  = bl_total - ai_total
    savings_pct = (savings/bl_total*100) if bl_total > 0 else 0
    roi         = ((savings-implementation_cost)/implementation_cost*100) if implementation_cost > 0 else 0
    payback     = (implementation_cost/(savings/(horizon_days/30))) if savings > 0 else float("inf")

    df["cost_saving"] = df["baseline_total_cost"] - df[TOTAL_COST]
    cols = [SKU_ID, "cost_saving", EXPIRY_LOSS, HOLDING_COST, STOCKOUT_PENALTY, TOTAL_COST, "baseline_total_cost"]
    if PRODUCT_NAME in df.columns:
        cols = [SKU_ID, PRODUCT_NAME] + cols[1:]
    top_savings = df[[c for c in cols if c in df.columns]].nlargest(10,"cost_saving").reset_index(drop=True)

    summary = {
        "ai":       {"expiry_loss":round(df[EXPIRY_LOSS].sum(),2),"holding_cost":round(df[HOLDING_COST].sum(),2),
                     "stockout_penalty":round(df[STOCKOUT_PENALTY].sum(),2),"total_cost":round(ai_total,2)},
        "baseline": {"expiry_loss":round(df["baseline_expiry_loss"].sum(),2),
                     "holding_cost":round(df["baseline_holding_cost"].sum(),2),
                     "stockout_penalty":round(df["baseline_stockout_penalty"].sum(),2),"total_cost":round(bl_total,2)},
    }
    roi_metrics = {"cost_savings_usd":round(savings,2),"savings_percent":round(savings_pct,1),
                   "roi_percent":round(roi,1),"payback_months":round(payback,1),"implementation_cost":implementation_cost}

    logger.info(f"Financial: savings=${savings:,.0f} ({savings_pct:.1f}%), ROI={roi:.1f}%")
    return {"sku_costs":df,"summary":summary,"roi_metrics":roi_metrics,"top_savings":top_savings}
