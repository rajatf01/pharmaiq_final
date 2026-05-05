"""
engine/decision_engine.py
Risk-aware reorder decision engine — core business logic.
"""
import os, sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (
    SKU_ID, PRODUCT_NAME, STOCK_LEVEL, SALES_VELOCITY, STOCK_COVER,
    DEMAND_FORECAST, EXPIRY_RISK, EXPIRY_PRESSURE, STOCKOUT_RISK,
    SUPPLIER_RISK, UNIT_COST, SELLING_PRICE, LEAD_TIME_DAYS,
    RECOMMENDED_QTY, ACTION, ACTION_REASON,
    ACTION_REORDER_NOW, ACTION_HOLD, ACTION_REVIEW,
    ACTION_REDUCE, ACTION_URGENT_EXPIRY,
)
from config.settings import (
    HIGH_EXPIRY_RISK, MEDIUM_EXPIRY_RISK, HIGH_STOCKOUT_RISK,
    HIGH_SUPPLIER_RISK, DEFAULT_SERVICE_LEVEL, Z_SCORE_95,
    FORECAST_HORIZON_DAYS, CRITICAL_EXPIRY_DAYS, WARNING_EXPIRY_DAYS,
)

logger = logging.getLogger(__name__)


def safety_stock(daily_std, lead_time, service_z=Z_SCORE_95):
    return max(0.0, service_z * daily_std * np.sqrt(max(lead_time, 1)))


def eoq(annual_demand, order_cost=50.0, unit_cost=5.0, holding_rate=0.25):
    if annual_demand <= 0 or unit_cost <= 0: return 0.0
    h = unit_cost * holding_rate
    return np.sqrt((2 * annual_demand * order_cost) / h)


def compute_stockout_risk(row, service_level=DEFAULT_SERVICE_LEVEL):
    stock_cover = float(row.get(STOCK_COVER, 30))
    eff_lead    = float(row.get("effective_lead_time", 14))
    velocity    = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    daily_std   = max(float(row.get("std_daily_demand", velocity*0.3)), 0.01)
    sup_risk    = float(row.get(SUPPLIER_RISK, 0.2))
    ss_days     = safety_stock(daily_std, eff_lead) / velocity
    buffer      = eff_lead + ss_days
    if stock_cover <= 0: return 1.0
    if stock_cover >= buffer*2: return 0.05
    base = 1 - (stock_cover / buffer)
    return float(np.clip(np.clip(base,0,1) * (1+sup_risk*0.4), 0.0, 1.0))


def compute_reorder_qty(row, service_level=DEFAULT_SERVICE_LEVEL, disruption_factor=1.0):
    daily_vel      = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    daily_std      = max(float(row.get("std_daily_demand", daily_vel*0.3)), 0.01)
    annual_demand  = daily_vel * 365
    unit_cost      = max(float(row.get(UNIT_COST, 5.0)), 0.01)
    eff_lead       = float(row.get("effective_lead_time", 14))
    expiry_risk    = float(row.get(EXPIRY_RISK, 0.0))
    sup_risk       = float(row.get(SUPPLIER_RISK, 0.2))
    current_stock  = float(row.get("current_stock", 0))
    nearest_exp    = float(row.get("nearest_expiry", 365))

    base_qty   = eoq(annual_demand, unit_cost=unit_cost)
    ss         = safety_stock(daily_std, eff_lead, Z_SCORE_95 if service_level>=0.95 else 1.28)
    sup_buffer = base_qty * sup_risk * 0.25 * disruption_factor
    gross_qty  = base_qty + ss + sup_buffer

    if expiry_risk >= HIGH_EXPIRY_RISK:
        gross_qty = min(gross_qty, daily_vel * eff_lead * 1.2)
    elif expiry_risk >= MEDIUM_EXPIRY_RISK:
        reduction = (expiry_risk - MEDIUM_EXPIRY_RISK) / (HIGH_EXPIRY_RISK - MEDIUM_EXPIRY_RISK)
        gross_qty = gross_qty * (1 - 0.40*reduction)

    if nearest_exp > 0 and nearest_exp < 180:
        sellable = daily_vel * nearest_exp
        gross_qty = min(gross_qty, max(0, sellable - current_stock))

    net_qty = max(0, gross_qty - max(0, current_stock - ss))
    return round(net_qty, 0)


def determine_action(row):
    expiry_risk   = float(row.get(EXPIRY_RISK, 0))
    stockout_risk = float(row.get(STOCKOUT_RISK, 0))
    sup_risk      = float(row.get(SUPPLIER_RISK, 0))
    nearest_exp   = float(row.get("nearest_expiry", 365))
    stock_cover   = float(row.get(STOCK_COVER, 30))
    reorder_qty   = float(row.get(RECOMMENDED_QTY, 0))

    if nearest_exp < 0:
        return (ACTION_URGENT_EXPIRY,
                "Batch already expired. Remove from inventory immediately.")

    if expiry_risk >= HIGH_EXPIRY_RISK and stockout_risk < HIGH_STOCKOUT_RISK:
        msg = (f"Expiry risk is critical ({expiry_risk:.0%}): stock likely to expire "
               f"in ~{int(nearest_exp)} days.")
        if nearest_exp <= CRITICAL_EXPIRY_DAYS:
            msg += " Consider promotions, discounts, or returns to supplier."
        return (ACTION_REDUCE, msg)

    if stockout_risk >= HIGH_STOCKOUT_RISK and expiry_risk < MEDIUM_EXPIRY_RISK:
        msg = (f"Stockout risk is high ({stockout_risk:.0%}): "
               f"only {stock_cover:.0f} days of cover remaining.")
        if sup_risk >= HIGH_SUPPLIER_RISK:
            msg += f" Supplier is unreliable ({sup_risk:.0%} risk) — order earlier."
        return (ACTION_REORDER_NOW, msg)

    if stockout_risk >= HIGH_STOCKOUT_RISK and expiry_risk >= MEDIUM_EXPIRY_RISK:
        return (ACTION_REORDER_NOW,
                f"Competing risks: stockout={stockout_risk:.0%}, expiry={expiry_risk:.0%}. "
                f"Reorder {int(reorder_qty)} units (reduced to limit expiry exposure).")

    if stockout_risk < 0.30 and expiry_risk < MEDIUM_EXPIRY_RISK:
        return (ACTION_HOLD,
                f"Inventory well-positioned: {stock_cover:.0f} days cover, low expiry risk.")

    return (ACTION_REVIEW,
            f"Mixed signals — stockout risk {stockout_risk:.0%}, expiry risk {expiry_risk:.0%}. "
            "Manual review recommended.")


def run_decision_engine(scored_df, service_level=DEFAULT_SERVICE_LEVEL, disruption_factor=1.0):
    df = scored_df.copy()
    if STOCKOUT_RISK not in df.columns:
        df[STOCKOUT_RISK] = df.apply(lambda r: compute_stockout_risk(r, service_level), axis=1).round(4)
    df[RECOMMENDED_QTY] = df.apply(lambda r: compute_reorder_qty(r, service_level, disruption_factor), axis=1)
    actions = df.apply(determine_action, axis=1)
    df[ACTION]        = actions.apply(lambda x: x[0])
    df[ACTION_REASON] = actions.apply(lambda x: x[1])
    df["priority_score"] = (
        0.40*df[EXPIRY_RISK].fillna(0) +
        0.35*df[STOCKOUT_RISK].fillna(0) +
        0.25*df[SUPPLIER_RISK].fillna(0)
    ).round(4)

    n_reorder = (df[ACTION] == ACTION_REORDER_NOW).sum()
    n_reduce  = (df[ACTION].isin([ACTION_REDUCE, ACTION_URGENT_EXPIRY])).sum()
    logger.info(f"Decision engine complete: {n_reorder} reorder, {n_reduce} reduce/urgent")
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)
