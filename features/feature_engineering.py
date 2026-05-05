"""
features/feature_engineering.py
Transforms raw sales, batch, and supplier data into ML-ready features.
"""
import os, sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (
    SKU_ID, DATE, QTY_SOLD, STOCK_LEVEL, EXPIRY_DATE, DAYS_TO_EXPIRY,
    UNIT_COST, SELLING_PRICE, LEAD_TIME_DAYS, LEAD_TIME_STD,
    SALES_VELOCITY, STOCK_COVER, PRODUCT_NAME, CATEGORY,
    SUPPLIER_ID, BATCH_ID, REORDER_POINT
)
from config.settings import ROLLING_WINDOW_DAYS, FORECAST_HORIZON_DAYS

logger = logging.getLogger(__name__)


def _check_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing columns: {missing}")

def _safe_merge(left, right, on, how="left"):
    overlap = [c for c in right.columns if c in left.columns and c != on]
    if overlap:
        right = right.drop(columns=overlap)
    return left.merge(right, on=on, how=how)


def build_sales_features(sales_df):
    _check_columns(sales_df, [SKU_ID, DATE, QTY_SOLD, STOCK_LEVEL], "sales_features")
    df = sales_df.copy()
    df[DATE]     = pd.to_datetime(df[DATE])
    df[QTY_SOLD] = pd.to_numeric(df[QTY_SOLD], errors="coerce").fillna(0)
    df = df.sort_values([SKU_ID, DATE])
    grp = df.groupby(SKU_ID)

    df[SALES_VELOCITY] = (grp[QTY_SOLD]
        .transform(lambda x: x.rolling(ROLLING_WINDOW_DAYS, min_periods=7).mean())
        .fillna(df[QTY_SOLD]))

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}d"] = grp[QTY_SOLD].transform(lambda x: x.shift(lag))

    df["rolling_std_7d"]   = grp[QTY_SOLD].transform(lambda x: x.rolling(7,  min_periods=3).std()).fillna(1)
    df["rolling_std_30d"]  = grp[QTY_SOLD].transform(lambda x: x.rolling(30, min_periods=7).std()).fillna(1)
    df["rolling_mean_7d"]  = grp[QTY_SOLD].transform(lambda x: x.rolling(7,  min_periods=3).mean()).fillna(df[QTY_SOLD])
    df["rolling_mean_30d"] = grp[QTY_SOLD].transform(lambda x: x.rolling(30, min_periods=7).mean()).fillna(df[QTY_SOLD])

    df["day_of_week"]  = df[DATE].dt.dayofweek
    df["month"]        = df[DATE].dt.month
    df["week_of_year"] = df[DATE].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = df[DATE].dt.quarter
    return df


def build_sku_summary(sales_df):
    df = sales_df.copy()
    df[QTY_SOLD]    = pd.to_numeric(df[QTY_SOLD],    errors="coerce").fillna(0)
    df[STOCK_LEVEL] = pd.to_numeric(df[STOCK_LEVEL], errors="coerce").fillna(0)
    summary = df.groupby(SKU_ID).agg(
        avg_daily_demand  = (QTY_SOLD,    "mean"),
        std_daily_demand  = (QTY_SOLD,    "std"),
        total_sold_30d    = (QTY_SOLD,    lambda x: x.tail(30).sum()),
        total_sold_90d    = (QTY_SOLD,    lambda x: x.tail(90).sum()),
        current_stock     = (STOCK_LEVEL, "last"),
        **{SALES_VELOCITY: (QTY_SOLD,    lambda x: x.tail(ROLLING_WINDOW_DAYS).mean())},
    ).reset_index()
    safe_demand = summary["avg_daily_demand"].clip(lower=0.01)
    summary[STOCK_COVER]      = (summary["current_stock"] / safe_demand).round(1)
    summary["std_daily_demand"] = summary["std_daily_demand"].fillna(1)
    return summary


def build_batch_features(batches_df, sku_summary):
    df = batches_df.copy()
    df[DAYS_TO_EXPIRY] = pd.to_numeric(df[DAYS_TO_EXPIRY], errors="coerce").fillna(0)
    df[STOCK_LEVEL]    = pd.to_numeric(df[STOCK_LEVEL],    errors="coerce").fillna(0)

    if SALES_VELOCITY in sku_summary.columns:
        vel_df = sku_summary[[SKU_ID, SALES_VELOCITY, "avg_daily_demand"]].copy()
        df = _safe_merge(df, vel_df, on=SKU_ID, how="left")
    else:
        df[SALES_VELOCITY]     = 1.0
        df["avg_daily_demand"] = 1.0

    df[SALES_VELOCITY]     = df[SALES_VELOCITY].fillna(1).clip(lower=0.01)
    df["avg_daily_demand"] = df["avg_daily_demand"].fillna(1).clip(lower=0.01)
    df["projected_sell_days"] = (df[STOCK_LEVEL] / df[SALES_VELOCITY]).clip(lower=0)
    df["will_expire"] = (df["projected_sell_days"] > df[DAYS_TO_EXPIRY].clip(lower=0)).astype(int)

    sellable = (df[DAYS_TO_EXPIRY].clip(lower=0) * df[SALES_VELOCITY])
    capped   = sellable.where(sellable <= df[STOCK_LEVEL], df[STOCK_LEVEL])
    df["at_risk_units"] = np.where(df["will_expire"] == 1, capped, 0)
    df["at_risk_units"] = df["at_risk_units"].clip(lower=0)

    dte_safe = df[DAYS_TO_EXPIRY].clip(lower=1)
    df["urgency_ratio"] = (df[STOCK_LEVEL] / (df[SALES_VELOCITY] * dte_safe)).clip(upper=10).fillna(1)
    return df


def build_sku_batch_summary(batch_features_df):
    df     = batch_features_df.copy()
    active = df[df[DAYS_TO_EXPIRY] >= 0].copy()
    if active.empty:
        active = df.copy()
    return active.groupby(SKU_ID).agg(
        nearest_expiry       = (DAYS_TO_EXPIRY, "min"),
        total_batch_stock    = (STOCK_LEVEL,    "sum"),
        n_critical_batches   = ("will_expire",  "sum"),
        max_urgency_ratio    = ("urgency_ratio","max"),
        total_at_risk_units  = ("at_risk_units","sum"),
    ).reset_index()


def build_supplier_features(suppliers_df):
    df = suppliers_df.copy()
    df[LEAD_TIME_DAYS] = pd.to_numeric(df[LEAD_TIME_DAYS], errors="coerce").fillna(14)

    if "actual_lead_time" in df.columns:
        df["actual_lead_time"] = pd.to_numeric(df["actual_lead_time"], errors="coerce").fillna(df[LEAD_TIME_DAYS])
        df["delay"]    = (df["actual_lead_time"] - df[LEAD_TIME_DAYS]).clip(lower=0)
        df["is_delay"] = (df["actual_lead_time"] > df[LEAD_TIME_DAYS]).astype(int)
        sup_summary = df.groupby(SKU_ID).agg(
            mean_lead_time    = ("actual_lead_time","mean"),
            std_lead_time     = ("actual_lead_time","std"),
            delay_probability = ("is_delay",        "mean"),
            avg_delay_days    = ("delay",            "mean"),
            n_orders          = ("actual_lead_time","count"),
        ).reset_index()
    else:
        std_col = LEAD_TIME_STD if LEAD_TIME_STD in df.columns else LEAD_TIME_DAYS
        sup_summary = df.groupby(SKU_ID).agg(
            mean_lead_time = (LEAD_TIME_DAYS, "first"),
            n_orders       = (LEAD_TIME_DAYS, "count"),
        ).reset_index()
        sup_summary["std_lead_time"]     = df.groupby(SKU_ID)[LEAD_TIME_DAYS].first().values * 0.2
        sup_summary["delay_probability"] = 0.20
        sup_summary["avg_delay_days"]    = 2.0

    sup_summary["std_lead_time"]     = sup_summary["std_lead_time"].fillna(1)
    sup_summary["delay_probability"] = sup_summary["delay_probability"].fillna(0.2)
    sup_summary["lead_time_cv"]      = (sup_summary["std_lead_time"] / sup_summary["mean_lead_time"].clip(lower=1))
    return sup_summary


def build_master_features(sales_df, batches_df, suppliers_df, products_df):
    sales_feat    = build_sales_features(sales_df)
    sku_summary   = build_sku_summary(sales_feat)
    batch_feat    = build_batch_features(batches_df, sku_summary)
    batch_summary = build_sku_batch_summary(batch_feat)
    sup_features  = build_supplier_features(suppliers_df)

    master = products_df.copy() if products_df is not None else pd.DataFrame({SKU_ID: sku_summary[SKU_ID].unique()})
    master = _safe_merge(master, sku_summary,   on=SKU_ID, how="left")
    master = _safe_merge(master, batch_summary, on=SKU_ID, how="left")
    master = _safe_merge(master, sup_features,  on=SKU_ID, how="left")

    master["nearest_expiry"]      = master["nearest_expiry"].fillna(365)
    master["total_at_risk_units"] = master["total_at_risk_units"].fillna(0)
    master["n_critical_batches"]  = master["n_critical_batches"].fillna(0)
    master["current_stock"]       = master["current_stock"].fillna(0)
    master[STOCK_COVER]           = master[STOCK_COVER].fillna(30)
    master[SALES_VELOCITY]        = master[SALES_VELOCITY].fillna(1)
    master["delay_probability"]   = master["delay_probability"].fillna(0.2)
    master["mean_lead_time"]      = master["mean_lead_time"].fillna(14)
    master["std_lead_time"]       = master["std_lead_time"].fillna(3)

    for col, default in [(UNIT_COST, 5.0), (SELLING_PRICE, 12.0)]:
        if col not in master.columns:
            master[col] = default
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(default)

    logger.info(f"Master feature table: {len(master)} SKUs, {len(master.columns)} features")
    return master
