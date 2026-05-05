"""
models/supplier_risk.py
Rule-based supplier reliability scoring.
"""
import os, sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (SKU_ID, SUPPLIER_ID, SUPPLIER_NAME, LEAD_TIME_DAYS, SUPPLIER_RISK)
from config.settings import HIGH_SUPPLIER_RISK, Z_SCORE_95

logger = logging.getLogger(__name__)


def compute_effective_lead_time(mean_lt, std_lt, service_z=Z_SCORE_95):
    return round(mean_lt + service_z * std_lt, 1)


def rule_based_supplier_risk(row):
    delay_p    = float(row.get("delay_probability", 0.2))
    lt_cv      = float(row.get("lead_time_cv",      0.2))
    avg_delay  = float(row.get("avg_delay_days",    2.0))
    n_orders   = max(int(row.get("n_orders", 10)), 1)
    confidence = min(1.0, n_orders / 20)
    delay_s    = delay_p
    cv_s       = min(1.0, lt_cv / 0.5)
    impact_s   = min(1.0, avg_delay / 14)
    raw        = 0.50*delay_s + 0.30*cv_s + 0.20*impact_s
    score      = 0.2*(1-confidence) + raw*confidence
    return float(np.clip(score, 0.0, 1.0))


class SupplierRiskModel:
    def __init__(self):
        self._trained = False
        self.metrics_ = {}

    def fit(self, supplier_features_df):
        required = [SKU_ID, "mean_lead_time", "delay_probability"]
        if any(c not in supplier_features_df.columns for c in required):
            logger.warning("Supplier features missing required columns")
            return self
        self.metrics_["n_skus"]          = len(supplier_features_df)
        self.metrics_["mean_delay_prob"] = round(supplier_features_df["delay_probability"].mean(), 3)
        self.metrics_["mean_lead_time"]  = round(supplier_features_df["mean_lead_time"].mean(), 1)
        self._trained = True
        logger.info(f"SupplierRiskModel ready: {self.metrics_['n_skus']} SKUs, avg delay prob={self.metrics_['mean_delay_prob']:.1%}")
        return self

    def predict_scores(self, master_df):
        df = master_df.copy()
        df[SUPPLIER_RISK] = df.apply(rule_based_supplier_risk, axis=1).round(4)
        df["effective_lead_time"] = df.apply(
            lambda r: compute_effective_lead_time(r.get("mean_lead_time",14), r.get("std_lead_time",3)), axis=1)
        df["supplier_risk_label"] = df[SUPPLIER_RISK].apply(
            lambda s: "High" if s>=HIGH_SUPPLIER_RISK else ("Medium" if s>=0.35 else "Low"))
        return df

    def get_metrics(self): return self.metrics_


def run_supplier_risk(master_df, supplier_features_df):
    model     = SupplierRiskModel()
    model.fit(supplier_features_df)
    scored_df = model.predict_scores(master_df)
    logger.info(f"Supplier risk scored: {(scored_df[SUPPLIER_RISK]>=HIGH_SUPPLIER_RISK).sum()} high-risk SKUs")
    return scored_df, model
