"""
models/expiry_risk.py
Expiry risk scoring — most critical model in PharmaIQ.
"""
import os, sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (SKU_ID, DAYS_TO_EXPIRY, STOCK_LEVEL, SALES_VELOCITY,
    EXPIRY_RISK, EXPIRY_PRESSURE, UNIT_COST, SELLING_PRICE)
from config.settings import (HIGH_EXPIRY_RISK, MEDIUM_EXPIRY_RISK,
    CRITICAL_EXPIRY_DAYS, WARNING_EXPIRY_DAYS)

logger = logging.getLogger(__name__)

EXPIRY_FEATURES = ["nearest_expiry","urgency_ratio","total_at_risk_units",
                   "n_critical_batches",SALES_VELOCITY,"total_batch_stock","max_urgency_ratio"]


def rule_based_expiry_score(row):
    dte      = float(row.get("nearest_expiry", 365))
    velocity = max(float(row.get(SALES_VELOCITY, 1)), 0.01)
    stock    = max(float(row.get("total_batch_stock", row.get(STOCK_LEVEL, 0))), 0)
    urgency  = float(row.get("max_urgency_ratio", row.get("urgency_ratio", 1)))

    if dte < 0: return 1.0
    days_of_stock = stock / velocity
    expiry_proximity = 1 - (min(dte, 180) / 180)

    if days_of_stock <= 0:
        sell_through_risk = 0.0
    else:
        ratio = days_of_stock / max(dte, 1)
        sell_through_risk = min(1.0, max(0.0, ratio-1)/3 + (ratio>1)*0.3)

    urgency_score = min(1.0, (urgency-1)/4) if urgency > 1 else 0.0
    score = 0.45*sell_through_risk + 0.35*expiry_proximity + 0.20*urgency_score
    return float(np.clip(score, 0.0, 1.0))


def compute_expiry_pressure(score, dte):
    if dte < 0:                                  return "Expired"
    if dte <= CRITICAL_EXPIRY_DAYS or score >= HIGH_EXPIRY_RISK:   return "Critical"
    if dte <= WARNING_EXPIRY_DAYS  or score >= MEDIUM_EXPIRY_RISK:  return "High"
    if score >= 0.20:                            return "Medium"
    return "Low"


class ExpiryRiskModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self._trained     = False
        self.metrics_     = {}

    def fit(self, batch_features_df):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        df = batch_features_df.copy()
        available = [c for c in EXPIRY_FEATURES if c in df.columns]
        alt = [DAYS_TO_EXPIRY, SALES_VELOCITY, STOCK_LEVEL, "urgency_ratio",
               "projected_sell_days", "at_risk_units"]
        available += [c for c in alt if c in df.columns and c not in available]

        if "will_expire" not in df.columns:
            logger.warning("'will_expire' missing — skipping ML training")
            return self

        df_clean = df[available + ["will_expire"]].dropna()
        if len(df_clean) < 50 or df_clean["will_expire"].sum() < 5:
            logger.warning(f"Insufficient expiry training data ({len(df_clean)} rows) — using rule-based")
            return self

        self._feature_cols = list(df_clean[available].columns)
        self._model = GradientBoostingClassifier(n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=self.random_state)
        self._model.fit(df_clean[available].values, df_clean["will_expire"].values)
        self._trained = True

        cv = cross_val_score(self._model, df_clean[available].values,
                              df_clean["will_expire"].values, cv=5, scoring="roc_auc")
        self.metrics_["cv_auc_mean"]    = round(cv.mean(), 3)
        self.metrics_["cv_auc_std"]     = round(cv.std(), 3)
        self.metrics_["n_train"]        = len(df_clean)
        self.metrics_["positive_rate"]  = round(df_clean["will_expire"].mean(), 3)
        self.metrics_["feature_importance"] = pd.DataFrame({
            "feature":    self._feature_cols,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info(f"ExpiryRiskModel trained: AUC={self.metrics_['cv_auc_mean']:.3f}")
        return self

    def predict_scores(self, master_df):
        df = master_df.copy()
        scores = df.apply(rule_based_expiry_score, axis=1).values
        df[EXPIRY_RISK] = np.round(scores, 4)
        dte_col  = "nearest_expiry" if "nearest_expiry" in df.columns else DAYS_TO_EXPIRY
        dte_vals = df[dte_col].fillna(365) if dte_col in df.columns else pd.Series([365]*len(df))
        df[EXPIRY_PRESSURE] = [compute_expiry_pressure(s,d) for s,d in zip(scores, dte_vals)]
        return df

    def get_metrics(self): return self.metrics_
    def get_feature_importance(self):
        fi = self.metrics_.get("feature_importance")
        return fi if fi is not None else pd.DataFrame({"feature":[],"importance":[]})


def run_expiry_risk(master_df, batch_features_df):
    model     = ExpiryRiskModel()
    model.fit(batch_features_df)
    scored_df = model.predict_scores(master_df)
    logger.info(f"Expiry risk scored: {(scored_df[EXPIRY_RISK]>=HIGH_EXPIRY_RISK).sum()} high-risk SKUs")
    return scored_df, model
