"""
models/demand_forecasting.py
Moving Average baseline + Random Forest demand forecaster.
"""
import os, sys, logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.schema import (SKU_ID, DATE, QTY_SOLD, SALES_VELOCITY,
    DEMAND_FORECAST, FORECAST_LOWER, FORECAST_UPPER)
from config.settings import MA_WINDOW_DAYS, FORECAST_HORIZON_DAYS, ROLLING_WINDOW_DAYS

logger = logging.getLogger(__name__)

RF_FEATURE_COLS = ["lag_7d","lag_14d","lag_30d","rolling_mean_7d","rolling_mean_30d",
                    "rolling_std_7d","rolling_std_30d","day_of_week","month",
                    "week_of_year","is_weekend","quarter"]


def moving_average_forecast(sales_df, window=MA_WINDOW_DAYS, horizon=FORECAST_HORIZON_DAYS):
    df = sales_df.copy()
    df[DATE]     = pd.to_datetime(df[DATE])
    df[QTY_SOLD] = pd.to_numeric(df[QTY_SOLD], errors="coerce").fillna(0)
    df = df.sort_values([SKU_ID, DATE])
    results = []
    for sku, grp in df.groupby(SKU_ID):
        recent = grp[QTY_SOLD].tail(window)
        mu     = recent.mean()
        sigma  = recent.std() if len(recent) > 1 else mu * 0.2
        results.append({
            SKU_ID:           sku,
            DEMAND_FORECAST:  round(mu * horizon, 1),
            FORECAST_LOWER:   round(max(0, (mu - 1.645*sigma/np.sqrt(window))*horizon), 1),
            FORECAST_UPPER:   round((mu + 1.645*sigma/np.sqrt(window))*horizon, 1),
            "daily_forecast": round(mu, 2),
            "forecast_std":   round(sigma, 2),
            "model_type":     "moving_average",
        })
    return pd.DataFrame(results)


class DemandForecaster:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state  = random_state
        self._model        = None
        self._feature_cols = None
        self._trained      = False
        self.metrics_      = {}

    def fit(self, sales_features_df):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        df = sales_features_df.dropna(subset=[QTY_SOLD]).copy()
        df[QTY_SOLD] = pd.to_numeric(df[QTY_SOLD], errors="coerce").fillna(0)
        available = [c for c in RF_FEATURE_COLS if c in df.columns]
        if len(available) < 4:
            logger.warning("Not enough features for RF — using moving average")
            return self
        df_clean = df[available + [QTY_SOLD]].dropna()
        if len(df_clean) < 100:
            logger.warning(f"Only {len(df_clean)} training rows — using moving average")
            return self

        X = df_clean[available].values
        y = df_clean[QTY_SOLD].values
        self._model = RandomForestRegressor(n_estimators=self.n_estimators,
            max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=self.random_state)
        self._model.fit(X, y)
        self._feature_cols = available
        self._trained      = True

        cv = cross_val_score(self._model, X, y, cv=5,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
        self.metrics_["cv_mae"]  = round(-cv.mean(), 3)
        self.metrics_["cv_std"]  = round(cv.std(), 3)
        self.metrics_["n_train"] = len(df_clean)
        self.metrics_["features"]= available
        self.metrics_["feature_importance"] = pd.DataFrame({
            "feature":    available,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info(f"DemandForecaster trained: MAE={self.metrics_['cv_mae']:.2f}, n={len(df_clean)}")
        return self

    def predict_sku_forecast(self, sales_features_df, horizon=FORECAST_HORIZON_DAYS):
        if not self._trained:
            return moving_average_forecast(sales_features_df[[SKU_ID, DATE, QTY_SOLD]], horizon=horizon)
        df      = sales_features_df.copy().sort_values([SKU_ID, DATE])
        latest  = df.groupby(SKU_ID).last().reset_index()
        for c in [x for x in self._feature_cols if x not in latest.columns]:
            latest[c] = 0
        X_pred     = latest[self._feature_cols].fillna(0).values
        preds      = self._model.predict(X_pred).clip(min=0)
        tree_preds = np.stack([t.predict(X_pred) for t in self._model.estimators_], axis=1)
        pred_std   = tree_preds.std(axis=1)
        hist_std   = (df.groupby(SKU_ID)[QTY_SOLD].std()
                      .reindex(latest[SKU_ID].values).fillna(1).values)
        sigma = np.maximum(pred_std, hist_std)
        results = []
        for i, sku in enumerate(latest[SKU_ID].values):
            mu = max(0, preds[i]); s = sigma[i]
            results.append({SKU_ID:sku, DEMAND_FORECAST:round(mu*horizon,1),
                FORECAST_LOWER:round(max(0,(mu-1.645*s)*horizon),1),
                FORECAST_UPPER:round((mu+1.645*s)*horizon,1),
                "daily_forecast":round(mu,2),"forecast_std":round(s,2),"model_type":"random_forest"})
        return pd.DataFrame(results)

    def get_metrics(self): return self.metrics_
    def get_feature_importance(self):
        fi = self.metrics_.get("feature_importance")
        return fi if fi is not None else pd.DataFrame({"feature":[],"importance":[]})


def run_demand_forecast(sales_features_df, horizon=FORECAST_HORIZON_DAYS):
    ma = moving_average_forecast(sales_features_df[[SKU_ID,DATE,QTY_SOLD]], horizon=horizon)
    ma = ma.rename(columns={DEMAND_FORECAST:"ma_forecast",FORECAST_LOWER:"ma_lower",FORECAST_UPPER:"ma_upper"})
    forecaster = DemandForecaster()
    forecaster.fit(sales_features_df)
    ml = forecaster.predict_sku_forecast(sales_features_df, horizon=horizon)
    result = ml.merge(ma[[SKU_ID,"ma_forecast","ma_lower","ma_upper"]], on=SKU_ID, how="left")
    if not forecaster._trained:
        result[DEMAND_FORECAST]  = result["ma_forecast"]
        result[FORECAST_LOWER]   = result["ma_lower"]
        result[FORECAST_UPPER]   = result["ma_upper"]
    result["primary_forecast"] = result[DEMAND_FORECAST]
    logger.info(f"Demand forecasts generated for {len(result)} SKUs")
    return result, forecaster
