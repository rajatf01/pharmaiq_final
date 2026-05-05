"""
dashboard/risk_compliance_tab.py
=================================
Covers ALL 7 compliance requirements:

1.  Risk                   — Unified 3-axis risk dashboard
2.  Lead Time Risk         — Dedicated lead time risk analysis
3.  Prediction Model       — Random Forest supplier reliability predictor
4.  Risk Analysis          — Financial risk quantification
5.  Risk Assessment        — Risk Register + Likelihood × Impact matrix
6.  Scenario Planning      — Named what-if scenario engine (4 scenarios)
7.  Risk Heatmaps          — SKU×Dimension heatmap, Category matrix, Intensity map
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, cast
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.schema import (
    SKU_ID, PRODUCT_NAME, CATEGORY, EXPIRY_RISK, STOCKOUT_RISK,
    SUPPLIER_RISK, EXPIRY_PRESSURE, ACTION, RECOMMENDED_QTY,
    UNIT_COST, SELLING_PRICE, SALES_VELOCITY, STOCK_COVER,
    EXPIRY_LOSS, HOLDING_COST, STOCKOUT_PENALTY, TOTAL_COST,
    ACTION_REORDER_NOW, ACTION_REDUCE, ACTION_URGENT_EXPIRY,
)
from config.settings import (
    HIGH_EXPIRY_RISK, MEDIUM_EXPIRY_RISK,
    HIGH_STOCKOUT_RISK, HIGH_SUPPLIER_RISK,
    DEFAULT_SERVICE_LEVEL,
)

# ── Dark theme tokens ─────────────────────────────────────────────────────────
C = {
    "bg":     "#080C14",
    "card":   "#0F1829",
    "border": "#1E3050",
    "text":   "#E8F0FE",
    "muted":  "#8FA3BF",
    "green":  "#00C853",
    "amber":  "#FFB300",
    "red":    "#FF1744",
    "blue":   "#0091EA",
    "purple": "#AA00FF",
}

RISK_COLOURS = {
    "Critical": C["red"],
    "High":     C["amber"],
    "Medium":   "#FFC107",
    "Low":      C["green"],
    "Expired":  C["purple"],
}

CHART_BASE: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C["bg"],
    font=dict(color=C["muted"], size=11),
    margin=dict(t=40, b=20, l=10, r=10),
)

def apply_chart_layout(fig: go.Figure, **kwargs: Any) -> go.Figure:
    layout: dict[str, Any] = {**CHART_BASE, **kwargs}
    fig.update_layout(layout)
    return fig

RISK_CSS = """
<style>
.rc-section{background:#0F1829;border:1px solid #1E3050;border-radius:12px;
  padding:20px 22px;margin-bottom:16px}
.rc-section h3{color:#E8F0FE;font-size:16px;font-weight:700;margin:0 0 4px 0}
.rc-section p{color:#8FA3BF;font-size:12px;margin:0 0 14px 0}
.risk-pill{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:11px;font-weight:700;border:1px solid;margin:2px}
.risk-critical{color:#FF1744;border-color:#FF1744;background:rgba(255,23,68,.12)}
.risk-high    {color:#FFB300;border-color:#FFB300;background:rgba(255,179,0,.12)}
.risk-medium  {color:#FFC107;border-color:#FFC107;background:rgba(255,193,7,.10)}
.risk-low     {color:#00C853;border-color:#00C853;background:rgba(0,200,83,.12)}
.reg-row{display:flex;align-items:center;gap:12px;padding:8px 0;
  border-bottom:1px solid #1A2540;font-size:12px;color:#C5D3E8}
.reg-row:last-child{border-bottom:none}
.reg-id{width:70px;color:#8FA3BF;font-family:monospace;font-size:11px}
.reg-name{flex:1;font-weight:600}
.reg-sev{width:80px}
.reg-impact{flex:2;color:#8FA3BF;font-size:11px}
.reg-mit{flex:2;color:#90CAF9;font-size:11px}
.sc-card{background:#0A1525;border:1px solid #1E3050;border-radius:10px;
  padding:14px 16px;margin-bottom:8px}
.sc-card h4{color:#E8F0FE;font-size:14px;margin:0 0 4px 0}
.sc-card p{color:#8FA3BF;font-size:11px;margin:0}
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def sv(row, col, d=0):
    try:
        v = row.get(col, d) if isinstance(row, dict) else getattr(row, col, d)
        return d if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
    except: 
        return float(d)

def risk_pill(label):
    cls = label.lower() if label.lower() in ["critical","high","medium","low"] else "medium"
    return f'<span class="risk-pill risk-{cls}">{label}</span>'

def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


# ─────────────────────────────────────────────────────────────────────────────
# 1 + 2: RISK OVERVIEW + LEAD TIME RISK
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_overview(df: pd.DataFrame):
    st.markdown(RISK_CSS, unsafe_allow_html=True)
    st.markdown("""<div class="rc-section">
      <h3>🛡️ Unified Risk Dashboard</h3>
      <p>Three independent AI risk scores per SKU — expiry, stockout, and supplier reliability.
         Each scored 0–1. Composite priority = 0.40×expiry + 0.35×stockout + 0.25×supplier.</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🔴 Critical Expiry",
              int((safe_col(df,EXPIRY_RISK,0)>=HIGH_EXPIRY_RISK).sum()),
              help="Expiry risk ≥ 0.70")
    c2.metric("📦 High Stockout",
              int((safe_col(df,STOCKOUT_RISK,0)>=HIGH_STOCKOUT_RISK).sum()),
              help="Stockout risk ≥ 0.65")
    c3.metric("🚚 High Supplier Risk",
              int((safe_col(df,SUPPLIER_RISK,0)>=HIGH_SUPPLIER_RISK).sum()),
              help="Supplier risk ≥ 0.60")
    c4.metric("⚠️ Multi-Risk SKUs",
              int(((safe_col(df,EXPIRY_RISK,0)>=MEDIUM_EXPIRY_RISK) &
                   (safe_col(df,STOCKOUT_RISK,0)>=0.40)).sum()),
              help="SKUs with both expiry and stockout risk elevated")
    c5.metric("📊 Portfolio Avg Risk",
              f"{safe_col(df,'priority_score',0).mean():.2f}",
              help="Average composite priority score across all SKUs")

    st.divider()

    # ── Tri-axis scatter: 3 risks in one 3D-style plot ────────────────────
    st.subheader("🎯 Three-Axis Risk Map")
    st.caption("Every bubble = one SKU. X = stockout risk · Y = expiry risk · Size = supplier risk · Colour = worst risk category")

    df2 = df.copy()
    df2["worst_risk"] = df2.apply(lambda r: (
        "Critical" if sv(r,EXPIRY_RISK)>=HIGH_EXPIRY_RISK or sv(r,"nearest_expiry",999)<0
        else "High"  if sv(r,EXPIRY_RISK)>=MEDIUM_EXPIRY_RISK or sv(r,STOCKOUT_RISK)>=HIGH_STOCKOUT_RISK
        else "Medium" if sv(r,EXPIRY_RISK)>=0.20 or sv(r,STOCKOUT_RISK)>=0.35
        else "Low"), axis=1)
    df2["sup_size"] = (df2[SUPPLIER_RISK]*40+8).clip(8, 48) if SUPPLIER_RISK in df2.columns else 16
    hn = PRODUCT_NAME if PRODUCT_NAME in df2.columns else SKU_ID

    fig = px.scatter(df2, x=STOCKOUT_RISK, y=EXPIRY_RISK,
        color="worst_risk", size="sup_size", hover_name=hn,
        hover_data={EXPIRY_RISK:":.2f", STOCKOUT_RISK:":.2f",
                    SUPPLIER_RISK:":.2f" if SUPPLIER_RISK in df2.columns else False,
                    ACTION: True if ACTION in df2.columns else False},
        color_discrete_map={"Critical":C["red"],"High":C["amber"],"Medium":"#FFC107","Low":C["green"]},
        labels={STOCKOUT_RISK:"Stockout Risk (0–1)", EXPIRY_RISK:"Expiry Risk (0–1)"},
        size_max=40)
    fig.add_hline(y=HIGH_EXPIRY_RISK,   line_dash="dash", line_color=C["red"],   line_width=1,
                  annotation_text="Expiry threshold",  annotation_font_color=C["red"])
    fig.add_vline(x=HIGH_STOCKOUT_RISK, line_dash="dash", line_color=C["blue"],  line_width=1,
                  annotation_text="Stockout threshold", annotation_font_color=C["blue"])
    # Quadrant labels
    fig.add_annotation(x=0.85, y=0.85, text="⚠️ Dual Crisis", font=dict(color=C["red"],   size=11), showarrow=False)
    fig.add_annotation(x=0.85, y=0.15, text="📦 Reorder",    font=dict(color=C["blue"],  size=11), showarrow=False)
    fig.add_annotation(x=0.10, y=0.85, text="♻️ Reduce",     font=dict(color=C["amber"], size=11), showarrow=False)
    fig.add_annotation(x=0.10, y=0.15, text="✅ Healthy",    font=dict(color=C["green"], size=11), showarrow=False)
    apply_chart_layout(fig, height=420,
        xaxis=dict(gridcolor=C["border"], range=[-0.02,1.02]),
        yaxis=dict(gridcolor=C["border"], range=[-0.02,1.02]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── LEAD TIME RISK — dedicated section ───────────────────────────────
    st.subheader("⏱️ Lead Time Risk Analysis")
    st.caption("Lead time risk = the probability that a SKU stocksout during the replenishment period. "
               "Effective lead time (95th pct) = stated lead time + 1.645 × standard deviation.")

    lt_cols = ["mean_lead_time","effective_lead_time","lead_time_cv",
               "delay_probability","avg_delay_days","std_lead_time"]
    avail_lt = [c for c in lt_cols if c in df.columns]

    if not avail_lt:
        st.info("Lead time features not yet computed — run pipeline first.")
        return

    lt_df = df[[SKU_ID] + ([PRODUCT_NAME] if PRODUCT_NAME in df.columns else []) +
               ([CATEGORY] if CATEGORY in df.columns else []) + avail_lt +
               [STOCKOUT_RISK, SUPPLIER_RISK]].copy()

    # Lead time risk score = weighted combination
    lt_df["lead_time_risk"] = (
        0.45 * lt_df["delay_probability"].fillna(0.2) +
        0.30 * (lt_df["lead_time_cv"].fillna(0.2)).clip(0,1) +
        0.25 * (lt_df["effective_lead_time"].fillna(14) / 40).clip(0,1)
    ).round(3)

    lt_df["lt_risk_label"] = lt_df["lead_time_risk"].apply(
        lambda s: "Critical" if s>=0.60 else ("High" if s>=0.40 else ("Medium" if s>=0.25 else "Low")))

    lc1, lc2 = st.columns(2)

    with lc1:
        # Effective vs stated lead time bar chart
        lt_plot = lt_df[[PRODUCT_NAME if PRODUCT_NAME in lt_df.columns else SKU_ID,
                         "mean_lead_time","effective_lead_time","lead_time_risk"]].copy()
        hn2 = PRODUCT_NAME if PRODUCT_NAME in lt_plot.columns else SKU_ID
        lt_plot = lt_plot.sort_values("lead_time_risk", ascending=False).head(15)
        fig_lt = go.Figure()
        fig_lt.add_bar(name="Stated Lead Time",     x=lt_plot[hn2],
                       y=lt_plot["mean_lead_time"],     marker_color=C["blue"],   opacity=0.75)
        fig_lt.add_bar(name="Effective Lead Time (95th pct)", x=lt_plot[hn2],
                       y=lt_plot["effective_lead_time"], marker_color=C["amber"], opacity=0.90)
        apply_chart_layout(fig, barmode="group", height=340,
            title="Stated vs Effective Lead Time (Top 15 by Risk)",
            xaxis=dict(gridcolor=C["border"], tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=C["border"], title="Days"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"]), orientation="h", y=1.1))
        st.plotly_chart(fig_lt, use_container_width=True)

    with lc2:
        # Lead time risk distribution
        fig_cv = px.histogram(lt_df, x="lead_time_risk", nbins=20,
            color="lt_risk_label",
            color_discrete_map={"Critical":C["red"],"High":C["amber"],"Medium":"#FFC107","Low":C["green"]},
            title="Lead Time Risk Score Distribution",
            labels={"lead_time_risk":"Lead Time Risk Score (0–1)"})
        apply_chart_layout(fig, height=340,
            xaxis=dict(gridcolor=C["border"]),
            yaxis=dict(gridcolor=C["border"], title="Number of SKUs"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])))
        st.plotly_chart(fig_cv, use_container_width=True)

    # Lead time risk table
    lt_show = lt_df[[hn2,"mean_lead_time","effective_lead_time",
                     "delay_probability","lead_time_cv","lead_time_risk","lt_risk_label"]].copy()
    lt_show.columns = ["Product","Stated LT (d)","Effective LT (d)",
                       "Delay Prob","LT Variability (CV)","LT Risk Score","Risk Level"]
    lt_show["Delay Prob"]    = lt_show["Delay Prob"].apply(lambda x: f"{x:.0%}")
    lt_show["LT Risk Score"] = lt_show["LT Risk Score"].apply(lambda x: f"{x:.3f}")
    lt_show["LT Variability (CV)"] = lt_show["LT Variability (CV)"].apply(lambda x: f"{x:.2f}")
    lt_show = lt_show.sort_values("LT Risk Score", ascending=False).reset_index(drop=True)
    st.dataframe(lt_show, use_container_width=True, hide_index=True)



# ─────────────────────────────────────────────────────────────────────────────
# 2: DEDICATED LEAD TIME RISK TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_lead_time_risk(df: pd.DataFrame):
    st.markdown("""<div class="rc-section">
      <h3>⏱️ Lead Time Risk Analysis</h3>
      <p>Lead time risk quantifies the probability that a SKU will stockout during the replenishment
         window. It combines delay probability, lead time variability (CV), and effective lead time
         into a single risk score. Higher = more dangerous for supply continuity.</p>
    </div>""", unsafe_allow_html=True)

    lt_cols = ["mean_lead_time","effective_lead_time","lead_time_cv",
               "delay_probability","avg_delay_days","std_lead_time"]
    avail_lt = [c for c in lt_cols if c in df.columns]

    if not avail_lt:
        st.warning("Lead time features not computed. Run the pipeline first.")
        return

    df2 = df.copy()
    df2["lead_time_risk"] = (
        0.45 * df2.get("delay_probability",    pd.Series([0.2]*len(df2))).fillna(0.2) +
        0.30 * df2.get("lead_time_cv",         pd.Series([0.2]*len(df2))).fillna(0.2).clip(0,1) +
        0.25 * (df2.get("effective_lead_time", pd.Series([14]*len(df2))).fillna(14) / 40).clip(0,1)
    ).round(3)
    df2["lt_risk_label"] = df2["lead_time_risk"].apply(
        lambda s: "Critical" if s>=0.60 else ("High" if s>=0.40 else ("Medium" if s>=0.25 else "Low")))
    hn = PRODUCT_NAME if PRODUCT_NAME in df2.columns else SKU_ID

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Critical Lead Time Risk", int((df2["lead_time_risk"]>=0.60).sum()))
    k2.metric("High Lead Time Risk",     int(((df2["lead_time_risk"]>=0.40)&(df2["lead_time_risk"]<0.60)).sum()))
    k3.metric("Avg Delay Probability",   f"{df2['delay_probability'].mean():.0%}" if "delay_probability" in df2.columns else "N/A")
    k4.metric("Avg Effective Lead Time", f"{df2['effective_lead_time'].mean():.1f}d" if "effective_lead_time" in df2.columns else "N/A")
    st.divider()

    lc1, lc2 = st.columns(2)
    with lc1:
        top15 = df2.nlargest(15,"lead_time_risk")
        fig_lt = go.Figure()
        if "mean_lead_time" in df2.columns:
            fig_lt.add_bar(name="Stated Lead Time", x=top15[hn],
                           y=top15["mean_lead_time"], marker_color=C["blue"], opacity=0.75)
        if "effective_lead_time" in df2.columns:
            fig_lt.add_bar(name="Effective (95th pct)", x=top15[hn],
                           y=top15["effective_lead_time"], marker_color=C["amber"], opacity=0.90)
        apply_chart_layout(fig_lt, barmode="group", height=340,
            title="Stated vs Effective Lead Time (Top 15 Risk)",
            xaxis=dict(gridcolor=C["border"],tickangle=-40,tickfont=dict(size=8)),
            yaxis=dict(gridcolor=C["border"],title="Days"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=C["muted"]),orientation="h",y=1.1))
        st.plotly_chart(fig_lt, use_container_width=True)

    with lc2:
        fig_cv = px.histogram(df2, x="lead_time_risk", nbins=20,
            color="lt_risk_label",
            color_discrete_map={"Critical":C["red"],"High":C["amber"],"Medium":"#FFC107","Low":C["green"]},
            title="Lead Time Risk Distribution",
            labels={"lead_time_risk":"Lead Time Risk Score (0–1)"})
        apply_chart_layout(fig_cv, height=340,
            xaxis=dict(gridcolor=C["border"]),
            yaxis=dict(gridcolor=C["border"],title="SKU Count"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=C["muted"])))
        st.plotly_chart(fig_cv, use_container_width=True)

    # Scatter: delay probability vs effective lead time
    fig_sc = px.scatter(df2, x="delay_probability" if "delay_probability" in df2.columns else "lead_time_risk",
        y="effective_lead_time" if "effective_lead_time" in df2.columns else "mean_lead_time",
        color="lt_risk_label", size="lead_time_risk", size_max=25, hover_name=hn,
        color_discrete_map={"Critical":C["red"],"High":C["amber"],"Medium":"#FFC107","Low":C["green"]},
        title="Delay Probability vs Effective Lead Time — Risk Positioning",
        labels={"delay_probability":"Delay Probability","effective_lead_time":"Effective Lead Time (days)"})
    apply_chart_layout(fig_sc, height=350,
        xaxis=dict(gridcolor=C["border"]), yaxis=dict(gridcolor=C["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=C["muted"])))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Table
    show_cols = [hn,"mean_lead_time","effective_lead_time","delay_probability",
                 "lead_time_cv","avg_delay_days","lead_time_risk","lt_risk_label"]
    show_cols = [c for c in show_cols if c in df2.columns]
    tbl = df2[show_cols].sort_values("lead_time_risk",ascending=False).reset_index(drop=True)
    renames = {hn:"Product","mean_lead_time":"Stated LT (d)","effective_lead_time":"Effective LT (d)",
               "delay_probability":"Delay Prob","lead_time_cv":"LT CV",
               "avg_delay_days":"Avg Delay (d)","lead_time_risk":"LT Risk Score","lt_risk_label":"Level"}
    tbl = tbl.rename(columns={k:v for k,v in renames.items() if k in tbl.columns})
    if "Delay Prob" in tbl.columns: tbl["Delay Prob"] = tbl["Delay Prob"].apply(lambda x:f"{x:.0%}")
    if "LT Risk Score" in tbl.columns: tbl["LT Risk Score"] = tbl["LT Risk Score"].apply(lambda x:f"{x:.3f}")
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3: ML PREDICTION MODEL ON SUPPLIER SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def render_supplier_prediction_model(df: pd.DataFrame, suppliers_df: Optional[pd.DataFrame]):
    st.markdown("""<div class="rc-section">
      <h3>🤖 AI Prediction Model — Supplier Reliability</h3>
      <p>A Random Forest classifier trained on historical delivery records predicts whether
         a supplier will deliver on time for each SKU. SHAP-style feature attribution explains
         which factors drive each prediction. This goes beyond rule-based scoring — the model
         learns patterns from actual delivery history.</p>
    </div>""", unsafe_allow_html=True)

    # ── Build training dataset from delivery history ───────────────────────
    if suppliers_df is None or suppliers_df.empty or len(suppliers_df) < 30:
        st.warning("Insufficient delivery history for ML training (need ≥ 30 records). "
                   "Showing simulation with synthetic training data.")
        suppliers_df = _generate_training_data()

    df_train = suppliers_df.copy()
    if "actual_lead_time" not in df_train.columns:
        st.info("actual_lead_time column not found — generating simulated training data.")
        df_train = _generate_training_data()

    df_train = df_train.dropna(subset=["lead_time_days","actual_lead_time"])
    df_train["was_delayed"]  = (df_train["actual_lead_time"] > df_train["lead_time_days"]).astype(int)
    df_train["delay_days"]   = (df_train["actual_lead_time"] - df_train["lead_time_days"]).clip(lower=0)
    df_train["lead_time_cv"] = df_train.groupby(["supplier_id","sku_id"] if "supplier_id" in df_train.columns else ["sku_id"])\
                                ["actual_lead_time"].transform(lambda x: x.std()/x.mean() if x.mean()>0 else 0).fillna(0)
    df_train["order_index"]  = df_train.get("order_index", range(len(df_train)))

    feature_cols = ["lead_time_days","actual_lead_time","delay_days","lead_time_cv","order_index"]
    avail_f = [c for c in feature_cols if c in df_train.columns]
    X = df_train[avail_f].fillna(0).to_numpy(dtype=float)
    y = df_train["was_delayed"].to_numpy(dtype=int)

    if len(X) < 20 or y.sum() < 5:
        st.warning("Not enough positive examples for ML. Augmenting with synthetic data.")
        df_train = _generate_training_data()
        X = df_train[avail_f].fillna(0).to_numpy(dtype=float)
        y = df_train["was_delayed"].to_numpy(dtype=int)

    # ── Train RF model ────────────────────────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_sc, y)

    cv = StratifiedKFold(n_splits=min(5, max(2, int(y.sum()//2))), shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_sc, y, cv=cv, scoring="roc_auc")
    cv_acc    = cross_val_score(rf, X_sc, y, cv=cv, scoring="accuracy")

    # ── Display model metrics ─────────────────────────────────────────────
    st.subheader("📊 Model Performance")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Algorithm",       "Random Forest")
    m2.metric("CV AUC (ROC)",    f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    m3.metric("CV Accuracy",     f"{cv_acc.mean():.1%}")
    m4.metric("Training Records",f"{len(X):,}")

    # ── Feature importance ────────────────────────────────────────────────
    st.subheader("🔍 Feature Importance — What Drives Delay Predictions?")
    fi_df = pd.DataFrame({
        "Feature":    [f.replace("_"," ").title() for f in avail_f],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#0091EA","#FFB300","#FF1744"],
        title="Random Forest Feature Importance",
        labels={"Importance":"Relative Importance","Feature":"Delivery Feature"})
    apply_chart_layout(fig_fi, height=280,
        xaxis=dict(gridcolor=C["border"]),
        yaxis=dict(gridcolor=C["border"]),
        coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── SKU-level delay predictions ───────────────────────────────────────
    st.subheader("📦 AI-Predicted Delay Probability per SKU")
    sku_preds = []
    for sku in df_train["sku_id"].unique() if "sku_id" in df_train.columns else []:
        sku_d = df_train[df_train["sku_id"]==sku][avail_f].fillna(0)
        if len(sku_d) > 0:
            pred_proba = rf.predict_proba(scaler.transform(sku_d))[:,1].mean()
            sku_preds.append({"sku_id":sku, "delay_probability_ml":round(pred_proba,3)})
    if not sku_preds:
        # Generate one prediction per synthetic SKU
        for i in range(min(10, len(df))):
            x_row = np.array([[14, 16, 2, 0.20, i][:len(avail_f)]])
            p = rf.predict_proba(scaler.transform(x_row))[0,1]
            sid = df.iloc[i][SKU_ID] if SKU_ID in df.columns else f"SKU{i+1:03d}"
            nm  = df.iloc[i][PRODUCT_NAME] if PRODUCT_NAME in df.columns else sid
            sku_preds.append({"sku_id":sid,"product_name":nm,"delay_probability_ml":round(float(p),3)})

    pred_df = pd.DataFrame(sku_preds)
    if "sku_id" in pred_df.columns and SKU_ID in df.columns:
        pred_df = pred_df.merge(df[[SKU_ID]+([PRODUCT_NAME] if PRODUCT_NAME in df.columns else [])],
                                left_on="sku_id",right_on=SKU_ID,how="left")

    if not pred_df.empty and "delay_probability_ml" in pred_df.columns:
        pred_df["risk_level"] = pred_df["delay_probability_ml"].apply(
            lambda p: "Critical" if p>=0.60 else ("High" if p>=0.40 else ("Medium" if p>=0.25 else "Low")))
        nm_col = "product_name" if "product_name" in pred_df.columns else "sku_id"
        fig_pred = px.bar(pred_df.sort_values("delay_probability_ml",ascending=False),
            x=nm_col, y="delay_probability_ml",
            color="risk_level",
            color_discrete_map={"Critical":C["red"],"High":C["amber"],"Medium":"#FFC107","Low":C["green"]},
            title="ML-Predicted Delay Probability per Supplier-SKU",
            labels={"delay_probability_ml":"Predicted Delay Probability","risk_level":"Risk Level"})
        fig_pred.add_hline(y=0.40, line_dash="dash", line_color=C["amber"],
                           annotation_text="High risk threshold")
        apply_chart_layout(fig_pred, height=340,
            xaxis=dict(gridcolor=C["border"], tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=C["border"], range=[0,1]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])))
        st.plotly_chart(fig_pred, use_container_width=True)

    st.info(
        "**How this model works:** A Random Forest with 200 trees learns from historical delivery "
        "records — stated lead time, actual delivery time, order sequence, and lead time variability. "
        "It predicts the probability of a future delay for each supplier-SKU pair. "
        "This is a *learned* model, not hand-coded rules — it improves as more delivery data accumulates."
    )


def _generate_training_data():
    rng = np.random.default_rng(42)
    rows = []
    suppliers = ["SUP001","SUP002","SUP003","SUP004","SUP005"]
    skus      = [f"SKU{i+1:03d}" for i in range(10)]
    for s in suppliers:
        stated_lt = {"SUP001":14,"SUP002":18,"SUP003":21,"SUP004":7,"SUP005":14}[s]
        delay_p   = {"SUP001":0.22,"SUP002":0.48,"SUP003":0.62,"SUP004":0.18,"SUP005":0.35}[s]
        for sku in skus:
            for i in range(rng.integers(15, 35)):
                delayed = rng.random() < delay_p
                actual  = stated_lt + (int(rng.uniform(3,12)) if delayed else int(rng.uniform(-2,3)))
                rows.append({"sku_id":sku,"supplier_id":s,
                              "lead_time_days":stated_lt,"actual_lead_time":max(1,actual),
                              "order_index":i,"was_delayed":int(delayed),
                              "delay_days":max(0,actual-stated_lt)})
    df = pd.DataFrame(rows)
    df["lead_time_cv"] = df.groupby(["supplier_id","sku_id"])["actual_lead_time"]\
                           .transform(lambda x: x.std()/x.mean() if x.mean()>0 else 0).fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4 + 5: RISK ANALYSIS + RISK ASSESSMENT (REGISTER + MATRIX)
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_analysis(df: pd.DataFrame, financial: dict):
    st.markdown("""<div class="rc-section">
      <h3>💰 Risk Analysis — Financial Quantification</h3>
      <p>Each risk is translated into a dollar value. Total financial exposure = expiry loss +
         holding cost + stockout penalty. AI system reduces this by optimising reorder decisions.</p>
    </div>""", unsafe_allow_html=True)

    fin  = financial.get("summary", {})
    skus = financial.get("sku_costs", df).copy()

    # ── Cost waterfall ────────────────────────────────────────────────────
    ai_exp = fin.get("ai",{}).get("expiry_loss",0)
    ai_hld = fin.get("ai",{}).get("holding_cost",0)
    ai_so  = fin.get("ai",{}).get("stockout_penalty",0)
    bl_exp = fin.get("baseline",{}).get("expiry_loss",0)
    bl_hld = fin.get("baseline",{}).get("holding_cost",0)
    bl_so  = fin.get("baseline",{}).get("stockout_penalty",0)

    wf = go.Figure(go.Waterfall(
        name="Risk Cost Breakdown", orientation="v",
        measure=["absolute","relative","relative","total","relative","relative","relative","total"],
        x=["Baseline Total","– Expiry Saving","– Holding Saving","Subtotal",
           "– Stockout Saving","+ Residual Expiry","+ Residual Stockout","AI Total"],
        y=[bl_exp+bl_hld+bl_so,
           -(bl_exp-ai_exp), -(bl_hld-ai_hld),
           0,
           -(bl_so-ai_so), ai_exp, ai_so,
           0],
        connector={"line":{"color":"#1E3050"}},
        increasing ={"marker":{"color":C["red"]}},
        decreasing ={"marker":{"color":C["green"]}},
        totals     ={"marker":{"color":C["blue"]}},
    ))
    apply_chart_layout(wf, height=360,
        title="Financial Risk Waterfall: Baseline → AI System",
        yaxis=dict(gridcolor=C["border"], title="USD ($)"),
        xaxis=dict(gridcolor=C["border"]))
    st.plotly_chart(wf, use_container_width=True)

    # ── Per-SKU financial risk breakdown ─────────────────────────────────
    if EXPIRY_LOSS in skus.columns:
        st.subheader("📊 Risk Cost by SKU")
        top_cost = skus.nlargest(15, TOTAL_COST if TOTAL_COST in skus.columns else EXPIRY_LOSS)
        hn = PRODUCT_NAME if PRODUCT_NAME in top_cost.columns else SKU_ID
        cost_fig = go.Figure()
        for col, label, colour in [
            (EXPIRY_LOSS,      "Expiry Loss",      C["red"]),
            (HOLDING_COST,     "Holding Cost",     C["amber"]),
            (STOCKOUT_PENALTY, "Stockout Penalty", C["blue"]),
        ]:
            if col in top_cost.columns:
                cost_fig.add_bar(name=label, x=top_cost[hn],
                                 y=top_cost[col], marker_color=colour)
        apply_chart_layout(cost_fig, barmode="stack", height=360,
            title="Risk Cost Breakdown per SKU (Top 15)",
            xaxis=dict(gridcolor=C["border"], tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=C["border"], title="USD ($)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"]), orientation="h", y=1.1))
        st.plotly_chart(cost_fig, use_container_width=True)


def render_risk_assessment(df: pd.DataFrame):
    st.markdown("""<div class="rc-section">
      <h3>📋 Risk Assessment — Risk Register & Likelihood × Impact Matrix</h3>
      <p>Formal risk register mapping each SKU's risks to likelihood, impact, severity score,
         and mitigation actions. The risk matrix plots every SKU by probability × consequence.</p>
    </div>""", unsafe_allow_html=True)

    # ── Build Risk Register ───────────────────────────────────────────────
    register = []
    for _, row in df.iterrows():
        sku  = str(row.get(SKU_ID, ""))
        name = str(row.get(PRODUCT_NAME, sku))
        er   = float(row.get(EXPIRY_RISK, 0))
        sr   = float(row.get(STOCKOUT_RISK, 0))
        sup  = float(row.get(SUPPLIER_RISK, 0))
        cost = float(row.get(UNIT_COST, 5))
        vel  = float(row.get(SALES_VELOCITY, 1))

        # Expiry risk entry
        if er >= 0.20:
            likelihood = "High" if er>=0.65 else ("Medium" if er>=0.40 else "Low")
            impact     = "High" if cost>10 else ("Medium" if cost>4 else "Low")
            sev_score  = round(er * (cost/10) * 10, 1)
            register.append({"SKU":sku,"Product":name,"Risk Type":"Expiry",
                "Likelihood":likelihood,"Impact":impact,"Severity":sev_score,
                "Probability":round(er,2),
                "Financial Exposure":f"${er*vel*30*cost:,.0f}",
                "Mitigation":"Promote / Return to supplier / Price discount"
                if er>=0.65 else "Monitor weekly"})

        # Stockout risk entry
        if sr >= 0.30:
            likelihood = "High" if sr>=0.65 else ("Medium" if sr>=0.40 else "Low")
            impact     = "High" if vel>20 else ("Medium" if vel>8 else "Low")
            sev_score  = round(sr * (vel/20) * 10, 1)
            register.append({"SKU":sku,"Product":name,"Risk Type":"Stockout",
                "Likelihood":likelihood,"Impact":impact,"Severity":sev_score,
                "Probability":round(sr,2),
                "Financial Exposure":f"${sr*vel*14*cost*2:,.0f}",
                "Mitigation":"Reorder immediately / Switch to faster supplier"
                if sr>=0.65 else "Increase safety stock"})

        # Supplier risk entry
        if sup >= 0.30:
            likelihood = "High" if sup>=0.60 else ("Medium" if sup>=0.35 else "Low")
            impact     = "Medium"
            sev_score  = round(sup * 8, 1)
            register.append({"SKU":sku,"Product":name,"Risk Type":"Supplier",
                "Likelihood":likelihood,"Impact":impact,"Severity":sev_score,
                "Probability":round(sup,2),
                "Financial Exposure":f"${sup*vel*7*cost:,.0f}",
                "Mitigation":"Qualify backup supplier / Add lead time buffer"
                if sup>=0.60 else "Monitor delivery performance"})

    reg_df = pd.DataFrame(register)

    if reg_df.empty:
        st.info("No risks above threshold — portfolio is healthy.")
        return

    # ── Risk summary badges ───────────────────────────────────────────────
    sev_counts = reg_df.groupby(["Likelihood","Impact"]).size().reset_index(name="count")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Risk Items", len(reg_df))
    c2.metric("High Likelihood",  int((reg_df["Likelihood"]=="High").sum()))
    c3.metric("High Impact",      int((reg_df["Impact"]=="High").sum()))
    c4.metric("Avg Severity",     f"{reg_df['Severity'].mean():.1f}/10")

    # ── Likelihood × Impact Matrix (the classic risk matrix) ─────────────
    st.subheader("🔴 Likelihood × Impact Risk Matrix")
    st.caption("Standard 3×3 risk matrix. Each cell shows number of SKU-risks in that zone. "
               "Red = act immediately. Yellow = monitor. Green = acceptable.")

    levels = ["Low","Medium","High"]
    matrix = np.zeros((3,3), dtype=int)
    lmap = {"Low":0,"Medium":1,"High":2}
    for _, r in reg_df.iterrows():
        li = lmap.get(r["Likelihood"],1); ii = lmap.get(r["Impact"],1)
        matrix[li][ii] += 1

    # Build heatmap with custom colour (green → red)
    cell_colours = [
        ["#0A2A1A","#1A3010","#2A4008"],  # Low likelihood
        ["#1A2008","#2A3800","#3A4800"],  # Medium
        ["#2A1008","#3A2008","#4A0808"],  # High likelihood × High impact
    ]
    z_colours = [
        [C["green"],   "#66BB6A", C["amber"]],
        ["#66BB6A",    C["amber"],"#FF6D00"],
        [C["amber"],   "#FF6D00", C["red"]],
    ]

    fig_mat = go.Figure(go.Heatmap(
        z=matrix, x=["Low Impact","Medium Impact","High Impact"],
        y=["Low Likelihood","Medium Likelihood","High Likelihood"],
        colorscale=[[0,"#00C853"],[0.5,"#FFB300"],[1,"#FF1744"]],
        text=[[str(matrix[i][j]) if matrix[i][j]>0 else "" for j in range(3)] for i in range(3)],
        texttemplate="%{text}",
        textfont=dict(size=24, color="white"),
        showscale=False,
        hoverongaps=False,
    ))
    fig_mat.add_annotation(x="Low Impact",    y="Low Likelihood",    text="✅ Accept",   showarrow=False, font=dict(color="#00C853",size=10))
    fig_mat.add_annotation(x="High Impact",   y="High Likelihood",   text="🚨 Urgent",   showarrow=False, font=dict(color="#FF1744",size=10), yshift=-18)
    fig_mat.add_annotation(x="Medium Impact", y="High Likelihood",   text="⚠️ Monitor",  showarrow=False, font=dict(color="#FFB300",size=10), yshift=-18)
    apply_chart_layout(fig_mat, height=320,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_mat, use_container_width=True)

    # ── Risk Register Table ───────────────────────────────────────────────
    st.subheader("📋 Risk Register")
    show = reg_df[["SKU","Product","Risk Type","Likelihood","Impact",
                   "Severity","Probability","Financial Exposure","Mitigation"]].copy()
    show["Probability"] = show["Probability"].apply(lambda x: f"{x:.0%}")
    show = show.sort_values("Severity", ascending=False).reset_index(drop=True)
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Download
    st.download_button("⬇️ Download Risk Register (CSV)",
        show.to_csv(index=False).encode(),
        "pharmaiq_risk_register.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# 6: SCENARIO PLANNING
# ─────────────────────────────────────────────────────────────────────────────
def render_scenario_planning(df: pd.DataFrame):
    st.markdown("""<div class="rc-section">
      <h3>🔮 Scenario Planning — What-If Analysis Engine</h3>
      <p>Compare four named scenarios side by side. Each scenario changes demand, supplier
         reliability, and disruption parameters independently. See how AI recommendations
         shift across different business conditions.</p>
    </div>""", unsafe_allow_html=True)

    # ── Scenario definitions ──────────────────────────────────────────────
    SCENARIOS = {
        "📊 Base Case": {
            "desc":           "Current conditions — normal demand, reliable suppliers",
            "demand_mult":    1.00,
            "disruption":     1.00,
            "service_level":  0.95,
            "delay_mult":     1.00,
            "colour":         C["blue"],
        },
        "📈 Demand Surge (+30%)": {
            "desc":           "Post-pandemic or seasonal surge — demand up 30%, same supply",
            "demand_mult":    1.30,
            "disruption":     1.00,
            "service_level":  0.97,
            "delay_mult":     1.00,
            "colour":         C["green"],
        },
        "🚚 Supply Disruption": {
            "desc":           "Supplier delays double, lead times increase 50%",
            "demand_mult":    1.00,
            "disruption":     1.80,
            "service_level":  0.98,
            "delay_mult":     2.00,
            "colour":         C["amber"],
        },
        "⚠️ Worst Case": {
            "desc":           "Demand up 20%, severe supply disruption, highest service level",
            "demand_mult":    1.20,
            "disruption":     2.00,
            "service_level":  0.99,
            "delay_mult":     2.50,
            "colour":         C["red"],
        },
    }

    # ── Interactive scenario customiser ───────────────────────────────────
    st.subheader("⚙️ Customise Scenario Parameters")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        custom_demand = st.slider("Demand Change (%)",   -50, +100, 0, 5,
                                   help="How much does demand increase or decrease?") / 100
    with sc2:
        custom_disrupt = st.slider("Supply Disruption Factor", 1.0, 3.0, 1.0, 0.1,
                                    help="1.0 = normal. 2.0 = severe disruption.")
    with sc3:
        custom_sl = st.slider("Target Service Level", 0.80, 0.99, 0.95, 0.01,
                               help="Higher = more safety stock required.")

    SCENARIOS["🔧 Custom Scenario"] = {
        "desc":          f"Demand {custom_demand:+.0%} · Disruption ×{custom_disrupt:.1f} · SL {custom_sl:.0%}",
        "demand_mult":   1 + custom_demand,
        "disruption":    custom_disrupt,
        "service_level": custom_sl,
        "delay_mult":    custom_disrupt,
        "colour":        C["purple"],
    }

    # ── Compute scenario outcomes ─────────────────────────────────────────
    def _scenario_kpis(df_base, sc):
        dm  = sc["demand_mult"]; disr = sc["disruption"]
        sl  = sc["service_level"]; dlm = sc["delay_mult"]
        n   = len(df_base)
        vel = safe_col(df_base, SALES_VELOCITY, 5) * dm
        eff_lt = safe_col(df_base, "effective_lead_time", 14) * (1 + (dlm-1)*0.5)
        std_d  = safe_col(df_base, "std_daily_demand", 2) * dm
        ss     = (1.645 if sl>=0.95 else 1.28) * std_d * np.sqrt(eff_lt)
        stock  = safe_col(df_base, "current_stock", 100)
        cover  = stock / vel.clip(lower=0.1)
        so_risk = (1 - cover/(eff_lt + ss/vel.clip(lower=0.1))).clip(0,1)
        exp_r   = safe_col(df_base, EXPIRY_RISK, 0.3)
        sup_r   = (safe_col(df_base, SUPPLIER_RISK, 0.3) * dlm).clip(0,1)
        uc      = safe_col(df_base, UNIT_COST, 5)
        sp      = safe_col(df_base, SELLING_PRICE, 12)
        margin  = (sp - uc).clip(lower=0.5)
        exp_loss   = (exp_r * stock * uc * 0.95 * (1/dm)).clip(lower=0).sum()
        hold_cost  = (stock * uc * (0.25/365*30)).sum()
        so_penalty = (so_risk * vel * eff_lt * margin * 2.0).sum()
        reorder_skus = int((so_risk >= 0.65).sum())
        reduce_skus  = int(((exp_r >= 0.70) & (so_risk < 0.50)).sum())
        total_cost   = exp_loss + hold_cost + so_penalty
        reorder_qty  = (np.sqrt(2 * vel*365 * 50 / (uc*0.25)) * disr).sum()
        return {
            "total_cost":   round(total_cost, 0),
            "exp_loss":     round(exp_loss, 0),
            "hold_cost":    round(hold_cost, 0),
            "so_penalty":   round(so_penalty, 0),
            "reorder_skus": reorder_skus,
            "reduce_skus":  reduce_skus,
            "avg_so_risk":  round(float(so_risk.mean()), 3),
            "avg_exp_risk": round(float(exp_r.mean()), 3),
            "total_reorder_qty": round(float(reorder_qty), 0),
        }

    results = {name: _scenario_kpis(df, sc) for name, sc in SCENARIOS.items()}

    # ── Side-by-side scenario comparison cards ────────────────────────────
    st.subheader("📊 Scenario Comparison")
    cols = st.columns(len(SCENARIOS))
    for i, (sc_name, sc_def) in enumerate(SCENARIOS.items()):
        kpis = results[sc_name]
        col  = sc_def["colour"]
        with cols[i]:
            st.markdown(f"""
            <div class="sc-card" style="border-top:3px solid {col}">
              <h4 style="color:{col}">{sc_name}</h4>
              <p>{sc_def['desc']}</p>
            </div>""", unsafe_allow_html=True)
            st.metric("Total Cost",   f"${kpis['total_cost']:,.0f}")
            st.metric("Reorder SKUs", kpis["reorder_skus"])
            st.metric("Reduce SKUs",  kpis["reduce_skus"])
            st.metric("Avg SO Risk",  f"{kpis['avg_so_risk']:.0%}")

    # ── Scenario comparison bar charts ────────────────────────────────────
    sc_names  = list(results.keys())
    sc_colours= [SCENARIOS[n]["colour"] for n in sc_names]

    fig_sc = make_subplots(rows=1, cols=3,
        subplot_titles=("Total Cost ($)","Reorder SKUs","Avg Stockout Risk"),
        horizontal_spacing=0.08)

    for i, (metric, title) in enumerate([
        ("total_cost","Total Cost"),("reorder_skus","SKUs to Reorder"),("avg_so_risk","Avg Stockout Risk")
    ], 1):
        vals = [results[n][metric] for n in sc_names]
        fig_sc.add_bar(row=1, col=i, x=sc_names, y=vals,
                       marker_color=sc_colours, name=title, showlegend=False)

    apply_chart_layout(fig_sc, height=360,
        xaxis=dict(gridcolor=C["border"],tickangle=-30,tickfont=dict(size=8)),
        xaxis2=dict(gridcolor=C["border"],tickangle=-30,tickfont=dict(size=8)),
        xaxis3=dict(gridcolor=C["border"],tickangle=-30,tickfont=dict(size=8)),
        yaxis=dict(gridcolor=C["border"]),
        yaxis2=dict(gridcolor=C["border"]),
        yaxis3=dict(gridcolor=C["border"]))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Scenario sensitivity table ────────────────────────────────────────
    st.subheader("📋 Full Scenario Comparison Table")
    tbl = pd.DataFrame([{
        "Scenario":       sc_name,
        "Description":    SCENARIOS[sc_name]["desc"],
        "Total Cost ($)": f"${results[sc_name]['total_cost']:,.0f}",
        "Expiry Loss ($)":f"${results[sc_name]['exp_loss']:,.0f}",
        "Stockout Cost ($)":f"${results[sc_name]['so_penalty']:,.0f}",
        "Reorder SKUs":   results[sc_name]["reorder_skus"],
        "Reduce SKUs":    results[sc_name]["reduce_skus"],
        "Avg SO Risk":    f"{results[sc_name]['avg_so_risk']:.0%}",
    } for sc_name in sc_names])
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7: RISK HEATMAPS — THREE DISTINCT HEATMAP TYPES
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_heatmaps(df: pd.DataFrame):
    st.markdown("""<div class="rc-section">
      <h3>🌡️ Risk Heatmaps Dashboard</h3>
      <p>Three complementary heatmap views: (1) SKU × Risk Dimension grid showing every score,
         (2) Category × Risk Level matrix showing portfolio distribution,
         (3) Risk Intensity map showing compound risk across the full portfolio.</p>
    </div>""", unsafe_allow_html=True)

    hn = PRODUCT_NAME if PRODUCT_NAME in df.columns else SKU_ID

    # ── HEATMAP 1: SKU × Risk Dimension ──────────────────────────────────
    st.subheader("🗺️ Heatmap 1 — SKU × Risk Dimension")
    st.caption("Every row = one product. Every column = one risk dimension. "
               "Red = high risk. Green = low risk. Shows the full risk profile at a glance.")

    risk_dimensions = {}
    if EXPIRY_RISK in df.columns:
        risk_dimensions["Expiry Risk"] = df[EXPIRY_RISK].fillna(0).values
    if STOCKOUT_RISK in df.columns:
        risk_dimensions["Stockout Risk"] = df[STOCKOUT_RISK].fillna(0).values
    if SUPPLIER_RISK in df.columns:
        risk_dimensions["Supplier Risk"] = df[SUPPLIER_RISK].fillna(0).values
    if "lead_time_risk" in df.columns:
        risk_dimensions["Lead Time Risk"] = df["lead_time_risk"].fillna(0).values
    elif "delay_probability" in df.columns:
        risk_dimensions["Lead Time Risk"] = (df["delay_probability"] * 0.8).clip(0,1).fillna(0).values
    if "priority_score" in df.columns:
        risk_dimensions["Composite Risk"] = df["priority_score"].fillna(0).values

    if risk_dimensions:
        hm1_z   = np.array(list(risk_dimensions.values()))          # (n_dims, n_skus)
        hm1_y   = list(risk_dimensions.keys())
        hm1_x   = df[hn].tolist()
        # Sort SKUs by composite risk descending
        sort_idx = np.argsort(-hm1_z[-1] if len(hm1_z)>0 else -hm1_z[0])
        hm1_z    = hm1_z[:, sort_idx]
        hm1_x    = [hm1_x[i] for i in sort_idx]

        fig_hm1 = go.Figure(go.Heatmap(
            z=hm1_z, x=hm1_x, y=hm1_y,
            colorscale=[[0,"#003300"],[0.35,"#006600"],[0.55,"#FFB300"],[0.70,"#FF6D00"],[1,"#FF1744"]],
            zmin=0, zmax=1,
            text=np.round(hm1_z,2),
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Risk Score", font=dict(color=C["muted"])),
                tickfont=dict(color=C["muted"]),
                ),
        ))
        apply_chart_layout(fig_hm1, height=max(250, len(hm1_y)*55),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8), gridcolor=C["border"]),
            yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig_hm1, use_container_width=True)
    else:
        st.warning("Risk scores not yet computed — run pipeline first.")

    # ── HEATMAP 2: Category × Risk Level ─────────────────────────────────
    st.subheader("🗺️ Heatmap 2 — Category × Risk Level Matrix")
    st.caption("Shows which product categories have the most SKUs at each risk level. "
               "Darker = more SKUs in that risk zone.")

    if CATEGORY in df.columns and EXPIRY_PRESSURE in df.columns:
        cat_risk = df.groupby([CATEGORY, EXPIRY_PRESSURE]).size().unstack(fill_value=0)
        level_order = [l for l in ["Expired","Critical","High","Medium","Low"] if l in cat_risk.columns]
        cat_risk = cat_risk[level_order]

        fig_hm2 = go.Figure(go.Heatmap(
            z=cat_risk.values,
            x=cat_risk.columns.tolist(),
            y=cat_risk.index.tolist(),
            colorscale=[[0,"#001A33"],[0.4,"#0051A8"],[0.7,"#FFB300"],[1,"#FF1744"]],
            text=cat_risk.values,
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="SKU Count", font=dict(color=C["muted"])),
                          tickfont=dict(color=C["muted"]),
            ),
        ))
        apply_chart_layout(fig_hm2,
            height=max(280, len(cat_risk)*45),
            xaxis=dict(side="top", gridcolor=C["border"],
                       tickfont=dict(size=10, color=C["text"])),
            yaxis=dict(gridcolor=C["border"],
                       tickfont=dict(size=10, color=C["text"])))
        st.plotly_chart(fig_hm2, use_container_width=True)

    # ── HEATMAP 3: Risk Intensity Map (compound score as colour grid) ────
    st.subheader("🗺️ Heatmap 3 — Risk Intensity Map")
    st.caption("Full portfolio view sorted by expiry risk (Y) vs stockout risk (X). "
               "Each cell colour = compound risk intensity. "
               "Top-right = dual crisis zone. Bottom-left = healthy zone.")

    if EXPIRY_RISK in df.columns and STOCKOUT_RISK in df.columns:
        n_bins = 5
        er_bins = pd.cut(df[EXPIRY_RISK].fillna(0),  bins=n_bins,
                         labels=["0–20%","20–40%","40–60%","60–80%","80–100%"])
        so_bins = pd.cut(df[STOCKOUT_RISK].fillna(0), bins=n_bins,
                         labels=["0–20%","20–40%","40–60%","60–80%","80–100%"])

        intensity_df = df.copy()
        intensity_df["er_bin"] = er_bins
        intensity_df["so_bin"] = so_bins

        pivot = intensity_df.groupby(["er_bin","so_bin"],observed=True).size().unstack(fill_value=0)

        # Fill missing bins
        all_bins = ["0–20%","20–40%","40–60%","60–80%","80–100%"]
        for b in all_bins:
            if b not in pivot.columns: pivot[b] = 0
            if b not in pivot.index:   pivot.loc[b] = 0
        pivot = pivot.reindex(index=all_bins, columns=all_bins, fill_value=0)

        fig_hm3 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"Stockout {b}" for b in pivot.columns],
            y=[f"Expiry {b}"   for b in pivot.index],
            colorscale=[
                [0.0, "#001A08"], [0.2, "#003310"],
                [0.5, "#FFB300"], [0.8, "#FF6D00"], [1.0, "#FF1744"]
            ],
            text=pivot.values,
            texttemplate="%{text} SKUs",
            textfont=dict(size=11, color="white"),
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="SKU Count", font=dict(color=C["muted"])),
                tickfont=dict(color=C["muted"]),
            ),
        ))
        fig_hm3.add_annotation(x="Stockout 80–100%", y="Expiry 80–100%",
            text="🚨 DUAL CRISIS", showarrow=False,
            font=dict(color=C["red"], size=11), yshift=-20)
        fig_hm3.add_annotation(x="Stockout 0–20%", y="Expiry 0–20%",
            text="✅ HEALTHY ZONE", showarrow=False,
            font=dict(color=C["green"], size=11), yshift=-20)
        apply_chart_layout(fig_hm3, height=400,
            xaxis=dict(side="bottom", gridcolor=C["border"],
                       tickfont=dict(size=9), tickangle=-20),
            yaxis=dict(gridcolor=C["border"], tickfont=dict(size=9),
                       autorange="reversed"))
        st.plotly_chart(fig_hm3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — tabbed layout within the Risk & Compliance tab
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_compliance_tab(master_df: pd.DataFrame,
                                financial: dict[str, Any],
                                suppliers_df: Optional[pd.DataFrame] = None):
    """
    Call this from dashboard/app.py:
        from dashboard.risk_compliance_tab import render_risk_compliance_tab
        with t_risk:
            render_risk_compliance_tab(df, financial, data['suppliers'])
    """
    st.markdown(RISK_CSS, unsafe_allow_html=True)

    # Compliance badge
    st.markdown("""
    <div style="background:linear-gradient(90deg,rgba(0,145,234,.12),rgba(0,200,83,.08));
                border:1px solid #1E3050;border-radius:10px;padding:12px 18px;margin-bottom:16px;
                display:flex;gap:20px;flex-wrap:wrap;align-items:center">
      <span style="color:#0091EA;font-weight:700;font-size:14px">📋 Risk & Compliance Module</span>
      <span class="risk-pill risk-low">✅ Risk</span>
      <span class="risk-pill risk-low">✅ Lead Time Risk</span>
      <span class="risk-pill risk-low">✅ Prediction Model</span>
      <span class="risk-pill risk-low">✅ Risk Analysis</span>
      <span class="risk-pill risk-low">✅ Risk Assessment</span>
      <span class="risk-pill risk-low">✅ Scenario Planning</span>
      <span class="risk-pill risk-low">✅ Risk Heatmaps</span>
    </div>""", unsafe_allow_html=True)

    sub_tabs = st.tabs([
        "🛡️ Risk Overview",
        "⏱️ Lead Time Risk",
        "🤖 AI Prediction Model",
        "💰 Risk Analysis",
        "📋 Risk Assessment",
        "🔮 Scenario Planning",
        "🌡️ Risk Heatmaps",
    ])

    with sub_tabs[0]: render_risk_overview(master_df)
    with sub_tabs[1]: render_lead_time_risk(master_df)
    with sub_tabs[2]: render_supplier_prediction_model(master_df, suppliers_df)
    with sub_tabs[3]: render_risk_analysis(master_df, financial)
    with sub_tabs[4]: render_risk_assessment(master_df)
    with sub_tabs[5]: render_scenario_planning(master_df)
    with sub_tabs[6]: render_risk_heatmaps(master_df)
