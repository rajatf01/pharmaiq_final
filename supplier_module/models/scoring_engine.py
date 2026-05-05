"""
supplier_module/models/scoring_engine.py
-----------------------------------------
AI scoring engine for the Supplier Comparison Module.

Outputs per supplier-product pair:
  - supplier_risk_score   : [0, 1]  lower = more reliable
  - mcdm_score            : [0, 100] higher = better overall value
  - cost_rank             : 1 = cheapest
  - mcdm_rank             : 1 = best holistic value
  - recommendation        : True/False
  - confidence            : [0, 1]
  - explanation           : dict with plain-English text
  - tradeoff_vs_next_best : dict with financial comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Import column constants from data layer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supplier_module.data.supplier_data import (
    SUPPLIER_ID, SUPPLIER_NAME, PRODUCT_ID, PRODUCT_NAME,
    UNIT_COST, STATED_LEAD_TIME, LEAD_TIME_STD, LEAD_TIME_CV,
    DEFECT_RATE, COMPLIANCE_SCORE, FULFILLMENT_RATE,
    DELAY_PROBABILITY, AVG_DELAY_DAYS, RETURNS_RATE, BUY_VS_USAGE,
    MIN_SHELF_LIFE, TOTAL_ORDERS,
)

# ── Output column names ───────────────────────────────────────────────────────
RISK_SCORE   = "supplier_risk_score"
RISK_LABEL   = "risk_label"
MCDM_SCORE   = "mcdm_score"
COST_RANK    = "cost_rank"
MCDM_RANK    = "mcdm_rank"
IS_RECOMMENDED      = "is_recommended"
RECOMMENDATION_CONF = "recommendation_confidence"
EFF_LEAD_TIME       = "effective_lead_time_days"

# ── Default MCDM weights (must sum to 1.0) ────────────────────────────────────
DEFAULT_WEIGHTS = {
    "cost":        0.35,
    "lead_time":   0.25,
    "quality":     0.20,
    "reliability": 0.15,
    "shelf_life":  0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. SUPPLIER RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(row: pd.Series) -> float:
    """
    Rule-based supplier risk score [0, 1].  Lower = more reliable.

    Components (weights):
      delay_probability  0.40
      lead_time_cv       0.25
      avg_delay_norm     0.20  (normalised to 14-day max)
      defect_rate_norm   0.15  (normalised to 5% max)

    Confidence shrinkage applied for < 20 historical orders.
    """
    delay_p    = float(row.get(DELAY_PROBABILITY,  0.20))
    lt_cv      = float(row.get(LEAD_TIME_CV,       0.20))
    avg_delay  = float(row.get(AVG_DELAY_DAYS,     2.0))
    defect     = float(row.get(DEFECT_RATE,        0.01))
    n_orders   = max(int(row.get(TOTAL_ORDERS,     10)),  1)

    avg_delay_norm = min(1.0, avg_delay / 14.0)
    defect_norm    = min(1.0, defect    / 0.05)

    raw = (0.40 * delay_p     +
           0.25 * lt_cv       +
           0.20 * avg_delay_norm +
           0.15 * defect_norm)

    # Confidence adjustment: shrink toward 0.20 baseline with sparse data
    confidence = min(1.0, n_orders / 20.0)
    score      = 0.20 * (1 - confidence) + raw * confidence

    return round(float(np.clip(score, 0.0, 1.0)), 4)


def risk_label(score: float) -> str:
    if score < 0.35: return "Low"
    if score < 0.60: return "Medium"
    if score < 0.80: return "High"
    return "Critical"


# ─────────────────────────────────────────────────────────────────────────────
# 2. EFFECTIVE LEAD TIME
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_lead_time(row: pd.Series, z: float = 1.645) -> float:
    """95th-percentile lead time = mean + z * std."""
    lt  = float(row.get(STATED_LEAD_TIME, 14))
    std = float(row.get(LEAD_TIME_STD,    2.0))
    return round(lt + z * std, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MCDM SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_mcdm_scores(sp_df: pd.DataFrame,
                         weights: Dict[str, float] | None = None) -> pd.Series:
    """
    Compute MCDM score [0–100] for all suppliers of a given product.
    Normalises each dimension relative to the shortlist (min-max scaling).

    Returns: pd.Series of scores aligned to sp_df.index.
    """
    w = weights or DEFAULT_WEIGHTS

    df = sp_df.copy()

    # ── Normalise dimensions ──────────────────────────────────────────────
    def norm(series, invert=False):
        """Min-max normalise; invert if lower is better."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        n = (series - mn) / (mx - mn)
        return (1 - n) if invert else n

    # Cost: lower is better → invert
    cost_n = norm(df[UNIT_COST], invert=True)

    # Lead time: lower eff lead time is better → invert
    eff_lt = df.get(EFF_LEAD_TIME,
                    df[STATED_LEAD_TIME] + 1.645 * df.get(LEAD_TIME_STD, 2))
    lead_n = norm(eff_lt, invert=True)

    # Quality: higher compliance better, lower defect better
    qual_n = (df[COMPLIANCE_SCORE] / 100.0) * (1 - df[DEFECT_RATE] / 0.05).clip(0, 1)

    # Reliability: higher fulfillment and lower delay better
    rel_n  = (df[FULFILLMENT_RATE] * (1 - df[DELAY_PROBABILITY])).clip(0, 1)

    # Shelf life: higher is better, normalise to 365 days
    shelf_n = (df[MIN_SHELF_LIFE] / 365.0).clip(0, 1)

    raw = (w["cost"]        * cost_n  +
           w["lead_time"]   * lead_n  +
           w["quality"]     * qual_n  +
           w["reliability"] * rel_n   +
           w["shelf_life"]  * shelf_n)

    return (raw * 100).round(1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRADE-OFF CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_tradeoff(sup_a: pd.Series, sup_b: pd.Series,
                      order_qty: int = 1000,
                      holding_rate: float = 0.25) -> Dict:
    """
    Quantify the monetary trade-off between two suppliers for one order.

    Returns a dict with:
      cost_saving, delay_cost_premium, quality_cost_diff, net_benefit,
      recommended_supplier, explanation_text
    """
    # ── Cost difference ────────────────────────────────────────────────────
    cost_saving = (float(sup_b[UNIT_COST]) - float(sup_a[UNIT_COST])) * order_qty

    # ── Extra safety stock cost due to higher delay prob of A vs B ──────────
    delay_diff     = float(sup_a[DELAY_PROBABILITY]) - float(sup_b[DELAY_PROBABILITY])
    daily_demand   = order_qty / 30.0
    extra_ss_units = max(0, delay_diff * float(sup_a.get(AVG_DELAY_DAYS, 4)) * daily_demand)
    daily_hold_r   = holding_rate / 365.0
    delay_hold_cost= extra_ss_units * float(sup_a[UNIT_COST]) * daily_hold_r * 30.0

    # ── Quality cost difference ────────────────────────────────────────────
    qual_diff = ((float(sup_a[DEFECT_RATE]) - float(sup_b[DEFECT_RATE]))
                 * order_qty * float(sup_a[UNIT_COST]))

    net_benefit = cost_saving - delay_hold_cost - qual_diff

    # ── Winner ─────────────────────────────────────────────────────────────
    winner = sup_a[SUPPLIER_NAME] if net_benefit >= 0 else sup_b[SUPPLIER_NAME]

    text = (
        f"Cost saving by choosing {sup_a[SUPPLIER_NAME]}: "
        f"${cost_saving:+.2f} ({order_qty:,} units × "
        f"${abs(float(sup_a[UNIT_COST]) - float(sup_b[UNIT_COST])):.2f}/unit)\n"
        f"Additional safety stock cost (delay risk): "
        f"-${delay_hold_cost:.2f}/month\n"
        f"Quality cost difference: "
        f"-${qual_diff:.2f}\n"
        f"{'─'*52}\n"
        f"Net benefit of choosing {sup_a[SUPPLIER_NAME]}: "
        f"${net_benefit:+.2f} per {order_qty:,}-unit order\n"
        f"\n✅ Recommendation: Choose {winner}."
    )

    return {
        "supplier_a":        sup_a[SUPPLIER_NAME],
        "supplier_b":        sup_b[SUPPLIER_NAME],
        "order_qty":         order_qty,
        "cost_saving":       round(cost_saving, 2),
        "delay_cost":        round(delay_hold_cost, 2),
        "quality_cost_diff": round(qual_diff, 2),
        "net_benefit":       round(net_benefit, 2),
        "recommended":       winner,
        "text":              text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPLANATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(row: pd.Series, all_rows: pd.DataFrame) -> Dict:
    """
    Generate plain-English explanation for a supplier-product pair.
    Returns dict: {headline, positives, risks, action, tradeoff_text}
    """
    name       = row[SUPPLIER_NAME]
    cost       = float(row[UNIT_COST])
    cost_rank  = int(row.get(COST_RANK, 1))
    n_sup      = len(all_rows)
    delay_p    = float(row.get(DELAY_PROBABILITY, 0.20))
    avg_delay  = float(row.get(AVG_DELAY_DAYS, 2.0))
    fulfill    = float(row.get(FULFILLMENT_RATE, 0.95))
    defect     = float(row.get(DEFECT_RATE, 0.01))
    compliance = int(row.get(COMPLIANCE_SCORE, 90))
    shelf      = int(row.get(MIN_SHELF_LIFE, 365))
    eff_lt     = float(row.get(EFF_LEAD_TIME, 14))
    mcdm_rank  = int(row.get(MCDM_RANK, 1))
    mcdm_score = float(row.get(MCDM_SCORE, 70))
    risk_s     = float(row.get(RISK_SCORE, 0.30))
    r_label    = row.get(RISK_LABEL, "Low")

    positives = []
    risks     = []

    # ── Cost ──────────────────────────────────────────────────────────────
    if cost_rank == 1:
        positives.append(
            f"Lowest unit cost among {n_sup} shortlisted suppliers at ${cost:.2f}/unit.")
    else:
        cheapest  = all_rows.nsmallest(1, UNIT_COST).iloc[0]
        delta     = cost - float(cheapest[UNIT_COST])
        positives.append(
            f"Unit cost is ${delta:.2f}/unit more than cheapest option "
            f"({cheapest[SUPPLIER_NAME]} at ${float(cheapest[UNIT_COST]):.2f}). "
            f"Ranked {cost_rank} of {n_sup} by cost.")

    # ── Fulfillment ───────────────────────────────────────────────────────
    if fulfill >= 0.97:
        positives.append(
            f"Excellent fulfillment rate: {fulfill:.1%} of orders delivered in full.")
    elif fulfill >= 0.92:
        positives.append(
            f"Good fulfillment rate: {fulfill:.1%} of orders fulfilled completely.")
    else:
        risks.append(
            f"Low fulfillment rate: only {fulfill:.1%} of orders arrive in full — "
            f"budget for partial deliveries.")

    # ── Compliance ────────────────────────────────────────────────────────
    if compliance >= 95:
        positives.append(
            f"WHO-GMP compliant with compliance score of {compliance}/100.")
    elif compliance >= 88:
        positives.append(f"Compliance score: {compliance}/100 — meets standards.")
    else:
        risks.append(
            f"Below-average compliance score ({compliance}/100) — verify certifications.")

    # ── Delay risk ────────────────────────────────────────────────────────
    if delay_p <= 0.20:
        positives.append(
            f"Highly reliable delivery: only {delay_p:.0%} of past orders were delayed.")
    elif delay_p <= 0.40:
        risks.append(
            f"Moderate delay risk: {delay_p:.0%} of orders arrive late by an average "
            f"of {avg_delay:.1f} days. Order {int(np.ceil(avg_delay))} days earlier "
            f"than the stated lead time.")
    else:
        risks.append(
            f"High delay risk: {delay_p:.0%} of orders have been late. Effective lead "
            f"time is {eff_lt:.1f} days (not {int(row[STATED_LEAD_TIME])}). "
            f"Add extra safety stock buffer.")

    # ── Defect rate ───────────────────────────────────────────────────────
    if defect < 0.01:
        positives.append(
            f"Excellent quality: defect rate of {defect:.1%} — well below 1% threshold.")
    elif defect < 0.03:
        pass  # neutral — don't mention
    else:
        risks.append(
            f"Quality concern: defect rate of {defect:.1%} exceeds the 3% warning threshold. "
            f"Consider requesting a quality guarantee clause.")

    # ── Shelf life ────────────────────────────────────────────────────────
    if shelf < 180:
        risks.append(
            f"Short shelf life warning: minimum {shelf} days remaining at delivery. "
            f"High risk of expiry before dispensing.")
    elif shelf >= 365:
        positives.append(
            f"Long shelf life: minimum {shelf} days remaining at delivery — "
            f"low expiry risk.")

    # ── Headline ──────────────────────────────────────────────────────────
    if mcdm_rank == 1:
        headline = (f"{name} is recommended as primary supplier. "
                    f"AI Score: {mcdm_score:.1f}/100 ({r_label} risk).")
    else:
        headline = (f"{name} is ranked #{mcdm_rank} overall. "
                    f"AI Score: {mcdm_score:.1f}/100 ({r_label} risk).")

    # ── Recommended action ────────────────────────────────────────────────
    if mcdm_rank == 1:
        action = (f"Select {name} as primary supplier. "
                  f"Set reorder trigger to {int(eff_lt)} days "
                  f"(not {int(row[STATED_LEAD_TIME])}) to account for delay risk.")
    else:
        action = (f"Consider {name} as backup supplier. "
                  f"Primary recommendation is "
                  f"{all_rows.loc[all_rows[MCDM_RANK]==1, SUPPLIER_NAME].values[0] if MCDM_RANK in all_rows.columns else 'the top-ranked supplier'}.")

    # ── Trade-off vs next best ────────────────────────────────────────────
    tradeoff_text = ""
    if MCDM_RANK in all_rows.columns and mcdm_rank == 1 and len(all_rows) > 1:
        next_best_rows = all_rows[all_rows[MCDM_RANK] == 2]
        if not next_best_rows.empty:
            nb = next_best_rows.iloc[0]
            tf = compute_tradeoff(row, nb, order_qty=1000)
            tradeoff_text = tf["text"]

    return {
        "headline":     headline,
        "positives":    positives,
        "risks":        risks,
        "action":       action,
        "tradeoff":     tradeoff_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN SCORING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def score_all_suppliers(sp_df: pd.DataFrame,
                         weights: Dict[str, float] | None = None) -> pd.DataFrame:
    """
    Run the full scoring pipeline on a supplier_products DataFrame.

    Input:  sp_df — one row per supplier-product pair
    Output: enriched DataFrame with risk scores, MCDM scores, ranks, explanations
    """
    from supplier_module.data.supplier_data import validate_sp_dataframe
    validate_sp_dataframe(sp_df)

    df = sp_df.copy()

    # ── Effective lead time ───────────────────────────────────────────────
    if EFF_LEAD_TIME not in df.columns:
        df[EFF_LEAD_TIME] = df.apply(compute_effective_lead_time, axis=1)

    # ── Risk score ────────────────────────────────────────────────────────
    df[RISK_SCORE] = df.apply(compute_risk_score, axis=1)
    df[RISK_LABEL] = df[RISK_SCORE].apply(risk_label)

    # ── MCDM scores per product ───────────────────────────────────────────
    mcdm_scores = []
    for pid, group in df.groupby(PRODUCT_ID):
        scores = compute_mcdm_scores(group, weights)
        mcdm_scores.append(scores)
    df[MCDM_SCORE] = pd.concat(mcdm_scores)

    # ── Rankings per product ──────────────────────────────────────────────
    df[COST_RANK] = df.groupby(PRODUCT_ID)[UNIT_COST].rank(
        method="min", ascending=True).astype(int)
    df[MCDM_RANK] = df.groupby(PRODUCT_ID)[MCDM_SCORE].rank(
        method="min", ascending=False).astype(int)

    # ── Recommendation ────────────────────────────────────────────────────
    df[IS_RECOMMENDED] = df[MCDM_RANK] == 1

    # ── Confidence (based on data volume and score margin) ────────────────
    def confidence(row):
        n          = max(int(row.get(TOTAL_ORDERS, 10)), 1)
        data_conf  = min(1.0, n / 20.0)
        score_gap  = (row.get(MCDM_SCORE, 70) - 50.0) / 50.0  # how far from neutral
        return round(float(np.clip(0.50 + 0.30 * data_conf + 0.20 * score_gap, 0.40, 0.97)), 2)

    df[RECOMMENDATION_CONF] = df.apply(confidence, axis=1)

    # ── Explanations per supplier-product ─────────────────────────────────
    explanations = []
    for pid, group in df.groupby(PRODUCT_ID):
        for idx, row in group.iterrows():
            exp = generate_explanation(row, group)
            explanations.append({"_idx": idx, **exp})

    exp_df = pd.DataFrame(explanations).set_index("_idx")
    df = df.join(exp_df[["headline", "positives", "risks",
                           "action", "tradeoff"]])

    return df.sort_values([PRODUCT_ID, MCDM_RANK]).reset_index(drop=True)


def score_product_suppliers(sp_df: pd.DataFrame,
                              product_id: str,
                              weights: Dict[str, float] | None = None) -> pd.DataFrame:
    """Score suppliers for a single product — used by the API endpoint."""
    product_sp = sp_df[sp_df[PRODUCT_ID] == product_id].copy()
    if product_sp.empty:
        raise ValueError(f"No suppliers found for product {product_id}")
    return score_all_suppliers(product_sp, weights)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    from supplier_module.data.supplier_data import generate_synthetic_supplier_products

    sp_df  = generate_synthetic_supplier_products()
    scored = score_all_suppliers(sp_df)

    print("\n" + "="*70)
    print("SUPPLIER SCORING RESULTS")
    print("="*70)

    for pid, group in scored.groupby(PRODUCT_ID):
        print(f"\n📦 {group.iloc[0][PRODUCT_NAME]}")
        print("-"*60)
        cols = [SUPPLIER_NAME, UNIT_COST, COST_RANK,
                MCDM_SCORE, MCDM_RANK, RISK_SCORE, RISK_LABEL, IS_RECOMMENDED]
        print(group[cols].to_string(index=False))

    print("\n" + "="*70)
    print("EXAMPLE AI EXPLANATION — Top-ranked supplier for Paracetamol")
    print("="*70)
    para = scored[(scored[PRODUCT_ID] == "SKU003") & (scored[MCDM_RANK] == 1)]
    if not para.empty:
        r = para.iloc[0]
        print(f"\n{r['headline']}")
        print("\n✅ Why recommended:")
        for p in r["positives"]: print(f"  • {p}")
        print("\n⚠️  Risk factors:")
        for rk in r["risks"]: print(f"  • {rk}")
        print(f"\n🎯 Action: {r['action']}")
        if r.get("tradeoff"):
            print(f"\n💰 Trade-off:\n{r['tradeoff']}")
