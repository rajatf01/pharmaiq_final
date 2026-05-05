"""
pipeline.py
End-to-end pipeline orchestrator.
Usage:
    from pipeline import run_pipeline
    results = run_pipeline()
"""
import os, sys, logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_pipeline(mode="manual", odoo_config=None, csv_paths=None,
                 service_level=0.95, disruption_factor=1.0,
                 implementation_cost=5000.0):
    logger.info("="*60)
    logger.info("PharmaIQ Pipeline Starting")
    logger.info("="*60)

    # Step 1: Load data
    logger.info("Step 1/6 — Loading data")
    from data.data_layer import DataLayer
    dl   = DataLayer(mode=mode, odoo_config=odoo_config, csv_paths=csv_paths)
    data = dl.load()
    sales_df     = data["sales"]
    batches_df   = data["batches"]
    suppliers_df = data["suppliers"]
    products_df  = data["products"]
    logger.info(f"  Data loaded: {len(sales_df)} sales rows, {len(batches_df)} batches, {len(suppliers_df)} supplier records")

    # Step 2: Feature engineering
    logger.info("Step 2/6 — Feature engineering")
    from features.feature_engineering import (build_sales_features, build_master_features,
        build_batch_features, build_sku_summary)
    sales_features = build_sales_features(sales_df)
    sku_summary    = build_sku_summary(sales_features)
    batch_features = build_batch_features(batches_df, sku_summary)
    master_df      = build_master_features(sales_df, batches_df, suppliers_df, products_df)
    logger.info(f"  Master features: {len(master_df)} SKUs, {len(master_df.columns)} cols")

    # Step 3: AI Models
    logger.info("Step 3/6 — Running AI models")
    from models.demand_forecasting import run_demand_forecast
    forecast_df, demand_model = run_demand_forecast(sales_features)
    master_df = master_df.merge(
        forecast_df.drop(columns=[c for c in forecast_df.columns
            if c in master_df.columns and c != "sku_id"], errors="ignore"),
        on="sku_id", how="left")

    from models.expiry_risk import run_expiry_risk
    master_df, expiry_model = run_expiry_risk(master_df, batch_features)

    from models.supplier_risk import run_supplier_risk
    from features.feature_engineering import build_supplier_features
    sup_features = build_supplier_features(suppliers_df)
    master_df, supplier_model = run_supplier_risk(master_df, sup_features)
    logger.info("  All models complete")

    # Step 4: Decision engine
    logger.info("Step 4/6 — Decision engine")
    from engine.decision_engine import run_decision_engine
    master_df = run_decision_engine(master_df, service_level=service_level,
                                    disruption_factor=disruption_factor)

    # Step 5: Financial
    logger.info("Step 5/6 — Financial analysis")
    from business.financial_layer import build_financial_summary
    financial = build_financial_summary(master_df, implementation_cost=implementation_cost)
    master_df = financial["sku_costs"]

    # Step 6: Explanations
    logger.info("Step 6/6 — Generating explanations")
    from explainability.explainer import add_explanations, get_top_risk_insights
    master_df = add_explanations(master_df)
    insights  = get_top_risk_insights(master_df)

    logger.info("="*60)
    logger.info("Pipeline complete ✓")
    logger.info("="*60)

    return {
        "master_df":  master_df,
        "sales_df":   sales_df,
        "batches_df": batches_df,
        "forecasts":  forecast_df,
        "financial":  financial,
        "models":     {"demand":demand_model,"expiry":expiry_model,"supplier":supplier_model},
        "insights":   insights,
        "data_layer": dl,
    }


if __name__ == "__main__":
    results = run_pipeline()
    df  = results["master_df"]
    fin = results["financial"]
    print(f"\nPipeline ran successfully for {len(df)} SKUs")
    cols = ["sku_id","product_name","expiry_risk_score","stockout_risk_score",
            "recommended_reorder_qty","recommended_action"]
    avail = [c for c in cols if c in df.columns]
    print("\nTop 5 priority SKUs:")
    print(df[avail].head(5).to_string(index=False))
    print(f"\nFinancial Summary:")
    print(f"  AI Total Cost:       ${fin['summary']['ai']['total_cost']:,.2f}")
    print(f"  Baseline Total Cost: ${fin['summary']['baseline']['total_cost']:,.2f}")
    print(f"  Savings:             ${fin['roi_metrics']['cost_savings_usd']:,.2f}")
    print(f"  ROI:                 {fin['roi_metrics']['roi_percent']:.1f}%")
