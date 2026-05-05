"""
config/schema.py
Single source of truth for ALL column names.
Import from here everywhere — never hardcode column strings.
"""

# ── Identity ─────────────────────────────────────────────────────────────────
SKU_ID          = "sku_id"
PRODUCT_NAME    = "product_name"
CATEGORY        = "category"
BATCH_ID        = "batch_id"

# ── Time ─────────────────────────────────────────────────────────────────────
DATE            = "date"

# ── Inventory ────────────────────────────────────────────────────────────────
STOCK_LEVEL     = "stock_level"
QTY_SOLD        = "quantity_sold"
REORDER_POINT   = "reorder_point"
REORDER_QTY     = "reorder_quantity"

# ── Expiry ───────────────────────────────────────────────────────────────────
EXPIRY_DATE     = "expiry_date"
DAYS_TO_EXPIRY  = "days_to_expiry"

# ── Supplier ─────────────────────────────────────────────────────────────────
SUPPLIER_ID     = "supplier_id"
SUPPLIER_NAME   = "supplier_name"
LEAD_TIME_DAYS  = "lead_time_days"
LEAD_TIME_STD   = "lead_time_std"

# ── Financial ────────────────────────────────────────────────────────────────
UNIT_COST       = "unit_cost"
SELLING_PRICE   = "selling_price"

# ── Engineered Features ──────────────────────────────────────────────────────
SALES_VELOCITY  = "sales_velocity_30d"
STOCK_COVER     = "stock_cover_days"
DEMAND_FORECAST = "demand_forecast"
FORECAST_LOWER  = "forecast_lower"
FORECAST_UPPER  = "forecast_upper"

# ── Risk Scores ──────────────────────────────────────────────────────────────
EXPIRY_RISK     = "expiry_risk_score"
STOCKOUT_RISK   = "stockout_risk_score"
SUPPLIER_RISK   = "supplier_risk_score"

# ── Decision Engine Output ───────────────────────────────────────────────────
RECOMMENDED_QTY  = "recommended_reorder_qty"
ACTION           = "recommended_action"
ACTION_REASON    = "action_reason"
EXPIRY_PRESSURE  = "expiry_pressure"

# ── Financial Output ─────────────────────────────────────────────────────────
EXPIRY_LOSS      = "expiry_loss_usd"
HOLDING_COST     = "holding_cost_usd"
STOCKOUT_PENALTY = "stockout_penalty_usd"
TOTAL_COST       = "total_cost_usd"

# ── Action labels ─────────────────────────────────────────────────────────────
ACTION_REORDER_NOW   = "Reorder Now"
ACTION_HOLD          = "Hold Inventory"
ACTION_REVIEW        = "Review Manually"
ACTION_REDUCE        = "Reduce / Promote Stock"
ACTION_URGENT_EXPIRY = "Urgent: Expiry Risk"

# ── Required columns for pipeline validation ──────────────────────────────────
REQUIRED_SALES_COLS    = [SKU_ID, DATE, QTY_SOLD, STOCK_LEVEL]
REQUIRED_BATCH_COLS    = [SKU_ID, BATCH_ID, EXPIRY_DATE, STOCK_LEVEL, UNIT_COST]
REQUIRED_SUPPLIER_COLS = [SKU_ID, SUPPLIER_ID, LEAD_TIME_DAYS]
