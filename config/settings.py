"""
config/settings.py
All configurable parameters — change here, affects the whole system.
"""

# ── Model Settings ────────────────────────────────────────────────────────────
FORECAST_HORIZON_DAYS  = 30
ROLLING_WINDOW_DAYS    = 30
MA_WINDOW_DAYS         = 14

# ── Risk Thresholds ───────────────────────────────────────────────────────────
HIGH_EXPIRY_RISK       = 0.70
MEDIUM_EXPIRY_RISK     = 0.40
HIGH_STOCKOUT_RISK     = 0.65
HIGH_SUPPLIER_RISK     = 0.60

# ── Safety Stock ──────────────────────────────────────────────────────────────
DEFAULT_SERVICE_LEVEL  = 0.95
Z_SCORE_95             = 1.645
Z_SCORE_99             = 2.326

# ── Financial Defaults ────────────────────────────────────────────────────────
HOLDING_COST_RATE      = 0.25
STOCKOUT_PENALTY_MULT  = 2.0
EXPIRY_SALVAGE_RATE    = 0.05

# ── Expiry Windows ────────────────────────────────────────────────────────────
CRITICAL_EXPIRY_DAYS   = 30
WARNING_EXPIRY_DAYS    = 90

# ── Odoo API ──────────────────────────────────────────────────────────────────
ODOO_URL               = "http://localhost:8069"
ODOO_DB                = "pharma_db"
ODOO_USERNAME          = "admin"
ODOO_PASSWORD          = "admin"
ODOO_TIMEOUT_SEC       = 30

# ── Synthetic Data ────────────────────────────────────────────────────────────
SYNTHETIC_SKUS         = 50
SYNTHETIC_HISTORY_DAYS = 730
SYNTHETIC_SEED         = 42

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE        = "PharmaIQ — Inventory Intelligence"
TOP_N_SKUS             = 10
