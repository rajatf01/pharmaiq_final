"""
PharmaIQ — Main Streamlit Dashboard  v1.5
Tabs: Executive | Operations | Historical | Supplier Comparison | Technical
Run: streamlit run dashboard/app.py

UI/UX update only: sky-blue light theme, rounded cards, hover effects, light title card.
"""
import os, sys, logging, warnings, tempfile
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Optional, cast
from risk_compliance_tab import render_risk_compliance_tab


# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "supplier_module"))

from config.schema import (
    SKU_ID, PRODUCT_NAME, CATEGORY, DATE, QTY_SOLD, STOCK_LEVEL,
    EXPIRY_RISK, EXPIRY_PRESSURE, STOCKOUT_RISK, SUPPLIER_RISK,
    RECOMMENDED_QTY, ACTION, ACTION_REASON,
    UNIT_COST, SELLING_PRICE, STOCK_COVER, SALES_VELOCITY,
    DEMAND_FORECAST, EXPIRY_LOSS, HOLDING_COST, STOCKOUT_PENALTY, TOTAL_COST,
    ACTION_REORDER_NOW, ACTION_HOLD, ACTION_REVIEW,
    ACTION_REDUCE, ACTION_URGENT_EXPIRY, EXPIRY_DATE, DAYS_TO_EXPIRY,
)
from config.settings import (
    DASHBOARD_TITLE, TOP_N_SKUS,
    HIGH_EXPIRY_RISK, MEDIUM_EXPIRY_RISK,
    HIGH_STOCKOUT_RISK, HIGH_SUPPLIER_RISK,
)

logger = logging.getLogger(__name__)

try:
    from risk_compliance_tab import render_risk_compliance_tab
    _RISK_TAB_AVAILABLE = True
except ImportError:
    _RISK_TAB_AVAILABLE = False

st.set_page_config(page_title=DASHBOARD_TITLE, page_icon="💊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* TRUE SaaS reset */
:root{
  --bg:#f4fbff;
  --panel:#ffffff;
  --ink:#071a3d;
  --muted:#64748b;
  --blue:#0284c7;
  --blue2:#0ea5e9;
  --line:#dbeafe;
  --green:#16a34a;
  --red:#ef4444;
  --orange:#f97316;
  --purple:#8b5cf6;
}
html, body, .stApp, [class*="css"]{font-family:'Inter','Segoe UI',sans-serif!important;}
.stApp{
  background:
    radial-gradient(circle at 82% 0%, rgba(14,165,233,.18), transparent 28%),
    linear-gradient(180deg,#eef9ff 0%,#ffffff 42%,#f6fbff 100%)!important;
  color:var(--ink)!important;
}
header[data-testid="stHeader"]{
  background: transparent !important;
}
.main,.block-container{background:transparent!important;}
.block-container{max-width:100%!important;padding:1.25rem 1.55rem 2rem 1.55rem!important;}
#MainMenu, footer{visibility:hidden!important;}
hr{border:none!important;border-top:1px solid var(--line)!important;margin:1.15rem 0!important;}

/* Sidebar: clean SaaS nav */
section[data-testid="stSidebar"]{
  background:#fff!important;
  border-right:1px solid var(--line)!important;
  box-shadow:14px 0 35px rgba(2,132,199,.08)!important;
}
section[data-testid="stSidebar"] > div{padding:1.25rem 1rem!important;}
section[data-testid="stSidebar"] *{color:#183b63!important;}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  color:#0265c9!important;font-weight:900!important;letter-spacing:-.03em!important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#64748b!important;}

.sidebar-brand{
  display:flex;align-items:center;gap:12px;margin:4px 0 24px 0;padding-bottom:22px;border-bottom:1px solid var(--line);
}
.sidebar-logo{
  width:52px;height:52px;border-radius:18px;background:linear-gradient(145deg,#0ea5e9,#006fe6);
  color:#fff!important;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:20px;
  box-shadow:0 16px 34px rgba(2,132,199,.28);
}
.sidebar-name{font-size:25px;font-weight:900;color:#071a3d!important;line-height:1;letter-spacing:-.04em;}
.sidebar-caption{font-size:11px;font-weight:700;color:#355678!important;margin-top:5px;}

/* Make sidebar widgets look like a product panel */
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label{
  font-weight:700!important;color:#26486f!important;font-size:13px!important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label{
  padding:7px 8px!important;border-radius:12px!important;margin:1px 0!important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#eff8ff!important;}
input, textarea, div[data-baseweb="select"] > div{
  background:#fff!important;border:1px solid var(--line)!important;border-radius:13px!important;color:#102a56!important;
}
div[data-baseweb="tag"]{background:#e0f2fe!important;color:#075985!important;border-radius:999px!important;font-weight:800!important;}
div[data-testid="stFileUploadDropzone"]{background:#fff!important;border:1.5px dashed #93c5fd!important;border-radius:18px!important;}

/* HERO and custom product UI */
.saas-hero{
  position:relative;overflow:hidden;border-radius:30px;padding:32px 34px;margin:0 0 18px 0;
  background:
    radial-gradient(circle at 69% 45%, rgba(14,165,233,.28), transparent 24%),
    linear-gradient(120deg,#ffffff 0%,#eaf8ff 42%,#cfeeff 100%);
  border:1px solid #c9e9ff;box-shadow:0 24px 60px rgba(2,132,199,.15);
}
.saas-hero:before{
  content:"";position:absolute;right:22%;top:18px;width:34%;height:140px;opacity:.38;
  background-image:radial-gradient(#7dd3fc 1.15px, transparent 1.15px);background-size:18px 18px;
}
.saas-hero:after{
  content:"";position:absolute;right:31%;bottom:26px;width:170px;height:118px;border-radius:26px;
  background:linear-gradient(145deg,#0ea5e9,#006fe6);box-shadow:0 24px 42px rgba(0,119,230,.28);
  clip-path:polygon(28% 0,72% 0,78% 12%,78% 100%,22% 100%,22% 12%);
  opacity:.95;
}
.hero-content{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;gap:30px;}
.hero-left{display:flex;align-items:center;gap:20px;}
.hero-logo{width:76px;height:76px;border-radius:24px;background:linear-gradient(145deg,#0ea5e9,#006fe6);display:flex;align-items:center;justify-content:center;color:#fff!important;font-weight:900;font-size:30px;box-shadow:0 18px 38px rgba(2,132,199,.30);}
.hero-title{font-size:56px;font-weight:900;color:#0277df!important;letter-spacing:-2.5px;line-height:.95;margin:0;}
.hero-sub{font-size:22px;font-weight:800;color:#071a3d!important;margin-top:8px;}
.hero-line{font-size:16px;color:#365a7d!important;margin-top:9px;}.hero-line b{color:#0077e6!important;}
.hero-pills{display:flex;gap:14px;align-items:center;flex-wrap:wrap;justify-content:flex-end;}
.hero-pill{width:96px;height:98px;border-radius:20px;background:rgba(255,255,255,.94);border:1px solid #dbeafe;box-shadow:0 14px 30px rgba(2,132,199,.12);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;font-size:11px;font-weight:800;color:#071a3d!important;}
.hero-pill .ico{font-size:29px;margin-bottom:8px;}

.mode-alert{border:1px solid #38bdf8;background:rgba(224,246,255,.80);border-radius:16px;padding:12px 16px;font-size:12px;color:#16406f!important;margin:0 0 18px 0;}.mode-alert strong{color:#0077e6!important;}.mode-alert.live{border-color:#22c55e;background:rgba(220,252,231,.75)}.mode-alert.live strong{color:#15803d!important;}

.kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(160px,1fr));gap:16px;margin:20px 0 22px 0;}
.kpi-card{background:#fff;border:1px solid #dbeafe;border-radius:22px;padding:20px 18px;min-height:118px;box-shadow:0 16px 35px rgba(2,132,199,.10);display:flex;align-items:center;gap:16px;transition:transform .18s ease,box-shadow .18s ease;}
.kpi-card:hover{transform:translateY(-5px);box-shadow:0 24px 45px rgba(2,132,199,.18);}
.kpi-icon{width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;flex-shrink:0;}.kpi-red{background:#ffe4e8;color:#ef4444}.kpi-blue{background:#e0f2fe;color:#0077e6}.kpi-green{background:#dcfce7;color:#16a34a}.kpi-purple{background:#ede9fe;color:#7c3aed}.kpi-orange{background:#ffedd5;color:#f97316}
.kpi-label{font-size:12px;color:#071a3d!important;font-weight:800;margin-bottom:5px;}.kpi-value{font-size:28px;color:#071a3d!important;font-weight:900;line-height:1;letter-spacing:-.04em;}.kpi-note{font-size:11px;font-weight:800;margin-top:8px}.note-red{color:#ef4444!important}.note-blue{color:#006fe6!important}.note-green{color:#16a34a!important}.note-orange{color:#f97316!important}.note-muted{color:#64748b!important}

/* Executive cards */
section.main div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]),
section.main div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]){background:#fff!important;border:1px solid #dbeafe!important;border-radius:22px!important;padding:12px 14px!important;box-shadow:0 15px 34px rgba(2,132,199,.09)!important;}
.insight-card{background:#fff;border:1px solid #dbeafe;border-left:6px solid #0ea5e9;padding:14px 18px;border-radius:16px;margin:10px 0;color:#102a56;font-size:13px;font-weight:700;box-shadow:0 10px 24px rgba(2,132,199,.07);}
.section-header{font-size:13px;color:#0077e6;font-weight:900;letter-spacing:.14em;text-transform:uppercase;margin:10px 0 14px 0;}

/* Tabs */
div[data-testid="stTabs"] button{background:#fff!important;color:#173b68!important;border:1px solid #dbeafe!important;border-radius:16px!important;padding:11px 18px!important;margin-right:9px!important;font-weight:800!important;font-size:12px!important;box-shadow:0 8px 18px rgba(2,132,199,.06)!important;transition:all .18s ease!important;}
div[data-testid="stTabs"] button:hover{background:#eff8ff!important;transform:translateY(-2px);}
div[data-testid="stTabs"] button[aria-selected="true"]{background:linear-gradient(145deg,#0ea5e9,#0077e6)!important;color:white!important;border-color:#0ea5e9!important;box-shadow:0 12px 24px rgba(0,119,230,.22)!important;}

/* Tables */
div[data-testid="stDataFrame"]{border-radius:16px!important;border:1px solid #dbeafe!important;overflow:hidden!important;}
div[data-testid="stDataFrame"] th{background:#f1f8ff!important;color:#0b3370!important;font-size:11px!important;font-weight:900!important;text-transform:uppercase!important;}
div[data-testid="stDataFrame"] td{background:#fff!important;color:#102a56!important;font-size:12px!important;}

/* Buttons */
.stButton>button,.stDownloadButton>button{background:linear-gradient(145deg,#0ea5e9,#0077e6)!important;color:#fff!important;border:none!important;border-radius:15px!important;font-weight:900!important;box-shadow:0 12px 24px rgba(0,119,230,.22)!important;transition:all .18s ease!important;}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);box-shadow:0 16px 32px rgba(0,119,230,.32)!important;}

details{background:#fff!important;border:1px solid #dbeafe!important;border-radius:18px!important;box-shadow:0 10px 24px rgba(2,132,199,.06)!important;} summary{font-weight:800!important;color:#102a56!important;} div[data-testid="stAlert"]{border-radius:16px!important;}
h1,h2,h3{color:#071a3d!important;font-weight:900!important;letter-spacing:-.04em!important;} p,span,label{color:#365a7d;}

@media(max-width:1200px){.hero-content{flex-direction:column;align-items:flex-start}.hero-pills{justify-content:flex-start}.kpi-grid{grid-template-columns:repeat(2,minmax(160px,1fr))}.hero-title{font-size:42px!important}}

.modebar {
  background: transparent !important;
  box-shadow: none !important;
}

.modebar-btn svg path {
  fill: #0f172a !important;   /* dark visible icons */
}

.modebar-btn:hover svg path {
  fill: #0284c7 !important;   /* hover color */
}
            
.modebar-btn {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.modebar-btn svg {
  vertical-align: middle !important;
}
.stButton>button {
  background: linear-gradient(145deg, #e0f2fe, #bae6fd) !important; /* light blue */
  color: #0f172a !important;  /* dark text */
  font-weight: 800 !important;
  border: 1px solid #93c5fd !important;
}

/* Hover effect */
.stButton>button:hover {
  background: linear-gradient(145deg, #bae6fd, #7dd3fc) !important;
  color: #020617 !important;
}

/* Make text inside cards bold + dark */
.stButton>button div {
  color: #0f172a !important;
  font-weight: 700 !important;
}

/* Optional: increase readability spacing */
.stButton>button {
  line-height: 1.4 !important;
}
                         
</style>""", unsafe_allow_html=True)

ACTION_COLOURS = {
    ACTION_URGENT_EXPIRY:"#D32F2F", ACTION_REDUCE:"#F57C00",
    ACTION_REORDER_NOW:"#1565C0",   ACTION_REVIEW:"#6A1B9A", ACTION_HOLD:"#2E7D32",
}
RISK_COLOURS = {"Critical":"#D32F2F","High":"#F57C00",
                "Medium":"#FBC02D","Low":"#388E3C","Expired":"#880E4F"}

UPLOAD_SCHEMA = {
    "sales": {
        "label":    "📈 Sales History  ←  upload  sales.csv",
        "required": ["sku_id", "date", "quantity_sold", "stock_level"],
        "example":  "sku_id | date       | quantity_sold | stock_level",
        "tip":      "One row per product per day. Minimum 14 days history per SKU.",
    },
    "batches": {
        "label":    "⏰ Batch & Expiry  ←  upload  batches.csv",
        "required": ["sku_id", "batch_id", "expiry_date", "stock_level", "unit_cost"],
        "example":  "sku_id | batch_id | expiry_date | stock_level | unit_cost",
        "tip":      "One row per batch. Critical for expiry risk detection.",
    },
    "suppliers": {
        "label":    "🚚 Supplier History  ←  upload  suppliers.csv",
        "required": ["sku_id", "supplier_id", "lead_time_days"],
        "example":  "sku_id | supplier_id | lead_time_days | actual_lead_time",
        "tip":      "One row per historical delivery order.",
    },
    "products": {
        "label":    "📦 Product Master  ←  upload  products.csv",
        "required": ["sku_id", "product_name"],
        "example":  "sku_id | product_name | category | unit_cost | selling_price",
        "tip":      "One row per product. Reference data.",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _detect_file_type(df_cols: list[str]) -> Optional[str]:
    """Guess which slot a file belongs to based on its columns."""
    col_set = set(df_cols)
    if {"date","quantity_sold","stock_level"}.issubset(col_set):
        return "sales"
    if {"batch_id","expiry_date"}.issubset(col_set):
        return "batches"
    if {"actual_lead_time","order_index"}.issubset(col_set) or        ({"supplier_id","lead_time_days"}.issubset(col_set) and "batch_id" not in col_set):
        return "suppliers"
    if {"product_name"}.issubset(col_set) and "date" not in col_set and "batch_id" not in col_set:
        return "products"
    return None


def _validate_csv(f, key):
    if f is None:
        return None, ""
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return None, f"Cannot read file: {e}"
    if len(df) == 0:
        return None, "File is empty — no data rows found."

    required = UPLOAD_SCHEMA[key]["required"]
    missing  = [c for c in required if c not in df.columns]

    if missing:
        # Auto-detect which slot this file actually belongs to
        detected = _detect_file_type(list(df.columns))
        if detected and detected != key:
            labels = {
                "sales":     "📈 Sales History",
                "batches":   "⏰ Batch & Expiry",
                "suppliers": "🚚 Supplier History",
                "products":  "📦 Product Master",
            }
            msg = (
                f"Wrong file in this slot!\n"
                f"This slot is for: {UPLOAD_SCHEMA[key]['label']}\n"
                f"Your file looks like: {labels.get(detected, detected)}\n\n"
                f"Please upload this file into the '{labels.get(detected,'correct')}' slot instead."
            )
        else:
            msg = (
                f"Missing required columns: {missing}\n"
                f"Your file has: {list(df.columns)}"
            )
        return None, msg

    return df, ""

def _to_temp(df):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(t.name, index=False); t.close(); return t.name

def safe_col(
    df: pd.DataFrame,
    col: str,
    default: float | str | None = 0,
) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)

def safe_val(row, col, default=0):
    try:
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except: return default

def fmt_usd(val):
    try: return f"${float(val):,.0f}"
    except: return "$0"

def safe_num_col(
    df: pd.DataFrame,
    col: str,
    default: float = 0.0,
) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")

def render_priority_actions(df, financial):
    if df is None or df.empty:
        return

    fin_roi = financial.get("roi_metrics", {}) if financial else {}

    critical_expiry = int((safe_num_col(df, EXPIRY_RISK, 0) >= HIGH_EXPIRY_RISK).sum())
    reorder_now = int((safe_col(df, ACTION) == ACTION_REORDER_NOW).sum())
    review_now = int((safe_col(df, ACTION) == ACTION_REVIEW).sum())
    estimated_savings = fmt_usd(fin_roi.get("cost_savings_usd", 0))

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#ffffff 0%,#eef9ff 100%);
        border:1px solid #bae6fd;
        border-radius:24px;
        padding:22px 26px;
        margin:18px 0 22px 0;
        box-shadow:0 18px 45px rgba(2,132,199,.10);
    ">
      <div style="display:flex;justify-content:space-between;gap:24px;align-items:flex-start;flex-wrap:wrap">
        <div>
          <div style="font-size:13px;font-weight:900;color:#0284c7;text-transform:uppercase;margin-bottom:8px">
            🚨 Priority Actions Today
          </div>
          <div style="font-size:26px;font-weight:900;color:#071a3d;margin-bottom:10px">
            Focus on SKUs that protect cash, availability, and expiry losses.
          </div>
          <div style="font-size:14px;color:#365a7d;font-weight:700;line-height:1.8">
            • <b style="color:#ef4444">{critical_expiry}</b> SKUs need expiry action<br>
            • <b style="color:#0077e6">{reorder_now}</b> SKUs should be reordered now<br>
            • <b style="color:#8b5cf6">{review_now}</b> SKUs need manual review
          </div>
        </div>
        <div style="
            min-width:240px;
            background:#ffffff;
            border:1px solid #dbeafe;
            border-radius:20px;
            padding:18px;
            text-align:center;
        ">
          <div style="font-size:12px;font-weight:900;color:#64748b">
            Estimated Savings
          </div>
          <div style="font-size:32px;font-weight:900;color:#16a34a">
            {estimated_savings}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Pipeline cache ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _run_cached(mode, sl, dis, sp, bp, sup_p, pp, cache_version):
    from pipeline import run_pipeline
    csv_paths = {}
    if sp:    csv_paths["sales"]     = sp
    if bp:    csv_paths["batches"]   = bp
    if sup_p: csv_paths["suppliers"] = sup_p
    if pp:    csv_paths["products"]  = pp
    return run_pipeline(mode=mode, csv_paths=csv_paths or None,
                        service_level=sl, disruption_factor=dis)

def get_results(mode, sl, dis, sp, bp, sup_p, pp):
    key = f"{mode}|{sl}|{dis}|{sp}|{bp}|{sup_p}|{pp}|cvs_synthetic_v5"
    if "results" not in st.session_state or st.session_state.get("_key") != key:
        with st.spinner("🔄 Running AI pipeline — approx. 60–90 seconds…"):
            st.session_state["results"] = _run_cached(
                mode, sl, dis, sp, bp, sup_p, pp, "cvs_synthetic_v5"
            )
            st.session_state["_key"]    = key
    return st.session_state["results"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-logo">PQ</div>
      <div>
        <div class="sidebar-name">PharmaIQ</div>
        <div class="sidebar-caption">Smart Pharma Inventory</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    st.sidebar.subheader("📂 Data Source")
    mode_choice = st.sidebar.radio("Choose input mode",
        ["🧪 Demo (Synthetic Data)", "📁 Upload Your CSV Files"])
    sp = bp = sup_p = pp = None

    if mode_choice == "📁 Upload Your CSV Files":
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "**Upload the 4 files below — each into its own slot.**  \n"
            "Each slot is clearly labelled with the exact filename.  \n"
            "Missing files fall back to synthetic data automatically."
        )
        st.sidebar.markdown(
            "> 📌 **Important:** Upload each file into the matching slot.  \n"
            "> Do NOT upload `batches.csv` into the Sales slot, etc."
        )
        st.sidebar.markdown("---")
        uploaded = {}
        for key, schema in UPLOAD_SCHEMA.items():
            # ── Clear section header with file name ──────────────────────
            st.sidebar.markdown(f"**{schema['label']}**")
            st.sidebar.caption(
                f"Required columns: `{'` · `'.join(schema['required'])}`"
            )
            st.sidebar.caption(f"Example row: `{schema['example']}`")
            st.sidebar.caption(f"💡 {schema['tip']}")

            f = st.sidebar.file_uploader(
                label=f"Drop {key}.csv here",
                type=["csv"],
                key=f"up_{key}",
                label_visibility="visible",   # show the label so user knows which file
            )
            df_up, err = _validate_csv(f, key)
            if f is None:
                st.sidebar.info(f"ℹ️ No file — using synthetic data for {key}")
            elif err:
                for line in err.split("\n"):
                    if line.strip():
                        st.sidebar.error(line)
            else:
                st.sidebar.success(
                    f"✅ {key}.csv loaded — {len(df_up):,} rows · "
                    f"{len(df_up.columns)} columns"
                )
                uploaded[key] = df_up
            st.sidebar.markdown("")
        sp    = _to_temp(uploaded["sales"])     if "sales"     in uploaded else None
        bp    = _to_temp(uploaded["batches"])   if "batches"   in uploaded else None
        sup_p = _to_temp(uploaded["suppliers"]) if "suppliers" in uploaded else None
        pp    = _to_temp(uploaded["products"])  if "products"  in uploaded else None
        if uploaded:
            with st.sidebar.expander("👁 Preview", expanded=False):
                for k, df_ in uploaded.items():
                    st.sidebar.markdown(f"**{k}**")
                    st.dataframe(df_.head(3), use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Parameters")
    sl  = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95, 0.01,
          help="0.95 = 95% no-stockout target. Higher = more safety stock.")
    dis = st.sidebar.slider("Supply Disruption Factor", 1.0, 2.0, 1.0, 0.1,
          help="1.0 = normal. 1.5 = moderate disruption. 2.0 = severe.")

    st.sidebar.divider()
    st.sidebar.subheader("🔍 Filters")
    rf  = st.sidebar.multiselect("Expiry Pressure",
          ["Expired","Critical","High","Medium","Low"],
          default=["Expired","Critical","High"])
    cf  = st.sidebar.text_input("Category contains", "")
    st.sidebar.divider()
    top_n = st.sidebar.slider("Top-N SKUs", 5, 20, TOP_N_SKUS)
    st.sidebar.caption("v1.2 — Supplier Module included")
    return "manual", sl, dis, rf, cf, top_n, sp, bp, sup_p, pp

# ── KPIs ──────────────────────────────────────────────────────────────────────
def set_action_filter(filter_type):
    st.session_state["kpi_filter"] = filter_type
    st.session_state["active_view_hint"] = "operations"

def render_kpis(df, fin_data):
    fin = fin_data["summary"]
    roi = fin_data["roi_metrics"]

    critical = int((safe_num_col(df, EXPIRY_RISK, 0) >= HIGH_EXPIRY_RISK).sum())
    reorder = int((safe_col(df, ACTION) == ACTION_REORDER_NOW).sum())
    savings = fmt_usd(roi.get("cost_savings_usd", 0))
    savings_pct = f"+{roi.get('savings_percent', 0):.1f}% vs baseline"
    total_cost = fmt_usd(fin["ai"]["total_cost"])
    cost_delta = fmt_usd(fin["ai"]["total_cost"] - fin["baseline"]["total_cost"])
    roi_pct = f"{roi.get('roi_percent', 0):.0f}%"
    payback = f"Payback: {roi.get('payback_months', 0):.1f} mo"

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        if st.button(
            f"⚠️\n\nCritical Expiry SKUs\n\n{critical}\n\nHigh priority items",
            key="kpi_critical_expiry",
            use_container_width=True,
        ):
            set_action_filter("critical_expiry")

    with k2:
        if st.button(
            f"📦\n\nReorder Now\n\n{reorder}\n\nImmediate action",
            key="kpi_reorder_now",
            use_container_width=True,
        ):
            set_action_filter("reorder_now")

    with k3:
        if st.button(
            f"💰\n\n30-Day Savings\n\n{savings}\n\n{savings_pct}",
            key="kpi_savings",
            use_container_width=True,
        ):
            set_action_filter("top_savings")

    with k4:
        st.button(
            f"📉\n\nAI Total Cost\n\n{total_cost}\n\n{cost_delta} vs baseline",
            key="kpi_total_cost",
            use_container_width=True,
            disabled=True,
        )

    with k5:
        st.button(
            f"🎯\n\nROI\n\n{roi_pct}\n\n{payback}",
            key="kpi_roi",
            use_container_width=True,
            disabled=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    active_filter = st.session_state.get("kpi_filter")
    if active_filter:
        labels = {
            "critical_expiry": "Critical Expiry SKUs",
            "reorder_now": "Reorder Now SKUs",
            "top_savings": "Top Savings Opportunities",
        }
        st.info(f"Active quick filter: {labels.get(active_filter, active_filter)}. Open Operations View to see filtered SKUs.")

        if st.button("Clear quick filter", key="clear_kpi_filter"):
            st.session_state.pop("kpi_filter", None)
            st.rerun()


# ── Executive Tab ─────────────────────────────────────────────────────────────
def render_executive_tab(df, financial, insights, top_n):
    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)
    for ins in insights:
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

    fin = financial["summary"]

    r1c1, r1c2, r1c3 = st.columns([1.25, 1, 1])
    with r1c1:
        st.subheader("Cost Breakdown: AI vs Baseline")
        cd = pd.DataFrame({"Category":["Expiry Loss","Holding Cost","Stockout Penalty"],
            "AI":[fin["ai"]["expiry_loss"],fin["ai"]["holding_cost"],fin["ai"]["stockout_penalty"]],
            "Baseline":[fin["baseline"]["expiry_loss"],fin["baseline"]["holding_cost"],fin["baseline"]["stockout_penalty"]]})
        f2=go.Figure()
        f2.add_bar(name="AI System",x=cd["Category"],y=cd["AI"],marker_color="#0EA5E9",width=.35)
        f2.add_bar(name="Baseline",x=cd["Category"],y=cd["Baseline"],marker_color="#CBD5E1",width=.35)
        f2.update_layout(barmode="group",height=320,margin=dict(t=18,b=20,l=10,r=10),yaxis_title="",legend=dict(orientation="h",font=dict(color="#102A56")),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="#FFFFFF",font=dict(color="#102A56"),xaxis=dict(gridcolor="#E6F2FC"),yaxis=dict(gridcolor="#E6F2FC",tickprefix="$"))
        st.plotly_chart(f2, use_container_width=True)

    with r1c2:
        st.subheader("Recommended Actions")
        if ACTION in df.columns:
            ac = df[ACTION].value_counts().reset_index(); ac.columns=["Action","Count"]
            fig=px.pie(ac,names="Action",values="Count",color="Action",color_discrete_map=ACTION_COLOURS,hole=0.52)
            fig.update_traces(textinfo="none", marker=dict(line=dict(color="#ffffff", width=2)))
            fig.update_layout(height=320,margin=dict(t=18,b=10,l=10,r=10),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)",legend=dict(font=dict(color="#102A56",size=11),orientation="v",x=1.02,y=.5),annotations=[dict(text=f"<b>{len(df)}</b><br>Total SKUs",x=.5,y=.5,showarrow=False,font=dict(size=14,color="#102A56"))])
            st.plotly_chart(fig, use_container_width=True)

    with r1c3:
        st.subheader("Expiry Risk Distribution")
        if EXPIRY_PRESSURE in df.columns:
            rc = df[EXPIRY_PRESSURE].value_counts().reset_index(); rc.columns=["Risk","Count"]
            fig_r=px.pie(rc,names="Risk",values="Count",color="Risk",color_discrete_map=RISK_COLOURS,hole=0.52)
            fig_r.update_traces(textinfo="none", marker=dict(line=dict(color="#ffffff", width=2)))
            fig_r.update_layout(height=320,margin=dict(t=18,b=10,l=10,r=10),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,0)",legend=dict(font=dict(color="#102A56",size=11),orientation="v",x=1.02,y=.5),annotations=[dict(text=f"<b>{len(df)}</b><br>Total SKUs",x=.5,y=.5,showarrow=False,font=dict(size=14,color="#102A56"))])
            st.plotly_chart(fig_r, use_container_width=True)

    r2c1, r2c2 = st.columns([1.35, 1])
    with r2c1:
        st.subheader(f"🔴 Top {top_n} High-Risk SKUs")
        cols  = [SKU_ID,PRODUCT_NAME,EXPIRY_PRESSURE,EXPIRY_RISK,STOCKOUT_RISK,"nearest_expiry",ACTION]
        avail = [c for c in cols if c in df.columns]
        top_r = df.nlargest(top_n,EXPIRY_RISK)[avail].copy() if EXPIRY_RISK in df.columns else df.head(top_n)[avail].copy()
        d = top_r.rename(columns={SKU_ID:"SKU",PRODUCT_NAME:"Product",EXPIRY_PRESSURE:"Pressure",EXPIRY_RISK:"Expiry Risk",STOCKOUT_RISK:"Stockout Risk","nearest_expiry":"Days to Expiry",ACTION:"Action"})
        for c in ["Expiry Risk","Stockout Risk"]:
            if c in d.columns: d[c]=d[c].apply(lambda x:f"{float(x):.0%}" if pd.notna(x) else "—")
        if "Days to Expiry" in d.columns: d["Days to Expiry"]=d["Days to Expiry"].apply(lambda x:f"{int(x)}d" if pd.notna(x) else "—")
        st.dataframe(d, use_container_width=True, hide_index=True)

    with r2c2:
        st.subheader(f"💎 Top {top_n} Savings Opportunities")
        ts = financial.get("top_savings",pd.DataFrame())
        if not ts.empty:
            dc=[SKU_ID,PRODUCT_NAME,"cost_saving",TOTAL_COST,"baseline_total_cost"]
            ts=ts[[c for c in dc if c in ts.columns]].copy()
            for c in ["cost_saving",TOTAL_COST,"baseline_total_cost"]:
                if c in ts.columns: ts[c]=ts[c].apply(lambda x:f"${float(x):,.0f}")
            ts.columns=[c.replace("_"," ").title() for c in ts.columns]
            st.dataframe(ts, use_container_width=True, hide_index=True)

# ── Operations Tab ────────────────────────────────────────────────────────────
def render_operations_tab(df, top_n, rf, cf):
    st.markdown('<div class="section-header">⚙️ Operations View</div>', unsafe_allow_html=True)

    cl, cr = st.columns([2, 1])

    with cl:
        st.subheader("📋 Action Queue")
        
        q = df.copy()

        kpi_filter = st.session_state.get("kpi_filter")

        if kpi_filter == "critical_expiry" and EXPIRY_RISK in q.columns:
            q = q[q[EXPIRY_RISK] >= HIGH_EXPIRY_RISK]
            st.success("Showing SKUs from KPI: Critical Expiry SKUs")

        elif kpi_filter == "reorder_now" and ACTION in q.columns:
            q = q[q[ACTION] == ACTION_REORDER_NOW]
            st.success("Showing SKUs from KPI: Reorder Now")

        elif kpi_filter == "top_savings" and "cost_saving" in q.columns:
            q = q.sort_values("cost_saving", ascending=False)
            st.success("Showing SKUs from KPI: Top Savings Opportunities")

        if rf and EXPIRY_PRESSURE in q.columns:
            q = q[q[EXPIRY_PRESSURE].isin(rf)]

        if cf and CATEGORY in q.columns:
            q = q[q[CATEGORY].str.contains(cf, case=False, na=False)]

        qc = [
            SKU_ID, PRODUCT_NAME, CATEGORY, ACTION,
            EXPIRY_RISK, STOCKOUT_RISK, RECOMMENDED_QTY,
            "nearest_expiry", STOCK_COVER
        ]

        qd = q[[c for c in qc if c in q.columns]].head(top_n).copy()

        for c in [EXPIRY_RISK, STOCKOUT_RISK]:
            if c in qd.columns:
                qd[c] = qd[c].apply(lambda x: f"{float(x):.0%}" if pd.notna(x) else "—")

        if "nearest_expiry" in qd.columns:
            qd["nearest_expiry"] = qd["nearest_expiry"].apply(
                lambda x: f"{int(x)}d" if pd.notna(x) else "—"
            )

        if STOCK_COVER in qd.columns:
            qd[STOCK_COVER] = qd[STOCK_COVER].apply(
                lambda x: f"{float(x):.1f}d" if pd.notna(x) else "—"
            )

        if RECOMMENDED_QTY in qd.columns:
            qd[RECOMMENDED_QTY] = qd[RECOMMENDED_QTY].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "0"
            )

        qd.columns = [c.replace("_", " ").title() for c in qd.columns]
        st.dataframe(qd, use_container_width=True, hide_index=True)

    with cr:
        if CATEGORY in df.columns and EXPIRY_PRESSURE in df.columns:
            st.subheader("🌡️ Expiry by Category")
            h = df.groupby([CATEGORY, EXPIRY_PRESSURE]).size().reset_index(name="count")

            if not h.empty:
                fh = px.bar(
                    h,
                    x="count",
                    y=CATEGORY,
                    color=EXPIRY_PRESSURE,
                    color_discrete_map=RISK_COLOURS,
                    orientation="h",
                    title=""
                )

                fh.update_layout(
                    height=350,
                    margin=dict(t=10, b=10),
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,.75)",
                    font=dict(color="#0f172a"),
                    xaxis=dict(gridcolor="#bae6fd"),
                    yaxis=dict(gridcolor="#bae6fd"),
                )

                st.plotly_chart(fh, use_container_width=True)

    st.divider()

    st.subheader("🔎 SKU Deep Dive")

    nm = dict(zip(df[SKU_ID], df[PRODUCT_NAME])) if PRODUCT_NAME in df.columns else {}
    sel = st.selectbox(
        "Select SKU",
        [f"{s} — {nm.get(s, '')}" for s in df[SKU_ID].tolist()]
    )

    if sel:
        sku = sel.split(" — ")[0]
        row = df[df[SKU_ID] == sku]

        if not row.empty:
            r = row.iloc[0]

            g1, g2, g3 = st.columns(3)

            risk_items = [
                ("Expiry Risk", safe_val(r, EXPIRY_RISK, 0)),
                ("Stockout Risk", safe_val(r, STOCKOUT_RISK, 0)),
                ("Supplier Risk", safe_val(r, SUPPLIER_RISK, 0)),
            ]

            for col_box, (title, val) in zip([g1, g2, g3], risk_items):
                score = float(val) * 100

                if score >= 70:
                    bar_color = "#ef4444"
                elif score >= 40:
                    bar_color = "#f97316"
                else:
                    bar_color = "#22c55e"

                with col_box:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={
                            "suffix": "%",
                            "font": {"size": 42, "color": "#071a3d"}
                        },
                        title={
                            "text": title,
                            "font": {"size": 18, "color": "#071a3d"}
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1,
                                "tickcolor": "#64748b"
                            },
                            "bar": {"color": bar_color},
                            "bgcolor": "white",
                            "borderwidth": 1,
                            "bordercolor": "#dbeafe",
                            "steps": [
                                {"range": [0, 40], "color": "#dcfce7"},
                                {"range": [40, 70], "color": "#fef3c7"},
                                {"range": [70, 100], "color": "#fee2e2"},
                            ],
                        },
                    ))

                    fig.update_layout(
                        height=270,
                        margin=dict(t=45, b=10, l=20, r=20),
                        paper_bgcolor="rgba(255,255,255,0)",
                        font=dict(color="#071a3d"),
                    )

                    st.plotly_chart(fig, use_container_width=True)

            reorder_qty = int(safe_val(r, RECOMMENDED_QTY, 0))

            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #dbeafe;
                    border-radius:18px;
                    padding:18px;
                    margin:12px 0 18px 0;
                    text-align:center;
                    box-shadow:0 12px 28px rgba(2,132,199,.08);
                ">
                    <div style="font-size:13px;color:#64748b;font-weight:800;">
                        Recommended Reorder Quantity
                    </div>
                    <div style="font-size:32px;color:#071a3d;font-weight:900;">
                        {reorder_qty:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("💬 AI Explanations")

            te = st.tabs(["📈 Demand", "⏰ Expiry", "🔄 Reorder", "🚚 Supplier"])

            with te[0]:
                st.info(r.get("explain_demand", "No explanation available"))

            with te[1]:
                st.warning(r.get("explain_expiry", "No explanation available"))

            with te[2]:
                st.info(r.get("explain_reorder", "No explanation available"))

            with te[3]:
                st.info(r.get("explain_supplier", "No explanation available"))

            if ACTION_REASON in r.index:
                st.subheader("🎯 Recommended Action")
                ac = safe_val(r, ACTION, "Review")
                col = ACTION_COLOURS.get(str(ac), "#666")

                st.markdown(
                    f'<span style="font-size:18px;color:{col};font-weight:800">{ac}</span>',
                    unsafe_allow_html=True
                )

                st.caption(safe_val(r, ACTION_REASON, ""))

# ── Historical Tab ────────────────────────────────────────────────────────────
def render_historical_tab(sales_df, batches_df, master_df):
    st.markdown('<div class="section-header">📊 Historical Data View</div>', unsafe_allow_html=True)
    st.caption("Analyse demand trends, seasonal patterns, and stock behaviour over time.")
    if sales_df is None or sales_df.empty:
        st.warning("No historical sales data available."); return
    sales_df = sales_df.copy()
    sales_df[DATE]     = pd.to_datetime(sales_df[DATE], errors="coerce")
    sales_df[QTY_SOLD] = pd.to_numeric(sales_df[QTY_SOLD], errors="coerce").fillna(0)
    sales_df = sales_df.dropna(subset=[DATE])
    today      = sales_df[DATE].max()
    total_days = (today - sales_df[DATE].min()).days

    cc1, cc2, cc3 = st.columns([1,1,2])
    with cc1:
        max_years  = min(3, max(1, total_days//365))
        yr_opts    = [f"Last {y} Year{'s' if y>1 else ''}" for y in range(1,max_years+1)]
        yr_labels  = {f"Last {y} Year{'s' if y>1 else ''}": y for y in range(1,max_years+1)}
        yr_sel     = st.selectbox("📅 Time Period", yr_opts, index=min(1,len(yr_opts)-1))
        n_years    = yr_labels[yr_sel]
        cutoff     = today - pd.DateOffset(years=n_years)
    with cc2:
        sku_list = sorted(sales_df[SKU_ID].unique().tolist())
        nm_map   = dict(zip(master_df[SKU_ID],master_df[PRODUCT_NAME])) if PRODUCT_NAME in master_df.columns else {}
        sku_disp = [f"{s} — {nm_map.get(s,'')}" for s in sku_list]
        sel_disp = st.selectbox("🔍 Select SKU", ["All SKUs"]+sku_disp)
        sel_sku  = None if sel_disp=="All SKUs" else sel_disp.split(" — ")[0]
    with cc3:
        agg_map = {"Daily":"D","Weekly":"W","Monthly":"ME"}
        agg_sel = st.radio("📐 Aggregation", list(agg_map.keys()), horizontal=True, index=1)
        agg_freq = agg_map[agg_sel]

    hist = sales_df[sales_df[DATE] >= cutoff].copy()
    if sel_sku: hist = hist[hist[SKU_ID]==sel_sku]
    if hist.empty: st.warning("No data for selected period."); return

    st.markdown("---")
    st.subheader(f"📌 Period Summary — {yr_sel}")
    total_sold = hist[QTY_SOLD].sum()
    avg_daily  = hist.groupby(DATE)[QTY_SOLD].sum().mean()
    peak_day   = hist.groupby(DATE)[QTY_SOLD].sum().idxmax()
    peak_val   = hist.groupby(DATE)[QTY_SOLD].sum().max()
    zero_days  = int((hist.groupby(DATE)[QTY_SOLD].sum()==0).sum())
    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric("📦 Total Units Sold",f"{int(total_sold):,}")
    k2.metric("📈 Avg Daily Demand",f"{avg_daily:.1f} units/day")
    k3.metric("🏆 Peak Day",f"{peak_day.strftime('%d %b %Y')}")
    k4.metric("🏆 Peak Volume",f"{int(peak_val):,} units")
    k5.metric("🚫 Zero-Demand Days",f"{zero_days}")
    st.markdown("---")

    r2l, r2r = st.columns(2)
    with r2l:
        st.subheader("📈 Demand Trend")
        if sel_sku:
            agg_d=hist.set_index(DATE)[QTY_SOLD].resample(agg_freq).sum().reset_index()
            agg_d.columns=[DATE,"quantity_sold"]
            agg_d["rolling_avg"]=agg_d["quantity_sold"].rolling(4,min_periods=1).mean()
            fig_t=px.line(agg_d,x=DATE,y="quantity_sold",labels={DATE:"Date","quantity_sold":"Units Sold"},
                          color_discrete_sequence=["#0EA5E9"])
            fig_t.add_scatter(x=agg_d[DATE],y=agg_d["rolling_avg"],mode="lines",name="4-period avg",
                              line=dict(color="#F57C00",dash="dash",width=2))
        else:
            agg_d=hist.groupby(DATE)[QTY_SOLD].sum().resample(agg_freq).sum().reset_index()
            agg_d.columns=[DATE,"quantity_sold"]
            fig_t=px.line(agg_d,x=DATE,y="quantity_sold",color_discrete_sequence=["#0EA5E9"])
        fig_t.update_layout(height=300,margin=dict(t=40,b=20),hovermode="x unified",paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
        st.plotly_chart(fig_t, use_container_width=True)
    with r2r:
        st.subheader("🏪 Stock Level Trend")
        if STOCK_LEVEL in hist.columns:
            if sel_sku:
                agg_s=hist.set_index(DATE)[STOCK_LEVEL].resample(agg_freq).last().reset_index()
            else:
                agg_s=hist.groupby(DATE)[STOCK_LEVEL].mean().resample(agg_freq).mean().reset_index()
            agg_s.columns=[DATE,"stock_level"]
            fig_s=px.area(agg_s,x=DATE,y="stock_level",color_discrete_sequence=["#22C55E"])
            fig_s.update_layout(height=300,margin=dict(t=40,b=20),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
            st.plotly_chart(fig_s, use_container_width=True)
        else: st.info("Stock level data not available.")

    st.markdown("---")
    st.subheader("📅 Year-over-Year Demand Comparison")
    yoy_df=sales_df.copy()
    if sel_sku: yoy_df=yoy_df[yoy_df[SKU_ID]==sel_sku]
    yoy_df["year"]=yoy_df[DATE].dt.year; yoy_df["month"]=yoy_df[DATE].dt.month
    yoy_agg=yoy_df.groupby(["year","month"])[QTY_SOLD].sum().reset_index()
    yoy_agg["month_name"]=pd.to_datetime(yoy_agg["month"],format="%m").dt.strftime("%b")
    month_order=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    years_avail=sorted(yoy_agg["year"].unique().tolist())
    if len(years_avail)>=2:
        fig_yoy=px.line(yoy_agg,x="month_name",y=QTY_SOLD,color="year",
            labels={"month_name":"Month",QTY_SOLD:"Units Sold"},
            category_orders={"month_name":month_order},
            color_discrete_sequence=["#0EA5E9","#F57C00","#22C55E"],markers=True)
        y1=yoy_agg[yoy_agg["year"]==years_avail[-2]]["quantity_sold"].sum()
        y2=yoy_agg[yoy_agg["year"]==years_avail[-1]]["quantity_sold"].sum()
        pct=(y2-y1)/y1*100 if y1>0 else 0
        arrow="📈" if pct>=0 else "📉"
        st.caption(f"{arrow} Total demand {'increased' if pct>=0 else 'decreased'} **{abs(pct):.1f}%** from {years_avail[-2]} → {years_avail[-1]}")
        fig_yoy.update_layout(height=340,margin=dict(t=20,b=20),hovermode="x unified",paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
        st.plotly_chart(fig_yoy, use_container_width=True)
    else: st.info("Need at least 2 years of data for YoY comparison.")

    st.markdown("---")
    s4l, s4r = st.columns(2)
    with s4l:
        st.subheader("🌡️ Seasonal Pattern")
        sea_df=sales_df.copy()
        if sel_sku: sea_df=sea_df[sea_df[SKU_ID]==sel_sku]
        sea_df["month_name"]=sea_df[DATE].dt.strftime("%b")
        sea_agg=sea_df.groupby("month_name")[QTY_SOLD].mean().reindex(month_order).reset_index()
        sea_agg.columns=["Month","Avg Daily Demand"]
        sea_agg["colour"]=sea_agg["Avg Daily Demand"].apply(
            lambda x:"#D32F2F" if x>=sea_agg["Avg Daily Demand"].quantile(0.75)
                     else ("#F57C00" if x>=sea_agg["Avg Daily Demand"].quantile(0.50) else "#0EA5E9"))
        fig_sea=px.bar(sea_agg,x="Month",y="Avg Daily Demand",color="colour",color_discrete_map="identity")
        fig_sea.update_layout(height=300,margin=dict(t=20,b=20),showlegend=False,paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
        st.plotly_chart(fig_sea, use_container_width=True)
    with s4r:
        st.subheader("⚡ Sales Velocity Trend")
        if sel_sku:
            vel=hist.set_index(DATE)[QTY_SOLD].resample("D").sum().rolling(30,min_periods=7).mean().reset_index()
            vel.columns=[DATE,"velocity"]
            fig_v=px.line(vel,x=DATE,y="velocity",color_discrete_sequence=["#8B5CF6"])
            fig_v.update_layout(height=300,margin=dict(t=20,b=20),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            top5=hist.groupby(SKU_ID)[QTY_SOLD].mean().nlargest(5).index.tolist()
            vtop=hist[hist[SKU_ID].isin(top5)]
            if not vtop.empty:
                fig_v=px.line(vtop,x=DATE,y=QTY_SOLD,color=SKU_ID,title="Top 5 SKUs by Volume")
                fig_v.update_layout(height=300,margin=dict(t=40,b=20),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
                st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top vs Slow Movers")
    sku_vel=hist.groupby(SKU_ID)[QTY_SOLD].mean().reset_index().rename(columns={QTY_SOLD:"avg_daily"})
    sku_vel["product_name"]=sku_vel[SKU_ID].map(nm_map).fillna(sku_vel[SKU_ID])
    tm_l, tm_r = st.columns(2)
    with tm_l:
        top10=sku_vel.nlargest(10,"avg_daily")
        ft=px.bar(top10,x="avg_daily",y="product_name",orientation="h",
                  title="🏆 Top 10 Fastest-Moving",color="avg_daily",color_continuous_scale="Blues")
        ft.update_layout(height=370,margin=dict(t=40,b=10),showlegend=False,yaxis=dict(autorange="reversed"),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"))
        st.plotly_chart(ft, use_container_width=True)
    with tm_r:
        slow10=sku_vel.nsmallest(10,"avg_daily")
        fs=px.bar(slow10,x="avg_daily",y="product_name",orientation="h",
                  title="🐢 Top 10 Slowest-Moving",color="avg_daily",color_continuous_scale="Reds_r")
        fs.update_layout(height=370,margin=dict(t=40,b=10),showlegend=False,yaxis=dict(autorange="reversed"),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"))
        st.plotly_chart(fs, use_container_width=True)

    if batches_df is not None and not batches_df.empty:
        st.markdown("---")
        st.subheader("⏰ Batch Expiry Timeline")
        b=batches_df.copy()
        b[EXPIRY_DATE]=pd.to_datetime(b[EXPIRY_DATE],errors="coerce")
        b=b.dropna(subset=[EXPIRY_DATE])
        b[DAYS_TO_EXPIRY]=(b[EXPIRY_DATE]-pd.Timestamp("today").normalize()).dt.days
        b["product_name"]=b[SKU_ID].map(nm_map).fillna(b[SKU_ID])
        b["status"]=b[DAYS_TO_EXPIRY].apply(lambda d:"Expired/Critical" if d<=30 else("Warning" if d<=90 else "OK"))
        if sel_sku: b=b[b[SKU_ID]==sel_sku]
        if not b.empty:
            cmap={"Expired/Critical":"#D32F2F","Warning":"#F57C00","OK":"#22C55E"}
            hover_cols=["batch_id"] if "batch_id" in b.columns else [SKU_ID]
            fb=px.bar(b.sort_values(DAYS_TO_EXPIRY),x=DAYS_TO_EXPIRY,y="product_name",
                      color="status",color_discrete_map=cmap,orientation="h",
                      title="Active Batch Expiry — Days Remaining",hover_data=hover_cols+[STOCK_LEVEL])
            fb.add_vline(x=30,line_dash="dash",line_color="#D32F2F",annotation_text="30d")
            fb.add_vline(x=90,line_dash="dash",line_color="#F57C00",annotation_text="90d")
            fb.update_layout(height=max(350,len(b)*18),margin=dict(t=50,b=20),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
            st.plotly_chart(fb, use_container_width=True)

    st.markdown("---")
    with st.expander("📥 Download Raw Historical Data"):
        dl_df=hist[[DATE,SKU_ID,QTY_SOLD,STOCK_LEVEL]].copy()
        dl_df[DATE]=dl_df[DATE].dt.strftime("%Y-%m-%d")
        st.dataframe(dl_df.head(200), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download as CSV",
            data=dl_df.to_csv(index=False).encode(),
            file_name=f"pharmaiq_history_{yr_sel.replace(' ','_').lower()}.csv",
            mime="text/csv")

# ── Technical Tab ─────────────────────────────────────────────────────────────
def render_technical_tab(models, df):
    st.markdown('<div class="section-header">🔬 Technical View</div>', unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.subheader("📊 Demand Forecasting Model")
        dm=models.get("demand")
        if dm:
            m=dm.get_metrics()
            if m:
                m1,m2,m3=st.columns(3)
                m1.metric("CV MAE",f"{m.get('cv_mae','N/A')}"); m2.metric("CV Std",f"{m.get('cv_std','N/A')}"); m3.metric("Train Rows",f"{m.get('n_train',0):,}")
                fi=dm.get_feature_importance()
                if not fi.empty:
                    ff=px.bar(fi.head(10),x="importance",y="feature",orientation="h",title="Feature Importance",color="importance",color_continuous_scale="Blues")
                    ff.update_layout(height=320,margin=dict(t=40,b=10),showlegend=False,yaxis=dict(autorange="reversed",gridcolor="#bae6fd"),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"))
                    st.plotly_chart(ff, use_container_width=True)
            else: st.info("Moving Average baseline used")
    with cr:
        st.subheader("⏰ Expiry Risk Model")
        em=models.get("expiry")
        if em:
            e=em.get_metrics()
            if e:
                e1,e2,e3=st.columns(3)
                e1.metric("CV AUC",f"{e.get('cv_auc_mean','N/A')}"); e2.metric("AUC Std",f"{e.get('cv_auc_std','N/A')}")
                pr=e.get("positive_rate"); e3.metric("Positive Rate",f"{pr:.1%}" if isinstance(pr,float) else "N/A")
                efi=em.get_feature_importance()
                if not efi.empty:
                    fe=px.bar(efi.head(8),x="importance",y="feature",orientation="h",title="Expiry Feature Importance",color="importance",color_continuous_scale="Reds")
                    fe.update_layout(height=290,margin=dict(t=40,b=10),showlegend=False,yaxis=dict(autorange="reversed",gridcolor="#bae6fd"),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"))
                    st.plotly_chart(fe, use_container_width=True)
            else: st.info("Rule-based scoring used")
    st.divider()
    st.subheader("📉 Risk Score Distributions")
    dc=st.columns(3)
    for i,(col,title,colour) in enumerate([(EXPIRY_RISK,"Expiry Risk","#D32F2F"),(STOCKOUT_RISK,"Stockout Risk","#0EA5E9"),(SUPPLIER_RISK,"Supplier Risk","#F57C00")]):
        with dc[i]:
            if col in df.columns:
                fd=px.histogram(df,x=col,nbins=20,title=f"{title} Distribution",color_discrete_sequence=[colour])
                fd.update_layout(height=250,margin=dict(t=40,b=20,l=20,r=10),xaxis_title="Score",yaxis_title="SKUs",bargap=0.1,paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"))
                st.plotly_chart(fd, use_container_width=True)
    st.divider()
    if EXPIRY_RISK in df.columns and STOCKOUT_RISK in df.columns:
        st.subheader("🎯 Portfolio View: Expiry Risk vs Stockout Risk")
        hn=PRODUCT_NAME if PRODUCT_NAME in df.columns else SKU_ID
        cc=CATEGORY if CATEGORY in df.columns else None
        sc=TOTAL_COST if TOTAL_COST in df.columns else None
        fs=px.scatter(df,x=STOCKOUT_RISK,y=EXPIRY_RISK,color=cc,size=sc,size_max=20,hover_name=hn,
            hover_data=[ACTION] if ACTION in df.columns else None,
            title="Each bubble = one SKU  |  Size = Total Cost",
            labels={STOCKOUT_RISK:"Stockout Risk",EXPIRY_RISK:"Expiry Risk"})
        fs.add_hline(y=HIGH_EXPIRY_RISK,line_dash="dash",line_color="#D32F2F",annotation_text="High Expiry Threshold")
        fs.add_vline(x=HIGH_STOCKOUT_RISK,line_dash="dash",line_color="#0EA5E9",annotation_text="High Stockout Threshold")
        fs.update_layout(height=450,margin=dict(t=50,b=30),paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(255,255,255,.75)",font=dict(color="#0f172a"),xaxis=dict(gridcolor="#bae6fd"),yaxis=dict(gridcolor="#bae6fd"),legend=dict(bgcolor="rgba(255,255,255,0)",font=dict(color="#0f172a")))
        st.plotly_chart(fs, use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mode, sl, dis, rf, cf, top_n, sp, bp, sup_p, pp = render_sidebar()

    # ── Page title ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="saas-hero">
      <div class="hero-content">
        <div class="hero-left">
          <div class="hero-logo">PQ</div>
          <div>
            <div class="hero-title">PharmaIQ</div>
            <div class="hero-sub">Smart Pharma Inventory</div>
            <div class="hero-line">Right stock. Right time. <b>Zero waste.</b></div>
          </div>
        </div>
        <div class="hero-pills">
          <div class="hero-pill"><div class="ico">🧠</div>AI<br>Insights</div>
          <div class="hero-pill"><div class="ico">📈</div>Demand<br>Forecast</div>
          <div class="hero-pill"><div class="ico">🛡️</div>Expiry<br>Control</div>
          <div class="hero-pill"><div class="ico">🛒</div>Smart<br>Reorder</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    n_up = sum(1 for p in [sp,bp,sup_p,pp] if p)
    if n_up:
        st.markdown(f"""<div class="mode-alert live">
          <strong>✅ Live Data</strong>
          <span> — Running on your uploaded data ({n_up} file(s) active).
          Missing files use synthetic fallback automatically.</span></div>""",
          unsafe_allow_html=True)
    else:
        st.markdown("""<div class="mode-alert">
          <strong>🧪 Demo Mode</strong>
          <span> — Running on built-in synthetic data.
          Switch to <strong>'Upload Your CSV Files'</strong>
          in the sidebar to analyse your own inventory.</span></div>""",
          unsafe_allow_html=True)
    st.divider()

    try:
        results = get_results(mode, sl, dis, sp, bp, sup_p, pp)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Run `pip install -r requirements.txt` and restart.")
        st.stop()

    df         = results.get("master_df",  pd.DataFrame())
    sales_df   = results.get("sales_df",   pd.DataFrame())
    batches_df = results.get("batches_df", pd.DataFrame())
    fin        = results.get("financial",  {})
    models     = results.get("models",     {})
    insights   = results.get("insights",   [])

    if df.empty:
        st.warning("No data loaded."); st.stop()

    if fin: 
        render_priority_actions(df, fin)
        render_kpis(df, fin)

    st.divider()

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🏢 Executive View",
        "⚙️ Operations View",
        "📊 Historical View",
        "🛡️ Risk & Compliance",
        "🏭 Supplier Comparison",
        "🔬 Technical View",
    ])

    with t1: render_executive_tab(df, fin, insights, top_n)
    with t2: render_operations_tab(df, top_n, rf, cf)
    with t3: render_historical_tab(sales_df, batches_df, df)
    with t4:
        if _RISK_TAB_AVAILABLE:
            _sup_df = None
            try:
                import os as _os
                _sup_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "sample_data", "suppliers.csv")
                if _os.path.exists(_sup_path):
                    _sup_df = pd.read_csv(_sup_path)
            except Exception:
                pass
            render_risk_compliance_tab(master_df=df, financial=fin, suppliers_df=_sup_df)
        else:
            st.error("Risk compliance module not found. Make sure dashboard/risk_compliance_tab.py exists.")
    with t5:
        try:
            import importlib

            supplier_tab = importlib.import_module("supplier_module.dashboard.supplier_tab")
            supplier_tab.render_supplier_tab()
        except Exception as e:
            st.error(f"Supplier module error: {e}")
            st.info("Make sure supplier_module/ folder is inside your pharmaiq_final/ directory.")
    with t6: render_technical_tab(models, df)

    st.divider()
    st.markdown("""<div style="text-align:center;padding:16px;margin-top:8px;
      border-top:1px solid #bae6fd;color:#64748b;font-size:11px">
      PharmaIQ &nbsp;·&nbsp; Smart Inventory Intelligence for Pharma SMEs &nbsp;·&nbsp;
      Reduce expiry losses · Improve availability · Drive profit
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
