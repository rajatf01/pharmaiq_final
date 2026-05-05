"""
data/data_layer.py
Dual-mode data ingestion: CSV/synthetic manual mode or Odoo ERP mode.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.schema import (
    SKU_ID, DATE, STOCK_LEVEL, BATCH_ID, EXPIRY_DATE,
    UNIT_COST, SELLING_PRICE, SUPPLIER_ID, SUPPLIER_NAME,
    LEAD_TIME_DAYS, PRODUCT_NAME, DAYS_TO_EXPIRY,
    REQUIRED_SALES_COLS, REQUIRED_BATCH_COLS, REQUIRED_SUPPLIER_COLS,
)

logger = logging.getLogger(__name__)


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {missing}")


def _coerce_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _add_days_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp("today").normalize()

    if EXPIRY_DATE in df.columns:
        df[DAYS_TO_EXPIRY] = (df[EXPIRY_DATE] - today).dt.days

    return df


class CSVLoader:
    def __init__(
        self,
        sales_path: str | None = None,
        batches_path: str | None = None,
        suppliers_path: str | None = None,
        products_path: str | None = None,
    ) -> None:
        self.paths: dict[str, str | None] = {
            "sales": sales_path,
            "batches": batches_path,
            "suppliers": suppliers_path,
            "products": products_path,
        }

    def _load_or_synthetic(self, key: str) -> pd.DataFrame:
        path = self.paths.get(key)

        if path and Path(path).exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded {key} from {path} ({len(df)} rows)")
            return df

        logger.warning(f"No CSV for '{key}' — falling back to synthetic data")

        from data.synthetic_data import generate_synthetic_dataset

        synthetic_data = generate_synthetic_dataset()
        return synthetic_data[key]

    def load(self) -> dict[str, pd.DataFrame]:
        sales = self._load_or_synthetic("sales")
        batches = self._load_or_synthetic("batches")
        suppliers = self._load_or_synthetic("suppliers")
        products = self._load_or_synthetic("products")

        sales = _coerce_dates(sales, [DATE])
        batches = _coerce_dates(batches, [EXPIRY_DATE])
        batches = _add_days_to_expiry(batches)

        _validate_columns(sales, REQUIRED_SALES_COLS, "sales")
        _validate_columns(batches, REQUIRED_BATCH_COLS, "batches")
        _validate_columns(suppliers, REQUIRED_SUPPLIER_COLS, "suppliers")

        return {
            "sales": sales,
            "batches": batches,
            "suppliers": suppliers,
            "products": products,
        }


class OdooLoader:
    def __init__(
        self,
        url: str,
        db: str,
        username: str,
        password: str,
        timeout: int = 30,
    ) -> None:
        self.url = url
        self.db = db
        self.username = username
        self.password = password
        self.timeout = timeout
        self._uid: int | None = None

    def _authenticate(self) -> int:
        import xmlrpc.client

        common = xmlrpc.client.ServerProxy(
            f"{self.url}/xmlrpc/2/common",
            allow_none=True,
        )

        uid = common.authenticate(self.db, self.username, self.password, {})

        if not uid:
            raise ConnectionError("Odoo authentication failed")

        uid_int = int(uid)
        self._uid = uid_int
        return uid_int

    def _execute(
        self,
        model: str,
        method: str,
        domain: list[list[Any]],
        fields: list[str],
    ) -> list[dict[str, Any]]:
        import xmlrpc.client

        if self._uid is None:
            self._authenticate()

        models = xmlrpc.client.ServerProxy(
            f"{self.url}/xmlrpc/2/object",
            allow_none=True,
        )

        result = models.execute_kw(
            self.db,
            self._uid,
            self.password,
            model,
            method,
            [domain],
            {"fields": fields, "limit": 10000},
        )

        return result if isinstance(result, list) else []

    def load(self) -> dict[str, pd.DataFrame]:
        try:
            self._authenticate()
            logger.info("Odoo authenticated successfully")

            prods = self._execute(
                "product.product",
                "search_read",
                [["active", "=", True]],
                ["id", "name", "categ_id", "list_price", "standard_price"],
            )

            quants = self._execute(
                "stock.quant",
                "search_read",
                [["location_id.usage", "=", "internal"]],
                ["product_id", "quantity", "lot_id"],
            )

            lots = self._execute(
                "stock.lot",
                "search_read",
                [],
                ["name", "product_id", "expiration_date"],
            )

            sups = self._execute(
                "product.supplierinfo",
                "search_read",
                [],
                ["product_id", "partner_id", "delay"],
            )

            moves = self._execute(
                "stock.move",
                "search_read",
                [["state", "=", "done"], ["picking_type_code", "=", "outgoing"]],
                ["product_id", "date", "product_qty"],
            )

            products_rows: list[dict[str, Any]] = []
            for r in prods:
                pid = r.get("id")
                products_rows.append({
                    SKU_ID: f"ODOO_{pid}",
                    PRODUCT_NAME: r.get("name", ""),
                    UNIT_COST: r.get("standard_price", 0),
                    SELLING_PRICE: r.get("list_price", 0),
                })

            products_df = pd.DataFrame(products_rows)

            stock_rows: list[dict[str, Any]] = []
            for r in quants:
                product_id = r.get("product_id")
                if isinstance(product_id, list) and product_id:
                    stock_rows.append({
                        SKU_ID: f"ODOO_{product_id[0]}",
                        STOCK_LEVEL: r.get("quantity", 0),
                    })

            stock_agg = pd.DataFrame(stock_rows)
            if not stock_agg.empty:
                stock_agg = stock_agg.groupby(SKU_ID, as_index=False)[STOCK_LEVEL].sum()

            batch_rows: list[dict[str, Any]] = []
            for r in lots:
                product_id = r.get("product_id")
                if isinstance(product_id, list) and product_id:
                    batch_rows.append({
                        SKU_ID: f"ODOO_{product_id[0]}",
                        BATCH_ID: r.get("name", ""),
                        EXPIRY_DATE: r.get("expiration_date"),
                    })

            batches_df = pd.DataFrame(batch_rows)
            if not batches_df.empty:
                batches_df = _coerce_dates(batches_df, [EXPIRY_DATE])
                batches_df = _add_days_to_expiry(batches_df)

                if not products_df.empty:
                    batches_df = batches_df.merge(
                        products_df[[SKU_ID, UNIT_COST, SELLING_PRICE]],
                        on=SKU_ID,
                        how="left",
                    )

                if not stock_agg.empty:
                    batches_df = batches_df.merge(stock_agg, on=SKU_ID, how="left")

            supplier_rows: list[dict[str, Any]] = []
            for r in sups:
                product_id = r.get("product_id")
                partner_id = r.get("partner_id")

                if (
                    isinstance(product_id, list)
                    and product_id
                    and isinstance(partner_id, list)
                    and partner_id
                ):
                    supplier_rows.append({
                        SKU_ID: f"ODOO_{product_id[0]}",
                        SUPPLIER_ID: f"ODOO_SUP_{partner_id[0]}",
                        SUPPLIER_NAME: partner_id[1],
                        LEAD_TIME_DAYS: r.get("delay", 14),
                    })

            suppliers_df = pd.DataFrame(supplier_rows)

            sales_rows: list[dict[str, Any]] = []
            for r in moves:
                product_id = r.get("product_id")
                if isinstance(product_id, list) and product_id:
                    sales_rows.append({
                        SKU_ID: f"ODOO_{product_id[0]}",
                        DATE: r.get("date"),
                        "quantity_sold": r.get("product_qty", 0),
                    })

            sales_df = pd.DataFrame(sales_rows)
            if not sales_df.empty:
                sales_df = _coerce_dates(sales_df, [DATE])

                if not stock_agg.empty:
                    sales_df = sales_df.merge(stock_agg, on=SKU_ID, how="left")
                    sales_df[STOCK_LEVEL] = sales_df[STOCK_LEVEL].fillna(0)

            logger.info("Odoo data loaded successfully")

            return {
                "sales": sales_df,
                "batches": batches_df,
                "suppliers": suppliers_df,
                "products": products_df,
            }

        except Exception as e:
            logger.error(f"Odoo load failed: {e} — falling back to CSV/synthetic")
            return CSVLoader().load()

    def write_alert(self, sku_id: str, message: str, severity: str = "warning") -> None:
        try:
            import xmlrpc.client

            if self._uid is None:
                self._authenticate()

            models = xmlrpc.client.ServerProxy(
                f"{self.url}/xmlrpc/2/object",
                allow_none=True,
            )

            models.execute_kw(
                self.db,
                self._uid,
                self.password,
                "mail.message",
                "create",
                [{
                    "subject": f"[PharmaIQ] {severity.upper()}: {sku_id}",
                    "body": message,
                    "message_type": "comment",
                }],
            )

        except Exception as e:
            logger.warning(f"Could not write alert to Odoo: {e}")


class DataLayer:
    def __init__(
        self,
        mode: str = "manual",
        odoo_config: dict[str, Any] | None = None,
        csv_paths: dict[str, str] | None = None,
    ) -> None:
        self.mode = mode

        if mode == "odoo" and odoo_config:
            self._loader: CSVLoader | OdooLoader = OdooLoader(**odoo_config)
        else:
            csv_paths = csv_paths or {}
            self._loader = CSVLoader(
                sales_path=csv_paths.get("sales"),
                batches_path=csv_paths.get("batches"),
                suppliers_path=csv_paths.get("suppliers"),
                products_path=csv_paths.get("products"),
            )

    def load(self) -> dict[str, pd.DataFrame]:
        data = self._loader.load()

        for key, df in data.items():
            if df.empty:
                raise ValueError(f"DataLayer: '{key}' DataFrame is empty after load")

        return data

    @property
    def supports_write(self) -> bool:
        return isinstance(self._loader, OdooLoader)

    def write_alert(self, sku_id: str, message: str, severity: str = "warning") -> None:
        if isinstance(self._loader, OdooLoader):
            self._loader.write_alert(sku_id, message, severity)
        else:
            logger.info(f"[ALERT — manual mode] {sku_id}: {message}")