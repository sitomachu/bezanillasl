"""
DataExporter corregido:
- Excel columnas > Z OK
- processed/raw coherente
- summary usa precio_mensual_equivalente si existe
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openpyxl.utils import get_column_letter

from config.settings import EXPORT_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataExporter:
    def __init__(self, data: List[Dict], output_dir: Optional[Path] = None):
        self.data = data
        self.output_dir = Path(output_dir) if output_dir else RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_output_dir(processed: bool, fallback_dir: Path) -> Path:
        out = PROCESSED_DATA_DIR if processed else fallback_dir
        out.mkdir(parents=True, exist_ok=True)
        return out

    def to_csv(self, filename: Optional[str] = None, processed: bool = False) -> Path:
        if not filename:
            filename = f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out = self._resolve_output_dir(processed, self.output_dir)
        path = out / filename
        pd.DataFrame(self.data).to_csv(
            path,
            index=False,
            sep=EXPORT_CONFIG["csv_separator"],
            encoding=EXPORT_CONFIG["csv_encoding"],
        )
        return path

    def to_excel(self, filename: Optional[str] = None, processed: bool = False, sheet_name: str = "Datos") -> Path:
        if not filename:
            filename = f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        out = self._resolve_output_dir(processed, self.output_dir)
        path = out / filename

        df = pd.DataFrame(self.data)
        with pd.ExcelWriter(path, engine=EXPORT_CONFIG["excel_engine"]) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns, start=1):
                series = df[col].astype(str)
                max_len = max(series.map(len).max() if len(series) else 0, len(str(col)))
                ws.column_dimensions[get_column_letter(idx)].width = min(max_len + 2, 50)

        return path

    def to_json(self, filename: Optional[str] = None, processed: bool = False, pretty: bool = True) -> Path:
        if not filename:
            filename = f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out = self._resolve_output_dir(processed, self.output_dir)
        path = out / filename
        with open(path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(self.data, f, ensure_ascii=False, indent=EXPORT_CONFIG["json_indent"], default=str)
            else:
                json.dump(self.data, f, ensure_ascii=False, default=str)
        return path

    def to_all_formats(self, base_filename: Optional[str] = None, processed: bool = False) -> Dict[str, Path]:
        if not base_filename:
            base_filename = f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return {
            "csv": self.to_csv(f"{base_filename}.csv", processed),
            "excel": self.to_excel(f"{base_filename}.xlsx", processed),
            "json": self.to_json(f"{base_filename}.json", processed),
        }

    def create_summary_report(self, filename: Optional[str] = None, processed: bool = True) -> Path:
        if not filename:
            filename = f"resumen_scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        out = self._resolve_output_dir(processed, self.output_dir)
        path = out / filename

        df = pd.DataFrame(self.data)
        price_col = "precio_mensual_equivalente" if "precio_mensual_equivalente" in df.columns else "precio_numerico"

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Datos Completos", index=False)

            if "portal" in df.columns and price_col in df.columns:
                agg = {price_col: ["count", "mean", "median", "min", "max"]}
                if "superficie_m2" in df.columns:
                    agg["superficie_m2"] = ["mean", "median"]
                if "habitaciones" in df.columns:
                    agg["habitaciones"] = ["mean"]

                df.groupby("portal").agg(agg).round(2).to_excel(writer, sheet_name="Resumen por Portal")

        return path
