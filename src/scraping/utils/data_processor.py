"""
DataProcessor corregido:
- normaliza a precio_mensual_equivalente
- valida por esa métrica
"""

import logging
import re
from typing import List, Dict, Optional

import pandas as pd

from config.settings import VALIDATION_RULES, INVALID_KEYWORDS

logger = logging.getLogger("scraping.data_processor")


class DataProcessor:
    def __init__(self, data: List[Dict]):
        self.raw_data = data
        self.processed_data: List[Dict] = []
        logger.info("Procesador inicializado con %d registros", len(data))

    def clean_all(self) -> List[Dict]:
        df = pd.DataFrame(self.raw_data)
        if df.empty:
            self.processed_data = []
            return []

        df = self._remove_duplicates(df)
        df = self._clean_text_fields(df)
        df = self._normalize_prices(df)
        df = self._validate_data(df)
        df = self._filter_invalid_listings(df)
        df = self._enrich_data(df)
        df = self._reorder_columns(df)

        self.processed_data = df.to_dict("records")
        return self.processed_data

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        if "url" in df.columns:
            df = df.drop_duplicates(subset=["url"], keep="first")
        return df

    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        for field in ["titulo", "ubicacion", "descripcion", "municipio", "provincia", "tipo_propiedad"]:
            if field in df.columns:
                df[field] = df[field].apply(self._clean_text)
        return df

    @staticmethod
    def _clean_text(text: Optional[str]) -> Optional[str]:
        if pd.isna(text) or text is None:
            return None
        text = re.sub(r"\s+", " ", str(text)).replace("\n", " ").replace("\r", " ").strip()
        return text if text else None

    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if "precio_numerico" in df.columns:
            df["precio_numerico"] = pd.to_numeric(df["precio_numerico"], errors="coerce")

        if "unidad_precio" not in df.columns:
            df["unidad_precio"] = None
            if "portal" in df.columns:
                df.loc[df["portal"].str.lower() == "airbnb", "unidad_precio"] = "noche"
                df.loc[df["portal"].str.lower().isin(["idealista", "fotocasa"]), "unidad_precio"] = "mes"

        df["precio_mensual_equivalente"] = pd.NA

        mask_mes = df["unidad_precio"] == "mes"
        df.loc[mask_mes, "precio_mensual_equivalente"] = df.loc[mask_mes, "precio_numerico"]

        mask_noche = df["unidad_precio"] == "noche"
        if mask_noche.any():
            df.loc[mask_noche, "precio_mensual_estimado"] = df.loc[mask_noche, "precio_numerico"] * 30
            df.loc[mask_noche, "precio_mensual_equivalente"] = df.loc[mask_noche, "precio_mensual_estimado"]

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "precio_mensual_equivalente" in df.columns:
            p = pd.to_numeric(df["precio_mensual_equivalente"], errors="coerce")
            ok = ((p >= VALIDATION_RULES["precio_min"]) & (p <= VALIDATION_RULES["precio_max"])) | p.isna()
            df = df[ok]

        if "superficie_m2" in df.columns:
            s = pd.to_numeric(df["superficie_m2"], errors="coerce")
            ok = ((s >= VALIDATION_RULES["superficie_min"]) & (s <= VALIDATION_RULES["superficie_max"])) | s.isna()
            df = df[ok]

        if "habitaciones" in df.columns:
            h = pd.to_numeric(df["habitaciones"], errors="coerce")
            ok = ((h >= VALIDATION_RULES["habitaciones_min"]) & (h <= VALIDATION_RULES["habitaciones_max"])) | h.isna()
            df = df[ok]

        return df

    def _filter_invalid_listings(self, df: pd.DataFrame) -> pd.DataFrame:
        if "titulo" not in df.columns:
            return df

        titles = df["titulo"].fillna("").astype(str)

        residencial_markers = ["piso", "casa", "chalet", "ático", "duplex", "dúplex", "estudio", "apartamento", "vivienda"]
        keep = pd.Series(True, index=df.index)

        has_res = pd.Series(False, index=df.index)
        for rm in residencial_markers:
            has_res = has_res | titles.str.contains(rf"\b{re.escape(rm)}\b", case=False, na=False)

        for kw in INVALID_KEYWORDS:
            kw_mask = titles.str.contains(rf"\b{re.escape(kw)}\b", case=False, na=False)
            keep = keep & ~(kw_mask & ~has_res)

        return df[keep]

    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "precio_mensual_equivalente" in df.columns and "superficie_m2" in df.columns:
            p = pd.to_numeric(df["precio_mensual_equivalente"], errors="coerce")
            s = pd.to_numeric(df["superficie_m2"], errors="coerce")
            df["precio_por_m2"] = (p / s).where((s > 0) & p.notna() & s.notna()).round(2)

        required = [c for c in ["precio_mensual_equivalente", "habitaciones", "superficie_m2"] if c in df.columns]
        df["datos_completos"] = df[required].notna().all(axis=1) if required else False

        if "precio_mensual_equivalente" in df.columns:
            p = pd.to_numeric(df["precio_mensual_equivalente"], errors="coerce")
            df["categoria_precio"] = pd.cut(
                p,
                bins=[0, 500, 750, 1000, 1500, float("inf")],
                labels=["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"],
            )

        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "portal",
            "fecha_scraping",
            "titulo",
            "tipo_propiedad",
            "precio",
            "precio_numerico",
            "unidad_precio",
            "precio_mensual_estimado",
            "precio_mensual_equivalente",
            "precio_por_m2",
            "categoria_precio",
            "habitaciones",
            "superficie_m2",
            "banos",
            "ubicacion",
            "municipio",
            "provincia",
            "descripcion",
            "url",
            "datos_completos",
        ]
        cols = [c for c in preferred if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        return df[cols + rest]
