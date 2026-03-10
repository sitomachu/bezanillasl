from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.idealistaAPI.config.idealista import PROCESSED_BASE, RAW_BASE


def extract_all_elements(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    json_files = sorted(input_dir.glob("*.json"))

    for fp in json_files:
        if fp.name.lower() == "manifest.json":
            continue

        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        element_list = payload.get("elementList")
        if isinstance(element_list, list):
            rows.extend([x for x in element_list if isinstance(x, dict)])

    return rows


def build_processed_path(input_dir: Path) -> Path:
    relative = input_dir.relative_to(RAW_BASE)
    processed_dir = PROCESSED_BASE / relative
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def _build_dedupe_key(df: pd.DataFrame) -> pd.Series:
    property_code = df.get("propertyCode", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    fallback = (
        df.get("price", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("size", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("latitude", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("longitude", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("address", df.get("streetName", pd.Series([""] * len(df), index=df.index))).fillna("").astype(str).str.strip().str.lower()
    )
    return property_code.where(property_code != "", fallback)


def clean_json_run(
    input_dir: Path,
    output_filename: str,
) -> Path:
    """
    Convierte todos los JSON de un run en un CSV raw.
    Devuelve la ruta del CSV generado.
    """

    rows = extract_all_elements(input_dir)

    if not rows:
        raise ValueError("No se encontraron anuncios en los JSON.")

    df = pd.json_normalize(rows)
    df["_dedupe_key"] = _build_dedupe_key(df)
    df = df.drop_duplicates(subset="_dedupe_key", keep="first").drop(columns=["_dedupe_key"])

    processed_dir = build_processed_path(input_dir)

    output_path = processed_dir / output_filename
    df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path

if __name__ == "__main__":
    # Carpeta donde están los JSON
    input_dir = Path("data/raw/idealistaAPI/raw/rent_homes_run_20260220_111903")

    # Carpeta de salida deseada
    PROCESSED_BASE = Path("data/raw/idealistaAPI/preprocess")

    # Nombre del CSV
    output_filename = "rent_homes_cantabria_bezana_like_raw.csv"

    # Ejecutar limpieza
    out = clean_json_run(input_dir=input_dir, output_filename=output_filename)

    print(f"CSV generado en: {out.resolve()}")
