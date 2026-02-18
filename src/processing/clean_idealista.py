from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


RAW_BASE = Path("data/raw/idealista")
PROCESSED_BASE = Path("data/processed/idealista")


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

    processed_dir = build_processed_path(input_dir)

    output_path = processed_dir / output_filename
    df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path
