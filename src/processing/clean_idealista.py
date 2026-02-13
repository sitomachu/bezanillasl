from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


RAW_BASE = Path("data/raw/idealista")
PROCESSED_BASE = Path("data/processed/idealista")


def ensure_dir(p: Path) -> None:
    """Crea directorio si no existe."""
    p.mkdir(parents=True, exist_ok=True)


def find_latest_run(raw_base: Path) -> Path:
    """
    Encuentra el último run_* (por orden alfabético, que coincide con timestamp).
    """
    runs = sorted([p for p in raw_base.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No hay runs en {raw_base.resolve()}")
    return runs[-1]


def read_jsonl(path: Path) -> list[dict]:
    """Lee JSONL (una fila JSON por línea)."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres y tipos.
    - columnas -> snake_case
    - conversión de tipos numéricos típicos
    - published -> datetime (si existe)
    """
    def snake(s: str) -> str:
        s = s.strip()
        s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
        s = re.sub(r"__+", "_", s)
        return s.lower().strip("_")

    df = df.copy()
    df.columns = [snake(c) for c in df.columns]

    # Convierte campos típicos si existen (depende del payload real)
    for col in ["price", "size", "rooms", "bathrooms", "floor", "distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicación:
    - preferente: propertyCode (ID de anuncio)
    - alternativa: url
    - fallback: combinación razonable
    """
    df = df.copy()

    if "propertycode" in df.columns:
        return df.drop_duplicates(subset=["propertycode"], keep="last")

    if "url" in df.columns:
        return df.drop_duplicates(subset=["url"], keep="last")

    key_cols = [c for c in ["title", "municipality", "district", "price"] if c in df.columns]
    if key_cols:
        return df.drop_duplicates(subset=key_cols, keep="last")

    return df.drop_duplicates(keep="last")


def clean_latest_run(
    raw_base: Path = RAW_BASE,
    processed_base: Path = PROCESSED_BASE,
) -> Path:
    """
    Pipeline de limpieza para el último run:
    1) carga todos los elementList.jsonl
    2) concatena en un único dataframe
    3) normaliza columnas y tipos
    4) deduplica
    5) guarda CSV y (si se puede) Parquet
    """
    run_dir = find_latest_run(raw_base)

    jsonl_files = list(run_dir.glob("**/elementList.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No encuentro elementList.jsonl en {run_dir.resolve()}")

    all_rows: list[dict] = []
    for f in jsonl_files:
        all_rows.extend(read_jsonl(f))

    if not all_rows:
        raise RuntimeError("No hay filas en los JSONL. La descarga probablemente devolvió 0 resultados.")

    # Aplana JSON anidado (si hay subcampos)
    df = pd.json_normalize(all_rows)

    # Metadato útil para trazabilidad
    df["run_id"] = run_dir.name

    df = normalize_columns(df)
    df = deduplicate(df)

    # Filtros de calidad mínimos (ajusta según tu análisis)
    if "price" in df.columns:
        df = df[df["price"].notna()]
    if "municipality" in df.columns:
        df = df[df["municipality"].notna()]

    out_dir = processed_base / run_dir.name
    ensure_dir(out_dir)

    # Salidas
    out_csv = out_dir / "idealista_rent_clean.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Parquet es útil, pero puede fallar si no tienes pyarrow/fastparquet
    out_parquet = out_dir / "idealista_rent_clean.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception:
        pass

    # Resumen (para el TFM esto es oro: cuántos registros, columnas, etc.)
    summary = {
        "run_dir": str(run_dir),
        "rows_raw": len(all_rows),
        "rows_clean": int(len(df)),
        "columns": list(df.columns),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_dir


def main() -> None:
    out = clean_latest_run()
    print(f"[OK] Limpieza finalizada. Salida en: {out}")


if __name__ == "__main__":
    main()
