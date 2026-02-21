from __future__ import annotations

from pathlib import Path

from src.geospatial_expansion.expand.enricher import enriquecer_csv_desde_pois

# -------------------------
# PARAMETROS (EDITAR AQUI)
# -------------------------
CSV_ENTRADA = Path(
    "data/processed/idealista/sale_homes_run_20260218_173035/sale_homes_cantabria_bezana_like_raw.csv"
)
CSV_POIS_SALIDA = Path("data/processed/geo/pois_cantabria.csv")
CSV_SALIDA_EXPANDIDO = Path("data/processed/geo/sale_homes_con_distancias.csv")
CATEGORIAS = ["playa", "supermercado", "colegio"]
COL_LAT = None  # Ejemplo: "LATITUD"
COL_LON = None  # Ejemplo: "LONGITUD"


def main() -> int:
    out = enriquecer_csv_desde_pois(
        csv_entrada=CSV_ENTRADA,
        csv_pois=CSV_POIS_SALIDA,
        csv_salida=CSV_SALIDA_EXPANDIDO,
        categorias=CATEGORIAS,
        col_lat=COL_LAT,
        col_lon=COL_LON,
    )
    print(f"OK: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
