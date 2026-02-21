# Modulo de Expansion Geoespacial

Modulo simplificado en dos procesos independientes:
1. `run_descargar_pois.py`: descarga POIs desde OpenStreetMap y genera `csv_pois`.
2. `run_expandir.py`: toma tu dataset + `csv_pois` y calcula distancias al POI mas cercano.

Estructura:
- `download/osm_downloader.py`: logica de descarga de POIs.
- `expand/enricher.py`: logica de expansion y calculo de distancias.
- `common/distance.py`: utilidades geodesicas (Haversine).

## Requisitos

- CSV de entrada con coordenadas (`latitude`/`longitude` por defecto).
- Dependencias Python: `pandas`, `osmnx`.

Instalacion:

```bash
pip install -r src/geospatial_expansion/requirements.txt
```

## Proceso 1: Descargar POIs

1. Edita parametros en `src/geospatial_expansion/run_descargar_pois.py`.
2. La descarga usa circulos fijos (`DEFAULT_CIRCLES`) para Cantabria/Bezana.
3. No necesita `csv_entrada`.
4. Ejecuta:

```bash
python -m src.geospatial_expansion.run_descargar_pois
```

Salida esperada:
- CSV con columnas `circulo,categoria,nombre,latitude,longitude`.

## Proceso 2: Expandir dataset

1. Edita parametros en `src/geospatial_expansion/run_expandir.py`.
2. Ejecuta:

```bash
python -m src.geospatial_expansion.run_expandir
```

Salida esperada:
- Dataset enriquecido con:
- `nearest_<categoria>_name`
- `nearest_<categoria>_distance_m`

## Uso desde cuaderno (Jupyter)

```python
import pandas as pd
from src.geospatial_expansion import expandir_dataset

df = pd.read_csv("data/processed/idealista/sale_homes_run_20260218_173035/sale_homes_cantabria_bezana_like_raw.csv")

# tipo_poi puede ser string o lista
df_out = expandir_dataset(
    dataset=df,
    tipo_poi=["playa", "colegio"],
    csv_pois="data/processed/geo/pois_cantabria.csv",
)
```

Notas:
- Si no pasas `col_lat` y `col_lon`, el modulo intenta detectarlas automaticamente.
- Si no encuentra columnas de coordenadas, lanza error.
