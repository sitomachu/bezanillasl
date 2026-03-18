# Modulo de Expansion Geoespacial

Modulo con flujo mixto:
- Descarga de POIs: permitida por terminal.
- Expansion de dataset: solo por funcion (Python/notebook).

Funciones principales:
1. `descargar_pois_desde_circulos_a_csv(...)`: descarga POIs desde OpenStreetMap y genera `csv_pois`.
2. `agregar_distancias_minimas_poi(dataset, tipos_poi)`: toma tu DataFrame y agrega distancias minimas al POI mas cercano.

Guia de usuario operativa:
- `src/geospatial_expansion/geospatial_expansion_userguide.md`

Estructura:
- `download/osm_downloader.py`: logica de descarga de POIs.
- `expand/enricher.py`: logica de expansion y calculo de distancias.
- `common/distance.py`: utilidades geodesicas (Haversine).

## Requisitos

- CSV de entrada con coordenadas (`latitude`/`longitude` por defecto).
- Dependencias Python: `pandas`, `osmnx`.

Instalacion:

```bash
python -m pip install -r requirements.txt
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

1. Llamar desde notebook/Python usando `agregar_distancias_minimas_poi(dataset, tipos_poi)`.

Salida esperada:
- Dataset enriquecido con:
- `distancia_min_<categoria>_km`

## Uso desde cuaderno (Jupyter)

```python
import pandas as pd
from src.geospatial_expansion import agregar_distancias_minimas_poi

df = pd.read_csv("data/processed/idealista/sale_homes_run_20260218_173035/sale_homes_cantabria_bezana_like_raw.csv")

df_out = agregar_distancias_minimas_poi(
    df,
    ["playa", "colegio"],
)
```

Notas:
- Si no encuentra coordenadas en espanol/ingles, devuelve `UserWarning` y retorna el DataFrame sin cambios.
