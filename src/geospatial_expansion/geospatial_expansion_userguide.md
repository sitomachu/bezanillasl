# Geospatial Expansion User Guide

Guia operativa del modulo `src/geospatial_expansion`, separada de `idealistaAPI`.

## 1. Objetivo del modulo

Este modulo enriquece un dataset con una columna de distancia minima por categoria:

- `distancia_min_<categoria>_km`

Las distancias se calculan en linea recta (Haversine) y se devuelven en kilometros.

## 2. Requisitos

- Python 3.10+
- Dependencias del modulo:

```bash
python -m pip install -r requirements.txt
```

- Dataset de entrada con coordenadas:
  - Detecta automaticamente: `latitude/longitude`, `lat/lon`, `LATITUD/LONGITUD`, `latitud/longitud`

## 3. Flujo recomendado (2 pasos)

## Paso 1: Descargar POIs desde OSM (terminal o API Python)

Funcion:

- `descargar_pois_desde_circulos_a_csv(...)`

Ejemplo:

```python
from pathlib import Path
from src.geospatial_expansion import descargar_pois_desde_circulos_a_csv

circles = [
    ("SantaCruzDeBezana", "43.4435,-3.9036", 12000),
]

descargar_pois_desde_circulos_a_csv(
    circles=circles,
    csv_pois_salida=Path("data/processed/geo/pois_cantabria.csv"),
    categorias=["playa", "supermercado"],
)
```

Opcion terminal:

```bash
python -m src.geospatial_expansion.run_descargar_pois
```

Salida esperada:

- CSV en `data/processed/geo/pois_cantabria.csv` (o la ruta que configures)
- Columnas: `circulo,categoria,nombre,latitude,longitude`

Categorias OSM mapeadas por defecto:

- `playa`
- `supermercado`
- `colegio`
- `hospital`
- `farmacia`

## Paso 2: Enriquecer DataFrame con distancias minimas (API Python)

## 4. Como llamar al modulo desde notebook

Import recomendado en cuadernos (separado en 2 celdas):

Celda 1: preparar ruta del proyecto

```python
import sys
from pathlib import Path

cwd = Path.cwd().resolve()
project_root = next((p for p in [cwd, *cwd.parents] if (p / "src").exists()), None)
if project_root is None:
    raise RuntimeError("No se encontro la raiz del proyecto (carpeta 'src').")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

Celda 2: importar el modulo geoespacial

```python
from src.geospatial_expansion import agregar_distancias_minimas_poi
```

Funcion recomendada:

- `agregar_distancias_minimas_poi(dataset, tipos_poi)`

Firma:

- `dataset`: `pandas.DataFrame`
- `tipos_poi`: `list[str]` (ejemplo: `["playa", "supermercado"]`)

```python
import pandas as pd
from src.geospatial_expansion import agregar_distancias_minimas_poi

df = pd.read_csv(
    "data/processed/idealista/sale_homes_run_20260218_173035/"
    "sale_homes_cantabria_bezana_like_raw.csv"
)

df_out = agregar_distancias_minimas_poi(
    df,
    ["playa", "supermercado"],
)
```

Resultado esperado:

- Se anaden columnas:
  - `distancia_min_playa_km`
  - `distancia_min_supermercado_km`
- Las distancias salen en kilometros.
- Usa el database de POIs en `data/processed/geo/pois_cantabria.csv`.

Validaciones:

- Si no detecta coordenadas en el DataFrame, devuelve un `UserWarning` y retorna el DataFrame sin cambios.
- Si `tipos_poi` viene vacio, lanza error.

## 5. Estructura del modulo

- `src/geospatial_expansion/run_descargar_pois.py`: runner de descarga OSM -> CSV.
- `src/geospatial_expansion/download/osm_downloader.py`: descarga/normalizacion de POIs.
- `src/geospatial_expansion/expand/enricher.py`: deteccion de coordenadas y calculo de nearest POI.
- `src/geospatial_expansion/common/distance.py`: Haversine y busqueda del punto mas cercano.

## 6. Troubleshooting

- `ImportError: No se pudo importar osmnx`
  - Ejecuta: `python -m pip install -r requirements.txt`

- `No se pudieron detectar columnas de coordenadas`
  - El modulo devuelve `UserWarning` y retorna el DataFrame sin cambios.

- `El CSV de POIs no contiene columnas requeridas`
  - Verifica que tenga: `categoria,nombre,latitude,longitude`.

- Columnas de distancia vacias (`distancia_min_*_km`)
  - Revisa que la categoria exista en el CSV de POIs y que haya datos para esa categoria.
