from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from src.geospatial_expansion.common.distance import nearest_point


LAT_CANDIDATES = ("latitude", "lat", "LATITUD", "latitud")
LON_CANDIDATES = ("longitude", "lon", "lng", "LONGITUD", "longitud")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POIS_CSV = PROJECT_ROOT / "data" / "processed" / "geo" / "pois_cantabria.csv"
COORD_WARNING_MSG = (
    "No se detectaron columnas de coordenadas (latitud/longitud). "
    "Se probaron nombres en espanol e ingles: "
    "latitude/longitude, lat/lon, LATITUD/LONGITUD, latitud/longitud. "
    "Se devuelve el DataFrame sin cambios."
)


def _resolve_csv_pois_path(csv_pois: Path) -> Path:
    if csv_pois.exists():
        return csv_pois

    candidates = []
    if not csv_pois.is_absolute():
        candidates.append(PROJECT_ROOT / csv_pois)
        candidates.append(PROJECT_ROOT / ".." / csv_pois)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return csv_pois


def _find_first_present(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    colset = {c.lower(): c for c in columns}
    for cand in candidates:
        hit = colset.get(cand.lower())
        if hit:
            return hit
    return None


def detect_coordinate_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lat_col = _find_first_present(df.columns, LAT_CANDIDATES)
    lon_col = _find_first_present(df.columns, LON_CANDIDATES)
    if not lat_col or not lon_col:
        raise ValueError("No se pudieron detectar columnas de coordenadas en el dataset de entrada.")
    return lat_col, lon_col


def cargar_pois_desde_csv(
    *,
    csv_pois: Path,
    categorias: List[str],
) -> Dict[str, List[Tuple[str, float, float]]]:
    csv_pois = _resolve_csv_pois_path(Path(csv_pois))
    if not csv_pois.exists():
        raise FileNotFoundError(f"No existe el CSV de POIs: {csv_pois.resolve()}")
    df = pd.read_csv(csv_pois)
    required = {"categoria", "nombre", "latitude", "longitude"}
    missing = required.difference(df.columns)
    if missing:
        faltantes = ", ".join(sorted(missing))
        raise ValueError(f"El CSV de POIs no contiene columnas requeridas: {faltantes}.")

    limpio = df.copy()
    limpio["categoria"] = limpio["categoria"].astype(str).str.strip().str.lower()
    limpio["nombre"] = limpio["nombre"].astype(str).str.strip()
    limpio["latitude"] = pd.to_numeric(limpio["latitude"], errors="coerce")
    limpio["longitude"] = pd.to_numeric(limpio["longitude"], errors="coerce")
    limpio = limpio.dropna(subset=["categoria", "nombre", "latitude", "longitude"])

    out: Dict[str, List[Tuple[str, float, float]]] = {}
    for cat in categorias:
        key = cat.strip().lower()
        subset = limpio[limpio["categoria"] == key]
        out[key] = [
            (str(nombre), float(lat), float(lon))
            for nombre, lat, lon in subset[["nombre", "latitude", "longitude"]].itertuples(index=False, name=None)
        ]
    return out


def enriquecer_dataset_con_pois(
    df: pd.DataFrame,
    *,
    pois_por_categoria: Dict[str, List[Tuple[str, float, float]]],
    categorias: List[str],
    lat_col: str,
    lon_col: str,
    redondeo_metros: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    for categoria in categorias:
        cat = categoria.strip().lower()
        points = pois_por_categoria.get(cat, [])
        col_name = f"nearest_{cat}_name"
        col_dist = f"nearest_{cat}_distance_m"
        if not points:
            out[col_name] = None
            out[col_dist] = None
            continue

        names: List[Optional[str]] = []
        dists: List[Optional[float]] = []
        for lat, lon in out[[lat_col, lon_col]].itertuples(index=False, name=None):
            if pd.isna(lat) or pd.isna(lon):
                names.append(None)
                dists.append(None)
                continue
            nearest = nearest_point(float(lat), float(lon), points)
            if nearest is None:
                names.append(None)
                dists.append(None)
                continue
            nombre, metros = nearest
            names.append(nombre)
            dists.append(round(metros, redondeo_metros))

        out[col_name] = names
        out[col_dist] = dists

    return out


def enriquecer_csv_desde_pois(
    *,
    csv_entrada: Path,
    csv_pois: Path,
    csv_salida: Path,
    categorias: List[str],
    col_lat: Optional[str] = None,
    col_lon: Optional[str] = None,
) -> Path:
    df = pd.read_csv(csv_entrada)
    if not col_lat or not col_lon:
        col_lat, col_lon = detect_coordinate_columns(df)

    pois_por_categoria = cargar_pois_desde_csv(csv_pois=csv_pois, categorias=categorias)
    enriched = enriquecer_dataset_con_pois(
        df,
        pois_por_categoria=pois_por_categoria,
        categorias=categorias,
        lat_col=col_lat,
        lon_col=col_lon,
    )
    csv_salida.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(csv_salida, index=False, encoding="utf-8")
    return csv_salida


def expandir_dataset(
    dataset: pd.DataFrame,
    tipo_poi: Union[str, List[str]],
    *,
    csv_pois: Union[str, Path] = "data/processed/geo/pois_cantabria.csv",
    col_lat: Optional[str] = None,
    col_lon: Optional[str] = None,
    redondeo_metros: int = 1,
) -> pd.DataFrame:
    """
    API para cuadernos:
    - Recibe un DataFrame.
    - Detecta columnas de lat/lon si no se pasan.
    - Calcula la distancia minima al POI mas cercano para el/los tipos solicitados.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset debe ser un pandas.DataFrame")

    categorias = [tipo_poi] if isinstance(tipo_poi, str) else list(tipo_poi)
    categorias = [c.strip().lower() for c in categorias if str(c).strip()]
    if not categorias:
        raise ValueError("tipo_poi no puede estar vacio.")

    if not col_lat or not col_lon:
        try:
            col_lat, col_lon = detect_coordinate_columns(dataset)
        except ValueError:
            warnings.warn(COORD_WARNING_MSG, UserWarning, stacklevel=2)
            return dataset.copy()

    pois_por_categoria = cargar_pois_desde_csv(
        csv_pois=Path(csv_pois),
        categorias=categorias,
    )
    return enriquecer_dataset_con_pois(
        dataset,
        pois_por_categoria=pois_por_categoria,
        categorias=categorias,
        lat_col=col_lat,
        lon_col=col_lon,
        redondeo_metros=redondeo_metros,
    )


def agregar_distancias_minimas_poi(
    dataset: pd.DataFrame,
    tipos_poi: List[str],
) -> pd.DataFrame:
    """
    API simplificada para notebooks:
    - Entrada: DataFrame + lista de tipos de POI.
    - Salida: DataFrame original + una columna de distancia por tipo de POI.
    Columnas agregadas:
    - distancia_min_<tipo_poi>_km
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset debe ser un pandas.DataFrame")
    if not isinstance(tipos_poi, list):
        raise TypeError("tipos_poi debe ser una lista de strings")

    categorias = [str(c).strip().lower() for c in tipos_poi if str(c).strip()]
    if not categorias:
        raise ValueError("tipos_poi no puede estar vacio.")

    try:
        lat_col, lon_col = detect_coordinate_columns(dataset)
    except ValueError:
        warnings.warn(COORD_WARNING_MSG, UserWarning, stacklevel=2)
        return dataset.copy()
    pois_por_categoria = cargar_pois_desde_csv(
        csv_pois=DEFAULT_POIS_CSV,
        categorias=categorias,
    )

    out = dataset.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    for categoria in categorias:
        points = pois_por_categoria.get(categoria, [])
        col_dist = f"distancia_min_{categoria}_km"
        if not points:
            out[col_dist] = None
            continue

        dists: List[Optional[float]] = []
        for lat, lon in out[[lat_col, lon_col]].itertuples(index=False, name=None):
            if pd.isna(lat) or pd.isna(lon):
                dists.append(None)
                continue
            nearest = nearest_point(float(lat), float(lon), points)
            if nearest is None:
                dists.append(None)
                continue
            _, metros = nearest
            dists.append(round(metros / 1000.0, 3))

        out[col_dist] = dists

    return out
