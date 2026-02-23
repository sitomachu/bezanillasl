from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


MAPEO_CATEGORIAS_OSM: Dict[str, Dict[str, object]] = {
    "playa": {"natural": "beach"},
    "supermercado": {"shop": "supermarket"},
    "colegio": {"amenity": "school"},
    "hospital": {"amenity": "hospital"},
    "farmacia": {"amenity": "pharmacy"},
}


def _normalizar_features(features: pd.DataFrame) -> List[Tuple[str, float, float]]:
    filas: List[Tuple[str, float, float]] = []
    if features.empty:
        return filas

    for row in features.itertuples():
        lat = getattr(row, "lat", None)
        lon = getattr(row, "lon", None)
        if lat is None or lon is None:
            geom = getattr(row, "geometry", None)
            if geom is None or getattr(geom, "is_empty", True):
                continue
            punto = geom.representative_point()
            lat = float(punto.y)
            lon = float(punto.x)
        nombre = getattr(row, "name", None) or "sin_nombre"
        filas.append((str(nombre), float(lat), float(lon)))
    return filas


def _bbox_desde_centro_radio_m(center: str, radius_m: int) -> Tuple[float, float, float, float]:
    lat_s, lon_s = center.split(",", 1)
    lat = float(lat_s.strip())
    lon = float(lon_s.strip())
    delta_lat = float(radius_m) / 111_320.0
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    delta_lon = float(radius_m) / (111_320.0 * abs(cos_lat))
    norte = lat + delta_lat
    sur = lat - delta_lat
    este = lon + delta_lon
    oeste = lon - delta_lon
    return norte, sur, este, oeste


def _obtener_pois_por_categoria(
    *,
    categorias: List[str],
    norte: float,
    sur: float,
    este: float,
    oeste: float,
) -> Dict[str, List[Tuple[str, float, float]]]:
    try:
        import osmnx as ox
    except Exception as exc:
        raise ImportError(
            "No se pudo importar osmnx. Instala dependencias con: "
            "pip install -r src/geospatial_expansion/requirements.txt"
        ) from exc

    out: Dict[str, List[Tuple[str, float, float]]] = {}
    for categoria in categorias:
        cat = categoria.strip().lower()
        tags = MAPEO_CATEGORIAS_OSM.get(cat)
        t0 = time.perf_counter()
        print(f"[DESCARGA] categoria={cat} inicio", flush=True)
        if not tags:
            out[cat] = []
            print(
                f"[DESCARGA] categoria={cat} sin_mapeo_osm tiempo_s={time.perf_counter() - t0:.2f}",
                flush=True,
            )
            continue

        features = ox.features_from_bbox(bbox=(oeste, sur, este, norte), tags=tags)
        if features is None or len(features) == 0:
            out[cat] = []
            print(
                f"[DESCARGA] categoria={cat} encontrados=0 tiempo_s={time.perf_counter() - t0:.2f}",
                flush=True,
            )
            continue

        out[cat] = _normalizar_features(features)
        print(
            f"[DESCARGA] categoria={cat} encontrados={len(out[cat])} tiempo_s={time.perf_counter() - t0:.2f}",
            flush=True,
        )
    return out


def descargar_pois_desde_circulos_a_csv(
    *,
    circles: List[Tuple[str, str, int]],
    csv_pois_salida: Path,
    categorias: List[str],
) -> Path:
    t_inicio = time.perf_counter()
    print(f"[POIS] csv_salida={csv_pois_salida}", flush=True)
    print(f"[POIS] categorias={','.join(categorias)}", flush=True)
    print(f"[POIS] circulos={len(circles)}", flush=True)

    filas: List[Dict[str, object]] = []
    for nombre_circulo, center, radius_m in circles:
        print(f"[POIS] circulo={nombre_circulo} center={center} radio_m={radius_m} inicio", flush=True)
        norte, sur, este, oeste = _bbox_desde_centro_radio_m(center=center, radius_m=radius_m)
        pois = _obtener_pois_por_categoria(
            categorias=categorias,
            norte=norte,
            sur=sur,
            este=este,
            oeste=oeste,
        )
        total_circulo = 0
        for categoria, puntos in pois.items():
            total_circulo += len(puntos)
            for nombre, lat, lon in puntos:
                filas.append(
                    {
                        "circulo": nombre_circulo,
                        "categoria": categoria,
                        "nombre": nombre,
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
        print(f"[POIS] circulo={nombre_circulo} total_registros={total_circulo}", flush=True)

    out_df = pd.DataFrame(filas, columns=["circulo", "categoria", "nombre", "latitude", "longitude"])
    if not out_df.empty:
        out_df["categoria"] = out_df["categoria"].astype(str).str.strip().str.lower()
        out_df["nombre"] = out_df["nombre"].astype(str).str.strip()
        out_df["latitude"] = pd.to_numeric(out_df["latitude"], errors="coerce")
        out_df["longitude"] = pd.to_numeric(out_df["longitude"], errors="coerce")
        out_df = out_df.dropna(subset=["categoria", "nombre", "latitude", "longitude"])
        out_df = out_df.drop_duplicates(subset=["categoria", "nombre", "latitude", "longitude"])

    csv_pois_salida.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_pois_salida, index=False, encoding="utf-8")
    print(f"[POIS] total_registros_unicos={len(out_df)} tiempo_total_s={time.perf_counter() - t_inicio:.2f}", flush=True)
    return csv_pois_salida
