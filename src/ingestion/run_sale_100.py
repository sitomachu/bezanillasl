from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from src.ingestion.clean_idealista import clean_json_run
from src.ingestion.client import IdealistaClient


# =========================
# Config por defecto
# =========================

RAW_BASE = Path("data/raw/idealista")
PROCESSED_BASE = Path("data/processed/idealista")

DEFAULT_DISTANCE_M = 18000
MAX_ITEMS = 50  # Idealista search: max 50
SLEEP_S = 0.2   # pequeño delay por cortesía

DEFAULT_CIRCLES: List[Tuple[str, str, int]] = [
    # --- PRIORIDAD BEZANA-LIKE (repetidos) ---
    ("SantaCruzDeBezana", "43.4435,-3.9036", 12000),
    ("SantaCruzDeBezana", "43.4435,-3.9036", 12000),
    ("SantaCruzDeBezana", "43.4435,-3.9036", 12000),

    ("SotoDeLaMarina", "43.4620,-3.9080", 12000),
    ("SotoDeLaMarina", "43.4620,-3.9080", 12000),

    ("Liencres", "43.4480,-3.9700", 12000),
    ("Liencres", "43.4480,-3.9700", 12000),

    ("Pielagos_Boo", "43.4230,-3.9520", 14000),
    ("Pielagos_Boo", "43.4230,-3.9520", 14000),

    ("Camargo_Muriedas", "43.4150,-3.8540", 14000),
    ("Camargo_Muriedas", "43.4150,-3.8540", 14000),

    # --- VOLUMEN (Santander) ---
    ("Santander", "43.4623,-3.8099", 18000),
    ("Santander", "43.4623,-3.8099", 18000),

    # --- COSTA CERCANA COMPLEMENTARIA ---
    ("Somo", "43.4470,-3.7450", 14000),
    ("Suances", "43.4260,-4.0430", 14000),

    # --- COSTA LEJANA (baja prioridad, 1 vez) ---
    ("Laredo", "43.4090,-3.4160", 16000),
    ("CastroUrdiales", "43.3830,-3.2140", 16000),
]


@dataclass(frozen=True)
class Circle:
    name: str
    center: str  # "lat,lon"
    distance_m: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_circles_from_file(path: Path, default_distance: int) -> List[Circle]:
    data = json.loads(path.read_text(encoding="utf-8"))
    circles: List[Circle] = []
    for it in data:
        circles.append(
            Circle(
                name=str(it["name"]),
                center=str(it["center"]),
                distance_m=int(it.get("distance_m", default_distance)),
            )
        )
    return circles


def _default_circles(_: int) -> List[Circle]:
    return [Circle(name=n, center=c, distance_m=d) for (n, c, d) in DEFAULT_CIRCLES]


def _flatten(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(rows)

    cols = {c.lower(): c for c in df.columns}
    if "propertycode" in cols:
        df = df.drop_duplicates(subset=[cols["propertycode"]], keep="last")

    return df


def _is_full_page(resp: Dict[str, Any], max_items: int) -> bool:
    el = resp.get("elementList") or []
    return isinstance(el, list) and len(el) >= max_items


def _search_one(
    client: IdealistaClient,
    *,
    circle: Circle,
    page: int,
    raw_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    resp = client.search(
        country="es",
        operation="sale",
        property_type="homes",
        num_page=page,
        max_items=MAX_ITEMS,
        center=circle.center,
        distance=circle.distance_m,
        extra_params={
            "order": "publicationDate",
            "sort": "desc",
        },
    )
    _write_json(raw_dir / f"{tag}__{circle.name}__p{page:03d}.json", resp)
    time.sleep(SLEEP_S)
    return resp


def run(
    *,
    max_requests: int,
    circles: List[Circle],
    max_pages_per_circle: int,
    adaptive_pages: bool,
    verify_ssl: bool,
    raw_csv_name: str,
) -> Path:
    """
    Ejecuta llamadas hasta agotar presupuesto de requests y:
      - Guarda JSON en data/raw/idealista/...
      - Genera CSV raw en data/processed/idealista/... usando clean_json_run()
      - (Opcional) además genera un CSV filtrado (tu lógica actual)
    """
    # Tu IdealistaClient no acepta verify=..., así que verify_ssl solo se guarda en manifest
    client = IdealistaClient()

    rid = _run_id()
    raw_dir = RAW_BASE / f"sale_homes_run_{rid}"
    out_dir = PROCESSED_BASE / f"sale_homes_run_{rid}"
    _ensure_dir(raw_dir)
    _ensure_dir(out_dir)

    manifest = {
        "run_id": rid,
        "operation": "sale",
        "property_type": "homes",
        "max_requests": max_requests,
        "max_pages_per_circle": max_pages_per_circle,
        "adaptive_pages": adaptive_pages,
        "verify_ssl": verify_ssl,
        "circles": [circle.__dict__ for circle in circles],
        "raw_csv_name": raw_csv_name,
    }
    _write_json(raw_dir / "manifest.json", manifest)

    used = 0
    all_rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    circle_idx = 0
    while used < max_requests:
        circle = circles[circle_idx % len(circles)]
        circle_idx += 1

        if used >= max_requests:
            break

        try:
            resp1 = _search_one(client, circle=circle, page=1, raw_dir=raw_dir, tag=f"req{used+1:03d}")
        except Exception as e:
            _write_json(
                raw_dir / f"req{used+1:03d}__ERROR.json",
                {"error": str(e), "circle": circle.__dict__, "page": 1},
            )
            used += 1
            continue

        used += 1

        el1 = resp1.get("elementList") or []
        if isinstance(el1, list) and el1:
            for it in el1:
                code = str(it.get("propertyCode", ""))
                if code and code not in seen:
                    seen.add(code)
                    all_rows.append(it)

        # páginas extra
        resp_prev = resp1  # <-- evita bug de variable no definida
        if max_pages_per_circle >= 2:
            for page in range(2, max_pages_per_circle + 1):
                if used >= max_requests:
                    break

                if adaptive_pages and not _is_full_page(resp_prev, MAX_ITEMS):
                    break

                try:
                    resp_prev = _search_one(
                        client, circle=circle, page=page, raw_dir=raw_dir, tag=f"req{used+1:03d}"
                    )
                except Exception as e:
                    _write_json(
                        raw_dir / f"req{used+1:03d}__ERROR.json",
                        {"error": str(e), "circle": circle.__dict__, "page": page},
                    )
                    used += 1
                    break  # no sigas con más páginas de este círculo

                used += 1
                el = resp_prev.get("elementList") or []
                if isinstance(el, list) and el:
                    for it in el:
                        code = str(it.get("propertyCode", ""))
                        if code and code not in seen:
                            seen.add(code)
                            all_rows.append(it)
                else:
                    break

    # 1) CSV RAW SIEMPRE (sale aunque no tengas all_rows en memoria, porque lee JSON)
    raw_csv_path = None
    try:
        raw_csv_path = clean_json_run(input_dir=raw_dir, output_filename=raw_csv_name)
    except Exception as e:
        _write_json(out_dir / "clean_raw_error.json", {"error": str(e)})

    # 2) CSV filtrado (tu lógica actual) solo si hay rows en memoria
    if not all_rows:
        summary = {
            "used_requests": used,
            "unique_rows": 0,
            "note": "Sin resultados en memoria (pero revisa el CSV raw si se generó)",
            "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
        }
        _write_json(out_dir / "summary.json", summary)
        return out_dir

    df = _flatten(all_rows)

    cols = {c.lower(): c for c in df.columns}
    prov = cols.get("province")
    size = cols.get("size")
    lat = cols.get("latitude")
    lon = cols.get("longitude")

    if prov:
        df = df[df[prov] == "Cantabria"]

    if size:
        df = df[pd.to_numeric(df[size], errors="coerce").notna()]
        df = df[pd.to_numeric(df[size], errors="coerce") > 0]

    if lat and lon:
        df = df[df[lat].notna() & df[lon].notna()]

    out_csv = out_dir / "sale_homes_cantabria_coast.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "used_requests": used,
        "unique_rows_raw_in_memory": len(all_rows),
        "rows_after_filters": int(len(df)),
        "filtered_csv_path": str(out_csv),
        "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
        "columns_filtered": list(df.columns),
    }
    _write_json(out_dir / "summary.json", summary)

    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Idealista: 100 requests automáticos para venta (homes) en costa de Cantabria")
    p.add_argument("--max-requests", type=int, default=100, help="Presupuesto de requests (por defecto 100)")
    p.add_argument("--max-pages-per-circle", type=int, default=3, help="Máx páginas por círculo (1-3 recomendado)")
    p.add_argument("--no-adaptive-pages", action="store_true", help="Si se activa, siempre intentará páginas 1..N")
    p.add_argument("--distance", type=int, default=DEFAULT_DISTANCE_M, help="Radio por defecto (m)")
    p.add_argument("--circles-file", type=str, default=None, help="JSON con círculos (name/center/distance_m)")
    p.add_argument("--raw-csv-name", type=str, default="sale_homes_raw.csv", help="Nombre del CSV raw en processed/")
    return p


def main() -> int:
    args = build_parser().parse_args()

    verify_ssl = _env_bool("IDEALISTA_VERIFY_SSL", default=True)

    if args.circles_file:
        circles = _load_circles_from_file(Path(args.circles_file), default_distance=int(args.distance))
    else:
        circles = _default_circles(int(args.distance))

    out_dir = run(
        max_requests=int(args.max_requests),
        circles=circles,
        max_pages_per_circle=max(1, int(args.max_pages_per_circle)),
        adaptive_pages=(not args.no_adaptive_pages),
        verify_ssl=verify_ssl,
        raw_csv_name=str(args.raw_csv_name),
    )

    print(f"OK. Resultados en: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
