from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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

# Prioridad "Bezana-like" (repetidos) + Santander volumen + costa complementaria.
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_circles_from_file(path: Path, default_distance: int) -> List[Circle]:
    """
    Formato esperado:
    [
      {"name": "Santander", "center": "43.4623,-3.8099", "distance_m": 18000},
      ...
    ]
    """
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


def _default_circles(default_distance: int) -> List[Circle]:
    """
    Si en algún momento quieres hacer distance dinámico, puedes poner distance_m=None
    en DEFAULT_CIRCLES y heredará default_distance.
    """
    circles: List[Circle] = []
    for (n, c, d) in DEFAULT_CIRCLES:
        dist = d if d is not None else default_distance
        circles.append(Circle(name=n, center=c, distance_m=dist))
    return circles


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
        operation="rent",          # alquiler
        property_type="homes",     # viviendas
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
    output_csv_name: str,
) -> Path:
    """
    Ejecuta llamadas hasta agotar presupuesto de requests.
    Este script SOLO descarga JSON y delega JSON->CSV a clean_idealista.py.

    Estrategia:
      - Recorre círculos (sesgo Bezana-like/costa).
      - Pide página 1 siempre.
      - Si adaptive_pages: solo pide páginas 2..N si la página anterior vino llena (50 items).
      - Dedup global por propertyCode durante el run (para no inflar memoria / repetidos).
    """
    client = IdealistaClient()

    rid = _run_id()
    raw_dir = RAW_BASE / f"rent_homes_run_{rid}"
    out_dir = PROCESSED_BASE / f"rent_homes_run_{rid}"
    _ensure_dir(raw_dir)
    _ensure_dir(out_dir)

    manifest = {
        "run_id": rid,
        "operation": "rent",
        "property_type": "homes",
        "max_requests": max_requests,
        "max_pages_per_circle": max_pages_per_circle,
        "adaptive_pages": adaptive_pages,
        "circles": [circle.__dict__ for circle in circles],
        "output_csv_name": output_csv_name,
    }
    _write_json(raw_dir / "manifest.json", manifest)

    used = 0
    seen: Set[str] = set()

    circle_idx = 0
    while used < max_requests:
        circle = circles[circle_idx % len(circles)]
        circle_idx += 1

        # -------- Página 1 --------
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
        if isinstance(el1, list):
            for it in el1:
                code = str((it or {}).get("propertyCode", ""))
                if code:
                    seen.add(code)

        # -------- Páginas extra (2..N) --------
        resp_prev = resp1
        for page in range(2, max_pages_per_circle + 1):
            if used >= max_requests:
                break

            if adaptive_pages and not _is_full_page(resp_prev, MAX_ITEMS):
                break

            try:
                resp_prev = _search_one(
                    client,
                    circle=circle,
                    page=page,  # <-- FIX CRÍTICO: aquí era page=1 en tu versión
                    raw_dir=raw_dir,
                    tag=f"req{used+1:03d}",
                )
            except Exception as e:
                _write_json(
                    raw_dir / f"req{used+1:03d}__ERROR.json",
                    {"error": str(e), "circle": circle.__dict__, "page": page},
                )
                used += 1
                break

            used += 1

            el = resp_prev.get("elementList") or []
            if isinstance(el, list):
                for it in el:
                    code = str((it or {}).get("propertyCode", ""))
                    if code:
                        seen.add(code)
            else:
                break

    # --------------------------
    # Postprocesado: JSON -> CSV
    # --------------------------
    csv_path = clean_json_run(
        input_dir=raw_dir,
        output_dir=out_dir,
        output_filename=output_csv_name,
        only_cantabria=True,
        require_valid_size=True,
        require_latlon=True,
    )

    summary = {
        "used_requests": used,
        "unique_property_codes_seen": len(seen),
        "raw_dir": str(raw_dir),
        "processed_dir": str(out_dir),
        "csv_path": str(csv_path),
    }
    _write_json(out_dir / "summary.json", summary)

    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Idealista: requests automáticos para ALQUILER (homes) priorizando zona Bezana-like en Cantabria"
    )
    p.add_argument("--max-requests", type=int, default=100, help="Presupuesto de requests (por defecto 100)")
    p.add_argument("--max-pages-per-circle", type=int, default=3, help="Máx páginas por círculo (1-3 recomendado)")
    p.add_argument("--no-adaptive-pages", action="store_true", help="Si se activa, siempre intentará páginas 1..N")
    p.add_argument("--distance", type=int, default=DEFAULT_DISTANCE_M, help="Radio por defecto (m)")
    p.add_argument("--circles-file", type=str, default=None, help="JSON con círculos (name/center/distance_m)")
    p.add_argument(
        "--output-csv",
        type=str,
        default="rent_homes_cantabria_bezana_like_raw.csv",
        help="Nombre del CSV en data/processed/idealista/<run>/",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.circles_file:
        circles = _load_circles_from_file(Path(args.circles_file), default_distance=int(args.distance))
    else:
        circles = _default_circles(int(args.distance))

    out_dir = run(
        max_requests=int(args.max_requests),
        circles=circles,
        max_pages_per_circle=max(1, int(args.max_pages_per_circle)),
        adaptive_pages=(not args.no_adaptive_pages),
        output_csv_name=str(args.output_csv),
    )

    print(f"OK. Resultados en: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
