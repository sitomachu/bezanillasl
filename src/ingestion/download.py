from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import typing as t

from src.ingestion.client import IdealistaClient


RAW_DIR_DEFAULT = Path("data/raw/idealista")


@dataclass(frozen=True)
class SearchCircle:
    """
    Define un círculo de búsqueda.
    - name: nombre para carpeta de salida (sin espacios idealmente)
    - center: "lat,lon"
    - distance_m: radio en metros
    """
    name: str
    center: str
    distance_m: int


def ensure_dir(p: Path) -> None:
    """Crea un directorio si no existe."""
    p.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict) -> None:
    """Guarda un JSON completo (respuesta completa de la API)."""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict]) -> None:
    """
    Guarda filas en formato JSON Lines (una fila JSON por línea).
    Útil para ir agregando resultados sin cargar todo en memoria.
    """
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now_tag() -> str:
    """Timestamp para identificar un run reproducible."""
    return time.strftime("%Y%m%d_%H%M%S")


def download_circles(
    *,
    client: IdealistaClient,
    circles: list[SearchCircle],
    out_dir: Path = RAW_DIR_DEFAULT,
    country: str = "es",
    operation: str = "rent",
    property_type: str = "homes",
    max_pages_per_circle: int = 3,
    max_items: int = 50,
    extra_params: dict[str, t.Any] | None = None,
    sleep_between_calls_s: float = 0.25,
) -> Path:
    """
    Descarga datos crudos a disco con un run_id único.

    Qué guarda:
    - manifest.json: configuración del run (para reproducibilidad)
    - run_xxx/<circle>/page_001.json ...: respuesta completa por página
    - run_xxx/<circle>/elementList.jsonl: agregado de anuncios (elementList)

    Ventajas:
    - mantienes raw “intocable”
    - puedes re-procesar sin volver a consumir cuota API
    """
    ensure_dir(out_dir)

    run_id = now_tag()
    run_dir = out_dir / f"run_{run_id}"
    ensure_dir(run_dir)

    # Manifiesto: fundamental para TFM (qué pediste, cuándo, con qué filtros)
    manifest = {
        "run_id": run_id,
        "country": country,
        "operation": operation,
        "property_type": property_type,
        "max_pages_per_circle": max_pages_per_circle,
        "max_items": max_items,
        "circles": [c.__dict__ for c in circles],
        "extra_params": extra_params or {},
    }
    dump_json(run_dir / "manifest.json", manifest)

    for circle in circles:
        circle_dir = run_dir / circle.name
        ensure_dir(circle_dir)

        # Archivo agregado (por círculo)
        agg_jsonl = circle_dir / "elementList.jsonl"
        if agg_jsonl.exists():
            agg_jsonl.unlink()  # evita mezclar runs

        for page in range(1, max_pages_per_circle + 1):
            # Llamada real a Idealista (consume 1 request)
            resp = client.search(
                country=country,
                operation=operation,
                property_type=property_type,
                num_page=page,
                max_items=max_items,
                center=circle.center,
                distance=circle.distance_m,
                extra_params=extra_params or {},
            )

            # Guardamos respuesta completa: útil si luego quieres campos fuera de elementList
            dump_json(circle_dir / f"page_{page:03d}.json", resp)

            # Agregamos elementList para explotación rápida
            element_list = resp.get("elementList") or []
            if isinstance(element_list, list) and element_list:
                save_jsonl(agg_jsonl, element_list)

            # Criterios de corte:
            # 1) si API te informa de totalPages y ya estás al final
            total_pages = resp.get("totalPages")
            actual_page = resp.get("actualPage")
            if total_pages and actual_page and int(actual_page) >= int(total_pages):
                break

            # 2) si la página viene vacía, no tiene sentido seguir
            if not element_list:
                break

            # Respetar la API (y tu cuota) con un pequeño sleep
            time.sleep(sleep_between_calls_s)

    return run_dir


def main() -> None:
    """
    Ejecuta:
      python -m src.ingestion.idealista.download

    Recomendación realista con 100 requests/mes:
    - 6 a 10 círculos
    - 1 a 3 páginas por círculo
    - maxItems=50
    """
    client = IdealistaClient()

    # Círculos ejemplo (aprox). Ajusta.
    circles = [
        SearchCircle("santander", "43.4623,-3.8099", 15000),
        SearchCircle("liencres_pielagos", "43.4287,-3.9620", 12000),
        SearchCircle("suances", "43.4260,-4.0437", 12000),
        SearchCircle("comillas", "43.3858,-4.2914", 12000),
        SearchCircle("san_vicente", "43.3861,-4.3988", 14000),
        SearchCircle("laredo", "43.4096,-3.4117", 14000),
        SearchCircle("castro_urdiales", "43.3829,-3.2204", 14000),
    ]

    # Filtros: ajusta a tu análisis
    extra = {
        "order": "publicationDate",
        "sort": "desc",
        # Ejemplos:
        # "minPrice": 600,
        # "maxPrice": 2000,
        # "minSize": 40,
    }

    run_dir = download_circles(
        client=client,
        circles=circles,
        out_dir=Path(os.environ.get("IDEALISTA_RAW_DIR", "data/raw/idealista")),
        country="es",
        operation="rent",
        property_type="homes",
        max_pages_per_circle=int(os.environ.get("IDEALISTA_MAX_PAGES", "3")),
        max_items=50,
        extra_params=extra,
        sleep_between_calls_s=0.25,
    )

    print(f"[OK] Descarga finalizada. Datos en: {run_dir}")


if __name__ == "__main__":
    main()
