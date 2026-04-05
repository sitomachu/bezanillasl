from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.idealistaAPI.config.idealista import DEFAULT_LOCATIONS, MAX_ITEMS, PROCESSED_BASE, RAW_BASE, SLEEP_S
from src.idealistaAPI.ingestion.api_types import SearchResponse
from src.idealistaAPI.ingestion.client import IdealistaAPIError, IdealistaClient
from src.idealistaAPI.processing.clean_idealista import clean_json_run

# Errores consecutivos en una ubicación antes de abandonarla y continuar con las demás.
_MAX_CONSECUTIVE_ERRORS: int = 3


@dataclass(frozen=True)
class Location:
    name: str
    location_id: str
    fallback_center: Optional[str] = None      # usado automáticamente si location_id devuelve 404
    fallback_distance_m: Optional[int] = None


@dataclass
class CircleState:
    location: Location
    next_page: int = 1
    exhausted: bool = False
    requests: int = 0
    consecutive_errors: int = 0
    total_pages: Optional[int] = None  # poblado desde resp["totalPages"] en la primera llamada


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_locations() -> List[Location]:
    return [
        Location(name=n, location_id=lid, fallback_center=fc, fallback_distance_m=fd)
        for (n, lid, fc, fd) in DEFAULT_LOCATIONS
    ]


def _dedupe_locations_keep_first(locations: List[Location]) -> List[Location]:
    seen: Set[str] = set()
    out: List[Location] = []
    for loc in locations:
        key = loc.location_id.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(loc)
    return out


def _is_location_id_valid(location_id: str) -> bool:
    """Returns False for placeholder IDs used in legacy manifests."""
    return bool(location_id) and not location_id.startswith("legacy:")


def _search_one(
    client: IdealistaClient,
    *,
    operation: str,
    location: Location,
    page: int,
    raw_dir: Path,
    tag: str,
) -> SearchResponse:
    """
    Estrategia (en orden):
    1. location_id  → búsqueda exacta por municipio, sin solapamiento geográfico.
    2. center+distance (fallback) → si location_id devuelve 404 o no está disponible.
    El fallback es automático: no requiere intervención del usuario.
    """
    if _is_location_id_valid(location.location_id):
        try:
            resp = client.search(
                country="es",
                operation=operation,
                property_type="homes",
                num_page=page,
                max_items=MAX_ITEMS,
                location_id=location.location_id,
                extra_params={"order": "publicationDate", "sort": "desc"},
            )
            _write_json(raw_dir / f"{tag}__{location.name}__p{page:03d}.json", resp)
            time.sleep(SLEEP_S)
            return resp
        except IdealistaAPIError as exc:
            if "404" in str(exc) and location.fallback_center:
                _log(
                    f"  AVISO: location_id '{location.location_id}' no válido (404). "
                    f"Activando fallback center+distance para {location.name}."
                )
            else:
                raise

    if not location.fallback_center or location.fallback_distance_m is None:
        raise IdealistaAPIError(
            f"Ubicacion '{location.name}' sin location_id válido ni fallback center+distance."
        )

    resp = client.search(
        country="es",
        operation=operation,
        property_type="homes",
        num_page=page,
        max_items=MAX_ITEMS,
        center=location.fallback_center,
        distance=location.fallback_distance_m,
        extra_params={"order": "publicationDate", "sort": "desc"},
    )
    _write_json(raw_dir / f"{tag}__{location.name}__p{page:03d}.json", resp)
    time.sleep(SLEEP_S)
    return resp


def _initial_states(locations: List[Location]) -> List[CircleState]:
    return [CircleState(location=loc) for loc in locations]


def _active_states(states: List[CircleState], max_pages_per_location: int) -> List[CircleState]:
    out: List[CircleState] = []
    for st in states:
        if st.exhausted:
            continue
        if st.total_pages is not None and st.next_page > st.total_pages:
            continue
        if st.next_page > max_pages_per_location:
            continue
        out.append(st)
    return out


def _active_states_force(
    states: List[CircleState],
    max_pages_per_location: int,
    force_max_requests: bool,
) -> List[CircleState]:
    if force_max_requests:
        return [
            st for st in states
            if not st.exhausted
            and not (st.total_pages is not None and st.next_page > st.total_pages)
        ]
    return _active_states(states, max_pages_per_location)


def _pick_state(states: List[CircleState]) -> CircleState:
    return min(states, key=lambda s: (s.requests, s.next_page, s.location.name))


def _is_quota_exhausted_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    flags = [
        "quota", "rate limit", "too many requests", "request limit",
        "limit exceeded", "monthly", "limite", "li\u00admite", "cupo", "429", "403",
    ]
    return any(f in txt for f in flags)


def _new_manifest(
    *,
    run_id: str,
    operation: str,
    max_requests: int,
    max_pages_per_location: int,
    output_csv_name: str,
    locations: List[Location],
    pool_expansion_enabled: bool,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "operation": operation,
        "property_type": "homes",
        "max_requests": max_requests,
        "max_pages_per_location": max_pages_per_location,
        "max_items": MAX_ITEMS,
        "output_csv_name": output_csv_name,
        "locations_effective": [loc.__dict__ for loc in locations],
        "pool_expansion_enabled": pool_expansion_enabled,
        "strategy": (
            "fair_round_robin_by_locationid_with_center_fallback_pool_expansion"
            if pool_expansion_enabled
            else "fair_round_robin_by_locationid_with_center_fallback_no_pool_expansion"
        ),
    }


def _write_summary(
    *,
    processed_dir: Path,
    used_requests: int,
    stopped_by_quota: bool,
    stopped_by_user: bool,
    quota_error: Optional[str],
    unexpected_error: Optional[str],
    output_csv_name: str,
    raw_dir: Path,
    csv_path: Optional[Path],
    postprocess_status: str,
    states: List[CircleState],
) -> None:
    payload = {
        "used_requests": used_requests,
        "stopped_by_quota": stopped_by_quota,
        "stopped_by_user": stopped_by_user,
        "quota_error": quota_error,
        "unexpected_error": unexpected_error,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "postprocess_status": postprocess_status,
        "suggested_output_csv": output_csv_name,
        "csv_path": str(csv_path) if csv_path is not None else None,
        "location_states": [
            {
                "name": st.location.name,
                "location_id": st.location.location_id,
                "next_page_final": st.next_page,
                "total_pages": st.total_pages,
                "requests": st.requests,
                "exhausted": st.exhausted,
                "consecutive_errors": st.consecutive_errors,
            }
            for st in states
        ],
    }
    _write_json(processed_dir / "summary.json", payload)


# Pool completo de municipios costeros/suburbanos de Cantabria ordenados por
# similitud a Santa Cruz de Bezana. run_new lo usa para expandir automáticamente
# cuando el lote inicial se agota y aún quedan requests disponibles.
# Válido tanto para operacion="rent" como "sale".
_CANTABRIA_POOL: List[Location] = [
    # Batch 1 — núcleo (bahía de Santander y costa central)
    Location("SantaCruzDeBezana",  "0-EU-ES-39-39016", "43.4435,-3.9036", 5000),
    Location("Camargo",            "0-EU-ES-39-39018", "43.4150,-3.8540", 6000),
    Location("Pielagos",           "0-EU-ES-39-39046", "43.4230,-3.9520", 7000),
    Location("Santander",          "0-EU-ES-39-39075", "43.4623,-3.8099", 7000),
    Location("MarinaDeCudeyo",     "0-EU-ES-39-39043", "43.4620,-3.9080", 5000),
    Location("Miengo",             "0-EU-ES-39-39040", "43.4100,-4.0050", 5000),
    Location("RibamontanAlMar",    "0-EU-ES-39-39056", "43.4470,-3.7450", 6000),
    Location("Suances",            "0-EU-ES-39-39084", "43.4260,-4.0430", 6000),
    Location("Laredo",             "0-EU-ES-39-39034", "43.4090,-3.4160", 7000),
    Location("CastroUrdiales",     "0-EU-ES-39-39021", "43.3830,-3.2140", 7000),
    # Batch 2 — suburbano cercano a la bahía
    Location("Villaescusa",        "0-EU-ES-39-39094", "43.4200,-3.9200", 4000),
    Location("ElAstillero",        "0-EU-ES-39-39007", "43.4010,-3.8190", 4000),
    Location("MedioCudeyo",        "0-EU-ES-39-39039", "43.4350,-3.8440", 5000),
    Location("Polanco",            "0-EU-ES-39-39053", "43.3940,-4.0210", 4000),
    Location("SantaMariadeCayon",  "0-EU-ES-39-39067", "43.3780,-3.8890", 6000),
    Location("Torrelavega",        "0-EU-ES-39-39087", "43.3520,-4.0490", 5000),
    Location("Cartes",             "0-EU-ES-39-39015", "43.3530,-4.0090", 4000),
    Location("Santona",            "0-EU-ES-39-39076", "43.4440,-3.4590", 5000),
    Location("Noja",               "0-EU-ES-39-39049", "43.4860,-3.5350", 5000),
    Location("Entrambasaguas",     "0-EU-ES-39-39025", "43.3780,-3.6890", 6000),
    # Batch 3 — costa oriental y occidental
    Location("Bareyo",             "0-EU-ES-39-39010", "43.4750,-3.6700", 5000),
    Location("Arnuero",            "0-EU-ES-39-39005", "43.4780,-3.5090", 4000),
    Location("BarcenadeCicero",    "0-EU-ES-39-39008", "43.4200,-3.4280", 5000),
    Location("RibamontanAlMonte",  "0-EU-ES-39-39057", "43.4000,-3.7600", 5000),
    Location("SantillanadelMar",   "0-EU-ES-39-39072", "43.3890,-4.1020", 5000),
    Location("AlfozDeLloredo",     "0-EU-ES-39-39001", "43.3750,-4.2080", 6000),
    Location("Comillas",           "0-EU-ES-39-39020", "43.3870,-4.2890", 5000),
    Location("SanVicenteBarquera", "0-EU-ES-39-39068", "43.3880,-4.3990", 5000),
    Location("Reocin",             "0-EU-ES-39-39059", "43.3680,-4.0870", 5000),
    Location("LosCorralesDeBuelna","0-EU-ES-39-39026", "43.2620,-4.0680", 5000),
    # Batch 4 — expansión adicional
    Location("Ampuero",            "0-EU-ES-39-39002", "43.3010,-3.4600", 5000),
    Location("Solorzano",          "0-EU-ES-39-39081", "43.4050,-3.6200", 5000),
    Location("PuenteViesgo",       "0-EU-ES-39-39055", "43.2830,-3.9600", 4000),
    Location("Valdaliga",          "0-EU-ES-39-39090", "43.3580,-4.3380", 6000),
    Location("Udias",              "0-EU-ES-39-39088", "43.3680,-4.2150", 4000),
    Location("CabezonDeLaSal",     "0-EU-ES-39-39012", "43.3110,-4.2360", 5000),
]


def run_new(
    *,
    operation: str,
    max_requests: int,
    max_pages_per_circle: int,
    output_csv_name: str,
    no_adaptive_pages: bool = False,
    force_max_requests: bool = False,
    locations: Optional[List[Location]] = None,
    allow_pool_expansion: bool = True,
) -> Path:
    client = IdealistaClient()
    run_id = _run_id()
    run_name = f"{operation}_homes_run_{run_id}"
    raw_dir = RAW_BASE / run_name
    processed_dir = PROCESSED_BASE / run_name
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    locations = _dedupe_locations_keep_first(locations if locations is not None else _default_locations())
    states = _initial_states(locations)
    _write_json(
        raw_dir / "manifest.json",
        _new_manifest(
            run_id=run_id,
            operation=operation,
            max_requests=max_requests,
            max_pages_per_location=max_pages_per_circle,
            output_csv_name=output_csv_name,
            locations=locations,
            pool_expansion_enabled=allow_pool_expansion,
        ),
    )

    used = 0
    stopped_by_quota = False
    stopped_by_user = False
    quota_error: Optional[str] = None
    unexpected_error: Optional[str] = None
    csv_path: Optional[Path] = None
    postprocess_status = "not_started"
    pending_exception: Optional[Exception] = None
    _log(
        f"Inicio de ejecucion: operacion={operation} run_id={run_id} "
        f"objetivo_requests={max_requests} max_paginas_por_ubicacion={max_pages_per_circle} "
        f"ubicaciones_iniciales={len(states)} pool_total={len(_CANTABRIA_POOL)}"
    )

    try:
        while used < max_requests:
            active = _active_states_force(states, max_pages_per_circle, force_max_requests)
            if not active:
                if not allow_pool_expansion:
                    _log("Ubicaciones objetivo agotadas. Se detiene sin expandir al pool general.")
                    break
                tried = {st.location.name for st in states}
                pending = [loc for loc in _CANTABRIA_POOL if loc.name not in tried]
                if not pending:
                    _log("Pool de municipios agotado. Se detiene la ejecucion.")
                    break
                states.extend(CircleState(location=loc) for loc in pending)
                _log(
                    f"Lote actual agotado. Incorporando {len(pending)} municipios nuevos del pool: "
                    f"{[l.name for l in pending]}"
                )
                continue

            st = _pick_state(active)
            tag = f"req{used + 1:03d}"
            page = st.next_page
            total_pages_str = str(st.total_pages) if st.total_pages is not None else "?"
            _log(
                f"Request {used + 1}/{max_requests}: tag={tag} ubicacion={st.location.name} "
                f"pagina={page}/{total_pages_str} requests_en_ubicacion={st.requests}"
            )

            try:
                resp = _search_one(
                    client,
                    operation=operation,
                    location=st.location,
                    page=page,
                    raw_dir=raw_dir,
                    tag=tag,
                )
            except IdealistaAPIError as exc:
                if _is_quota_exhausted_error(exc):
                    stopped_by_quota = True
                    quota_error = str(exc)
                    _write_json(raw_dir / f"{tag}__STOP_QUOTA.json", {"error": quota_error})
                    _log(f"Cupo o limite de peticiones. Se detiene la ejecucion. error={quota_error}")
                    break

                _write_json(
                    raw_dir / f"{tag}__ERROR.json",
                    {"error": str(exc), "location": st.location.__dict__, "page": page},
                )
                used += 1
                st.requests += 1
                st.consecutive_errors += 1
                _log(
                    f"Error en request (no-quota). Se continua. tag={tag} "
                    f"errores_consecutivos={st.consecutive_errors} error={exc}"
                )
                if st.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    st.exhausted = True
                    _log(
                        f"  → {_MAX_CONSECUTIVE_ERRORS} errores consecutivos. "
                        f"Se omite ubicacion: {st.location.name}"
                    )
                continue

            used += 1
            st.requests += 1
            st.next_page += 1
            st.consecutive_errors = 0

            if st.total_pages is None:
                tp = resp.get("totalPages")
                if isinstance(tp, int) and tp > 0:
                    st.total_pages = tp
                    _log(f"  → {st.location.name}: {resp.get('total', '?')} anuncios en {tp} paginas totales")

            element_list = resp.get("elementList") or []

            if not force_max_requests:
                if not element_list:
                    st.exhausted = True
                    _log(f"Pagina vacia. Ubicacion agotada: {st.location.name}")
                    continue

                if st.total_pages is not None and st.next_page > st.total_pages:
                    st.exhausted = True
                    _log(
                        f"Todas las paginas obtenidas ({st.total_pages}). "
                        f"Ubicacion agotada: {st.location.name}"
                    )
                elif (not no_adaptive_pages) and (len(element_list) < MAX_ITEMS) and (st.total_pages is None):
                    st.exhausted = True
                    _log(f"Pagina incompleta (totalPages no disponible). Ubicacion agotada: {st.location.name}")

            _log(
                f"Request OK. tag={tag} anuncios={len(element_list)} "
                f"proxima_pagina={st.next_page} requests_usadas={used}"
            )
    except KeyboardInterrupt:
        stopped_by_user = True
        _log("Interrupcion manual recibida. Se finaliza el run con los datos ya descargados.")
    except Exception as exc:
        unexpected_error = f"{type(exc).__name__}: {exc}"
        pending_exception = exc
        _log(f"ERROR inesperado. Se finaliza el run con los datos ya descargados. error={unexpected_error}")
    finally:
        try:
            csv_path = clean_json_run(input_dir=raw_dir, output_filename=output_csv_name)
            postprocess_status = "csv_generated"
            _log(f"CSV generado: {csv_path.resolve()}")
        except Exception as exc:
            postprocess_status = f"csv_generation_failed: {type(exc).__name__}"
            _log(f"AVISO: no se pudo generar el CSV automaticamente. error={exc}")

        _write_summary(
            processed_dir=processed_dir,
            used_requests=used,
            stopped_by_quota=stopped_by_quota,
            stopped_by_user=stopped_by_user,
            quota_error=quota_error,
            unexpected_error=unexpected_error,
            output_csv_name=output_csv_name,
            raw_dir=raw_dir,
            csv_path=csv_path,
            postprocess_status=postprocess_status,
            states=states,
        )
        _log(f"Ejecucion finalizada. requests_usadas={used} summary={processed_dir / 'summary.json'}")

    if pending_exception is not None:
        raise pending_exception
    return processed_dir


def add_common_args(parser: argparse.ArgumentParser, *, default_csv: str) -> None:
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-pages-per-circle", type=int, default=20)
    parser.add_argument("--output-csv", type=str, default=default_csv)
    parser.add_argument("--no-adaptive-pages", action="store_true")
    parser.add_argument(
        "--force-max-requests",
        action="store_true",
        help="Consume requests until max-requests even if pages are empty/partial.",
    )
