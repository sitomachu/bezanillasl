from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.idealistaAPI.config.idealista import DEFAULT_LOCATIONS, MAX_ITEMS, PROCESSED_BASE, RAW_BASE, SLEEP_S
from src.idealistaAPI.ingestion.api_types import SearchResponse
from src.idealistaAPI.ingestion.client import IdealistaAPIError, IdealistaClient
from src.idealistaAPI.processing.clean_idealista import clean_json_run

# Número de errores consecutivos en una ubicación antes de abandonarla y seguir con las demás.
_MAX_CONSECUTIVE_ERRORS: int = 3


@dataclass(frozen=True)
class Location:
    name: str
    location_id: str
    fallback_center: Optional[str] = None       # usado automáticamente si location_id devuelve 404
    fallback_distance_m: Optional[int] = None


@dataclass
class CircleState:
    location: Location
    next_page: int = 1
    exhausted: bool = False
    requests: int = 0
    consecutive_errors: int = 0
    total_pages: Optional[int] = None   # poblado desde resp["totalPages"] en la primera llamada


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


def _rounded_distance_m(distance_m: int) -> int:
    return int(round(distance_m / 500.0) * 500)


def _build_backup_circles(circles: List[Circle]) -> List[Circle]:
    out: List[Circle] = []
    for factor in BACKUP_RADIUS_FACTORS:
        for circle in circles:
            scaled_distance = _rounded_distance_m(int(circle.distance_m * factor))
            scaled_distance = max(MIN_BACKUP_DISTANCE_M, min(MAX_BACKUP_DISTANCE_M, scaled_distance))
            if scaled_distance == circle.distance_m:
                continue
            suffix = f"{scaled_distance // 1000:02d}km"
            out.append(
                Circle(
                    name=f"{circle.name}__alt_{suffix}",
                    center=circle.center,
                    distance_m=scaled_distance,
                )
            )
    return _dedupe_circles_keep_first(out)


def _raw_run_dirs_for_operation(operation: str) -> List[Path]:
    return sorted(
        [
            p
            for p in RAW_BASE.glob(f"{operation}_homes_run_*")
            if p.is_dir()
        ]
    )


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
    Realiza una búsqueda para la ubicación dada.

    Estrategia (en orden):
    1. location_id  → búsqueda exacta por municipio, sin solapamiento geográfico.
    2. center+distance (fallback) → si location_id devuelve 404 o no está disponible.

    El fallback es automático: no requiere intervención del usuario.
    """
    use_location_id = _is_location_id_valid(location.location_id)

    if use_location_id:
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
                # fall through to center+distance below
            else:
                raise  # propagar errores que no sean 404 (quota, 5xx, red, etc.)

    # Fallback: center + distance
    if not location.fallback_center or location.fallback_distance_m is None:
        raise IdealistaAPIError(
            f"Ubicacion '{location.name}' no tiene location_id válido ni fallback center+distance."
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
        "strategy": "fair_round_robin_by_locationid_with_center_fallback",
    }


def _sync_manifest_circles(manifest_path: Path, manifest: Dict[str, Any], states: List[CircleState]) -> None:
    payload = dict(manifest)
    payload["circles_effective"] = [st.circle.__dict__ for st in states]
    _write_json(manifest_path, payload)


def _write_summary(
    *,
    processed_dir: Path,
    used_requests: int,
    stopped_by_quota: bool,
    quota_error: Optional[str],
    csv_path: Optional[Path],
    raw_dir: Path,
    states: List[CircleState],
    total_items: int,
    unique_items: int,
) -> None:
    payload = {
        "used_requests": used_requests,
        "stopped_by_quota": stopped_by_quota,
        "quota_error": quota_error,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "csv_path": str(csv_path) if csv_path else None,
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


def _update_state_metrics(
    st: CircleState,
    resp: SearchResponse,
    seen_property_keys: Set[str],
) -> Tuple[int, int, int]:
    page_keys = _extract_property_keys(resp)
    page_total = len(page_keys)
    new_count = 0
    duplicate_count = 0
    for key in page_keys:
        if key in seen_property_keys:
            duplicate_count += 1
            continue
        seen_property_keys.add(key)
        new_count += 1

    st.total_items += page_total
    st.unique_items += new_count
    st.duplicate_items += duplicate_count
    st.last_item_count = page_total
    st.last_new_items = new_count
    st.last_duplicate_items = duplicate_count
    st.last_new_ratio = (new_count / page_total) if page_total else 0.0
    st.zero_new_streak = (st.zero_new_streak + 1) if page_total and new_count == 0 else 0
    return page_total, new_count, duplicate_count


def _derive_circle_priors_from_raw_dir(raw_dir: Path) -> Dict[str, Dict[str, float]]:
    priors: Dict[str, Dict[str, float]] = {}
    states: Dict[str, CircleState] = {}
    seen_property_keys: Set[str] = set()
    for fp in sorted(raw_dir.glob("req*.json")):
        m = _REQ_PATTERN.match(fp.name)
        if not m:
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        circle_name = m.group(2)
        st = states.get(circle_name)
        if st is None:
            st = CircleState(circle=Circle(name=circle_name, center="", distance_m=0))
            states[circle_name] = st
        st.requests += 1
        _update_state_metrics(st, payload, seen_property_keys)

    for circle_name, st in states.items():
        total_items = st.total_items
        priors[circle_name] = {
            "prior_unique_per_request": st.unique_items / max(1, st.requests),
            "prior_duplicate_rate": (st.duplicate_items / total_items) if total_items else 1.0,
        }
    return priors


def _load_circle_priors(operation: str) -> Dict[str, Dict[str, float]]:
    raw_dirs = _raw_run_dirs_for_operation(operation)
    if not raw_dirs:
        return {}
    latest_raw_dir = raw_dirs[-1]
    return _derive_circle_priors_from_raw_dir(latest_raw_dir)


def _apply_circle_priors(states: List[CircleState], priors: Dict[str, Dict[str, float]]) -> None:
    for st in states:
        prior = priors.get(st.circle.name)
        if not prior:
            continue
        st.prior_unique_per_request = float(prior.get("prior_unique_per_request", 0.0) or 0.0)
        st.prior_duplicate_rate = float(prior.get("prior_duplicate_rate", 1.0) or 1.0)


def _should_exhaust_circle(
    *,
    st: CircleState,
    resp: SearchResponse,
    no_adaptive_pages: bool,
) -> Tuple[bool, str]:
    if st.last_item_count == 0:
        return True, "pagina_vacia"

    if st.zero_new_streak >= 5:
        return True, "cinco_paginas_sin_novedad"

    if (not no_adaptive_pages) and (not _is_full_page(resp)):
        return True, "pagina_incompleta"

    return False, ""


def run_new(
    *,
    operation: str,
    max_requests: int,
    max_pages_per_circle: int,
    output_csv_name: str,
    no_adaptive_pages: bool = False,
    force_max_requests: bool = False,
) -> Path:
    client = IdealistaClient()
    run_id = _run_id()
    run_name = f"{operation}_homes_run_{run_id}"
    raw_dir = RAW_BASE / run_name
    processed_dir = PROCESSED_BASE / run_name
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    locations = _dedupe_locations_keep_first(_default_locations())
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
        ),
    )
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup_circles = _build_backup_circles(circles)

    used = 0
    seen_property_keys: Set[str] = set()
    stopped_by_quota = False
    quota_error: Optional[str] = None
    _log(
        f"Inicio de ejecucion: operacion={operation} run_id={run_id} "
        f"objetivo_requests={max_requests} max_paginas_por_ubicacion={max_pages_per_circle} "
        f"ubicaciones={len(states)}"
    )

    while used < max_requests:
        active = _active_states_force(states, max_pages_per_circle, force_max_requests)
        if not active:
            _log("No quedan ubicaciones activas. Se detiene la ejecucion.")
            break

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

            # Error no-quota: registrar, penalizar ubicación, continuar con las demás.
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
        st.consecutive_errors = 0  # reset en cada respuesta exitosa

        # Capturar total de páginas en la primera respuesta exitosa de esta ubicación
        if st.total_pages is None:
            tp = resp.get("totalPages")
            if isinstance(tp, int) and tp > 0:
                st.total_pages = tp
                total = resp.get("total", "?")
                _log(f"  → {st.location.name}: {total} anuncios en {tp} paginas totales")

        element_list = resp.get("elementList") or []

        if not force_max_requests:
            if not element_list:
                st.exhausted = True
                _log(f"Pagina vacia. Ubicacion agotada: {st.location.name}")
                continue

            # Agotamiento preciso usando totalPages de la API
            if st.total_pages is not None and st.next_page > st.total_pages:
                st.exhausted = True
                _log(
                    f"Todas las paginas obtenidas ({st.total_pages}). "
                    f"Ubicacion agotada: {st.location.name}"
                )
            elif (not no_adaptive_pages) and (len(element_list) < MAX_ITEMS) and (st.total_pages is None):
                # Heurístico de respaldo: página incompleta = última página
                st.exhausted = True
                _log(f"Pagina incompleta (totalPages no disponible). Ubicacion agotada: {st.location.name}")

        _log(
            f"Request OK. tag={tag} anuncios={len(element_list)} "
            f"proxima_pagina={st.next_page} requests_usadas={used}"
        )

    csv_path: Optional[Path]
    try:
        csv_path = clean_json_run(input_dir=raw_dir, output_filename=output_csv_name)
        _log(f"CSV generado correctamente: {csv_path}")
    except Exception:
        csv_path = None
        _log("No se pudo generar el CSV (sin filas o error de procesamiento).")

    _write_summary(
        processed_dir=processed_dir,
        used_requests=used,
        stopped_by_quota=stopped_by_quota,
        quota_error=quota_error,
        csv_path=csv_path,
        raw_dir=raw_dir,
        states=states,
        total_items=sum(st.total_items for st in states),
        unique_items=len(seen_property_keys),
    )
    _log(f"Ejecucion finalizada. requests_usadas={used} summary={processed_dir / 'summary.json'}")
    return processed_dir


_REQ_PATTERN = re.compile(r"^req(\d+)__([^_].*?)__p(\d+)\.json$")


def _load_resume_state(raw_dir: Path) -> Tuple[Dict[str, Any], List[CircleState], int]:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe manifest.json en {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Nuevo formato: locations_effective → {name, location_id, fallback_center, fallback_distance_m}
    raw_locations = manifest.get("locations_effective")
    if raw_locations:
        locations = [
            Location(
                name=r["name"],
                location_id=r["location_id"],
                fallback_center=r.get("fallback_center"),
                fallback_distance_m=r.get("fallback_distance_m"),
            )
            for r in raw_locations
            if r.get("name") and r.get("location_id")
        ]
    else:
        # Formato antiguo: circles_effective / circles → {name, center, distance_m}
        # Recupera center+distance como fallback para poder continuar el run.
        raw_circles = manifest.get("circles_effective") or manifest.get("circles") or []
        if not raw_circles and manifest.get("circle_states"):
            raw_circles = [
                {"name": c.get("name"), "center": c.get("center"), "distance_m": c.get("distance_m")}
                for c in manifest["circle_states"]
            ]
        _log(
            "AVISO: manifest con formato antiguo (center+distance). "
            "Se usará center+distance como fallback para la reanudacion."
        )
        locations = [
            Location(
                name=r["name"],
                location_id=f"legacy:{r.get('center','')},{r.get('distance_m','')}",
                fallback_center=r.get("center"),
                fallback_distance_m=int(r["distance_m"]) if r.get("distance_m") is not None else None,
            )
            for r in raw_circles
            if r.get("name")
        ]

    locations = _dedupe_locations_keep_first(locations)
    by_name = {loc.name: CircleState(location=loc) for loc in locations}

    used = 0
    for fp in sorted(raw_dir.glob("req*.json")):
        m = _REQ_PATTERN.match(fp.name)
        if not m:
            continue
        used = max(used, int(m.group(1)))
        lname = m.group(2)
        page = int(m.group(3))
        if lname in by_name:
            st = by_name[lname]
            st.next_page = max(st.next_page, page + 1)
            st.requests += 1

    return manifest, list(by_name.values()), used


def _summary_path_for_raw_dir(raw_dir: Path) -> Path:
    return PROCESSED_BASE / raw_dir.name / "summary.json"


def _load_exhausted_circle_names(raw_dir: Path) -> Set[str]:
    summary_path = _summary_path_for_raw_dir(raw_dir)
    if not summary_path.exists():
        return set()
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {
        str(circle_state.get("name"))
        for circle_state in (payload.get("circle_states") or [])
        if circle_state.get("name") and bool(circle_state.get("exhausted"))
    }


def _merge_missing_circles(states: List[CircleState], circles: List[Circle]) -> List[CircleState]:
    by_name = {st.circle.name for st in states}
    merged = list(states)
    for circle in circles:
        if circle.name in by_name:
            continue
        merged.append(CircleState(circle=circle))
        by_name.add(circle.name)
    return merged


def _expand_with_backup_circles(
    *,
    states: List[CircleState],
    manifest: Dict[str, Any],
    manifest_path: Path,
    backup_circles: List[Circle],
) -> Tuple[List[CircleState], int]:
    existing_names = {st.circle.name for st in states}
    existing_keys = {(st.circle.center.strip(), int(st.circle.distance_m)) for st in states}
    merged = list(states)
    added = 0
    for circle in backup_circles:
        key = (circle.center.strip(), int(circle.distance_m))
        if circle.name in existing_names or key in existing_keys:
            continue
        merged.append(CircleState(circle=circle))
        existing_names.add(circle.name)
        existing_keys.add(key)
        added += 1
    if added:
        _sync_manifest_circles(manifest_path, manifest, merged)
    return merged, added


def _find_latest_rent_raw_dir() -> Path:
    candidates = _raw_run_dirs_for_operation("rent")
    if not candidates:
        raise FileNotFoundError("No hay ejecuciones previas en data/raw/idealistaAPI/raw/rent_homes_run_*")
    return candidates[-1]


def run_resume_latest_rent(
    *,
    max_requests_override: Optional[int] = None,
    max_pages_per_circle_override: Optional[int] = None,
    output_csv_override: Optional[str] = None,
    no_adaptive_pages: bool = False,
    force_max_requests: bool = False,
) -> Path:
    client = IdealistaClient()
    raw_dir = _find_latest_rent_raw_dir()
    manifest, states, used = _load_resume_state(raw_dir)
    manifest_path = raw_dir / "manifest.json"
    run_id = str(manifest.get("run_id") or raw_dir.name.replace("rent_homes_run_", ""))

    processed_dir = PROCESSED_BASE / f"rent_homes_run_{run_id}"
    _ensure_dir(processed_dir)

    max_requests = int(max_requests_override or manifest.get("max_requests", 100))
    max_pages_per_location = int(
        max_pages_per_circle_override or manifest.get("max_pages_per_location") or manifest.get("max_pages_per_circle", 20)
    )
    output_csv_name = str(
        output_csv_override or manifest.get("output_csv_name", "rent_homes_cantabria_bezana_like_raw.csv")
    )

    stopped_by_quota = False
    quota_error: Optional[str] = None
    seen_property_keys: Set[str] = set()

    for fp in sorted(raw_dir.glob("req*.json")):
        m = _REQ_PATTERN.match(fp.name)
        if not m:
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        cname = m.group(2)
        st = next((state for state in states if state.circle.name == cname), None)
        if st is None:
            continue
        _update_state_metrics(st, payload, seen_property_keys)

    _log(
        f"Inicio de reanudacion: run_id={run_id} requests_ya_usadas={used} "
        f"objetivo_requests={max_requests} max_paginas_por_ubicacion={max_pages_per_location} "
        f"ubicaciones={len(states)}"
    )

    while used < max_requests:
        active = _active_states_force(states, max_pages_per_location, force_max_requests)
        if not active:
            _log("No quedan ubicaciones activas. Se detiene la reanudacion.")
            break

        st = _pick_state(active)
        tag = f"req{used + 1:03d}"
        page = st.next_page
        _log(
            f"Request reanudada {used + 1}/{max_requests}: tag={tag} ubicacion={st.location.name} "
            f"pagina={page}/{st.total_pages or '?'} requests_en_ubicacion={st.requests}"
        )

        out_json = raw_dir / f"{tag}__{st.location.name}__p{page:03d}.json"
        if out_json.exists():
            used += 1
            st.requests += 1
            st.next_page += 1
            _log(f"El archivo del request ya existe. Se omite tag={tag}")
            continue

        try:
            resp = _search_one(
                client,
                operation="rent",
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
                _log(f"Cupo o limite de peticiones. Se detiene la reanudacion. error={quota_error}")
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
                _log(f"  → {_MAX_CONSECUTIVE_ERRORS} errores consecutivos. Se omite: {st.location.name}")
            continue

        used += 1
        st.requests += 1
        st.next_page += 1
        st.consecutive_errors = 0

        if st.total_pages is None:
            tp = resp.get("totalPages")
            if isinstance(tp, int) and tp > 0:
                st.total_pages = tp

        element_list = resp.get("elementList") or []

        if not force_max_requests:
            if not element_list:
                st.exhausted = True
                _log(f"Pagina vacia. Ubicacion agotada: {st.location.name}")
                continue

            if st.total_pages is not None and st.next_page > st.total_pages:
                st.exhausted = True
                _log(f"Todas las paginas obtenidas ({st.total_pages}). Ubicacion agotada: {st.location.name}")
            elif (not no_adaptive_pages) and (len(element_list) < MAX_ITEMS) and (st.total_pages is None):
                st.exhausted = True
                _log(f"Pagina incompleta (totalPages no disponible). Ubicacion agotada: {st.location.name}")

        _log(
            f"Request OK. tag={tag} anuncios={len(element_list)} "
            f"proxima_pagina={st.next_page} requests_usadas={used}"
        )

    csv_path: Optional[Path]
    try:
        csv_path = clean_json_run(input_dir=raw_dir, output_filename=output_csv_name)
        _log(f"CSV generado correctamente: {csv_path}")
    except Exception:
        csv_path = None
        _log("No se pudo generar el CSV (sin filas o error de procesamiento).")

    _write_summary(
        processed_dir=processed_dir,
        used_requests=used,
        stopped_by_quota=stopped_by_quota,
        quota_error=quota_error,
        csv_path=csv_path,
        raw_dir=raw_dir,
        states=states,
        total_items=sum(st.total_items for st in states),
        unique_items=len(seen_property_keys),
    )
    _log(f"Reanudacion finalizada. requests_usadas={used} summary={processed_dir / 'summary.json'}")
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
