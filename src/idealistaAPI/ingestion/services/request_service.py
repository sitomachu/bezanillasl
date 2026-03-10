from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.idealistaAPI.config.idealista import DEFAULT_CIRCLES, MAX_ITEMS, PROCESSED_BASE, RAW_BASE, SLEEP_S
from src.idealistaAPI.ingestion.api_types import SearchResponse
from src.idealistaAPI.ingestion.client import IdealistaAPIError, IdealistaClient
from src.idealistaAPI.processing.clean_idealista import clean_json_run


@dataclass(frozen=True)
class Circle:
    name: str
    center: str
    distance_m: int


@dataclass
class CircleState:
    circle: Circle
    next_page: int = 1
    exhausted: bool = False
    requests: int = 0
    bad_streak: int = 0
    total_items: int = 0
    unique_items: int = 0
    duplicate_items: int = 0
    zero_new_streak: int = 0
    last_item_count: int = 0
    last_new_items: int = 0
    last_duplicate_items: int = 0
    last_new_ratio: float = 0.0
    prior_unique_per_request: float = 0.0
    prior_duplicate_rate: float = 1.0


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_circles() -> List[Circle]:
    return [Circle(name=n, center=c, distance_m=d) for (n, c, d) in DEFAULT_CIRCLES]


def _dedupe_circles_keep_first(circles: List[Circle]) -> List[Circle]:
    seen: Set[Tuple[str, int]] = set()
    out: List[Circle] = []
    for c in circles:
        key = (c.center.strip(), int(c.distance_m))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _raw_run_dirs_for_operation(operation: str) -> List[Path]:
    return sorted(
        [
            p
            for p in RAW_BASE.glob(f"{operation}_homes_run_*")
            if p.is_dir()
        ]
    )


def _is_full_page(resp: SearchResponse) -> bool:
    el = resp.get("elementList") or []
    return len(el) >= MAX_ITEMS


def _extract_property_keys(resp: SearchResponse) -> List[str]:
    keys: List[str] = []
    for item in resp.get("elementList") or []:
        if not isinstance(item, dict):
            continue
        key = _safe_property_key(item)
        if key:
            keys.append(key)
    return keys


def _is_quota_exhausted_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    flags = [
        "quota",
        "rate limit",
        "too many requests",
        "request limit",
        "limit exceeded",
        "monthly",
        "limite",
        "li­mite",
        "cupo",
        "429",
        "403",
    ]
    return any(f in txt for f in flags)


def _safe_property_key(it: Dict[str, Any]) -> Optional[str]:
    code = str((it or {}).get("propertyCode", "")).strip()
    if code:
        return f"pc:{code}"
    price = (it or {}).get("price")
    size = (it or {}).get("size")
    lat = (it or {}).get("latitude")
    lon = (it or {}).get("longitude")
    addr = (it or {}).get("address") or (it or {}).get("streetName") or ""
    if price is None and size is None and (lat is None or lon is None) and not addr:
        return None
    s = f"{price}|{size}|{lat}|{lon}|{str(addr).strip().lower()}"
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return f"fb:{h}"

def _search_one(
    client: IdealistaClient,
    *,
    operation: str,
    circle: Circle,
    page: int,
    raw_dir: Path,
    tag: str,
) -> SearchResponse:
    resp = client.search(
        country="es",
        operation=operation,
        property_type="homes",
        num_page=page,
        max_items=MAX_ITEMS,
        center=circle.center,
        distance=circle.distance_m,
        extra_params={"order": "publicationDate", "sort": "desc"},
    )
    _write_json(raw_dir / f"{tag}__{circle.name}__p{page:03d}.json", resp)
    time.sleep(SLEEP_S)
    return resp


def _initial_states(circles: List[Circle]) -> List[CircleState]:
    return [CircleState(circle=c) for c in circles]


def _active_states(states: List[CircleState], max_pages_per_circle: int) -> List[CircleState]:
    out: List[CircleState] = []
    for st in states:
        if st.exhausted:
            continue
        if st.next_page > max_pages_per_circle:
            continue
        out.append(st)
    return out


def _active_states_force(
    states: List[CircleState],
    max_pages_per_circle: int,
    force_max_requests: bool,
) -> List[CircleState]:
    if force_max_requests:
        return [st for st in states if not st.exhausted]
    return _active_states(states, max_pages_per_circle)


def _pick_state(states: List[CircleState]) -> CircleState:
    unexplored = [st for st in states if st.requests == 0]
    if unexplored:
        return max(
            unexplored,
            key=lambda s: (
                s.prior_unique_per_request,
                -s.prior_duplicate_rate,
                -s.circle.distance_m,
                s.circle.name,
            ),
        )
    return max(
        states,
        key=lambda s: (
            s.last_new_items,
            s.last_new_ratio,
            s.unique_items / max(1, s.requests),
            s.prior_unique_per_request,
            -s.requests,
            -s.next_page,
            s.circle.name,
        ),
    )


def _new_manifest(
    *,
    run_id: str,
    operation: str,
    max_requests: int,
    max_pages_per_circle: int,
    output_csv_name: str,
    circles: List[Circle],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "operation": operation,
        "property_type": "homes",
        "max_requests": max_requests,
        "max_pages_per_circle": max_pages_per_circle,
        "max_items": MAX_ITEMS,
        "output_csv_name": output_csv_name,
        "circles_effective": [c.__dict__ for c in circles],
        "strategy": "novelty_prioritized_over_circles",
    }


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
        "total_items": total_items,
        "unique_items": unique_items,
        "duplicate_items": max(0, total_items - unique_items),
        "duplicate_rate": round(max(0, total_items - unique_items) / total_items, 4) if total_items else 0.0,
        "circle_states": [
            {
                "name": st.circle.name,
                "next_page_final": st.next_page,
                "requests": st.requests,
                "exhausted": st.exhausted,
                "total_items": st.total_items,
                "unique_items": st.unique_items,
                "duplicate_items": st.duplicate_items,
                "last_item_count": st.last_item_count,
                "last_new_items": st.last_new_items,
                "last_new_ratio": round(st.last_new_ratio, 4),
                "zero_new_streak": st.zero_new_streak,
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

    circles = _dedupe_circles_keep_first(_default_circles())
    states = _initial_states(circles)
    _apply_circle_priors(states, _load_circle_priors(operation))
    _write_json(
        raw_dir / "manifest.json",
        _new_manifest(
            run_id=run_id,
            operation=operation,
            max_requests=max_requests,
            max_pages_per_circle=max_pages_per_circle,
            output_csv_name=output_csv_name,
            circles=circles,
        ),
    )

    used = 0
    seen_property_keys: Set[str] = set()
    stopped_by_quota = False
    quota_error: Optional[str] = None
    _log(
        f"Inicio de ejecucion: operacion={operation} run_id={run_id} "
        f"objetivo_requests={max_requests} max_paginas_por_circulo={max_pages_per_circle} "
        f"circulos={len(states)}"
    )

    while used < max_requests:
        active = _active_states_force(states, max_pages_per_circle, force_max_requests)
        if not active:
            _log("No quedan circulos activos. Se detiene la ejecucion.")
            break

        st = _pick_state(active)
        tag = f"req{used + 1:03d}"
        page = st.next_page
        _log(
            f"Request {used + 1}/{max_requests}: tag={tag} circulo={st.circle.name} "
            f"pagina={page} requests_en_circulo={st.requests}"
        )

        try:
            resp = _search_one(
                client,
                operation=operation,
                circle=st.circle,
                page=page,
                raw_dir=raw_dir,
                tag=tag,
            )
        except IdealistaAPIError as exc:
            if _is_quota_exhausted_error(exc):
                stopped_by_quota = True
                quota_error = str(exc)
                _write_json(raw_dir / f"{tag}__STOP_QUOTA.json", {"error": quota_error})
                _log(f"Cupo o limite de peticiones detectado. Se detiene la ejecucion. error={quota_error}")
                break
            _write_json(
                raw_dir / f"{tag}__ERROR.json",
                {"error": str(exc), "circle": st.circle.__dict__, "page": page},
            )
            used += 1
            st.requests += 1
            _log(f"Request con error. Se detiene la ejecucion. tag={tag} error={exc}")
            break

        used += 1
        st.requests += 1
        st.next_page += 1
        element_count, new_count, duplicate_count = _update_state_metrics(st, resp, seen_property_keys)
        if not force_max_requests:
            should_exhaust, reason = _should_exhaust_circle(
                st=st,
                resp=resp,
                no_adaptive_pages=no_adaptive_pages,
            )
            if should_exhaust:
                st.bad_streak += 1
                st.exhausted = True
                _log(
                    f"Circulo agotado: {st.circle.name} motivo={reason} "
                    f"anuncios={element_count} nuevos={new_count} duplicados={duplicate_count}"
                )
            else:
                st.bad_streak = 0
        _log(
            f"Request OK. tag={tag} anuncios={element_count} nuevos={new_count} "
            f"duplicados={duplicate_count} ratio_novedad={st.last_new_ratio:.2f} "
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
    raw_circles = manifest.get("circles_effective") or manifest.get("circles") or []
    if (not raw_circles) and manifest.get("circle_states"):
        raw_circles = [
            {
                "name": c.get("name"),
                "center": c.get("center"),
                "distance_m": c.get("distance_m"),
            }
            for c in manifest.get("circle_states", [])
        ]
    circles = [Circle(**c) for c in raw_circles if c.get("name") and c.get("center") and c.get("distance_m") is not None]
    circles = _dedupe_circles_keep_first(circles)
    by_name = {c.name: CircleState(circle=c) for c in circles}

    used = 0
    for fp in sorted(raw_dir.glob("req*.json")):
        m = _REQ_PATTERN.match(fp.name)
        if not m:
            continue
        used = max(used, int(m.group(1)))
        cname = m.group(2)
        page = int(m.group(3))
        if cname in by_name:
            st = by_name[cname]
            st.next_page = max(st.next_page, page + 1)
            st.requests += 1

    return manifest, list(by_name.values()), used


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
    force_max_requests: bool = True,
) -> Path:
    client = IdealistaClient()
    raw_dir = _find_latest_rent_raw_dir()
    manifest, states, used = _load_resume_state(raw_dir)
    run_id = str(manifest.get("run_id") or raw_dir.name.replace("rent_homes_run_", ""))

    processed_dir = PROCESSED_BASE / f"rent_homes_run_{run_id}"
    _ensure_dir(processed_dir)

    max_requests = int(max_requests_override or manifest.get("max_requests", 100))
    max_pages_per_circle = int(max_pages_per_circle_override or manifest.get("max_pages_per_circle", 20))
    output_csv_name = str(output_csv_override or manifest.get("output_csv_name", "rent_homes_cantabria_bezana_like_raw.csv"))

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
        f"objetivo_requests={max_requests} max_paginas_por_circulo={max_pages_per_circle} "
        f"circulos={len(states)}"
    )

    while used < max_requests:
        active = _active_states_force(states, max_pages_per_circle, force_max_requests)
        if not active:
            _log("No quedan circulos activos. Se detiene la reanudacion.")
            break

        st = _pick_state(active)
        tag = f"req{used + 1:03d}"
        page = st.next_page
        _log(
            f"Request reanudada {used + 1}/{max_requests}: tag={tag} circulo={st.circle.name} "
            f"pagina={page} requests_en_circulo={st.requests}"
        )
        out_json = raw_dir / f"{tag}__{st.circle.name}__p{page:03d}.json"
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
                circle=st.circle,
                page=page,
                raw_dir=raw_dir,
                tag=tag,
            )
        except IdealistaAPIError as exc:
            if _is_quota_exhausted_error(exc):
                stopped_by_quota = True
                quota_error = str(exc)
                _write_json(raw_dir / f"{tag}__STOP_QUOTA.json", {"error": quota_error})
                _log(f"Cupo o limite de peticiones detectado. Se detiene la reanudacion. error={quota_error}")
                break
            _write_json(
                raw_dir / f"{tag}__ERROR.json",
                {"error": str(exc), "circle": st.circle.__dict__, "page": page},
            )
            used += 1
            st.requests += 1
            _log(f"Request con error. Se detiene la reanudacion. tag={tag} error={exc}")
            break

        used += 1
        st.requests += 1
        st.next_page += 1
        element_count, new_count, duplicate_count = _update_state_metrics(st, resp, seen_property_keys)
        if not force_max_requests:
            should_exhaust, reason = _should_exhaust_circle(
                st=st,
                resp=resp,
                no_adaptive_pages=no_adaptive_pages,
            )
            if should_exhaust:
                st.bad_streak += 1
                st.exhausted = True
                _log(
                    f"Circulo agotado: {st.circle.name} motivo={reason} "
                    f"anuncios={element_count} nuevos={new_count} duplicados={duplicate_count}"
                )
            else:
                st.bad_streak = 0
        _log(
            f"Request OK. tag={tag} anuncios={element_count} nuevos={new_count} "
            f"duplicados={duplicate_count} ratio_novedad={st.last_new_ratio:.2f} "
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
