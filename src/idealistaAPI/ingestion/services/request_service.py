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


def _is_full_page(resp: SearchResponse) -> bool:
    el = resp.get("elementList") or []
    return len(el) >= MAX_ITEMS


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
        "límite",
        "cupo",
        "429",
        "403",
    ]
    return any(f in txt for f in flags)


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
    return min(states, key=lambda s: (s.requests, s.next_page, s.circle.name))


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
        "strategy": "fair_round_robin_over_circles",
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
) -> None:
    payload = {
        "used_requests": used_requests,
        "stopped_by_quota": stopped_by_quota,
        "quota_error": quota_error,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "csv_path": str(csv_path) if csv_path else None,
        "circle_states": [
            {
                "name": st.circle.name,
                "next_page_final": st.next_page,
                "requests": st.requests,
                "exhausted": st.exhausted,
            }
            for st in states
        ],
    }
    _write_json(processed_dir / "summary.json", payload)


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
    stopped_by_quota = False
    quota_error: Optional[str] = None

    while used < max_requests:
        active = _active_states_force(states, max_pages_per_circle, force_max_requests)
        if not active:
            break

        st = _pick_state(active)
        tag = f"req{used + 1:03d}"
        page = st.next_page

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
                break
            _write_json(
                raw_dir / f"{tag}__ERROR.json",
                {"error": str(exc), "circle": st.circle.__dict__, "page": page},
            )
            used += 1
            st.requests += 1
            continue

        used += 1
        st.requests += 1
        st.next_page += 1

        if not force_max_requests:
            element_list = resp.get("elementList") or []
            if not element_list:
                st.exhausted = True
                continue

            if (not no_adaptive_pages) and (not _is_full_page(resp)):
                st.exhausted = True

    csv_path: Optional[Path]
    try:
        csv_path = clean_json_run(input_dir=raw_dir, output_filename=output_csv_name)
    except Exception:
        csv_path = None

    _write_summary(
        processed_dir=processed_dir,
        used_requests=used,
        stopped_by_quota=stopped_by_quota,
        quota_error=quota_error,
        csv_path=csv_path,
        raw_dir=raw_dir,
        states=states,
    )
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
    candidates = sorted(RAW_BASE.glob("rent_homes_run_*"))
    if not candidates:
        raise FileNotFoundError("No hay ejecuciones previas en data/raw/idealista/rent_homes_run_*")
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

    while used < max_requests:
        active = _active_states_force(states, max_pages_per_circle, force_max_requests)
        if not active:
            break

        st = _pick_state(active)
        tag = f"req{used + 1:03d}"
        page = st.next_page
        out_json = raw_dir / f"{tag}__{st.circle.name}__p{page:03d}.json"
        if out_json.exists():
            used += 1
            st.requests += 1
            st.next_page += 1
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
                break
            _write_json(
                raw_dir / f"{tag}__ERROR.json",
                {"error": str(exc), "circle": st.circle.__dict__, "page": page},
            )
            used += 1
            st.requests += 1
            continue

        used += 1
        st.requests += 1
        st.next_page += 1

        if not force_max_requests:
            element_list = resp.get("elementList") or []
            if not element_list:
                st.exhausted = True
                continue

            if (not no_adaptive_pages) and (not _is_full_page(resp)):
                st.exhausted = True

    csv_path: Optional[Path]
    try:
        csv_path = clean_json_run(input_dir=raw_dir, output_filename=output_csv_name)
    except Exception:
        csv_path = None

    _write_summary(
        processed_dir=processed_dir,
        used_requests=used,
        stopped_by_quota=stopped_by_quota,
        quota_error=quota_error,
        csv_path=csv_path,
        raw_dir=raw_dir,
        states=states,
    )
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
