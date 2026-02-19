from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ingestion.clean_idealista import clean_json_run
from src.ingestion.client import IdealistaClient

# =========================
# Config por defecto
# =========================

RAW_BASE = Path("data/raw/idealista")
PROCESSED_BASE = Path("data/processed/idealista")

DEFAULT_DISTANCE_M = 18000
MAX_ITEMS = 50
SLEEP_S = 0.2

DEFAULT_CIRCLES: List[Tuple[str, str, int]] = [
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

    ("Santander", "43.4623,-3.8099", 18000),
    ("Santander", "43.4623,-3.8099", 18000),

    ("Somo", "43.4470,-3.7450", 14000),
    ("Suances", "43.4260,-4.0430", 14000),

    ("Laredo", "43.4090,-3.4160", 16000),
    ("CastroUrdiales", "43.3830,-3.2140", 16000),
]


@dataclass(frozen=True)
class Circle:
    name: str
    center: str  # "lat,lon"
    distance_m: int


@dataclass
class CircleState:
    circle: Circle
    next_page: int = 1
    exhausted: bool = False
    bad_streak: int = 0  # consecutive “bad yield” pages


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _default_circles(default_distance: int) -> List[Circle]:
    circles: List[Circle] = []
    for (n, c, d) in DEFAULT_CIRCLES:
        dist = d if d is not None else default_distance
        circles.append(Circle(name=n, center=c, distance_m=dist))
    return circles


def _dedupe_circles_keep_first(circles: List[Circle]) -> List[Circle]:
    """
    Remove exact duplicates by (center, distance_m). Keeps the first occurrence (priority).
    """
    seen: Set[Tuple[str, int]] = set()
    out: List[Circle] = []
    for c in circles:
        key = (c.center.strip(), int(c.distance_m))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _is_full_page(resp: Dict[str, Any], max_items: int) -> bool:
    el = resp.get("elementList") or []
    return isinstance(el, list) and len(el) >= max_items


def _safe_property_key(it: Dict[str, Any]) -> Optional[str]:
    """
    Primary dedupe key: propertyCode.
    Fallback: deterministic hash of a few stable-ish fields when propertyCode is missing.
    """
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
    circle: Circle,
    page: int,
    raw_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    resp = client.search(
        country="es",
        operation="rent",
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


def run(
    *,
    max_requests: int,
    circles: List[Circle],
    max_pages_per_circle: int,
    adaptive_pages: bool,
    output_csv_name: str,
    # duplicate control knobs
    min_new_items_to_continue: int = 5,
    stop_if_dup_ratio_ge: float = 0.85,
    stop_after_consecutive_bad_pages: int = 2,
) -> Path:
    """
    Duplicate-minimizing strategy:
      - Deduplicate circles by (center, distance_m)
      - Keep per-circle next_page state (never restart at page 1 for same circle)
      - Stop a circle early when it mostly returns duplicates / too few new items
      - Still uses publicationDate desc to preserve “quality” (recentness)
    """
    client = IdealistaClient()

    rid = _run_id()
    raw_dir = RAW_BASE / f"rent_homes_run_{rid}"
    out_dir = PROCESSED_BASE / f"rent_homes_run_{rid}"
    _ensure_dir(raw_dir)
    _ensure_dir(out_dir)

    circles = _dedupe_circles_keep_first(circles)
    states: List[CircleState] = [CircleState(circle=c) for c in circles]

    manifest = {
        "run_id": rid,
        "operation": "rent",
        "property_type": "homes",
        "max_requests": max_requests,
        "max_pages_per_circle": max_pages_per_circle,
        "adaptive_pages": adaptive_pages,
        "circles_effective": [s.circle.__dict__ for s in states],
        "output_csv_name": output_csv_name,
        "anti_duplicates": {
            "min_new_items_to_continue": min_new_items_to_continue,
            "stop_if_dup_ratio_ge": stop_if_dup_ratio_ge,
            "stop_after_consecutive_bad_pages": stop_after_consecutive_bad_pages,
            "dedupe_circles_by": "(center, distance_m)",
            "pagination_state": "per-circle next_page",
        },
    }
    _write_json(raw_dir / "manifest.json", manifest)

    used = 0
    seen_keys: Set[str] = set()

    idx = 0
    active_left = sum(1 for s in states if not s.exhausted)

    while used < max_requests and active_left > 0:
        st = states[idx % len(states)]
        idx += 1

        if st.exhausted:
            continue

        if st.next_page > max_pages_per_circle:
            st.exhausted = True
            active_left = sum(1 for s in states if not s.exhausted)
            continue

        tag = f"req{used+1:03d}"
        page = st.next_page

        try:
            resp = _search_one(client, circle=st.circle, page=page, raw_dir=raw_dir, tag=tag)
        except Exception as e:
            _write_json(
                raw_dir / f"{tag}__ERROR.json",
                {"error": str(e), "circle": st.circle.__dict__, "page": page},
            )
            used += 1
            continue

        used += 1

        el = resp.get("elementList") or []
        if not isinstance(el, list) or not el:
            st.exhausted = True
            active_left = sum(1 for s in states if not s.exhausted)
            continue

        # Count new vs duplicates for this page
        new_items = 0
        dup_items = 0
        total_considered = 0

        for it in el:
            key = _safe_property_key(it)
            if not key:
                # Can't dedupe reliably; keep data but don't count as "new"
                continue
            total_considered += 1
            if key in seen_keys:
                dup_items += 1
                continue
            seen_keys.add(key)
            new_items += 1

        dup_ratio = (dup_items / total_considered) if total_considered > 0 else 1.0

        # Adaptive stop: if API says it's not a full page, inventory likely exhausted for that query
        if adaptive_pages and not _is_full_page(resp, MAX_ITEMS):
            st.exhausted = True
            active_left = sum(1 for s in states if not s.exhausted)
            continue

        # Duplicate-aware stop
        is_bad = (new_items < min_new_items_to_continue) or (dup_ratio >= stop_if_dup_ratio_ge)
        st.bad_streak = (st.bad_streak + 1) if is_bad else 0

        if st.bad_streak >= stop_after_consecutive_bad_pages:
            st.exhausted = True
            active_left = sum(1 for s in states if not s.exhausted)
            continue

        # If still active, advance page for this circle
        st.next_page += 1
        active_left = sum(1 for s in states if not s.exhausted)

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
        "unique_keys_seen": len(seen_keys),
        "raw_dir": str(raw_dir),
        "processed_dir": str(out_dir),
        "csv_path": str(csv_path),
        "circle_states": [
            {
                "name": s.circle.name,
                "center": s.circle.center,
                "distance_m": s.circle.distance_m,
                "next_page_final": s.next_page,
                "exhausted": s.exhausted,
                "bad_streak": s.bad_streak,
            }
            for s in states
        ],
    }
    _write_json(out_dir / "summary.json", summary)

    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Idealista: duplicate-minimizing requests for RENT (homes) in Cantabria"
    )
    p.add_argument("--max-requests", type=int, default=100)
    p.add_argument("--max-pages-per-circle", type=int, default=5, help="More pages -> more uniques until exhausted")
    p.add_argument("--no-adaptive-pages", action="store_true")
    p.add_argument("--distance", type=int, default=DEFAULT_DISTANCE_M)
    p.add_argument("--circles-file", type=str, default=None)
    p.add_argument(
        "--output-csv",
        type=str,
        default="rent_homes_cantabria_bezana_like_raw.csv",
    )

    # knobs
    p.add_argument("--min-new-items", type=int, default=5)
    p.add_argument("--stop-if-dup-ratio-ge", type=float, default=0.85)
    p.add_argument("--stop-after-bad-pages", type=int, default=2)

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
        min_new_items_to_continue=max(0, int(args.min_new_items)),
        stop_if_dup_ratio_ge=float(args.stop_if_dup_ratio_ge),
        stop_after_consecutive_bad_pages=max(1, int(args.stop_after_bad_pages)),
    )

    print(f"OK. Results in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
