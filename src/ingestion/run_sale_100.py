from __future__ import annotations

import argparse
import json
import os
import time
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.ingestion.clean_idealista import clean_json_run
from src.ingestion.client import IdealistaClient


# =========================
# Config base
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


# =========================
# Modelos
# =========================

@dataclass(frozen=True)
class Circle:
    name: str
    center: str  # "lat,lon"
    distance_m: int


@dataclass
class Source:
    """
    Fuente = (circle + filtros + orden) con estado de paginación y métricas.
    """
    source_id: str
    circle: Circle
    extra_params: Dict[str, Any]

    next_page: int = 1
    exhausted: bool = False

    # métricas online
    requests: int = 0
    new_items: int = 0
    dup_items: int = 0
    bad_streak: int = 0
    last_dup_ratio: float = 0.0
    last_new: int = 0
    last_total: int = 0

    # si un filtro no es soportado por el client/api
    disabled_due_to_error: bool = False
    last_error: Optional[str] = None


# =========================
# Utilidades
# =========================

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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _dedupe_circles_keep_first(circles: List[Circle]) -> List[Circle]:
    """
    Elimina duplicados por (center, distance_m). Mantiene el primero.
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


def _flatten(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(rows)
    cols = {c.lower(): c for c in df.columns}
    if "propertycode" in cols:
        df = df.drop_duplicates(subset=[cols["propertycode"]], keep="last")
    return df


def _is_full_page(resp: Dict[str, Any], max_items: int) -> bool:
    el = resp.get("elementList") or []
    return isinstance(el, list) and len(el) >= max_items


def _stable_item_key(it: Dict[str, Any]) -> Optional[str]:
    """
    Deduplicación robusta:
    - usa propertyCode si existe
    - si no existe, usa hash (ligero) de atributos comunes
    """
    code = str(it.get("propertyCode", "")).strip()
    if code:
        return f"pc:{code}"

    # Fallback: combinación de campos típicos
    # (si faltan, no inventamos; devolvemos None)
    price = it.get("price")
    size = it.get("size")
    lat = it.get("latitude")
    lon = it.get("longitude")
    addr = it.get("address") or it.get("streetName") or ""
    if price is None and size is None and (lat is None or lon is None) and not addr:
        return None

    s = f"{price}|{size}|{lat}|{lon}|{str(addr).strip().lower()}"
    # hash determinista simple (sin depender de hash randomizado de Python)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return f"fb:{h}"


# =========================
# Construcción de fuentes
# =========================

def _make_price_bins(spec: str) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    spec ejemplo: "0-200000,200000-350000,350000-600000,600000-"
    Devuelve lista [(min,max), ...] donde max None significa abierto.
    """
    bins: List[Tuple[Optional[int], Optional[int]]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            continue
        a, b = part.split("-", 1)
        a = a.strip()
        b = b.strip()
        minp = int(a) if a else None
        maxp = int(b) if b else None
        bins.append((minp, maxp))
    return bins


def _source_id(circle: Circle, extra_params: Dict[str, Any]) -> str:
    # id estable: center|dist|params_ordenados
    items = sorted((k, str(v)) for k, v in extra_params.items())
    s = f"{circle.center}|{circle.distance_m}|" + "|".join([f"{k}={v}" for k, v in items])
    # hash determinista corto
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return f"s{h}"


def build_sources(
    circles: List[Circle],
    *,
    use_price_segmentation: bool,
    price_bins: List[Tuple[Optional[int], Optional[int]]],
) -> List[Source]:
    sources: List[Source] = []

    base_params = {"order": "publicationDate", "sort": "desc"}

    for c in circles:
        # fuente base (sin segmentación)
        sid = _source_id(c, base_params)
        sources.append(Source(source_id=sid, circle=c, extra_params=dict(base_params)))

        if use_price_segmentation and price_bins:
            for (minp, maxp) in price_bins:
                params = dict(base_params)
                if minp is not None:
                    params["minPrice"] = int(minp)
                if maxp is not None:
                    params["maxPrice"] = int(maxp)
                sid = _source_id(c, params)
                sources.append(Source(source_id=sid, circle=c, extra_params=params))

    return sources


# =========================
# Planificador (UCB)
# =========================

def ucb_score(src: Source, total_requests: int, *, c: float = 1.2) -> float:
    """
    Score para maximizar "nuevos por request", penalizar duplicados, y explorar fuentes poco muestreadas.

    - exploitation: yield = new_items / requests
    - penalty: dup_ratio reciente
    - exploration: sqrt(log(total)/requests)
    """
    if src.exhausted or src.disabled_due_to_error:
        return -1e18

    r = max(1, src.requests)
    yield_rate = src.new_items / r

    # dup_ratio reciente (si no hay datos, 0)
    dup_ratio = src.last_dup_ratio

    # exploración estilo UCB1
    explore = math.sqrt(max(0.0, math.log(max(1, total_requests))) / r)

    # penaliza fuerte el duplicado: si dup_ratio -> 1, score cae
    return (yield_rate * (1.0 - dup_ratio)) + (c * explore)


def pick_next_source(sources: List[Source], total_requests: int) -> Optional[Source]:
    best: Optional[Source] = None
    best_score = -1e18
    for s in sources:
        sc = ucb_score(s, total_requests)
        if sc > best_score:
            best_score = sc
            best = s
    return best


# =========================
# API call + lógica anti-duplicados
# =========================

def _search_one(
    client: IdealistaClient,
    *,
    src: Source,
    raw_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    resp = client.search(
        country="es",
        operation="sale",
        property_type="homes",
        num_page=src.next_page,
        max_items=MAX_ITEMS,
        center=src.circle.center,
        distance=src.circle.distance_m,
        extra_params=src.extra_params,
    )
    _write_json(raw_dir / f"{tag}__{src.circle.name}__{src.source_id}__p{src.next_page:03d}.json", resp)
    time.sleep(SLEEP_S)
    return resp


def apply_result_and_update(
    src: Source,
    resp: Dict[str, Any],
    seen_keys: Set[str],
    all_rows: List[Dict[str, Any]],
    *,
    min_new_items_to_continue: int,
    stop_if_dup_ratio_ge: float,
    stop_after_consecutive_high_dup_pages: int,
    adaptive_pages: bool,
) -> None:
    el = resp.get("elementList") or []
    if not isinstance(el, list) or not el:
        src.exhausted = True
        return

    new_items = 0
    dup_items = 0
    total_considered = 0

    for it in el:
        key = _stable_item_key(it)
        if not key:
            # sin clave, no podemos deduplicar bien: aún así lo guardamos,
            # pero NO lo contamos como "nuevo" para optimización (evita sesgo).
            all_rows.append(it)
            continue

        total_considered += 1
        if key in seen_keys:
            dup_items += 1
            continue

        seen_keys.add(key)
        all_rows.append(it)
        new_items += 1

    src.requests += 1
    src.new_items += new_items
    src.dup_items += dup_items
    src.last_new = new_items
    src.last_total = total_considered

    dup_ratio = (dup_items / total_considered) if total_considered > 0 else 1.0
    src.last_dup_ratio = dup_ratio

    # Reglas de corte / penalización
    is_bad = (new_items < min_new_items_to_continue) or (dup_ratio >= stop_if_dup_ratio_ge)
    src.bad_streak = (src.bad_streak + 1) if is_bad else 0

    # adaptativo por “página incompleta”
    if adaptive_pages and not _is_full_page(resp, MAX_ITEMS):
        # normalmente indica agotamiento para ese query concreto
        src.exhausted = True
        return

    if src.bad_streak >= stop_after_consecutive_high_dup_pages:
        src.exhausted = True
        return

    # si no se agotó, avanzar página
    src.next_page += 1


# =========================
# Persistencia de estado
# =========================

def save_state(state_path: Path, sources: List[Source], seen_keys: Set[str]) -> None:
    payload = {
        "saved_at": datetime.now().isoformat(),
        "sources": [asdict(s) for s in sources],
        "seen_keys": list(seen_keys),
    }
    _write_json(state_path, payload)


def load_state(state_path: Path) -> Tuple[List[Source], Set[str]]:
    data = _load_json(state_path)
    sources: List[Source] = []
    for s in data.get("sources", []):
        circle = Circle(**s["circle"])
        src = Source(
            source_id=s["source_id"],
            circle=circle,
            extra_params=s["extra_params"],
            next_page=s.get("next_page", 1),
            exhausted=s.get("exhausted", False),
            requests=s.get("requests", 0),
            new_items=s.get("new_items", 0),
            dup_items=s.get("dup_items", 0),
            bad_streak=s.get("bad_streak", 0),
            last_dup_ratio=s.get("last_dup_ratio", 0.0),
            last_new=s.get("last_new", 0),
            last_total=s.get("last_total", 0),
            disabled_due_to_error=s.get("disabled_due_to_error", False),
            last_error=s.get("last_error"),
        )
        sources.append(src)
    seen = set(data.get("seen_keys", []))
    return sources, seen


# =========================
# Run principal
# =========================

def run(
    *,
    max_requests: int,
    circles: List[Circle],
    adaptive_pages: bool,
    verify_ssl: bool,
    raw_csv_name: str,
    max_pages_per_source: int,
    resume_state: Optional[Path],
    save_state_every: int,
    use_price_segmentation: bool,
    price_bins_spec: str,
    # knobs anti-duplicados
    min_new_items_to_continue: int,
    stop_if_dup_ratio_ge: float,
    stop_after_consecutive_high_dup_pages: int,
) -> Path:
    client = IdealistaClient()

    rid = _run_id()
    raw_dir = RAW_BASE / f"sale_homes_run_{rid}"
    out_dir = PROCESSED_BASE / f"sale_homes_run_{rid}"
    _ensure_dir(raw_dir)
    _ensure_dir(out_dir)

    circles = _dedupe_circles_keep_first(circles)

    price_bins = _make_price_bins(price_bins_spec) if price_bins_spec else []
    sources: List[Source]
    seen_keys: Set[str]

    state_path = out_dir / "state.json"

    if resume_state and resume_state.exists():
        sources, seen_keys = load_state(resume_state)
    else:
        sources = build_sources(
            circles,
            use_price_segmentation=use_price_segmentation,
            price_bins=price_bins,
        )
        seen_keys = set()

    manifest = {
        "run_id": rid,
        "operation": "sale",
        "property_type": "homes",
        "max_requests": max_requests,
        "adaptive_pages": adaptive_pages,
        "verify_ssl": verify_ssl,
        "raw_csv_name": raw_csv_name,
        "max_pages_per_source": max_pages_per_source,
        "use_price_segmentation": use_price_segmentation,
        "price_bins_spec": price_bins_spec,
        "anti_duplicates": {
            "min_new_items_to_continue": min_new_items_to_continue,
            "stop_if_dup_ratio_ge": stop_if_dup_ratio_ge,
            "stop_after_consecutive_high_dup_pages": stop_after_consecutive_high_dup_pages,
        },
        "circles_effective": [asdict(c) for c in circles],
        "sources_count": len(sources),
        "resumed_from": str(resume_state) if resume_state else None,
    }
    _write_json(raw_dir / "manifest.json", manifest)

    used = 0
    all_rows: List[Dict[str, Any]] = []

    while used < max_requests:
        # filtra fuentes agotadas o sin páginas
        active = [s for s in sources if (not s.exhausted) and (not s.disabled_due_to_error)]
        if not active:
            break

        src = pick_next_source(sources, total_requests=max(1, used + 1))
        if src is None:
            break

        # límite de páginas por fuente (para no atascarte en una sola query)
        if src.next_page > max_pages_per_source:
            src.exhausted = True
            continue

        tag = f"req{used+1:04d}"

        try:
            resp = _search_one(client, src=src, raw_dir=raw_dir, tag=tag)
        except Exception as e:
            src.requests += 1
            src.last_error = str(e)

            # Si el error parece venir de parámetros no soportados, desactiva esa fuente.
            # (No hacemos regex complicado: si falla, esa fuente es mala y no merece presupuesto.)
            src.disabled_due_to_error = True

            _write_json(
                raw_dir / f"{tag}__ERROR__{src.source_id}.json",
                {
                    "error": str(e),
                    "source": {
                        "source_id": src.source_id,
                        "circle": asdict(src.circle),
                        "extra_params": src.extra_params,
                        "page": src.next_page,
                    },
                },
            )
            used += 1
            continue

        apply_result_and_update(
            src,
            resp,
            seen_keys,
            all_rows,
            min_new_items_to_continue=min_new_items_to_continue,
            stop_if_dup_ratio_ge=stop_if_dup_ratio_ge,
            stop_after_consecutive_high_dup_pages=stop_after_consecutive_high_dup_pages,
            adaptive_pages=adaptive_pages,
        )

        used += 1

        if save_state_every > 0 and (used % save_state_every == 0):
            save_state(state_path, sources, seen_keys)

    # Estado final
    save_state(state_path, sources, seen_keys)

    # 1) CSV raw (desde JSON)
    raw_csv_path = None
    try:
        raw_csv_path = clean_json_run(input_dir=raw_dir, output_filename=raw_csv_name)
    except Exception as e:
        _write_json(out_dir / "clean_raw_error.json", {"error": str(e)})

    # 2) CSV filtrado desde memoria (únicos)
    if all_rows:
        df = _flatten(all_rows)

        cols = {c.lower(): c for c in df.columns}
        prov = cols.get("province")
        size = cols.get("size")
        lat = cols.get("latitude")
        lon = cols.get("longitude")

        if prov:
            df = df[df[prov] == "Cantabria"]
        if size:
            s = pd.to_numeric(df[size], errors="coerce")
            df = df[s.notna() & (s > 0)]
        if lat and lon:
            df = df[df[lat].notna() & df[lon].notna()]

        out_csv = out_dir / "sale_homes_cantabria_coast_unique.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
    else:
        out_csv = None

    # Summary con métricas reales (lo que te interesa)
    def src_row(s: Source) -> Dict[str, Any]:
        denom = max(1, s.requests)
        return {
            "source_id": s.source_id,
            "circle": s.circle.name,
            "center": s.circle.center,
            "distance_m": s.circle.distance_m,
            "params": s.extra_params,
            "requests": s.requests,
            "new_items": s.new_items,
            "dup_items": s.dup_items,
            "yield_new_per_req": s.new_items / denom,
            "last_dup_ratio": s.last_dup_ratio,
            "next_page": s.next_page,
            "exhausted": s.exhausted,
            "disabled_due_to_error": s.disabled_due_to_error,
            "last_error": s.last_error,
        }

    sources_sorted = sorted(sources, key=lambda s: (s.new_items / max(1, s.requests)), reverse=True)

    summary = {
        "used_requests": used,
        "unique_seen_keys": len(seen_keys),
        "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
        "filtered_unique_csv_path": str(out_csv) if out_csv else None,
        "sources_total": len(sources),
        "sources_active_end": sum(1 for s in sources if (not s.exhausted and not s.disabled_due_to_error)),
        "top_sources_by_yield": [src_row(s) for s in sources_sorted[:10]],
        "worst_sources_by_yield": [src_row(s) for s in sources_sorted[-10:]],
        "state_path": str(state_path),
    }
    _write_json(out_dir / "summary.json", summary)

    return out_dir


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Idealista: adquisición optimizada de registros únicos (sale/homes)")

    p.add_argument("--max-requests", type=int, default=100)
    p.add_argument("--max-pages-per-source", type=int, default=8, help="Límite duro por fuente (evita atasco)")
    p.add_argument("--no-adaptive-pages", action="store_true", help="Si se activa, no corta por página incompleta")

    p.add_argument("--distance", type=int, default=DEFAULT_DISTANCE_M)
    p.add_argument("--circles-file", type=str, default=None)

    p.add_argument("--raw-csv-name", type=str, default="sale_homes_raw.csv")

    # estado
    p.add_argument("--resume-state", type=str, default=None, help="Ruta a state.json previo (reanudar sin repetir)")
    p.add_argument("--save-state-every", type=int, default=10, help="Cada cuántos requests guardar state.json")

    # segmentación por precio (reduce colisión)
    p.add_argument("--use-price-segmentation", action="store_true", help="Activa fuentes por rangos de precio")
    p.add_argument(
        "--price-bins",
        type=str,
        default="0-200000,200000-350000,350000-600000,600000-",
        help='Ej: "0-200000,200000-350000,350000-600000,600000-"',
    )

    # knobs anti-duplicados
    p.add_argument("--min-new-items", type=int, default=5, help="Corta si una página aporta menos de N nuevos")
    p.add_argument("--stop-if-dup-ratio-ge", type=float, default=0.85, help="Corta si dup_ratio >= X")
    p.add_argument("--stop-after-high-dup-pages", type=int, default=2, help="Corta tras N páginas malas seguidas")

    return p


def main() -> int:
    args = build_parser().parse_args()
    verify_ssl = _env_bool("IDEALISTA_VERIFY_SSL", default=True)

    if args.circles_file:
        circles = _load_circles_from_file(Path(args.circles_file), default_distance=int(args.distance))
    else:
        circles = _default_circles(int(args.distance))

    resume_state = Path(args.resume_state) if args.resume_state else None

    out_dir = run(
        max_requests=int(args.max_requests),
        circles=circles,
        adaptive_pages=(not args.no_adaptive_pages),
        verify_ssl=verify_ssl,
        raw_csv_name=str(args.raw_csv_name),
        max_pages_per_source=max(1, int(args.max_pages_per_source)),
        resume_state=resume_state,
        save_state_every=max(0, int(args.save_state_every)),
        use_price_segmentation=bool(args.use_price_segmentation),
        price_bins_spec=str(args.price_bins),
        min_new_items_to_continue=max(0, int(args.min_new_items)),
        stop_if_dup_ratio_ge=float(args.stop_if_dup_ratio_ge),
        stop_after_consecutive_high_dup_pages=max(1, int(args.stop_after_high_dup_pages)),
    )

    print(f"OK. Resultados en: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
