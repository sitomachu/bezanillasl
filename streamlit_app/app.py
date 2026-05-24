"""
Bezanilla SL — Predictor de precios de viviendas
=================================================

Página principal: estimación de precio de venta y alquiler con modelos
XGBoost replicados desde `notebooks/05_ML/55_input_result.ipynb`,
más un panel con viviendas reales scrapeadas de Idealista.

Ejecutar:  streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import json
import html
import re
import unicodedata
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from bs4 import BeautifulSoup
from folium.plugins import MarkerCluster
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from xgboost import XGBRegressor


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bezanilla SL · Predictor de precios",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
GOLD_DIR     = PROJECT_ROOT / "data" / "gold"
PARAMS_DIR   = PROJECT_ROOT / "data" / "model_results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "idealistaAPI"
MODEL_SALE_PATH = GOLD_DIR / "final_sale_idealistaAPI.csv"
MODEL_RENT_PATH = GOLD_DIR / "final_rent_idealistaAPI.csv"
STREAMLIT_SALE_PATH = GOLD_DIR / "streamlit_sale.csv"
STREAMLIT_RENT_PATH = GOLD_DIR / "streamlit_rent.csv"


PISO_ONLY_FEATURES = {
    "tiene_ascensor_piso",
    "es_exterior_piso",
    "planta_num",
    "interaccion_planta_sin_ascensor_piso",
}


# ─────────────────────────────────────────────────────────────────────────────
# Carga de artefactos: replica notebooks/05_ML/55_input_result.ipynb
# ─────────────────────────────────────────────────────────────────────────────
def build_training_X(df: pd.DataFrame, base_features: list[str], min_muni_obs: int) -> tuple[pd.DataFrame, list[str], dict]:
    df2 = df.copy()
    base = [f for f in base_features if f in df2.columns]
    mun_cols = sorted([c for c in df2.columns if c.startswith("municipio_")])
    if mun_cols:
        counts = df2[mun_cols].sum()
        small = counts[counts < min_muni_obs].index.tolist()
        if small:
            df2["municipio_otros"] = df2[small].max(axis=1)
            df2 = df2.drop(columns=small)
        mun_final = sorted(c for c in df2.columns if c.startswith("municipio_"))
    else:
        mun_final = []

    all_feats = base + [m for m in mun_final if m not in base]
    X_raw = df2[all_feats].copy()

    if "tipologia_unificada_unifamiliar" in df2.columns:
        is_unifamiliar = df2["tipologia_unificada_unifamiliar"] == 1
        for feature in PISO_ONLY_FEATURES:
            if feature in X_raw.columns:
                X_raw.loc[is_unifamiliar, feature] = np.nan

    medians = X_raw.median().to_dict()
    piso_cols = [f for f in all_feats if f in PISO_ONLY_FEATURES]
    other_cols = [f for f in all_feats if f not in PISO_ONLY_FEATURES]
    imputer = SimpleImputer(strategy="median")
    X_other = pd.DataFrame(
        imputer.fit_transform(X_raw[other_cols]),
        columns=other_cols,
        index=X_raw.index,
    )
    X = pd.concat([X_other, X_raw[piso_cols]], axis=1)[all_feats]
    return X, all_feats, medians


GEO_COLS = [
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
    "precio_m2_municipio_media",
    "distancia_centro_municipio_km",
    "score_cercania_servicios",
    "latitud",
    "longitud",
]


def build_geo_ref(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mun_col in [c for c in df.columns if c.startswith("municipio_")]:
        municipio = mun_col.replace("municipio_", "")
        subset = df[df[mun_col] == 1]
        if len(subset) == 0:
            continue
        row = {"municipio": municipio}
        for geo_col in GEO_COLS:
            row[geo_col] = subset[geo_col].median() if geo_col in subset.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("municipio")


def extend_rent_geo_ref(rent_geo_ref: pd.DataFrame, rent_cfg: dict) -> pd.DataFrame:
    proc_rent_path = PROCESSED_DIR / "total_rent_cantabria_outliers.csv"
    rent_path = MODEL_RENT_PATH
    if not proc_rent_path.exists() or not rent_path.exists():
        return rent_geo_ref

    proc = pd.read_csv(proc_rent_path)
    proc["_lat"] = proc["latitude"].round(5)
    proc["_lon"] = proc["longitude"].round(5)

    gold = pd.read_csv(rent_path)
    gold["_lat"] = gold["latitud"].round(5)
    gold["_lon"] = gold["longitud"].round(5)

    geo_feats = [c for c in GEO_COLS if c != "precio_m2_municipio_media"]
    merged = proc.merge(
        gold[["_lat", "_lon", "municipio_otro"] + geo_feats],
        on=["_lat", "_lon"],
        how="inner",
    )

    means = rent_cfg.get("mun_means_sale", {})
    global_mean = rent_cfg.get("mun_global_mean_sale", float(rent_geo_ref["precio_m2_municipio_media"].mean()))
    for municipio, group in merged[merged["municipio_otro"] == 1].groupby("municipality"):
        if municipio in rent_geo_ref.index:
            continue
        row = {c: group[c].median() if c in group.columns else np.nan for c in geo_feats}
        row["precio_m2_municipio_media"] = means.get(municipio, global_mean)
        rent_geo_ref.loc[municipio] = row
    return rent_geo_ref


@st.cache_resource(show_spinner="Cargando modelos…")
def load_artifacts() -> dict:
    sale_params_path = PARAMS_DIR / "params_sale.json"
    rent_params_path = PARAMS_DIR / "params_rent.json"
    sale_path = MODEL_SALE_PATH
    rent_path = MODEL_RENT_PATH

    if not sale_params_path.exists() or not rent_params_path.exists() or not sale_path.exists() or not rent_path.exists():
        raise FileNotFoundError(
            "Faltan datos/configuracion para replicar "
            "`notebooks/05_ML/55_input_result.ipynb`."
        )

    sale_cfg = json.loads(sale_params_path.read_text(encoding="utf-8"))
    rent_cfg = json.loads(rent_params_path.read_text(encoding="utf-8"))

    random_state = sale_cfg["random_state"]
    test_size = sale_cfg["test_size"]

    sale_xgb_params = {**sale_cfg["xgb_params"], "random_state": random_state, "n_jobs": -1, "verbosity": 0}
    rent_xgb_params = {**rent_cfg["xgb_params"], "random_state": random_state, "n_jobs": -1, "verbosity": 0}

    df_sale = pd.read_csv(sale_path)
    df_sale = df_sale[df_sale[sale_cfg["target_col"]].notna()].copy()
    X_sale, feats_sale, medians_sale = build_training_X(
        df_sale,
        sale_cfg["base_features"],
        sale_cfg["min_muni_obs"],
    )
    y_sale = df_sale[sale_cfg["target_col"]].values
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_sale,
        y_sale,
        test_size=test_size,
        random_state=random_state,
    )
    model_sale = XGBRegressor(**sale_xgb_params)
    model_sale.fit(Xs_tr, ys_tr)

    df_rent = pd.read_csv(rent_path)
    df_rent = df_rent[df_rent[rent_cfg["target_col"]].notna() & df_rent["precio_m2"].notna()].copy()
    X_rent, feats_rent, medians_rent = build_training_X(
        df_rent,
        rent_cfg["base_features"],
        rent_cfg["min_muni_obs"],
    )
    y_rent = df_rent[rent_cfg["target_col"]].values
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_rent,
        y_rent,
        test_size=test_size,
        random_state=random_state,
    )
    model_rent = XGBRegressor(**rent_xgb_params)
    model_rent.fit(Xr_tr, yr_tr)

    sale_geo_ref = build_geo_ref(df_sale)
    rent_geo_ref = build_geo_ref(df_rent)

    sale_means = sale_cfg.get("mun_means_sale", {})
    sale_global = sale_cfg.get("mun_global_mean_sale", float(sale_geo_ref["precio_m2_municipio_media"].mean()))
    sale_geo_ref["precio_m2_municipio_media"] = sale_geo_ref.index.map(
        lambda municipio: sale_means.get(municipio, sale_global)
    )

    rent_means = rent_cfg.get("mun_means_sale", {})
    rent_global = rent_cfg.get("mun_global_mean_sale", float(rent_geo_ref["precio_m2_municipio_media"].mean()))
    rent_geo_ref["precio_m2_municipio_media"] = rent_geo_ref.index.map(
        lambda municipio: rent_means.get(municipio, rent_global)
    )
    rent_geo_ref = extend_rent_geo_ref(rent_geo_ref, rent_cfg)

    meta = {
        "feats_sale": feats_sale,
        "feats_rent": feats_rent,
        "medians_sale": medians_sale,
        "medians_rent": medians_rent,
        "sale_geo_ref": sale_geo_ref,
        "rent_geo_ref": rent_geo_ref,
        "sale_rmse_test": float(np.sqrt(mean_squared_error(ys_te, model_sale.predict(Xs_te)))),
        "rent_rmse_test": float(np.sqrt(mean_squared_error(yr_te, model_rent.predict(Xr_te)))),
        "planta_num_values": sorted(df_sale["planta_num"].dropna().astype(int).unique().tolist()),
    }

    return {
        "model_sale": model_sale,
        "model_rent": model_rent,
        "meta": meta,
    }


@st.cache_data(show_spinner=False)
def load_municipio_universe() -> list[str]:
    """Fallback: lee municipios soportados por el modelo de venta."""
    municipios: set[str] = set()
    if MODEL_SALE_PATH.exists():
        for c in pd.read_csv(MODEL_SALE_PATH, nrows=0).columns:
            if c.startswith("municipio_"):
                name = c.replace("municipio_", "")
                if name not in {"otro", "otros"}:
                    municipios.add(name)
    return sorted(municipios) + ["Otros"]


def municipios_from_meta(meta: dict) -> list[str]:
    municipios: set[str] = set()
    sale_geo_ref = meta.get("sale_geo_ref")
    if sale_geo_ref is not None:
        municipios.update(str(m) for m in sale_geo_ref.index if str(m) not in {"otro", "otros"})
    if not municipios:
        return load_municipio_universe()
    return sorted(municipios) + ["Otros"]


def rmse_from_meta(meta: dict, prefix: str) -> float:
    for key in (f"{prefix}_rmse_log", f"{prefix}_rmse_test"):
        if key in meta:
            return float(meta[key])
    raise KeyError(f"No se encontro RMSE para {prefix} en encoders.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del vector de features (replica _build_row del notebook)
# ─────────────────────────────────────────────────────────────────────────────
def build_input_row(
    *,
    municipio: str,
    superficie_m2: float,
    n_dormitorios: int,
    n_banos: int,
    tipologia: str,             # "piso" | "unifamiliar"
    tiene_garaje: bool,
    obra_nueva: bool,
    planta_num: int | None,
    es_exterior: bool | None,
    tiene_ascensor: bool | None,
    feature_cols: list[str],
    geo_ref: pd.DataFrame,
    medians: dict,
) -> pd.DataFrame:
    row = pd.Series(np.nan, index=feature_cols, dtype="float64")

    def _set(k, v):
        if k in row.index and v is not None:
            row[k] = float(v)

    _set("superficie_construida_m2",        superficie_m2)
    _set("numero_dormitorios",              n_dormitorios)
    _set("numero_banos",                    n_banos)
    _set("tiene_garaje",                    int(tiene_garaje))
    _set("obra_nueva",                      int(obra_nueva))
    _set("tipologia_unificada_piso",        1 if tipologia == "piso"        else 0)
    _set("tipologia_unificada_unifamiliar", 1 if tipologia == "unifamiliar" else 0)

    if tipologia == "piso":
        _set("planta_num",          planta_num)
        _set("es_exterior_piso",    int(es_exterior)    if es_exterior    is not None else None)
        _set("tiene_ascensor_piso", int(tiene_ascensor) if tiene_ascensor is not None else None)
        if planta_num is not None and tiene_ascensor is not None:
            _set("interaccion_planta_sin_ascensor_piso",
                 planta_num * (1 - int(tiene_ascensor)))

    # ── Geo features por municipio (igual que _build_row del notebook) ───────
    if municipio in geo_ref.index:
        for col, val in geo_ref.loc[municipio].to_dict().items():
            _set(col, val)

    # ── Activar one-hot del municipio ────────────────────────────────────────
    for c in feature_cols:
        if c.startswith("municipio_"):
            row[c] = 0.0
    mun_col = f"municipio_{municipio}"
    if mun_col in row.index:
        row[mun_col] = 1.0
    elif "municipio_otro" in row.index:
        row["municipio_otro"] = 1.0
    elif "municipio_otros" in row.index:
        row["municipio_otros"] = 1.0

    # ── Imputación final ──────────────────────────────────────────────────────
    for col in feature_cols:
        if pd.isna(row[col]):
            if tipologia == "unifamiliar" and col in PISO_ONLY_FEATURES:
                continue  # se mantiene NaN — XGBoost lo enruta
            row[col] = float(medians.get(col, 0.0))

    return pd.DataFrame([row])


def predict_log_price(model, X: pd.DataFrame, rmse_log: float) -> tuple[float, float, float]:
    log_pred = float(model.predict(X)[0])
    pred = float(np.exp(log_pred))
    lo   = pred * float(np.exp(-rmse_log))
    hi   = pred * float(np.exp( rmse_log))
    return pred, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Idealista — construcción de URL + scraping
# ─────────────────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    """Slug compatible con Idealista (sin tildes, en minúsculas, con guiones)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


_NUM_WORD = {1: "un", 2: "dos", 3: "tres", 4: "cuatro"}


def dormitorios_token(n: int) -> str | None:
    if n is None or n <= 0:
        return None
    if n == 1:
        return "de-un-dormitorio"
    if n in _NUM_WORD:
        return f"de-{_NUM_WORD[n]}-dormitorios"
    return f"de-{n}-dormitorios"


def banos_token(n: int) -> str | None:
    if n is None or n <= 0:
        return None
    if n == 1:
        return "un-bano"
    if n in _NUM_WORD:
        return f"{_NUM_WORD[n]}-banos"
    return f"{n}-banos"


def build_idealista_url(
    *,
    modo: str,                # "venta" | "alquiler"
    municipio: str,
    tipologia: str,           # "piso" | "unifamiliar"
    n_dormitorios: int,
    n_banos: int,
    es_exterior: bool | None,
    tiene_ascensor: bool | None,
    tiene_garaje: bool,
    obra_nueva: bool,
) -> str:
    base = "https://www.idealista.com"
    section = "venta-viviendas" if modo == "venta" else "alquiler-viviendas"

    if municipio == "Otros":
        return f"{base}/{section}/cantabria-provincia/"

    muni_slug = f"{slugify(municipio)}-cantabria"

    filters: list[str] = []
    if tipologia == "piso":
        filters.append("con-pisos")
    elif tipologia == "unifamiliar":
        filters.append("con-chalets")

    for tk in (dormitorios_token(n_dormitorios), banos_token(n_banos)):
        if tk:
            filters.append(tk)
    if tiene_garaje:
        filters.append("garaje")
    if es_exterior:
        filters.append("exterior")
    if tiene_ascensor:
        filters.append("ascensor")
    if obra_nueva:
        filters.append("obra-nueva")

    if filters:
        return f"{base}/{section}/{muni_slug}/con-{','.join(filters).replace('con-', '', 1)}/"
    return f"{base}/{section}/{muni_slug}/"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_idealista_listings(url: str, max_results: int = 6) -> list[dict]:
    """Devuelve hasta `max_results` viviendas reales del primer SERP de Idealista.

    Idealista bloquea agresivamente bots: si la petición falla devolvemos lista
    vacía y la UI muestra solo el link directo.
    """
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        if r.status_code != 200 or "captcha" in r.text.lower():
            return []
    except requests.RequestException:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    items: list[dict] = []

    for art in soup.select("article.item")[:max_results]:
        a = art.select_one("a.item-link")
        if a is None:
            continue
        href  = urljoin("https://www.idealista.com", a.get("href", ""))
        title = a.get_text(strip=True)

        precio_el = art.select_one(".item-price, .price-row")
        precio_txt = precio_el.get_text(strip=True) if precio_el else ""
        precio_eur = _parse_price(precio_txt)

        img = art.select_one("img")
        img_url = ""
        if img is not None:
            img_url = img.get("data-src") or img.get("src") or ""

        details = [d.get_text(strip=True) for d in art.select(".item-detail")]

        items.append({
            "title": title,
            "url": href,
            "precio_txt": precio_txt,
            "precio_eur": precio_eur,
            "img": img_url,
            "details": details,
            "source": "idealista",
        })
    return items


def _parse_price(txt: str) -> float | None:
    if not txt:
        return None
    digits = re.sub(r"[^\d]", "", txt)
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Idealista local — listings desde los CSV gold completos para Streamlit
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_LISTING_PATHS = {
    "venta": [
        STREAMLIT_SALE_PATH,
    ],
    "alquiler": [
        STREAMLIT_RENT_PATH,
    ],
}


LOCAL_LISTING_COLS = [
    "propertyCode", "thumbnail", "url", "price", "priceByArea", "propertyType", "operation", "size",
    "rooms", "bathrooms", "address", "municipality", "district",
    "status", "newDevelopment", "parkingSpace.hasParkingSpace", "suggestedTexts.title",
    "suggestedTexts.subtitle", "detailedType.typology", "detailedType.subTypology",
    "municipio", "tipologia_unificada", "numero_dormitorios", "numero_banos",
    "superficie_construida_m2", "obra_nueva", "tiene_garaje", "es_exterior",
    "tiene_ascensor", "planta_num", "exterior", "hasLift", "floor",
]


@st.cache_data(show_spinner=False)
def load_local_listings(modo: str) -> pd.DataFrame:
    for path in LOCAL_LISTING_PATHS[modo]:
        if path.exists():
            cols = pd.read_csv(path, nrows=0).columns
            usecols = [c for c in LOCAL_LISTING_COLS if c in cols]
            return pd.read_csv(path, usecols=usecols)
    return pd.DataFrame(columns=LOCAL_LISTING_COLS)


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "1", "1.0", "yes", "y", "si", "sí", "s"}:
            return True
        if text in {"false", "f", "0", "0.0", "no", "n", "nan", "none", ""}:
            return False
    return bool(value)


def _first_listing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _strict_numeric_filter(df: pd.DataFrame, col: str | None, expected: int | float) -> pd.DataFrame:
    if col is None:
        return df
    values = pd.to_numeric(df[col], errors="coerce")
    return df[values == expected]


def _strict_bool_filter(df: pd.DataFrame, col: str | None, expected: bool = True) -> pd.DataFrame:
    if col is None:
        return df
    values = df[col].map(_truthy)
    return df[values == bool(expected)]


def _listing_title(row: pd.Series) -> str:
    for col in ("suggestedTexts.title", "address", "suggestedTexts.subtitle"):
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return "Vivienda en Idealista"


def _listing_details(row: pd.Series) -> list[str]:
    details = []
    for col, suffix in (("size", "m²"), ("rooms", "hab."), ("bathrooms", "baños")):
        value = row.get(col)
        if pd.notna(value):
            details.append(f"{int(value)} {suffix}")
    district = row.get("district")
    if pd.notna(district) and str(district).strip():
        details.append(str(district).strip())
    return details


def normalize_listing_url(value, property_code=None) -> str:
    if pd.notna(property_code):
        code = str(property_code).strip()
        if code.endswith(".0"):
            code = code[:-2]
        if code and code.lower() != "nan":
            return f"https://www.idealista.com/inmueble/{code}/"

    if pd.isna(value):
        return ""
    url = str(value).strip()
    if not url or url.lower() == "nan":
        return ""
    if url.startswith("/"):
        return urljoin("https://www.idealista.com", url)
    if not url.startswith("http"):
        return f"https://www.idealista.com/{url.lstrip('/')}"
    return url


def find_local_listings(
    *,
    modo: str,
    municipio: str,
    tipologia: str,
    superficie_m2: float,
    n_dormitorios: int,
    n_banos: int,
    planta_num: int | None,
    es_exterior: bool | None,
    tiene_ascensor: bool | None,
    tiene_garaje: bool,
    obra_nueva: bool,
    max_results: int | None = None,
) -> list[dict]:
    df = load_local_listings(modo)
    if df.empty:
        return []

    filtered = df.copy()

    municipio_col = _first_listing_col(filtered, ["municipio", "municipality"])
    if municipio != "Otros" and municipio_col is not None:
        filtered = filtered[filtered[municipio_col].astype(str).str.strip() == municipio]

    tipologia_col = _first_listing_col(filtered, ["tipologia_unificada", "propertyType", "detailedType.typology"])
    if tipologia_col is not None:
        values = filtered[tipologia_col].astype(str).str.lower().str.strip()
        if tipologia_col == "tipologia_unificada":
            type_mask = values == tipologia
        elif tipologia == "piso":
            type_mask = values.isin({"flat", "penthouse", "duplex"})
        else:
            type_mask = values.isin({"chalet", "countryhouse"})
        filtered = filtered[type_mask]

    rooms_col = _first_listing_col(filtered, ["numero_dormitorios", "rooms"])
    filtered = _strict_numeric_filter(filtered, rooms_col, int(n_dormitorios))

    bathrooms_col = _first_listing_col(filtered, ["numero_banos", "bathrooms"])
    filtered = _strict_numeric_filter(filtered, bathrooms_col, int(n_banos))

    if tipologia == "piso":
        floor_col = _first_listing_col(filtered, ["planta_num"])
        if planta_num is not None:
            filtered = _strict_numeric_filter(filtered, floor_col, int(planta_num))

        exterior_col = _first_listing_col(filtered, ["es_exterior", "exterior"])
        if es_exterior is not None:
            filtered = _strict_bool_filter(filtered, exterior_col, bool(es_exterior))

        lift_col = _first_listing_col(filtered, ["tiene_ascensor", "hasLift"])
        if tiene_ascensor is not None:
            filtered = _strict_bool_filter(filtered, lift_col, bool(tiene_ascensor))

    if tiene_garaje:
        garage_col = _first_listing_col(filtered, ["tiene_garaje", "parkingSpace.hasParkingSpace"])
        filtered = _strict_bool_filter(filtered, garage_col, True)

    if obra_nueva:
        new_build_col = _first_listing_col(filtered, ["obra_nueva", "newDevelopment"])
        filtered = _strict_bool_filter(filtered, new_build_col, True)

    if filtered.empty:
        return []

    score = pd.Series(0.0, index=filtered.index)
    if "rooms" in filtered.columns:
        score += (pd.to_numeric(filtered["rooms"], errors="coerce").fillna(n_dormitorios) - n_dormitorios).abs() * 3
    if "bathrooms" in filtered.columns:
        score += (pd.to_numeric(filtered["bathrooms"], errors="coerce").fillna(n_banos) - n_banos).abs() * 2
    if "size" in filtered.columns:
        score += (pd.to_numeric(filtered["size"], errors="coerce").fillna(superficie_m2) - superficie_m2).abs() / 50
    if "parkingSpace.hasParkingSpace" in filtered.columns and tiene_garaje:
        score += filtered["parkingSpace.hasParkingSpace"].map(lambda v: 0 if _truthy(v) else 1)
    if "newDevelopment" in filtered.columns and obra_nueva:
        score += filtered["newDevelopment"].map(lambda v: 0 if _truthy(v) else 2)

    filtered = filtered.assign(_score=score).sort_values(["_score", "price"], ascending=[True, True])

    items = []
    if max_results is not None:
        filtered = filtered.head(max_results)

    for _, row in filtered.iterrows():
        price = pd.to_numeric(row.get("price"), errors="coerce")
        price_eur = None if pd.isna(price) else float(price)
        price_txt = ""
        if price_eur is not None:
            suffix = "€/mes" if modo == "alquiler" else "€"
            price_txt = f"{price_eur:,.0f} {suffix}".replace(",", ".")
        items.append({
            "title": _listing_title(row),
            "url": normalize_listing_url(row.get("url", ""), row.get("propertyCode")),
            "precio_txt": price_txt,
            "precio_eur": price_eur,
            "img": row.get("thumbnail", ""),
            "details": _listing_details(row),
            "property_code": "" if pd.isna(row.get("propertyCode")) else str(row.get("propertyCode")).replace(".0", ""),
            "local_status": "" if pd.isna(row.get("status")) else str(row.get("status")),
            "source": "local",
        })
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_image_bytes(url: str) -> bytes | None:
    if not url:
        return None
    headers = {
        **_HEADERS,
        "Referer": "https://www.idealista.com/",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code == 200 and content_type.startswith("image/"):
            return response.content
    except requests.RequestException:
        return None
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def check_listing_availability(url: str) -> str:
    """Devuelve sold solo cuando Idealista permite identificarlo claramente.

    Idealista suele devolver 403 a peticiones automáticas, incluso para anuncios
    vivos. En ese caso devolvemos unknown para no marcar falsos vendidos.
    """
    if not url:
        return "unknown"

    headers = {
        **_HEADERS,
        "Referer": "https://www.idealista.com/",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
    except requests.RequestException:
        return "unknown"

    if response.status_code in {404, 410}:
        return "sold"
    if response.status_code in {401, 403, 429}:
        return "unknown"

    final_url = response.url.lower()
    body = response.text.lower()
    sold_markers = (
        "inmueble no encontrado",
        "anuncio no encontrado",
        "este anuncio ya no",
        "el anuncio ya no",
        "no existe",
        "no está disponible",
        "no esta disponible",
    )
    if "inmueble" not in final_url and any(marker in body for marker in sold_markers):
        return "sold"
    if any(marker in body for marker in sold_markers):
        return "sold"
    return "available"


def render_listings_section(
    *,
    modo_url: str,
    titulo: str,
    precio_teorico: float,
    municipio: str,
    tipologia: str,
    superficie_m2: float,
    n_dormitorios: int,
    n_banos: int,
    planta_num: int | None,
    es_exterior: bool | None,
    tiene_ascensor: bool | None,
    tiene_garaje: bool,
    obra_nueva: bool,
) -> None:
    idealista_url = build_idealista_url(
        modo=modo_url,
        municipio=municipio,
        tipologia=tipologia,
        n_dormitorios=int(n_dormitorios),
        n_banos=int(n_banos),
        es_exterior=es_exterior,
        tiene_ascensor=tiene_ascensor,
        tiene_garaje=tiene_garaje,
        obra_nueva=obra_nueva,
    )

    st.subheader(titulo)
    st.link_button("Abrir búsqueda en Idealista", idealista_url, use_container_width=False)

    listings = find_local_listings(
        modo=modo_url,
        municipio=municipio,
        tipologia=tipologia,
        superficie_m2=superficie_m2,
        n_dormitorios=int(n_dormitorios),
        n_banos=int(n_banos),
        planta_num=planta_num,
        es_exterior=es_exterior,
        tiene_ascensor=tiene_ascensor,
        tiene_garaje=tiene_garaje,
        obra_nueva=obra_nueva,
    )
    if not listings:
        with st.spinner(f"Buscando viviendas de {titulo.lower()} en Idealista…"):
            listings = fetch_idealista_listings(idealista_url, max_results=6)

    if not listings:
        st.warning(
            "No hay registros locales similares y no se han podido recuperar "
            f"resultados directamente desde Idealista para {titulo.lower()}."
        )
        return

    source = "en el dataset" if all(item.get("source") == "local" for item in listings) else "en Idealista"
    st.caption(f"{len(listings)} viviendas encontradas {source} para esta configuración.")

    cols = st.columns(min(3, len(listings)))
    for i, item in enumerate(listings):
        with cols[i % len(cols)]:
            if item["img"]:
                st.image(item["img"], use_container_width=True)
            st.markdown(f"**{item['title']}**")
            st.write(item["precio_txt"])
            if item["precio_eur"] and precio_teorico:
                diff = item["precio_eur"] - precio_teorico
                pct = (diff / precio_teorico) * 100
                arrow = "🔺" if diff > 0 else "🔻"
                st.caption(
                    f"{arrow} {diff:+,.0f} € ({pct:+.1f}%) vs precio teórico"
                    .replace(",", ".")
                )
            if item["details"]:
                st.caption(" · ".join(item["details"][:3]))
            if item["url"]:
                st.link_button("Abrir anuncio", item["url"], use_container_width=True)
            property_code = item.get("property_code")
            if property_code:
                st.caption(f"Ref. Idealista: {property_code}")


# ─────────────────────────────────────────────────────────────────────────────
# Mapa Folium
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_map_listings() -> pd.DataFrame:
    frames = []
    for modo in ("venta", "alquiler"):
        df = load_local_listings(modo).copy()
        if df.empty:
            continue
        path = next((p for p in LOCAL_LISTING_PATHS[modo] if p.exists()), None)
        if path is not None:
            cols = pd.read_csv(path, nrows=0).columns
            geo_cols = [c for c in ("latitude", "longitude") if c in cols]
            if geo_cols:
                geo = pd.read_csv(path, usecols=["propertyCode", *geo_cols])
                df = df.merge(geo, on="propertyCode", how="left")
        df["modo"] = modo
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data["latitude"] = pd.to_numeric(data.get("latitude"), errors="coerce")
    data["longitude"] = pd.to_numeric(data.get("longitude"), errors="coerce")
    data["price"] = pd.to_numeric(data.get("price"), errors="coerce")
    data["priceByArea"] = pd.to_numeric(data.get("priceByArea"), errors="coerce")
    data["size"] = pd.to_numeric(data.get("size"), errors="coerce")
    data["rooms"] = pd.to_numeric(data.get("rooms"), errors="coerce")
    data["bathrooms"] = pd.to_numeric(data.get("bathrooms"), errors="coerce")
    data = data.dropna(subset=["latitude", "longitude", "price"])
    if "priceByArea" not in data.columns:
        data["priceByArea"] = np.nan
    data["priceByArea"] = data["priceByArea"].fillna(data["price"] / data["size"].replace(0, np.nan))
    data["url_norm"] = data.apply(lambda row: normalize_listing_url(row.get("url", ""), row.get("propertyCode")), axis=1)
    data["title"] = data.apply(_listing_title, axis=1)
    return data


def _format_price(value: float, modo: str) -> str:
    suffix = "€/mes" if modo == "alquiler" else "€"
    return f"{value:,.0f} {suffix}".replace(",", ".")


def _format_eur_m2(value: float, modo: str) -> str:
    suffix = "€/m²/mes" if modo == "alquiler" else "€/m²"
    return f"{value:,.0f} {suffix}".replace(",", ".")


def _map_popup_html(row: pd.Series) -> str:
    title = html.escape(str(row.get("title", "Vivienda")))
    modo = row.get("modo", "venta")
    price = _format_price(float(row["price"]), modo)
    price_area = _format_eur_m2(float(row["priceByArea"]), modo) if pd.notna(row.get("priceByArea")) else ""
    img = row.get("thumbnail", "")
    url = row.get("url_norm", "")
    details = []
    for col, suffix in (("size", "m²"), ("rooms", "hab."), ("bathrooms", "baños")):
        value = row.get(col)
        if pd.notna(value):
            details.append(f"{int(value)} {suffix}")
    detail_txt = " · ".join(details)
    image_html = f'<img src="{img}" style="width:220px;height:135px;object-fit:cover;border-radius:6px;margin-bottom:8px;">' if pd.notna(img) and str(img).strip() else ""
    link_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer">Abrir anuncio</a>' if url else ""
    return f"""
    <div style="width:230px">
      {image_html}
      <div style="font-weight:700;margin-bottom:4px;">{title}</div>
      <div style="font-size:15px;margin-bottom:4px;">{price}</div>
      <div style="color:#555;margin-bottom:4px;">{price_area}</div>
      <div style="color:#555;margin-bottom:8px;">{detail_txt}</div>
      {link_html}
    </div>
    """


def haversine_km(lat: float, lon: float, lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lat_series.astype(float))
    lon2 = np.radians(lon_series.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * (2 * np.arcsin(np.sqrt(a)))


def _zone_color(value: float, low: float, high: float) -> str:
    if value <= low:
        return "#16a34a"
    if value >= high:
        return "#dc2626"
    return "#f59e0b"


def _operation_filter(data: pd.DataFrame, operacion: str) -> pd.DataFrame:
    if operacion == "Compra-venta":
        return data[data["modo"] == "venta"].copy()
    if operacion == "Alquiler":
        return data[data["modo"] == "alquiler"].copy()
    return data.copy()


def render_map_page() -> None:
    st.title("🗺️ Mapa de viviendas")

    data = load_map_listings()
    if data.empty:
        st.warning("No hay viviendas con coordenadas disponibles para pintar el mapa.")
        return

    tab_zones, tab_search = st.tabs(["Zonas caras/baratas", "Buscar viviendas"])

    with tab_zones:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            operacion = st.radio("Operación", ["Compra-venta", "Alquiler"], horizontal=False, key="zone_operation")
            min_obs = st.number_input("Mínimo de viviendas por zona", min_value=1, max_value=100, value=5, step=1)

        filtered = _operation_filter(data, operacion).dropna(subset=["priceByArea", "municipality"])
        zones = (
            filtered.groupby("municipality", as_index=False)
            .agg(
                precio_m2_mediano=("priceByArea", "median"),
                precio_mediano=("price", "median"),
                viviendas=("propertyCode", "count"),
                latitud=("latitude", "mean"),
                longitud=("longitude", "mean"),
            )
        )
        zones = zones[zones["viviendas"] >= min_obs].copy()

        with col_b:
            if zones.empty:
                st.warning("No hay zonas suficientes para los filtros seleccionados.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Zonas", len(zones))
                c2.metric("Zona más cara", zones.sort_values("precio_m2_mediano", ascending=False).iloc[0]["municipality"])
                c3.metric("Zona más barata", zones.sort_values("precio_m2_mediano", ascending=True).iloc[0]["municipality"])

        if not zones.empty:
            low = float(zones["precio_m2_mediano"].quantile(0.33))
            high = float(zones["precio_m2_mediano"].quantile(0.67))
            fmap = folium.Map(
                location=[float(zones["latitud"].mean()), float(zones["longitud"].mean())],
                zoom_start=9,
                tiles="CartoDB positron",
            )
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

            for _, row in zones.iterrows():
                color = _zone_color(float(row["precio_m2_mediano"]), low, high)
                html_popup = f"""
                <div style="width:220px">
                  <div style="font-weight:700;margin-bottom:4px;">{html.escape(str(row["municipality"]))}</div>
                  <div>{_format_eur_m2(float(row["precio_m2_mediano"]), "alquiler" if operacion == "Alquiler" else "venta")}</div>
                  <div>{_format_price(float(row["precio_mediano"]), "alquiler" if operacion == "Alquiler" else "venta")}</div>
                  <div>{int(row["viviendas"])} viviendas</div>
                </div>
                """
                folium.CircleMarker(
                    location=[row["latitud"], row["longitud"]],
                    radius=max(8, min(28, 6 + np.sqrt(float(row["viviendas"])))),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.65,
                    popup=folium.Popup(html_popup, max_width=260),
                    tooltip=f"{row['municipality']} · {_format_eur_m2(float(row['precio_m2_mediano']), 'alquiler' if operacion == 'Alquiler' else 'venta')}",
                ).add_to(fmap)

            legend = """
            <div style="position: fixed; bottom: 32px; left: 32px; z-index: 9999;
                        background: white; padding: 10px 12px; border: 1px solid #bbb;
                        border-radius: 6px; font-size: 13px;">
              <div><span style="color:#16a34a;">●</span> Zona barata</div>
              <div><span style="color:#f59e0b;">●</span> Zona media</div>
              <div><span style="color:#dc2626;">●</span> Zona cara</div>
            </div>
            """
            fmap.get_root().html.add_child(folium.Element(legend))
            st_folium(fmap, use_container_width=True, height=720, returned_objects=[])

    with tab_search:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            municipios = sorted(data["municipality"].dropna().astype(str).unique().tolist())
            default_municipio = "Santa Cruz de Bezana" if "Santa Cruz de Bezana" in municipios else municipios[0]
            municipio_busqueda = st.selectbox("Municipio de referencia", municipios, index=municipios.index(default_municipio))
        with col_b:
            operacion = st.radio("Operación", ["Compra-venta", "Alquiler", "Todas"], horizontal=False, key="near_operation")
            radius_km = st.slider("Radio de búsqueda (km)", min_value=0.5, max_value=25.0, value=5.0, step=0.5)
        with col_c:
            only_flats = st.checkbox("Solo pisos", value=True)
            max_points = st.number_input("Máximo de puntos", min_value=10, max_value=1000, value=300, step=10)

        nearby = _operation_filter(data, operacion)
        if only_flats:
            nearby = nearby[nearby["propertyType"].astype(str).str.lower().isin({"flat", "penthouse", "duplex"})]
        reference_rows = data[data["municipality"].astype(str) == municipio_busqueda]
        ref_lat = float(reference_rows["latitude"].mean())
        ref_lon = float(reference_rows["longitude"].mean())
        nearby = nearby.copy()
        nearby["distancia_km"] = haversine_km(ref_lat, ref_lon, nearby["latitude"], nearby["longitude"])
        nearby = nearby[nearby["distancia_km"] <= radius_km].sort_values("distancia_km")

        c1, c2, c3 = st.columns(3)
        c1.metric("Viviendas cercanas", len(nearby))
        c2.metric("Distancia mínima", f"{nearby['distancia_km'].min():.2f} km" if len(nearby) else "-")
        c3.metric("Precio mediano", _format_price(float(nearby["price"].median()), "alquiler" if operacion == "Alquiler" else "venta") if len(nearby) else "-")

        fmap = folium.Map(location=[ref_lat, ref_lon], zoom_start=12, tiles="CartoDB positron")
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
        folium.Marker(
            location=[ref_lat, ref_lon],
            tooltip=f"Referencia: {municipio_busqueda}",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(fmap)
        folium.Circle(
            location=[ref_lat, ref_lon],
            radius=float(radius_km) * 1000,
            color="#dc2626",
            fill=False,
            weight=2,
        ).add_to(fmap)

        cluster = MarkerCluster(name="Viviendas cercanas").add_to(fmap)
        for _, row in nearby.head(int(max_points)).iterrows():
            tooltip = (
                f"{_format_price(float(row['price']), row['modo'])} · "
                f"{int(row['size']) if pd.notna(row.get('size')) else '-'} m² · "
                f"{int(row['rooms']) if pd.notna(row.get('rooms')) else '-'} hab. · "
                f"{row['distancia_km']:.2f} km"
            )
            color = "blue" if row["modo"] == "venta" else "green"
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                tooltip=tooltip,
                popup=folium.Popup(_map_popup_html(row), max_width=280),
                icon=folium.Icon(color=color, icon="home", prefix="fa"),
            ).add_to(cluster)

        folium.LayerControl(collapsed=True).add_to(fmap)
        st_folium(fmap, use_container_width=True, height=720, returned_objects=[])


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def render_predictor_page(artifacts: dict) -> None:
    meta = artifacts["meta"]

    st.title("🏠 Predictor de precios")
    st.caption("Estima precio de venta y alquiler con los modelos XGBoost definitivos.")

    # ── Sidebar: formulario ───────────────────────────────────────────────────
    municipios = municipios_from_meta(meta)

    with st.sidebar:
        st.header("📋 Datos de la vivienda")

        default_idx = municipios.index("Santander") if "Santander" in municipios else 0
        municipio = st.selectbox("Municipio", municipios, index=default_idx)

        tipologia_label = st.selectbox("Tipo de vivienda", ["Piso", "Unifamiliar"])
        tipologia = "piso" if tipologia_label == "Piso" else "unifamiliar"

        col_a, col_b = st.columns(2)
        with col_a:
            superficie_m2 = st.number_input("Superficie (m²)", min_value=20.0, max_value=2000.0, value=90.0, step=5.0)
            n_dormitorios = st.number_input("Dormitorios", min_value=0, max_value=10, value=2, step=1)
        with col_b:
            n_banos = st.number_input("Baños", min_value=1, max_value=8, value=1, step=1)

        tiene_garaje = st.checkbox("Garaje", value=False)
        obra_nueva   = st.checkbox("Obra nueva", value=False)

        planta_num = es_exterior = tiene_ascensor = None
        if tipologia == "piso":
            st.divider()
            st.subheader("🏢 Características del piso")
            planta_values = meta.get("planta_num_values", list(range(0, 11)))
            planta_options = [None, *planta_values]
            planta_num = st.selectbox(
                "Planta",
                options=planta_options,
                format_func=lambda x: "Indiferente" if x is None else ("Bajo" if x == 0 else f"Planta {int(x)}"),
                index=0,
                help="'Indiferente' usa una planta típica (mediana) y no filtra los listados por planta.",
            )
            es_exterior    = st.checkbox("Exterior", value=True)
            tiene_ascensor = st.checkbox("Ascensor", value=True)

        run = st.button("🎯 Estimar precio", type="primary", use_container_width=True)

    # ── Main: estado vacío ───────────────────────────────────────────────────
    if not run:
        st.info("Rellena el formulario en el lateral y pulsa **Estimar precio** para ver los resultados.")
        return

    # ── Predicción ────────────────────────────────────────────────────────────
    with st.spinner("Calculando estimaciones…"):
        X_sale = build_input_row(
            municipio=municipio, superficie_m2=superficie_m2,
            n_dormitorios=n_dormitorios, n_banos=n_banos,
            tipologia=tipologia, tiene_garaje=tiene_garaje, obra_nueva=obra_nueva,
            planta_num=planta_num, es_exterior=es_exterior, tiene_ascensor=tiene_ascensor,
            feature_cols=meta["feats_sale"],
            geo_ref=meta["sale_geo_ref"],
            medians=meta["medians_sale"],
        )
        X_rent = build_input_row(
            municipio=municipio, superficie_m2=superficie_m2,
            n_dormitorios=n_dormitorios, n_banos=n_banos,
            tipologia=tipologia, tiene_garaje=tiene_garaje, obra_nueva=obra_nueva,
            planta_num=planta_num, es_exterior=es_exterior, tiene_ascensor=tiene_ascensor,
            feature_cols=meta["feats_rent"],
            geo_ref=meta["rent_geo_ref"],
            medians=meta["medians_rent"],
        )

        precio_venta, venta_lo, venta_hi = predict_log_price(
            artifacts["model_sale"], X_sale, rmse_from_meta(meta, "sale")
        )
        precio_alquiler, alq_lo, alq_hi = predict_log_price(
            artifacts["model_rent"], X_rent, rmse_from_meta(meta, "rent")
        )

    tab_sale, tab_rent = st.tabs(["Compra-venta", "Alquiler"])

    with tab_sale:
        st.subheader("📊 Estimación de compra-venta")
        st.metric("Precio de venta", f"{precio_venta:,.0f} €".replace(",", "."))
        st.caption(f"Rango ±1σ: {venta_lo:,.0f} € – {venta_hi:,.0f} €".replace(",", "."))
        st.divider()
        render_listings_section(
            modo_url="venta",
            titulo="Viviendas en venta",
            precio_teorico=precio_venta,
            municipio=municipio,
            tipologia=tipologia,
            superficie_m2=superficie_m2,
            n_dormitorios=int(n_dormitorios),
            n_banos=int(n_banos),
            planta_num=planta_num,
            es_exterior=es_exterior,
            tiene_ascensor=tiene_ascensor,
            tiene_garaje=tiene_garaje,
            obra_nueva=obra_nueva,
        )

    with tab_rent:
        st.subheader("📊 Estimación de alquiler")
        st.metric("Alquiler mensual", f"{precio_alquiler:,.0f} €/mes".replace(",", "."))
        st.caption(f"Rango ±1σ: {alq_lo:,.0f} – {alq_hi:,.0f} €/mes".replace(",", "."))
        st.divider()
        render_listings_section(
            modo_url="alquiler",
            titulo="Viviendas en alquiler",
            precio_teorico=precio_alquiler,
            municipio=municipio,
            tipologia=tipologia,
            superficie_m2=superficie_m2,
            n_dormitorios=int(n_dormitorios),
            n_banos=int(n_banos),
            planta_num=planta_num,
            es_exterior=es_exterior,
            tiene_ascensor=tiene_ascensor,
            tiene_garaje=tiene_garaje,
            obra_nueva=obra_nueva,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    with st.sidebar:
        page = option_menu(
            menu_title="Bezanilla SL",
            options=["Predictor", "Mapa"],
            icons=["calculator", "geo-alt"],
            default_index=0,
        )

    if page == "Predictor":
        try:
            artifacts = load_artifacts()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        render_predictor_page(artifacts)
    else:
        render_map_page()


if __name__ == "__main__":
    main()
