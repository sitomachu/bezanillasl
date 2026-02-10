"""
Configuración centralizada del sistema de webscraping
BezanillaSL - Real Estate Analytics
"""

from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRAPED_DATA_DIR = DATA_DIR / "scraped"
RAW_DATA_DIR = SCRAPED_DATA_DIR / "raw"
PROCESSED_DATA_DIR = SCRAPED_DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"


def ensure_data_dirs() -> None:
    """Crea directorios necesarios si no existen."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURACIÓN DE SCRAPING
# ============================================================================
MAX_PAGES_DEFAULT = 15
MAX_RESULTS_AIRBNB = 100
HEADLESS_MODE = True  # producción

CANTABRIA_COORDS = {"latitude": 43.1828, "longitude": -3.9878}

# ============================================================================
# DELAYS (ms)
# ============================================================================
DELAYS = {
    "page_load": (1500, 3000),
    "between_pages": (2000, 5000),
    "scroll": (800, 1500),
    "cookie_accept": (800, 1500),
}

# ============================================================================
# NAVEGADOR
# ============================================================================
BROWSER_CONFIG = {
    "viewport": {"width": 1920, "height": 1080},
    "locale": "es-ES",
    "timezone_id": "Europe/Madrid",
    "color_scheme": "light",
    "device_scale_factor": 1,
    "has_touch": False,
    "is_mobile": False,
}

BROWSER_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-setuid-sandbox",
]

# ============================================================================
# URLS
# ============================================================================
URLS = {
    "idealista": {
        "base": "https://www.idealista.com",
        "alquiler_cantabria": "https://www.idealista.com/alquiler-viviendas/cantabria/",
        "cookie_button": "button#didomi-notice-agree-button",
    },
    "fotocasa": {
        "base": "https://www.fotocasa.es",
        "alquiler_cantabria": "https://www.fotocasa.es/es/alquiler/viviendas/cantabria/todas-las-zonas/l",
        "cookie_button": 'button[data-testid="TcfAccept"]',
    },
    "airbnb": {
        "base": "https://www.airbnb.es",
        "search_cantabria": "https://www.airbnb.es/s/Cantabria--España/homes",
        "cookie_button": 'button[data-testid="accept-btn"]',
    },
}

# ============================================================================
# SELECTORES (pueden cambiar)
# ============================================================================
SELECTORS = {
    "idealista": {
        "listings": "article.item",
        "price": "span.item-price",
        "title": "a.item-link",
        "details": "span.item-detail",
        "location": "span.item-location",
        "description": "div.item-description",
    },
    "fotocasa": {
        "listings": "article.re-CardPackMinimal",
        "price": "span.re-CardPrice",
        "title": "p.re-CardTitle",
        "location": "p.re-CardLocation",
        "features": "span.re-CardFeaturesWithIcons-feature",
    },
    "airbnb": {
        "listings": 'div[data-testid="card-container"]',
    },
}

# ============================================================================
# EXPORT
# ============================================================================
EXPORT_CONFIG = {
    "csv_separator": ",",
    "csv_encoding": "utf-8",
    "excel_engine": "openpyxl",
    "json_indent": 2,
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# ============================================================================
# VALIDACIÓN Y LIMPIEZA
# ============================================================================
VALIDATION_RULES = {
    "precio_min": 100,
    "precio_max": 10000,
    "superficie_min": 15,
    "superficie_max": 500,
    "habitaciones_min": 0,
    "habitaciones_max": 10,
}

INVALID_KEYWORDS = [
    "garaje",
    "parking",
    "trastero",
    "local comercial",
    "oficina",
    "nave",
    "terreno",
    "solar",
]
