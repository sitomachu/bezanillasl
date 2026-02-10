"""
Configuración centralizada del sistema de webscraping
BezanillaSL - Real Estate Analytics
"""

from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRAPED_DATA_DIR = DATA_DIR / "scraped"
RAW_DATA_DIR = SCRAPED_DATA_DIR / "raw"
PROCESSED_DATA_DIR = SCRAPED_DATA_DIR / "processed"

# Crear directorios si no existen
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURACIÓN DE SCRAPING
# ============================================================================

# Zonas de búsqueda en Cantabria
LOCATIONS = {
    'cantabria': 'Cantabria',
    'santander': 'Santander',
    'torrelavega': 'Torrelavega',
    'castro_urdiales': 'Castro Urdiales',
    'reinosa': 'Reinosa',
    'laredo': 'Laredo',
    'camargo': 'Camargo',
    'pielagos': 'Piélagos',
    'bezana': 'Santa Cruz de Bezana',
    'astillero': 'El Astillero',
    'villaescusa': 'Villaescusa',
    'miengo': 'Miengo',
    'polanco': 'Polanco',
    'suances': 'Suances',
    'santillana': 'Santillana del Mar',
    'reocin': 'Reocín',
    'entrambasaguas': 'Entrambasaguas',
    'marina_cudeyo': 'Marina de Cudeyo',
    'medio_cudeyo': 'Medio Cudeyo',
    'ribamontan_mar': 'Ribamontán al Mar'
}

# Coordenadas de Cantabria
CANTABRIA_COORDS = {
    'latitude': 43.1828,
    'longitude': -3.9878
}

# Parámetros generales
MAX_PAGES_DEFAULT = 15
MAX_RESULTS_AIRBNB = 100
HEADLESS_MODE = False  # True para producción, False para debugging

# ============================================================================
# CONFIGURACIÓN DE DELAYS (milisegundos)
# ============================================================================
DELAYS = {
    'page_load': (2000, 4000),       # Entre carga de páginas
    'between_pages': (3000, 6000),   # Entre páginas del mismo portal
    'scroll': (1000, 2000),          # Durante scroll
    'mouse_move': (100, 300),        # Entre movimientos de mouse
    'cookie_accept': (1000, 2000),   # Después de aceptar cookies
}

# ============================================================================
# CONFIGURACIÓN DEL NAVEGADOR
# ============================================================================
BROWSER_CONFIG = {
    'viewport': {'width': 1920, 'height': 1080},
    'locale': 'es-ES',
    'timezone_id': 'Europe/Madrid',
    'color_scheme': 'light',
    'device_scale_factor': 1,
    'has_touch': False,
    'is_mobile': False,
}

BROWSER_ARGS = [
    '--disable-blink-features=AutomationControlled',
    '--disable-dev-shm-usage',
    '--disable-web-security',
    '--disable-features=IsolateOrigins,site-per-process',
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-gpu',
]

# ============================================================================
# URLS DE PORTALES
# ============================================================================
URLS = {
    'idealista': {
        'base': 'https://www.idealista.com',
        'alquiler_cantabria': 'https://www.idealista.com/alquiler-viviendas/cantabria/',
        'cookie_button': 'button#didomi-notice-agree-button',
    },
    'fotocasa': {
        'base': 'https://www.fotocasa.es',
        'alquiler_cantabria': 'https://www.fotocasa.es/es/alquiler/viviendas/cantabria/todas-las-zonas/l',
        'cookie_button': 'button[data-testid="TcfAccept"]',
    },
    'airbnb': {
        'base': 'https://www.airbnb.es',
        'search_cantabria': 'https://www.airbnb.es/s/Cantabria--España/homes',
        'cookie_button': 'button[data-testid="accept-btn"]',
    },
}

# ============================================================================
# SELECTORES CSS (pueden cambiar con el tiempo)
# ============================================================================
SELECTORS = {
    'idealista': {
        'listings': 'article.item',
        'price': 'span.item-price',
        'title': 'a.item-link',
        'details': 'span.item-detail',
        'location': 'span.item-location',
        'description': 'div.item-description',
    },
    'fotocasa': {
        'listings': 'article.re-CardPackMinimal',
        'price': 'span.re-CardPrice',
        'title': 'p.re-CardTitle',
        'location': 'p.re-CardLocation',
        'features': 'span.re-CardFeaturesWithIcons-feature',
    },
    'airbnb': {
        'listings': 'div[data-testid="card-container"]',
        'price': 'span._tyxjp1',
        'title': 'div[data-testid="listing-card-title"]',
        'details': 'span._1d1qzab',
    },
}

# ============================================================================
# CONFIGURACIÓN DE EXPORTACIÓN
# ============================================================================
EXPORT_CONFIG = {
    'csv_separator': ',',
    'csv_encoding': 'utf-8',
    'excel_engine': 'openpyxl',
    'json_indent': 2,
    'date_format': '%Y-%m-%d %H:%M:%S',
}

# Campos del dataset final
DATASET_FIELDS = [
    'portal',
    'titulo',
    'precio',
    'precio_numerico',
    'habitaciones',
    'superficie_m2',
    'banos',
    'ubicacion',
    'municipio',
    'provincia',
    'descripcion',
    'url',
    'latitud',
    'longitud',
    'tipo_propiedad',
    'fecha_scraping',
    'fecha_publicacion',
]

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(PROJECT_ROOT / 'logs' / 'scraping.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        'scraping': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}

# ============================================================================
# VALIDACIÓN Y LIMPIEZA DE DATOS
# ============================================================================
VALIDATION_RULES = {
    'precio_min': 100,      # Euros/mes
    'precio_max': 10000,    # Euros/mes
    'superficie_min': 15,   # m²
    'superficie_max': 500,  # m²
    'habitaciones_min': 0,
    'habitaciones_max': 10,
}

# Palabras clave para filtrar anuncios no válidos
INVALID_KEYWORDS = [
    'garaje', 'parking', 'trastero', 'local comercial',
    'oficina', 'nave', 'terreno', 'solar'
]

# ============================================================================
# RATE LIMITING Y POLÍTICAS
# ============================================================================
RATE_LIMITING = {
    'requests_per_minute': 10,
    'requests_per_hour': 200,
    'respect_robots_txt': True,
    'max_retries': 3,
    'retry_delay': 5,  # segundos
}

# ============================================================================
# CONFIGURACIÓN DE PROXIES (opcional)
# ============================================================================
USE_PROXIES = False
PROXY_LIST = [
    # 'http://proxy1.example.com:8080',
    # 'http://proxy2.example.com:8080',
]

# ============================================================================
# MODO DEBUG
# ============================================================================
DEBUG_MODE = False
SAVE_HTML_SNAPSHOTS = False  # Guardar HTML para debugging
MAX_PAGES_DEBUG = 2  # Límite en modo debug