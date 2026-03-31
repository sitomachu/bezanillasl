from __future__ import annotations

from pathlib import Path
from typing import Final, List, Tuple


RAW_BASE: Final[Path] = Path("data/raw/idealistaAPI/raw")
PROCESSED_BASE: Final[Path] = Path("data/raw/idealistaAPI/preprocess")

MAX_ITEMS: Final[int] = 50
SLEEP_S: Final[float] = 0.5

# (name, location_id, fallback_center, fallback_distance_m)
#
# location_id: código de municipio de Idealista (formato: 0-EU-ES-{provincia}-{código_INE})
#   → sin solapamiento geográfico entre municipios (zero overlap)
#   → si la API devuelve 404, se usa automáticamente el fallback center+distance
#
# fallback_center/fallback_distance_m: radio ajustado a ~municipio sin solapar con vecinos.
#   Solo se activa si location_id falla — no requiere intervención manual.
DEFAULT_LOCATIONS: Final[List[Tuple[str, str, str, int]]] = [
    ("SantaCruzDeBezana", "0-EU-ES-39-39016", "43.4435,-3.9036", 5000),
    ("Camargo",           "0-EU-ES-39-39018", "43.4150,-3.8540", 6000),
    ("Pielagos",          "0-EU-ES-39-39046", "43.4230,-3.9520", 7000),
    ("Santander",         "0-EU-ES-39-39075", "43.4623,-3.8099", 7000),
    ("MarinaDeCudeyo",    "0-EU-ES-39-39043", "43.4620,-3.9080", 5000),
    ("Miengo",            "0-EU-ES-39-39040", "43.4100,-4.0050", 5000),
    ("RibamontanAlMar",   "0-EU-ES-39-39056", "43.4470,-3.7450", 6000),
    ("Suances",           "0-EU-ES-39-39084", "43.4260,-4.0430", 6000),
    ("Laredo",            "0-EU-ES-39-39034", "43.4090,-3.4160", 7000),
    ("CastroUrdiales",    "0-EU-ES-39-39021", "43.3830,-3.2140", 7000),
]
