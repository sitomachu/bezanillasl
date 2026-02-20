from __future__ import annotations

from pathlib import Path
from typing import Final, List, Tuple


RAW_BASE: Final[Path] = Path("data/raw/idealista")
PROCESSED_BASE: Final[Path] = Path("data/processed/idealista")

MAX_ITEMS: Final[int] = 50
SLEEP_S: Final[float] = 0.2

DEFAULT_CIRCLES: Final[List[Tuple[str, str, int]]] = [
    ("SantaCruzDeBezana", "43.4435,-3.9036", 12000),
    ("SotoDeLaMarina", "43.4620,-3.9080", 12000),
    ("Liencres", "43.4480,-3.9700", 12000),
    ("Pielagos_Boo", "43.4230,-3.9520", 14000),
    ("Camargo_Muriedas", "43.4150,-3.8540", 14000),
    ("Santander", "43.4623,-3.8099", 18000),
    ("Somo", "43.4470,-3.7450", 14000),
    ("Suances", "43.4260,-4.0430", 14000),
    ("Laredo", "43.4090,-3.4160", 16000),
    ("CastroUrdiales", "43.3830,-3.2140", 16000),
]
