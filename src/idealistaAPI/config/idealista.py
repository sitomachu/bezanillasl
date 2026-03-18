from __future__ import annotations

from pathlib import Path
from typing import Final, List, Tuple


RAW_BASE: Final[Path] = Path("data/raw/idealistaAPI/raw")
PROCESSED_BASE: Final[Path] = Path("data/raw/idealistaAPI/preprocess")

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
    ("Astillero", "43.4019,-3.8186", 12000),
    ("MedioCudeyo_Solares", "43.3814,-3.7362", 14000),
    ("SantaMariaCayon", "43.3084,-3.8537", 16000),
    ("Torrelavega", "43.3493,-4.0476", 18000),
    ("Santoña", "43.4422,-3.4525", 14000),
    ("Noja", "43.4893,-3.5231", 14000),
    ("Colindres", "43.3971,-3.4497", 12000),
    ("Reinosa", "43.0007,-4.1380", 18000),
    ("CorralesDeBuelna", "43.2599,-4.0695", 12000),
    ("Cartes", "43.3222,-4.0703", 10000),
    ("LosCorralesCentro", "43.2602,-4.0724", 9000),
    ("CabezonDeLaSal", "43.3082,-4.2363", 12000),
    ("SanVicenteDeLaBarquera", "43.3851,-4.3980", 14000),
    ("Comillas", "43.3860,-4.2912", 10000),
    ("SantillanaDelMar", "43.3890,-4.1092", 10000),
    ("PuenteViesgo", "43.2996,-3.9644", 10000),
    ("Lierganes", "43.3449,-3.7422", 10000),
    ("Entrambasaguas", "43.3798,-3.6827", 10000),
    ("RibamontanAlMar", "43.4559,-3.6823", 10000),
    ("Ampuero", "43.3431,-3.4168", 10000),
    ("RamalesDeLaVictoria", "43.2578,-3.4637", 12000),
    ("Voto_Badames", "43.3805,-3.4840", 10000),
    ("CabezonDeLaSalInterior", "43.3070,-4.2350", 9000),
    ("Potes", "43.1536,-4.6227", 16000),
    ("CabezonDeLiebanA", "43.1330,-4.5828", 14000),
    ("ValDeSanVicente", "43.3772,-4.4831", 12000),
    ("ArenasDeIguna", "43.1872,-4.0478", 12000),
    ("Molledo", "43.1518,-4.0420", 12000),
]
