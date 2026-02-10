# src/scraping/scrapers/__init__.py
"""
Módulo de scrapers (opción B)
"""

from .idealista import IdealistaScraper
from .fotocasa import FotocasaScraper
from .airbnb import AirbnbScraper

__all__ = [
    "IdealistaScraper",
    "FotocasaScraper",
    "AirbnbScraper",
]
