"""
Módulo de scrapers
Contiene scrapers específicos para cada portal inmobiliario
"""

from .base_scraper import AntiDetectionScraper
from .idealista import IdealistaScraper
from .fotocasa import FotocasaScraper
from .airbnb import AirbnbScraper

__all__ = [
    'AntiDetectionScraper',
    'IdealistaScraper',
    'FotocasaScraper',
    'AirbnbScraper',
]