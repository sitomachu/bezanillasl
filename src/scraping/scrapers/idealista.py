"""
Scraper específico para Idealista (BaseScraper)
"""

import logging
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from config.settings import URLS, SELECTORS, MAX_PAGES_DEFAULT
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger("scraping.idealista")


class IdealistaScraper(BaseScraper):
    def __init__(self, headless: bool = True):
        super().__init__(headless=headless)
        self.portal_name = "Idealista"
        self.base_url = URLS["idealista"]["base"]
        self.search_url = URLS["idealista"]["alquiler_cantabria"]
        self.cookie_selector = URLS["idealista"]["cookie_button"]
        self.data: List[Dict] = []

    async def scrape(self, max_pages: int = MAX_PAGES_DEFAULT) -> List[Dict]:
        async with self:
            page = await self.new_page()

            for page_num in range(1, max_pages + 1):
                url = self._get_page_url(page_num)
                ok = await self.safe_goto(page, url)
                if not ok:
                    continue

                if page_num == 1:
                    await self.accept_cookies(page, self.cookie_selector)

                await self.scroll_to_bottom(page)

                html = await page.content()
                self._parse_page(html)

                await self.human_delay("between_pages")

        return self.data

    def _get_page_url(self, page_num: int) -> str:
        if page_num == 1:
            return self.search_url
        return f"{self.search_url}pagina-{page_num}.htm"

    def _parse_page(self, html: str) -> None:
        soup = BeautifulSoup(html, "html.parser")
        listings = soup.select(SELECTORS["idealista"]["listings"])
        for listing in listings:
            item = self._parse_listing(listing)
            if item:
                self.data.append(item)

    def _parse_listing(self, listing) -> Optional[Dict]:
        try:
            price_elem = listing.select_one(SELECTORS["idealista"]["price"])
            price_text = price_elem.get_text(strip=True) if price_elem else None
            price_numeric = self._extract_numeric_price(price_text)

            title_elem = listing.select_one(SELECTORS["idealista"]["title"])
            title = title_elem.get_text(strip=True) if title_elem else None
            url = title_elem.get("href") if title_elem else None
            if url and not url.startswith("http"):
                url = self.base_url + url

            location_elem = listing.select_one(SELECTORS["idealista"]["location"])
            location = location_elem.get_text(strip=True) if location_elem else None

            details_elem = listing.select_one(SELECTORS["idealista"]["details"])
            rooms, size, baths = self._parse_details(details_elem.get_text(" ", strip=True) if details_elem else "")

            desc_elem = listing.select_one(SELECTORS["idealista"]["description"])
            desc = desc_elem.get_text(" ", strip=True) if desc_elem else None

            return {
                "portal": self.portal_name,
                "titulo": title,
                "precio": price_text,
                "precio_numerico": price_numeric,
                "unidad_precio": "mes",
                "habitaciones": rooms,
                "superficie_m2": size,
                "banos": baths,
                "ubicacion": location,
                "municipio": self._extract_municipio(location),
                "provincia": "Cantabria",
                "descripcion": desc,
                "url": url,
                "tipo_propiedad": self._extract_property_type(title, desc),
                "fecha_scraping": self.timestamp_iso(),
            }
        except Exception:
            logger.debug("Fallo parseando listing Idealista", exc_info=True)
            return None

    @staticmethod
    def _parse_details(text: str) -> tuple:
        rooms = None
        size = None
        baths = None

        m = re.search(r"(\d+)\s*hab", text, re.I)
        if m:
            rooms = int(m.group(1))

        m = re.search(r"(\d+)\s*m²", text)
        if m:
            size = int(m.group(1))

        m = re.search(r"(\d+)\s*bañ", text, re.I)
        if m:
            baths = int(m.group(1))

        return rooms, size, baths

    @staticmethod
    def _extract_numeric_price(price_text: Optional[str]) -> Optional[float]:
        if not price_text:
            return None
        s = re.sub(r"[^\d]", "", price_text)
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _extract_property_type(title: Optional[str], description: Optional[str]) -> str:
        t = ((title or "") + " " + (description or "")).lower()
        if "ático" in t:
            return "Ático"
        if "dúplex" in t or "duplex" in t:
            return "Dúplex"
        if "estudio" in t:
            return "Estudio"
        if "piso" in t or "apartamento" in t:
            return "Piso"
        if "casa" in t or "chalet" in t:
            return "Casa"
        return "Vivienda"

    @staticmethod
    def _extract_municipio(location: Optional[str]) -> Optional[str]:
        if not location:
            return None
        parts = [p.strip() for p in location.split(",")]
        return parts[-1] if parts else location.strip()
