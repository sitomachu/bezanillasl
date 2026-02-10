"""
Scraper específico para Fotocasa (BaseScraper)
"""

import logging
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from config.settings import URLS, SELECTORS, MAX_PAGES_DEFAULT
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger("scraping.fotocasa")


class FotocasaScraper(BaseScraper):
    def __init__(self, headless: bool = True):
        super().__init__(headless=headless)
        self.portal_name = "Fotocasa"
        self.base_url = URLS["fotocasa"]["base"]
        self.search_url = URLS["fotocasa"]["alquiler_cantabria"]
        self.cookie_selector = URLS["fotocasa"]["cookie_button"]
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
        return f"{self.search_url}/{page_num}"

    def _parse_page(self, html: str) -> None:
        soup = BeautifulSoup(html, "html.parser")
        listings = soup.select(SELECTORS["fotocasa"]["listings"])
        if not listings:
            listings = soup.find_all("article", class_=re.compile(r"re-Card"))

        for listing in listings:
            item = self._parse_listing(listing)
            if item:
                self.data.append(item)

    def _parse_listing(self, listing) -> Optional[Dict]:
        try:
            price_elem = listing.find("span", class_=re.compile(r"re-CardPrice"))
            price_text = price_elem.get_text(strip=True) if price_elem else None
            price_numeric = self._extract_numeric_price(price_text)

            title_elem = listing.find(["p", "h3"], class_=re.compile(r"re-CardTitle"))
            title = title_elem.get_text(strip=True) if title_elem else None

            link_elem = listing.find("a", href=True)
            url = link_elem.get("href") if link_elem else None
            if url and not url.startswith("http"):
                url = self.base_url + url

            location_elem = listing.find(["p", "span"], class_=re.compile(r"re-CardLocation"))
            location = location_elem.get_text(strip=True) if location_elem else None

            features = listing.find_all("span", class_=re.compile(r"re-CardFeaturesWithIcons-feature"))
            rooms, size, baths = self._parse_features(features)

            desc_elem = listing.find("p", class_=re.compile(r"re-CardDescription"))
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
            logger.debug("Fallo parseando listing Fotocasa", exc_info=True)
            return None

    @staticmethod
    def _parse_features(features) -> tuple:
        rooms = None
        size = None
        baths = None

        for f in features:
            t = f.get_text(" ", strip=True).lower()
            if "hab" in t:
                m = re.search(r"(\d+)", t)
                if m:
                    rooms = int(m.group(1))
            elif "m²" in t or "m2" in t:
                m = re.search(r"(\d+)", t)
                if m:
                    size = int(m.group(1))
            elif "bañ" in t:
                m = re.search(r"(\d+)", t)
                if m:
                    baths = int(m.group(1))

        return rooms, size, baths

    @staticmethod
    def _extract_numeric_price(price_text: Optional[str]) -> Optional[float]:
        if not price_text:
            return None
        s = price_text.replace(".", "").replace(",", "")
        s = re.sub(r"[^\d]", "", s)
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
        if len(parts) >= 2:
            return parts[-2]
        return parts[0] if parts else location.strip()
