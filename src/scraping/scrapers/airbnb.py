"""
Scraper específico para Airbnb (BaseScraper)
NOTA: desactivado por defecto desde main.py.
"""

import logging
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from config.settings import URLS, MAX_RESULTS_AIRBNB
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger("scraping.airbnb")


class AirbnbScraper(BaseScraper):
    def __init__(self, headless: bool = True):
        super().__init__(headless=headless)
        self.portal_name = "Airbnb"
        self.base_url = URLS["airbnb"]["base"]
        self.search_url = URLS["airbnb"]["search_cantabria"]
        self.cookie_selector = URLS["airbnb"]["cookie_button"]
        self.data: List[Dict] = []

    async def scrape(self, max_results: int = MAX_RESULTS_AIRBNB) -> List[Dict]:
        async with self:
            page = await self.new_page()

            ok = await self.safe_goto(page, self.search_url)
            if not ok:
                return []

            await self.accept_cookies(page, self.cookie_selector)

            collected = 0
            scrolls = 0
            max_scrolls = 20
            seen_urls = set()

            while collected < max_results and scrolls < max_scrolls:
                await self.scroll_to_bottom(page)
                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")

                cards = soup.find_all("div", attrs={"data-testid": "card-container"})
                for c in cards:
                    item = self._parse_listing(c)
                    if not item:
                        continue
                    u = item.get("url")
                    if u and u in seen_urls:
                        continue
                    if u:
                        seen_urls.add(u)

                    self.data.append(item)
                    collected += 1
                    if collected >= max_results:
                        break

                scrolls += 1
                await self.human_delay("scroll")

        return self.data

    def _parse_listing(self, listing) -> Optional[Dict]:
        try:
            link_elem = listing.find("a", href=True)
            url = link_elem.get("href") if link_elem else None
            if url and not url.startswith("http"):
                url = self.base_url + url.split("?")[0]

            title_elem = listing.find(["h3", "h2"])
            title = title_elem.get_text(" ", strip=True) if title_elem else None

            price_elem = listing.find(string=re.compile(r"€", re.I))
            price_text = price_elem.strip() if isinstance(price_elem, str) else None
            price_numeric = self._extract_numeric_price(price_text)

            return {
                "portal": self.portal_name,
                "titulo": title,
                "precio": price_text,
                "precio_numerico": price_numeric,  # €/noche
                "unidad_precio": "noche",
                "ubicacion": "Cantabria, España",
                "municipio": None,
                "provincia": "Cantabria",
                "descripcion": None,
                "url": url,
                "tipo_propiedad": "Alojamiento",
                "tipo_alquiler": "Temporal",
                "fecha_scraping": self.timestamp_iso(),
            }
        except Exception:
            logger.debug("Fallo parseando listing Airbnb", exc_info=True)
            return None

    @staticmethod
    def _extract_numeric_price(price_text: Optional[str]) -> Optional[float]:
        if not price_text:
            return None
        s = re.sub(r"[^\d]", "", price_text)
        try:
            return float(s)
        except ValueError:
            return None
