#!/usr/bin/env python3
"""
Main corregido:
- Airbnb apagado por defecto
- si se activa (--include-airbnb) exporta Airbnb SEPARADO
- export raw vs processed coherente
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from config.settings import ensure_data_dirs, PROJECT_ROOT, MAX_PAGES_DEFAULT, MAX_RESULTS_AIRBNB, HEADLESS_MODE
ensure_data_dirs()

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from scrapers import IdealistaScraper, FotocasaScraper, AirbnbScraper
from utils import DataProcessor, DataExporter

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"scraping_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scraping.main")


class RealEstateScraper:
    VALID_PORTALS = {"idealista", "fotocasa", "airbnb"}

    def __init__(self, portals: list[str], max_pages: int, headless: bool):
        self.portals = portals
        self.max_pages = max_pages
        self.headless = headless
        self.all_data: list = []

        invalid = [p for p in self.portals if p not in self.VALID_PORTALS]
        if invalid:
            raise ValueError(f"Portales no válidos: {invalid}")

    async def scrape_all(self) -> list:
        for portal in self.portals:
            try:
                data = await self._scrape_portal(portal)
                if data:
                    self.all_data.extend(data)
                    logger.info("✓ %s: %d registros", portal, len(data))
                else:
                    logger.warning("⚠ %s: sin datos", portal)
            except Exception:
                logger.exception("Error scrapeando %s", portal)
        return self.all_data

    async def _scrape_portal(self, portal: str) -> list:
        if portal == "idealista":
            return await IdealistaScraper(headless=self.headless).scrape(max_pages=self.max_pages)
        if portal == "fotocasa":
            return await FotocasaScraper(headless=self.headless).scrape(max_pages=self.max_pages)
        if portal == "airbnb":
            return await AirbnbScraper(headless=self.headless).scrape(max_results=MAX_RESULTS_AIRBNB)
        return []


async def main() -> None:
    parser = argparse.ArgumentParser(description="Webscraper Cantabria")
    parser.add_argument("--pages", type=int, default=MAX_PAGES_DEFAULT)
    parser.add_argument("--format", choices=["csv", "excel", "json", "all"], default="all")

    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=HEADLESS_MODE,
        help=f"Headless (default {HEADLESS_MODE})",
    )

    parser.add_argument("--no-process", action="store_true", help="Exportar raw (sin limpiar)")
    parser.add_argument("--include-airbnb", action="store_true", help="Activar Airbnb (por defecto apagado)")
    args = parser.parse_args()

    if args.pages < 1:
        parser.error("--pages debe ser >= 1")

    portals = ["idealista", "fotocasa"] + (["airbnb"] if args.include_airbnb else [])

    scraper = RealEstateScraper(portals=portals, max_pages=args.pages, headless=args.headless)
    raw_data = await scraper.scrape_all()

    if not raw_data:
        logger.error("No se obtuvieron datos. Finalizando.")
        return

    residential_raw = [d for d in raw_data if d.get("portal") != "Airbnb"]
    airbnb_raw = [d for d in raw_data if d.get("portal") == "Airbnb"]

    processed_flag = not args.no_process
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- residencial (principal) ---
    residential = DataProcessor(residential_raw).clean_all() if processed_flag else residential_raw

    exporter_res = DataExporter(residential)
    base_res = f"cantabria_alquiler_residencial_{timestamp}"
    exporter_res.to_all_formats(base_filename=base_res, processed=processed_flag)
    if processed_flag:
        exporter_res.create_summary_report(
            filename=f"resumen_alquiler_residencial_{timestamp}.xlsx",
            processed=True,
        )

    # --- airbnb (separado) ---
    if args.include_airbnb and airbnb_raw:
        airbnb = DataProcessor(airbnb_raw).clean_all() if processed_flag else airbnb_raw
        exporter_ab = DataExporter(airbnb)
        base_ab = f"cantabria_airbnb_temporal_{timestamp}"
        exporter_ab.to_all_formats(base_filename=base_ab, processed=processed_flag)

    logger.info("✓ Proceso completado")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")
        sys.exit(0)
    except Exception:
        logger.exception("Error fatal")
        sys.exit(1)
