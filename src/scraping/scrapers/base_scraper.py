"""
BaseScraper (opción B):
- Playwright async
- Delays y utilidades
- Sin scripts de evasión anti-bot
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from config.settings import (
    BROWSER_ARGS,
    BROWSER_CONFIG,
    DELAYS,
    HEADLESS_MODE,
    CANTABRIA_COORDS,
)

logger = logging.getLogger("scraping.base")


@dataclass
class DelayRange:
    min_ms: int
    max_ms: int

    def sample_seconds(self) -> float:
        return random.uniform(self.min_ms / 1000, self.max_ms / 1000)


class BaseScraper:
    def __init__(self, headless: bool = HEADLESS_MODE):
        self.headless = headless
        self._playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=BROWSER_ARGS,
        )
        self.context = await self.browser.new_context(
            viewport=BROWSER_CONFIG["viewport"],
            locale=BROWSER_CONFIG["locale"],
            timezone_id=BROWSER_CONFIG["timezone_id"],
            geolocation={
                "latitude": CANTABRIA_COORDS["latitude"],
                "longitude": CANTABRIA_COORDS["longitude"],
            },
            permissions=["geolocation"],
            color_scheme=BROWSER_CONFIG["color_scheme"],
            device_scale_factor=BROWSER_CONFIG["device_scale_factor"],
            has_touch=BROWSER_CONFIG["has_touch"],
            is_mobile=BROWSER_CONFIG["is_mobile"],
            java_script_enabled=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def new_page(self) -> Page:
        if not self.context:
            raise RuntimeError("Context no inicializado. Usa 'async with scraper:'")
        return await self.context.new_page()

    @staticmethod
    def timestamp_iso() -> str:
        return datetime.now().isoformat()

    async def human_delay(
        self,
        delay_type: str = "page_load",
        custom: Optional[Tuple[int, int]] = None,
    ) -> None:
        if custom:
            d = DelayRange(custom[0], custom[1])
        else:
            mn, mx = DELAYS.get(delay_type, (800, 1500))
            d = DelayRange(mn, mx)
        await asyncio.sleep(d.sample_seconds())

    async def safe_goto(
        self,
        page: Page,
        url: str,
        wait_until: str = "networkidle",
        timeout: int = 60000,
    ) -> bool:
        try:
            logger.info("Navegando a: %s", url)
            await page.goto(url, wait_until=wait_until, timeout=timeout)
            await self.human_delay("page_load")
            return True
        except Exception:
            logger.exception("Error navegando a %s", url)
            return False

    async def accept_cookies(self, page: Page, selector: str, timeout: int = 5000) -> bool:
        try:
            btn = await page.wait_for_selector(selector, timeout=timeout)
            if btn:
                await btn.click()
                await self.human_delay("cookie_accept")
                return True
        except Exception:
            return False
        return False

    async def scroll_to_bottom(self, page: Page) -> None:
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await self.human_delay("scroll")
        except Exception:
            logger.debug("Scroll falló (no crítico).")
