"""
Clase base para scrapers con técnicas anti-detección
BezanillaSL - Real Estate Analytics
"""

import asyncio
import random
import logging
from typing import Optional, Dict, List
from datetime import datetime

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from fake_useragent import UserAgent

from config.settings import (
    BROWSER_CONFIG,
    BROWSER_ARGS,
    DELAYS,
    CANTABRIA_COORDS,
    HEADLESS_MODE,
)

logger = logging.getLogger('scraping.base')


class AntiDetectionScraper:
    """
    Clase base para scrapers con técnicas anti-detección avanzadas.
    Implementa comportamiento humano y stealth mode.
    """
    
    def __init__(self, headless: bool = HEADLESS_MODE):
        """
        Inicializa el scraper con configuración anti-detección.
        
        Args:
            headless: Si True, ejecuta el navegador sin interfaz gráfica
        """
        self.ua = UserAgent()
        self.data: List[Dict] = []
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        logger.info(f"Scraper inicializado - Headless: {headless}")
    
    async def create_stealth_browser(self, playwright) -> Browser:
        """
        Crea un navegador con configuración stealth para evitar detección.
        
        Returns:
            Browser: Instancia del navegador Chromium configurado
        """
        logger.debug("Creando navegador stealth...")
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=BROWSER_ARGS
        )
        return browser
    
    async def create_stealth_context(self, browser: Browser) -> BrowserContext:
        """
        Crea un contexto de navegación con headers y configuración realista.
        
        Args:
            browser: Instancia del navegador
            
        Returns:
            BrowserContext: Contexto configurado con técnicas anti-detección
        """
        logger.debug("Creando contexto de navegación stealth...")
        
        context = await browser.new_context(
            viewport=BROWSER_CONFIG['viewport'],
            user_agent=self.ua.random,
            locale=BROWSER_CONFIG['locale'],
            timezone_id=BROWSER_CONFIG['timezone_id'],
            geolocation={
                'latitude': CANTABRIA_COORDS['latitude'],
                'longitude': CANTABRIA_COORDS['longitude']
            },
            permissions=['geolocation'],
            color_scheme=BROWSER_CONFIG['color_scheme'],
            device_scale_factor=BROWSER_CONFIG['device_scale_factor'],
            has_touch=BROWSER_CONFIG['has_touch'],
            is_mobile=BROWSER_CONFIG['is_mobile'],
            java_script_enabled=True,
        )
        
        # Inyectar scripts anti-detección en todas las páginas
        await context.add_init_script(self._get_stealth_script())
        
        logger.info("Contexto stealth creado exitosamente")
        return context
    
    def _get_stealth_script(self) -> str:
        """
        Retorna el script JavaScript para ocultar automatización.
        
        Returns:
            str: Script de inyección anti-detección
        """
        return """
        () => {
            // Ocultar propiedad webdriver
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Simular plugins del navegador
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {
                        0: {type: "application/x-google-chrome-pdf", suffixes: "pdf"},
                        description: "Portable Document Format",
                        filename: "internal-pdf-viewer",
                        name: "Chrome PDF Plugin"
                    },
                    {
                        0: {type: "application/pdf", suffixes: "pdf"},
                        description: "Portable Document Format",
                        filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                        name: "Chrome PDF Viewer"
                    }
                ]
            });
            
            // Simular idiomas del navegador
            Object.defineProperty(navigator, 'languages', {
                get: () => ['es-ES', 'es', 'en-US', 'en']
            });
            
            // Añadir objeto chrome
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
            
            // Modificar permissions API
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Ocultar automation
            delete navigator.__proto__.webdriver;
            
            // Hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8
            });
            
            // Device memory
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8
            });
            
            // Platform
            Object.defineProperty(navigator, 'platform', {
                get: () => 'Win32'
            });
            
            // Vendor
            Object.defineProperty(navigator, 'vendor', {
                get: () => 'Google Inc.'
            });
        }
        """
    
    async def human_like_delay(
        self,
        min_ms: Optional[int] = None,
        max_ms: Optional[int] = None,
        delay_type: str = 'page_load'
    ):
        """
        Simula delays humanos entre acciones.
        
        Args:
            min_ms: Delay mínimo en milisegundos
            max_ms: Delay máximo en milisegundos
            delay_type: Tipo de delay ('page_load', 'scroll', 'mouse_move', etc.)
        """
        if min_ms is None or max_ms is None:
            min_ms, max_ms = DELAYS.get(delay_type, (1000, 2000))
        
        delay_seconds = random.uniform(min_ms / 1000, max_ms / 1000)
        logger.debug(f"Delay {delay_type}: {delay_seconds:.2f}s")
        await asyncio.sleep(delay_seconds)
    
    async def scroll_page(self, page: Page, slow: bool = True):
        """
        Simula scroll humano en la página.
        
        Args:
            page: Página de Playwright
            slow: Si True, hace scroll más lento y realista
        """
        logger.debug("Iniciando scroll de página...")
        
        if slow:
            # Scroll lento y realista
            await page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = Math.floor(Math.random() * 100) + 100;
                        const timer = setInterval(() => {
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            
                            if(totalHeight >= scrollHeight - window.innerHeight){
                                clearInterval(timer);
                                resolve();
                            }
                        }, Math.floor(Math.random() * 50) + 50);
                    });
                }
            """)
        else:
            # Scroll rápido
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        
        logger.debug("Scroll completado")
    
    async def move_mouse_randomly(self, page: Page, num_movements: int = 3):
        """
        Mueve el mouse de forma aleatoria por la página.
        
        Args:
            page: Página de Playwright
            num_movements: Número de movimientos a realizar
        """
        logger.debug(f"Moviendo mouse aleatoriamente ({num_movements} movimientos)...")
        
        for i in range(random.randint(2, num_movements)):
            x = random.randint(100, 1800)
            y = random.randint(100, 900)
            await page.mouse.move(x, y)
            await self.human_like_delay(delay_type='mouse_move')
        
        logger.debug("Movimientos de mouse completados")
    
    async def accept_cookies(
        self,
        page: Page,
        selector: str,
        timeout: int = 5000
    ) -> bool:
        """
        Intenta aceptar cookies si aparece el banner.
        
        Args:
            page: Página de Playwright
            selector: Selector CSS del botón de cookies
            timeout: Timeout en milisegundos
            
        Returns:
            bool: True si se aceptaron cookies, False si no apareció el banner
        """
        try:
            logger.debug(f"Buscando botón de cookies: {selector}")
            cookie_button = await page.wait_for_selector(
                selector,
                timeout=timeout
            )
            
            if cookie_button:
                await cookie_button.click()
                await self.human_like_delay(delay_type='cookie_accept')
                logger.info("Cookies aceptadas")
                return True
                
        except Exception as e:
            logger.debug(f"No se encontró banner de cookies: {e}")
        
        return False
    
    async def safe_goto(
        self,
        page: Page,
        url: str,
        wait_until: str = 'networkidle',
        timeout: int = 60000
    ) -> bool:
        """
        Navega a una URL de forma segura con manejo de errores.
        
        Args:
            page: Página de Playwright
            url: URL de destino
            wait_until: Condición de espera ('load', 'networkidle', etc.)
            timeout: Timeout en milisegundos
            
        Returns:
            bool: True si la navegación fue exitosa
        """
        try:
            logger.info(f"Navegando a: {url}")
            await page.goto(url, wait_until=wait_until, timeout=timeout)
            await self.human_like_delay(delay_type='page_load')
            return True
            
        except Exception as e:
            logger.error(f"Error navegando a {url}: {e}")
            return False
    
    async def extract_page_content(self, page: Page) -> str:
        """
        Extrae el contenido HTML de la página.
        
        Args:
            page: Página de Playwright
            
        Returns:
            str: Contenido HTML de la página
        """
        content = await page.content()
        logger.debug(f"Contenido extraído: {len(content)} caracteres")
        return content
    
    async def close(self):
        """Cierra el navegador y limpia recursos."""
        if self.browser:
            await self.browser.close()
            logger.info("Navegador cerrado")
    
    def get_timestamp(self) -> str:
        """
        Retorna timestamp actual en formato ISO.
        
        Returns:
            str: Timestamp actual
        """
        return datetime.now().isoformat()
    
    def add_data(self, item: Dict):
        """
        Añade un item al dataset.
        
        Args:
            item: Diccionario con datos del inmueble
        """
        if item:
            self.data.append(item)
            logger.debug(f"Item añadido. Total: {len(self.data)}")
    
    def get_data(self) -> List[Dict]:
        """
        Retorna todos los datos scrapeados.
        
        Returns:
            List[Dict]: Lista de diccionarios con datos
        """
        return self.data
    
    def clear_data(self):
        """Limpia el dataset."""
        self.data = []
        logger.info("Dataset limpiado")