"""
Scraper específico para Idealista
BezanillaSL - Real Estate Analytics
"""

import logging
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright
from scrapers.base_scraper import AntiDetectionScraper
from config.settings import URLS, SELECTORS, MAX_PAGES_DEFAULT

logger = logging.getLogger('scraping.idealista')


class IdealistaScraper(AntiDetectionScraper):
    """Scraper específico para el portal Idealista"""
    
    def __init__(self, headless: bool = False):
        """
        Inicializa el scraper de Idealista.
        
        Args:
            headless: Si True, ejecuta sin interfaz gráfica
        """
        super().__init__(headless)
        self.portal_name = "Idealista"
        self.base_url = URLS['idealista']['base']
        self.search_url = URLS['idealista']['alquiler_cantabria']
        self.cookie_selector = URLS['idealista']['cookie_button']
        logger.info(f"{self.portal_name} scraper inicializado")
    
    async def scrape(self, max_pages: int = MAX_PAGES_DEFAULT) -> List[Dict]:
        """
        Ejecuta el scraping de Idealista.
        
        Args:
            max_pages: Número máximo de páginas a scrapear
            
        Returns:
            List[Dict]: Lista de propiedades scrapeadas
        """
        logger.info(f"Iniciando scraping de {self.portal_name} - {max_pages} páginas")
        
        async with async_playwright() as p:
            self.browser = await self.create_stealth_browser(p)
            self.context = await self.create_stealth_context(self.browser)
            page = await self.context.new_page()
            
            try:
                for page_num in range(1, max_pages + 1):
                    url = self._get_page_url(page_num)
                    
                    # Navegar a la página
                    success = await self.safe_goto(page, url)
                    if not success:
                        logger.warning(f"No se pudo cargar página {page_num}, continuando...")
                        continue
                    
                    # Aceptar cookies en la primera página
                    if page_num == 1:
                        await self.accept_cookies(page, self.cookie_selector)
                    
                    # Comportamiento humano
                    await self.scroll_page(page)
                    await self.move_mouse_randomly(page)
                    
                    # Extraer contenido
                    content = await self.extract_page_content(page)
                    listings_count = await self._parse_page(content)
                    
                    logger.info(
                        f"Página {page_num}/{max_pages}: {listings_count} anuncios "
                        f"(Total acumulado: {len(self.data)})"
                    )
                    
                    # Delay entre páginas
                    await self.human_like_delay(delay_type='between_pages')
                
            except Exception as e:
                logger.error(f"Error durante scraping de {self.portal_name}: {e}")
                
            finally:
                await self.close()
        
        logger.info(
            f"Scraping de {self.portal_name} completado: "
            f"{len(self.data)} propiedades"
        )
        return self.get_data()
    
    def _get_page_url(self, page_num: int) -> str:
        """
        Construye la URL para una página específica.
        
        Args:
            page_num: Número de página
            
        Returns:
            str: URL completa
        """
        if page_num == 1:
            return self.search_url
        return f"{self.search_url}pagina-{page_num}.htm"
    
    async def _parse_page(self, html_content: str) -> int:
        """
        Parsea el contenido HTML de una página.
        
        Args:
            html_content: Contenido HTML de la página
            
        Returns:
            int: Número de anuncios encontrados
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        selectors = SELECTORS['idealista']
        
        listings = soup.find_all('article', class_='item')
        logger.debug(f"Encontrados {len(listings)} elementos 'article.item'")
        
        for listing in listings:
            try:
                property_data = self._parse_listing(listing, selectors)
                if property_data:
                    self.add_data(property_data)
            except Exception as e:
                logger.warning(f"Error parseando anuncio individual: {e}")
        
        return len(listings)
    
    def _parse_listing(self, listing, selectors: Dict) -> Optional[Dict]:
        """
        Parsea un anuncio individual de Idealista.
        
        Args:
            listing: Elemento BeautifulSoup del anuncio
            selectors: Diccionario de selectores CSS
            
        Returns:
            Dict: Datos del inmueble o None si falla
        """
        try:
            # Precio
            price_elem = listing.find('span', class_='item-price')
            price_text = price_elem.text.strip() if price_elem else None
            price_numeric = self._extract_numeric_price(price_text)
            
            # Título
            title_elem = listing.find('a', class_='item-link')
            title = title_elem.text.strip() if title_elem else None
            url = title_elem.get('href', '') if title_elem else None
            if url and not url.startswith('http'):
                url = self.base_url + url
            
            # Ubicación
            location_elem = listing.find('span', class_='item-location')
            location = location_elem.text.strip() if location_elem else None
            
            # Detalles (habitaciones, superficie, baños)
            details_elem = listing.find('span', class_='item-detail')
            rooms, size, bathrooms = self._parse_details(details_elem)
            
            # Descripción
            desc_elem = listing.find('div', class_='item-description')
            description = desc_elem.text.strip() if desc_elem else None
            
            # Tipo de propiedad
            property_type = self._extract_property_type(title, description)
            
            return {
                'portal': self.portal_name,
                'titulo': title,
                'precio': price_text,
                'precio_numerico': price_numeric,
                'habitaciones': rooms,
                'superficie_m2': size,
                'banos': bathrooms,
                'ubicacion': location,
                'municipio': self._extract_municipio(location),
                'provincia': 'Cantabria',
                'descripcion': description,
                'url': url,
                'tipo_propiedad': property_type,
                'fecha_scraping': self.get_timestamp(),
            }
            
        except Exception as e:
            logger.debug(f"Error parseando listing: {e}")
            return None
    
    def _parse_details(self, details_elem) -> tuple:
        """
        Extrae habitaciones, superficie y baños del elemento de detalles.
        
        Args:
            details_elem: Elemento BeautifulSoup con los detalles
            
        Returns:
            tuple: (habitaciones, superficie_m2, baños)
        """
        rooms = None
        size = None
        bathrooms = None
        
        if details_elem:
            detail_text = details_elem.text.strip()
            
            # Habitaciones: "3 hab."
            rooms_match = re.search(r'(\d+)\s*hab', detail_text, re.IGNORECASE)
            if rooms_match:
                rooms = int(rooms_match.group(1))
            
            # Superficie: "85 m²"
            size_match = re.search(r'(\d+)\s*m²', detail_text)
            if size_match:
                size = int(size_match.group(1))
            
            # Baños: "2 baños"
            bath_match = re.search(r'(\d+)\s*baño', detail_text, re.IGNORECASE)
            if bath_match:
                bathrooms = int(bath_match.group(1))
        
        return rooms, size, bathrooms
    
    def _extract_numeric_price(self, price_text: Optional[str]) -> Optional[float]:
        """
        Extrae el precio numérico del texto.
        
        Args:
            price_text: Texto del precio (ej: "850 €/mes")
            
        Returns:
            float: Precio numérico o None
        """
        if not price_text:
            return None
        
        # Eliminar símbolos y texto, mantener solo dígitos
        price_clean = re.sub(r'[^\d]', '', price_text)
        
        try:
            return float(price_clean)
        except ValueError:
            return None
    
    def _extract_property_type(
        self,
        title: Optional[str],
        description: Optional[str]
    ) -> str:
        """
        Extrae el tipo de propiedad del título o descripción.
        
        Args:
            title: Título del anuncio
            description: Descripción del anuncio
            
        Returns:
            str: Tipo de propiedad
        """
        text = (title or '') + ' ' + (description or '')
        text = text.lower()
        
        if 'piso' in text or 'apartamento' in text:
            return 'Piso'
        elif 'casa' in text or 'chalet' in text:
            return 'Casa'
        elif 'ático' in text:
            return 'Ático'
        elif 'estudio' in text:
            return 'Estudio'
        elif 'dúplex' in text:
            return 'Dúplex'
        else:
            return 'Vivienda'
    
    def _extract_municipio(self, location: Optional[str]) -> Optional[str]:
        """
        Extrae el municipio de la ubicación.
        
        Args:
            location: Texto de ubicación completa
            
        Returns:
            str: Municipio extraído
        """
        if not location:
            return None
        
        # Intentar extraer el municipio (usualmente la última parte)
        parts = location.split(',')
        if parts:
            return parts[-1].strip()
        
        return location.strip()


if __name__ == "__main__":
    # Test del scraper
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test():
        scraper = IdealistaScraper(headless=False)
        data = await scraper.scrape(max_pages=2)
        print(f"\n✓ Scrapeadas {len(data)} propiedades de Idealista")
        if data:
            print(f"\nEjemplo de datos:")
            print(data[0])
    
    asyncio.run(test())