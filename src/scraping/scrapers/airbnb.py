"""
Scraper específico para Airbnb
BezanillaSL - Real Estate Analytics
"""

import logging
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright
from scrapers.base_scraper import AntiDetectionScraper
from config.settings import URLS, SELECTORS, MAX_RESULTS_AIRBNB

logger = logging.getLogger('scraping.airbnb')


class AirbnbScraper(AntiDetectionScraper):
    """Scraper específico para el portal Airbnb"""
    
    def __init__(self, headless: bool = False):
        """
        Inicializa el scraper de Airbnb.
        
        Args:
            headless: Si True, ejecuta sin interfaz gráfica
        """
        super().__init__(headless)
        self.portal_name = "Airbnb"
        self.base_url = URLS['airbnb']['base']
        self.search_url = URLS['airbnb']['search_cantabria']
        self.cookie_selector = URLS['airbnb']['cookie_button']
        logger.info(f"{self.portal_name} scraper inicializado")
    
    async def scrape(self, max_results: int = MAX_RESULTS_AIRBNB) -> List[Dict]:
        """
        Ejecuta el scraping de Airbnb.
        
        Args:
            max_results: Número máximo de resultados a scrapear
            
        Returns:
            List[Dict]: Lista de propiedades scrapeadas
        """
        logger.info(f"Iniciando scraping de {self.portal_name} - hasta {max_results} resultados")
        
        async with async_playwright() as p:
            self.browser = await self.create_stealth_browser(p)
            self.context = await self.create_stealth_context(self.browser)
            page = await self.context.new_page()
            
            try:
                # Navegar a la página de búsqueda
                success = await self.safe_goto(page, self.search_url)
                if not success:
                    logger.error("No se pudo cargar la página de Airbnb")
                    return []
                
                # Aceptar cookies
                await self.accept_cookies(page, self.cookie_selector)
                
                # Airbnb carga resultados dinámicamente con scroll
                results_collected = 0
                scroll_attempts = 0
                max_scrolls = 20
                previous_count = 0
                
                while results_collected < max_results and scroll_attempts < max_scrolls:
                    # Scroll para cargar más resultados
                    await self.scroll_page(page, slow=True)
                    await self.human_like_delay(delay_type='scroll')
                    
                    # Extraer contenido
                    content = await self.extract_page_content(page)
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Buscar listings (los selectores de Airbnb cambian frecuentemente)
                    listings = self._find_listings(soup)
                    
                    current_count = len(listings)
                    logger.debug(f"Scroll {scroll_attempts + 1}: {current_count} listings en página")
                    
                    # Si no se encontraron más resultados, salir
                    if current_count == previous_count:
                        logger.info("No se encontraron más resultados nuevos")
                        break
                    
                    # Parsear solo los nuevos listings
                    for i, listing in enumerate(listings):
                        if i >= previous_count and results_collected < max_results:
                            try:
                                property_data = self._parse_listing(listing)
                                if property_data and self._is_new_property(property_data):
                                    self.add_data(property_data)
                                    results_collected += 1
                            except Exception as e:
                                logger.warning(f"Error parseando anuncio: {e}")
                    
                    logger.info(f"Resultados acumulados: {len(self.data)}")
                    
                    previous_count = current_count
                    scroll_attempts += 1
                    
                    # Movimiento de mouse aleatorio
                    await self.move_mouse_randomly(page)
                
            except Exception as e:
                logger.error(f"Error durante scraping de {self.portal_name}: {e}")
                
            finally:
                await self.close()
        
        logger.info(
            f"Scraping de {self.portal_name} completado: "
            f"{len(self.data)} propiedades"
        )
        return self.get_data()
    
    def _find_listings(self, soup: BeautifulSoup) -> List:
        """
        Encuentra los listings en la página usando múltiples selectores.
        Los selectores de Airbnb cambian frecuentemente.
        
        Args:
            soup: Objeto BeautifulSoup de la página
            
        Returns:
            List: Lista de elementos de listings encontrados
        """
        # Intentar con diferentes selectores
        selectors_to_try = [
            ('div', {'data-testid': 'card-container'}),
            ('div', {'itemprop': 'itemListElement'}),
            ('div', {'role': 'group'}),
            ('div', {'class': re.compile(r'.*listing.*', re.I)}),
        ]
        
        for tag, attrs in selectors_to_try:
            listings = soup.find_all(tag, attrs)
            if listings:
                logger.debug(f"Encontrados listings con selector: {tag} {attrs}")
                return listings
        
        logger.warning("No se encontraron listings con ningún selector conocido")
        return []
    
    def _parse_listing(self, listing) -> Optional[Dict]:
        """
        Parsea un anuncio individual de Airbnb.
        
        Args:
            listing: Elemento BeautifulSoup del anuncio
            
        Returns:
            Dict: Datos del inmueble o None si falla
        """
        try:
            # Título
            title_elem = listing.find(
                ['div', 'span'],
                attrs={'data-testid': re.compile(r'.*title.*', re.I)}
            )
            if not title_elem:
                title_elem = listing.find(['h3', 'h2'])
            title = title_elem.text.strip() if title_elem else None
            
            # Precio (Airbnb muestra precios por noche)
            price_elem = listing.find('span', class_=re.compile(r'.*price.*', re.I))
            if not price_elem:
                price_elem = listing.find('span', string=re.compile(r'€.*noche', re.I))
            price_text = price_elem.text.strip() if price_elem else None
            price_numeric = self._extract_numeric_price(price_text)
            
            # URL
            link_elem = listing.find('a', href=True)
            url = link_elem.get('href', '') if link_elem else None
            if url and not url.startswith('http'):
                url = self.base_url + url.split('?')[0]  # Limpiar parámetros
            
            # Ubicación
            location_elem = listing.find('span', string=re.compile(r'.*,.*Cantabria.*', re.I))
            if not location_elem:
                location_elem = listing.find(['span', 'div'], string=re.compile(r'.*España.*', re.I))
            location = location_elem.text.strip() if location_elem else "Cantabria, España"
            
            # Detalles (huéspedes, habitaciones, camas, baños)
            details_text = listing.get_text()
            rooms, size, bathrooms, guests = self._parse_details(details_text)
            
            # Tipo de propiedad
            property_type = self._extract_property_type(title, details_text)
            
            return {
                'portal': self.portal_name,
                'titulo': title,
                'precio': price_text,
                'precio_numerico': price_numeric,
                'habitaciones': rooms,
                'superficie_m2': size,
                'banos': bathrooms,
                'huespedes_max': guests,
                'ubicacion': location,
                'municipio': self._extract_municipio(location),
                'provincia': 'Cantabria',
                'descripcion': None,  # Airbnb no muestra descripción en listados
                'url': url,
                'tipo_propiedad': property_type,
                'tipo_alquiler': 'Temporal',  # Airbnb es principalmente temporal
                'fecha_scraping': self.get_timestamp(),
            }
            
        except Exception as e:
            logger.debug(f"Error parseando listing de Airbnb: {e}")
            return None
    
    def _parse_details(self, text: str) -> tuple:
        """
        Extrae detalles del texto del anuncio.
        
        Args:
            text: Texto completo del anuncio
            
        Returns:
            tuple: (habitaciones, superficie, baños, huéspedes)
        """
        rooms = None
        size = None
        bathrooms = None
        guests = None
        
        # Huéspedes: "4 huéspedes"
        guests_match = re.search(r'(\d+)\s*huésped', text, re.IGNORECASE)
        if guests_match:
            guests = int(guests_match.group(1))
        
        # Habitaciones: "2 dormitorios" o "2 habitaciones"
        rooms_match = re.search(r'(\d+)\s*(?:dormitorio|habitación)', text, re.IGNORECASE)
        if rooms_match:
            rooms = int(rooms_match.group(1))
        
        # Baños: "1 baño"
        bath_match = re.search(r'(\d+)\s*baño', text, re.IGNORECASE)
        if bath_match:
            bathrooms = int(bath_match.group(1))
        
        # Airbnb no suele mostrar m², dejamos None
        
        return rooms, size, bathrooms, guests
    
    def _extract_numeric_price(self, price_text: Optional[str]) -> Optional[float]:
        """
        Extrae el precio numérico del texto.
        Nota: Airbnb muestra precio por noche.
        
        Args:
            price_text: Texto del precio (ej: "45 € por noche")
            
        Returns:
            float: Precio numérico o None
        """
        if not price_text:
            return None
        
        # Eliminar símbolos y texto
        price_clean = re.sub(r'[^\d]', '', price_text)
        
        try:
            return float(price_clean)
        except ValueError:
            return None
    
    def _extract_property_type(self, title: Optional[str], text: str) -> str:
        """
        Extrae el tipo de propiedad.
        
        Args:
            title: Título del anuncio
            text: Texto del anuncio
            
        Returns:
            str: Tipo de propiedad
        """
        combined_text = (title or '') + ' ' + text
        combined_text = combined_text.lower()
        
        if 'apartamento' in combined_text or 'piso' in combined_text:
            return 'Apartamento'
        elif 'casa' in combined_text or 'chalet' in combined_text:
            return 'Casa'
        elif 'villa' in combined_text:
            return 'Villa'
        elif 'habitación' in combined_text:
            return 'Habitación privada'
        elif 'estudio' in combined_text:
            return 'Estudio'
        else:
            return 'Alojamiento'
    
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
        
        # Airbnb suele mostrar: "Municipio, Provincia, País"
        parts = [p.strip() for p in location.split(',')]
        if parts:
            return parts[0]  # El municipio suele ser la primera parte
        
        return location.strip()
    
    def _is_new_property(self, property_data: Dict) -> bool:
        """
        Verifica si la propiedad ya fue scrapeada (evitar duplicados).
        
        Args:
            property_data: Datos de la propiedad
            
        Returns:
            bool: True si es nueva, False si ya existe
        """
        url = property_data.get('url')
        if not url:
            return True
        
        # Verificar si la URL ya existe en el dataset
        for item in self.data:
            if item.get('url') == url:
                return False
        
        return True


if __name__ == "__main__":
    # Test del scraper
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test():
        scraper = AirbnbScraper(headless=False)
        data = await scraper.scrape(max_results=20)
        print(f"\n✓ Scrapeadas {len(data)} propiedades de Airbnb")
        if data:
            print(f"\nEjemplo de datos:")
            print(data[0])
    
    asyncio.run(test())