#!/usr/bin/env python3
"""
Script principal para ejecutar el webscraping de portales inmobiliarios
BezanillaSL - Real Estate Analytics

Uso:
    python main.py                                    # Scrapear todos los portales
    python main.py --portals idealista fotocasa       # Portales específicos
    python main.py --pages 20                         # Configurar páginas
    python main.py --format csv                       # Formato de salida
    python main.py --headless                         # Modo sin interfaz gráfica
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Añadir el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent))

from scrapers import IdealistaScraper, FotocasaScraper, AirbnbScraper
from utils import DataProcessor, DataExporter
from config.settings import (
    MAX_PAGES_DEFAULT,
    MAX_RESULTS_AIRBNB,
    HEADLESS_MODE,
    PROJECT_ROOT
)

# Configurar logging
log_dir = PROJECT_ROOT / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'scraping_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('scraping.main')


class RealEstateScraper:
    """Orquestador principal del sistema de scraping"""
    
    def __init__(
        self,
        portals: list = None,
        max_pages: int = MAX_PAGES_DEFAULT,
        headless: bool = HEADLESS_MODE
    ):
        """
        Inicializa el scraper principal.
        
        Args:
            portals: Lista de portales a scrapear
            max_pages: Número de páginas por portal
            headless: Ejecutar en modo headless
        """
        self.portals = portals or ['idealista', 'fotocasa', 'airbnb']
        self.max_pages = max_pages
        self.headless = headless
        self.all_data = []
        
        logger.info(
            f"RealEstateScraper inicializado - Portales: {self.portals}, "
            f"Páginas: {max_pages}, Headless: {headless}"
        )
    
    async def scrape_all(self):
        """Ejecuta el scraping de todos los portales configurados."""
        logger.info("="*70)
        logger.info("INICIANDO SCRAPING DE PORTALES INMOBILIARIOS")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Scrapear cada portal
        for portal_name in self.portals:
            logger.info(f"\n{'='*70}")
            logger.info(f"Portal: {portal_name.upper()}")
            logger.info(f"{'='*70}")
            
            try:
                data = await self._scrape_portal(portal_name)
                if data:
                    self.all_data.extend(data)
                    logger.info(f"✓ {portal_name}: {len(data)} propiedades scrapeadas")
                else:
                    logger.warning(f"⚠ {portal_name}: No se obtuvieron datos")
                    
            except Exception as e:
                logger.error(f"✗ Error scrapeando {portal_name}: {e}")
        
        # Resumen final
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("SCRAPING COMPLETADO")
        logger.info("="*70)
        logger.info(f"Total de propiedades: {len(self.all_data)}")
        logger.info(f"Duración: {duration:.1f} segundos ({duration/60:.1f} minutos)")
        logger.info("="*70 + "\n")
        
        return self.all_data
    
    async def _scrape_portal(self, portal_name: str) -> list:
        """
        Scrapea un portal específico.
        
        Args:
            portal_name: Nombre del portal ('idealista', 'fotocasa', 'airbnb')
            
        Returns:
            list: Datos scrapeados
        """
        portal_name = portal_name.lower()
        
        if portal_name == 'idealista':
            scraper = IdealistaScraper(headless=self.headless)
            return await scraper.scrape(max_pages=self.max_pages)
            
        elif portal_name == 'fotocasa':
            scraper = FotocasaScraper(headless=self.headless)
            return await scraper.scrape(max_pages=self.max_pages)
            
        elif portal_name == 'airbnb':
            scraper = AirbnbScraper(headless=self.headless)
            return await scraper.scrape(max_results=MAX_RESULTS_AIRBNB)
            
        else:
            logger.error(f"Portal desconocido: {portal_name}")
            return []
    
    def process_data(self):
        """Procesa y limpia los datos scrapeados."""
        if not self.all_data:
            logger.warning("No hay datos para procesar")
            return []
        
        logger.info("\n" + "="*70)
        logger.info("PROCESANDO DATOS")
        logger.info("="*70)
        
        processor = DataProcessor(self.all_data)
        clean_data = processor.clean_all()
        processor.print_statistics()
        
        return clean_data
    
    def export_data(self, data, format: str = 'all'):
        """
        Exporta los datos al formato especificado.
        
        Args:
            data: Datos a exportar
            format: Formato ('csv', 'excel', 'json', 'all')
        """
        if not data:
            logger.warning("No hay datos para exportar")
            return
        
        logger.info("\n" + "="*70)
        logger.info("EXPORTANDO DATOS")
        logger.info("="*70)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exporter = DataExporter(data)
        
        if format == 'all':
            results = exporter.to_all_formats(
                base_filename=f'cantabria_alquileres_{timestamp}',
                processed=True
            )
            
            logger.info("\n📁 Archivos generados:")
            for fmt, path in results.items():
                logger.info(f"   • {fmt.upper()}: {path}")
            
            # Crear informe resumen
            logger.info("\n📊 Generando informe resumen...")
            report_path = exporter.create_summary_report()
            logger.info(f"   • RESUMEN: {report_path}")
            
        elif format == 'csv':
            path = exporter.to_csv(
                filename=f'cantabria_alquileres_{timestamp}.csv',
                processed=True
            )
            logger.info(f"   • CSV: {path}")
            
        elif format == 'excel':
            path = exporter.to_excel(
                filename=f'cantabria_alquileres_{timestamp}.xlsx',
                processed=True
            )
            logger.info(f"   • EXCEL: {path}")
            
        elif format == 'json':
            path = exporter.to_json(
                filename=f'cantabria_alquileres_{timestamp}.json',
                processed=True
            )
            logger.info(f"   • JSON: {path}")
        
        logger.info("="*70 + "\n")


async def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Webscraper de portales inmobiliarios para Cantabria',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py
  python main.py --portals idealista fotocasa
  python main.py --pages 20 --format excel
  python main.py --headless --format csv
        """
    )
    
    parser.add_argument(
        '--portals',
        nargs='+',
        choices=['idealista', 'fotocasa', 'airbnb'],
        default=['idealista', 'fotocasa', 'airbnb'],
        help='Portales a scrapear (por defecto: todos)'
    )
    
    parser.add_argument(
        '--pages',
        type=int,
        default=MAX_PAGES_DEFAULT,
        help=f'Número de páginas a scrapear por portal (por defecto: {MAX_PAGES_DEFAULT})'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'excel', 'json', 'all'],
        default='all',
        help='Formato de exportación (por defecto: all)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Ejecutar en modo headless (sin interfaz gráfica)'
    )
    
    parser.add_argument(
        '--no-process',
        action='store_true',
        help='No procesar/limpiar datos (exportar raw)'
    )
    
    args = parser.parse_args()
    
    # Crear scraper principal
    scraper = RealEstateScraper(
        portals=args.portals,
        max_pages=args.pages,
        headless=args.headless
    )
    
    # Ejecutar scraping
    raw_data = await scraper.scrape_all()
    
    if not raw_data:
        logger.error("No se obtuvieron datos. Finalizando.")
        return
    
    # Procesar datos (si no se especifica --no-process)
    if args.no_process:
        final_data = raw_data
        logger.info("Datos exportados sin procesamiento (raw)")
    else:
        final_data = scraper.process_data()
    
    # Exportar datos
    scraper.export_data(final_data, format=args.format)
    
    logger.info("✓ Proceso completado exitosamente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Error fatal: {e}", exc_info=True)
        sys.exit(1)