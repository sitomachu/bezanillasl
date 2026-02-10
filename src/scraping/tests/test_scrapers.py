"""
Tests unitarios para los scrapers
BezanillaSL - Real Estate Analytics
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Añadir el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers import IdealistaScraper, FotocasaScraper, AirbnbScraper
from utils import DataProcessor, DataExporter


class TestScrapers:
    """Tests para los scrapers"""
    
    @pytest.mark.asyncio
    async def test_idealista_scraper_init(self):
        """Test de inicialización de IdealistaScraper"""
        scraper = IdealistaScraper(headless=True)
        assert scraper.portal_name == "Idealista"
        assert scraper.headless == True
        assert len(scraper.data) == 0
    
    @pytest.mark.asyncio
    async def test_fotocasa_scraper_init(self):
        """Test de inicialización de FotocasaScraper"""
        scraper = FotocasaScraper(headless=True)
        assert scraper.portal_name == "Fotocasa"
        assert scraper.headless == True
        assert len(scraper.data) == 0
    
    @pytest.mark.asyncio
    async def test_airbnb_scraper_init(self):
        """Test de inicialización de AirbnbScraper"""
        scraper = AirbnbScraper(headless=True)
        assert scraper.portal_name == "Airbnb"
        assert scraper.headless == True
        assert len(scraper.data) == 0
    
    def test_scraper_methods(self):
        """Test de métodos básicos del scraper"""
        scraper = IdealistaScraper()
        
        # Test timestamp
        timestamp = scraper.get_timestamp()
        assert timestamp is not None
        assert len(timestamp) > 0
        
        # Test add_data
        scraper.add_data({'test': 'data'})
        assert len(scraper.get_data()) == 1
        
        # Test clear_data
        scraper.clear_data()
        assert len(scraper.get_data()) == 0


class TestDataProcessor:
    """Tests para el procesador de datos"""
    
    def test_processor_init(self):
        """Test de inicialización del procesador"""
        test_data = [
            {'precio_numerico': 850, 'habitaciones': 3},
            {'precio_numerico': 750, 'habitaciones': 2},
        ]
        processor = DataProcessor(test_data)
        assert len(processor.raw_data) == 2
    
    def test_remove_duplicates(self):
        """Test de eliminación de duplicados"""
        test_data = [
            {'portal': 'Idealista', 'url': 'http://test1.com', 'precio_numerico': 850},
            {'portal': 'Idealista', 'url': 'http://test1.com', 'precio_numerico': 850},  # Duplicado
            {'portal': 'Fotocasa', 'url': 'http://test2.com', 'precio_numerico': 750},
        ]
        processor = DataProcessor(test_data)
        clean_data = processor.clean_all()
        assert len(clean_data) == 2  # Debe haber eliminado 1 duplicado
    
    def test_price_validation(self):
        """Test de validación de precios"""
        test_data = [
            {'precio_numerico': 850, 'portal': 'Idealista'},
            {'precio_numerico': 50, 'portal': 'Idealista'},  # Precio inválido (muy bajo)
            {'precio_numerico': 15000, 'portal': 'Idealista'},  # Precio inválido (muy alto)
        ]
        processor = DataProcessor(test_data)
        clean_data = processor.clean_all()
        # Solo debe quedar el primer registro con precio válido
        assert all(100 <= item['precio_numerico'] <= 10000 for item in clean_data if 'precio_numerico' in item)
    
    def test_statistics_calculation(self):
        """Test de cálculo de estadísticas"""
        test_data = [
            {'portal': 'Idealista', 'precio_numerico': 850, 'habitaciones': 3, 'superficie_m2': 90, 'municipio': 'Santander'},
            {'portal': 'Fotocasa', 'precio_numerico': 750, 'habitaciones': 2, 'superficie_m2': 75, 'municipio': 'Torrelavega'},
        ]
        processor = DataProcessor(test_data)
        processor.clean_all()
        stats = processor.get_statistics()
        
        assert stats['total_registros'] == 2
        assert 'precio' in stats
        assert 'portales' in stats
        assert stats['precio']['media'] == 800.0


class TestDataExporter:
    """Tests para el exportador de datos"""
    
    def test_exporter_init(self):
        """Test de inicialización del exportador"""
        test_data = [
            {'precio_numerico': 850, 'habitaciones': 3},
        ]
        exporter = DataExporter(test_data)
        assert len(exporter.data) == 1
        assert exporter.output_dir.exists()
    
    def test_csv_export(self, tmp_path):
        """Test de exportación a CSV"""
        test_data = [
            {'portal': 'Idealista', 'precio_numerico': 850, 'habitaciones': 3},
        ]
        exporter = DataExporter(test_data, output_dir=tmp_path)
        
        csv_path = exporter.to_csv('test.csv')
        assert csv_path.exists()
        assert csv_path.suffix == '.csv'
    
    def test_json_export(self, tmp_path):
        """Test de exportación a JSON"""
        test_data = [
            {'portal': 'Idealista', 'precio_numerico': 850, 'habitaciones': 3},
        ]
        exporter = DataExporter(test_data, output_dir=tmp_path)
        
        json_path = exporter.to_json('test.json')
        assert json_path.exists()
        assert json_path.suffix == '.json'
    
    def test_excel_export(self, tmp_path):
        """Test de exportación a Excel"""
        test_data = [
            {'portal': 'Idealista', 'precio_numerico': 850, 'habitaciones': 3},
        ]
        exporter = DataExporter(test_data, output_dir=tmp_path)
        
        excel_path = exporter.to_excel('test.xlsx')
        assert excel_path.exists()
        assert excel_path.suffix == '.xlsx'


# Fixtures para pytest
@pytest.fixture
def sample_property_data():
    """Fixture con datos de ejemplo de una propiedad"""
    return {
        'portal': 'Idealista',
        'titulo': 'Piso en alquiler en Centro',
        'precio': '850 €/mes',
        'precio_numerico': 850,
        'habitaciones': 3,
        'superficie_m2': 90,
        'banos': 2,
        'ubicacion': 'Centro, Santander, Cantabria',
        'municipio': 'Santander',
        'provincia': 'Cantabria',
        'url': 'http://idealista.com/test',
        'tipo_propiedad': 'Piso',
    }


@pytest.fixture
def sample_dataset():
    """Fixture con un dataset de ejemplo"""
    return [
        {
            'portal': 'Idealista',
            'precio_numerico': 850,
            'habitaciones': 3,
            'superficie_m2': 90,
            'municipio': 'Santander',
        },
        {
            'portal': 'Fotocasa',
            'precio_numerico': 750,
            'habitaciones': 2,
            'superficie_m2': 75,
            'municipio': 'Torrelavega',
        },
        {
            'portal': 'Airbnb',
            'precio_numerico': 65,
            'habitaciones': 1,
            'superficie_m2': 45,
            'municipio': 'Santander',
        },
    ]


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, '-v'])