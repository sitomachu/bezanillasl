"""
Tests unitarios (sin scraping en vivo)
Ubicación real: src/scraping/tests/
"""

from pathlib import Path
import sys
import pytest

# Añade src/scraping al path para importar scrapers y utils
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrapers import IdealistaScraper, FotocasaScraper, AirbnbScraper
from utils import DataProcessor, DataExporter


class TestScrapers:
    @pytest.mark.asyncio
    async def test_init(self):
        s1 = IdealistaScraper(headless=True)
        s2 = FotocasaScraper(headless=True)
        s3 = AirbnbScraper(headless=True)
        assert s1.portal_name == "Idealista"
        assert s2.portal_name == "Fotocasa"
        assert s3.portal_name == "Airbnb"

    def test_timestamp(self):
        s = IdealistaScraper()
        assert isinstance(s.timestamp_iso(), str)


class TestDataProcessor:
    def test_remove_duplicates(self):
        data = [
            {"portal": "Idealista", "url": "a", "precio_numerico": 800, "unidad_precio": "mes"},
            {"portal": "Idealista", "url": "a", "precio_numerico": 800, "unidad_precio": "mes"},
            {"portal": "Fotocasa", "url": "b", "precio_numerico": 700, "unidad_precio": "mes"},
        ]
        clean = DataProcessor(data).clean_all()
        assert len(clean) == 2

    def test_monthly_equivalent_validation(self):
        data = [
            {"portal": "Idealista", "precio_numerico": 850, "unidad_precio": "mes"},
            {"portal": "Airbnb", "precio_numerico": 60, "unidad_precio": "noche"},  # equiv 1800/mes
            {"portal": "Airbnb", "precio_numerico": 2, "unidad_precio": "noche"},   # equiv 60/mes => cae
        ]
        clean = DataProcessor(data).clean_all()
        assert any(d["portal"] == "Idealista" for d in clean)
        assert any(d["portal"] == "Airbnb" for d in clean)
        for d in clean:
            p = d.get("precio_mensual_equivalente")
            if p is not None:
                assert 100 <= float(p) <= 10000


class TestDataExporter:
    def test_exports(self, tmp_path):
        data = [{"portal": "Idealista", "precio_numerico": 850, "unidad_precio": "mes"}]
        ex = DataExporter(data, output_dir=tmp_path)
        assert ex.to_csv("a.csv").exists()
        assert ex.to_json("a.json").exists()
        assert ex.to_excel("a.xlsx").exists()
