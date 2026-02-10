"""
Utilidades para exportar datos a diferentes formatos
BezanillaSL - Real Estate Analytics
"""

import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd

from config.settings import (
    EXPORT_CONFIG,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger('scraping.exporter')


class DataExporter:
    """Exporta datos scrapeados a diferentes formatos"""
    
    def __init__(self, data: List[Dict], output_dir: Optional[Path] = None):
        """
        Inicializa el exportador.
        
        Args:
            data: Lista de diccionarios con los datos
            output_dir: Directorio de salida (por defecto RAW_DATA_DIR)
        """
        self.data = data
        self.output_dir = Path(output_dir) if output_dir else RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exportador inicializado con {len(data)} registros")
    
    def to_csv(
        self,
        filename: Optional[str] = None,
        processed: bool = False
    ) -> Path:
        """
        Exporta datos a formato CSV.
        
        Args:
            filename: Nombre del archivo (auto-generado si es None)
            processed: Si True, guarda en PROCESSED_DATA_DIR
            
        Returns:
            Path: Ruta del archivo creado
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'real_estate_data_{timestamp}.csv'
        
        output_dir = PROCESSED_DATA_DIR if processed else self.output_dir
        filepath = output_dir / filename
        
        try:
            df = pd.DataFrame(self.data)
            df.to_csv(
                filepath,
                index=False,
                sep=EXPORT_CONFIG['csv_separator'],
                encoding=EXPORT_CONFIG['csv_encoding']
            )
            logger.info(f"✓ Datos exportados a CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando a CSV: {e}")
            raise
    
    def to_excel(
        self,
        filename: Optional[str] = None,
        processed: bool = False,
        sheet_name: str = 'Datos'
    ) -> Path:
        """
        Exporta datos a formato Excel.
        
        Args:
            filename: Nombre del archivo (auto-generado si es None)
            processed: Si True, guarda en PROCESSED_DATA_DIR
            sheet_name: Nombre de la hoja de cálculo
            
        Returns:
            Path: Ruta del archivo creado
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'real_estate_data_{timestamp}.xlsx'
        
        output_dir = PROCESSED_DATA_DIR if processed else self.output_dir
        filepath = output_dir / filename
        
        try:
            df = pd.DataFrame(self.data)
            
            # Crear archivo Excel con formato
            with pd.ExcelWriter(
                filepath,
                engine=EXPORT_CONFIG['excel_engine']
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Ajustar ancho de columnas
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    )
                    worksheet.column_dimensions[
                        chr(65 + idx)
                    ].width = min(max_len + 2, 50)
            
            logger.info(f"✓ Datos exportados a Excel: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando a Excel: {e}")
            raise
    
    def to_json(
        self,
        filename: Optional[str] = None,
        processed: bool = False,
        pretty: bool = True
    ) -> Path:
        """
        Exporta datos a formato JSON.
        
        Args:
            filename: Nombre del archivo (auto-generado si es None)
            processed: Si True, guarda en PROCESSED_DATA_DIR
            pretty: Si True, formatea el JSON con indentación
            
        Returns:
            Path: Ruta del archivo creado
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'real_estate_data_{timestamp}.json'
        
        output_dir = PROCESSED_DATA_DIR if processed else self.output_dir
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(
                        self.data,
                        f,
                        ensure_ascii=False,
                        indent=EXPORT_CONFIG['json_indent']
                    )
                else:
                    json.dump(self.data, f, ensure_ascii=False)
            
            logger.info(f"✓ Datos exportados a JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando a JSON: {e}")
            raise
    
    def to_all_formats(
        self,
        base_filename: Optional[str] = None,
        processed: bool = False
    ) -> Dict[str, Path]:
        """
        Exporta a todos los formatos disponibles.
        
        Args:
            base_filename: Nombre base del archivo (sin extensión)
            processed: Si True, guarda en PROCESSED_DATA_DIR
            
        Returns:
            Dict[str, Path]: Diccionario con formato -> ruta
        """
        if not base_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f'real_estate_data_{timestamp}'
        
        results = {}
        
        # CSV
        try:
            csv_path = self.to_csv(f'{base_filename}.csv', processed)
            results['csv'] = csv_path
        except Exception as e:
            logger.error(f"Fallo exportación CSV: {e}")
        
        # Excel
        try:
            excel_path = self.to_excel(f'{base_filename}.xlsx', processed)
            results['excel'] = excel_path
        except Exception as e:
            logger.error(f"Fallo exportación Excel: {e}")
        
        # JSON
        try:
            json_path = self.to_json(f'{base_filename}.json', processed)
            results['json'] = json_path
        except Exception as e:
            logger.error(f"Fallo exportación JSON: {e}")
        
        logger.info(f"✓ Exportación completa: {len(results)} formatos")
        return results
    
    def export_by_portal(
        self,
        format: str = 'csv',
        processed: bool = False
    ) -> Dict[str, Path]:
        """
        Exporta datos separados por portal.
        
        Args:
            format: Formato de exportación ('csv', 'excel', 'json')
            processed: Si True, guarda en PROCESSED_DATA_DIR
            
        Returns:
            Dict[str, Path]: Diccionario con portal -> ruta
        """
        if not self.data:
            logger.warning("No hay datos para exportar")
            return {}
        
        df = pd.DataFrame(self.data)
        
        if 'portal' not in df.columns:
            logger.error("Los datos no tienen columna 'portal'")
            return {}
        
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for portal in df['portal'].unique():
            portal_data = df[df['portal'] == portal].to_dict('records')
            portal_exporter = DataExporter(portal_data, self.output_dir)
            
            filename = f'{portal.lower()}_{timestamp}'
            
            try:
                if format == 'csv':
                    path = portal_exporter.to_csv(f'{filename}.csv', processed)
                elif format == 'excel':
                    path = portal_exporter.to_excel(f'{filename}.xlsx', processed)
                elif format == 'json':
                    path = portal_exporter.to_json(f'{filename}.json', processed)
                else:
                    logger.error(f"Formato desconocido: {format}")
                    continue
                
                results[portal] = path
                logger.info(f"✓ {portal}: {len(portal_data)} registros exportados")
                
            except Exception as e:
                logger.error(f"Error exportando datos de {portal}: {e}")
        
        return results
    
    def create_summary_report(
        self,
        filename: Optional[str] = None
    ) -> Path:
        """
        Crea un informe resumen en Excel con múltiples hojas.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Path: Ruta del archivo creado
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'resumen_scraping_{timestamp}.xlsx'
        
        filepath = PROCESSED_DATA_DIR / filename
        
        try:
            df = pd.DataFrame(self.data)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Hoja 1: Todos los datos
                df.to_excel(writer, sheet_name='Datos Completos', index=False)
                
                # Hoja 2: Resumen por portal
                if 'portal' in df.columns:
                    summary_portal = df.groupby('portal').agg({
                        'precio_numerico': ['count', 'mean', 'median', 'min', 'max'],
                        'superficie_m2': ['mean', 'median'],
                        'habitaciones': ['mean']
                    }).round(2)
                    summary_portal.to_excel(writer, sheet_name='Resumen por Portal')
                
                # Hoja 3: Resumen por municipio
                if 'municipio' in df.columns and 'precio_numerico' in df.columns:
                    summary_municipio = df.groupby('municipio').agg({
                        'precio_numerico': ['count', 'mean', 'median'],
                        'superficie_m2': 'mean'
                    }).round(2).sort_values(('precio_numerico', 'count'), ascending=False)
                    summary_municipio.head(20).to_excel(writer, sheet_name='Top 20 Municipios')
                
                # Hoja 4: Distribución de precios
                if 'categoria_precio' in df.columns:
                    dist_precio = df['categoria_precio'].value_counts().to_frame()
                    dist_precio.to_excel(writer, sheet_name='Distribución Precios')
            
            logger.info(f"✓ Informe resumen creado: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creando informe resumen: {e}")
            raise


if __name__ == "__main__":
    # Test del exportador
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Datos de prueba
    test_data = [
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
    ]
    
    exporter = DataExporter(test_data)
    results = exporter.to_all_formats('test_export')
    
    print("\nArchivos creados:")
    for format, path in results.items():
        print(f"  • {format.upper()}: {path}")