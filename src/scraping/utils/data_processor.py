"""
Utilidades para limpieza y procesamiento de datos scrapeados
BezanillaSL - Real Estate Analytics
"""

import logging
import re
from typing import List, Dict, Optional
import pandas as pd

from config.settings import VALIDATION_RULES, INVALID_KEYWORDS

logger = logging.getLogger('scraping.data_processor')


class DataProcessor:
    """Procesa y limpia datos scrapeados de portales inmobiliarios"""
    
    def __init__(self, data: List[Dict]):
        """
        Inicializa el procesador con datos.
        
        Args:
            data: Lista de diccionarios con datos scrapeados
        """
        self.raw_data = data
        self.processed_data = []
        logger.info(f"Procesador inicializado con {len(data)} registros")
    
    def clean_all(self) -> List[Dict]:
        """
        Ejecuta todos los pasos de limpieza.
        
        Returns:
            List[Dict]: Datos limpios
        """
        logger.info("Iniciando proceso de limpieza...")
        
        # Convertir a DataFrame para facilitar el procesamiento
        df = pd.DataFrame(self.raw_data)
        
        initial_count = len(df)
        logger.info(f"Registros iniciales: {initial_count}")
        
        # 1. Eliminar duplicados
        df = self._remove_duplicates(df)
        
        # 2. Limpiar campos de texto
        df = self._clean_text_fields(df)
        
        # 3. Normalizar precios
        df = self._normalize_prices(df)
        
        # 4. Validar datos
        df = self._validate_data(df)
        
        # 5. Filtrar anuncios inválidos
        df = self._filter_invalid_listings(df)
        
        # 6. Enriquecer datos
        df = self._enrich_data(df)
        
        # 7. Ordenar campos
        df = self._reorder_columns(df)
        
        final_count = len(df)
        removed = initial_count - final_count
        logger.info(
            f"Limpieza completada: {final_count} registros válidos "
            f"({removed} eliminados, {removed/initial_count*100:.1f}%)"
        )
        
        self.processed_data = df.to_dict('records')
        return self.processed_data
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros duplicados.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame sin duplicados
        """
        initial_count = len(df)
        
        # Eliminar duplicados exactos
        df = df.drop_duplicates()
        
        # Eliminar duplicados por URL (mismo anuncio)
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'], keep='first')
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Eliminados {removed} registros duplicados")
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia campos de texto.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame con textos limpios
        """
        text_fields = ['titulo', 'ubicacion', 'descripcion', 'municipio']
        
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].apply(self._clean_text)
        
        logger.debug("Campos de texto limpiados")
        return df
    
    @staticmethod
    def _clean_text(text: Optional[str]) -> Optional[str]:
        """
        Limpia un campo de texto individual.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio o None
        """
        if pd.isna(text) or not text:
            return None
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', str(text))
        
        # Eliminar saltos de línea
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Strip
        text = text.strip()
        
        return text if text else None
    
    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza y valida precios.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame con precios normalizados
        """
        if 'precio_numerico' in df.columns:
            # Convertir a float
            df['precio_numerico'] = pd.to_numeric(
                df['precio_numerico'],
                errors='coerce'
            )
            
            # Para Airbnb, estimar precio mensual (precio_noche * 30)
            mask_airbnb = df['portal'] == 'Airbnb'
            if mask_airbnb.any():
                df.loc[mask_airbnb, 'precio_mensual_estimado'] = (
                    df.loc[mask_airbnb, 'precio_numerico'] * 30
                )
                logger.info("Precios mensuales estimados para Airbnb")
        
        logger.debug("Precios normalizados")
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida datos según reglas configuradas.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame con datos válidos
        """
        initial_count = len(df)
        
        # Validar precio
        if 'precio_numerico' in df.columns:
            mask = (
                (df['precio_numerico'] >= VALIDATION_RULES['precio_min']) &
                (df['precio_numerico'] <= VALIDATION_RULES['precio_max'])
            ) | df['precio_numerico'].isna()
            df = df[mask]
        
        # Validar superficie
        if 'superficie_m2' in df.columns:
            mask = (
                (df['superficie_m2'] >= VALIDATION_RULES['superficie_min']) &
                (df['superficie_m2'] <= VALIDATION_RULES['superficie_max'])
            ) | df['superficie_m2'].isna()
            df = df[mask]
        
        # Validar habitaciones
        if 'habitaciones' in df.columns:
            mask = (
                (df['habitaciones'] >= VALIDATION_RULES['habitaciones_min']) &
                (df['habitaciones'] <= VALIDATION_RULES['habitaciones_max'])
            ) | df['habitaciones'].isna()
            df = df[mask]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Eliminados {removed} registros con datos inválidos")
        
        return df
    
    def _filter_invalid_listings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra anuncios que no son viviendas (garajes, locales, etc.).
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame filtrado
        """
        initial_count = len(df)
        
        # Buscar keywords inválidas en título y descripción
        mask = pd.Series([True] * len(df), index=df.index)
        
        for field in ['titulo', 'descripcion']:
            if field in df.columns:
                for keyword in INVALID_KEYWORDS:
                    field_mask = ~df[field].str.contains(
                        keyword,
                        case=False,
                        na=False
                    )
                    mask = mask & field_mask
        
        df = df[mask]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Eliminados {removed} anuncios no válidos (garajes, locales, etc.)")
        
        return df
    
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece los datos con campos calculados.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame enriquecido
        """
        # Precio por m²
        if 'precio_numerico' in df.columns and 'superficie_m2' in df.columns:
            df['precio_por_m2'] = (
                df['precio_numerico'] / df['superficie_m2']
            ).round(2)
        
        # Indicador de dataset completo
        required_fields = ['precio_numerico', 'habitaciones', 'superficie_m2']
        df['datos_completos'] = df[required_fields].notna().all(axis=1)
        
        # Categoría de precio
        if 'precio_numerico' in df.columns:
            df['categoria_precio'] = pd.cut(
                df['precio_numerico'],
                bins=[0, 500, 750, 1000, 1500, float('inf')],
                labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']
            )
        
        logger.debug("Datos enriquecidos con campos calculados")
        return df
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reordena columnas en un orden lógico.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame con columnas reordenadas
        """
        # Orden preferido de columnas
        preferred_order = [
            'portal',
            'fecha_scraping',
            'titulo',
            'tipo_propiedad',
            'precio',
            'precio_numerico',
            'precio_mensual_estimado',
            'precio_por_m2',
            'categoria_precio',
            'habitaciones',
            'superficie_m2',
            'banos',
            'ubicacion',
            'municipio',
            'provincia',
            'descripcion',
            'url',
            'datos_completos',
        ]
        
        # Mantener solo columnas que existen
        columns_order = [col for col in preferred_order if col in df.columns]
        
        # Añadir columnas restantes
        remaining_cols = [col for col in df.columns if col not in columns_order]
        columns_order.extend(remaining_cols)
        
        df = df[columns_order]
        
        logger.debug("Columnas reordenadas")
        return df
    
    def get_statistics(self) -> Dict:
        """
        Calcula estadísticas de los datos procesados.
        
        Returns:
            Dict: Estadísticas de los datos
        """
        if not self.processed_data:
            logger.warning("No hay datos procesados para calcular estadísticas")
            return {}
        
        df = pd.DataFrame(self.processed_data)
        
        stats = {
            'total_registros': len(df),
            'portales': df['portal'].value_counts().to_dict() if 'portal' in df.columns else {},
            'municipios_top': df['municipio'].value_counts().head(10).to_dict() if 'municipio' in df.columns else {},
        }
        
        # Estadísticas de precio
        if 'precio_numerico' in df.columns:
            precios = df['precio_numerico'].dropna()
            stats['precio'] = {
                'media': round(precios.mean(), 2),
                'mediana': round(precios.median(), 2),
                'min': round(precios.min(), 2),
                'max': round(precios.max(), 2),
                'desviacion_std': round(precios.std(), 2),
            }
        
        # Estadísticas de superficie
        if 'superficie_m2' in df.columns:
            superficies = df['superficie_m2'].dropna()
            stats['superficie'] = {
                'media': round(superficies.mean(), 2),
                'mediana': round(superficies.median(), 2),
                'min': round(superficies.min(), 2),
                'max': round(superficies.max(), 2),
            }
        
        # Completitud de datos
        stats['completitud'] = {
            'con_precio': int(df['precio_numerico'].notna().sum()),
            'con_habitaciones': int(df['habitaciones'].notna().sum()) if 'habitaciones' in df.columns else 0,
            'con_superficie': int(df['superficie_m2'].notna().sum()) if 'superficie_m2' in df.columns else 0,
            'datos_completos': int(df['datos_completos'].sum()) if 'datos_completos' in df.columns else 0,
        }
        
        return stats
    
    def print_statistics(self):
        """Imprime estadísticas de forma legible."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("ESTADÍSTICAS DEL DATASET")
        print("="*70)
        
        print(f"\n📊 Total de registros: {stats.get('total_registros', 0)}")
        
        if 'portales' in stats:
            print("\n📁 Distribución por portal:")
            for portal, count in stats['portales'].items():
                print(f"   • {portal}: {count}")
        
        if 'precio' in stats:
            print("\n💰 Estadísticas de precio:")
            for key, value in stats['precio'].items():
                print(f"   • {key.capitalize()}: {value} €/mes")
        
        if 'superficie' in stats:
            print("\n📐 Estadísticas de superficie:")
            for key, value in stats['superficie'].items():
                print(f"   • {key.capitalize()}: {value} m²")
        
        if 'completitud' in stats:
            print("\n✅ Completitud de datos:")
            total = stats['total_registros']
            for key, value in stats['completitud'].items():
                pct = (value / total * 100) if total > 0 else 0
                print(f"   • {key.replace('_', ' ').capitalize()}: {value} ({pct:.1f}%)")
        
        if 'municipios_top' in stats and stats['municipios_top']:
            print("\n🏘️  Top 10 municipios:")
            for i, (municipio, count) in enumerate(stats['municipios_top'].items(), 1):
                print(f"   {i}. {municipio}: {count}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test del procesador
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Datos de prueba
    test_data = [
        {
            'portal': 'Idealista',
            'precio': '850 €/mes',
            'precio_numerico': 850,
            'habitaciones': 3,
            'superficie_m2': 90,
            'ubicacion': 'Centro, Santander, Cantabria',
            'titulo': 'Piso en alquiler',
        },
        {
            'portal': 'Idealista',
            'precio': '850 €/mes',
            'precio_numerico': 850,
            'habitaciones': 3,
            'superficie_m2': 90,
            'ubicacion': 'Centro, Santander, Cantabria',
            'titulo': 'Piso en alquiler',
        },  # Duplicado
    ]
    
    processor = DataProcessor(test_data)
    clean_data = processor.clean_all()
    processor.print_statistics()