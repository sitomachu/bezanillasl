# BezanillaSL: Real Estate Analytics & Business Feasibility

Este repositorio contiene el desarrollo técnico y estratégico derivado del estudio conjunto de dos Trabajos de Fin de Máster (TFM) para los programas de **MBA Tech** y **Master en Business Analytics**. El proyecto integra el análisis de viabilidad económica de un desarrollo inmobiliario en Cantabria, España, con modelos de analítica de datos para la predicción de precios de suelo y rentas.

## 1. Visión del Proyecto
El objetivo central es validar la factibilidad de una empresa patrimonial familiar orientada al segmento de vivienda asequible (**Affordable Housing**). Este ecosistema analítico busca sustituir la intuición tradicional del sector inmobiliario por un sistema de apoyo a las decisiones basado en evidencia cuantitativa y modelado predictivo, unificando la visión de negocio (MBA) con la precisión técnica (Analytics).

## 2. Equipo y Colaboradores
**Alejandro (Project Owner & Technical Lead)**: Responsable de la arquitectura de datos, desarrollo del código fuente, procesamiento de datasets y liderazgo del modelado predictivo (Analytics).
**Pablo (Technical Collaborator & Theoretical Lead)**: Responsable de la fundamentación teórica, elaboración del plan de negocio, proyecciones financieras de viabilidad y análisis estratégico de mercado.

## 3. Estructura del Repositorio
La arquitectura del proyecto organiza la información desde la ingesta de fuentes oficiales hasta el modelado predictivo:

```text
├── data/
│   ├── raw/                 # Datos originales sin procesar.
│   └── mivau/               # Datasets del Ministerio de Vivienda y Agenda Urbana.
│       ├── datos_alquiler/  # Sistema de Referencia del Precio del Alquiler (SERPAVI).
│       ├── datos_suelo/     # Estadísticas de precios de suelo urbano.
│       └── datos_vivienda/  # Estimación del parque de viviendas.
├── models/                  # Almacenamiento de modelos entrenados (p.ej. predicción de precios).
├── notebooks/               # Experimentación y Análisis Exploratorio de Datos (EDA).
├── src/                     # Código fuente del proyecto (scripts de limpieza y modelado).
│   ├── idealistaAPI/        # Módulo de ingesta y procesamiento vía API Idealista.
│   └── geospatial_expansion/ # Módulo de distancias a POIs (playa, colegio, supermercado, etc.).
├── requirements.txt         # Listado de dependencias y versiones.
└── README.md                # Documentación principal del proyecto.
```

## 4. Módulo Idealista API
El repositorio incluye un módulo específico para descargar y preparar datos de Idealista, ubicado en `src/idealistaAPI`.

Flujos principales:
1. Descarga de datos de venta o alquiler con autenticación OAuth.
2. Almacenamiento de respuestas crudas en `data/raw/idealista/...`.
3. Limpieza y conversión a CSV en `data/processed/idealista/...`.

Guías y uso:
- Documentación del módulo: `src/idealistaAPI/README.md`
- Guía operativa: `src/idealistaAPI/idealista_API_userguide.md`

## 5. Módulo de Expansión Geoespacial
El repositorio incluye `src/geospatial_expansion` para enriquecer datasets de venta/alquiler con distancia mínima al punto de interés más cercano por categoría (playa, supermercado, colegio, etc.) usando OpenStreetMap (`osmnx`).

Entradas:
- CSV objetivo con coordenadas (`latitude`/`longitude` o columnas equivalentes).

Salida:
- DataFrame enriquecido con columnas como `distancia_min_playa_km`.

Paso 1: descargar POIs (config en `run_descargar_pois.py`)
```bash
python -m src.geospatial_expansion.run_descargar_pois
```
La descarga usa circulos geograficos fijos definidos en `DEFAULT_CIRCLES`.

Paso 2: expandir dataset desde notebook/Python:
```python
from src.geospatial_expansion import agregar_distancias_minimas_poi
df_out = agregar_distancias_minimas_poi(df, ["playa", "supermercado"])
```
