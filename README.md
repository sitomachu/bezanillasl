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
BezanillaSL/
├── data/
│   ├── raw/                  # Datos originales sin procesar
│   ├── mivau/                # Datasets del Ministerio de Vivienda y Agenda Urbana
│   │   ├── datos_alquiler/   # Sistema de Referencia del Precio del Alquiler (SERPAVI)
│   │   ├── datos_suelo/      # Estadísticas de precios de suelo urbano
│   │   └── datos_vivienda/   # Estimación del parque de viviendas
│   ├── scraped/              # Datos extraídos por webscraping
│   │   ├── raw/              # Datos sin procesar de portales
│   │   └── processed/        # Datos limpios y normalizados
│   └── combined/             # Datasets unificados (MIVAU + Scraping)
├── models/                   # Modelos entrenados (predicción de precios)
├── notebooks/                # Experimentación y Análisis Exploratorio de Datos (EDA)
├── src/
│   ├── scraping/             # Sistema de webscraping
│   │   ├── config/           # Configuración centralizada
│   │   │   ├── __init__.py
│   │   │   └── settings.py
│   │   ├── scrapers/         # Scrapers por portal
│   │   │   ├── __init__.py
│   │   │   ├── base_scraper.py
│   │   │   ├── idealista.py
│   │   │   ├── fotocasa.py
│   │   │   └── airbnb.py
│   │   ├── utils/            # Utilidades de procesamiento
│   │   │   ├── __init__.py
│   │   │   ├── data_processor.py
│   │   │   └── exporter.py
│   │   ├── tests/            # Tests unitarios
│   │   │   └── test_scrapers.py
│   │   └── main.py           # Script principal de scraping
│   ├── preprocessing/        # Limpieza y transformación de datos
│   ├── features/             # Ingeniería de características
│   └── modeling/             # Desarrollo de modelos predictivos
├── requirements.txt          # Dependencias del proyecto completo
├── scraping_requirements.txt # Dependencias específicas de scraping
└── README.md                 # Documentación principal

## 4. Estrategia de Ingesta y Datos
El sistema utiliza una arquitectura de datos híbrida para garantizar tanto la profundidad histórica como la relevancia actual del mercado:

## Pipeline de datos
┌─────────────────────┐
│  Fuentes Oficiales  │
│     (MIVAU)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Webscraping        │────▶│  data/scraped/raw   │
│  (Idealista, etc)   │     └──────────┬──────────┘
└─────────────────────┘                │
                                       ▼
                            ┌─────────────────────┐
                            │  Limpieza y         │
                            │  Normalización      │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  data/combined      │
                            │  Dataset Unificado  │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Feature            │
                            │  Engineering        │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Modelado           │
                            │  Predictivo         │
                            └─────────────────────┘

* **Indicadores Oficiales**: Integración de fuentes consolidadas como el Sistema de Referencia del Precio del Alquiler (**SERPAVI**), estadísticas de suelo urbano y censos de vivienda del **MIVAU** e **INE**.
* **Dinámicas de Mercado**: Ingesta masiva de datos provenientes de plataformas inmobiliarias para capturar la oferta activa, variables físicas de los inmuebles y tendencias de precios en tiempo real dentro de la región de Cantabria.

## 5. Pipeline de Trabajo
1.  **Ingesta**: Captura masiva y distribuida de datos por secciones censales y municipios.
2.  **Normalización**: Procesamiento de variables heterogéneas y unificación de estructuras de datos.
3.  **Enriquecimiento**: Cruce de la oferta actual con indicadores socioeconómicos y geoespaciales.
4.  **Modelado**: Aplicación de algoritmos de regresión y aprendizaje supervisado para la estimación de valor.

## 6. Objetivos del Modelado Predictivo
Los modelos desarrollados se orientan a minimizar la incertidumbre en las decisiones de inversión:

* **Estimación de Rentas**: Predicción de precios de alquiler basada en micro-localización y atributos específicos del activo.
* **Valoración de Activos**: Identificación de oportunidades mediante el análisis de desviaciones entre precio de oferta y valor intrínseco.
* **Análisis de Viabilidad**: Integración de predicciones en modelos financieros para el cálculo de ROI y sensibilidad de negocio.

## 7. Consideraciones Éticas y Legales
Este proyecto se enmarca en un contexto estrictamente académico e investigador. El sistema de captura de datos ha sido diseñado para respetar las políticas de acceso de las fuentes consultadas, limitando el uso de la información al análisis estadístico agregado y garantizando la integridad de los datos de origen.

---
**Última actualización**: Febrero 2026