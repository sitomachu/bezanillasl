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
├── requirements.txt         # Listado de dependencias y versiones.
└── README.md                # Documentación principal del proyecto.