# BezanillaSL: Real Estate Analytics & Business Feasibility

Este repositorio contiene el desarrollo técnico y estratégico derivado del estudio conjunto de dos Trabajos de Fin de Máster (TFM) para los programas de **MBA Tech** y **Master en Business Analytics**. El proyecto integra el análisis de viabilidad económica de un desarrollo inmobiliario en Cantabria, España, con modelos de analítica de datos para la predicción de precios de suelo y rentas.

## 1. Visión del Proyecto
El objetivo central es validar la factibilidad de una empresa patrimonial familiar orientada al segmento de vivienda asequible (**Affordable Housing**). Este ecosistema analítico busca sustituir la intuición tradicional del sector inmobiliario por un sistema de apoyo a las decisiones basado en evidencia cuantitativa y modelado predictivo, unificando la visión de negocio (MBA) con la precisión técnica (Analytics).

## 2. Equipo y Colaboradores
**Alejandro (Project Owner & Technical Lead)**: Responsable de la arquitectura de datos, desarrollo del código fuente, procesamiento de datasets y liderazgo del modelado predictivo (Analytics).
**Pablo (Technical Collaborator & Theoretical Lead)**: Responsable de la fundamentación teórica, elaboración del plan de negocio, proyecciones financieras de viabilidad y análisis estratégico de mercado.

## 3. Estructura del Repositorio

```text
BezanillaSL/
├── data/
│   ├── raw/                    # Datos originales sin procesar
│   │   ├── idealistaAPI/       # JSON por run de API + CSVs preprocesados por run
│   │   ├── scraping_manual/    # CSVs obtenidos por scraping manual (venta, alquiler, terrenos)
│   │   ├── MIVAU/              # Ministerio de Vivienda: SERPAVI, suelo, parque de viviendas
│   │   ├── INE/                # Censo de Viviendas 2021
│   │   └── euribor_raw.txt     # Serie histórica del Euribor
│   ├── processed/              # Datos limpios y normalizados
│   │   ├── idealistaAPI/       # Totales unificados de todas las runs, sin outliers
│   │   ├── scraping_manual/    # Terrenos scraping sin outliers
│   │   └── geo/                # POIs descargados de OpenStreetMap
│   ├── gold/                   # Datasets finales listos para ML y para la app Streamlit
│   └── model_results/          # Parámetros, hiperparámetros y métricas de los modelos XGBoost
├── notebooks/                  # Cuadernos de análisis por fase (ver notebooks/README.md)
│   ├── 01_manual_scraping_processing/
│   ├── 02_idealista_API_processing/
│   ├── 03_macro_and_structural_analysis/
│   ├── 04_transformations/
│   ├── 05_ML/
│   └── 06_ML_scraping_land/
├── src/                        # Código fuente de producción
│   ├── idealistaAPI/           # Módulo de ingesta y procesamiento vía API Idealista
│   └── geospatial_expansion/   # Módulo de distancias a POIs (playa, colegio, supermercado, etc.)
├── models/                     # Modelos XGBoost serializados (.pkl y .json) + encoders
├── streamlit_app/              # Aplicación web de predicción y comparación de precios
├── docs/                       # Documentación técnica: estructura del repo, modelos ML, diagramas
├── excel/                      # Modelos económicos de viabilidad (Excel)
├── requirements.txt
└── README.md
```

## 4. Pipeline de datos

El flujo de datos sigue una arquitectura de capas:

```
raw → processed → gold → ML (modelos) → streamlit_app
```

1. **Ingesta** — API Idealista (`src/idealistaAPI`) o scraping manual → `data/raw/`
2. **Normalización y outliers** — notebooks `01_*` y `02_*` → `data/processed/`
3. **Feature engineering** — notebooks `04_transformations/` → `data/gold/`
4. **Modelado** — notebooks `05_ML/` y `06_ML_scraping_land/` → `models/` + `data/model_results/`
5. **Explotación** — `streamlit_app/app.py` sirve predicciones desde los modelos serializados

## 5. Módulo Idealista API

Módulo en `src/idealistaAPI` para descargar y preparar datos de Idealista con autenticación OAuth2.

Flujos principales:
1. Descarga de datos de venta o alquiler con autenticación OAuth.
2. Almacenamiento de respuestas crudas en `data/raw/idealistaAPI/raw/<run>/`.
3. Conversión JSON → CSV desde el notebook `idealistaAPI_raw_to_preprocess.ipynb`.
4. Procesamiento analítico posterior en `data/processed/idealistaAPI/`.

Guías y uso:
- Documentación del módulo: `src/idealistaAPI/README.md`
- Guía operativa: `src/idealistaAPI/idealista_API_userguide.md`

## 6. Módulo de Expansión Geoespacial

Módulo en `src/geospatial_expansion` para enriquecer datasets con distancia mínima al POI más cercano por categoría (playa, supermercado, colegio, etc.) usando OpenStreetMap (`osmnx`).

Paso 1: descargar POIs (una sola vez)
```bash
python -m src.geospatial_expansion.run_descargar_pois
```

Paso 2: enriquecer dataset desde notebook/Python:
```python
from src.geospatial_expansion import agregar_distancias_minimas_poi
df_out = agregar_distancias_minimas_poi(df, ["playa", "supermercado"])
```

Guías y uso:
- Documentación del módulo: `src/geospatial_expansion/README.md`
- Guía operativa: `src/geospatial_expansion/geospatial_expansion_userguide.md`

## 7. Aplicación Streamlit

Aplicación web de estimación interactiva de precios de venta y alquiler, visualización de propiedades reales en mapa y comparación con el precio estimado.

```bash
streamlit run streamlit_app/app.py
```

Usa los modelos serializados en `models/` y los datasets gold en `data/gold/streamlit_sale.csv` y `data/gold/streamlit_rent.csv`.

## 8. Documentación técnica

- `docs/estructura_completa_repositorio.md` — arquitectura completa del repositorio, catálogo de notebooks, flujo de datos y deuda técnica
- `docs/ML_MODELS_DOCUMENTATION.md` — especificación de modelos: targets, features, datasets y métricas
- `docs/modelos_regresion_lineal.md` — análisis de modelos OLS, Ridge y Lasso
- `docs/modelos_bagging_random_forest.md` — análisis de Random Forest y Extra Trees
- `docs/modelos_boosting.md` — análisis de XGBoost, GBR y AdaBoost

## 9. Entorno Python recomendado

Crear y activar el entorno virtual desde la raíz del proyecto:

```bash
# Crear (si no existe)
python3 -m venv .venv

# Activar en macOS / Linux
source .venv/bin/activate

# Activar en Windows PowerShell
.venv\Scripts\Activate.ps1
```

Actualizar herramientas base:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Instalar todas las dependencias:

```bash
python -m pip install -r requirements.txt
```

Verificar que el entorno está activo:

```bash
which python   # macOS/Linux
python -V
```
