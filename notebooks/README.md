# Notebooks del proyecto

Este directorio agrupa los cuadernos por fase del pipeline analítico.
Ejecutar siempre desde la raíz del repositorio (o ajustar `sys.path` en el notebook).

## Estructura

### `01_manual_scraping_processing` — Procesamiento de scraping manual
- `scraping_sale_processing.ipynb`: limpieza y estandarización de datos de venta scraping.
- `scraping_rent_processing.ipynb`: limpieza y estandarización de datos de alquiler scraping.
- `scraping_land_processing.ipynb`: limpieza y estandarización de datos de terrenos.

### `02_idealista_API_processing` — Procesamiento de datos API Idealista
- `idealistaAPI_raw_to_preprocess.ipynb`: consolida los JSON crudos de cada run en un CSV preprocess por operación (usa `src/idealistaAPI/processing/clean_idealista.py`).
- `idealistaAPI_data.ipynb`: limpieza y unificación de CSVs de todas las ejecuciones; cambia entre `rent` y `sale` con el trigger `OPERATION`.
- `idealistaAPI_processing_outliers.ipynb`: eliminación de outliers (IQR×1.5 sobre `log_precio` y filtros de dominio) y consolidación en `data/processed/idealistaAPI/`.

### `03_macro_and_structural_analysis` — Análisis macro y estructural
- `analisis_SERPAVI.ipynb`: precios de referencia de alquiler por municipio 2011–2023 (MIVAU).
- `analisis_censoviviendas.ipynb`: análisis del parque residencial cántabro (INE 2021).
- `analisis_euribor_tipos.ipynb`: tipos de interés y contexto macroeconómico.
- `analisis_pestle.ipynb`: análisis estratégico PESTLE del entorno inmobiliario.

### `04_transformations` — Transformación processed → gold
- `idealistaAPI_processed_to_gold.ipynb`: genera los datasets ML (`final_sale_idealistaAPI.csv`, `final_rent_idealistaAPI.csv`) con feature engineering completo: encoding, distancias POI, dummies de municipio y log-target.
- `idealistaAPI_processed_to_gold_streamlit_full.ipynb`: genera los datasets para la app Streamlit (`streamlit_sale.csv`, `streamlit_rent.csv`), conservando columnas originales de Idealista (URLs, imágenes, precios reales, coordenadas).
- `scraping_processed_to_gold.ipynb`: genera el gold layer de terrenos (`final_land_scraping.csv`): filtrado, encoding de municipio (target encoding), OHE de tipo_suelo y log-target.

### `05_ML` — Experimentos ML sobre datos API Idealista

Contiene los experimentos completos de las tres familias de modelos. Los notebooks marcados como **[DEFINITIVO]** son los que generan los artefactos finales del pipeline.

**Unificación de dataset:**
- `50_unificar_dataset.ipynb`: combina venta + alquiler para análisis comparativos.

**Regresión lineal (familia 51):**
- `51_linear_regression_1.py`: experimento inicial (script Python).
- `51_linear_regression_2.ipynb`: segunda iteración.
- `51_linear_regression_ridge.ipynb`: experimento específico Ridge.
- `51_linear_regression_lasso.ipynb`: experimento específico Lasso.
- `51_linear_regression_def.ipynb`: OLS, Ridge y Lasso comparados con CV (rutas gold históricas).
- `51_linear_regression_def_2.ipynb`: **[DEFINITIVO]** versión revisada de regresión lineal.

**Random Forest / Extra Trees (familia 52):**
- `52_random_forest_1.ipynb`: primer experimento RF.
- `52_random_forest_2.ipynb`: segunda iteración RF.
- `52_random_forest_scraping.ipynb`: RF sobre datos de scraping manual (venta + alquiler).
- `52_random_forest_def.ipynb`: RF + Extra Trees + RF regularizado con GridSearchCV (rutas gold históricas).
- `52_random_forest_def_2.ipynb`: **[DEFINITIVO]** versión revisada de Random Forest.

**Boosting / XGBoost (familia 53):**
- `53_boost_1.ipynb`: primer experimento boosting.
- `53_boost_reg.ipynb`: boosting con regularización.
- `53_boost_def.ipynb`: XGBoost + GBR + AdaBoost con GridSearchCV.
- `53_boost_def_2.ipynb`: XGBoost con Optuna (primera versión optimizada).
- `53_boost_def_3.ipynb`: XGBoost optimizado por operación (venta/alquiler).
- `53_boost_sale.ipynb`: versión previa al notebook con Optuna (obsoleta).
- `53_boost_sale_optuna.ipynb`: **[DEFINITIVO VENTA]** XGBoost + Optuna 100 trials para M-SALE. Exporta `data/model_results/params_sale.json`.
- `53_boost_rent.ipynb`: **[DEFINITIVO ALQUILER]** XGBoost + Optuna 100 trials para M-RENT. Exporta `data/model_results/params_rent.json`.

**Modelos híbridos (familia 54):**
- `54_hibrido.ipynb`: ensemble combinando familias de modelos.
- `54_hibrido_2.ipynb`: ensemble híbrido v2.

**Consolidación e inferencia (familia 55):**
- `55_sale_rent_models.ipynb`: **[INTEGRACIÓN]** lee `params_*.json`, reentrena M-SALE y M-RENT y exporta modelos finales a `models/` (`.pkl`, `.json`, `encoders.pkl`).
- `55_input_result.ipynb`: **[PREDICCIÓN INDIVIDUAL]** herramienta interactiva de estimación precio venta + alquiler + rentabilidad bruta. Entrena sobre 80% de datos.
- `55_input_result_no_k_fold.ipynb`: **[PREDICCIÓN INDIVIDUAL — PRODUCCIÓN]** igual que el anterior pero entrena sobre el 100% de los datos. Usa CV-RMSE del JSON como intervalo de incertidumbre.

### `06_ML_scraping_land` — Experimentos ML sobre datos de terrenos
- `61_linear_regression.ipynb`: Ridge + Lasso con GridSearchCV (80 alphas, 5-fold CV, `StandardScaler`).
- `62_random_forest.ipynb`: RF + Extra Trees en 4 variantes; optimización con Optuna (40 trials).
- `63_boost.ipynb`: XGBoost baseline + Optuna (50 trials).
- `64_valoracion_suelo.ipynb`: valoración de suelo basada en los modelos anteriores.

---

## Convención de uso

1. Ejecutar los cuadernos desde la raíz del repositorio o ajustar `sys.path` en el notebook.
2. Usar datos de entrada desde `data/raw/`, `data/processed/` o `data/gold/` según la fase.
3. Para cuadernos de API Idealista:
   - `idealistaAPI_raw_to_preprocess.ipynb` parte de `data/raw/idealistaAPI/raw/<run>/`.
   - `idealistaAPI_data.ipynb` parte de `data/raw/idealistaAPI/preprocess/<run>/` y cambia entre `rent` y `sale` con `OPERATION`.
4. Para el pipeline ML principal, el orden canónico es: `53_boost_sale_optuna` / `53_boost_rent` → `55_sale_rent_models` → `streamlit_app/app.py`.

## Dependencias

- Crear y activar `.venv` desde la raíz del repo.
- Instalar con `python -m pip install -r requirements.txt`.
- Usar siempre el mismo `requirements.txt` global para evitar divergencias entre notebooks.
