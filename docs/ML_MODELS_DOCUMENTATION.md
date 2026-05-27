# Documentación técnica de modelos predictivos

## 1. Visión general

Este documento describe el estado actual de los modelos predictivos que estiman precios de vivienda en Cantabria dentro del proyecto BezanillaSL. La documentación se basa en los siguientes notebooks y en los artefactos reales presentes en el repositorio:

1. `notebooks/05_ML/53_boost_sale_optuna.ipynb`
2. `notebooks/05_ML/53_boost_rent.ipynb`
3. `notebooks/05_ML/55_sale_rent_models.ipynb`
4. `notebooks/05_ML/55_input_result.ipynb`
5. `notebooks/05_ML/55_input_result_no_k_fold.ipynb`

El pipeline no está formado por análisis aislados. Los notebooks `53_*` optimizan los modelos individuales, `55_sale_rent_models` consolida ambos modelos y genera artefactos persistidos, y los notebooks `55_input_result*` preparan la lógica de inferencia usada por la aplicación.

```text
53_boost_sale_optuna
  -> optimización del modelo de venta
  -> data/model_results/params_sale.json

53_boost_rent
  -> optimización del modelo de alquiler
  -> data/model_results/params_rent.json

55_sale_rent_models
  -> reconstrucción de modelos finales
  -> evaluación conjunta
  -> models/modelo_venta.*
  -> models/modelo_alquiler.*
  -> models/encoders.pkl

55_input_result / 55_input_result_no_k_fold
  -> preparación de inferencia y rangos de error
  -> base para comparar precio estimado y precio observado en la app

streamlit_app/app.py
  -> herramienta final de valoración y comparación inmobiliaria
```

La variable objetivo de ambos modelos es `log_precio`, definida como el logaritmo natural del precio. Las predicciones se transforman a euros mediante `exp(prediccion_log)`.

## 2. Objetivo dentro del TFM

Los modelos permiten convertir información inmobiliaria estructurada en una estimación cuantitativa de precio de venta y de alquiler. En la aplicación final, estas estimaciones se usan como referencia para comparar inmuebles reales del dataset, interpretar desviaciones frente al precio observado y apoyar decisiones de análisis inmobiliario e inversión. El sistema debe entenderse como una herramienta de apoyo a la decisión, no como una garantía de valoración exacta ni de resultado económico.

## 3. Datasets utilizados

| Uso | Ruta | Filas | Columnas | Observación |
|---|---:|---:|---:|---|
| Entrenamiento venta | `data/gold/final_sale_idealistaAPI.csv` | 2532 | 70 | Dataset limpio de venta |
| Entrenamiento alquiler | `data/gold/final_rent_idealistaAPI.csv` | 661 | 47 | Dataset limpio de alquiler |
| Extensión geográfica de alquiler | `data/processed/idealistaAPI/total_rent_cantabria_outliers.csv` | 674 | 49 | Se usa para recuperar municipios agrupados en `municipio_otro` mediante coordenadas |
| App Streamlit, anuncios venta | `data/gold/streamlit_sale.csv` | 2532 | 154 | Dataset enriquecido para mostrar anuncios, filtros y mapa |
| App Streamlit, anuncios alquiler | `data/gold/streamlit_rent.csv` | 674 | 147 | Dataset enriquecido para mostrar anuncios, filtros y mapa |

Los notebooks de modelización no aplican limpieza de outliers. Cargan datasets `gold` que ya llegan depurados por pasos anteriores del pipeline. En los notebooks analizados, solo se eliminan filas sin target válido; en alquiler también se exige `precio_m2` no nulo.

## 4. Variables objetivo

| Modelo | Target principal | Uso | Métrica de optimización |
|---|---|---|---|
| Venta | `log_precio` | Logaritmo natural del precio de venta | RMSE en escala log |
| Alquiler | `log_precio` | Logaritmo natural del alquiler mensual | RMSE en escala log |

En `53_boost_rent` se evalúa además `log_precio_m2` como target alternativo, pero se descarta porque obtiene peor validación cruzada que `log_precio`:

| Target evaluado en alquiler | CV-RMSE | CV-R2 |
|---|---:|---:|
| `log_precio` | 0.15041 | 0.62604 |
| `log_precio_m2` | 0.15501 | 0.50005 |

## 5. Features principales

### 5.1 Modelo de venta

`53_boost_sale_optuna` define 17 `BASE_FEATURES` y añade dinámicamente las columnas `municipio_*` presentes en el dataset. El modelo final usa 47 features.

`BASE_FEATURES` de venta:

```python
[
    "superficie_construida_m2",
    "numero_dormitorios",
    "numero_banos",
    "planta_num",
    "es_exterior_piso",
    "tiene_ascensor_piso",
    "tiene_garaje",
    "obra_nueva",
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
    "precio_m2_municipio_media",
    "interaccion_planta_sin_ascensor_piso",
    "distancia_centro_municipio_km",
    "score_cercania_servicios",
    "tipologia_unificada_piso",
    "tipologia_unificada_unifamiliar",
]
```

En el modelo de venta, `precio_m2_municipio_media` se recalcula en `53_boost_sale_optuna` usando únicamente el conjunto de entrenamiento para reducir riesgo de leakage. El notebook exporta esas medias a `params_sale.json`.

### 5.2 Modelo de alquiler

`53_boost_rent` define 16 `BASE_FEATURES` y añade dinámicamente columnas `municipio_*`. El modelo final usa 23 features.

`BASE_FEATURES` de alquiler:

```python
[
    "superficie_construida_m2",
    "numero_dormitorios",
    "numero_banos",
    "planta_num",
    "es_exterior_piso",
    "tiene_ascensor_piso",
    "tiene_garaje",
    "obra_nueva",
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
    "interaccion_planta_sin_ascensor_piso",
    "distancia_centro_municipio_km",
    "score_cercania_servicios",
    "tipologia_unificada_piso",
    "tipologia_unificada_unifamiliar",
]
```

En alquiler, `precio_m2_municipio_media` queda excluida de las features finales. El notebook calcula medias de precio por metro cuadrado de venta por municipio y las guarda en `params_rent.json`, pero esas medias se usan como referencia geográfica para inferencia, no como feature del modelo de alquiler actual.

### 5.3 Transformaciones comunes

La función `build_X()` aplica una lógica equivalente en los notebooks analizados:

- Selecciona las `BASE_FEATURES` disponibles.
- Añade las columnas `municipio_*`.
- Agrupa municipios con menos de `MIN_MUNI_OBS = 10` observaciones en `municipio_otros`.
- Asigna `NaN` a features específicas de piso en viviendas unifamiliares:
  `planta_num`, `es_exterior_piso`, `tiene_ascensor_piso` e `interaccion_planta_sin_ascensor_piso`.
- Imputa con mediana las features no específicas de piso mediante `SimpleImputer(strategy="median")`. Las features piso-only se dejan en `NaN` para los unifamiliares, ya que XGBoost aprende a enrutar esos `NaN` durante el entrenamiento.
- Como las features piso-only se ponen en `NaN` para unifamiliares antes de calcular `medians`, las medianas de `planta_num`, `es_exterior_piso`, `tiene_ascensor_piso` e `interaccion_planta_sin_ascensor_piso` se calculan únicamente sobre pisos. Estas medianas se reutilizan en inferencia cuando el usuario no especifica esos atributos para un piso (ver sección 11).
- No aplica escalado, porque XGBoost no lo requiere.

Features excluidas o comentadas en los notebooks: `latitud`, `longitud`, `ratio_dormitorios_superficie`, `ratio_banos_superficie`, `precio`, `precio_m2`, `precio_m2_raw`, `log_precio_m2` y `rentabilidad_bruta_zona`.

## 6. Modelos empleados

Los modelos finales de venta y alquiler utilizan `xgboost.XGBRegressor`. No se emplean modelos lineales, Random Forest, SHAP ni modelos híbridos en los cinco notebooks documentados.

Parámetros fijos añadidos al instanciar el estimador:

```python
random_state = 42
n_jobs = -1
verbosity = 0
```

La división de evaluación es:

```python
train_test_split(..., test_size=0.20, random_state=42)
```

## 7. Optimización con Optuna

Los notebooks `53_boost_sale_optuna` y `53_boost_rent` usan Optuna para minimizar el RMSE en validación cruzada:

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

La función objetivo usa:

- `KFold(n_splits=5, shuffle=True, random_state=42)`.
- `cross_val_score(..., scoring="neg_root_mean_squared_error", n_jobs=-1)`.
- Entrenamiento sobre `X_train`, no sobre el conjunto de test.

### 7.1 Espacio de búsqueda, venta

| Hiperparámetro | Rango |
|---|---|
| `n_estimators` | 200 a 1000, `step=50` |
| `max_depth` | 3 a 7 |
| `learning_rate` | 0.01 a 0.20, escala log |
| `subsample` | 0.5 a 1.0 |
| `colsample_bytree` | 0.5 a 1.0 |
| `min_child_weight` | 1 a 15 |
| `reg_lambda` | 0.1 a 10.0, escala log |
| `reg_alpha` | 0.001 a 5.0, escala log |
| `gamma` | 0.0 a 5.0 |

### 7.2 Espacio de búsqueda, alquiler

| Hiperparámetro | Rango |
|---|---|
| `n_estimators` | 200 a 1500 |
| `max_depth` | 3 a 5 |
| `learning_rate` | 0.01 a 0.30, escala log |
| `subsample` | 0.5 a 0.85 |
| `colsample_bytree` | 0.5 a 1.0 |
| `min_child_weight` | 1 a 6 |
| `reg_lambda` | 0.1 a 10.0, escala log |
| `reg_alpha` | 0.0001 a 1.0, escala log |
| `gamma` | 0.0 a 0.05 |

## 8. Modelo de venta: `53_boost_sale_optuna`

Este notebook entrena y optimiza el modelo individual de venta. Sus principales resultados actuales son:

| Elemento | Valor |
|---|---|
| Dataset | `data/gold/final_sale_idealistaAPI.csv` |
| Filas modeladas | 2532 |
| Features finales | 47 |
| Target | `log_precio` |
| Mejor trial Optuna | 76 |
| CV-RMSE | 0.23397 |
| Test RMSE | 0.23625 |
| Test R2 | 0.82947 |
| Artefacto exportado | `data/model_results/params_sale.json` |

Hiperparámetros óptimos exportados:

```json
{
  "n_estimators": 1000,
  "max_depth": 7,
  "learning_rate": 0.012144278135361338,
  "subsample": 0.6586720837240191,
  "colsample_bytree": 0.8255269184092968,
  "min_child_weight": 4,
  "reg_lambda": 0.559036725832902,
  "reg_alpha": 0.554110039059858,
  "gamma": 0.005187403899303554
}
```

El notebook incluye una comparación con hiperparámetros hardcodeados anteriores, usada solo como benchmark interno:

| Métrica | Hardcoded anterior | Optuna actual |
|---|---:|---:|
| CV-RMSE train | 0.24015 | 0.23397 |
| Test RMSE | 0.24837 | 0.23625 |
| Test R2 | 0.81152 | 0.82947 |
| Test MAE | 0.18046 | 0.16921 |

## 9. Modelo de alquiler: `53_boost_rent`

Este notebook entrena y optimiza el modelo individual de alquiler. Sus principales resultados actuales son:

| Elemento | Valor |
|---|---|
| Dataset | `data/gold/final_rent_idealistaAPI.csv` |
| Filas modeladas | 661 |
| Features finales | 23 |
| Target seleccionado | `log_precio` |
| Mejor trial Optuna | 88 |
| CV-RMSE | 0.14785 |
| Test RMSE | 0.15398 |
| Test R2 | 0.60393 |
| Artefacto exportado | `data/model_results/params_rent.json` |

Hiperparámetros óptimos exportados:

```json
{
  "n_estimators": 1014,
  "max_depth": 5,
  "learning_rate": 0.027848574074057768,
  "subsample": 0.8127736802804786,
  "colsample_bytree": 0.5634433497740899,
  "min_child_weight": 3,
  "reg_lambda": 8.702344235467123,
  "reg_alpha": 0.0008762154191077866,
  "gamma": 0.046414487564645036
}
```

El notebook evita explícitamente variables derivadas directamente del target de alquiler. En particular, `rentabilidad_bruta_zona` se descarta por leakage y `precio_m2_municipio_media` no entra en el vector final de features de alquiler.

## 10. Consolidación: `55_sale_rent_models`

`55_sale_rent_models` es el punto de consolidación de los modelos finales de venta y alquiler. Lee los JSON exportados por los notebooks individuales:

- `data/model_results/params_sale.json`
- `data/model_results/params_rent.json`

No copia manualmente los hiperparámetros optimizados. Los reconstruye a partir de los JSON y les añade `random_state=42`, `n_jobs=-1` y `verbosity=0`.

El notebook:

- Carga los datasets `final_sale_idealistaAPI.csv` y `final_rent_idealistaAPI.csv`.
- Reconstruye las features con la misma lógica de `build_X()`.
- En venta, sustituye `precio_m2_municipio_media` por las medias train-only guardadas en `params_sale.json`.
- En alquiler, carga las medias de venta guardadas en `params_rent.json`; en el modelo actual no afectan a `X_rent` porque esa variable no está en `BASE_FEATURES_RENT`.
- Entrena modelos con split 80/20.
- Calcula métricas de train, CV y test.
- Exporta modelos y metadatos a `models/`.

Métricas consolidadas:

| Modelo | Split | MSE | RMSE | MAE | R2 | MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Venta | Train | 0.01098 | 0.10478 | 0.07719 | 0.96593 | 0.00614 |
| Venta | CV | n/a | 0.23397 | n/a | n/a | n/a |
| Venta | Test | 0.05581 | 0.23625 | 0.16921 | 0.82947 | 0.01354 |
| Alquiler | Train | 0.01216 | 0.11027 | 0.08707 | 0.80076 | 0.01271 |
| Alquiler | CV | n/a | 0.14785 | n/a | n/a | n/a |
| Alquiler | Test | 0.02371 | 0.15398 | 0.12093 | 0.60393 | 0.01764 |

Artefactos exportados por `55_sale_rent_models`:

| Ruta | Formato | Contenido |
|---|---|---|
| `models/modelo_venta.json` | XGBoost native JSON | Modelo de venta serializado |
| `models/modelo_alquiler.json` | XGBoost native JSON | Modelo de alquiler serializado |
| `models/modelo_venta.pkl` | joblib/pickle | Modelo de venta serializado |
| `models/modelo_alquiler.pkl` | joblib/pickle | Modelo de alquiler serializado |
| `models/encoders.pkl` | joblib/pickle | Features, medianas, referencias geográficas, RMSE de test y listas auxiliares |

`encoders.pkl` contiene, entre otros campos: `feats_sale`, `feats_rent`, `medians_sale`, `medians_rent`, `sale_geo_ref`, `rent_geo_ref`, `sale_rmse_test`, `rent_rmse_test`, `valid_municipios_sale`, `valid_municipios_rent`, `planta_num_values`, `min_muni_obs_sale` y `min_muni_obs_rent`.

No se exportan CSV de predicciones ni archivos independientes de métricas desde este notebook.

## 11. Preparación de resultados: `55_input_result`

`55_input_result` es un notebook de inferencia interactiva. No es un notebook de optimización. Lee:

- `data/model_results/params_sale.json`
- `data/model_results/params_rent.json`
- `data/gold/final_sale_idealistaAPI.csv`
- `data/gold/final_rent_idealistaAPI.csv`
- `data/processed/idealistaAPI/total_rent_cantabria_outliers.csv`

El notebook entrena en memoria dos modelos XGBoost con split 80/20 y calcula `sale_rmse_test` y `rent_rmse_test` para construir intervalos de error en escala log. El intervalo se transforma a euros mediante:

```python
precio_min = precio_estimado * exp(-rmse_log)
precio_max = precio_estimado * exp(rmse_log)
```

La celda final permite introducir atributos de una vivienda y devuelve:

- Precio estimado de venta.
- Intervalo de error de venta.
- Alquiler mensual estimado.
- Intervalo de error de alquiler.

### 11.1 Entrada de atributos específicos de piso

Las variables `PLANTA`, `ES_EXTERIOR` y `TIENE_ASCENSOR` admiten `None` con dos significados diferenciados, gestionados ambos por `_build_row`:

- Para `TIPOLOGIA = "unifamiliar"`, `None` significa que la feature no aplica. La columna se deja en `NaN` y XGBoost la enruta usando la rama aprendida durante el entrenamiento sobre unifamiliares.
- Para `TIPOLOGIA = "piso"`, `None` significa "indiferente". `_build_row` deja la columna en `NaN` durante el ensamblado y, en el bloque de imputación final, la rellena con la mediana de entrenamiento. Como esas medianas se calculan sobre pisos (las celdas para unifamiliares quedan en `NaN` en `build_X` antes de medianar), el resultado equivale a predecir para un piso con planta, exterior o ascensor "típico".

En la práctica, en el repositorio actual la planta indiferente se traduce en `planta_num ≈ 2`. La interacción `interaccion_planta_sin_ascensor_piso` solo se calcula explícitamente si `PLANTA` y `TIENE_ASCENSOR` son no nulos; en otro caso también cae a su mediana. Si el usuario fija una planta concreta el comportamiento del modelo no cambia respecto a versiones anteriores.

El notebook no compara automáticamente contra un precio real observado. Esa comparación se implementa de forma explícita en `streamlit_app/app.py`, donde los anuncios reales se muestran junto al precio teórico estimado.

También construye una referencia geográfica por municipio y amplía la cobertura de alquiler mediante un join por coordenadas redondeadas entre `total_rent_cantabria_outliers.csv` y `final_rent_idealistaAPI.csv`. En la ejecución guardada, la referencia geográfica queda en 30 municipios para venta y 54 para alquiler.

Este notebook no exporta archivos. Sus salidas impresas pueden quedar desactualizadas si cambian los JSON y no se reejecutan todas las celdas.

## 12. Preparación de resultados sin k-fold operativo: `55_input_result_no_k_fold`

`55_input_result_no_k_fold` replica la lógica de input de `55_input_result`, pero entrena los modelos sobre el 100% de los datos limpios:

```python
model_sale.fit(X_sale, y_sale)
model_rent.fit(X_rent, y_rent)
```

No calcula un RMSE de test propio, porque no reserva conjunto de test. Para los intervalos usa directamente los CV-RMSE exportados en los JSON:

- Venta: `SALE_CV_RMSE = sale_cfg["optuna_cv_rmse"] = 0.23397`
- Alquiler: `RENT_CV_RMSE = rent_cfg["optuna_cv_rmse"] = 0.14785`

Esta versión está más orientada a inferencia con todo el histórico disponible. Sin embargo, la aplicación actual de Streamlit declara y ejecuta una lógica equivalente a `55_input_result`, es decir, reentrena con split 80/20 y usa RMSE de test recalculado en sesión.

Este notebook tampoco exporta archivos.

## 13. Métricas actualizadas

Resumen comparativo de los modelos optimizados y consolidados:

| Modelo | Fase | CV-RMSE | Test RMSE | Test MAE | Test R2 | Test MAPE | Error aproximado en escala original |
|---|---|---:|---:|---:|---:|---:|---:|
| Venta | `53_boost_sale_optuna` y `55_sale_rent_models` | 0.23397 | 0.23625 | 0.16921 | 0.82947 | 0.01354 | `exp(0.23625)-1 = 26.6%` |
| Alquiler | `53_boost_rent` y `55_sale_rent_models` | 0.14785 | 0.15398 | 0.12093 | 0.60393 | 0.01764 | `exp(0.15398)-1 = 16.6%` |

Los notebooks no calculan R2 ajustado.

## 14. Diferencias entre venta y alquiler

| Aspecto | Venta | Alquiler |
|---|---|---|
| Dataset de entrenamiento | 2532 filas | 661 filas |
| Features finales | 47 | 23 |
| Municipios OHE finales | 30 columnas de municipio, incluido `municipio_otro` | 7 columnas de municipio, incluido `municipio_otro` |
| Feature `precio_m2_municipio_media` | Sí, recalculada desde train en `53_boost_sale_optuna` | No entra en el modelo actual |
| Target alternativo evaluado | No | Sí, `log_precio_m2`, descartado |
| Test R2 consolidado | 0.82947 | 0.60393 |
| Test RMSE consolidado | 0.23625 | 0.15398 |

La diferencia de R2 debe interpretarse junto con el tamaño muestral y la naturaleza del mercado. El dataset de alquiler es más pequeño y puede estar afectado por factores no observados en las features disponibles, como estacionalidad, amueblamiento, duración contractual o negociación individual.

## 15. Relación con `data/model_results`

La ruta real del repositorio es `data/model_results`, en plural. No existe una carpeta `data/model_result` en el estado actual.

Archivos actuales:

| Ruta | Generado por | Contenido |
|---|---|---|
| `data/model_results/params_sale.json` | `53_boost_sale_optuna` | Configuración, features base, hiperparámetros, métricas y medias municipales de venta |
| `data/model_results/params_rent.json` | `53_boost_rent` | Configuración, features base, hiperparámetros, métricas y medias municipales de venta usadas como referencia externa |

Estos JSON son la fuente principal para reconstruir los modelos en `55_sale_rent_models`, `55_input_result`, `55_input_result_no_k_fold` y `streamlit_app/app.py`.

## 16. Relación con `models/`

La carpeta `models/` contiene artefactos generados por `55_sale_rent_models`:

```text
models/modelo_venta.json
models/modelo_alquiler.json
models/modelo_venta.pkl
models/modelo_alquiler.pkl
models/encoders.pkl
```

Estos archivos representan una versión persistida de los modelos y de sus metadatos auxiliares. En el estado actual del repositorio, `streamlit_app/app.py` no carga estos `.pkl` ni `.json`; reentrena los modelos al iniciar usando los JSON de `data/model_results` y los CSV de `data/gold`.

Por tanto, `models/` documenta y conserva una consolidación ejecutada, pero la aplicación actual depende directamente de `data/model_results` y `data/gold` para reconstruir los modelos.

## 17. Relación con Streamlit

`streamlit_app/app.py` implementa la herramienta final de valoración. Su cabecera indica que replica `notebooks/05_ML/55_input_result.ipynb`.

La aplicación lee:

- `data/model_results/params_sale.json`
- `data/model_results/params_rent.json`
- `data/gold/final_sale_idealistaAPI.csv`
- `data/gold/final_rent_idealistaAPI.csv`
- `data/gold/streamlit_sale.csv`
- `data/gold/streamlit_rent.csv`
- `data/processed/idealistaAPI/total_rent_cantabria_outliers.csv`

La app:

- Construye matrices de features equivalentes a las del notebook de input.
- Reentrena `XGBRegressor` para venta y alquiler en memoria y cachea los artefactos con `st.cache_resource`.
- Genera precio de venta y alquiler estimados.
- Calcula rangos de error usando RMSE en escala log.
- Muestra anuncios reales del dataset local o de Idealista.
- Compara el precio observado de cada anuncio con el precio teórico estimado.
- Facilita identificar inmuebles relativamente por encima o por debajo del precio estimado por el modelo.
- Incluye visualización de zonas caras, medias y baratas en el mapa a partir de `priceByArea`.

### 17.1 Entrada de planta en la interfaz

Cuando la tipología seleccionada es "Piso", la app muestra un desplegable de planta cuya primera opción y valor por defecto es "Indiferente". Esa opción se propaga internamente como `planta_num = None` y tiene dos efectos:

- En la predicción del modelo, `build_input_row` deja `planta_num` e `interaccion_planta_sin_ascensor_piso` en `NaN` y los rellena con la mediana de pisos del entrenamiento, equivalente al comportamiento descrito en la sección 11.1 del notebook `55_input_result`. La predicción corresponde a un piso con planta típica.
- En el listado de inmuebles reales, `find_local_listings` omite el filtro por planta cuando recibe `planta_num = None`, mostrando todos los inmuebles que cumplen el resto de filtros.

Si el usuario selecciona una planta concreta en el desplegable, tanto la predicción como el filtrado de listados se comportan exactamente como antes del cambio. No se introduce ningún cambio en los modelos entrenados, sólo en cómo se construye el vector de entrada y cómo se filtran los anuncios cuando la planta no se especifica.

La comparación de precio observado frente a precio estimado debe interpretarse como una señal analítica, no como una recomendación automática de compra o alquiler.

## 18. Limitaciones metodológicas

- Los modelos son estimadores hedónicos basados en las variables disponibles en los CSV. No incorporan variables como estado interior detallado, eficiencia energética, amueblamiento, orientación, vistas, ruido, transporte público o restricciones contractuales.
- El modelo de alquiler usa menos observaciones que el de venta, lo que aumenta la sensibilidad a cambios de muestra y a pequeñas variaciones de entrenamiento.
- Los notebooks no calculan R2 ajustado.
- La validación se basa en un split aleatorio 80/20 y KFold con `random_state=42`; no hay validación temporal ni geográfica.
- La app de Streamlit y `55_sale_rent_models` son dos caminos de ejecución relacionados pero no idénticos: `55_sale_rent_models` exporta modelos a `models/`, mientras que la app reentrena desde JSON y CSV.
- Los outputs impresos guardados en notebooks pueden no reflejar los JSON actuales si no se reejecutan todas las celdas después de modificar artefactos.

## 19. Reproducibilidad y consistencia del pipeline

Para mantener la documentación y los resultados alineados, el orden recomendado de ejecución es:

1. Ejecutar `53_boost_sale_optuna`.
2. Ejecutar `53_boost_rent`.
3. Verificar que se han actualizado `data/model_results/params_sale.json` y `data/model_results/params_rent.json`.
4. Ejecutar `55_sale_rent_models` para regenerar `models/`.
5. Ejecutar `55_input_result` o `55_input_result_no_k_fold` según el objetivo de inferencia.
6. Ejecutar Streamlit desde `streamlit_app/app.py`.

Recomendaciones de control:

- Versionar los JSON de parámetros junto con los datasets usados.
- Registrar hashes de los CSV `final_sale_idealistaAPI.csv` y `final_rent_idealistaAPI.csv`.
- Guardar fecha, versión de librerías y entorno de ejecución.
- Evitar mezclar salidas impresas antiguas con JSON actualizados.
- Si se requiere reproducibilidad estricta, fijar versiones exactas de `xgboost` y `optuna` y evaluar `n_jobs=1` durante la optimización.
- Separar de forma explícita artefactos experimentales, artefactos consolidados y artefactos consumidos por la aplicación.
