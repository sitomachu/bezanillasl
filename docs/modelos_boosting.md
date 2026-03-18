# Documentación: Modelos de Boosting — BezanillaSL

## Contexto del proyecto

BezanillaSL es un proyecto de predicción de precios de inmuebles en **Cantabria** (España). Los dos datasets en `data/gold/` son:
- **sale** (`final_sale.csv`): inmuebles en venta
- **rent** (`final_rent.csv`): inmuebles en alquiler

La variable objetivo es **`log_precio`** (logaritmo natural del precio en euros). Se trabaja en escala logarítmica para reducir la heterocedasticidad y simetrizar la distribución. Para convertir predicciones a euros: `precio = exp(log_precio_predicho)`.

---

## Notebook: `53_boost_def.ipynb`

Implementa y compara tres algoritmos de boosting en ambos datasets: **XGBoost**, **Gradient Boosting Regressor (GBR de sklearn)** y **AdaBoost**. Para cada uno se ajusta primero un modelo base con hiperparámetros por defecto y luego se busca el óptimo mediante `GridSearchCV`.

| Modelo | Implementación | Estrategia de boosting | Regularización explícita |
|--------|---------------|----------------------|--------------------------|
| XGBoost | `xgboost.XGBRegressor` | Gradient boosting con segunda derivada (Newton steps) | L1/L2 nativa (`reg_alpha`, `reg_lambda`) + subsampling |
| GBR | `sklearn.GradientBoostingRegressor` | Gradient boosting stagewise (primera derivada) | Subsampling estocástico + `min_samples_leaf` |
| AdaBoost | `sklearn.AdaBoostRegressor` | Reweighting adaptativo de residuos | Ninguna — depende del árbol base |

---

## Configuración global

```python
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
TARGET_COL        = "log_precio"
CV_FOLDS          = 5
MIN_OBS_MUNICIPIO = 10   # municipios con < 10 obs → municipio_otro
```

---

## Features

### Features base (11 activas)

| Feature | Tipo | Nota |
|---------|------|------|
| `superficie_construida_m2` | Continua | Sin log — los árboles son invariantes a transformaciones monotónicas |
| `numero_dormitorios` | Discreta | |
| `numero_banos` | Discreta | |
| `tiene_garaje` | Dummy (0/1) | |
| `obra_nueva` | Dummy (0/1) | |
| `distancia_min_playa_km` | Continua | |
| `distancia_min_supermercado_km` | Continua | |
| `distancia_min_colegio_km` | Continua | |
| `distancia_centro_municipio_km` | Continua | |
| `tipologia_unificada_piso` | Dummy (0/1) | |
| `tipologia_unificada_unifamiliar` | Dummy (0/1) | |

**Features comentadas (no activas):** `latitud`, `longitud` y `score_cercania_servicios`. Las coordenadas aumentan la dimensionalidad sin mejorar resultados con estos tamaños de muestra. El score de cercanía es un compuesto de las distancias individuales — el boosting puede capturar sus interacciones no lineales directamente desde las distancias, haciendo el score redundante.

### Features de municipio

38 municipios candidatos en `MUNICIPIO_FEATURES`. Los que tienen < 10 observaciones en el dataset se colapsan en `municipio_otro` mediante la función `collapse_rare_municipios()`.

| Dataset | Municipios ≥ 10 obs (mantenidos) | Municipios < 10 obs → `municipio_otro` | Total features finales |
|---------|----------------------------------|----------------------------------------|------------------------|
| Sale | 13 | 22 | **25** (11 base + 13 municipio + `municipio_otro`) |
| Rent | 9 | 29 | **21** (11 base + 9 municipio + `municipio_otro`) |

**Sale — municipios mantenidos (13):** `Camargo`, `Castro-Urdiales`, `Laredo`, `Noja`, `Piélagos`, `Polanco`, `Santa Cruz de Bezana`, `Santander`, `Santoña`, `Santurtzi`, `Suances`, `Torrelavega`, `Voto`

**Rent — municipios mantenidos (9):** `Camargo`, `Castro-Urdiales`, `El Astillero`, `Laredo`, `Piélagos`, `Santa Cruz de Bezana`, `Santander`, `Torrelavega`, + `municipio_otro`

> 3 municipios de la lista candidata (`Sobremazas`, `Villaescusa`, `Viveda`) no están presentes en el dataset sale — no aparecen en ninguna fila del CSV.

### Preprocesado

Función `prepare_X(df, feature_cols)`: selecciona las columnas disponibles e imputa valores nulos con la **mediana** de cada columna (`SimpleImputer(strategy="median")`). Los modelos de boosting basados en árboles son invariantes a la escala — no se aplica estandarización.

---

## Pipeline común

1. Carga de `final_sale.csv` o `final_rent.csv`
2. Filtrado de filas con `log_precio` nulo (`df[TARGET_COL].notna()`)
3. Colapso de municipios raros → `municipio_otro`
4. Preparación de features con imputación de mediana
5. **Split 80/20** `train_test_split(random_state=42)`
6. Para cada modelo: modelo base → `plot_diagnostics()` + `plot_feature_importance()` → GridSearchCV → `plot_diagnostics()` + `plot_feature_importance()`
7. Resumen comparativo por dataset y global al final

**Nota importante:** a diferencia del notebook 51 (regresión lineal), aquí **no se eliminan outliers** del target antes del split. Los modelos de árboles no asumen linealidad ni distribución gaussiana de los residuos — los outliers son manejables sin eliminarlos.

---

## Datos de entrada

### Dataset SALE (venta)

| Etapa | Valor |
|-------|-------|
| Filas cargadas | 588 |
| Municipios ≥ 10 obs | 13 (+ `municipio_otro`) |
| Features finales | 25 |
| Train | 470 |
| Test | 118 |

### Dataset RENT (alquiler)

| Etapa | Valor |
|-------|-------|
| Filas cargadas | 477 |
| Municipios ≥ 10 obs | 9 (+ `municipio_otro`) |
| Features finales | 21 |
| Train | 381 |
| Test | 96 |

---

## Grids de hiperparámetros

### XGBoost — 192 combinaciones

```python
PARAM_GRID_XGB = {
    "n_estimators":     [200, 400],       # número de árboles / rondas de boosting
    "max_depth":        [3, 4, 5],        # profundidad máxima de cada árbol base
    "learning_rate":    [0.05, 0.1],      # shrinkage: escala la contribución de cada árbol
    "subsample":        [0.7, 0.9],       # fracción de filas muestreadas por árbol (stochastic boosting)
    "colsample_bytree": [0.7, 0.9],       # fracción de features muestreadas por árbol
    "min_child_weight": [3, 5],           # suma mínima de pesos en una hoja (regularización)
    "reg_lambda":       [1, 5],           # regularización L2 sobre los pesos de las hojas
}
```

> XGBoost tiene el grid más grande porque tiene más mecanismos de regularización que los otros dos. `reg_lambda` y `min_child_weight` son específicos de XGBoost — actúan directamente sobre el cálculo del gradiente de segundo orden (Newton boosting). `colsample_bytree` introduce aleatoriedad adicional no presente en GBR estándar.

### Gradient Boosting Regressor — 48 combinaciones

```python
PARAM_GRID_GBR = {
    "n_estimators":     [200, 400],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.7, 0.9],       # subsampling estocástico (Stochastic GBM)
    "min_samples_leaf": [5, 10],          # número mínimo de muestras en hoja
}
```

> GBR de sklearn no tiene equivalente a `reg_lambda` ni `colsample_bytree`. El grid es más pequeño (48 combinaciones). La principal diferencia frente a XGBoost es que usa solo la primera derivada (gradiente), mientras que XGBoost usa también la segunda derivada (curvatura) para calcular los valores óptimos de las hojas.

### AdaBoost — 27 combinaciones

```python
PARAM_GRID_ADA = {
    "n_estimators":         [100, 200, 400],
    "learning_rate":        [0.5, 1.0, 1.5],
    "estimator__max_depth": [3, 5, 7],      # profundidad del árbol base (DecisionTreeRegressor)
}
```

> AdaBoost no tiene subsampling ni regularización L2 explícita. El estimador base es un `DecisionTreeRegressor()` genérico en el GridSearch (sin restricciones de profundidad salvo `estimator__max_depth`), y `DecisionTreeRegressor(max_depth=3)` en el modelo base. La sintaxis `estimator__max_depth` usa el doble guión bajo de sklearn para pasar parámetros al estimador base a través de la interfaz de Pipeline. En AdaBoost, `learning_rate` no es un shrinkage del gradiente sino un escalar multiplicativo del peso asignado a cada árbol en el ensemble.

**Scoring en GridSearchCV:** `"neg_root_mean_squared_error"` (RMSE negativo) con `CV_FOLDS=5`. `n_jobs=-1` en todos los GridSearchCV y en XGBoost para paralelizar en todos los cores disponibles.

---

## Resultados — Dataset SALE (venta)

### 1. XGBoost — SALE

#### Modelo base (hiperparámetros por defecto de XGBRegressor)

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.00006 0.00766 0.00494 0.99982 0.00040
 test 0.11461 0.33854 0.24241 0.57907 0.01969
Sobreajuste → ratio RMSE test/train: 44.20 | delta R2: 0.4208
```

XGBoost base memoriza el train de forma casi perfecta (R²=0.9998, RMSE=0.008). Los hiperparámetros por defecto de `XGBRegressor` (`max_depth=6`, `learning_rate=0.3`, `subsample=1.0`, `colsample_bytree=1.0`, sin regularización extra) son agresivos y producen sobreajuste severo. El test R² (0.579) es el más bajo de todos los modelos boosting en sale.

**Feature importances — XGBoost base SALE (top 10):**

| Feature | Importancia |
|---------|------------|
| `municipio_Santoña` | 0.2581 |
| `tiene_garaje` | 0.1343 |
| `superficie_construida_m2` | 0.0879 |
| `distancia_min_playa_km` | 0.0573 |
| `numero_banos` | 0.0553 |
| `municipio_Santander` | 0.0459 |
| `municipio_otro` | 0.0415 |
| `municipio_Castro-Urdiales` | 0.0374 |
| `municipio_Voto` | 0.0371 |
| `municipio_Torrelavega` | 0.0347 |

> El modelo base sobreajustado concentra importancia en municipios individuales (`Santoña` con 25.8%) — señal de memorización de patrones específicos del train. Con regularización esto se distribuye más uniformemente.

#### Modelo óptimo (GridSearch, 192 combinaciones)

```
Mejores parámetros: {
    'colsample_bytree': 0.7,
    'learning_rate':    0.05,
    'max_depth':        3,
    'min_child_weight': 3,
    'n_estimators':     200,
    'reg_lambda':       5,
    'subsample':        0.7
}
CV RMSE (mejor): 0.36396

split     MSE    RMSE     MAE      R2    MAPE
train 0.05526 0.23506 0.17334 0.83087 0.01414
   CV     NaN 0.36396     NaN     NaN     NaN
 test 0.09936 0.31522 0.23112 0.63506 0.01877
Sobreajuste → ratio RMSE test/train: 1.3410 | delta R2: 0.1958
```

El GridSearch selecciona `max_depth=3` (árboles poco profundos), `learning_rate=0.05` (shrinkage fuerte), doble subsampling (`subsample=0.7`, `colsample_bytree=0.7`) y `reg_lambda=5` (regularización L2 alta). La combinación reduce el ratio de sobreajuste de 44.20 a 1.34. Mejora el test R² de 0.579 a 0.635.

**Feature importances — XGBoost óptimo SALE (top 15):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.1317 |
| `municipio_Noja` | 0.0880 |
| `numero_dormitorios` | 0.0866 |
| `numero_banos` | 0.0857 |
| `tiene_garaje` | 0.0813 |
| `distancia_min_playa_km` | 0.0606 |
| `municipio_Castro-Urdiales` | 0.0491 |
| `municipio_Torrelavega` | 0.0459 |
| `tipologia_unificada_piso` | 0.0418 |
| `municipio_otro` | 0.0415 |
| `municipio_Santander` | 0.0378 |
| `municipio_Suances` | 0.0293 |
| `municipio_Santoña` | 0.0256 |
| `municipio_Polanco` | 0.0236 |
| `distancia_min_colegio_km` | 0.0235 |

> Con regularización, la importancia se redistribuye: `superficie_construida_m2` sube al primer puesto (0.13), los municipios se equilibran (ninguno supera 0.09) y `numero_dormitorios` aparece como relevante. `municipio_Santoña` pasa de 0.258 a 0.026 — confirmando que el modelo base memorizaba una particularidad del train.

---

### 2. Gradient Boosting Regressor — SALE

#### Modelo base (hiperparámetros por defecto de sklearn)

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.04299 0.20734 0.15802 0.86841 0.01287
 test 0.09885 0.31440 0.23482 0.63696 0.01904
Sobreajuste → ratio RMSE test/train: 1.5163 | delta R2: 0.2315
```

GBR base ya generaliza mucho mejor que XGBoost base. Los hiperparámetros por defecto de sklearn (`max_depth=3`, `learning_rate=0.1`, `n_estimators=100`, sin subsampling) son más conservadores. Nótese que GBR base (test R²=0.637) supera a XGBoost óptimo (test R²=0.635) — la configuración por defecto de sklearn GBR está bien calibrada para este tipo de problema.

**Feature importances — GBR base SALE (top 10):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.4550 |
| `distancia_min_playa_km` | 0.1977 |
| `numero_banos` | 0.0811 |
| `tiene_garaje` | 0.0791 |
| `distancia_min_colegio_km` | 0.0487 |
| `distancia_min_supermercado_km` | 0.0311 |
| `distancia_centro_municipio_km` | 0.0292 |
| `numero_dormitorios` | 0.0278 |
| `municipio_Santander` | 0.0156 |
| `municipio_Santoña` | 0.0119 |

> GBR concentra la importancia en `superficie_construida_m2` (45.5%) y `distancia_min_playa_km` (19.8%). Es un patrón mucho más estable que el de XGBoost base — confirma que en venta la superficie y la proximidad al mar son los dos predictores dominantes.

#### Modelo óptimo (GridSearch, 48 combinaciones)

```
Mejores parámetros: {
    'learning_rate':    0.05,
    'max_depth':        5,
    'min_samples_leaf': 5,
    'n_estimators':     200,
    'subsample':        0.9
}
CV RMSE (mejor): 0.37350

split     MSE    RMSE     MAE      R2    MAPE
train 0.01124 0.10600 0.07717 0.96561 0.00629
   CV     NaN 0.37350     NaN     NaN     NaN
 test 0.10184 0.31912 0.22423 0.62597 0.01827
Sobreajuste → ratio RMSE test/train: 3.0106 | delta R2: 0.3396
```

⚠️ **Anomalía notable:** el GridSearch de GBR sale selecciona `max_depth=5` (más profundo que el base que usa `max_depth=3` por defecto), lo que **empeora** el test R² respecto al base (0.626 vs 0.637) y dispara el ratio de sobreajuste a 3.01. El CV RMSE (0.374) es además el más alto de todos los modelos boosting en sale. El GridSearch no encuentra una configuración mejor que el base para GBR en venta.

**Feature importances — GBR óptimo SALE (top 15):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.4271 |
| `distancia_min_playa_km` | 0.1922 |
| `tiene_garaje` | 0.0719 |
| `distancia_min_colegio_km` | 0.0680 |
| `distancia_centro_municipio_km` | 0.0581 |
| `distancia_min_supermercado_km` | 0.0553 |
| `numero_banos` | 0.0426 |
| `numero_dormitorios` | 0.0355 |
| `municipio_otro` | 0.0185 |
| `municipio_Santander` | 0.0091 |
| `municipio_Santoña` | 0.0082 |
| `municipio_Torrelavega` | 0.0029 |
| `tipologia_unificada_piso` | 0.0018 |
| `tipologia_unificada_unifamiliar` | 0.0015 |
| `municipio_Santurtzi` | 0.0014 |

> El patrón de importancias GBR es muy estable entre base y óptimo — `superficie_construida_m2` y `distancia_min_playa_km` siguen dominando juntas (~62% de la importancia total). Los municipios tienen importancias residuales (< 2% cada uno), a diferencia de XGBoost donde los municipios individuales reciben más peso.

---

### 3. AdaBoost — SALE

#### Modelo base (árbol base `max_depth=3`)

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.10190 0.31922 0.26463 0.68808 0.02142
 test 0.13101 0.36195 0.28181 0.51884 0.02275
Sobreajuste → ratio RMSE test/train: 1.1339 | delta R2: 0.1692
```

AdaBoost base tiene el menor sobreajuste de los tres modelos en sale (ratio=1.13, delta_R2=0.17), pero el peor test R² (0.519) y el peor RMSE_test (0.362). El árbol base de `max_depth=3` con los hiperparámetros por defecto de AdaBoost no tiene capacidad suficiente para capturar la complejidad del mercado de venta.

**Feature importances — AdaBoost base SALE (top 10):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.3135 |
| `distancia_min_playa_km` | 0.1642 |
| `numero_banos` | 0.1594 |
| `tiene_garaje` | 0.1017 |
| `distancia_centro_municipio_km` | 0.0696 |
| `distancia_min_supermercado_km` | 0.0428 |
| `numero_dormitorios` | 0.0400 |
| `municipio_otro` | 0.0381 |
| `municipio_Santander` | 0.0306 |
| `distancia_min_colegio_km` | 0.0224 |

#### Modelo óptimo (GridSearch, 27 combinaciones)

```
Mejores parámetros: {
    'estimator__max_depth': 7,
    'learning_rate':        1.5,
    'n_estimators':         400
}
CV RMSE (mejor): 0.35606

split     MSE    RMSE     MAE      R2    MAPE
train 0.01165 0.10792 0.08396 0.96435 0.00677
   CV     NaN 0.35606     NaN     NaN     NaN
 test 0.09784 0.31280 0.23176 0.64065 0.01886
Sobreajuste → ratio RMSE test/train: 2.8984 | delta R2: 0.3237
```

El GridSearch selecciona `max_depth=7` y `learning_rate=1.5` (valores máximos del grid) con `n_estimators=400` (también el máximo). La mejora es drástica: test R² de 0.519 → 0.641 y RMSE_test de 0.362 → 0.313. AdaBoost óptimo resulta ser el **mejor modelo boosting para sale** (test R²=0.641, RMSE_test=0.313, CV_RMSE=0.356), aunque con sobreajuste significativo (ratio=2.90).

**Feature importances — AdaBoost óptimo SALE (top 15):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.3198 |
| `distancia_min_playa_km` | 0.1652 |
| `numero_banos` | 0.1006 |
| `tiene_garaje` | 0.0849 |
| `distancia_min_colegio_km` | 0.0770 |
| `distancia_centro_municipio_km` | 0.0752 |
| `distancia_min_supermercado_km` | 0.0670 |
| `numero_dormitorios` | 0.0423 |
| `municipio_otro` | 0.0218 |
| `municipio_Santander` | 0.0127 |
| `municipio_Santoña` | 0.0121 |
| `tipologia_unificada_unifamiliar` | 0.0033 |
| `municipio_Torrelavega` | 0.0031 |
| `tipologia_unificada_piso` | 0.0025 |
| `municipio_Suances` | 0.0021 |

> Las importancias de AdaBoost son muy similares entre base y óptimo — la estructura de importancias es estable. La diferencia clave es que el óptimo captura relaciones más complejas con `max_depth=7`, no cambia la jerarquía de variables sino la profundidad con la que las explota.

---

### Resumen comparativo — SALE

| Modelo | Fase | MSE_train | RMSE_train | MAE_train | R²_train | CV_RMSE | MSE_test | RMSE_test | MAE_test | R²_test | MAPE_test | ratio_RMSE | delta_R2 |
|--------|------|----------|-----------|---------|---------|---------|---------|----------|---------|---------|----------|-----------|---------|
| XGBoost | base | 0.00006 | 0.00766 | 0.00494 | 0.99982 | — | 0.11461 | 0.33854 | 0.24241 | 0.57907 | 0.01969 | 44.20 | 0.421 |
| XGBoost | **óptimo** | 0.05526 | 0.23506 | 0.17334 | 0.83087 | 0.36396 | 0.09936 | 0.31522 | 0.23112 | 0.63506 | 0.01877 | 1.34 | 0.196 |
| GBR | **base** | 0.04299 | 0.20734 | 0.15802 | 0.86841 | — | 0.09885 | 0.31440 | 0.23482 | 0.63696 | 0.01904 | 1.52 | 0.232 |
| GBR | óptimo | 0.01124 | 0.10600 | 0.07717 | 0.96561 | 0.37350 | 0.10184 | 0.31912 | 0.22423 | 0.62597 | 0.01827 | 3.01 | 0.340 |
| AdaBoost | base | 0.10190 | 0.31922 | 0.26463 | 0.68808 | — | 0.13101 | 0.36195 | 0.28181 | 0.51884 | 0.02275 | 1.13 | 0.169 |
| AdaBoost | **óptimo** | 0.01165 | 0.10792 | 0.08396 | 0.96435 | **0.35606** | 0.09784 | **0.31280** | 0.23176 | **0.64065** | 0.01886 | 2.90 | 0.324 |

**Sale: AdaBoost óptimo es el mejor** por RMSE_test (0.313) y R²_test (0.641). GBR base es el segundo mejor y el más robusto (menor sobreajuste entre los que generalizan bien).

---

## Resultados — Dataset RENT (alquiler)

### 1. XGBoost — RENT

#### Modelo base

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.00075 0.02734 0.00449 0.99529 0.00061
 test 0.10948 0.33088 0.23956 0.19691 0.03407
Sobreajuste → ratio RMSE test/train: 12.10 | delta R2: 0.7984
```

XGBoost base en rent es prácticamente inútil sin regularización: test R²=0.197. El delta_R2=0.798 es el más alto de todo el proyecto. El modelo memoriza casi perfectamente el train (R²=0.995) pero falla completamente en generalizar al conjunto de alquiler.

**Feature importances — XGBoost base RENT (top 10):**

| Feature | Importancia |
|---------|------------|
| `municipio_Laredo` | **0.6735** |
| `municipio_Torrelavega` | 0.0683 |
| `municipio_Suances` | 0.0357 |
| `municipio_Santander` | 0.0304 |
| `municipio_Piélagos` | 0.0265 |
| `superficie_construida_m2` | 0.0205 |
| `municipio_Castro-Urdiales` | 0.0165 |
| `numero_banos` | 0.0163 |
| `municipio_Santa Cruz de Bezana` | 0.0161 |
| `municipio_Camargo` | 0.0145 |

> Señal inequívoca de memorización: `municipio_Laredo` acapara el 67.4% de toda la importancia. El modelo ha encontrado que los alquileres en Laredo tienen un precio particular en el train y lo está memorizando en lugar de aprender el patrón general.

#### Modelo óptimo (GridSearch, 192 combinaciones)

```
Mejores parámetros: {
    'colsample_bytree': 0.7,
    'learning_rate':    0.05,
    'max_depth':        3,
    'min_child_weight': 5,
    'n_estimators':     200,
    'reg_lambda':       5,
    'subsample':        0.9
}
CV RMSE (mejor): 0.29769

split     MSE    RMSE     MAE      R2    MAPE
train 0.04246 0.20605 0.14276 0.73232 0.02013
   CV     NaN 0.29769     NaN     NaN     NaN
 test 0.08343 0.28883 0.20178 0.38804 0.02845
Sobreajuste → ratio RMSE test/train: 1.4017 | delta R2: 0.3443
```

El óptimo de rent y sale eligen configuraciones casi idénticas — solo difiere `min_child_weight` (5 vs 3) y `subsample` (0.9 vs 0.7). Con regularización, el test R² pasa de 0.197 a 0.388 — la mejora más grande de cualquier modelo en cualquier dataset.

**Feature importances — XGBoost óptimo RENT (top 15):**

| Feature | Importancia |
|---------|------------|
| `municipio_Laredo` | 0.2637 |
| `superficie_construida_m2` | 0.1240 |
| `numero_dormitorios` | 0.1131 |
| `municipio_Santander` | 0.0797 |
| `tipologia_unificada_piso` | 0.0466 |
| `tipologia_unificada_unifamiliar` | 0.0455 |
| `distancia_min_playa_km` | 0.0444 |
| `numero_banos` | 0.0408 |
| `municipio_otro` | 0.0368 |
| `distancia_min_supermercado_km` | 0.0337 |
| `municipio_Piélagos` | 0.0324 |
| `distancia_centro_municipio_km` | 0.0278 |
| `distancia_min_colegio_km` | 0.0274 |
| `municipio_Castro-Urdiales` | 0.0255 |
| `municipio_Suances` | 0.0231 |

> Con regularización, `municipio_Laredo` baja de 0.674 a 0.264 — sigue siendo la feature más importante pero ya no domina abrumadoramente. `superficie_construida_m2` y `numero_dormitorios` recuperan peso como predictores fundamentales.

---

### 2. Gradient Boosting Regressor — RENT

#### Modelo base

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.02840 0.16852 0.12040 0.82095 0.01704
 test 0.08497 0.29150 0.21043 0.37669 0.02978
Sobreajuste → ratio RMSE test/train: 1.7298 | delta R2: 0.4443
```

GBR base en rent tiene sobreajuste moderado (ratio=1.73) y test R²=0.377. Similar al patrón en sale: los defaults de sklearn GBR son más conservadores que XGBoost.

**Feature importances — GBR base RENT (top 10):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.3377 |
| `municipio_Laredo` | 0.2203 |
| `distancia_min_playa_km` | 0.1099 |
| `distancia_min_colegio_km` | 0.0818 |
| `distancia_centro_municipio_km` | 0.0667 |
| `distancia_min_supermercado_km` | 0.0656 |
| `numero_dormitorios` | 0.0383 |
| `municipio_Santander` | 0.0293 |
| `numero_banos` | 0.0238 |
| `municipio_Suances` | 0.0065 |

> En rent, `municipio_Laredo` es la segunda feature más importante en GBR (22%) — refleja que Laredo tiene un mercado de alquiler diferencial dentro de Cantabria, probablemente por su demanda turística y tamaño poblacional.

#### Modelo óptimo (GridSearch, 48 combinaciones)

```
Mejores parámetros: {
    'learning_rate':    0.05,
    'max_depth':        3,
    'min_samples_leaf': 10,
    'n_estimators':     200,
    'subsample':        0.9
}
CV RMSE (mejor): 0.29690

split     MSE    RMSE     MAE      R2    MAPE
train 0.03998 0.19994 0.13751 0.74795 0.01942
   CV     NaN 0.29690     NaN     NaN     NaN
 test 0.08756 0.29590 0.21130 0.35774 0.02986
Sobreajuste → ratio RMSE test/train: 1.4799 | delta R2: 0.3902
```

GBR óptimo en rent empeora ligeramente el test R² respecto al base (0.358 vs 0.377) — mismo patrón que en sale. El GridSearch reduce el sobreajuste (ratio 1.73 → 1.48) pero a costa del test R². El CV RMSE de GBR rent (0.29690) es casi idéntico al de XGBoost (0.29769).

**Feature importances — GBR óptimo RENT (top 15):**

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.3379 |
| `municipio_Laredo` | 0.2217 |
| `distancia_min_playa_km` | 0.1261 |
| `distancia_min_colegio_km` | 0.0796 |
| `distancia_min_supermercado_km` | 0.0783 |
| `distancia_centro_municipio_km` | 0.0502 |
| `numero_dormitorios` | 0.0332 |
| `numero_banos` | 0.0319 |
| `municipio_Santander` | 0.0309 |
| `tiene_garaje` | 0.0041 |
| `municipio_otro` | 0.0032 |
| `tipologia_unificada_unifamiliar` | 0.0016 |
| `municipio_Piélagos` | 0.0006 |
| `municipio_Castro-Urdiales` | 0.0003 |
| `tipologia_unificada_piso` | 0.0003 |

> El patrón de importancias GBR rent es extremadamente estable entre base y óptimo — misma jerarquía, prácticamente mismas proporciones. Las 3 primeras features (`superficie_construida_m2`, `municipio_Laredo`, `distancia_min_playa_km`) explican el 68% de la importancia total.

---

### 3. AdaBoost — RENT

#### Modelo base (árbol base `max_depth=3`)

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.09841 0.31370 0.26789 0.37951 0.03857
 test 0.11480 0.33883 0.29327 0.15787 0.04223
Sobreajuste → ratio RMSE test/train: 1.0801 | delta R2: 0.2216
```

AdaBoost base en rent es el peor modelo de todo el proyecto: test R²=0.158. Incluso el train R² (0.380) es bajo — el árbol base de `max_depth=3` no logra capturar la complejidad del mercado de alquiler ni en train. El MAPE de test (4.22%) es el más alto de todos los modelos y datasets.

**Feature importances — AdaBoost base RENT (top 10):**

| Feature | Importancia |
|---------|------------|
| `distancia_min_playa_km` | 0.2130 |
| `superficie_construida_m2` | 0.1851 |
| `distancia_centro_municipio_km` | 0.1740 |
| `distancia_min_colegio_km` | 0.1361 |
| `municipio_Laredo` | 0.1145 |
| `distancia_min_supermercado_km` | 0.0847 |
| `numero_dormitorios` | 0.0313 |
| `municipio_otro` | 0.0270 |
| `tiene_garaje` | 0.0134 |
| `municipio_Santander` | 0.0095 |

#### Modelo óptimo (GridSearch, 27 combinaciones)

```
Mejores parámetros: {
    'estimator__max_depth': 7,
    'learning_rate':        1.5,
    'n_estimators':         400
}
CV RMSE (mejor): 0.31359

split     MSE    RMSE     MAE      R2    MAPE
train 0.01344 0.11591 0.08985 0.91529 0.01295
   CV     NaN 0.31359     NaN     NaN     NaN
 test 0.08811 0.29683 0.19956 0.35367 0.02818
Sobreajuste → ratio RMSE test/train: 2.5609 | delta R2: 0.5616
```

El GridSearch selecciona exactamente los mismos parámetros que en sale (`max_depth=7`, `learning_rate=1.5`, `n_estimators=400`). La mejora es enorme: test R² de 0.158 → 0.354, RMSE_test de 0.339 → 0.297. Aun así no alcanza a XGBoost óptimo ni a GBR base. El delta_R2=0.562 sigue siendo alto — sobreajuste significativo.

**Feature importances — AdaBoost óptimo RENT (top 15):**

| Feature | Importancia |
|---------|------------|
| `distancia_centro_municipio_km` | 0.2236 |
| `superficie_construida_m2` | 0.2001 |
| `distancia_min_playa_km` | 0.1891 |
| `distancia_min_colegio_km` | 0.1402 |
| `distancia_min_supermercado_km` | 0.1049 |
| `municipio_Laredo` | 0.0654 |
| `numero_dormitorios` | 0.0288 |
| `municipio_otro` | 0.0119 |
| `numero_banos` | 0.0110 |
| `tiene_garaje` | 0.0088 |
| `municipio_Santander` | 0.0057 |
| `municipio_Suances` | 0.0034 |
| `tipologia_unificada_piso` | 0.0020 |
| `municipio_Castro-Urdiales` | 0.0015 |
| `tipologia_unificada_unifamiliar` | 0.0012 |

> AdaBoost rent basa su predicción casi exclusivamente en variables de distancia (~65% de la importancia total entre las 4 distancias). A diferencia de XGBoost y GBR que colocan `superficie_construida_m2` primero, AdaBoost da más peso a la ubicación que a las características físicas del inmueble en alquiler.

---

### Resumen comparativo — RENT

| Modelo | Fase | MSE_train | RMSE_train | MAE_train | R²_train | CV_RMSE | MSE_test | RMSE_test | MAE_test | R²_test | MAPE_test | ratio_RMSE | delta_R2 |
|--------|------|----------|-----------|---------|---------|---------|---------|----------|---------|---------|----------|-----------|---------|
| XGBoost | base | 0.00075 | 0.02734 | 0.00449 | 0.99529 | — | 0.10948 | 0.33088 | 0.23956 | 0.19691 | 0.03407 | 12.10 | 0.798 |
| XGBoost | **óptimo** | 0.04246 | 0.20605 | 0.14276 | 0.73232 | **0.29769** | 0.08343 | **0.28883** | **0.20178** | **0.38804** | 0.02845 | 1.40 | 0.344 |
| GBR | **base** | 0.02840 | 0.16852 | 0.12040 | 0.82095 | — | 0.08497 | 0.29150 | 0.21043 | 0.37669 | 0.02978 | 1.73 | 0.444 |
| GBR | óptimo | 0.03998 | 0.19994 | 0.13751 | 0.74795 | 0.29690 | 0.08756 | 0.29590 | 0.21130 | 0.35774 | 0.02986 | 1.48 | 0.390 |
| AdaBoost | base | 0.09841 | 0.31370 | 0.26789 | 0.37951 | — | 0.11480 | 0.33883 | 0.29327 | 0.15787 | 0.04223 | 1.08 | 0.222 |
| AdaBoost | óptimo | 0.01344 | 0.11591 | 0.08985 | 0.91529 | 0.31359 | 0.08811 | 0.29683 | 0.19956 | 0.35367 | 0.02818 | 2.56 | 0.562 |

**Rent: XGBoost óptimo es el mejor** por RMSE_test (0.289), R²_test (0.388) y MAE_test (0.202). GBR base es el segundo mejor y el más estable.

---

## Resumen global completo

| Dataset | Modelo | Fase | RMSE_train | CV_RMSE | RMSE_test | R²_test | MAE_test | MAPE_test | ratio_RMSE | delta_R2 |
|---------|--------|------|-----------|---------|----------|---------|---------|----------|-----------|---------|
| sale | XGBoost | base | 0.00766 | — | 0.33854 | 0.57907 | 0.24241 | 0.01969 | 44.20 | 0.421 |
| sale | XGBoost | óptimo | 0.23506 | 0.36396 | 0.31522 | 0.63506 | 0.23112 | 0.01877 | 1.34 | 0.196 |
| sale | GBR | base | 0.20734 | — | 0.31440 | 0.63696 | 0.23482 | 0.01904 | 1.52 | 0.232 |
| sale | GBR | óptimo | 0.10600 | 0.37350 | 0.31912 | 0.62597 | 0.22423 | 0.01827 | 3.01 | 0.340 |
| sale | AdaBoost | base | 0.31922 | — | 0.36195 | 0.51884 | 0.28181 | 0.02275 | 1.13 | 0.169 |
| sale | AdaBoost | **óptimo** | 0.10792 | **0.35606** | **0.31280** | **0.64065** | 0.23176 | 0.01886 | 2.90 | 0.324 |
| rent | XGBoost | base | 0.02734 | — | 0.33088 | 0.19691 | 0.23956 | 0.03407 | 12.10 | 0.798 |
| rent | XGBoost | **óptimo** | 0.20605 | **0.29769** | **0.28883** | **0.38804** | **0.20178** | **0.02845** | 1.40 | 0.344 |
| rent | GBR | base | 0.16852 | — | 0.29150 | 0.37669 | 0.21043 | 0.02978 | 1.73 | 0.444 |
| rent | GBR | óptimo | 0.19994 | 0.29690 | 0.29590 | 0.35774 | 0.21130 | 0.02986 | 1.48 | 0.390 |
| rent | AdaBoost | base | 0.31370 | — | 0.33883 | 0.15787 | 0.29327 | 0.04223 | 1.08 | 0.222 |
| rent | AdaBoost | óptimo | 0.11591 | 0.31359 | 0.29683 | 0.35367 | 0.19956 | 0.02818 | 2.56 | 0.562 |

---

## Análisis de los hiperparámetros óptimos

### Patrón consistente entre modelos y datasets

| Hiperparámetro | Sale óptimo | Rent óptimo | Conclusión |
|---------------|------------|------------|-----------|
| `max_depth` XGB/GBR | 3 / 5 | 3 / 3 | Árboles poco profundos casi siempre óptimos en boosting |
| `learning_rate` XGB/GBR | 0.05 / 0.05 | 0.05 / 0.05 | Shrinkage fuerte consistentemente mejor en ambos datasets |
| `subsample` XGB | 0.7 | 0.9 | Pequeña variación — ambos por debajo de 1.0 |
| `reg_lambda` XGB | 5 | 5 | Regularización L2 alta siempre óptima |
| `min_child_weight` XGB | 3 | 5 | Rent necesita hojas más pesadas (más conservador) |
| `colsample_bytree` XGB | 0.7 | 0.7 | Mismo valor en ambos datasets |
| `max_depth` AdaBoost | 7 | 7 | AdaBoost necesita árboles más profundos que GBR/XGB |
| `learning_rate` AdaBoost | 1.5 | 1.5 | Valor máximo del grid elegido en ambos datasets |
| `n_estimators` AdaBoost | 400 | 400 | Más iteraciones siempre mejor en AdaBoost |

> La convergencia de hiperparámetros entre datasets es una señal de robustez estructural: XGBoost con `max_depth=3`, `learning_rate=0.05` y `reg_lambda=5` es el régimen óptimo para este problema independientemente de si es venta o alquiler. Lo mismo para AdaBoost: `max_depth=7`, `learning_rate=1.5`, `n_estimators=400` es el máximo del grid en ambos casos — sugiere que el grid podría ampliarse.

### Por qué `max_depth=3` es óptimo para XGBoost y GBR

En boosting, los árboles individuales son **weak learners** — no se busca que cada árbol capture toda la señal. Con `max_depth=3`:
- Cada árbol puede capturar interacciones de hasta 3 variables
- Se necesitan más iteraciones para corregir el error residual acumulado
- El sesgo de cada árbol es alto pero la varianza es baja
- El ensemble acumula correcciones pequeñas y precisas → menor riesgo de overfitting

Con `max_depth` alto, cada árbol sobreajusta su corrección → el ensemble diverge en lugar de converger suavemente.

### Por qué AdaBoost necesita `max_depth=7`

AdaBoost no usa gradient descent sobre una función de pérdida diferenciable — reasigna pesos a las observaciones mal predichas en cada iteración. Con árboles poco profundos, AdaBoost converge lento y no alcanza la complejidad necesaria. Con `max_depth=7`, el árbol base puede capturar relaciones más ricas en cada iteración, compensando la ausencia de regularización explícita de AdaBoost (sin `reg_lambda`, sin `colsample_bytree`).

### Por qué GBR óptimo empeora al base (sale y rent)

El GridSearch de GBR selecciona `max_depth=5` en sale (vs `max_depth=3` del base por defecto), lo que aumenta el sobreajuste sin mejorar el test. Esto sugiere que el espacio de búsqueda de GBR no exploró suficientemente la región `max_depth ≤ 3` con `learning_rate < 0.05`. En rent ocurre algo similar aunque menos pronunciado. El base de GBR (con sus defaults conservadores) resulta ser una configuración difícil de superar con este tamaño de dataset.

---

## Comparación con todos los modelos del proyecto

### SALE — ranking completo

| Posición | Familia | Modelo | RMSE_test | R²_test | MAE_test |
|---------|---------|--------|----------|---------|---------|
| 🥇 1 | Bagging (nb 52) | Extra Trees óptimo | **0.2827** | **0.707** | ~0.206 |
| 🥈 2 | Bagging (nb 52) | RF óptimo | 0.3056 | 0.657 | ~0.212 |
| 🥉 3 | **Boosting (nb 53)** | **AdaBoost óptimo** | **0.3128** | **0.641** | 0.232 |
| 4 | Lineal (nb 51) | Ridge | 0.2997 | 0.638 | 0.229 |
| 5 | **Boosting (nb 53)** | GBR base | 0.3144 | 0.637 | 0.235 |
| 6 | **Boosting (nb 53)** | XGBoost óptimo | 0.3152 | 0.635 | 0.231 |
| 7 | Lineal (nb 51) | OLS Base | 0.3021 | 0.633 | 0.228 |
| 8 | **Boosting (nb 53)** | GBR óptimo | 0.3191 | 0.626 | 0.224 |
| 9 | Lineal (nb 51) | Lasso+OLS | 0.3146 | 0.602 | 0.236 |
| 10 | Bagging (nb 52) | RF Reg óptimo | 0.3285 | 0.604 | ~0.231 |
| 11 | **Boosting (nb 53)** | AdaBoost base | 0.3620 | 0.519 | 0.282 |

### RENT — ranking completo

| Posición | Familia | Modelo | RMSE_test | R²_test | MAE_test |
|---------|---------|--------|----------|---------|---------|
| 🥇 1 | Lineal (nb 51) | Lasso+OLS | **0.2133** | **0.576** | **0.165** |
| 🥈 2 | Lineal (nb 51) | OLS Base | 0.2161 | 0.564 | 0.167 |
| 🥉 3 | Lineal (nb 51) | Ridge | 0.2170 | 0.561 | 0.166 |
| 4 | Bagging (nb 52) | RF óptimo | 0.2739 | 0.450 | ~0.193 |
| 5 | Bagging (nb 52) | RF Reg óptimo | 0.2742 | 0.448 | ~0.192 |
| 6 | **Boosting (nb 53)** | **XGBoost óptimo** | **0.2888** | **0.388** | **0.202** |
| 7 | Bagging (nb 52) | RF Reg base | 0.2773 | 0.436 | ~0.195 |
| 8 | **Boosting (nb 53)** | GBR base | 0.2915 | 0.377 | 0.210 |
| 9 | **Boosting (nb 53)** | AdaBoost óptimo | 0.2968 | 0.354 | 0.200 |
| 10 | **Boosting (nb 53)** | GBR óptimo | 0.2959 | 0.358 | 0.211 |
| 11 | Bagging (nb 52) | Extra Trees óptimo | 0.2879 | 0.392 | ~0.202 |
| 12 | **Boosting (nb 53)** | AdaBoost base | 0.3388 | 0.158 | 0.293 |

**Conclusión de la comparativa:**

- **Sale:** Extra Trees (nb 52) sigue siendo el mejor modelo con amplia ventaja (R²=0.707). Los modelos boosting quedan en posiciones intermedias, comparables a los lineales pero sin superarlos claramente.
- **Rent:** los modelos lineales dominan (R²≈0.56–0.58). En alquiler, con n=477 y alta variabilidad idiosincrática, la regularización lineal es más eficiente que la complejidad de los ensembles. El boosting no supera al bagging ni a los lineales en rent.
- **Los modelos base de XGBoost son peligrosamente sobreajustados** — en ningún caso deben usarse sin ajuste de hiperparámetros.

---

## Interpretación del error en euros

| Dataset | Modelo | MAE_test (log) | Error mediano aprox. | Ejemplo concreto |
|---------|--------|---------------|----------------------|-----------------|
| Sale | AdaBoost óptimo | 0.2318 | e^0.232 − 1 ≈ **+26.1%** | 200.000€ → error ≈ 52.200€ |
| Sale | GBR base | 0.2348 | ≈ **+26.5%** | 200.000€ → error ≈ 53.000€ |
| Sale | XGBoost óptimo | 0.2311 | ≈ **+26.0%** | 200.000€ → error ≈ 52.000€ |
| Rent | XGBoost óptimo | 0.2018 | e^0.202 − 1 ≈ **+22.4%** | 1.000€/mes → error ≈ 224€ |
| Rent | GBR base | 0.2104 | ≈ **+23.4%** | 1.000€/mes → error ≈ 234€ |
| Rent | AdaBoost óptimo | 0.1996 | ≈ **+22.1%** | 1.000€/mes → error ≈ 221€ |

Los errores de boosting en sale son ligeramente superiores a los del mejor modelo lineal (Ridge, MAE=0.229) pero sin diferencia práctica relevante. En rent, los modelos lineales siguen siendo más precisos (MAE≈0.165 vs 0.200 de los mejores boosting) — diferencia de ~3.5 puntos porcentuales.

---

## Estructura del código

### Imports y configuración (`Cell 1`)

Librerías clave: `xgboost.XGBRegressor`, `sklearn.ensemble.GradientBoostingRegressor`, `sklearn.ensemble.AdaBoostRegressor`, `sklearn.tree.DecisionTreeRegressor`, `sklearn.model_selection.GridSearchCV`.

### Features y grids (`Cell 2`)

Define `BASE_FEATURES` (11 activas + 2 comentadas), `MUNICIPIO_FEATURES` (38 candidatos), y los 3 grids de hiperparámetros.

### Funciones auxiliares (`Cell 3`)

| Función | Descripción |
|---------|-------------|
| `get_metrics(y_real, y_pred)` | MSE, RMSE, MAE, R², MAPE — devuelve DataFrame de 1 fila |
| `collapse_rare_municipios(df, muni_cols, min_obs)` | Colapsa municipios con < `min_obs` obs en `municipio_otro` |
| `prepare_X(df, feature_cols)` | Selecciona features disponibles + imputación mediana |
| `plot_diagnostics(y_test, pred_test, title_prefix)` | 3 gráficos: Real vs Predicho, histograma de residuos, Q-Q plot (16×4) |
| `plot_feature_importance(importances, feature_names, title, top_n=15)` | Barplot horizontal top-N features (8×5) — devuelve DataFrame |
| `run_base_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name)` | Entrena, imprime métricas train/test y ratio sobreajuste — devuelve `(model, m_train, m_test)` |
| `run_grid_search(estimator, param_grid, X_train, X_test, y_train, y_test, model_name, dataset_name)` | GridSearchCV, imprime best params + métricas train/CV/test — devuelve `(best_model, gs, m_train, m_test, cv_rmse)` |

### Bucle principal (`Cell 4`)

Para cada dataset (`sale`, `rent`), en orden:
1. Carga y filtrado de nulos en target
2. `collapse_rare_municipios()` → determina features de municipio finales
3. `prepare_X()` → X con imputación mediana
4. `train_test_split(test_size=0.20, random_state=42)`
5. **XGBoost**: `run_base_model()` → plots → `run_grid_search()` → plots
6. **GBR**: `run_base_model()` → plots → `run_grid_search()` → plots
7. **AdaBoost**: `run_base_model()` → plots → `run_grid_search()` → plots
8. Resumen del dataset como DataFrame + acumulación en `all_summary_rows`
9. Al finalizar todos los datasets: resumen global `pd.DataFrame(all_summary_rows)`

---

## Decisiones de diseño importantes

| Decisión | Justificación |
|----------|--------------|
| **Sin eliminación de outliers** | Los árboles no asumen distribución gaussiana de los residuos ni linealidad — los outliers extremos son manejados como observaciones más difíciles de predecir |
| **Sin estandarización de features** | Los árboles son invariantes a la escala — escalar features no cambia los puntos de corte ni los resultados |
| **`score_cercania_servicios` comentado** | El boosting puede capturar interacciones no lineales directamente desde las 3 distancias individuales — el score compuesto es redundante |
| **`latitud`/`longitud` comentados** | Con n<600 y 25 features activas, añadir coordenadas incrementa la dimensionalidad sin mejora observada — las distancias son proxy suficiente |
| **Árbol base de AdaBoost en GridSearch** | `DecisionTreeRegressor()` sin restricciones adicionales — solo `max_depth` se controla vía `estimator__max_depth` |
| **`verbosity=0` en XGBRegressor** | Suprime el output verboso de XGBoost durante entrenamiento y GridSearchCV |
| **`scoring="neg_root_mean_squared_error"`** | Consistencia con las métricas de evaluación final (RMSE) |
| **`n_jobs=-1`** | Paralelización total en XGBoost y en todos los GridSearchCV |
| **`random_state=42`** | Reproducibilidad en train/test split, modelos estocásticos y GridSearchCV |

---

## Archivos relacionados

```
notebooks/05_ML/
├── 53_boost_def.ipynb     ← ESTE NOTEBOOK (definitivo, unificado)
├── 53_boost_1.ipynb       ← versión exploratoria inicial
└── 53_boost_2.ipynb       ← versión intermedia de desarrollo

data/gold/
├── final_sale.csv         ← 588 filas, 25 features efectivas en boosting
└── final_rent.csv         ← 477 filas, 21 features efectivas en boosting
```

---

## Modelo recomendado por dataset

| Dataset | Modelo recomendado | RMSE_test | R²_test | CV_RMSE | Razón |
|---------|--------------------|----------|---------|---------|-------|
| **Sale** | **AdaBoost óptimo** | 0.313 | 0.641 | 0.356 | Mejor R²_test y RMSE_test entre los boosting de sale. GBR base es la alternativa más robusta (menor sobreajuste) |
| **Rent** | **XGBoost óptimo** | 0.289 | 0.388 | 0.298 | Mejor en todas las métricas para rent. El CV_RMSE (0.298) es el más bajo de todos los ensembles en rent |

**Contexto general:** para rent, los modelos lineales del notebook 51 siguen siendo superiores (Lasso+OLS R²=0.576 vs 0.388 de XGBoost). El boosting aporta valor principalmente en sale, donde los no-linealismos son más capturables. En rent, con el tamaño muestral disponible (n=477), la regularización lineal generaliza mejor que la complejidad de los árboles.
