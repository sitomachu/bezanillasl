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

---

## Notebooks definitivos — Dataset API Idealista (versión final)

Los notebooks documentados en las secciones anteriores trabajan sobre los datasets combinados (`final_sale.csv`, `final_rent.csv`) con muestras de 588 y 477 observaciones respectivamente. Los notebooks definitivos de esta sección trabajan sobre los gold datasets exclusivos de la API de Idealista (`final_sale_idealistaAPI.csv`, `final_rent_idealistaAPI.csv`), con muestras significativamente mayores (2.543 y 662 observaciones tras limpieza) y feature engineering más extenso (48 y 27 features). Los resultados son sustancialmente mejores y estos notebooks constituyen el **modelo definitivo del proyecto**.

---

## Notebook: `53_boost_sale.ipynb`

Implementa XGBoost para la predicción de precios de **compra-venta** sobre el dataset API Idealista completo. Los hiperparámetros están fijados a partir del modelo óptimo obtenido en `53_boost_def_3.ipynb` — no se realiza búsqueda de hiperparámetros en este notebook, ya que los parámetros provienen de la experimentación previa. El objetivo es entrenar el modelo definitivo con el dataset más completo disponible y documentar su rendimiento final.

### Configuración global

```python
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TARGET_COL     = "log_precio"
IQR_FACTOR     = 1.5
CV_FOLDS       = 5
```

---

### Dataset y limpieza

Fuente: `data/gold/final_sale_idealistaAPI.csv`

| Paso | Criterio | Filas eliminadas | % | Razón |
|------|---------|-----------------|---|-------|
| Carga inicial | — | — | — | 2.694 filas, 71 columnas |
| Paso 1: IQR×1.5 sobre `log_precio` | Fuera de [Q1-1.5×IQR, Q3+1.5×IQR] | 0 | 0.0% | Rango válido: [53.914€, 1.550.605€] — engloba toda la muestra |
| Paso 2: Suelo `precio_m2 < 1.000 €/m²` | Propiedades anómalamente baratas por m² | 151 | 5.6% | Ruinas, errores de entrada o propiedades no residenciales — responsables de la cola inferior del Q-Q plot |
| **Filas tras limpieza** | | 151 | 5.6% | **2.543 filas** |

> El IQR sobre `log_precio` no elimina ningún registro en venta — el único filtro activo es el de `precio_m2`. El suelo de 1.000 €/m² actúa como umbral de coherencia: propiedades por debajo de ese valor son sistemáticamente anómalas (no reflejan el mercado residencial).

---

### Features

#### Grupos de features

| Categoría | Features |
|-----------|---------|
| **Físicas** | `superficie_construida_m2`, `numero_dormitorios`, `numero_banos`, `planta_num` |
| **Características** | `tiene_garaje`, `obra_nueva` |
| **Geoespaciales** | `distancia_min_playa_km`, `distancia_min_supermercado_km`, `distancia_min_colegio_km`, `distancia_centro_municipio_km`, `score_cercania_servicios` |
| **Mercado** | `precio_m2_municipio_media` |
| **Interacciones / ratios** | `ratio_dormitorios_superficie`, `ratio_banos_superficie`, `interaccion_planta_sin_ascensor_piso` |
| **Tipología** | `tipologia_unificada_piso`, `tipologia_unificada_unifamiliar` |
| **Municipio (OHE)** | 31 dummies del gold: `municipio_Ampuero`, `municipio_Barcena de Cicero`, `municipio_Camargo`, `municipio_Castro-Urdiales`, `municipio_Colindres`, `municipio_Cudon`, `municipio_El Astillero`, `municipio_Guarnizo`, `municipio_Laredo`, `municipio_Liendo`, `municipio_Limpias`, `municipio_Marina de Cudeyo`, `municipio_Miengo`, `municipio_Mogro`, `municipio_Noja`, `municipio_Ortuella`, `municipio_Piélagos`, `municipio_Polanco`, `municipio_Ribamontan al Mar`, `municipio_Ribamontan al Monte`, `municipio_Santa Cruz de Bezana`, `municipio_Santander`, `municipio_Santillana del Mar`, `municipio_Santoña`, `municipio_Santurtzi`, `municipio_Suances`, `municipio_Torrelavega`, `municipio_Villaescusa`, `municipio_Viveda`, `municipio_Voto`, `municipio_otro` |

**Total features activas: 48**

**Features comentadas (excluidas):** `latitud`, `longitud`, `es_exterior_piso`, `tiene_ascensor_piso`. Las coordenadas brutas añaden ruido sin mejora con este tamaño de muestra; `es_exterior_piso` y `tiene_ascensor_piso` son relevantes en alquiler pero no en venta.

#### Correlaciones relevantes con `log_precio` (documentadas en el notebook)

| Feature | Correlación |
|---------|------------|
| `superficie_construida_m2` | 0.6945 |
| `numero_banos` | 0.6228 |
| `numero_dormitorios` | 0.5500 |
| `precio_m2_municipio_media` | 0.1335 |

> `numero_banos` tiene la segunda correlación más alta con `log_precio` (0.623) — su alta importancia en el modelo es legítima, no un artefacto.

#### Preprocesado

Función `prepare_X()`: imputación por **mediana** de columna (`SimpleImputer(strategy="median")`). Sin estandarización — los árboles son invariantes a la escala.

---

### Split train/test

```
Train: 2.034 | Test: 509 | Features: 48
```

Proporción 80/20, `random_state=42`.

---

### Hiperparámetros XGBoost (hardcoded desde `53_boost_def_3`)

No hay búsqueda de hiperparámetros — los parámetros están fijados a partir de la experimentación previa.

```python
XGB_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_lambda       = 3,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
```

> `max_depth=4` (vs `max_depth=3` del GridSearch en `53_boost_def`) — un nivel más de profundidad, justificado por el mayor tamaño de muestra (2.034 vs 470 en train). `reg_lambda=3` mantiene regularización L2 moderada. `learning_rate=0.05` con 400 árboles garantiza convergencia gradual.

Se aplica validación cruzada sobre el conjunto de train para estimar el error antes de evaluar en test:

```
CV-RMSE (5-fold, train): 0.25111
```

---

### Resultados — SALE

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.02480 0.15747 0.11509 0.92387 0.00918
   CV     NaN 0.25111     NaN     NaN     NaN
 test 0.06033 0.24562 0.18063 0.82054 0.01451

Sobreajuste → ratio RMSE test/train: 1.5598 | delta R²: 0.1033
```

El modelo definitivo de venta alcanza **R²_test=0.820**, mejora de +17.9 puntos sobre el mejor resultado anterior (AdaBoost óptimo, R²=0.641 en `53_boost_def`). Esta mejora se explica principalmente por el mayor tamaño de muestra (2.034 vs 470 en train) y las features adicionales (`precio_m2_municipio_media`, ratios, interacciones). El sobreajuste es moderado (ratio=1.56, delta_R²=0.10) — significativamente menor que el AdaBoost óptimo del notebook anterior (ratio=2.90, delta_R²=0.32).

---

### Feature importances — SALE (top 10)

| Feature | Importancia | Nota |
|---------|------------|------|
| `superficie_construida_m2` | 0.2302 | Predictor dominante (23%) |
| `numero_banos` | 0.1162 | Segunda feature — correlación legítima alta |
| `interaccion_planta_sin_ascensor_piso` | 0.0780 | Penalización pisos altos sin ascensor |
| `tiene_garaje` | 0.0654 | Alta en venta vs alquiler |
| `numero_dormitorios` | 0.0617 | — |
| `precio_m2_municipio_media` | 0.0607 | Nivel de precio medio del municipio |
| `distancia_min_playa_km` | 0.0244 | Efecto costero en Cantabria |
| `municipio_Santander` | 0.0233 | Mayor mercado de la región |

> Con regularización correcta, `superficie_construida_m2` domina con un 23% — muy diferente del 13% del modelo anterior. La feature `interaccion_planta_sin_ascensor_piso` emerge como tercer predictor (7.8%), capturando un efecto estructural de depreciación. `precio_m2_municipio_media` aparece en el top-6, complementando las dummies de municipio con información continua del mercado zonal.

---

### Resumen — `53_boost_sale.ipynb`

| Métrica | Valor |
|---------|-------|
| Dataset | `final_sale_idealistaAPI.csv` |
| Filas tras limpieza | **2.543** |
| Train / Test | 2.034 / 509 |
| Features | **48** |
| Estrategia de tuning | Hiperparámetros fijos (procedentes de `53_boost_def_3`) |
| CV-RMSE (5-fold, train) | 0.25111 |
| **Test RMSE (log)** | **0.24562** |
| **Test R²** | **0.82054** |
| Test MAE (log) | 0.18063 |
| Error mediano aprox. | e^0.181 − 1 ≈ **+19.8%** |
| Ratio RMSE test/train | 1.56 |
| delta R² | 0.103 |

---

## Notebook: `53_boost_rent.ipynb`

Implementa XGBoost para la predicción de precios de **alquiler** sobre el dataset API Idealista, con limpieza específica para detectar alquileres vacacionales y búsqueda de hiperparámetros mediante **Optuna** (100 trials). Incluye comparación entre dos targets alternativos (`log_precio` vs `log_precio_m2`).

### Configuración global

```python
RANDOM_STATE                = 42
TEST_SIZE                   = 0.20
TARGET_COL                  = "log_precio"   # seleccionado tras comparación
CV_FOLDS                    = 5
IQR_FACTOR                  = 1.5
PRECIO_M2_VACACIONAL_UMBRAL = 18.0           # €/m²/mes
PRECIO_M2_MINIMO            = 6.0            # €/m²/mes
```

---

### Dataset y limpieza

Fuente: `data/gold/final_rent_idealistaAPI.csv`

El alquiler requiere **tres pasos** de limpieza en lugar de dos, dado que la distribución de `precio_m2` muestra una cola derecha con alquileres turísticos y una cola izquierda con anomalías baratas.

| Paso | Criterio | Filas eliminadas | % | Razón |
|------|---------|-----------------|---|-------|
| Carga inicial | — | — | — | 754 filas, 50 columnas |
| Paso 1: Vacacionales | `precio_m2 > 18.0 €/m²/mes` | 61 | 8.1% | Alquileres turísticos que distorsionan la señal del mercado residencial |
| Paso 2: Anomalías baratas | `precio_m2 < 6.0 €/m²/mes` | 12 | 1.7% | Garajes, errores de entrada o propiedades no residenciales |
| Paso 3: IQR×1.5 | Sobre `log_precio` | 19 | 2.8% | Outliers extremos de precio absoluto tras filtrar vacacionales |
| **Filas tras limpieza** | | 92 | 12.2% | **662 filas** |

> La detección de vacacionales por `precio_m2 > 18€/m²/mes` es la diferencia principal con el pipeline de venta. Sin este paso, el modelo sobreestima precios en zonas costeras (mezcla demanda turística + residencial). El umbral de 18€ separa la distribución bimodal observable en el histograma de `precio_m2`.

---

### Comparación de targets

Se comparan dos targets alternativos con XGBoost base (sin tuning) antes de lanzar Optuna:

| Target | CV-RMSE | CV-R² | Selección |
|--------|---------|-------|-----------|
| `log_precio` | **0.14070** | **0.67246** | ✓ Seleccionado |
| `log_precio_m2` | 0.14430 | 0.56896 | — |

> `log_precio` produce menor CV-RMSE y mayor CV-R² que `log_precio_m2`. Predecir el precio total es más preciso que predecir el precio por m² porque la superficie ya está capturada en las features. **Importante:** `precio_m2`, `precio` y `log_precio_m2` nunca entran como features — son derivadas del target y causarían data leakage.

---

### Features

| Categoría | Features |
|-----------|---------|
| **Físicas** | `superficie_construida_m2`, `numero_dormitorios`, `numero_banos`, `planta_num` |
| **Habitabilidad** | `es_exterior_piso`, `tiene_ascensor_piso` *(activos en rent; excluidos en sale)* |
| **Características** | `tiene_garaje`, `obra_nueva` |
| **Geoespaciales** | `distancia_min_playa_km`, `distancia_min_supermercado_km`, `distancia_min_colegio_km`, `distancia_centro_municipio_km`, `score_cercania_servicios` |
| **Mercado** | `precio_m2_municipio_media` |
| **Interacciones / ratios** | `ratio_dormitorios_superficie`, `ratio_banos_superficie`, `interaccion_planta_sin_ascensor_piso` |
| **Tipología** | `tipologia_unificada_piso`, `tipologia_unificada_unifamiliar` |
| **Municipio (OHE)** | 9 dummies: `municipio_Camargo`, `municipio_Castro-Urdiales`, `municipio_El Astillero`, `municipio_Laredo`, `municipio_Piélagos`, `municipio_Santa Cruz de Bezana`, `municipio_Santander`, `municipio_Suances`, `municipio_Torrelavega` + `municipio_otro` |

**Total features activas: 27**

> Rent incluye `es_exterior_piso` y `tiene_ascensor_piso` (excluidas en sale) — en alquiler la habitabilidad cotidiana importa más que en una decisión de compra. La cobertura municipal es menor (9 municipios + otro) porque muchos tienen < mínimo de observaciones en la muestra de alquiler.

---

### Split train/test

```
Train: 529 | Test: 133 | Features: 27
```

Proporción 80/20, `random_state=42`.

---

### Búsqueda de hiperparámetros con Optuna

A diferencia de `53_boost_sale`, en alquiler se usa **Optuna** para optimizar los hiperparámetros directamente sobre el dataset actual.

**Protocolo:**
- **Framework:** Optuna con `TPESampler` (por defecto)
- **Función objetivo:** minimizar CV-RMSE (5-fold) sobre el conjunto de train
- **El test no se toca durante la búsqueda** — evaluación holdout estricta
- **Número de trials:** 100

**Espacio de búsqueda:**

```python
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 1000, step=50),
        max_depth        = trial.suggest_int("max_depth", 3, 7),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
        gamma            = trial.suggest_float("gamma", 0.0, 1.0),
    )
    # CV 5-fold sobre train; score: neg. RMSE
```

> `learning_rate` se muestrea en escala logarítmica (`log=True`) para explorar con mayor densidad los valores bajos (0.01–0.05). `reg_alpha` añade regularización L1 sobre los pesos de las hojas, complementando a `reg_lambda` (L2). `gamma` es el gain mínimo de split para abrir un nodo — regulariza la estructura del árbol directamente.

**Resultado de la búsqueda:**

```
Best trial: 97.  Best value (CV-RMSE): 0.14622
100 trials completados
```

**Mejores hiperparámetros encontrados por Optuna:**

```python
BEST_PARAMS = {
    "n_estimators"    : 1000,
    "max_depth"       : 6,
    "learning_rate"   : 0.011737724486287017,
    "subsample"       : 0.6046283826235364,
    "colsample_bytree": 0.8531891359517659,
    "min_child_weight": 6,
    "reg_lambda"      : 3.378990393436524,
    "reg_alpha"       : 0.06467616791725625,
    "gamma"           : 0.03464694507245766,
}
```

> Optuna selecciona `max_depth=6` (más profundo que venta) y `n_estimators=1000` con `learning_rate≈0.012` (muy bajo). Esta combinación de learning rate bajo + muchos árboles + profundidad alta es característica de datasets pequeños: el modelo necesita más iteraciones para acumular correcciones graduales. `min_child_weight=6` (vs 3 en sale) exige hojas con más observaciones — el modelo es más conservador dado el tamaño reducido.

---

### Resultados — RENT

```
Target seleccionado: log_precio

split     MSE    RMSE     MAE      R2    MAPE
train 0.00864 0.09295 0.07143 0.86194 0.01043
   CV     NaN 0.14622     NaN     NaN     NaN
 test 0.02096 0.14478 0.11452 0.61790 0.01677

Sobreajuste → ratio RMSE test/train: 1.5576 | delta R²: 0.2440
```

El modelo definitivo de alquiler alcanza **R²_test=0.618** y **RMSE_test=0.145**. Respecto al mejor resultado anterior (XGBoost óptimo GridSearch en `53_boost_def`, R²=0.388), la mejora es de **+23.0 puntos** — la mayor mejora absoluta de cualquier modelo en cualquier dataset del proyecto. El sobreajuste es moderado y comparable al de venta (ratio≈1.56 en ambos casos).

> Los modelos lineales del notebook 51 obtenían R²=0.576 (Lasso+OLS) con el dataset combinado. El XGBoost definitivo con el dataset API-only obtiene R²=0.618 — los modelos de boosting superan finalmente a los lineales en alquiler cuando se dispone de más datos y mejor feature engineering.

---

### Feature importances — RENT (top 10)

| Feature | Importancia | Nota |
|---------|------------|------|
| `superficie_construida_m2` | 0.1803 | Predictor dominante |
| `numero_dormitorios` | 0.1142 | Alta relevancia en alquiler |
| `numero_banos` | 0.0800 | — |
| `precio_m2_municipio_media` | 0.0684 | Nivel de precio medio del municipio |
| `municipio_Camargo` | 0.0582 | Mercado diferencial Camargo |
| `municipio_Santander` | 0.0510 | Capital de provincia |
| `municipio_otro` | 0.0422 | Municipios con baja representación |
| `distancia_min_playa_km` | 0.0343 | Efecto costero en alquiler |

> En alquiler, `superficie_construida_m2` sigue siendo la feature más importante (18.0%) pero no domina como en venta. `numero_dormitorios` (11.4%) tiene más peso relativo que en sale — en alquiler la configuración de habitaciones es un criterio de búsqueda primario. Los municipios tienen importancias más equilibradas que en `53_boost_def` — señal de que la regularización de Optuna evita la memorización de patrones municipales.

---

### Resumen — `53_boost_rent.ipynb`

| Métrica | Valor |
|---------|-------|
| Dataset | `final_rent_idealistaAPI.csv` |
| Filas tras limpieza | **662** |
| Train / Test | 529 / 133 |
| Features | **27** |
| Estrategia de tuning | Optuna (100 trials, TPESampler, CV-RMSE 5-fold) |
| Best trial | 97 |
| CV-RMSE (Optuna, train) | 0.14622 |
| **Test RMSE (log)** | **0.14478** |
| **Test R²** | **0.61790** |
| Test MAE (log) | 0.11452 |
| Error mediano aprox. | e^0.115 − 1 ≈ **+12.2%** |
| Ratio RMSE test/train | 1.56 |
| delta R² | 0.244 |

---

## Notebook: `55_sale_rent_models.ipynb`

Notebook que centraliza y replica los modelos definitivos de `53_boost_sale` y `53_boost_rent` en un único lugar, documentando los hiperparámetros exportables y sirviendo como referencia consolidada de producción.

### Descripción de modelos integrados

| Aspecto | Sale | Rent |
|---------|------|------|
| Dataset | `final_sale_idealistaAPI.csv` | `final_rent_idealistaAPI.csv` |
| Hiperparámetros | Hardcodeados (de `53_boost_def_3`) | Optuna — transferidos de `53_boost_rent` |
| Target | `log_precio` | `log_precio` |
| Features | 48 | 27 |
| Limpieza | IQR×1.5 + suelo `precio_m2 ≥ 1.000 €/m²` | Vacacionales (`>18 €/m²`) + baratos (`<6 €/m²`) + IQR×1.5 |
| Train | 2.034 | 529 |
| Test | 509 | 133 |

### Hiperparámetros documentados

**Sale — XGBoost (hardcoded):**
```python
XGB_PARAMS_SALE = {
    "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_lambda": 3, "random_state": 42
}
```

**Rent — XGBoost (Optuna):**
```python
XGB_PARAMS_RENT = {
    "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.011737724486287017,
    "subsample": 0.6046283826235364, "colsample_bytree": 0.8531891359517659,
    "min_child_weight": 6, "reg_lambda": 3.378990393436524,
    "reg_alpha": 0.06467616791725625, "gamma": 0.03464694507245766,
    "random_state": 42
}
```

### Resultados consolidados

| Dataset | Filas tras limpieza | Features | CV-RMSE | Test RMSE | Test R² | Error aprox. |
|---------|---------------------|----------|---------|----------|---------|--------------|
| Sale | 2.543 | 48 | 0.25111 | **0.24562** | **0.82054** | ±27.8% |
| Rent | 662 | 27 | 0.14622 | **0.14478** | **0.61790** | ±15.6% |

> El error aproximado se calcula como e^RMSE − 1 en escala logarítmica — es el margen de error porcentual medio. Para una vivienda de 200.000€: error de venta ≈ ±55.600€. Para 1.000€/mes de alquiler: error ≈ ±156€/mes.

### Funciones auxiliares

| Función | Descripción |
|---------|-------------|
| `get_metrics(y_real, y_pred)` | MSE, RMSE, MAE, R², MAPE — devuelve DataFrame de 1 fila |
| `remove_outliers_iqr(df)` | IQR×1.5 sobre `log_precio` — devuelve df filtrado |
| `build_X(df, base_features)` | Construye X: features base + municipio_* OHE + imputación mediana |
| `build_geo_ref(df)` | Mediana de variables geográficas por municipio (para el estimador) |

---

## Notebook: `55_input_result.ipynb`

Estimador de precio interactivo que combina los modelos definitivos (sale + rent) para obtener, dado un conjunto de características de una vivienda, el **precio de venta estimado**, el **alquiler mensual estimado** y la **rentabilidad bruta**.

### Protocolo de entrenamiento

Los modelos se re-entrenan al completo dentro del notebook con los mismos parámetros que `55_sale_rent_models.ipynb`:
- Split 80/20 para evaluación holdout reproducible
- Sale: 2.034 train + 509 test | Rent: 529 train + 133 test

```
Sale  — test RMSE (log): 0.2456  →  ±27.8%
Rent  — test RMSE (log): 0.1448  →  ±15.6%
```

### Cobertura geográfica

El estimador construye una **referencia geográfica** por municipio (mediana de distancias POI y `precio_m2_municipio_media`) a partir del propio dataset limpio.

| Modelo | Municipios disponibles (n) |
|--------|--------------------------|
| **Sale** | 31: Ampuero, Barcena de Cicero, Camargo, Castro-Urdiales, Colindres, Cudon, El Astillero, Guarnizo, Laredo, Liendo, Limpias, Marina de Cudeyo, Miengo, Mogro, Noja, Ortuella, Piélagos, Polanco, Ribamontan al Mar, Ribamontan al Monte, Santa Cruz de Bezana, Santander, Santillana del Mar, Santoña, Santurtzi, Suances, Torrelavega, Villaescusa, Viveda, Voto, otro |
| **Rent** | 10: Camargo, Castro-Urdiales, El Astillero, Laredo, Piélagos, Santa Cruz de Bezana, Santander, Suances, Torrelavega, otro |

### Input del estimador

El usuario configura los atributos de la vivienda en una celda:

```python
MUNICIPIO      = "Santa Cruz de Bezana"   # nombre exacto de la lista anterior
SUPERFICIE_M2  = 90
DORMITORIOS    = 3
BANOS          = 2
PLANTA         = 1
TIPO           = "piso"          # "piso" | "unifamiliar"
ES_EXTERIOR    = True
TIENE_ASCENSOR = True
TIENE_GARAJE   = True
OBRA_NUEVA     = True
```

### Ejemplo de estimación

```
══════════════════════════════════════════════════════════
  90 m²  ·  3 dorm.  ·  2 baños  —  Santa Cruz de Bezana
  PISO  ·  Planta 1 · exterior · con ascensor · garaje · obra nueva
══════════════════════════════════════════════════════════

  Precio de venta estimado   :      314.381 €
  Intervalo error (±1σ)      : [   245.914 €  —     401.910 €]  (±28%)

  Alquiler mensual estimado  :        1.039 €/mes
  Intervalo error (±1σ)      : [       899 €/mes  —       1.201 €/mes]  (±16%)

  Rentabilidad bruta estim.  :         4.0 %
```

> El intervalo de error se calcula como `precio × (e^RMSE − 1)` usando el RMSE del modelo en escala logarítmica.

---

## Notebook: `55_input_result_no_k_fold.ipynb`

Variante de `55_input_result` que entrena los modelos sobre el **100% de las observaciones** (sin reservar test) para maximizar la información disponible en la predicción.

### Diferencia clave con `55_input_result`

| Aspecto | `55_input_result` | `55_input_result_no_k_fold` |
|---------|-----------------|--------------------------|
| Split | 80/20 (train/test) | Sin split — 100% de datos para entrenar |
| Evaluación interna | Test RMSE (holdout) | CV-RMSE (5-fold, from 53_boost notebooks) |
| Uso recomendado | Auditoría / comparación de modelos | Predicción de viviendas reales (maximiza datos) |

### CV-RMSE de referencia (transferidos desde notebooks 53)

| Modelo | Observaciones | Features | CV-RMSE | Error aprox. |
|--------|--------------|----------|---------|--------------|
| Sale (100% datos) | 2.543 | 48 | **0.25922** | ±29.6% |
| Rent (100% datos) | 662 | 27 | **0.14622** | ±16.8% |

### Ejemplo de estimación (misma vivienda)

```
══════════════════════════════════════════════════════════
  90 m²  ·  3 dorm.  ·  2 baños  —  Santa Cruz de Bezana
  PISO  ·  Planta 1 · exterior · con ascensor · garaje · obra nueva
══════════════════════════════════════════════════════════

  Precio de venta estimado   :      317.039 €
  Intervalo error (±1σ)      : [   244.644 €  —     410.857 €]  (±30%)

  Alquiler mensual estimado  :          994 €/mes
  Intervalo error (±1σ)      : [       859 €/mes  —       1.151 €/mes]  (±16%)

  Rentabilidad bruta estim.  :         3.8 %
```

> La estimación varía ligeramente respecto al modelo con split (314.381€ vs 317.039€ en venta) — el modelo entrenado sobre el 100% de los datos ajusta de forma marginalmente diferente. El intervalo es algo más amplio (±30% vs ±28%) al usar CV-RMSE en lugar de test RMSE.

---

## Resumen global definitivo — Modelos boosting finales

### Evolución de resultados: `53_boost_def` → modelos definitivos API

| Dataset | Notebook | Gold dataset | n_train | Mejor modelo | R²_test | RMSE_test | Mejora R² |
|---------|---------|-------------|---------|-------------|---------|----------|-----------|
| Sale | `53_boost_def.ipynb` | `final_sale.csv` (588 obs.) | 470 | AdaBoost óptimo | 0.641 | 0.313 | — |
| Sale | **`53_boost_sale.ipynb`** | `final_sale_idealistaAPI.csv` (2.543 obs.) | 2.034 | **XGBoost** | **0.821** | **0.246** | **+17.9pp** |
| Rent | `53_boost_def.ipynb` | `final_rent.csv` (477 obs.) | 381 | XGBoost óptimo | 0.388 | 0.289 | — |
| Rent | **`53_boost_rent.ipynb`** | `final_rent_idealistaAPI.csv` (662 obs.) | 529 | **XGBoost** | **0.618** | **0.145** | **+23.0pp** |

### Modelo definitivo recomendado (versión final)

| Dataset | Modelo | Tuning | Test RMSE | Test R² | CV-RMSE | Error en euros |
|---------|--------|--------|----------|---------|---------|----------------|
| **Sale** | XGBoost | Hardcoded | **0.2456** | **0.8205** | 0.2511 | ±27.8% (≈±49k€ para 175k€) |
| **Rent** | XGBoost | Optuna (100 trials) | **0.1448** | **0.6179** | 0.1462 | ±15.6% (≈±156€ para 1.000€/mes) |

### Posición en el ranking global del proyecto

**Sale — XGBoost definitivo (R²=0.821):**
- Supera al anterior líder global Extra Trees (R²=0.707 en notebook 52) por +11.4 puntos.
- Es el **mejor modelo del proyecto para venta**.
- Factor principal: muestra 4.3× más grande en train (2.034 vs 470) + features adicionales de mercado y de interacción.

**Rent — XGBoost definitivo (R²=0.618):**
- Supera a los modelos lineales anteriores (Lasso+OLS, R²=0.576) por +4.2 puntos.
- **Primera vez que un modelo no-lineal supera a los lineales en alquiler** en este proyecto.
- Factor principal: muestra mayor (529 vs 381 en train) + eliminación de vacacionales + feature `precio_m2_municipio_media`.

### Análisis comparativo de hiperparámetros: sale vs rent (modelos definitivos)

| Hiperparámetro | Sale | Rent | Interpretación |
|---------------|------|------|----------------|
| `n_estimators` | 400 | 1000 | Rent necesita más árboles con learning rate más bajo |
| `max_depth` | 4 | 6 | Rent permite árboles más profundos — más observaciones por árbol relativas al espacio de features |
| `learning_rate` | 0.050 | 0.0117 | Shrinkage mucho más agresivo en rent — compensado por más árboles |
| `subsample` | 0.8 | 0.60 | Rent muestrea menos filas por árbol — más conservador con n pequeño |
| `colsample_bytree` | 0.8 | 0.85 | Similar en ambos |
| `min_child_weight` | 3 | 6 | Rent exige hojas más densas — regularización más fuerte |
| `reg_lambda` | 3.0 | 3.38 | L2 similar en ambos — parámetro robusto |
| `reg_alpha` | — | 0.065 | L1 solo en rent (Optuna lo encuentra útil) |
| `gamma` | — | 0.035 | Split gain mínimo solo en rent (Optuna) |

### Decisiones de diseño específicas de los notebooks definitivos

| Decisión | Notebook | Justificación |
|----------|---------|--------------|
| Hiperparámetros fijos en sale | `53_boost_sale` | Los params de `53_boost_def_3` ya fueron optimizados — re-buscar con GridSearch en 2.034 obs. sería costoso sin garantía de mejora |
| Optuna en rent | `53_boost_rent` | El dataset de alquiler con limpieza de vacacionales es diferente al usado en `53_boost_def` — justifica re-optimizar desde cero |
| Filtro `precio_m2 < 1.000 €/m²` en sale | `53_boost_sale` | Detecta ruinas y errores que el IQR sobre `log_precio` no elimina (precios bajos con superficies altas que distorsionan el Q-Q plot) |
| Filtro vacacionales `precio_m2 > 18 €/m²/mes` | `53_boost_rent` | Sin este paso el modelo mezcla demanda turística y residencial — umbral que separa la distribución bimodal |
| Todas las OHE del gold en sale | `53_boost_sale` | El gold API tiene 30 municipios con representación suficiente — se usan directamente sin colapsar |
| `precio_m2_municipio_media` en ambos | Ambos | Feature continua de mercado que complementa las dummies OHE, capturando el nivel de precio zonal de forma suave |
| `es_exterior_piso`/`tiene_ascensor_piso` en rent | `53_boost_rent` | Relevantes para arrendatarios (habitabilidad cotidiana) pero no para compradores (decisión a largo plazo) |
