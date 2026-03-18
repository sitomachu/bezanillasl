# Documentación: Modelos de Bagging / Random Forest — BezanillaSL

## Contexto del proyecto

Continuación del trabajo de predicción de precios de inmuebles en **Cantabria** (España). Se usa la misma variable objetivo `log_precio` (logaritmo natural del precio en euros) y los mismos datasets gold que en el notebook de regresión lineal. Los modelos de este notebook son el segundo bloque de ML del proyecto, tras establecer la línea base con regresión lineal.

Datasets en `data/gold/`:
- **sale** (`final_sale.csv`): inmuebles en venta
- **rent** (`final_rent.csv`): inmuebles en alquiler

---

## Notebook: `52_random_forest_def.ipynb`

Implementa, evalúa y compara **cuatro variantes de modelos basados en árboles con bagging** aplicadas a ambos datasets. Para cada variante se ejecuta primero un modelo base con hiperparámetros por defecto (o fijos), y luego una búsqueda exhaustiva de hiperparámetros con `GridSearchCV`.

| Variante | Descripción | Grid |
|---------|-------------|------|
| **Random Forest (RF)** | Modelo de referencia: bagging de árboles con splits optimizados | 54 combinaciones |
| **Extra Trees (ET)** | Bagging con splits totalmente aleatorios — mayor decorrelación | 54 combinaciones |
| **RF Regularizado (RRF)** | RF con poda por complejidad de coste (`ccp_alpha`) | 108 combinaciones |
| **Quantile RF (QRF)** | RF con predicción de cuantiles — intervalos de confianza | 54 combinaciones |

> **Nota:** Quantile RF requiere `pip install quantile-forest`. En la ejecución documentada el paquete **no estaba instalado** (`QRF_AVAILABLE = False`), por lo que todos los resultados QRF fueron omitidos.

---

## Configuración global

```python
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
TARGET_COL        = "log_precio"
CV_FOLDS          = 5
MIN_OBS_MUNICIPIO = 10
```

No hay eliminación de outliers en este notebook — se trabaja con todas las filas donde `log_precio` no es nulo (a diferencia del notebook 51 donde se aplicaba IQR). El split es 80/20 con `random_state=42`.

---

## Features

### Features base (13)

```python
BASE_FEATURES = [
    "superficie_construida_m2",   # sin log (diferencia con el OLS base del nb 51)
    "numero_dormitorios",
    "numero_banos",
    # "latitud",   ← comentada
    # "longitud",  ← comentada
    "tiene_garaje",
    "obra_nueva",
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
    "distancia_centro_municipio_km",
    "score_cercania_servicios",
    "tipologia_unificada_piso",
    "tipologia_unificada_unifamiliar",
]
```

Latitud y longitud están **comentadas** — se decidió no incluirlas en este notebook. El resto de las distancias y el score de cercanía se mantienen. La superficie se usa en su escala natural (sin logaritmo), ya que los árboles no requieren transformaciones.

### Features de municipio

Se parte de una lista larga de 38 dummies de municipio (`municipio_*`) que cubren todos los municipios de Cantabria y País Vasco presentes en el dataset. Antes del entrenamiento, la función `collapse_rare_municipios` agrupa las dummies con menos de 10 observaciones en `municipio_otro`.

**Sale** (venta):
- 3 municipios no presentes en el dataset: `municipio_Sobremazas`, `municipio_Villaescusa`, `municipio_Viveda`
- 13 municipios con ≥ 10 obs (se mantienen): `Camargo`, `Castro-Urdiales`, `Laredo`, `Noja`, `Piélagos`, `Polanco`, `Santa Cruz de Bezana`, `Santander`, `Santoña`, `Santurtzi`, `Suances`, `Torrelavega`, `Voto`
- 22 municipios con < 10 obs → `municipio_otro`
- **Features totales: 13 base + 13 municipio + 1 otro = 26**

**Rent** (alquiler):
- 9 municipios con ≥ 10 obs: `Camargo`, `Castro-Urdiales`, `El Astillero`, `Laredo`, `Piélagos`, `Santa Cruz de Bezana`, `Santander`, `Torrelavega` + uno más
- 29 municipios con < 10 obs → `municipio_otro`
- **Features totales: 13 base + 9 municipio + 1 otro = 22**

> El alquiler está más concentrado geográficamente (9 vs 13 municipios con suficientes datos), lo que es consistente con que el mercado de alquiler es menos disperso que el de compraventa.

### Preprocesado de features (`prepare_X`)

Los árboles no requieren escalado. Solo se aplica **imputación por mediana** (`SimpleImputer(strategy="median")`) para manejar valores nulos. Las columnas que no existen en el DataFrame simplemente se descartan.

---

## Datos de entrada

### Dataset SALE (venta)

| Métrica | Valor |
|---------|-------|
| Filas cargadas | 588 (todas con `log_precio` no nulo) |
| Outliers eliminados | 0 (no se aplica IQR en este notebook) |
| Features | 26 |
| Train | 470 |
| Test | 118 |

> Diferencia respecto al nb 51: 588 vs 582 filas — en el nb 51 se eliminaban 6 outliers con IQR, aquí no.

### Dataset RENT (alquiler)

| Métrica | Valor |
|---------|-------|
| Filas cargadas | 477 |
| Outliers eliminados | 0 |
| Features | 22 |
| Train | 381 |
| Test | 96 |

---

## Grids de hiperparámetros

### Random Forest

```python
PARAM_GRID_RF = {
    "n_estimators":     [200, 400],
    "max_depth":        [10, 20, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features":     ["sqrt", 0.4, 0.6],
}
# 2 × 3 × 3 × 3 = 54 combinaciones
```

### Extra Trees

```python
PARAM_GRID_ET = {
    "n_estimators":     [200, 400],
    "max_depth":        [10, 20, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features":     ["sqrt", 0.4, 0.6],
}
# 54 combinaciones — mismo grid que RF para poder comparar directamente
```

### RF Regularizado

```python
PARAM_GRID_RRF = {
    "n_estimators":     [200, 400],
    "max_depth":        [10, 15, 20],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     ["sqrt", 0.4],
    "ccp_alpha":        [0.0001, 0.001, 0.01],
}
# 2 × 3 × 3 × 2 × 3 = 108 combinaciones
```

`max_depth=None` está excluido del grid regularizado (no tiene sentido combinar `ccp_alpha` con árboles de profundidad ilimitada). El grid está sesgado hacia configuraciones más restringidas: `min_samples_leaf` empieza en 5 en lugar de 1.

### Quantile RF

```python
PARAM_GRID_QRF = {
    "n_estimators":     [200, 400],
    "max_depth":        [10, 20, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     ["sqrt", 0.4, 0.6],
}
# 54 combinaciones — min_samples_leaf empieza en 5 (los QRF son más lentos)
```

---

## Modelo base para RF Regularizado

Antes del GridSearch, se prueba un modelo base fijo:

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    ccp_alpha=0.001,
    random_state=42,
    n_jobs=-1,
)
```

Este punto de partida manual representa una configuración "moderadamente restrictiva" que se usa para tener una referencia antes de buscar el óptimo.

---

## Resultados — Dataset SALE (venta)

### Random Forest — SALE

**Modelo base** (hiperparámetros por defecto de sklearn):

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.01735 0.13174 0.09102 0.94688 0.00742
 test 0.10952 0.33094 0.23481 0.59774 0.01916

Sobreajuste → ratio RMSE test/train: 2.5121 | delta R2: 0.3491
```

**GridSearch óptimo:**

```
Mejores params: {max_depth: None, max_features: sqrt, min_samples_leaf: 1, n_estimators: 400}
CV RMSE (mejor): 0.35348

split     MSE    RMSE     MAE      R2    MAPE
train 0.01527 0.12357 0.08767 0.95326 0.00715
   CV     NaN 0.35348     NaN     NaN     NaN
 test 0.09340 0.30561 0.21158 0.65697 0.01725

Sobreajuste → ratio RMSE test/train: 2.4732 | delta R2: 0.2963
```

> El GridSearch selecciona `max_depth=None` y `min_samples_leaf=1` — los árboles crecen sin restricción. El CV RMSE (0.353) es mayor que el test RMSE (0.306), confirmando sobreajuste. El modelo mejora notablemente sobre la base: R² test 0.597 → 0.657.

### Extra Trees — SALE

**Modelo base:**

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.00002 0.00456 0.00086 0.99994 0.00007
 test 0.08224 0.28677 0.20582 0.69796 0.01676

Sobreajuste → ratio RMSE test/train: 62.8882 | delta R2: 0.3020
```

**GridSearch óptimo:**

```
Mejores params: {max_depth: 20, max_features: 0.4, min_samples_leaf: 1, n_estimators: 400}
CV RMSE (mejor): 0.33306

split     MSE    RMSE     MAE      R2    MAPE
train 0.00006 0.00806 0.00319 0.99980 0.00026
   CV     NaN 0.33306     NaN     NaN     NaN
 test 0.07991 0.28268 0.20556 0.70651 0.01674

Sobreajuste → ratio RMSE test/train: 35.0720 | delta R2: 0.2933
```

> El ratio RMSE test/train = 62.9 en el modelo base es un número llamativo. El train RMSE es 0.005 y el test RMSE es 0.287 — los árboles memorizan perfectamente el train pero el ensemble promediado generaliza bien (R² test = 0.698). El GridSearch mejora ligeramente hasta R² test = 0.707 con `max_depth=20` y `max_features=0.4`.

### RF Regularizado — SALE

**Modelo base** (`ccp_alpha=0.001`, `max_depth=15`, `min_samples_leaf=5`):

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.06731 0.25944 0.18516 0.79396 0.01509
 test 0.10996 0.33160 0.23586 0.59614 0.01919

Sobreajuste → ratio RMSE test/train: 1.2781 | delta R2: 0.1978
```

**GridSearch óptimo:**

```
Mejores params: {ccp_alpha: 0.0001, max_depth: 15, max_features: 0.4, min_samples_leaf: 5, n_estimators: 200}
CV RMSE (mejor): 0.36225

split     MSE    RMSE     MAE      R2    MAPE
train 0.06829 0.26132 0.18447 0.79097 0.01505
   CV     NaN 0.36225     NaN     NaN     NaN
 test 0.10791 0.32849 0.23149 0.60368 0.01883

Sobreajuste → ratio RMSE test/train: 1.2570 | delta R2: 0.1873
```

> El `ccp_alpha` óptimo es 0.0001 (el valor más bajo del grid) — la poda agresiva no ayuda. El ratio RMSE test/train cae a 1.26 (vs 2.47 en RF estándar), pero a costa de peor R² test (0.60 vs 0.66). La regularización reduce el sobreajuste visible pero no mejora la generalización efectiva.

---

## Resultados — Dataset RENT (alquiler)

### Random Forest — RENT

**Modelo base:**

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.01389 0.11785 0.08215 0.91243 0.01158
 test 0.08021 0.28321 0.20445 0.41163 0.02900

Sobreajuste → ratio RMSE test/train: 2.4031 | delta R2: 0.5008
```

**GridSearch óptimo:**

```
Mejores params: {max_depth: 10, max_features: 0.6, min_samples_leaf: 5, n_estimators: 400}
CV RMSE (mejor): 0.29789

split     MSE    RMSE     MAE      R2    MAPE
train 0.05045 0.22462 0.15440 0.68189 0.02170
   CV     NaN 0.29789     NaN     NaN     NaN
 test 0.07501 0.27388 0.19304 0.44976 0.02729

Sobreajuste → ratio RMSE test/train: 1.2193 | delta R2: 0.2321
```

> El GridSearch selecciona `max_depth=10` y `min_samples_leaf=5` para rent — configuración mucho más restrictiva que para sale (`max_depth=None`, `min_samples_leaf=1`). El CV RMSE (0.298) está muy cerca del test RMSE (0.274), indicando que el modelo está bien calibrado. El R² test sube de 0.412 a 0.450.

### Extra Trees — RENT

**Modelo base:**

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.00074 0.02719 0.00257 0.99534 0.00034
 test 0.10736 0.32765 0.22325 0.21248 0.03166

Sobreajuste → ratio RMSE test/train: 12.0504 | delta R2: 0.7829
```

**GridSearch óptimo:**

```
Mejores params: {max_depth: 20, max_features: 0.6, min_samples_leaf: 5, n_estimators: 400}
CV RMSE (mejor): 0.29248

split     MSE    RMSE     MAE      R2    MAPE
train 0.05726 0.23929 0.16463 0.63897 0.02318
   CV     NaN 0.29248     NaN     NaN     NaN
 test 0.08290 0.28792 0.20241 0.39191 0.02864

Sobreajuste → ratio RMSE test/train: 1.2032 | delta R2: 0.2471
```

> El ET base en rent es dramático: R² train = 0.995 pero R² test = 0.212 (delta = 0.783). En rent el ensemble base de ET no generaliza bien. El GridSearch corrige completamente el problema seleccionando `min_samples_leaf=5` — el R² test sube a 0.392.

### RF Regularizado — RENT

**Modelo base** (`ccp_alpha=0.001`, `max_depth=15`, `min_samples_leaf=5`):

```
split     MSE    RMSE     MAE      R2    MAPE
train 0.05407 0.23254 0.16494 0.65905 0.02325
 test 0.07691 0.27733 0.19473 0.43582 0.02750

Sobreajuste → ratio RMSE test/train: 1.1926 | delta R2: 0.2232
```

**GridSearch óptimo:**

```
Mejores params: {ccp_alpha: 0.0001, max_depth: 10, max_features: 0.4, min_samples_leaf: 5, n_estimators: 400}
CV RMSE (mejor): 0.30049

split     MSE    RMSE     MAE      R2    MAPE
train 0.05443 0.23329 0.16172 0.65684 0.02272
   CV     NaN 0.30049     NaN     NaN     NaN
 test 0.07520 0.27423 0.19247 0.44835 0.02722

Sobreajuste → ratio RMSE test/train: 1.1755 | delta R2: 0.2085
```

> Para rent, el RF Regularizado base (fijo sin búsqueda) ya es bastante bueno (R² test 0.436) y el GridSearch solo mejora marginalmente a 0.448. El `ccp_alpha` óptimo es de nuevo el más bajo (0.0001), lo que sugiere que la poda agresiva siempre empeora los resultados.

---

## Resumen comparativo global

### SALE (venta)

| Modelo | Fase | RMSE_train | CV_RMSE | RMSE_test | R²_test | MAE_test | ratio RMSE | delta R² |
|--------|------|-----------|---------|----------|---------|---------|-----------|---------|
| RF | base | 0.13174 | — | 0.33094 | 0.59774 | 0.23481 | 2.51 | 0.349 |
| RF | optimo | 0.12357 | 0.35348 | 0.30561 | 0.65697 | 0.21158 | 2.47 | 0.296 |
| ExtraTrees | base | 0.00456 | — | 0.28677 | 0.69796 | 0.20582 | 62.9 | 0.302 |
| **ExtraTrees** | **optimo** | **0.00806** | **0.33306** | **0.28268** | **0.70651** | **0.20556** | 35.1 | 0.293 |
| RF_Reg | base | 0.25944 | — | 0.33160 | 0.59614 | 0.23586 | 1.28 | 0.198 |
| RF_Reg | optimo | 0.26132 | 0.36225 | 0.32849 | 0.60368 | 0.23149 | 1.26 | 0.187 |

### RENT (alquiler)

| Modelo | Fase | RMSE_train | CV_RMSE | RMSE_test | R²_test | MAE_test | ratio RMSE | delta R² |
|--------|------|-----------|---------|----------|---------|---------|-----------|---------|
| **RF** | **optimo** | **0.22462** | **0.29789** | **0.27388** | **0.44976** | **0.19304** | **1.22** | **0.232** |
| ExtraTrees | base | 0.02719 | — | 0.32765 | 0.21248 | 0.22325 | 12.1 | 0.783 |
| ExtraTrees | optimo | 0.23929 | 0.29248 | 0.28792 | 0.39191 | 0.20241 | 1.20 | 0.247 |
| RF_Reg | base | 0.23254 | — | 0.27733 | 0.43582 | 0.19473 | 1.19 | 0.223 |
| RF_Reg | optimo | 0.23329 | 0.30049 | 0.27423 | 0.44835 | 0.19247 | 1.18 | 0.209 |

---

## Análisis del fenómeno Extra Trees: R²≈0.99 en train

Este es el análisis más importante del notebook. El comportamiento de ET con `min_samples_leaf=1` genera una discrepancia extrema entre train y test que se estudia en detalle con 4 experimentos.

### ¿Por qué ET memoriza el train perfectamente?

`ExtraTreesRegressor` difiere de RF en un aspecto clave: **los puntos de corte de cada split se eligen completamente al azar** (RF busca el mejor split entre candidatos aleatorios; ET solo aleatoriza el umbral, sin optimización). Consecuencias:

- Los árboles necesitan crecer **más profundo** para separar los puntos correctamente
- Con `min_samples_leaf=1` y `max_depth=None`, cada árbol crece hasta tener exactamente 1 muestra por hoja → memorización total del train
- Este es un overfitting **estructural**, no estadístico — es inherente a cómo ET construye los árboles

### ¿Por qué aun así generaliza bien?

1. **Decorrelación extrema entre árboles**: splits totalmente aleatorios producen árboles mucho más distintos entre sí que en RF. Al promediar predictores fuertemente decorrelacionados, la varianza del ensemble cae más rápido.
2. **El promedio cancela el ruido**: cada árbol individual tiene sesgo alto y ruido alto, pero los ruidos van en direcciones distintas. Al promediar, el ruido se cancela y la señal persiste.
3. **Condición**: esto solo funciona si hay suficientes árboles. Con 1 árbol, el ET base tiene R² test negativo (sale: 0.325; rent: −0.193).

---

## Experimento 1 — Barrido de `min_samples_leaf`

ET con `n_estimators=200`, `max_depth=None`. Se barren: `[1, 2, 5, 10, 20, 50, 100]`.

### Sale

| min_samples_leaf | train_R2 | test_R2 | delta_R2 | test_RMSE |
|-----------------|---------|---------|---------|----------|
| 1 | 0.99994 | 0.69972 | 0.300 | 0.28593 |
| 2 | 0.96370 | 0.68547 | 0.278 | 0.29264 |
| 5 | 0.81070 | 0.62520 | 0.185 | 0.31945 |
| 10 | 0.70379 | 0.58304 | 0.121 | 0.33694 |
| 20 | 0.59893 | 0.50787 | 0.091 | 0.36605 |
| 50 | 0.48805 | 0.41765 | 0.070 | 0.39819 |
| 100 | 0.34825 | 0.26934 | 0.079 | 0.44603 |

### Rent

| min_samples_leaf | train_R2 | test_R2 | delta_R2 | test_RMSE |
|-----------------|---------|---------|---------|----------|
| 1 | 0.99534 | 0.23579 | 0.760 | 0.32277 |
| 2 | 0.89378 | 0.32165 | 0.572 | 0.30410 |
| 5 | 0.68279 | 0.36385 | 0.319 | 0.29449 |
| 10 | 0.55081 | 0.36831 | 0.183 | 0.29345 |
| 20 | 0.45293 | **0.36938** | 0.084 | **0.29321** |
| 50 | 0.17926 | 0.25317 | −0.074 | 0.31908 |
| 100 | 0.09625 | 0.14342 | −0.047 | 0.34172 |

**Observaciones clave:**

- **Sale**: al subir `min_samples_leaf` de 1 a 20, el gap de overfitting se reduce de 0.300 a 0.091. Sin embargo, el R² test **también cae** de 0.700 a 0.508 — la regularización reduce el overfitting pero a costa de peor generalización. El modelo con `min_samples_leaf=1` es el mejor en test para sale.

- **Rent**: comportamiento inverso. Con `min_samples_leaf=1`, el R² test es solo 0.236 — el ET base no generaliza en rent. El R² test mejora al aumentar la hoja hasta `min_samples_leaf=20` (R² test = 0.369). Con `min_samples_leaf=50` se empieza a sobrerestritur y el R² test cae de nuevo. **Para rent, se necesita `min_samples_leaf` entre 10 y 20.**

---

## Experimento 2 — Profundidad de los árboles individuales

ET con `n_estimators=100` sobre el dataset sale. Muestra la distribución de profundidades por valor de `min_samples_leaf`:

| min_samples_leaf | depth_media | depth_max | depth_min | train_R2 | test_R2 |
|-----------------|-------------|-----------|-----------|---------|---------|
| 1 | 20.6 | 26 | 17 | 0.99994 | 0.69796 |
| 5 | 12.1 | 15 | 9 | 0.80822 | 0.62051 |
| 20 | 6.3 | 8 | 5 | 0.59801 | 0.50920 |

Con `min_samples_leaf=1`, los árboles tienen **~21 niveles de profundidad media** (máximo 26). Con `min_samples_leaf=20`, los árboles son mucho más superficiales (~6 niveles). La profundidad de los árboles individuales correlaciona directamente con el train R² — a más profundidad, más memorización.

---

## Experimento 3 — El número de árboles como regularizador

ET con `min_samples_leaf=1` (sin poda). Se barren: `[1, 5, 10, 25, 50, 100, 200, 400]`.

### Sale

| n_trees | train_R2 | test_R2 | delta_R2 | test_RMSE |
|---------|---------|---------|---------|----------|
| 1 | 0.99994 | 0.32511 | 0.675 | 0.42867 |
| 5 | 0.99994 | 0.63826 | 0.362 | 0.31383 |
| 10 | 0.99994 | 0.65910 | 0.341 | 0.30466 |
| 25 | 0.99994 | 0.69407 | 0.306 | 0.28861 |
| 50 | 0.99994 | 0.69800 | 0.302 | 0.28675 |
| 100 | 0.99994 | 0.69796 | 0.302 | 0.28677 |
| 200 | 0.99994 | 0.69972 | 0.300 | 0.28593 |
| 400 | 0.99994 | 0.69697 | 0.303 | 0.28724 |

### Rent

| n_trees | train_R2 | test_R2 | delta_R2 | test_RMSE |
|---------|---------|---------|---------|----------|
| 1 | 0.99534 | −0.19326 | 1.189 | 0.40332 |
| 5 | 0.99534 | 0.07920 | 0.916 | 0.35430 |
| 25 | 0.99534 | 0.23405 | 0.761 | 0.32314 |
| 50 | 0.99534 | 0.21345 | 0.782 | 0.32745 |
| 100 | 0.99534 | 0.21248 | 0.783 | 0.32765 |
| 200 | 0.99534 | 0.23579 | 0.760 | 0.32277 |
| 400 | 0.99534 | 0.23558 | 0.760 | 0.32281 |

**Observaciones clave:**

- **Sale**: el train R² se mantiene en 0.99994 para cualquier número de árboles — la memorización individual no desaparece. El test R² mejora rápidamente: de 0.325 con 1 árbol a ≈0.700 con ≥50 árboles. A partir de 50 árboles la mejora es marginal — **el ensemble converge a los 50 árboles en sale**.

- **Rent**: el comportamiento es más errático. Con 1 árbol, R² test = −0.193 (peor que predecir la media). El test R² mejora hasta ≈0.24 con 200 árboles, pero no supera 0.25 — el problema de rent con ET y `min_samples_leaf=1` no se resuelve aumentando el número de árboles. Esto confirma que para rent se necesita cambiar `min_samples_leaf`, no solo añadir árboles.

---

## Experimento 4 — ET vs RF: comparación directa con mismo grid

5 configuraciones compartidas entre ET y RF:

| config | ET sale R²_test | RF sale R²_test | ET ventaja sale | ET rent R²_test | RF rent R²_test | ET ventaja rent |
|--------|----------------|----------------|----------------|----------------|----------------|----------------|
| d=None, l=1 | 0.69972 | 0.60105 | **+0.099** | 0.23579 | 0.41640 | **−0.181** |
| d=None, l=5 | 0.62520 | 0.60349 | +0.022 | 0.36385 | 0.42888 | **−0.065** |
| d=20, l=1 | 0.70014 | 0.60031 | **+0.100** | 0.23312 | 0.41271 | **−0.176** |
| d=20, l=5 | 0.62520 | 0.60349 | +0.022 | 0.36385 | 0.42888 | **−0.065** |
| d=10, l=5 | 0.62342 | 0.60371 | +0.020 | 0.37305 | 0.42964 | **−0.057** |

**Conclusiones del experimento 4:**

- **Sale**: ET **siempre supera** a RF (ventaja de +0.02 a +0.10 en R² test). Con `min_samples_leaf=1`, la ventaja es máxima (+0.10). La mayor decorrelación de ET compensa su mayor sobreajuste en el dataset de venta.

- **Rent**: RF **siempre supera** a ET (ET pierde entre −0.06 y −0.18 R²). La decorrelación extrema de ET con `min_samples_leaf=1` es contraproducente en rent — el dataset es más pequeño (477 vs 588 filas) y más ruidoso, y ET no puede promediar suficiente señal.

- La profundidad máxima (`max_depth=None` vs `max_depth=20`) no marca diferencia cuando `min_samples_leaf=5` — los árboles ya no son tan profundos.

---

## Resumen de modelos recomendados por dataset

### SALE — Mejor modelo: Extra Trees óptimo

```
RMSE test  : 0.28268
R² test    : 0.70651
MAE test   : 0.20556
CV RMSE    : 0.33306
Params     : n_estimators=400, max_depth=20, max_features=0.4, min_samples_leaf=1
```

Extra Trees con `min_samples_leaf=1` es claramente superior para sale. El R² test de 0.707 supera al RF óptimo (0.657) y representa una mejora sustancial sobre el mejor modelo de regresión lineal del nb 51 (Ridge: R² test 0.638).

### RENT — Mejor modelo: RF óptimo

```
RMSE test  : 0.27388
R² test    : 0.44976
MAE test   : 0.19304
CV RMSE    : 0.29789
Params     : n_estimators=400, max_depth=10, max_features=0.6, min_samples_leaf=5
```

RF con `max_depth=10` y `min_samples_leaf=5` es el mejor para rent. El R² test de 0.450 es modestamente superior al mejor modelo lineal del nb 51 (Lasso+OLS: R² test 0.576) — en realidad, **los modelos lineales superan a los árboles en rent**.

---

## Comparación con los modelos lineales del notebook 51

| Dataset | Mejor lineal (nb 51) | R² test (lineal) | Mejor árbol (nb 52) | R² test (árbol) | Ganador |
|---------|---------------------|----------------|--------------------|--------------------|---------|
| Sale | Ridge | 0.638 | ET óptimo | **0.707** | Árboles (+6.9 pp) |
| Rent | Lasso+OLS | **0.576** | RF óptimo | 0.450 | Lineal (+12.6 pp) |

Este resultado es contraintuitivo para rent: los modelos lineales con regularización generalizan mejor que los ensemble de árboles. Posibles explicaciones:
- El dataset de alquiler (477 filas) es pequeño para ensemble complejos — la regresión lineal funciona mejor con pocos datos
- El mercado de alquiler puede tener relaciones más lineales y aditivas
- Los precios de alquiler están más concentrados geográficamente (9 municipios vs 13) y la regresión puede capturar esto con dummies y el precio medio de municipio

---

## Interpretación del error en euros

La métrica MAE en escala log se interpreta como porcentaje de error aproximado:

- **Sale ET óptimo: MAE = 0.206** → error mediano ≈ e^0.206 − 1 ≈ **+22.9%**
- **Rent RF óptimo: MAE = 0.193** → error mediano ≈ e^0.193 − 1 ≈ **+21.3%**

Para un inmueble de 200.000€ en venta, el error mediano es ≈ 45.800€.
Para un alquiler de 1.000€/mes, el error mediano es ≈ 213€.

Comparado con el mejor modelo lineal:
- Sale lineal (Ridge): MAE = 0.229 → **+25.7%** → **Los árboles mejoran 2.8 pp en MAE**
- Rent lineal (Lasso+OLS): MAE = 0.165 → **+17.9%** → **Los lineales son mejores en 3.4 pp de MAE**

---

## Estructura del código

### Funciones auxiliares (`cell 3`)

| Función | Descripción |
|---------|-------------|
| `get_metrics(y_real, y_pred)` | Devuelve DataFrame con MSE, RMSE, MAE, R², MAPE |
| `collapse_rare_municipios(df, muni_cols, min_obs)` | Colapsa dummies de municipio con < `min_obs` en `municipio_otro` |
| `prepare_X(df, feature_cols)` | Selecciona y imputa features por mediana |
| `plot_diagnostics(y_test, pred_test, title)` | Real vs Predicho, histograma de residuos, Q-Q plot |
| `plot_feature_importance(importances, feature_names, title, top_n)` | Barplot de top N importancias |
| `plot_prediction_intervals(y_test, pred_low, pred_median, pred_high, title)` | Intervalos de predicción para QRF |
| `run_base_model(model, X_train, X_test, y_train, y_test, model_name)` | Entrena y evalúa modelo base, imprime métricas y ratios de sobreajuste |
| `run_grid_search(estimator, param_grid, X_train, X_test, y_train, y_test, model_name)` | GridSearchCV con CV=5, scoring=neg_RMSE, imprime mejores params y métricas |
| `run_qrf_base(model, X_train, X_test, y_train, y_test, model_name)` | Versión para QRF (predice mediana q=0.5) |
| `run_qrf_grid_search(param_grid, X_train, X_test, y_train, y_test, model_name)` | GridSearchCV para QRF |

### Bucle principal (`cell 4`)

Para cada dataset (sale, rent):
1. Cargar CSV
2. Colapsar municipios raros
3. Preparar X con imputación por mediana
4. Split 80/20
5. Para cada modelo: base → diagnósticos → feature importance → GridSearch → diagnósticos → feature importance
6. Construir tabla resumen del dataset
7. Acumular en `all_summary_rows`

Al final: resumen global con todos los modelos y datasets.

---

## Decisiones de diseño importantes

| Decisión | Justificación |
|----------|--------------|
| **No eliminar outliers** | Los árboles son más robustos a outliers que la regresión lineal — no es necesario el IQR previo |
| **Sin escalado de features** | Los árboles son invariantes a la escala — no se aplica StandardScaler |
| **Superficie sin log** | La transformación logarítmica no es necesaria para árboles |
| **Latitud/longitud comentadas** | Decisión explícita de no incluir coordenadas geográficas directas (se usan distancias en su lugar) |
| **GridSearchCV con n_jobs=-1** | Paralelismo completo para reducir el tiempo de entrenamiento |
| **Métrica CV: neg_RMSE** | Coherente con la métrica principal de evaluación |
| **QRF con `min_samples_leaf≥5`** | Los QRF son computacionalmente más costosos — se excluye `min_samples_leaf=1` del grid |
| **`random_state=42`** | Reproducibilidad en split y en los modelos estocásticos |

---

## Archivos relacionados

```
notebooks/05_ML/
├── 52_random_forest_def.ipynb   ← ESTE NOTEBOOK (definitivo)
└── (versiones exploratorias anteriores implícitas en el naming convention _def)

data/gold/
├── final_sale.csv    ← dataset venta (588 filas)
└── final_rent.csv    ← dataset alquiler (477 filas)
```

---

## Conclusiones y observaciones finales

| Observación | Explicación |
|------------|-------------|
| R²_train ≈ 0.99–1.0 en ET | Con `min_samples_leaf=1`, los árboles crecen hasta 1 muestra por hoja. Memorización total. No es señal de que el modelo sea malo. |
| R²_test sigue siendo bueno (sale) | El promedio de cientos de árboles decorrelacionados cancela el ruido individual. |
| ET falla en rent sin tuning | Con `min_samples_leaf=1`, R² test = 0.212 en rent. Se necesita `min_samples_leaf≥5`. |
| Al aumentar `min_samples_leaf` | El gap de overfitting se reduce pero el test R² también cae (sale) o mejora (rent) — el efecto depende del dataset. |
| Al aumentar `n_estimators` | Reduce progresivamente el gap train-test sin cambiar el train R². El ensemble converge en ~50 árboles para sale. |
| RF supera a ET en rent | La decorrelación extrema de ET con pequeños datasets ruidosos es perjudicial. RF con splits optimizados es más estable. |
| **`ccp_alpha` no ayuda** | El alpha óptimo siempre resulta ser el más bajo del grid (0.0001). La poda agresiva empeora los resultados en ambos datasets. |
| Los lineales ganan en rent | Lasso+OLS (R² 0.576) supera al mejor árbol (RF, R² 0.450). El mercado de alquiler con n=477 favorece modelos más simples. |
