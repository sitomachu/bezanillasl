# Documentación técnica: Modelos XGBoost optimizados — BezanillaSL

## 1. Resumen ejecutivo del sistema de modelización

BezanillaSL modela los precios del mercado inmobiliario de **Cantabria** (España) mediante dos XGBoost hedónicos, uno para **venta** y otro para **alquiler**. La variable objetivo es siempre **`log_precio`** (logaritmo natural del precio en euros), lo que reduce la heterocedasticidad y simetriza la distribución. Para convertir una predicción a euros: `precio = exp(log_precio_predicho)`.

### Modelos definidos

| ID | Nombre funcional | Dataset | Target | Metodología |
|----|-----------------|---------|--------|-------------|
| M-SALE | XGBoost Sale | `final_sale_idealistaAPI.csv` | `log_precio` | Optuna 100 trials, CV-RMSE 5-fold (`53_boost_sale_optuna`) |
| M-RENT | XGBoost Rent | `final_rent_idealistaAPI.csv` | `log_precio` | Optuna 100 trials, CV-RMSE 5-fold (`53_boost_rent`) |

### Flujo de dependencias entre notebooks

```
53_boost_rent.ipynb ──────────────────────────────────────────────────────────┐
  └─ Optuna 100 trials → XGB_PARAMS_RENT (mejores hiperparámetros)            │
                                                                               ▼
53_boost_sale_optuna.ipynb ─────────────────────────────────────┐    55_sale_rent_models.ipynb
  └─ Optuna 100 trials → XGB_PARAMS_SALE (mejores hiperparámetros)│   └─ Reentrena M-SALE y M-RENT
                                                                   │       con los mismos params
                                                                   ▼       y evalúa ambos juntos
                                                      55_input_result.ipynb
                                                          └─ Modelos entrenados sobre 80% datos
                                                             Predicción individual con intervalo ±1σ
                                                     
                                                      55_input_result_no_k_fold.ipynb
                                                          └─ Modelos entrenados sobre 100% datos
                                                             Usa CV-RMSE como intervalo de error
```

Los cinco notebooks son:
1. **`53_boost_rent.ipynb`**: exploración EDA + limpieza de outliers + Optuna para M-RENT
2. **`53_boost_sale_optuna.ipynb`**: exploración EDA + limpieza de outliers + Optuna para M-SALE (reemplaza a `53_boost_sale`)
3. **`55_sale_rent_models.ipynb`**: copia exacta de params de los 53_* y reentrena ambos modelos con evaluación conjunta
4. **`55_input_result.ipynb`**: herramienta interactiva de predicción individual (split 80/20)
5. **`55_input_result_no_k_fold.ipynb`**: herramienta interactiva de predicción individual (100% datos, más robusto)

---

## 2. Datos de entrada de cada modelo

### 2.1 Notebook `53_boost_rent.ipynb` — Dataset de alquiler

| Parámetro | Valor |
|-----------|-------|
| Ruta del archivo | `data/gold/final_rent_idealistaAPI.csv` |
| Filas totales cargadas | 754 |
| Columnas totales | 50 |
| Columnas con nulos | `planta` (147), `distrito` (137), `subtipologia` (644) |

**Variable objetivo**

| Parámetro | Valor |
|-----------|-------|
| Nombre | `log_precio` |
| Descripción | Logaritmo natural del precio mensual en euros |
| Media (tras limpieza) | 6.8380 |
| Std (tras limpieza) | 0.2473 |
| Rango válido (IQR×1.5) | [6.207, 7.481] → [496 €, 1774 €] |

**Target alternativo evaluado (descartado):** `log_precio_m2` — log(precio/m²). Se compara con `log_precio` mediante CV-RMSE en XGBoost base:

| Target | CV-RMSE | CV-R² |
|--------|---------|-------|
| `log_precio` | 0.14070 | 0.67246 |
| `log_precio_m2` | 0.14430 | 0.56896 |

`log_precio` es seleccionado por mayor CV-R² (0.672 vs 0.569).

**Features finales (27)**

| Feature | Tipo | Nota |
|---------|------|------|
| `superficie_construida_m2` | Continua | Feature hedónica principal |
| `numero_dormitorios` | Discreta | |
| `numero_banos` | Discreta | |
| `planta_num` | Discreta | Número de planta |
| `es_exterior_piso` | Dummy (0/1) | Específica de rent — no está en sale |
| `tiene_ascensor_piso` | Dummy (0/1) | Específica de rent — no está en sale |
| `tiene_garaje` | Dummy (0/1) | |
| `obra_nueva` | Dummy (0/1) | |
| `distancia_min_playa_km` | Continua | Distancia mínima a playa |
| `distancia_min_supermercado_km` | Continua | |
| `distancia_min_colegio_km` | Continua | |
| `precio_m2_municipio_media` | Continua | Precio medio de VENTA por municipio (no deriva del target) |
| `ratio_dormitorios_superficie` | Continua | Feature derivada: `dormitorios / m²` |
| `ratio_banos_superficie` | Continua | Feature derivada: `baños / m²` |
| `interaccion_planta_sin_ascensor_piso` | Continua | Feature derivada: `planta × (1 - tiene_ascensor)` |
| `distancia_centro_municipio_km` | Continua | |
| `score_cercania_servicios` | Continua | Índice compuesto de distancias |
| `tipologia_unificada_piso` | Dummy (0/1) | |
| `tipologia_unificada_unifamiliar` | Dummy (0/1) | |
| `municipio_Camargo` | Dummy OHE | Municipios con ≥ 10 obs |
| `municipio_Castro-Urdiales` | Dummy OHE | |
| `municipio_El Astillero` | Dummy OHE | |
| `municipio_Piélagos` | Dummy OHE | |
| `municipio_Santander` | Dummy OHE | |
| `municipio_Torrelavega` | Dummy OHE | |
| `municipio_otro` | Dummy OHE | Municipio presente en los datos pero < 10 obs |
| `municipio_otros` | Dummy OHE | Municipios con < 10 obs agrupados dinámicamente |

**Exclusiones explícitas por leakage:** `precio`, `precio_m2`, `precio_m2_raw`, `log_precio_m2`, `rentabilidad_bruta_zona` — esta última fue descartada porque usaba `precio` (= exp(target)) directamente como feature, disparando el R² artificialmente a > 0.9.

**Tamaño train/test:**

| Split | Filas | % |
|-------|-------|---|
| Train | 529 | 80% |
| Test | 133 | 20% |
| Total | 662 | 100% (tras limpieza) |

---

### 2.2 Notebook `53_boost_sale_optuna.ipynb` — Dataset de venta

| Parámetro | Valor |
|-----------|-------|
| Ruta del archivo | `data/gold/final_sale_idealistaAPI.csv` |
| Filas totales cargadas | 2694 |
| Columnas totales | 71 |
| Columnas con nulos | `planta` (1335), `distrito` (793), `subtipologia` (1628) |

**Variable objetivo**

| Parámetro | Valor |
|-----------|-------|
| Nombre | `log_precio` |
| Rango válido IQR×1.5 | [10.895, 14.254] → [53.914 €, 1.550.605 €] |

**Features finales (52)**

Las 21 features base (incluyendo `latitud`, `longitud`, `es_exterior_piso` y `tiene_ascensor_piso`, ahora activas en sale) más 31 dummies de municipio OHE:

| Feature | Tipo | Nota |
|---------|------|------|
| `superficie_construida_m2` | Continua | |
| `numero_dormitorios` | Discreta | |
| `numero_banos` | Discreta | Correlación con log_precio: 0.6228 (la más alta entre continuas) |
| `latitud` | Geoespacial | Activa en sale desde `53_boost_sale_optuna` |
| `longitud` | Geoespacial | Activa en sale desde `53_boost_sale_optuna` |
| `planta_num` | Discreta | |
| `es_exterior_piso` | Dummy (0/1) | Activa en sale desde `53_boost_sale_optuna` |
| `tiene_ascensor_piso` | Dummy (0/1) | Activa en sale desde `53_boost_sale_optuna` |
| `tiene_garaje` | Dummy (0/1) | |
| `obra_nueva` | Dummy (0/1) | |
| `distancia_min_playa_km` | Continua | |
| `distancia_min_supermercado_km` | Continua | |
| `distancia_min_colegio_km` | Continua | |
| `precio_m2_municipio_media` | Continua | Correlación con log_precio: 0.1335 |
| `ratio_dormitorios_superficie` | Continua | |
| `ratio_banos_superficie` | Continua | |
| `interaccion_planta_sin_ascensor_piso` | Continua | |
| `distancia_centro_municipio_km` | Continua | |
| `score_cercania_servicios` | Continua | |
| `tipologia_unificada_piso` | Dummy (0/1) | |
| `tipologia_unificada_unifamiliar` | Dummy (0/1) | |
| `municipio_Ampuero` … `municipio_otros` | Dummy OHE | 31 municipios (ver lista completa abajo) |

**Municipios OHE en sale (31):** `Ampuero`, `Barcena de Cicero`, `Camargo`, `Castro-Urdiales`, `Colindres`, `Cudon`, `El Astillero`, `Guarnizo`, `Laredo`, `Liendo`, `Limpias`, `Marina de Cudeyo`, `Miengo`, `Mogro`, `Noja`, `Ortuella`, `Piélagos`, `Polanco`, `Ribamontan al Mar`, `Ribamontan al Monte`, `Santa Cruz de Bezana`, `Santander`, `Santoña`, `Santurtzi`, `Suances`, `Torrelavega`, `Villaescusa`, `Viveda`, `Voto`, `municipio_otro`, `municipio_otros`.

**Correlaciones destacadas con log_precio:**

| Feature | Correlación |
|---------|------------|
| `superficie_construida_m2` | 0.6945 |
| `numero_banos` | 0.6228 |
| `numero_dormitorios` | 0.5500 |
| `precio_m2_municipio_media` | 0.1335 |

**Tamaño train/test:**

| Split | Filas | % |
|-------|-------|---|
| Train | 2034 | 80% |
| Test | 509 | 20% |
| Total | 2543 | 100% (tras limpieza) |

---

## 3. Preprocesamiento y feature engineering

### 3.1 Limpieza aplicada

#### Modelo RENT (`53_boost_rent`)

Se aplican tres pasos en secuencia:

**Paso 1 — Filtro vacacional:** eliminar alquileres turísticos con `precio_m2 > 18 €/m²/mes`. Se eliminan 61 filas (8.1%).

**Paso 2 — Filtro suelo:** eliminar anomalías baratas con `precio_m2 < 6 €/m²/mes` (garajes, errores, propiedades con precio no de mercado). Se eliminan 12 filas (1.7%). Las 12 propiedades eliminadas son pisos grandes (108–220 m²) con `precio_m2` entre 4.52 y 5.99 €/m²/mes.

**Paso 3 — IQR×1.5 sobre `log_precio`:** elimina extremos de precio absoluto. Se eliminan 19 filas (2.8%). Rango válido resultante: `log_precio ∈ [6.207, 7.481]` → precios entre 496 € y 1.774 €/mes.

**Resultado:** 662 filas (de 754 originales).

#### Modelo SALE (`53_boost_sale_optuna`)

Se aplican dos pasos:

**Paso 1 — IQR×1.5 sobre `log_precio`:** 0 filas eliminadas (0.0%). El rango válido es [10.895, 14.254] → [53.914 €, 1.550.605 €]. La distribución de venta no tiene outliers extremos en precio absoluto.

**Paso 2 — Suelo de `precio_m2 ≥ 1000 €/m²`:** elimina propiedades anómalamente baratas por metro cuadrado (ruinas, no-residencial, errores). Se eliminan 151 filas (5.6%). Estas propiedades son responsables de la cola inferior del Q-Q plot (residuos hasta −1.25). Los eliminados incluyen propiedades de hasta 562 m² con `precio_m2` desde 357 €/m².

**Resultado:** 2.543 filas (de 2.694 originales).

### 3.2 Transformaciones sobre la variable objetivo

- **`log_precio`** es la variable objetivo en ambos modelos (logaritmo natural del precio en euros). Ya está precalculada en el CSV gold.
- Para convertir predicciones a precio real: `precio = exp(log_precio_predicho)`.
- No se aplica ninguna transformación adicional al target dentro de los notebooks.
- El target alternativo `log_precio_m2 = log(precio / superficie_m2)` se calcula en `53_boost_rent` solo para comparar y se descarta.

### 3.3 Encoding de variables categóricas

- **Municipios**: One-Hot Encoding (OHE) dinámico. Se detectan automáticamente las columnas `municipio_*` del CSV gold. Los municipios con menos de `MIN_MUNI_OBS = 10` observaciones se colapsan en `municipio_otros` (operación `max` por fila de los dummies de municipios raros). Se conserva `municipio_otro` como categoría explícita si ya está presente en el gold.
- **Tipología**: `tipologia_unificada_piso` y `tipologia_unificada_unifamiliar` ya están precalculadas en el gold como dummies.

### 3.4 Feature engineering (variables derivadas)

| Feature derivada | Fórmula | Propósito |
|-----------------|---------|-----------|
| `ratio_dormitorios_superficie` | `numero_dormitorios / superficie_construida_m2` | Densidad habitacional |
| `ratio_banos_superficie` | `numero_banos / superficie_construida_m2` | Densidad de baños |
| `interaccion_planta_sin_ascensor_piso` | `planta_num × (1 - tiene_ascensor_piso)` | Penalización por accesibilidad (planta alta sin ascensor) |
| `log_precio_m2` | `log(precio / superficie_construida_m2)` | Solo en rent, evaluado como target alternativo y descartado |

### 3.5 Preprocesamiento en función `build_X()`

La función `build_X(df, base_features)` aplica en este orden:
1. Selección de las columnas de `BASE_FEATURES` disponibles en el dataframe.
2. Colapso de municipios raros a `municipio_otros`.
3. Construcción de `X_raw` con `base_features + mun_final`.
4. Cálculo de medianas (`medians = X_raw.median().to_dict()`) — se guardan para imputación de inputs externos.
5. **Imputación** de nulos con `SimpleImputer(strategy="median")`.
6. Los árboles de decisión son invariantes a la escala — **no se aplica estandarización**.

### 3.6 Separación train/test

```python
train_test_split(X, y, test_size=0.20, random_state=42)
```

- Método: división aleatoria simple (sin estratificación).
- `random_state = 42` en todos los notebooks y modelos.
- El test se usa exclusivamente para evaluación final — nunca entra en la optimización de Optuna ni en la validación cruzada interna.

---

## 4. Optimización de hiperparámetros con Optuna

### 4.1 Contexto

Ambos modelos (M-SALE y M-RENT) usan Optuna con 100 trials y el mismo espacio de búsqueda. `53_boost_sale_optuna.ipynb` realiza la búsqueda para venta; `53_boost_rent.ipynb` para alquiler.

### 4.2 Configuración del estudio (Sale y Rent)

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

- **Dirección:** `minimize` (minimizar CV-RMSE).
- **Sampler:** TPE (Tree-structured Parzen Estimator) con `seed=42`.
- **Trials:** 100.
- **Criterio de parada:** número fijo de trials (no hay early stopping de Optuna).
- **Verbosidad:** `optuna.logging.set_verbosity(optuna.logging.WARNING)` — solo se muestra la barra de progreso.

### 4.3 Función objetivo

```python
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1000, step=50),
        max_depth         = trial.suggest_int("max_depth", 3, 7),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 15),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        gamma             = trial.suggest_float("gamma", 0.0, 5.0),
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 0,
    )
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(
        XGBRegressor(**params), X_train, y_train,
        cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    return float(-cv.mean())
```

La métrica es **CV-RMSE negativo** (5-fold KFold sobre `X_train`). El test no se toca durante la búsqueda. La función es idéntica en ambos notebooks (`53_boost_sale_optuna` y `53_boost_rent`).

### 4.4 Espacio de búsqueda completo (Sale y Rent)

| Hiperparámetro | Tipo | Rango | Escala |
|----------------|------|-------|--------|
| `n_estimators` | `int` | [200, 1000] | step=50 |
| `max_depth` | `int` | [3, 7] | lineal |
| `learning_rate` | `float` | [0.01, 0.20] | log |
| `subsample` | `float` | [0.5, 1.0] | lineal |
| `colsample_bytree` | `float` | [0.5, 1.0] | lineal |
| `min_child_weight` | `int` | [1, 15] | lineal |
| `reg_lambda` | `float` | [0.1, 10.0] | log |
| `reg_alpha` | `float` | [1e-3, 5.0] | log |
| `gamma` | `float` | [0.0, 5.0] | lineal |

> `reg_alpha` y `gamma` son específicos de XGBoost: `reg_alpha` es regularización L1 sobre los pesos de las hojas; `gamma` es el umbral mínimo de reducción de loss para hacer una partición en un nodo. Ambos se buscan en sale y en rent con los mismos rangos.

### 4.5 Validación cruzada dentro de cada trial

En cada uno de los 100 trials se entrena un XGBoost con `KFold(n_splits=5, shuffle=True, random_state=42)` sobre `X_train`. El CV-RMSE resultante es el valor devuelto por la función objetivo. Idéntico en sale y rent.

### 4.6 Mejores hiperparámetros encontrados

#### Sale (`53_boost_sale_optuna`)

**Trial ganador: #76 de 100. Mejor CV-RMSE: 0.23347.**

```python
XGB_PARAMS_SALE = {
    "n_estimators":     950,
    "max_depth":        6,
    "learning_rate":    0.026390709496515886,
    "subsample":        0.6705806157309522,
    "colsample_bytree": 0.7312116009128224,
    "min_child_weight": 9,
    "reg_lambda":       1.6752171349993321,
    "reg_alpha":        0.3505729220414384,
    "gamma":            0.005208850498085864,
}
```

#### Rent (`53_boost_rent`)

**Trial ganador: #97 de 100. Mejor CV-RMSE: 0.14622.**

```python
XGB_PARAMS_RENT = {
    "n_estimators":     1000,
    "max_depth":        6,
    "learning_rate":    0.011737724486287017,
    "subsample":        0.6046283826235364,
    "colsample_bytree": 0.8531891359517659,
    "min_child_weight": 6,
    "reg_lambda":       3.378990393436524,
    "reg_alpha":        0.06467616791725625,
    "gamma":            0.03464694507245766,
}
```

Parámetros fijos no buscados en ambos modelos: `random_state=42`, `n_jobs=-1`, `verbosity=0`.

### 4.7 Gráficas de Optuna (Sale y Rent)

Ambos notebooks generan las mismas dos gráficas:

**Gráfica 1 — Curva de convergencia acumulada** (`axes[0]`):
- Tipo: línea
- Eje X: número de trial (0–99)
- Eje Y: CV-RMSE (mejor acumulado)
- Muestra cómo el mejor valor mejora con más trials

**Gráfica 2 — Dispersión de todos los trials** (`axes[1]`):
- Tipo: scatter + línea de mejor acumulado
- Eje X: trial
- Eje Y: CV-RMSE de cada trial individual (puntos en azul) y mejor acumulado (línea roja)
- Permite ver la varianza del espacio de búsqueda

Ambas en una figura de `figsize=(14, 4)`, `plt.subplots(1, 2)`.

---

## 5. Validación cruzada (K-Fold)

### 5.1 Configuración

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

`CV_FOLDS = 5` en todos los notebooks.

### 5.2 Dónde se aplica

| Notebook | K-Fold dentro de Optuna | K-Fold fuera de Optuna (evaluación final) |
|----------|------------------------|------------------------------------------|
| `53_boost_rent` | Sí — en cada trial | Sí — CV-RMSE final sobre `X_train` |
| `53_boost_sale_optuna` | Sí — en cada trial | Sí — CV-RMSE final sobre `X_train` |
| `55_sale_rent_models` | No | Sí — CV-RMSE sobre `X_train` de cada modelo |
| `55_input_result` | No | Sí — split 80/20; el RMSE del test se usa como intervalo |
| `55_input_result_no_k_fold` | No | No — entrena sobre 100%; usa CV-RMSE hardcodeado |

### 5.3 Métricas CV — `53_boost_rent`

- CV-RMSE (5-fold, sobre `X_train` de 529 filas): **0.14622**
- No se imprimen resultados por fold individual — solo la media.

### 5.4 Métricas CV — `53_boost_sale_optuna`

- CV-RMSE (5-fold, sobre `X_train` de 2034 filas): **0.23347** (trial #76)
- No se imprimen resultados por fold individual — solo la media.

### 5.5 Diferencia entre `55_input_result` y `55_input_result_no_k_fold`

| Aspecto | `55_input_result` | `55_input_result_no_k_fold` |
|---------|------------------|---------------------------|
| Datos de entrenamiento | 80% (2034 sale / 529 rent) | 100% (2543 sale / 662 rent) |
| Evaluación | RMSE sobre test 20% | CV-RMSE hardcodeado del modelo split |
| Intervalo de error | `±1σ = ±RMSE_test` | `±1σ = ±CV_RMSE` (0.23347 sale / 0.14622 rent) |
| Municipios disponibles | 31 sale / 10 rent | 31 sale / 10 rent (idénticos) |
| Fiabilidad del intervalo | Basado en un único split 20% | Basado en 5-fold cross-val — más robusto |
| Justificación | Evaluación + predicción | Producción — el modelo ve más datos |

---

## 6. Entrenamiento final del modelo

### 6.1 Modelo SALE

**Hiperparámetros finales (Optuna, `53_boost_sale_optuna`, trial #76):**

```python
XGB_PARAMS_SALE = {
    "n_estimators":     950,
    "max_depth":        6,
    "learning_rate":    0.026390709496515886,
    "subsample":        0.6705806157309522,
    "colsample_bytree": 0.7312116009128224,
    "min_child_weight": 9,
    "reg_lambda":       1.6752171349993321,
    "reg_alpha":        0.3505729220414384,
    "gamma":            0.005208850498085864,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
```

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `objective` | `reg:squarederror` (defecto XGBoost) | Minimiza MSE |
| `eval_metric` | No configurado explícitamente | |
| `early_stopping_rounds` | No aplicado | |
| Tamaño train | 2034 filas | 80% de 2543 |
| `n_jobs` | -1 | Todos los cores disponibles |

### 6.2 Modelo RENT

**Hiperparámetros finales (Optuna):**

```python
XGB_PARAMS_RENT = {
    "n_estimators":     1000,
    "max_depth":        6,
    "learning_rate":    0.011737724486287017,
    "subsample":        0.6046283826235364,
    "colsample_bytree": 0.8531891359517659,
    "min_child_weight": 6,
    "reg_lambda":       3.378990393436524,
    "reg_alpha":        0.06467616791725625,
    "gamma":            0.03464694507245766,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
```

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `objective` | `reg:squarederror` (defecto XGBoost) | Minimiza MSE |
| `eval_metric` | No configurado explícitamente | |
| `early_stopping_rounds` | No aplicado | |
| Tamaño train | 529 filas | 80% de 662 |
| `n_jobs` | -1 | Todos los cores disponibles |

### 6.3 Persistencia de modelos en disco

**No se guarda ningún modelo en disco** en estos cinco notebooks. No hay llamadas a `pickle.dump()`, `joblib.dump()`, `model.save_model()` ni equivalentes. Los modelos se entrenan en memoria y se usan directamente para predecir en la misma sesión. El directorio `data/ML/` contiene archivos de modelos anteriores (linear_regression, random_forest) pero no de estos XGBoost.

---

## 7. Evaluación y métricas de rendimiento

### 7.1 Métricas sobre conjunto de test

#### Modelo SALE (split 80/20, `53_boost_sale_optuna` y `55_sale_rent_models`)

| Split | MSE | RMSE | MAE | R² | MAPE |
|-------|-----|------|-----|-----|------|
| Train | 0.00602 | 0.07757 | 0.05728 | 0.98153 | 0.00456 |
| CV (5-fold) | — | 0.23347 | — | — | — |
| Test | 0.04938 | 0.22222 | 0.16137 | 0.85310 | 0.01292 |

- Sobreajuste: ratio RMSE test/train = **2.8648** | delta R² = **0.1284**
- RMSE_test en escala original: exp(0.22222) − 1 = **±24.9%**

**Comparativa con hiperparámetros anteriores (hardcodeados de `53_boost_def_3`):**

| Métrica | Hardcodeados | Optuna | Mejora |
|---------|-------------|--------|--------|
| CV-RMSE (train) | 0.23904 | 0.23347 | −2.3% |
| Test RMSE | 0.23515 | 0.22222 | −5.5% |
| Test R² | 0.83552 | 0.85310 | +1.8 pp |
| Test MAE | 0.17506 | 0.16137 | −7.8% |

#### Modelo RENT (split 80/20, `53_boost_rent` y `55_sale_rent_models`)

| Split | MSE | RMSE | MAE | R² | MAPE |
|-------|-----|------|-----|-----|------|
| Train | 0.00864 | 0.09295 | 0.07143 | 0.86194 | 0.01043 |
| CV (5-fold) | — | 0.14622 | — | — | — |
| Test | 0.02096 | 0.14478 | 0.11452 | 0.61790 | 0.01677 |

- Sobreajuste: ratio RMSE test/train = **1.5576** | delta R² = **0.2440**
- RMSE_test en escala original: exp(0.14478) − 1 = **±15.6%**

### 7.2 Comparativa entre modelos

| Modelo | RMSE_train | CV-RMSE | RMSE_test | R²_test | Error % (escala original) |
|--------|-----------|---------|-----------|---------|--------------------------|
| M-SALE | 0.07757 | 0.23347 | 0.22222 | 0.853 | ±24.9% |
| M-RENT | 0.09295 | 0.14622 | 0.14478 | 0.618 | ±15.6% |

M-SALE tiene mayor R² de test (0.85 vs 0.62), y un error porcentual menor en euros que con los hiperparámetros anteriores (±25% vs ±28%). M-RENT predice con menor error relativo en euros pero explica menos varianza — probablemente porque el mercado de alquiler en Cantabria tiene más factores no capturados (turismo, estacionalidad, negociación individual).

### 7.3 Modelo `55_input_result_no_k_fold` (100% datos)

| Modelo | CV-RMSE usado | Filas entrenamiento | Error % |
|--------|---------------|--------------------|---------| 
| Sale | 0.23347 | 2543 | ±26.3% |
| Rent | 0.14622 | 662 | ±16% |

El CV-RMSE de sale (0.23347) proviene del mejor trial de Optuna en `53_boost_sale_optuna` (5-fold sobre X_train de 2034 filas). El CV-RMSE de rent es idéntico al del modo 80/20 (0.14622).

### 7.4 Análisis de residuos

Para cada modelo se genera un panel de tres gráficas diagnósticas sobre el conjunto de test:

1. **Real vs Predicho (scatter):** puntos en azul, línea identidad en rojo discontinuo. Permite detectar sesgos sistemáticos.
2. **Histograma de residuos:** `y_test - pred_test`, línea vertical en 0. Evalúa simetría y curtosis.
3. **Q-Q plot de residuos:** `scipy.stats.probplot(residuals, dist="norm")`. Evalúa normalidad de los residuos.

En `53_boost_sale_optuna`, el autor menciona que las propiedades con `precio_m2 < 1000 €/m²` generaban residuos extremos hasta −1.25 en el Q-Q plot — por eso se aplica el suelo en el Paso 2 de la limpieza.

---

## 8. Gráficas e interpretación visual

### 8.1 `53_boost_rent.ipynb`

**Gráfica 1 — Distribución precio_m2 y log_precio (EDA)**
- Tipo: histograma doble (`plt.subplots(1, 2)`, figsize=(14, 4))
- Eje izquierdo: `precio_m2` (€/m²/mes), bins=50, azul; línea roja discontinua en `umbral_vacacional = 18 €/m²/mes`
- Eje derecho: `log_precio`, bins=40, naranja
- Conclusión: el 8.1% de los registros supera el umbral vacacional — hay una cola derecha clara que distorsionaría el modelo

**Gráfica 2 — Optuna: curva de convergencia y dispersión de trials**
- Tipo: línea + scatter (`plt.subplots(1, 2)`, figsize=(14, 4))
- Eje izquierdo: mejor CV-RMSE acumulado a lo largo de los 100 trials
- Eje derecho: CV-RMSE de cada trial (scatter azul) + mejor acumulado (línea roja)
- Conclusión: el trial ganador (#97) encuentra `learning_rate≈0.012` con `n_estimators=1000` — la búsqueda converge hacia tasas de aprendizaje bajas y muchos estimadores

**Gráfica 3 — Diagnósticos del modelo final (test)**
- Tipo: panel triple (`plt.subplots(1, 3)`, figsize=(16, 4))
- (a) Real vs Predicho (scatter), (b) Histograma de residuos, (c) Q-Q plot
- Conclusión: los residuos son aproximadamente simétricos; el Q-Q muestra colas algo más pesadas de lo normal

**Gráfica 4 — Feature importance (Top 20)**
- Tipo: barplot horizontal (`plt.subplots`, figsize=(9, 6))
- Eje X: importancia (gain por defecto de XGBoost)
- Eje Y: nombre de la feature
- Top 20 (ver sección 9)

### 8.2 `53_boost_sale_optuna.ipynb`

**Gráfica 1 — Distribución precio_m2 y log_precio (EDA)**
- Tipo: histograma doble (`plt.subplots(1, 2)`, figsize=(14, 4))
- Eje izquierdo: `precio_m2` (€/m²), bins=50, azul
- Eje derecho: `log_precio`, bins=40, naranja
- Conclusión: la distribución de venta no tiene outliers en precio absoluto (IQR elimina 0 filas); el suelo de 1.000 €/m² elimina el 5.6%

**Gráfica 2 — Optuna Sale: curva de convergencia y dispersión de trials**
- Tipo: línea + scatter (`plt.subplots(1, 2)`, figsize=(14, 4))
- Eje izquierdo: mejor CV-RMSE acumulado a lo largo de los 100 trials
- Eje derecho: CV-RMSE de cada trial individual (scatter azul) + mejor acumulado (línea roja)
- Conclusión: el trial ganador (#76) alcanza CV-RMSE=0.23347 con `n_estimators=950`, `max_depth=6`, `learning_rate≈0.026`

**Gráfica 3 — Diagnósticos del modelo final (test)**
- Tipo: panel triple (`plt.subplots(1, 3)`, figsize=(16, 4))
- (a) Real vs Predicho, (b) Histograma de residuos, (c) Q-Q plot
- Título: `"sale | XGBoost Optuna | Real vs Predicho"`

**Gráfica 4 — Feature importance (Top 20)**
- Tipo: barplot horizontal, figsize=(9, 6)
- Título: `"sale | XGBoost Optuna — Top 20 importancias"`
- Top 20 (ver sección 9)

### 8.3 `55_sale_rent_models.ipynb`

**Gráfica 1 — Diagnósticos SALE**
- Panel triple (Real vs Predicho, Residuos, Q-Q), figsize=(16, 4)
- Título: `"sale | XGBoost | Real vs Predicho"`

**Gráfica 2 — Feature importance SALE**
- Barplot horizontal, Top 20, figsize=(9, 6)
- Título: `"sale | XGBoost — Top 20 importancias"`

**Gráfica 3 — Diagnósticos RENT**
- Panel triple, figsize=(16, 4)
- Título: `"rent | XGBoost | Real vs Predicho"`

**Gráfica 4 — Feature importance RENT**
- Barplot horizontal, Top 20, figsize=(9, 6)
- Título: `"rent | XGBoost — Top 20 importancias"`

### 8.4 `55_input_result.ipynb` y `55_input_result_no_k_fold.ipynb`

Sin gráficas — son notebooks de predicción interactiva con output exclusivamente textual.

---

## 9. Interpretabilidad del modelo

### 9.1 Feature importance nativa de XGBoost

Se usa `model.feature_importances_` (gain-based por defecto en XGBoost). No se usa SHAP ni ninguna otra librería de interpretabilidad.

#### Feature importance — Modelo RENT (`53_boost_rent`, hiperparámetros Optuna)

Top 20 features:

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.180315 |
| `numero_dormitorios` | 0.114159 |
| `numero_banos` | 0.079953 |
| `precio_m2_municipio_media` | 0.068444 |
| `municipio_Camargo` | 0.058170 |
| `municipio_Santander` | 0.050953 |
| `municipio_otro` | 0.042223 |
| `distancia_min_playa_km` | 0.034337 |
| `municipio_Castro-Urdiales` | 0.034190 |
| `ratio_banos_superficie` | 0.033210 |
| `tipologia_unificada_piso` | 0.028880 |
| `score_cercania_servicios` | 0.025464 |
| `ratio_dormitorios_superficie` | 0.025460 |
| `es_exterior_piso` | 0.024330 |
| `tipologia_unificada_unifamiliar` | 0.023795 |
| `distancia_centro_municipio_km` | 0.022190 |
| `distancia_min_colegio_km` | 0.021699 |
| `tiene_garaje` | 0.021230 |
| `interaccion_planta_sin_ascensor_piso` | 0.021001 |
| `tiene_ascensor_piso` | 0.020587 |

> Insights económicos: en alquiler, la superficie (18%) y el número de dormitorios (11.4%) son los predictores dominantes. El precio medio de venta del municipio (`precio_m2_municipio_media`, 6.8%) actúa como proxy de zona sin causar leakage. `es_exterior_piso` (2.4%) tiene importancia modest pero presente, confirmando la preferencia del mercado por pisos exteriores. `municipio_Camargo` y `municipio_Santander` tienen las importancias más altas entre municipios (5.8% y 5.1%).

#### Feature importance — Modelo SALE (`53_boost_sale_optuna`, hiperparámetros Optuna)

Top 20 features:

| Feature | Importancia |
|---------|------------|
| `superficie_construida_m2` | 0.153270 |
| `numero_banos` | 0.120529 |
| `tiene_ascensor_piso` | 0.115420 |
| `municipio_Santoña` | 0.066357 |
| `numero_dormitorios` | 0.052209 |
| `tipologia_unificada_piso` | 0.051666 |
| `precio_m2_municipio_media` | 0.044408 |
| `tiene_garaje` | 0.039557 |
| `interaccion_planta_sin_ascensor_piso` | 0.032321 |
| `municipio_Noja` | 0.023757 |
| `municipio_Santander` | 0.023160 |
| `es_exterior_piso` | 0.019480 |
| `tipologia_unificada_unifamiliar` | 0.019462 |
| `latitud` | 0.016556 |
| `distancia_min_playa_km` | 0.013373 |
| `municipio_Ribamontan al Mar` | 0.013175 |
| `obra_nueva` | 0.012297 |
| `longitud` | 0.011906 |
| `municipio_Barcena de Cicero` | 0.011744 |
| `municipio_El Astillero` | 0.011700 |

> Insights económicos: en venta con Optuna, la superficie sigue liderando (15.3%) aunque con menor dominancia que en el modelo hardcodeado (23%). La incorporación de `tiene_ascensor_piso` como feature activa le permite ocupar el tercer puesto (11.5%), revelando que el ascensor es un factor de precio determinante en venta. `municipio_Santoña` destaca como el municipio individual más importante (6.6%), posiblemente por la brecha de precios respecto a municipios adyacentes. `latitud` y `longitud` tienen importancia conjunta de 2.8%, confirmando que la posición geográfica aporta señal más allá de los dummies de municipio.

### 9.2 SHAP

No se usa SHAP en ninguno de los cinco notebooks.

---

## 10. Herramienta de predicción individual (notebooks 55_input_*)

### 10.1 `55_input_result.ipynb` (split 80/20)

#### Carga y entrenamiento de modelos

No se cargan modelos de disco. Los modelos se entrenan en tiempo de ejecución al correr el notebook, replicando exactamente el proceso de `55_sale_rent_models.ipynb`:
1. Carga de CSVs gold
2. Limpieza de outliers
3. `build_X()` con medianas guardadas en `medians_sale` y `medians_rent`
4. Split 80/20
5. Entrenamiento de `model_sale` y `model_rent`
6. Cálculo de `sale_rmse_test` y `rent_rmse_test` (sobre el 20% de test) para usar como intervalo

#### Referencia geográfica

```python
GEO_COLS = [
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
    "precio_m2_municipio_media",
    "distancia_centro_municipio_km",
    "score_cercania_servicios",
    "latitud",     # añadido: feature activa en el nuevo modelo sale
    "longitud",    # añadido: feature activa en el nuevo modelo sale
]
```

`build_geo_ref(df)` calcula la mediana de estas variables por municipio. Permite asignar automáticamente los valores geográficos al inmueble de input a partir del municipio seleccionado. `latitud` y `longitud` se imputan con la mediana del municipio, no son inputs del usuario.

- Municipios disponibles en venta: 31
- Municipios disponibles en alquiler: 10

#### Lista completa de atributos de input

```python
MUNICIPIO      = "Santa Cruz de Bezana"  # nombre exacto del municipio
SUPERFICIE_M2  = 90                      # m² construidos (float)
N_DORMITORIOS  = 3                       # entero
N_BANOS        = 2                       # entero
TIENE_GARAJE   = True                    # bool
OBRA_NUEVA     = True                    # bool
TIPOLOGIA      = "piso"                  # "piso" o "unifamiliar"
PLANTA         = 1                       # entero (0=bajo); None si unifamiliar
ES_EXTERIOR    = True                    # bool; None si unifamiliar
TIENE_ASCENSOR = True                    # bool; None si unifamiliar
```

#### Preprocesamiento del input

La función `_build_row()` construye la fila de predicción:
1. Asigna features hedónicas directas del input.
2. Calcula features derivadas: `ratio_dormitorios_superficie`, `ratio_banos_superficie`, `interaccion_planta_sin_ascensor_piso`.
3. Extrae valores geográficos de `geo_ref.loc[municipio]` (medianas por municipio).
4. Pone a 0 todos los dummies de municipio y activa `municipio_{municipio}` o `municipio_otros` si no está en la lista.
5. Imputa cualquier valor `NaN` restante con `medians.get(col, 0)`.

#### Formato de salida

```
══════════════════════════════════════════════════════════
  90 m²  ·  3 dorm.  ·  2 baños  —  Santa Cruz de Bezana
  PISO  ·  Planta 1 · exterior · con ascensor · garaje · obra nueva
══════════════════════════════════════════════════════════

  Precio de venta estimado   :      314,381 €
  Intervalo error (±1σ)      : [   245,914 €  —     401,910 €]  (±28%)

  Alquiler mensual estimado  :        1,039 €/mes
  Intervalo error (±1σ)      : [       899 €/mes  —       1,201 €/mes]  (±16%)

  Rentabilidad bruta estim.  :         4.0 %
```

#### Lógica de negocio sobre la predicción

- **Intervalo ±1σ:** calculado como `[precio × exp(-RMSE_log), precio × exp(+RMSE_log)]` — intervalo asimétrico en euros porque la predicción es en escala logarítmica.
- **Rentabilidad bruta:** `(precio_alquiler × 12) / precio_venta × 100`.
- Si el municipio no está disponible en venta o alquiler, se muestra una advertencia y se omite ese precio — no hay fallback automático a `municipio_otro`.

---

### 10.2 `55_input_result_no_k_fold.ipynb` (100% datos)

#### Diferencias clave respecto a `55_input_result`

El notebook es structuralmente idéntico con una diferencia fundamental: **los modelos se entrenan sobre el 100% de los datos limpios**, sin split train/test.

```python
# En lugar de:
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(...)
model_sale.fit(Xs_tr, ys_tr)

# Aquí:
model_sale.fit(X_sale, y_sale)   # todo el dataset limpio
```

El intervalo de error ya no se puede calcular sobre un test real, por lo que se usan los **CV-RMSE hardcodeados** del modelo con split 80/20:

```python
SALE_CV_RMSE = 0.25922
RENT_CV_RMSE = 0.14622
```

> Nota: esta celda tiene un bug de escritura — el bloque de código se repite múltiples veces dentro del mismo contenido de la celda (artefacto de edición). El valor efectivo al ejecutar es el último: `SALE_CV_RMSE = 0.25922`, `RENT_CV_RMSE = 0.14622`.

**Output de entrenamiento:**

```
Sale  — filas para entrenamiento: 2543
Sale  — modelo entrenado sobre 100% (48 features)  |  CV-RMSE: 0.25922  →  ±29.6%
Rent  — filas para entrenamiento: 662
Rent  — modelo entrenado sobre 100% (27 features)  |  CV-RMSE: 0.14622  →  ±16%
```

#### Métricas sobre el dataset global

No se calculan métricas sobre el dataset global de entrenamiento. No hay cálculo explícito de R² o RMSE sobre `X_sale`/`X_rent` completo. El notebook solo usa el CV-RMSE importado como referencia de error.

---

### 10.3 Ejemplos de predicción

#### Ejemplo 1 — `55_input_result.ipynb` (modelo split 80/20)

**Input:**

| Atributo | Valor |
|----------|-------|
| Municipio | Santa Cruz de Bezana |
| Superficie | 90 m² |
| Dormitorios | 3 |
| Baños | 2 |
| Garaje | Sí |
| Obra nueva | Sí |
| Tipología | Piso |
| Planta | 1 |
| Exterior | Sí |
| Ascensor | Sí |

**Output:**

| Métrica | Valor |
|---------|-------|
| Precio de venta estimado | 314.381 € |
| Intervalo venta (±1σ) | [245.914 €, 401.910 €] |
| Error venta | ±28% |
| Alquiler mensual estimado | 1.039 €/mes |
| Intervalo alquiler (±1σ) | [899 €/mes, 1.201 €/mes] |
| Error alquiler | ±16% |
| Rentabilidad bruta estimada | 4.0% |

#### Ejemplo 2 — `55_input_result_no_k_fold.ipynb` (modelo 100% datos)

**Input:**

| Atributo | Valor |
|----------|-------|
| Municipio | Santander |
| Superficie | 200 m² |
| Dormitorios | 5 |
| Baños | 3 |
| Garaje | Sí |
| Obra nueva | No |
| Tipología | Unifamiliar |
| Planta | N/A (unifamiliar) |
| Exterior | N/A |
| Ascensor | N/A |

**Output:**

| Métrica | Valor |
|---------|-------|
| Precio de venta estimado | 688.664 € |
| Intervalo venta (±1σ) | [531.410 €, 892.453 €] |
| Error venta | ±30% |
| Alquiler mensual estimado | 1.560 €/mes |
| Intervalo alquiler (±1σ) | [1.348 €/mes, 1.806 €/mes] |
| Error alquiler | ±16% |
| Rentabilidad bruta estimada | 2.7% |

---

## 11. Notebook `55_sale_rent_models.ipynb`: integración de modelos

### 11.1 Función

`55_sale_rent_models.ipynb` replica los dos modelos de los notebooks `53_*` en un único notebook de referencia. Permite evaluar M-SALE y M-RENT de forma conjunta y produce los outputs canónicos que se copian a los notebooks `55_input_*`.

### 11.2 Reutilización de hiperparámetros

Los hiperparámetros **no se cargan de disco ni se importan de un módulo**. Están **hardcodeados directamente** en el notebook, copiados manualmente de los outputs de los notebooks `53_*`:

```python
# Sale: Optuna (copiado del output de 53_boost_sale_optuna, trial #76)
XGB_PARAMS = dict(
    n_estimators=950, max_depth=6, learning_rate=0.026390709496515886,
    subsample=0.6705806157309522, colsample_bytree=0.7312116009128224,
    min_child_weight=9, reg_lambda=1.6752171349993321,
    reg_alpha=0.3505729220414384, gamma=0.005208850498085864, ...
)

# Rent: hardcodeado (copiado del output de Optuna en 53_boost_rent)
XGB_PARAMS_RENT = dict(
    n_estimators=1000, max_depth=6, learning_rate=0.011737724486287017,
    subsample=0.6046283826235364, colsample_bytree=0.8531891359517659,
    ...
)
```

### 11.3 Entrenamiento

Entrena los modelos de nuevo (no carga pesos de disco). El proceso es idéntico al de los notebooks `53_*`.

### 11.4 Evaluación conjunta

Produce los mismos outputs de métricas que `53_boost_sale_optuna` y `53_boost_rent`, pero en un único notebook. Permite comparar visualmente M-SALE y M-RENT side by side. No hay una celda de resumen combinado — son dos secciones independientes (`## Modelo SALE` y `## Modelo RENT`).

### 11.5 Lógica adicional respecto a los 53_*

- Ambos modelos usan Optuna: sale con `XGB_PARAMS` de `53_boost_sale_optuna`, rent con `XGB_PARAMS_RENT` de `53_boost_rent`.
- `BASE_FEATURES` (sale) ahora incluye `latitud`, `longitud`, `es_exterior_piso`, `tiene_ascensor_piso` — features activas en sale desde `53_boost_sale_optuna`.
- Define una función `build_X_rent()` específica para rent (vs la `build_X()` genérica de sale), ya que rent sigue sin tener `latitud`/`longitud`.
- Los comentarios explicitan el origen de los parámetros: `"Hiperparámetros óptimos de Optuna (53_boost_sale_optuna, trial #76, CV-RMSE=0.23347, test R²=0.853)"` para sale.
- Los CV-RMSE usados en `55_input_result_no_k_fold` son: sale = 0.23347, rent = 0.14622.

---

## 12. Dependencias técnicas

### 12.1 Librerías externas

| Librería | Uso | Versión |
|----------|-----|---------|
| `xgboost` | `XGBRegressor` — modelo principal | No determinable en los notebooks |
| `optuna` | Búsqueda de hiperparámetros (`53_boost_rent` y `53_boost_sale_optuna`) | No determinable |
| `sklearn` | `SimpleImputer`, `KFold`, `cross_val_score`, `train_test_split`, métricas | No determinable |
| `pandas` | Manipulación de datos | No determinable |
| `numpy` | Cálculos numéricos | No determinable |
| `matplotlib` | Visualizaciones | No determinable |
| `scipy` | `scipy.stats.probplot` — Q-Q plots | No determinable |
| `pathlib` | Detección de raíz del proyecto | Stdlib |

El entorno es Python 3.14 (visible en el path del venv: `.venv/lib/python3.14/`).

**Advertencia de entorno:**

```
TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.
```

La barra de progreso de Optuna puede no mostrarse correctamente si no está instalado `ipywidgets`.

### 12.2 Módulos internos (`src/`)

No se importa ningún módulo interno del directorio `src/`. Todo el código es self-contained en cada notebook.

### 12.3 Detección de raíz del proyecto

Función `find_project_root()` en todos los notebooks:

```python
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "gold").exists():
            return p
    raise FileNotFoundError("No se encontró la raíz del proyecto (data/gold)")

PROJECT_ROOT = find_project_root(Path.cwd().resolve())
```

Esto hace los notebooks portables sin hardcodear rutas absolutas.

---

## 13. Limitaciones, advertencias y observaciones

### 13.1 TODOs y comentarios del autor

**En `53_boost_rent.ipynb`:**

> "El gold no incluye distancia a hospitales, universidades ni transporte público. Son las variables con mayor potencial de mejora — incorporarlas en el pipeline de transformaciones sería el siguiente salto."

Features sugeridas no disponibles aún:
- `distancia_min_hospital_km`
- `distancia_min_universidad_km`
- `distancia_min_parada_bus_km` / `distancia_min_tren_km`

### 13.2 Features comentadas (inactivas)

En `53_boost_sale_optuna.ipynb`, todas las features relevantes están activas: `latitud`, `longitud`, `es_exterior_piso` y `tiene_ascensor_piso` se incluyen en `BASE_FEATURES` (al contrario que en el notebook anterior `53_boost_sale`, donde estaban comentadas). Esto eleva el total de features de 48 a 52.

### 13.3 Leakage evitado — advertencia explícita

`rentabilidad_bruta_zona` fue descartada explícitamente en `53_boost_rent` con comentario:

> "Fue descartada — usaba `precio` (= exp(log_precio)) directamente como feature, lo que hacía que el modelo pudiera reconstruir el target casi perfectamente y disparaba el R² artificialmente a >0.9."

Esto es un caso documentado de **data leakage** que el autor detectó y corrigió.

### 13.4 Consistencia entre notebooks

Con `53_boost_sale_optuna`, los CV-RMSE son coherentes en todos los notebooks:
- `53_boost_sale_optuna`: CV-RMSE = 0.23347 (sobre train de 2034 filas)
- `55_sale_rent_models`: mismo valor al usar los mismos params y split
- `55_input_result_no_k_fold`: `SALE_CV_RMSE = 0.23347` (hardcodeado, coincide)

`es_exterior_piso` y `tiene_ascensor_piso` ahora están activas en sale y en rent — la asimetría de features entre modelos queda reducida a `latitud`/`longitud` (presentes en sale, ausentes en rent).

### 13.5 Bug de edición en `55_input_result_no_k_fold.ipynb` (corregido)

La celda 4 del notebook tenía el mismo bloque de código repetido muchas veces (artefacto de edición). **Este bug ha sido corregido** — la celda ahora contiene una única asignación limpia:

```python
# CV-RMSE de 55_sale_rent_models (5-fold sobre train completo)
# Sale: 53_boost_sale_optuna (Optuna, trial #76)
SALE_CV_RMSE = 0.23347
RENT_CV_RMSE = 0.14622
```

### 13.6 Sin persistencia de modelos

Los modelos no se guardan en disco. Cada vez que se ejecuta un notebook de predicción (`55_input_result*`) se re-entrena completo el modelo desde cero. Para un sistema de producción esto implica un tiempo de espera de entrenamiento en cada sesión.

### 13.7 Sobreajuste del modelo RENT

El ratio RMSE test/train de M-RENT es 1.56 y el delta R² es 0.244, lo que indica sobreajuste moderado. Con solo 529 filas de train y 9 hiperparámetros optimizados por Optuna, existe riesgo de haber sobreajustado el espacio de búsqueda al CV de train. El CV-RMSE (0.146) y el RMSE_test (0.145) están muy alineados, lo que sugiere que el sobreajuste de hiperparámetros es pequeño.

### 13.8 Sobreajuste pronunciado en M-SALE (Optuna)

El ratio RMSE test/train de M-SALE con Optuna es **2.86** (0.222 / 0.078), considerablemente mayor que el del modelo hardcodeado (1.56). Esto indica que el modelo Optuna sobreajusta más el conjunto de train. Sin embargo, el CV-RMSE (0.233) y el RMSE_test (0.222) están razonablemente alineados, lo que sugiere que el sobreajuste se detecta a través de la validación cruzada. El resultado neto en test es mejor (R²=0.853 vs 0.836), por lo que el modelo Optuna generaliza mejor aunque muestre mayor overfit aparente en train.

### 13.9 Municipios no disponibles en alquiler

Solo 10 municipios están disponibles para predicción de alquiler vs 31 para venta. Si el usuario introduce un municipio disponible en venta pero no en alquiler (e.g., `Noja`, `Suances`), el estimador muestra una advertencia y omite la predicción de alquiler (y por tanto la rentabilidad bruta).

### 13.10 Decisiones metodológicas sin justificación explícita en el código

- **`MIN_MUNI_OBS = 10`**: umbral para colapsar municipios. Valor elegido sin justificación estadística visible.
- **`IQR_FACTOR = 1.5`**: más estricto que el IQR×3 usado en notebooks anteriores de la serie. El autor menciona que es "más estricto" pero no justifica el cambio.
- **`PRECIO_M2_VACACIONAL_UMBRAL = 18 €/m²`**: umbral para detectar alquileres turísticos. Elegido por inspección visual de la distribución, no por criterio estadístico formal.
- **`PRECIO_M2_MIN = 1000 €/m²`** (sale): suelo para detectar propiedades no-residenciales. Criterio ad-hoc basado en la cola inferior del Q-Q plot.
