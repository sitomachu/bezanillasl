# Documentación técnica: Modelos XGBoost optimizados — BezanillaSL

> **Cómo mantener este documento actualizado:**
> Los valores numéricos clave (hiperparámetros, métricas, features) se almacenan en
> `data/model_results/params_rent.json` y `data/model_results/params_sale.json`.
> Cuando se re-ejecute un notebook `53_boost_*`, actualiza las secciones 4, 6, 7 y 9
> con los nuevos valores de esos JSON. El resto del documento (arquitectura, flujo, preprocesamiento)
> solo cambia si se modifica la lógica de los notebooks.

---

## 1. Resumen ejecutivo del sistema de modelización

BezanillaSL modela los precios del mercado inmobiliario de **Cantabria** (España) mediante dos XGBoost hedónicos, uno para **venta** y otro para **alquiler**. La variable objetivo es siempre **`log_precio`** (logaritmo natural del precio en euros), lo que reduce la heterocedasticidad y simetriza la distribución. Para convertir una predicción a euros: `precio = exp(log_precio_predicho)`.

### Modelos definidos

| ID | Nombre funcional | Dataset | Target | Metodología |
|----|-----------------|---------|--------|-------------|
| M-SALE | XGBoost Sale | `final_sale_idealistaAPI.csv` | `log_precio` | Optuna 100 trials, CV-RMSE 5-fold (`53_boost_sale_optuna`) |
| M-RENT | XGBoost Rent | `final_rent_idealistaAPI.csv` | `log_precio` | Optuna 100 trials, CV-RMSE 5-fold (`53_boost_rent`) |

### Flujo de dependencias entre notebooks

```
53_boost_rent.ipynb ──────────────────────────────────────────────────┐
  └─ Optuna 100 trials → params_rent.json                             │
                                                                       ▼
53_boost_sale_optuna.ipynb ────────────────────────────┐    55_sale_rent_models.ipynb
  └─ Optuna 100 trials → params_sale.json              │    └─ Lee params_*.json
                                                        │       Reentrena M-SALE y M-RENT
                                                        │       Evalúa ambos conjuntamente
                                                        ▼
                                           55_input_result.ipynb
                                               └─ Lee params_*.json
                                                  Modelos sobre 80% datos
                                                  Predicción individual ±1σ

                                           55_input_result_no_k_fold.ipynb
                                               └─ Lee params_*.json
                                                  Modelos sobre 100% datos
                                                  Usa CV-RMSE como intervalo
```

Los cinco notebooks son:
1. **`53_boost_rent.ipynb`**: EDA + Optuna para M-RENT → exporta `params_rent.json`
2. **`53_boost_sale_optuna.ipynb`**: EDA + Optuna para M-SALE → exporta `params_sale.json`
3. **`55_sale_rent_models.ipynb`**: Lee params JSON, reentrena ambos modelos con evaluación conjunta
4. **`55_input_result.ipynb`**: Predicción interactiva individual (split 80/20)
5. **`55_input_result_no_k_fold.ipynb`**: Predicción interactiva individual (100% datos, más robusto)

### Sistema de persistencia de parámetros (params JSON)

Los notebooks `53_boost_*` exportan al finalizar Optuna un JSON con todos los parámetros relevantes para reproducir el modelo. Los notebooks `55_*` los leen directamente en lugar de hardcodear valores:

```python
# 53_boost_rent.ipynb — exporta al terminar Optuna
params_out = {
    "notebook": "53_boost_rent",
    "generated_at": "<timestamp ISO>",
    "target_col": "log_precio",
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "min_muni_obs": 10,
    "optuna_trials": 100,
    "base_features": [...],
    "xgb_params": {...},
    "optuna_best_trial": <int>,
    "optuna_cv_rmse": <float>,
    "test_rmse": <float>,
    "test_r2": <float>,
    "mun_means_sale": {...},    # medianas precio_m2 venta por municipio
    "mun_global_mean_sale": <float>,
}

# 55_input_result.ipynb — lee al inicio
with open(RENT_PARAMS_PATH) as f:
    rent_cfg = json.load(f)
XGB_PARAMS_RENT = rent_cfg["xgb_params"]
BASE_FEATURES_RENT = rent_cfg["base_features"]
```

Esto garantiza consistencia entre todos los notebooks `55_*` y elimina el riesgo de inconsistencias por copiar-pegar valores manualmente.

---

## 2. Datos de entrada de cada modelo

### 2.1 Notebook `53_boost_rent.ipynb` — Dataset de alquiler

| Parámetro | Valor |
|-----------|-------|
| Ruta del archivo | `data/gold/final_rent_idealistaAPI.csv` |
| Filas cargadas (outliers eliminados upstream) | **661** |
| Columnas totales | 47 |

> **Nota:** La limpieza de outliers se realiza en `idealistaAPI_processing_outliers.ipynb`, no en este notebook. El CSV gold ya contiene los datos limpios cuando llega aquí.

**Variable objetivo**

| Parámetro | Valor |
|-----------|-------|
| Nombre | `log_precio` |
| Descripción | Logaritmo natural del precio mensual en euros |
| Media | 6.8370 |
| Std | 0.2470 |

**Target alternativo evaluado (descartado):** `log_precio_m2` — log(precio/m²). Se compara con `log_precio` mediante CV-RMSE en XGBoost base:

| Target | CV-RMSE | CV-R² |
|--------|---------|-------|
| `log_precio` | 0.14070 | 0.67246 |
| `log_precio_m2` | 0.14430 | 0.56896 |

`log_precio` es seleccionado por mayor CV-R² (0.672 vs 0.569).

**Features finales (23 total)**

| Feature | Tipo | Nota |
|---------|------|------|
| `superficie_construida_m2` | Continua | Feature hedónica principal |
| `numero_dormitorios` | Discreta | |
| `numero_banos` | Discreta | |
| `planta_num` | Discreta | NaN para propiedades unifamiliar |
| `es_exterior_piso` | Dummy (0/1) | Específica de pisos; NaN para unifamiliar |
| `tiene_ascensor_piso` | Dummy (0/1) | Específica de pisos; NaN para unifamiliar |
| `tiene_garaje` | Dummy (0/1) | |
| `obra_nueva` | Dummy (0/1) | Importancia ≈ 0 en modelo actual |
| `distancia_min_playa_km` | Continua | |
| `distancia_min_supermercado_km` | Continua | |
| `distancia_min_colegio_km` | Continua | |
| `interaccion_planta_sin_ascensor_piso` | Continua | `planta_num × (1 - tiene_ascensor_piso)`; NaN para unifamiliar |
| `distancia_centro_municipio_km` | Continua | |
| `score_cercania_servicios` | Continua | Índice compuesto de distancias |
| `tipologia_unificada_piso` | Dummy (0/1) | |
| `tipologia_unificada_unifamiliar` | Dummy (0/1) | |
| `municipio_Camargo` | Dummy OHE | Municipio con ≥ 10 obs en rent |
| `municipio_Castro-Urdiales` | Dummy OHE | |
| `municipio_El Astillero` | Dummy OHE | |
| `municipio_Piélagos` | Dummy OHE | |
| `municipio_Santander` | Dummy OHE | |
| `municipio_Torrelavega` | Dummy OHE | |
| `municipio_otro` | Dummy OHE | Municipios con < 10 obs en rent |

> **Features eliminadas respecto a versiones anteriores:** `precio_m2_municipio_media` (eliminada por el usuario al experimentar con el espacio de features), `ratio_dormitorios_superficie`, `ratio_banos_superficie`.

**Exclusiones explícitas por leakage:** `precio`, `precio_m2`, `precio_m2_raw`, `log_precio_m2`, `rentabilidad_bruta_zona` — esta última fue descartada porque usaba `precio` (= exp(target)) directamente como feature, disparando el R² artificialmente a > 0.9.

**Tamaño train/test:**

| Split | Filas | % |
|-------|-------|---|
| Train | ~529 | 80% |
| Test | ~132 | 20% |
| Total | 661 | 100% |

---

### 2.2 Notebook `53_boost_sale_optuna.ipynb` — Dataset de venta

| Parámetro | Valor |
|-----------|-------|
| Ruta del archivo | `data/gold/final_sale_idealistaAPI.csv` |
| Filas cargadas (outliers eliminados upstream) | **2532** |
| Columnas totales | 70 |

> **Nota:** La limpieza de outliers se realiza en `idealistaAPI_processing_outliers.ipynb`, no en este notebook.

**Variable objetivo**

| Parámetro | Valor |
|-----------|-------|
| Nombre | `log_precio` |
| Descripción | Logaritmo natural del precio de venta en euros |
| Media | 12.609 |
| Std | 0.569 |

**Features finales (~47 total)**

| Feature | Tipo | Nota |
|---------|------|------|
| `superficie_construida_m2` | Continua | |
| `numero_dormitorios` | Discreta | |
| `numero_banos` | Discreta | Correlación con log_precio: 0.6228 |
| `planta_num` | Discreta | NaN para unifamiliar |
| `es_exterior_piso` | Dummy (0/1) | NaN para unifamiliar |
| `tiene_ascensor_piso` | Dummy (0/1) | NaN para unifamiliar |
| `tiene_garaje` | Dummy (0/1) | |
| `obra_nueva` | Dummy (0/1) | |
| `distancia_min_playa_km` | Continua | |
| `distancia_min_supermercado_km` | Continua | |
| `distancia_min_colegio_km` | Continua | |
| `precio_m2_municipio_media` | Continua | Precio medio de VENTA por municipio (anti-leakage) |
| `interaccion_planta_sin_ascensor_piso` | Continua | NaN para unifamiliar |
| `distancia_centro_municipio_km` | Continua | |
| `score_cercania_servicios` | Continua | |
| `tipologia_unificada_piso` | Dummy (0/1) | |
| `tipologia_unificada_unifamiliar` | Dummy (0/1) | |
| `municipio_Ampuero` … `municipio_otros` | Dummy OHE | ~30 municipios (ver lista completa abajo) |

> **Features eliminadas respecto a versiones anteriores:** `latitud`, `longitud`, `ratio_dormitorios_superficie`, `ratio_banos_superficie`.

**Municipios OHE en sale (~30):** `Ampuero`, `Barcena de Cicero`, `Camargo`, `Castro-Urdiales`, `Colindres`, `Cudon`, `El Astillero`, `Guarnizo`, `Laredo`, `Liendo`, `Limpias`, `Marina de Cudeyo`, `Miengo`, `Mogro`, `Noja`, `Ortuella`, `Piélagos`, `Polanco`, `Ribamontan al Mar`, `Ribamontan al Monte`, `Santa Cruz de Bezana`, `Santander`, `Santoña`, `Santurtzi`, `Suances`, `Torrelavega`, `Villaescusa`, `Viveda`, `Voto`, `municipio_otro`, `municipio_otros`.

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
| Train | ~2026 | 80% |
| Test | ~506 | 20% |
| Total | 2532 | 100% |

---

## 3. Preprocesamiento y feature engineering

### 3.1 Pipeline de limpieza de outliers (migrado a notebook upstream)

**A partir de la versión actual, toda la limpieza de outliers se realiza en `idealistaAPI_processing_outliers.ipynb` y el CSV gold ya está limpio al llegar a los notebooks ML.** Los notebooks `53_boost_*` y `55_*` no aplican ningún filtro de outliers — simplemente cargan el CSV gold.

#### Filtros de alquiler (aplicados en `idealistaAPI_processing_outliers.ipynb`)

| Paso | Filtro | Criterio | Filas eliminadas aprox. |
|------|--------|----------|------------------------|
| 1 | Filtro vacacional | `precio_m2 > 18 €/m²/mes` | ~61 filas (8.1%) |
| 2 | Filtro suelo | `precio_m2 < 6 €/m²/mes` | ~12 filas (1.7%) |
| 3 | IQR×1.5 sobre `log_precio` | Extremos de precio absoluto | ~19 filas (2.8%) |

> El umbral vacacional de 18 €/m²/mes se eligió por inspección visual de la distribución de `precio_m2`. El suelo de 6 €/m²/mes elimina pisos grandes con precios anómalamente bajos (garajes, errores de entrada, propiedades fuera de mercado).

#### Filtros de venta (aplicados en `idealistaAPI_processing_outliers.ipynb`)

| Paso | Filtro | Criterio | Filas eliminadas aprox. |
|------|--------|----------|------------------------|
| 1 | IQR×1.5 sobre `log_precio` | Extremos de precio absoluto | 0 filas (0%) |
| 2 | Suelo precio/m² | `precio_m2 >= 1000 €/m²` | ~151 filas (5.6%) |

> El suelo de 1.000 €/m² elimina propiedades anómalamente baratas (ruinas, no-residencial, errores). Se detectó que estas propiedades generaban residuos extremos (hasta −1.25) en el Q-Q plot.

> **Nota adicional (gold notebook):** El notebook gold también aplica un filtro exacto de `precio_m2` para capturar casos límite del redondeo de `priceByArea` de la API de Idealista. Este filtro está en el notebook gold (`idealistaAPI_processing_gold.ipynb` o equivalente) y actúa antes de que el CSV llegue a `idealistaAPI_processing_outliers.ipynb`.

### 3.2 Transformaciones sobre la variable objetivo

- **`log_precio`** es la variable objetivo en ambos modelos (logaritmo natural del precio en euros). Ya está precalculada en el CSV gold.
- Para convertir predicciones a precio real: `precio = exp(log_precio_predicho)`.
- No se aplica ninguna transformación adicional al target dentro de los notebooks ML.
- El target alternativo `log_precio_m2 = log(precio / superficie_m2)` se calcula en `53_boost_rent` solo para comparar y se descarta.

### 3.3 Encoding de variables categóricas

**Municipios — proceso en dos etapas:**

**Etapa 1 (notebook gold):** Los municipios con menos de `MIN_MUNI_OBS = 10` observaciones se colapsan en la columna `municipio_otro` dentro del CSV gold. Esta operación es definitiva — no es reversible una vez generado el CSV.

**Etapa 2 (función `build_X()`):** Al construir la matriz de features, `build_X()` puede colapsar adicionalmente cualquier columna `municipio_*` que tenga menos de `min_muni_obs` observaciones en el dataframe activo en ese momento en la columna `municipio_otros` (operación `max` por fila de los dummies de municipios raros).

> **Distinción importante:** `municipio_otro` (sin 's') es una categoría del gold CSV, generada por el pipeline de datos. `municipio_otros` (con 's') es una columna generada dinámicamente por `build_X()` para colapsar municipios insuficientes en el split de train. Son dos categorías distintas.

**Tipología:** `tipologia_unificada_piso` y `tipologia_unificada_unifamiliar` ya están precalculadas en el gold como dummies.

### 3.4 Manejo de NaN para features específicas de piso

Las siguientes features solo tienen sentido para propiedades de tipo piso. Para propiedades unifamiliares, se asigna `NaN` en lugar de un valor imputado:

```python
# En build_X() — propiedades unifamiliares reciben NaN en features de piso
PISO_ONLY_FEATURES = [
    "planta_num",
    "es_exterior_piso",
    "tiene_ascensor_piso",
    "interaccion_planta_sin_ascensor_piso",
]
# df.loc[df["tipologia_unificada_unifamiliar"] == 1, PISO_ONLY_FEATURES] = NaN
```

XGBoost maneja los `NaN` de forma nativa: en cada nodo de decisión, aprende si los registros con `NaN` deben ir a la rama izquierda o derecha, optimizando la ganancia. Esto es correcto conceptualmente — una unifamiliar no tiene "planta", el valor es realmente desconocido/no aplica, no un cero ni una mediana.

`SimpleImputer(strategy="median")` solo se aplica a las features no-piso para rellenar nulos genuinos del dataset (valores perdidos en la API).

### 3.5 Función `build_X()`

La función `build_X(df, base_features, min_muni_obs)` aplica en este orden:
1. Selección de las columnas de `base_features` disponibles en el dataframe.
2. Detección de columnas `municipio_*` y colapso de las que tienen < `min_muni_obs` en `municipio_otros`.
3. Construcción de `X_raw` con `base_features + mun_final`.
4. Asignación de `NaN` a features piso-only en registros unifamiliares.
5. Cálculo de medianas (`medians = X_raw.median()`) — guardadas para imputación de inputs externos.
6. **Imputación** de nulos (salvo NaN de unifamiliar) con `SimpleImputer(strategy="median")`.
7. Los árboles de decisión son invariantes a la escala — **no se aplica estandarización**.

### 3.6 Feature engineering (variables derivadas activas)

| Feature derivada | Fórmula | Modelos |
|-----------------|---------|---------|
| `interaccion_planta_sin_ascensor_piso` | `planta_num × (1 - tiene_ascensor_piso)` | Sale y Rent |

> **Nota:** `ratio_dormitorios_superficie` y `ratio_banos_superficie` estaban en versiones anteriores pero fueron eliminadas. `log_precio_m2` solo se usa en rent para comparar targets y no se incluye como feature.

### 3.7 Separación train/test

```python
train_test_split(X, y, test_size=0.20, random_state=42)
```

- Método: división aleatoria simple (sin estratificación).
- `random_state = 42` en todos los notebooks y modelos.
- El test se usa exclusivamente para evaluación final — nunca entra en la optimización de Optuna ni en la validación cruzada interna.

---

## 4. Optimización de hiperparámetros con Optuna

### 4.1 Contexto

Ambos modelos usan Optuna con 100 trials. `53_boost_sale_optuna.ipynb` realiza la búsqueda para venta; `53_boost_rent.ipynb` para alquiler. Los mejores params se exportan a JSON para ser consumidos por los notebooks `55_*`.

### 4.2 Configuración del estudio

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
- **Verbosidad:** `optuna.logging.set_verbosity(optuna.logging.WARNING)`.

### 4.3 Función objetivo

```python
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators      = ...,
        max_depth         = ...,
        ...
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

La métrica es **CV-RMSE** (5-fold KFold sobre `X_train`). El test no se toca durante la búsqueda.

### 4.4 Espacio de búsqueda

#### Modelo SALE (`53_boost_sale_optuna`)

| Hiperparámetro | Tipo | Rango | Escala |
|----------------|------|-------|--------|
| `n_estimators` | `int` | [200, 1000], step=50 | lineal |
| `max_depth` | `int` | [3, 7] | lineal |
| `learning_rate` | `float` | [0.01, 0.20] | log |
| `subsample` | `float` | [0.5, 1.0] | lineal |
| `colsample_bytree` | `float` | [0.5, 1.0] | lineal |
| `min_child_weight` | `int` | [1, 15] | lineal |
| `reg_lambda` | `float` | [0.1, 10.0] | log |
| `reg_alpha` | `float` | [1e-3, 5.0] | log |
| `gamma` | `float` | [0.0, 5.0] | lineal |

#### Modelo RENT (`53_boost_rent`) — espacio corregido

El espacio de búsqueda de rent fue ajustado después de detectar que la configuración original producía importancias cero en muchos municipios y riesgo de sobreajuste. Ver sección 13.7 para la descripción del problema y su diagnóstico.

| Hiperparámetro | Tipo | Rango actual | Rango anterior | Razón del cambio |
|----------------|------|-------------|----------------|-----------------|
| `n_estimators` | `int` | [200, 1500] | [200, 1000], step=50 | Más espacio para learning rates bajos |
| `max_depth` | `int` | [3, 5] | [3, 6] | Limitar capacidad, reducir sobreajuste |
| `learning_rate` | `float` | [0.01, 0.30], log | [0.01, 0.20], log | — |
| `subsample` | `float` | [0.5, 0.85] | [0.5, 1.0] | Evitar subsample≈1 (árboles correlados, sobreajuste) |
| `colsample_bytree` | `float` | [0.5, 1.0] | [0.5, 1.0] | Sin cambio |
| `min_child_weight` | `int` | [1, 6] | [1, 15] | Valores altos bloqueaban splits de municipios pequeños |
| `reg_lambda` | `float` | [0.1, 10.0], log | [0.1, 10.0], log | Sin cambio |
| `reg_alpha` | `float` | [1e-4, 1.0], log | [1e-3, 5.0], log | Evitar regularización L1 excesiva |
| `gamma` | `float` | [0.0, 0.05] | [0.0, 5.0] | Valores altos bloqueaban todos los splits de municipio |

> **Diagnóstico:** Con el espacio original, Optuna encontraba soluciones degeneradas donde `gamma=0.162` y `min_child_weight=13` combinados con `subsample=0.52` impedían que los municipios pequeños (< `min_child_weight / subsample` filas visibles por árbol) pudieran participar en ningún split, resultando en importancia=0 para casi todas las columnas de municipio.

### 4.5 Mejores hiperparámetros encontrados

> Los valores de esta sección provienen directamente de los archivos JSON. Para actualizar, editar los bloques de código con los valores del nuevo JSON.

#### Sale — `data/model_results/params_sale.json`

**Trial ganador: #68 de 100 | CV-RMSE: 0.23445**

```python
XGB_PARAMS_SALE = {
    "n_estimators":     900,
    "max_depth":        6,
    "learning_rate":    0.013693938318582058,
    "subsample":        0.6250722201248828,
    "colsample_bytree": 0.7441334899841819,
    "min_child_weight": 1,
    "reg_lambda":       0.8039267216172503,
    "reg_alpha":        0.22243915579180695,
    "gamma":            0.0010446114637011929,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
```

#### Rent — `data/model_results/params_rent.json`

**Trial ganador: #62 de 100 | CV-RMSE: 0.14791**

```python
XGB_PARAMS_RENT = {
    "n_estimators":     962,
    "max_depth":        4,
    "learning_rate":    0.014822599377963932,
    "subsample":        0.7067172942609928,
    "colsample_bytree": 0.5183256429375686,
    "min_child_weight": 2,
    "reg_lambda":       8.30220900347487,
    "reg_alpha":        0.0011392773182910022,
    "gamma":            0.04803719866132008,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
```

### 4.6 Validación cruzada dentro de cada trial

En cada trial se entrena XGBoost con `KFold(n_splits=5, shuffle=True, random_state=42)` sobre `X_train`. El CV-RMSE resultante es el valor de la función objetivo. Idéntico en sale y rent.

### 4.7 Gráficas de Optuna (Sale y Rent)

Ambos notebooks generan las mismas dos gráficas:

**Gráfica 1 — Curva de convergencia acumulada:**
- Tipo: línea; Eje X: trial (0–99); Eje Y: mejor CV-RMSE acumulado

**Gráfica 2 — Dispersión de todos los trials:**
- Tipo: scatter + línea; CV-RMSE por trial (azul) + mejor acumulado (rojo)

Ambas en `figsize=(14, 4)`, `plt.subplots(1, 2)`.

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
| `55_input_result` | No | Sí — split 80/20; RMSE_test usado como intervalo |
| `55_input_result_no_k_fold` | No | No — entrena sobre 100%; usa CV-RMSE del JSON |

### 5.3 Métricas CV — `53_boost_rent`

- CV-RMSE (5-fold, sobre X_train): **0.14791** (trial #62)

### 5.4 Métricas CV — `53_boost_sale_optuna`

- CV-RMSE (5-fold, sobre X_train): **0.23445** (trial #68)

### 5.5 Diferencia entre `55_input_result` y `55_input_result_no_k_fold`

| Aspecto | `55_input_result` | `55_input_result_no_k_fold` |
|---------|------------------|---------------------------|
| Datos de entrenamiento | 80% | 100% |
| Evaluación | RMSE sobre test 20% | CV-RMSE del JSON |
| Intervalo de error | `±1σ = ±RMSE_test` | `±1σ = ±CV_RMSE` (del JSON) |
| Municipios rent disponibles | 54 (via geo_ref extendido) | 54 (via geo_ref extendido) |
| Fiabilidad del intervalo | Basado en un único split 20% | Basado en 5-fold CV — más robusto |
| Justificación | Evaluación + predicción | Producción — el modelo ve más datos |

---

## 6. Entrenamiento final del modelo

### 6.1 Modelo SALE

**Hiperparámetros finales (del JSON `params_sale.json`):**

Idénticos a la sección 4.5. Parámetros fijos: `objective = reg:squarederror`, `random_state=42`, `n_jobs=-1`.

| Parámetro | Valor |
|-----------|-------|
| Tamaño train | ~2026 filas (80% de 2532) |
| `early_stopping_rounds` | No aplicado |

### 6.2 Modelo RENT

**Hiperparámetros finales (del JSON `params_rent.json`):**

Idénticos a la sección 4.5. Parámetros fijos: `objective = reg:squarederror`, `random_state=42`, `n_jobs=-1`.

| Parámetro | Valor |
|-----------|-------|
| Tamaño train | ~529 filas (80% de 661) |
| `early_stopping_rounds` | No aplicado |

### 6.3 Persistencia de modelos en disco

**No se guarda ningún modelo en disco** en ninguno de los cinco notebooks. Los modelos se entrenan en memoria y se usan directamente en la misma sesión. Para reutilizar un modelo es necesario re-ejecutar el notebook.

---

## 7. Evaluación y métricas de rendimiento

> Los valores de test R² y RMSE provienen de los JSON (fuente autoritativa). Los valores de train son aproximados, obtenidos en la misma sesión de ejecución.

### 7.1 Métricas sobre conjunto de test

#### Modelo SALE

| Split | RMSE | R² | Notas |
|-------|------|----|-------|
| Train (aprox.) | 0.110 | 0.963 | Aproximado |
| CV-RMSE (5-fold) | 0.23445 | — | Fuente: `params_sale.json` |
| Test | **0.23498** | **0.8313** | Fuente: `params_sale.json` |

- Error en escala original: exp(0.23498) − 1 = **±26.5%**
- Delta R² (train − test) ≈ 0.132 (sobreajuste moderado; ver sección 13.8)

#### Modelo RENT

| Split | RMSE | R² | Notas |
|-------|------|----|-------|
| Train (aprox.) | 0.115 | 0.783 | Aproximado |
| CV-RMSE (5-fold) | 0.14791 | — | Fuente: `params_rent.json` |
| Test | **0.15489** | **0.59922** | Fuente: `params_rent.json` |

- Error en escala original: exp(0.15489) − 1 = **±16.7%**
- Delta R² (train − test) ≈ 0.184 (sobreajuste moderado)

### 7.2 Comparativa entre modelos

| Modelo | CV-RMSE | RMSE_test | R²_test | Error % (escala original) |
|--------|---------|-----------|---------|--------------------------|
| M-SALE | 0.23445 | 0.23498 | 0.831 | ±26.5% |
| M-RENT | 0.14791 | 0.15489 | 0.599 | ±16.7% |

M-SALE tiene mayor R² de test (0.831 vs 0.599) pero mayor error porcentual en euros. M-RENT predice con menor error relativo pero explica menos varianza — probablemente porque el mercado de alquiler en Cantabria tiene más factores no capturados (turismo, estacionalidad, negociación individual). El CV-RMSE de sale (0.235) y el RMSE_test (0.235) están muy alineados, lo que indica que la validación cruzada es un buen estimador del error real. En rent también (0.148 vs 0.155).

### 7.3 Modelo `55_input_result_no_k_fold` (100% datos)

| Modelo | CV-RMSE (del JSON) | Filas entrenamiento | Error % |
|--------|-------------------|--------------------|---------| 
| Sale | 0.23445 | 2532 | ±26.5% |
| Rent | 0.14791 | 661 | ±16.0% |

### 7.4 Análisis de residuos

Para cada modelo se genera un panel de tres gráficas diagnósticas sobre el conjunto de test:

1. **Real vs Predicho (scatter):** puntos en azul, línea identidad en rojo discontinuo.
2. **Histograma de residuos:** `y_test - pred_test`, línea vertical en 0. Evalúa simetría y curtosis.
3. **Q-Q plot de residuos:** `scipy.stats.probplot(residuals, dist="norm")`. Evalúa normalidad.

---

## 8. Gráficas e interpretación visual

### 8.1 `53_boost_rent.ipynb`

**Gráfica 1 — Distribución precio_m2 y log_precio (EDA)**
- Histograma doble, figsize=(14, 4); eje izquierdo: `precio_m2` con línea roja en 18 €/m²/mes; eje derecho: `log_precio`

**Gráfica 2 — Optuna: convergencia y trials**
- Línea + scatter, figsize=(14, 4); trial ganador: #62, CV-RMSE=0.14791

**Gráfica 3 — Diagnósticos del modelo final (test)**
- Panel triple, figsize=(16, 4): Real vs Predicho, Histograma residuos, Q-Q plot

**Gráfica 4 — Feature importance (Top N)**
- Barplot horizontal, figsize=(9, 6); top features: `numero_dormitorios`, `numero_banos`, `superficie_construida_m2`

### 8.2 `53_boost_sale_optuna.ipynb`

**Gráfica 1 — Distribución precio_m2 y log_precio (EDA)**
- Histograma doble, figsize=(14, 4)

**Gráfica 2 — Optuna: convergencia y trials**
- Línea + scatter, figsize=(14, 4); trial ganador: #68, CV-RMSE=0.23445

**Gráfica 3 — Diagnósticos del modelo final (test)**
- Panel triple, figsize=(16, 4); título: `"sale | XGBoost Optuna | Real vs Predicho"`

**Gráfica 4 — Feature importance (Top 20)**
- Barplot horizontal, figsize=(9, 6); título: `"sale | XGBoost Optuna — Top 20 importancias"`

### 8.3 `55_sale_rent_models.ipynb`

Cuatro paneles: diagnósticos SALE, importancias SALE, diagnósticos RENT, importancias RENT. Sin gráficas adicionales.

### 8.4 `55_input_result.ipynb` y `55_input_result_no_k_fold.ipynb`

Sin gráficas — notebooks de predicción con output exclusivamente textual.

---

## 9. Interpretabilidad del modelo

### 9.1 Feature importance nativa de XGBoost

Se usa `model.feature_importances_` (gain-based, defecto en XGBoost). No se usa SHAP ni ninguna otra librería de interpretabilidad.

#### Feature importance — Modelo RENT (tras corrección del espacio Optuna)

Cambio notable respecto a versiones anteriores: `numero_dormitorios` y `numero_banos` lideran las importancias (antes lo hacía `superficie_construida_m2`). Esto refleja que el mercado de alquiler en Cantabria valora más el número de habitaciones que el tamaño bruto del inmueble.

| Feature | Importancia | Notas |
|---------|------------|-------|
| `numero_dormitorios` | 0.143 | Feature más importante en rent actual |
| `numero_banos` | 0.117 | |
| `superficie_construida_m2` | 0.113 | |
| `municipio_Santander` | 0.078 | Municipio más relevante |
| `tiene_ascensor_piso` | 0.055 | XGBoost aprende correctamente con NaN para unifamiliares |
| `municipio_otro` | 0.054 | Agrupa municipios con < 10 obs |
| *(resto de features)* | *ver notebook* | |
| `obra_nueva` | ~0.000 | Importancia cero — insuficiente señal en datos de rent |

> `obra_nueva` tiene importancia ≈ 0 en el modelo actual de rent, lo que indica que no hay suficiente señal en el conjunto de datos de alquiler para que este feature contribuya. Esto podría reflejar escasez de obra nueva en el mercado de alquiler de Cantabria, o que el precio no difiere significativamente entre obra nueva y segunda mano en alquiler.

> Las importancias de los municipios individuales aumentaron respecto a la versión anterior gracias a la corrección del espacio de búsqueda de Optuna (ver sección 13.7). Con `gamma` y `min_child_weight` más bajos, XGBoost puede ahora utilizar los dummies de municipio efectivamente.

#### Feature importance — Modelo SALE (top features)

| Feature | Importancia | Notas |
|---------|------------|-------|
| `tipologia_unificada_unifamiliar` | 0.154 | Feature más importante en sale actual |
| `tiene_ascensor_piso` | 0.112 | Alto impacto en precio de venta |
| `superficie_construida_m2` | 0.098 | |
| *(resto de features)* | *ver notebook* | |

> En sale, la tipología (unifamiliar vs piso) es el predictor más importante, seguido de la presencia de ascensor. El ascensor tiene mayor importancia en sale (11.2%) que su equivalente en rent, probablemente porque en venta el comprador valora más la comodidad a largo plazo.

### 9.2 SHAP

No se usa SHAP en ninguno de los cinco notebooks.

---

## 10. Herramienta de predicción individual (notebooks 55_input_*)

### 10.1 `55_input_result.ipynb` (split 80/20)

#### Carga de parámetros y entrenamiento

Los parámetros se cargan desde los JSON al inicio del notebook:

```python
RENT_PARAMS_PATH = PROJECT_ROOT / "data" / "model_results" / "params_rent.json"
SALE_PARAMS_PATH = PROJECT_ROOT / "data" / "model_results" / "params_sale.json"

with open(RENT_PARAMS_PATH) as f:
    rent_cfg = json.load(f)
with open(SALE_PARAMS_PATH) as f:
    sale_cfg = json.load(f)
```

Los modelos se entrenan en ejecución (no se cargan de disco):
1. Carga de CSVs gold (ya limpios — outliers eliminados upstream)
2. `build_X()` con medianas guardadas en `medians_sale` y `medians_rent`
3. Split 80/20
4. Entrenamiento de `model_sale` y `model_rent`
5. Cálculo de `sale_rmse_test` y `rent_rmse_test` para usar como intervalo

#### Referencia geográfica

```python
GEO_COLS = [
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
    "precio_m2_municipio_media",
    "distancia_centro_municipio_km",
    "score_cercania_servicios",
]
```

`build_geo_ref(df)` calcula la mediana de estas variables por municipio. Permite asignar automáticamente valores geográficos al input a partir del municipio seleccionado.

#### Extensión del geo_ref de alquiler mediante join por coordenadas

En versiones anteriores, el geo_ref de rent solo cubría los 6 municipios con columna OHE explícita + `municipio_otro`. Esto dejaba fuera municipios como Santa Cruz de Bezana que, aun teniendo propiedades en el dataset, estaban colapsados en `municipio_otro`.

La solución implementada es un **join por coordenadas redondeadas** entre el CSV processed (que tiene `latitude`/`longitude` y nombre de municipio) y el CSV gold (que tiene los valores geográficos computados):

```python
_proc = pd.read_csv(PROC_RENT_PATH)   # CSV processed con latitude/longitude
_proc["_lat"] = _proc["latitude"].round(5)
_proc["_lon"] = _proc["longitude"].round(5)

_gdf = pd.read_csv(RENT_PATH)         # CSV gold con distancias/scores
_gdf["_lat"] = _gdf["latitud"].round(5)
_gdf["_lon"] = _gdf["longitud"].round(5)

_merged = _proc.merge(_gdf[["_lat","_lon","municipio_otro"] + _geo_feats],
                      on=["_lat","_lon"], how="inner")
_otro_m = _merged[_merged["municipio_otro"] == 1]

for _mun, _grp in _otro_m.groupby("municipality"):
    if _mun in rent_geo_ref.index:
        continue
    _row = {gc: _grp[gc].median() for gc in _geo_feats if gc in _grp.columns}
    _row["precio_m2_municipio_media"] = _r_means.get(_mun, _r_global)
    rent_geo_ref.loc[_mun] = _row
```

**Resultado:** el geo_ref de rent se expande de ~7 a **54 municipios** disponibles para predicción, incluyendo Santa Cruz de Bezana y todos los municipios de la tabla `mun_means_sale` del JSON.

- **Municipios disponibles en venta:** ~31
- **Municipios disponibles en alquiler:** ~54 (extendido)

#### Municipios disponibles para alquiler (de `mun_means_sale` en params_rent.json)

`Ampuero`, `Barcena de Cicero`, `Camargo`, `Castro-Urdiales`, `Colindres`, `Cudon`, `El Astillero`, `Guarnizo`, `Laredo`, `Liendo`, `Limpias`, `Marina de Cudeyo`, `Miengo`, `Mogro`, `Noja`, `Ortuella`, `Piélagos`, `Polanco`, `Ribamontan al Mar`, `Ribamontan al Monte`, `Santa Cruz de Bezana`, `Santander`, `Santoña`, `Santurtzi`, `Suances`, `Torrelavega`, `Villaescusa`, `Viveda`, `Voto` + municipios adicionales vía join.

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

#### Preprocesamiento del input — función `_build_row()`

1. Asigna features hedónicas directas del input.
2. Calcula `interaccion_planta_sin_ascensor_piso`; para unifamiliares asigna NaN a todas las features piso-only.
3. Extrae valores geográficos de `geo_ref.loc[municipio]`.
4. Pone a 0 todos los dummies de municipio y activa el correcto según prioridad:
   - Si `municipio_XXX` existe en la matriz → activa esa columna
   - Si no existe pero `municipio_otro` existe → activa `municipio_otro`
   - Si no existe pero `municipio_otros` existe → activa `municipio_otros`
5. Imputa cualquier NaN restante (no de unifamiliar) con `medians.get(col, 0)`.

> **Bug corregido:** versiones anteriores del fallback solo comprobaban `municipio_otros`, omitiendo `municipio_otro`. Esto causaba que municipios como Santa Cruz de Bezana (colapsados en `municipio_otro` en el gold) no recibieran el dummy correcto en la predicción.

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

- **Intervalo ±1σ:** `[precio × exp(-RMSE_log), precio × exp(+RMSE_log)]` — asimétrico en euros.
- **Rentabilidad bruta:** `(precio_alquiler × 12) / precio_venta × 100`.
- Si el municipio no está en el geo_ref, se muestra advertencia y se omite esa predicción.

---

### 10.2 `55_input_result_no_k_fold.ipynb` (100% datos)

#### Diferencias clave respecto a `55_input_result`

El notebook es estructuralmente idéntico con una diferencia fundamental: **los modelos se entrenan sobre el 100% de los datos limpios**, sin split train/test.

```python
# Sin split:
model_sale.fit(X_sale, y_sale)
model_rent.fit(X_rent, y_rent)
```

El intervalo de error usa los **CV-RMSE del JSON** en lugar del RMSE del test:

```python
SALE_CV_RMSE = sale_cfg["optuna_cv_rmse"]   # 0.23445
RENT_CV_RMSE = rent_cfg["optuna_cv_rmse"]   # 0.14791
```

La extensión del geo_ref de alquiler por coordenadas y el resto de la lógica son **idénticos** a `55_input_result`.

---

### 10.3 Ejemplos de predicción

#### Ejemplo — `55_input_result.ipynb` (modelo split 80/20)

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

**Output (valores de referencia — cambiarán al re-ejecutar el notebook):**

| Métrica | Valor |
|---------|-------|
| Precio de venta estimado | 314.381 € |
| Intervalo venta (±1σ) | [245.914 €, 401.910 €] |
| Alquiler mensual estimado | 1.039 €/mes |
| Intervalo alquiler (±1σ) | [899 €/mes, 1.201 €/mes] |
| Rentabilidad bruta estimada | 4.0% |

---

## 11. Notebook `55_sale_rent_models.ipynb`: integración de modelos

### 11.1 Función

Replica los dos modelos de los notebooks `53_*` en un único notebook de referencia para evaluación conjunta. Produce los mismos outputs de métricas e importancias que los notebooks de entrenamiento.

### 11.2 Carga de hiperparámetros desde JSON

A diferencia de versiones anteriores (donde los params estaban hardcodeados), ahora **lee los JSON** exportados por los notebooks `53_*`:

```python
with open(PROJECT_ROOT / "data/model_results/params_sale.json") as f:
    sale_cfg = json.load(f)
with open(PROJECT_ROOT / "data/model_results/params_rent.json") as f:
    rent_cfg = json.load(f)

XGB_PARAMS_SALE = sale_cfg["xgb_params"]
XGB_PARAMS_RENT = rent_cfg["xgb_params"]
BASE_FEATURES_SALE = sale_cfg["base_features"]
BASE_FEATURES_RENT = rent_cfg["base_features"]
```

Esto garantiza que `55_sale_rent_models` siempre usa exactamente los mismos parámetros que los notebooks `53_*`.

### 11.3 Manejo de NaN y municipios

La función `build_X()` es la misma que en los notebooks `53_*`, con el mismo manejo de NaN para unifamiliares y el mismo colapso de municipios raros.

---

## 12. Dependencias técnicas

### 12.1 Librerías externas

| Librería | Uso | Versión |
|----------|-----|---------|
| `xgboost` | `XGBRegressor` — modelo principal | No determinable en los notebooks |
| `optuna` | Búsqueda de hiperparámetros (`53_*`) | No determinable |
| `sklearn` | `SimpleImputer`, `KFold`, `cross_val_score`, `train_test_split`, métricas | No determinable |
| `pandas` | Manipulación de datos | No determinable |
| `numpy` | Cálculos numéricos | No determinable |
| `matplotlib` | Visualizaciones | No determinable |
| `scipy` | `scipy.stats.probplot` — Q-Q plots | No determinable |
| `json` | Lectura/escritura de params JSON | Stdlib |
| `pathlib` | Detección de raíz del proyecto | Stdlib |

Entorno: Python 3.14 (path del venv: `.venv/lib/python3.14/`).

### 12.2 Módulos internos

No se importa ningún módulo del directorio `src/`. Todo el código es self-contained en cada notebook.

### 12.3 Detección de raíz del proyecto

```python
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "gold").exists():
            return p
    raise FileNotFoundError("No se encontró la raíz del proyecto (data/gold)")

PROJECT_ROOT = find_project_root(Path.cwd().resolve())
```

Hace los notebooks portables sin hardcodear rutas absolutas.

---

## 13. Limitaciones, advertencias y observaciones

### 13.1 Variables geográficas no disponibles

El gold no incluye distancia a hospitales, universidades ni transporte público. Son las variables con mayor potencial de mejora — incorporarlas en el pipeline de transformaciones sería el siguiente salto:
- `distancia_min_hospital_km`
- `distancia_min_universidad_km`
- `distancia_min_parada_bus_km` / `distancia_min_tren_km`

### 13.2 Leakage evitado — advertencia explícita

`rentabilidad_bruta_zona` fue descartada explícitamente en `53_boost_rent`:
> "Fue descartada — usaba `precio` (= exp(log_precio)) directamente como feature, lo que hacía que el modelo pudiera reconstruir el target casi perfectamente y disparaba el R² artificialmente a >0.9."

`precio_m2_municipio_media` en el modelo de rent se calcula a partir de datos de **venta** (no del target de alquiler), lo que lo hace legítimo como proxy de zona sin causar leakage.

### 13.3 Consistencia entre notebooks

Con el sistema de params JSON, los CV-RMSE son coherentes en todos los notebooks:
- `53_boost_rent`: CV-RMSE = 0.14791 (origen)
- `55_sale_rent_models`: mismo valor al usar los mismos params y split
- `55_input_result_no_k_fold`: lee `optuna_cv_rmse` del JSON → 0.14791

### 13.4 Sin persistencia de modelos

Los modelos no se guardan en disco. Cada ejecución de un notebook `55_*` re-entrena el modelo desde cero. Para un sistema de producción esto implica un tiempo de espera en cada sesión.

### 13.5 `obra_nueva` con importancia ≈ 0 en rent

La feature `obra_nueva` tiene importancia prácticamente nula en el modelo de alquiler actual. Esto puede indicar: (a) escasez de obra nueva en el mercado de alquiler de Cantabria, (b) que el precio de alquiler no difiere significativamente entre obra nueva y segunda mano, o (c) que la señal está ya capturada por otras features. No se descarta por ahora porque podría recuperar importancia con más datos.

### 13.6 Sobreajuste moderado en M-RENT

- Delta R² ≈ 0.18 (train 0.783 − test 0.599)
- CV-RMSE (0.148) y RMSE_test (0.155) muy alineados → el sobreajuste es de train vs. generalización, no de los hiperparámetros de Optuna

El sobreajuste es esperado con 529 filas de train. El mercado de alquiler tiene más ruido que el de venta (estacionalidad, negociación individual, amueblado vs. no amueblado) que el modelo no puede capturar con las features disponibles.

### 13.7 Diagnóstico y corrección de importancias cero en rent (histórico)

Con el espacio de búsqueda original de Optuna (`gamma: [0, 5]`, `min_child_weight: [1, 15]`, `subsample: [0.5, 1.0]`), se detectó que muchas features — incluyendo casi todos los municipios — tenían importancia = 0.

**Causa raíz:** el mecanismo es una cadena de tres factores:
1. `subsample = 0.52` → cada árbol ve solo el 52% de las filas de train
2. `min_child_weight = 13` → un nodo necesita ≥ 13 filas para poder hacer un split
3. Un municipio con N observaciones totales necesita `N × subsample ≥ min_child_weight` para aparecer en suficientes árboles, es decir `N ≥ 13 / 0.52 ≈ 25` observaciones. Municipios con 10–24 observaciones son bloqueados.
4. `gamma = 0.162` → incluso cuando un municipio supera el umbral de `min_child_weight`, el split adicional debe mejorar la función de pérdida en al menos 0.162 para ser aceptado. Esto bloqueaba los splits restantes.

**Solución aplicada:** restricción del espacio de búsqueda (ver sección 4.4). Con `gamma ≤ 0.05` y `min_child_weight ≤ 6`, todos los municipios con ≥ 10 observaciones pueden participar en splits con cualquier valor de `subsample ≥ 0.5`. La limitación adicional de `subsample ≤ 0.85` evita que Optuna encuentre soluciones con árboles altamente correlados que sobreajusten el train.

### 13.8 Sobreajuste pronunciado en M-SALE

- Delta R² ≈ 0.132 (train ≈ 0.963 − test 0.831)
- El ratio RMSE_test/RMSE_train ≈ 2.1 es alto
- Sin embargo, CV-RMSE (0.234) ≈ RMSE_test (0.235), lo que confirma que la validación cruzada detecta correctamente el nivel de error de generalización

El sobreajuste de train es esperado con 900 estimadores y learning rate bajo — el modelo memoriza el training set. Pero generaliza correctamente al test porque el CV-RMSE es un buen estimador.

### 13.9 Decisiones metodológicas sin justificación estadística formal

- **`MIN_MUNI_OBS = 10`**: umbral para colapsar municipios. Valor pragmático.
- **`IQR_FACTOR = 1.5`**: más estricto que el IQR×3 de notebooks anteriores.
- **`PRECIO_M2_VACACIONAL_UMBRAL = 18 €/m²`**: elegido por inspección visual.
- **`PRECIO_M2_MIN_VENTA = 1000 €/m²`**: suelo para detectar propiedades no-residenciales, basado en la cola inferior del Q-Q plot.

### 13.10 Cobertura de municipios en alquiler vs. venta

| Aspecto | Venta | Alquiler |
|---------|-------|----------|
| Municipios con columna OHE propia | ~29 | 6 |
| Municipios disponibles en input | ~31 | ~54 |
| Cómo se amplía en input | geo_ref directo del gold | geo_ref extendido por join lat/lon |

Los municipios de alquiler sin columna OHE propia se predicen con `municipio_otro = 1` en la matriz de features. El modelo los trata todos igual en terms de influencia del municipio, pero reciben sus valores geográficos correctos (distancias, scores) gracias al geo_ref extendido.
