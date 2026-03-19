# Documentación: Modelos de Regresión Lineal — BezanillaSL

## Contexto del proyecto

BezanillaSL es un proyecto de predicción de precios de inmuebles en **Cantabria** (España). Los municipios del dataset incluyen Santander, Torrelavega, Laredo, Castro-Urdiales, Camargo, Santa Cruz de Bezana, Suances, Santoña, etc. — zona costera cantábrica.

Hay dos datasets en `data/gold/`:
- **sale** (`final_sale.csv`): pisos y casas en venta
- **rent** (`final_rent.csv`): pisos y casas en alquiler

La variable objetivo es **`log_precio`** (logaritmo natural del precio en euros). Se trabaja en escala logarítmica para:
- Reducir la heterocedasticidad (los precios tienen distribución sesgada a la derecha)
- Hacer los residuos más simétricos y aproximadamente normales
- Facilitar la interpretación de coeficientes como efectos porcentuales

Para convertir una predicción al precio real: `precio = exp(log_precio_predicho)`.

---

## Notebook: `51_linear_regression_def.ipynb`

Unifica y compara tres enfoques de regresión lineal en un único notebook, aplicándolos a ambos datasets.

| Modelo | Regularización | Inferencia estadística | n features |
|--------|---------------|----------------------|-----------|
| OLS Base | Ninguna | Sí (statsmodels: coef, p-value, IC95) | 11 |
| Ridge | L2 (RidgeCV) | No | 35 (sale) / 29 (rent) |
| Lasso + OLS | L1 selección + OLS | Sí (statsmodels) | 26 (sale) / 13 (rent) |

---

## Pipeline común

Todos los modelos siguen estos pasos en el mismo orden:

1. **Carga** de `final_sale.csv` o `final_rent.csv`
2. **Eliminación de outliers** en `log_precio` con IQR (factor 1.5) **antes** del split — se eliminan registros cuyo `log_precio` cae fuera de `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`
3. **Split 80/20** train/test con `random_state=42`
4. **Validación cruzada 5-fold** (KFold shuffle, `random_state=42`)
5. **Métricas**: MSE, RMSE, MAE, MAPE, R², R² ajustado en train, CV y test
6. **Gráficos diagnósticos**: Real vs Predicho, Residuos vs Predicho, Histograma de residuos

---

## Datos de entrada

### Dataset SALE (venta)

| Etapa | Registros |
|-------|----------|
| Filas originales | 588 |
| Outliers eliminados (IQR) | 6 (1.0%) |
| Rango válido log_precio | [10.9828, 13.8581] → [e^10.98, e^13.86] ≈ [59.000€, 1.050.000€] |
| Filas usadas | 582 |
| Train | 465 |
| Test | 117 |

### Dataset RENT (alquiler)

| Etapa | Registros |
|-------|----------|
| Filas originales | 477 |
| Outliers eliminados (IQR) | 26 (5.5%) — mucho más que en venta |
| Rango válido log_precio | [6.1079, 7.8074] → [e^6.11, e^7.81] ≈ [450€, 2.460€/mes] |
| Filas usadas | 451 |
| Train | 360 |
| Test | 91 |

> El alquiler tiene muchos más outliers proporcionales (5.5% vs 1%) — los datos de alquiler tienen más variabilidad extrema.

---

## Modelo 1 — OLS Base

### Descripción

Regresión lineal ordinaria ajustada con `statsmodels.OLS`. Sin regularización. 11 features seleccionadas manualmente por criterio parsimonioso (mezcla de características físicas del inmueble + ubicación + mercado). Es el modelo más interpretable.

### Features (11)

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `log_superficie_construida_m2` | Continua | Superficie en log — relación log-log con precio |
| `numero_dormitorios` | Discreta | Nº dormitorios |
| `numero_banos` | Discreta | Nº baños |
| `tiene_garaje` | Dummy (0/1) | Si incluye plaza de garaje |
| `obra_nueva` | Dummy (0/1) | Si es obra nueva |
| `distancia_min_playa_km` | Continua | Distancia a la playa más cercana (km) |
| `distancia_min_supermercado_km` | Continua | Distancia al supermercado más cercano (km) |
| `distancia_min_colegio_km` | Continua | Distancia al colegio más cercano (km) |
| `distancia_centro_municipio_km` | Continua | Distancia al centro del municipio (km) |
| `tipologia_unificada_unifamiliar` | Dummy (0/1) | Si es casa unifamiliar (vs piso) |
| `precio_m2_municipio_media` | Continua | Precio medio del m² en el municipio (variable de mercado) |

### Resultados SALE

```
               split     MSE    RMSE     MAE    MAPE      R2  R2_ajustado
               train 0.11680 0.34176 0.24848 0.02014 0.61301      0.60361
          CV (5-fold) 0.11799 0.34267 0.24977 0.02021 0.59274      0.54983
                test  0.09124 0.30206 0.22771 0.01816 0.63262      0.59414

delta RMSE test-train : -0.03970
delta R2   train-test : -0.01962
F-stat: 65.23  p-value F: 3.69e-86
```

### Coeficientes OLS Base — SALE (ordenados por |coef|)

| Feature | coef | std_err | t | p_value | IC_95_inf | IC_95_sup | Significativa |
|---------|------|---------|---|---------|-----------|-----------|---------------|
| `log_superficie_construida_m2` | +0.5051 | 0.0544 | 9.29 | 0.0000 | 0.3982 | 0.6120 | ✅ |
| `tiene_garaje` | +0.2118 | 0.0356 | 5.95 | 0.0000 | 0.1418 | 0.2817 | ✅ |
| `numero_banos` | +0.0842 | 0.0232 | 3.64 | 0.0003 | 0.0387 | 0.1298 | ✅ |
| `obra_nueva` | −0.0399 | 0.0558 | −0.72 | 0.4743 | — | — | ❌ |
| `distancia_min_colegio_km` | +0.0299 | 0.0248 | 1.20 | 0.2289 | — | — | ❌ |
| `distancia_min_playa_km` | −0.0171 | 0.0070 | −2.45 | 0.0147 | −0.0309 | −0.0034 | ✅ |
| `tipologia_unificada_unifamiliar` | −0.0142 | 0.0600 | −0.24 | 0.8129 | — | — | ❌ |
| `numero_dormitorios` | +0.0097 | 0.0186 | 0.52 | 0.6031 | — | — | ❌ |
| `distancia_min_supermercado_km` | −0.0082 | 0.0237 | −0.34 | 0.7315 | — | — | ❌ |
| `distancia_centro_municipio_km` | −0.0069 | 0.0132 | −0.52 | 0.5999 | — | — | ❌ |
| `precio_m2_municipio_media` | +0.0003 | 0.00003 | 8.10 | 0.0000 | 0.0002 | 0.0003 | ✅ |

**Interpretación de los coeficientes significativos (sale):**
- `log_superficie_construida_m2` = 0.505: una variación del 1% en superficie produce un aumento del 0.505% en el precio (elasticidad)
- `tiene_garaje` = 0.212: tener garaje sube el precio en ≈ e^0.212 − 1 ≈ **+23.6%**
- `numero_banos` = 0.084: cada baño adicional sube el precio ≈ **+8.8%**
- `distancia_min_playa_km` = −0.017: cada km más de playa reduce el precio ≈ **−1.7%** (en Cantabria la proximidad al mar tiene valor)
- `precio_m2_municipio_media` = 0.00027: captura el efecto del mercado local

**Variables no significativas (p > 0.05) en sale:** `obra_nueva`, `tipologia_unificada_unifamiliar`, `numero_dormitorios`, distancias a supermercado, colegio y centro. Esto sugiere que en venta lo que más importa es la superficie, el garaje, los baños, la distancia a la playa y el precio de mercado del municipio.

### Resultados RENT

```
               split     MSE    RMSE     MAE    MAPE      R2  R2_ajustado
               train 0.04092 0.20228 0.15526 0.02231 0.54897      0.53471
          CV (5-fold) 0.04461 0.21054 0.16125 0.02317 0.52386      0.45686
                test  0.04671 0.21613 0.16674 0.02398 0.56433      0.50366

delta RMSE test-train : +0.01385
F-stat: 42.48  p-value F: 1.73e-54
```

### Coeficientes OLS Base — RENT

| Feature | coef | p_value | Significativa |
|---------|------|---------|---------------|
| `log_superficie_construida_m2` | +0.3802 | 0.0000 | ✅ |
| `precio_m2_municipio_media` | +0.0267 | 0.0000 | ✅ |
| `distancia_min_playa_km` | −0.0257 | 0.0000 | ✅ |
| `numero_dormitorios` | +0.0675 | 0.0004 | ✅ |
| `numero_banos` | +0.0525 | 0.0360 | ✅ |
| `tipologia_unificada_unifamiliar` | −0.0759 | 0.1368 | ❌ |
| `tiene_garaje` | −0.0351 | 0.1726 | ❌ |
| `distancia_min_supermercado_km` | +0.0172 | 0.4047 | ❌ |
| `distancia_centro_municipio_km` | −0.0101 | 0.3421 | ❌ |
| `distancia_min_colegio_km` | −0.0126 | 0.6280 | ❌ |
| `obra_nueva` | ≈ 0.0000 | 0.0000 | ⚠️ coef = 0 |

**Diferencias clave sale vs rent:**
- En rent el garaje **no es significativo** (p=0.17) — el alquiler no prima tanto el garaje como la venta
- `numero_dormitorios` sí es significativo en rent pero no en sale (en alquiler el nº de habitaciones importa más para fijar precio)
- `precio_m2_municipio_media` tiene un efecto mucho mayor en rent (0.027 vs 0.0003) por diferencia de escala

---

## Modelo 2 — Ridge (L2)

### Descripción

Regresión Ridge con penalización L2 que encoge los coeficientes hacia cero sin eliminarlos. El alpha óptimo se selecciona con `RidgeCV` evaluando 200 valores entre 10⁻³ y 10⁵ mediante CV=5. Las features se estandarizan (`StandardScaler`) dentro de cada fold para evitar data leakage. Usa un conjunto más amplio de features que incluye dummies de municipio.

### Municipios en el dataset

**Sale** — 59 municipios totales → 14 con ≥10 registros (se mantienen) + 45 agrupados en `municipio_otros`:

Municipios mantenidos (sale): `Camargo`, `Castro-Urdiales`, `Laredo`, `Noja`, `Ortuella`, `Piélagos`, `Polanco`, `Santa Cruz de Bezana`, `Santander`, `Santoña`, `Santurtzi`, `Suances`, `Torrelavega`, `Voto`

**Rent** — municipios totales → 8 con ≥10 registros: `Camargo`, `Castro-Urdiales`, `El Astillero`, `Laredo`, `Piélagos`, `Santa Cruz de Bezana`, `Santander`, `Torrelavega`

> En alquiler hay menos municipios con suficientes datos — el mercado de alquiler está más concentrado en las ciudades principales.

### Features Ridge (20 base + municipios)

Respecto al OLS base, se añaden:
- `superficie_construida_m2` (sin log, en lugar de versión log)
- `latitud`, `longitud` — posición geográfica exacta
- `planta` — número de planta del piso
- `es_exterior_piso` — si es exterior
- `tiene_ascensor_piso` — si tiene ascensor
- `ratio_dormitorios_superficie` — dormitorios / superficie
- `ratio_banos_superficie` — baños / superficie
- `interaccion_planta_sin_ascensor_piso` — interacción: planta alta sin ascensor
- `score_cercania_servicios` — score compuesto de proximidad a servicios
- `tipologia_unificada_piso` — dummy: es piso (vs otros tipos)
- Dummies de municipio (14 en sale, 8 en rent)

### Alphas óptimos y resultados

| Dataset | Alpha óptimo | Interpretación |
|---------|-------------|----------------|
| Sale | 13.83 | Regularización moderada-baja |
| Rent | 46.06 | Regularización más fuerte — los datos de alquiler tienen más ruido |

### Resultados SALE

```
               split     MSE    RMSE     MAE    MAPE      R2  R2_ajustado
               train 0.09902 0.31467 0.22816 0.01850 0.67191      0.64515
          CV (5-fold) 0.11026 0.33154 0.23996 0.01943 0.61819      0.45202
                test  0.08981 0.29968 0.22913 0.01829 0.63838      0.48212

delta RMSE test-train : -0.01499
delta R2   train-test : +0.03354
```

### Coeficientes Ridge estandarizados — SALE (top 10)

| Feature | coef_std | coef_original | Interpretación |
|---------|---------|---------------|----------------|
| `tipologia_unificada_piso` | −0.1939 | −0.4219 | Los pisos valen menos que unifamiliares/otros |
| `score_cercania_servicios` | +0.1534 | +0.9620 | Mejor acceso a servicios = precio mayor |
| `numero_dormitorios` | +0.1523 | +0.1084 | Más dormitorios = precio mayor |
| `precio_m2_municipio_media` | +0.1390 | +0.0002 | Mercado local |
| `tiene_ascensor_piso` | +0.1298 | +0.2621 | El ascensor sube el precio |
| `ratio_dormitorios_superficie` | — | — | Relación dormitorios/superficie |

> Los coeficientes estandarizados permiten comparar la importancia relativa de cada variable independientemente de su escala. `|coef_std|` mayor = mayor influencia.

### Resultados RENT

```
               split     MSE    RMSE     MAE    MAPE      R2  R2_ajustado
               train 0.03996 0.19991 0.15305 0.02196 0.55946      0.52075
          CV (5-fold) 0.04543 0.21253 0.16206 0.02325 0.51432      0.28027
                test  0.04707 0.21695 0.16608 0.02388 0.56102      0.35233

delta RMSE test-train : +0.01704
```

### Coeficientes Ridge estandarizados — RENT (top 10)

| Feature | coef_std | coef_original |
|---------|---------|---------------|
| `numero_dormitorios` | +0.0825 | +0.0886 |
| `superficie_construida_m2` | +0.0761 | +0.0020 |
| `precio_m2_municipio_media` | +0.0648 | +0.0170 |
| `numero_banos` | +0.0645 | +0.1014 |
| `distancia_min_playa_km` | −0.0467 | −0.0210 |
| `ratio_banos_superficie` | −0.0295 | −5.2192 |
| `tiene_ascensor_piso` | +0.0244 | +0.0510 |
| `municipio_Laredo` | +0.0141 | +0.0590 |
| `municipio_Piélagos` | −0.0139 | −0.0889 |

---

## Modelo 3 — Lasso + OLS

### Descripción

Pipeline en dos fases:
1. **LassoCV**: selecciona features automáticamente poniendo a cero los coeficientes de variables irrelevantes (α óptimo vía CV=5, 200 alphas, features estandarizadas internamente)
2. **OLS statsmodels**: re-estima los coeficientes **sin penalización** sobre las features seleccionadas, recuperando inferencia estadística válida (p-values, IC95)

Este enfoque combina lo mejor de Lasso (selección automática) con lo mejor de OLS (interpretabilidad estadística). El OLS re-estima sin sesgo, mientras que Lasso solo actúa como filtro.

Parte del conjunto candidato más amplio: **28 features base + municipios** (43 en sale, 37 en rent), que incluye términos cuadráticos e interacciones.

### Features candidatas adicionales respecto a Ridge

| Feature | Descripción |
|---------|-------------|
| `interaccion_superficie_banos` | superficie × baños |
| `planta_num` | Número de planta (numérico, sin categorizar) |
| `latitud_2`, `longitud_2` | Términos cuadráticos geográficos |
| `interaccion_latitud_longitud` | Producto lat × lon |
| `superficie_construida_m2_2` | Superficie al cuadrado |
| `numero_banos_2` | Baños al cuadrado |
| `numero_dormitorios_2` | Dormitorios al cuadrado |
| `tipologia_unificada_unifamiliar` | Dummy: es unifamiliar |

### Resultados SALE

**Alpha Lasso óptimo:** 0.007916

**Features eliminadas por Lasso (17):**
`latitud`, `longitud`, `interaccion_superficie_banos`, `interaccion_planta_sin_ascensor_piso`, `latitud_2`, `longitud_2`, `interaccion_latitud_longitud`, `superficie_construida_m2_2`, `numero_dormitorios_2`, `tipologia_unificada_unifamiliar`, `municipio_Castro-Urdiales`, `municipio_Noja`, `municipio_Ortuella`, `municipio_Santander`, `municipio_Suances`, `municipio_Torrelavega`, `municipio_otros`

**Features seleccionadas (26):**
`superficie_construida_m2`, `numero_dormitorios`, `numero_banos`, `planta`, `es_exterior_piso`, `tiene_ascensor_piso`, `tiene_garaje`, `obra_nueva`, `distancia_min_playa_km`, `distancia_min_supermercado_km`, `distancia_min_colegio_km`, `precio_m2_municipio_media`, `ratio_banos_superficie`, `planta_num`, `distancia_centro_municipio_km`, `score_cercania_servicios`, `numero_banos_2`, `tipologia_unificada_piso`, `municipio_Camargo`, `municipio_Laredo`, `municipio_Piélagos`, `municipio_Polanco`, `municipio_Santa Cruz de Bezana`, `municipio_Santoña`, `municipio_Santurtzi`, `municipio_Voto`

> Lasso elimina todas las interacciones geográficas (lat, lon y sus cuadrados) y la mayoría de los municipios grandes como Santander y Torrelavega. Mantiene baños al cuadrado (`numero_banos_2`) pero elimina dormitorios al cuadrado.

```
               split     MSE    RMSE     MAE    MAPE      R2  R2_ajustado
               train 0.08914 0.29857 0.21929 0.01782 0.70463      0.68710
          CV (5-fold) 0.13813 0.37165 0.28047 0.02274 0.48283      0.35124
                test  0.09895 0.31455 0.23628 0.01890 0.60162      0.47684

delta RMSE test-train : +0.01597
delta R2   train-test : +0.10301   ← señal de sobreajuste
```

⚠️ **El RMSE de CV (0.372) es notablemente mayor que el de test (0.315)** — incoherencia que indica que el modelo sale es inestable en este conjunto de datos. El modelo memoriza más el train de lo que generaliza.

### Resultados RENT

**Alpha Lasso óptimo:** 0.010692

**Features eliminadas por Lasso (24):**
`latitud`, `longitud`, `planta`, `tiene_garaje`, `obra_nueva`, `distancia_min_supermercado_km`, `distancia_min_colegio_km`, `ratio_banos_superficie`, `interaccion_superficie_banos`, `interaccion_planta_sin_ascensor_piso`, `latitud_2`, `longitud_2`, `interaccion_latitud_longitud`, `distancia_centro_municipio_km`, `score_cercania_servicios`, `superficie_construida_m2_2`, `numero_banos_2`, `tipologia_unificada_piso`, `tipologia_unificada_unifamiliar`, `municipio_Camargo`, `municipio_Castro-Urdiales`, `municipio_Laredo`, `municipio_Santa Cruz de Bezana`, `municipio_Torrelavega`

**Features seleccionadas (13):**
`superficie_construida_m2`, `numero_dormitorios`, `numero_banos`, `es_exterior_piso`, `tiene_ascensor_piso`, `distancia_min_playa_km`, `precio_m2_municipio_media`, `planta_num`, `numero_dormitorios_2`, `municipio_El Astillero`, `municipio_Piélagos`, `municipio_Santander`, `municipio_otros`

> Para alquiler, Lasso es mucho más agresivo: de 37 candidatas solo mantiene 13 (65% eliminadas). Elimina `tiene_garaje`, `obra_nueva`, la mayoría de municipios y todas las interacciones complejas. Mantiene `numero_dormitorios_2` (efecto cuadrático de los dormitorios).

**Coeficientes Lasso estandarizados no nulos (rent, top):**
`superficie_construida_m2` (0.114), `precio_m2_municipio_media` (0.087), `numero_dormitorios` (0.066), `distancia_min_playa_km` (negativo)

```
               split     MSE    RMSE     MAE    MAPE      R2  R2_ajustado
               train 0.04062 0.20157 0.15457 0.02218 0.55213      0.53530
          CV (5-fold) 0.04437 0.21040 0.16073 0.02304 0.52348      0.47011
                test  0.04552 0.21334 0.16510 0.02373 0.57550      0.52416

delta RMSE test-train : +0.01178
delta R2   train-test : -0.02337   ← generaliza bien
```

---

## Resumen comparativo global

### SALE (venta)

| Modelo | n_feat | Alpha | RMSE_train | RMSE_CV | RMSE_test | R²_test | MAE_test | delta_RMSE | delta_R2 |
|--------|--------|-------|-----------|---------|----------|---------|---------|-----------|---------|
| OLS Base | 11 | — | 0.3418 | 0.3427 | 0.3021 | 0.6326 | 0.2277 | −0.0397 | −0.020 |
| Ridge | 35 | 13.83 | 0.3147 | 0.3315 | **0.2997** | **0.6384** | 0.2291 | −0.0150 | +0.034 |
| Lasso+OLS | 26 | 0.0079 | 0.2986 | 0.3717 | 0.3146 | 0.6016 | 0.2363 | +0.0160 | +0.103 |

### RENT (alquiler)

| Modelo | n_feat | Alpha | RMSE_train | RMSE_CV | RMSE_test | R²_test | MAE_test | delta_RMSE | delta_R2 |
|--------|--------|-------|-----------|---------|----------|---------|---------|-----------|---------|
| OLS Base | 11 | — | 0.2023 | 0.2105 | 0.2161 | 0.5643 | 0.1667 | +0.0139 | −0.015 |
| Ridge | 29 | 46.06 | 0.1999 | 0.2125 | 0.2170 | 0.5610 | 0.1661 | +0.0170 | −0.002 |
| Lasso+OLS | 13 | 0.0107 | 0.2016 | 0.2104 | **0.2133** | **0.5755** | **0.1651** | +0.0118 | −0.023 |

---

## Análisis detallado de resultados

### SALE — ¿Quién gana?

**Ridge** es el mejor modelo para venta:
- Mejor RMSE test (0.2997) y mejor R² test (0.638)
- delta RMSE = −0.015: el test rinde ligeramente mejor que el train → sin sobreajuste
- OLS Base casi iguala a Ridge en test (RMSE 0.302 vs 0.300) con solo 11 features — notable para un modelo tan simple

**Lasso+OLS** en sale tiene problemas:
- Mejor RMSE en train (0.299) pero peor RMSE en test (0.315) → **sobreajuste**
- delta_R2 = +0.103 es el mayor de todos los modelos: el R² cae 10 puntos del train al test
- El CV RMSE (0.372) es incoherentemente mayor que el test RMSE (0.315) — anomalía posiblemente por el pequeño tamaño muestral (n=582) que hace el CV inestable con 26 features

**OLS Base delta negativo:** RMSE test (0.302) < RMSE train (0.342) significa que el modelo predice *mejor* en test. Estadísticamente posible en muestras pequeñas — el test puede caer en una zona más "sencilla" del espacio de features.

### RENT — ¿Quién gana?

**Lasso+OLS** es el ganador claro para alquiler:
- Mejor RMSE test (0.2133), mejor R² test (0.576) y mejor MAE test (0.165) — **con solo 13 features**
- delta_R2 = −0.023: el modelo generaliza mejor en test que en train → ningún sobreajuste
- CV y test son muy similares → modelo estable

**Ridge** no aporta mejora sobre OLS base en rent (RMSE 0.217 vs 0.216) y tiene alpha=46 muy alto → la regularización fuerte está penalizando demasiado.

### Interpretación del error en euros (precio real)

La métrica MAE en escala log se puede interpretar como:
- **Sale OLS/Ridge: MAE ≈ 0.228** → error mediano ≈ e^0.228 − 1 ≈ **+25.6%** del precio real
- **Rent Lasso+OLS: MAE ≈ 0.165** → error mediano ≈ e^0.165 − 1 ≈ **+17.9%** del precio real

Para un inmueble de 200.000€ en venta, el error mediano es ≈ 51.200€.
Para un alquiler de 1.000€/mes, el error mediano es ≈ 179€.

El alquiler se predice mejor que la venta — los precios de alquiler son más uniformes y predecibles dentro de cada municipio.

### Nivel de R² — ¿es bueno?

R² de 0.56–0.64 en test es razonablemente bueno para datos inmobiliarios reales (no sintéticos). Los precios de inmuebles tienen alta variabilidad idiosincrática (orientación, estado de conservación, negociación, etc.) que no está capturada en las features disponibles. Un R² perfecto no es esperable ni deseable — sugeriría overfitting.

---

## Decisiones de diseño importantes

| Decisión | Justificación |
|----------|--------------|
| **Variable objetivo en log** | Distribución de precios sesgada — log la normaliza y reduce heterocedasticidad |
| **Outliers antes del split** | Evita que outliers extremos en el test set distorsionen la evaluación del modelo |
| **Estandarizar solo Ridge/Lasso** | OLS base mantiene coeficientes en unidades originales para interpretabilidad directa |
| **Estandarizar dentro del fold** | Evita data leakage: el scaler se ajusta solo al fold de entrenamiento |
| **Municipios pequeños → `otros`** | Municipios con < 10 registros generan dummies ruidosas e inestables |
| **Dos pasos en Lasso+OLS** | Lasso tiene bias de shrinkage — los coeficientes Lasso son sesgados. OLS re-estima sin sesgo para obtener p-values e IC95 válidos |
| **RidgeCV con 200 alphas** | Grid fino en escala log para encontrar el alpha óptimo con precisión |
| **random_state=42** | Reproducibilidad en split y CV |

---

## Estructura del código

### Configuración global (`config-cell`)

```python
RANDOM_STATE            = 42
TEST_SIZE               = 0.20
N_SPLITS                = 5
TARGET_COL              = "log_precio"
IQR_FACTOR              = 1.5
MIN_MUNICIPIO_REGISTROS = 10
RIDGE_ALPHAS            = np.logspace(-3, 5, 200)
```

### Funciones auxiliares (`functions-cell`)

| Función | Descripción |
|---------|-------------|
| `remove_outliers_iqr(df, col)` | Limpieza IQR en la columna target antes del split |
| `compute_metrics(y_true, y_pred, n_features)` | MSE, RMSE, MAE, MAPE, R², R² ajustado |
| `compute_vif(X)` | VIF por regresor para detectar multicolinealidad |
| `cv_ols_metrics(X, y, n_features)` | CV 5-fold con LinearRegression |
| `cv_ridge_metrics(X, y, alpha, n_features)` | CV 5-fold con Ridge (Pipeline StandardScaler+Ridge) |
| `fit_ols_statsmodels(X_train, y_train)` | Ajusta OLS con statsmodels (añade constante) |
| `print_coef_summary(result)` | Tabla coeficientes OLS ordenada por \|coef\| |
| `group_small_municipios(df)` | Agrupa municipios_* con < 10 registros en `municipio_otros` |
| `build_base_X(df)` | Matriz features OLS base (11 features, imputa con mediana) |
| `build_candidate_X(df, candidate_features)` | Matriz features Ridge/Lasso + municipios |
| `select_features_lasso(X_train, y_train)` | LassoCV estandarizado → devuelve (features_seleccionadas, alpha_opt, coefs) |
| `plot_diagnostics(y_test, pred_test, title)` | 3 gráficos diagnósticos en una figura |

---

## Archivos relacionados

```
notebooks/05_ML/
├── 51_linear_regression_def.ipynb   ← ESTE NOTEBOOK (definitivo, unificado)
├── 51_linear_regression_1.py        ← versión script Python inicial
├── 51_linear_regression_2.ipynb     ← versión exploratoria anterior
├── 51_linear_regression_ridge.ipynb ← desarrollo Ridge por separado
└── 51_linear_regression_lasso.ipynb ← desarrollo Lasso por separado

data/gold/
├── final_sale.csv    ← dataset venta (gold layer, ya procesado)
└── final_rent.csv    ← dataset alquiler (gold layer, ya procesado)
```

---

## Conclusión y modelo recomendado

| Dataset | Modelo recomendado | Razón |
|---------|--------------------|-------|
| **Sale** | **Ridge** | Mejor RMSE test (0.300), sin sobreajuste, estable en CV |
| **Rent** | **Lasso+OLS** | Mejor R² y MAE test con solo 13 features, generaliza bien, inferencia estadística completa |

El OLS Base es una referencia sólida dada su simplicidad — con solo 11 features consigue resultados muy cercanos a los modelos más complejos. Esto sugiere que el poder predictivo adicional de más features es limitado, probablemente por el tamaño moderado del dataset (n ≈ 450–582).
