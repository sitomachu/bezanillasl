# Notebooks 05_ML

Este directorio contiene los cuadernos de modelado y evaluacion para venta y alquiler. A continuacion se resume el objetivo de cada uno.

## 50_unificar_dataset.ipynb
Unifica los datasets de venta y alquiler para uso en modelos ML.

## 51_linear_regression_1.ipynb
Primer experimento de regresion lineal para venta/alquiler (baseline inicial).

## 51_linear_regression_2.ipynb
Regresion OLS con limpieza por IQR, CV, VIF y diagnosticos completos para sale y rent.

## 51_linear_regression_def.ipynb
Version "definitiva" de OLS con comparativas, graficos y diagnostico de overfitting.

## 51_linear_regression_def_2.ipynb
Iteracion definitiva OLS (ajustes adicionales y comparativas extendidas).

## 51_linear_regression_lasso.ipynb
OLS + LassoCV para seleccion/regularizacion de variables.

## 51_linear_regression_ridge.ipynb
Ridge para regularizacion y comparativa de metricas.

## 52_random_forest_1.ipynb
Primer modelo Random Forest con pipeline basico y evaluacion.

## 52_random_forest_2.ipynb
Random Forest con busqueda de hiperparametros y evaluacion.

## 52_random_forest_def.ipynb
Random Forest definitivo con comparativas de modelos.

## 52_random_forest_def_2.ipynb
Random Forest definitivo con Optuna, seleccion de features y modelo final.

## 52_random_forest_scraping.ipynb
Random Forest aplicado a datos de scraping (pipeline especifico).

## 53_boost_1.ipynb
Primer modelo de boosting con preprocesado y evaluacion.

## 53_boost_def.ipynb
Boosting definitivo con grids y comparativas.

## 53_boost_def_2.ipynb
Boosting definitivo con Optuna, seleccion de features y modelo final.

## 53_boost_def_3.ipynb
Boosting directo con exportacion de resultados.

## 53_boost_reg.ipynb
Boosting con regularizacion (Lasso/Ridge) y analisis de coeficientes.

## 53_boost_rent.ipynb
Boosting orientado al modelo de alquiler.

## 53_boost_sale.ipynb
Boosting orientado al modelo de venta.

## 53_boost_sale_optuna.ipynb
Boosting venta con Optuna para optimizar hiperparametros.

## 54_hibrido.ipynb
Ensamble hibrido (Ridge + RF regularizado + Boost) con pesos por CV-RMSE.

## 54_hibrido_2.ipynb
Ensamble hibrido con componentes optimizados de los notebooks def_2.

## 55_input_result.ipynb
Estimador de precio usando parametros JSON para sale y rent (split 80/20).

## 55_input_result_no_k_fold.ipynb
Estimador de precio entrenado con 100% de datos (usa CV-RMSE del JSON).

## 55_sale_rent_models.ipynb
Modelos XGBoost definitivos para sale y rent a partir de los JSON exportados.
