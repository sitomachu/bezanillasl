# Dataset: Ministerio de Vivienda y Agenda Urbana (MIVAU)

Este directorio centraliza la información estadística obtenida del **Ministerio de Vivienda y Agenda Urbana (MIVAU)**. La estructura de datos está diseñada para facilitar el análisis técnico del sector inmobiliario en España, dividiéndose en tres pilares fundamentales: mercado del suelo, stock de vivienda y mercado de arrendamiento.

---

## Estructura del Directorio

* **`datos_alquiler/`**: Datos del Sistema Estatal de Referencia del Precio del Alquiler de Vivienda (SERPAVI).
* **`datos_suelo/`**: Estadísticas sobre transacciones y valor del suelo urbano.
* **`datos_vivienda/`**: Series históricas y estimaciones sobre el parque de viviendas en España.

---

## Origen de los Datos y Fuentes Oficiales

La integridad de este dataset depende de la trazabilidad hacia sus fuentes originales. A continuación se detallan los repositorios oficiales de los cuales se ha extraído la información:

| Categoría | Descripción Técnica | Fuente de Datos |
| :--- | :--- | :--- |
| **Suelo Urbano** | Análisis de precios medios de venta y volumen de transacciones por m². | [Estadísticas de Precios de Suelo](https://www.mivau.gob.es/el-ministerio/observatorios-y-estadisticas/estadisticas/precios-suelo-urbano) |
| **Parque de Viviendas** | Estimación del stock de viviendas por tipología y distribución geográfica. | [Estimación del Parque de Viviendas](https://www.mivau.gob.es/el-ministerio/observatorios-y-estadisticas/estadisticas/estimacion-parque-viviendas) |
| **Alquiler** | Índices de rentas y precios de referencia para el mercado de arrendamiento. | [Sistema de Referencia del Alquiler (SERPAVI)](https://www.mivau.gob.es/vivienda/alquila-bien-es-tu-derecho/serpavi) |

---

## Notas de Implementación

1.  **Formato de Datos**: Se asume que los archivos contenidos en estas subcarpetas mantienen la coherencia estructural de las fuentes gubernamentales (frecuentemente archivos `.xlsx` o `.csv` con separadores decimales europeos).
2.  **Referencia Temporal**: Los datos del MIVAU están sujetos a actualizaciones periódicas. Es imperativo verificar la fecha de extracción de los ficheros para asegurar la vigencia del análisis.
3.  **Uso**: Este repositorio tiene fines de análisis estadístico y de investigación. Se debe citar al MIVAU como autor original de los datos en cualquier publicación derivada.

---

**Última revisión:** Febrero 2026