3.1.1. Arquitectura conceptual del pipeline de datos

El presente TFM se plantea como un proyecto de **pipeline analítico aplicado** al mercado inmobiliario, no como un ejercicio aislado de predicción. La unidad de análisis no es solo el modelo final, sino el flujo completo que transforma datos heterogéneos en evidencia útil para decisiones de inversión y viabilidad empresarial.

Desde una perspectiva metodológica, la arquitectura sigue una lógica **secuencial, modular y reproducible**: captura de datos, almacenamiento por capas, preparación analítica, análisis/modelización, evaluación y, como destino de negocio, integración con la capa financiera. Este enfoque permite separar responsabilidades técnicas, reducir dependencia de procesos manuales y mantener trazabilidad de extremo a extremo.

### Tipo de arquitectura: pipeline estructurado con separación por capas (y convergencia medallion)

La arquitectura se estructura como un **pipeline por capas**, con separación explícita entre datos en bruto y datos transformados. Conceptualmente, esta organización se formaliza bajo una lógica **medallion**:

- **Bronze (en el repositorio, capa `raw`):** datos crudos, sin transformación analítica, preservando estructura original para auditoría y reprocesado. Ejemplos reales: `data/raw/scraping_manual/alquiler_idealista.csv`, `data/raw/scraping_manual/venta_idealista.csv`, `data/raw/scraping_manual/terrenos_idealista.csv`, y ejecuciones API versionadas por run en `data/raw/idealista/*` con `manifest.json` y respuestas `reqXXX__*.json`.
- **Silver (en el repositorio, capa `processed`):** datos limpios, tipados y normalizados, con reglas de calidad e integración entre fuentes. Ejemplos reales: `data/processed/idealista/rent_homes_run_20260220_111903/rent_homes_cantabria_bezana_like_raw.csv`, `data/processed/idealista/sale_homes_run_20260218_173035/sale_homes_cantabria_bezana_like_raw.csv`, métricas por ejecución en `summary.json`, y salida geoespacial `data/processed/geo/pois_cantabria.csv`.
- **Gold:** tablas y vistas semánticas orientadas a decisión, incluyendo datasets analíticos finales, variables derivadas, salidas de modelización y métricas de negocio para consumo analítico y financiero.

Esta arquitectura garantiza gobernanza de datos, reproducibilidad y escalabilidad, y evita que la explotación analítica dependa de procesos ad hoc no trazables.

### Flujo metodológico del pipeline

La arquitectura conceptual se articula en seis bloques interrelacionados:

1. **Captura e ingesta de datos**  
   Integra fuentes de distinta naturaleza: extracción inmobiliaria, APIs sectoriales y fuentes macroestructurales/oficiales. El criterio de diseño es preservar la señal original, documentar procedencia y mantener trazabilidad temporal de las extracciones. En la práctica se apoya en componentes como `src/idealistaAPI/ingestion/run_sale_requests.py`, `src/idealistaAPI/ingestion/run_rent_requests.py` y en datasets institucionales en `data/MIVAU/*`. En el caso del web scraping manual, la captura se incorpora directamente por carga manual a la capa de almacenamiento crudo (`data/raw/scraping_manual/*`), desde donde continúa el flujo de preparación y análisis.

2. **Almacenamiento y organización por capas**  
   Separa estado crudo, estado curado y estado analítico para evitar contaminación entre etapas y permitir reejecución controlada. Esta decisión soporta auditoría de transformaciones, versionado analítico y comparabilidad en el tiempo. La separación se materializa en rutas diferenciadas `data/raw/*` y `data/processed/*`, con versionado por ejecución en directorios `*_run_*`.

3. **Preparación y calidad de dato**  
   Incluye limpieza, estandarización, tipado, normalización semántica, detección de inconsistencias e ingeniería de variables. En esta fase, registros heterogéneos se transforman en datasets consistentes para análisis cuantitativo y modelización. Ejemplos reales de implementación son `notebooks/01_manual_scraping_processing/EDA_sale.ipynb`, `EDA_rent.ipynb`, `EDA_terreno.ipynb`, los notebooks de procesamiento API `notebooks/02_idealista_API_processing/idealistaAPI_data_rent.ipynb` y `idealistaAPI_data_sale.ipynb`, y el script `src/idealistaAPI/processing/clean_idealista.py`.

4. **Análisis descriptivo y estructural**  
   Desarrolla análisis exploratorio por segmento (venta, alquiler, suelo) y análisis contextual macroeconómico y sectorial. Esta capa cumple doble función: validación analítica del dato y generación de hipótesis robustas para la modelización. Ejemplos reales: `notebooks/03_macro_and_structural_analysis/analisis_euribor_tipos.ipynb`, `analisis_censoviviendas.ipynb` y `analisis_pestle.ipynb`.

5. **Modelización predictiva y machine learning**  
   Implementa modelos de estimación para precios de suelo, precios de venta y rentas de alquiler, con protocolos de partición, validación, comparación de algoritmos y evaluación mediante métricas homogéneas de rendimiento y error. Esta capa se apoya en la base de features ya construida en fases previas (tipologías normalizadas, variables derivadas y enriquecimiento espacial mediante `src/geospatial_expansion/expand/enricher.py`).

6. **Evaluación de negocio e integración financiera**  
   Traduce resultados analíticos a variables de decisión (ingresos esperados, sensibilidad por escenarios, riesgo y rentabilidad) y los integra en el marco económico-financiero del proyecto para soporte directo a decisiones estratégicas. Esta integración conecta la capa analítica con el objetivo de viabilidad inmobiliaria y con la lógica de decisión propia de Business Analytics.

### Criterios de diseño metodológico

El diseño responde a cuatro criterios centrales del enfoque de Business Analytics:

- **Reproducibilidad:** cada etapa del pipeline es reejecutable con reglas explícitas y salidas verificables.
- **Trazabilidad:** la procedencia del dato y las transformaciones quedan documentadas por capa.
- **Modularidad:** cada componente puede evolucionar sin rediseñar la arquitectura completa.
- **Orientación a decisión:** la cadena analítica culmina en indicadores y escenarios útiles para evaluación de viabilidad e inversión.

En consecuencia, el marco metodológico define una arquitectura analítica integral, preparada para operar de forma continua y para sostener tanto la dimensión técnica predictiva como la dimensión empresarial-financiera del TFM.
