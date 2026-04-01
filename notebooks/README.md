# Notebooks del proyecto

Este directorio agrupa los cuadernos por fase de trabajo analítico.

## Estructura

### `01_manual_scraping_processing`
- `EDA_rent.ipynb`: exploración de datos de alquiler (scraping manual).
- `EDA_sale.ipynb`: exploración de datos de venta (scraping manual).
- `EDA_terreno.ipynb`: exploración de datos de terrenos (scraping manual).

### `02_idealista_API_processing`
- `idealistaAPI_raw_to_preprocess.ipynb`: consolida todos los runs raw de una operacion en un CSV total preprocess.
- `idealistaAPI_data.ipynb`: procesamiento/limpieza unificado de datos de `rent` o `sale` mediante trigger `OPERATION`.

### `03_macro_and_structural_analysis`
- `analisis_censoviviendas.ipynb`: análisis estructural de censo de vivienda.
- `analisis_euribor_tipos.ipynb`: análisis macro de tipos y euríbor.
- `analisis_pestle.ipynb`: análisis estratégico PESTLE.

### `04_terrenos_initial_analysis`
- `analisis_inicial_terrenos.ipynb`: primer análisis descriptivo sobre `data/processed/scraping_manual/terrenos_idealista_clean.csv` con métricas y visualizaciones.

## Convención de uso

1. Ejecutar los cuadernos desde la raíz del repositorio o ajustar `sys.path` en el notebook.
2. Usar datos de entrada en `data/raw/...` y `data/processed/...` según corresponda.
3. Para cuadernos de Idealista API:
   - `idealistaAPI_raw_to_preprocess.ipynb` parte de `data/raw/idealistaAPI/raw/...`.
   - `idealistaAPI_data.ipynb` parte de `data/raw/idealistaAPI/preprocess/...` y cambia entre `rent` y `sale` con `OPERATION`.

## Dependencias recomendadas

- Flujo recomendado:
  - Crear y activar `.venv` desde la raiz del repo.
  - Instalar `requirements.txt` si vas a trabajar con notebooks.
  - Usar siempre el mismo `requirements.txt` global para evitar divergencias.
