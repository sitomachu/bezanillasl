# Idealista API User Guide

Guia operativa del modulo `src/idealistaAPI` con la estructura actual.

## 1. Requisitos

- Python 3.10+
- Credenciales de Idealista:
  - `IDEALISTA_CLIENT_ID`
  - `IDEALISTA_CLIENT_SECRET`

## 2. Instalacion de dependencias

Desde la raiz del repo:

```bash
python -m pip install -r requirements.txt
```

## 3. Variables de entorno

### Windows (PowerShell)

```powershell
$env:IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
$env:IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"
$env:PYTHONPATH="."
```

### macOS / Linux (bash/zsh)

```bash
export IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
export IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"
export PYTHONPATH="."
```

### Verificar variables de entorno

Windows (PowerShell):

```powershell
echo $env:IDEALISTA_CLIENT_ID
echo $env:IDEALISTA_CLIENT_SECRET
echo $env:PYTHONPATH
```

macOS / Linux:

```bash
echo $IDEALISTA_CLIENT_ID
echo $IDEALISTA_CLIENT_SECRET
echo $PYTHONPATH
```

## 4. Test de conectividad

```bash
python -m src.idealistaAPI.ingestion.test_one_request
```

## 5. Runners principales

### Run nuevo de venta

```bash
python src/idealistaAPI/ingestion/run_sale_requests.py --max-requests 100
```

### Run nuevo de alquiler

```bash
python src/idealistaAPI/ingestion/run_rent_requests.py --max-requests 100
```

### Reanudar el ultimo run de alquiler

Esto reutiliza el ultimo `rent_homes_run_*`, conserva las  requests ya consumidas y, si un circulo ya no aporta novedad, pasa a municipios adicionales definidos en configuracion.

```bash
python src/idealistaAPI/ingestion/run_resume_rent_requests.py --max-requests 100
```

### Limpieza manual de raw -> CSV total

```bash
jupyter notebook notebooks/02_idealista_API_processing/idealistaAPI_raw_to_preprocess.ipynb
```

### Test de conectividad

```bash
python -m src.idealistaAPI.ingestion.test_one_request
```

## 6. Parametros comunes

- `--max-requests`: limite total de requests del run.
- `--max-pages-per-circle`: paginas maximas por circulo.
- `--output-csv`: nombre sugerido en manifest, aunque el notebook consolida por operacion en un CSV total.
- `--no-adaptive-pages`: no cortar por pagina parcial.

Ejemplo:

```bash
python src/idealistaAPI/ingestion/run_rent_requests.py --max-requests 100 --max-pages-per-circle 25 --output-csv rent_custom.csv
```

## 7. Estructura de salida

Por cada run:

- `data/raw/idealistaAPI/raw/<operation>_homes_run_<timestamp>/manifest.json`
- `data/raw/idealistaAPI/raw/<operation>_homes_run_<timestamp>/reqXXX__<circle>__pYYY.json`
- `data/raw/idealistaAPI/preprocess/<operation>_homes_run_<timestamp>/summary.json`

Despues, desde notebook:

- `data/raw/idealistaAPI/preprocess/total_<operation>_properties.csv`
- `data/raw/idealistaAPI/preprocess/total_<operation>_properties_summary.json`

## 8. Regla de parada por cupo

Si la API devuelve cupo agotado/rate limit:

1. El runner se detiene.
2. Se guarda `reqXXX__STOP_QUOTA.json` en `raw`.
3. Se escribe el estado final en `summary.json`.

## 9. Estructura actual del modulo

- `src/idealistaAPI/config/idealista.py`: configuracion central.
- `src/idealistaAPI/ingestion/api_types.py`: tipos de respuesta.
- `src/idealistaAPI/ingestion/client.py`: OAuth + `search`.
- `src/idealistaAPI/ingestion/services/request_service.py`: logica de run/resume.
- `src/idealistaAPI/ingestion/run_*_requests.py`: CLIs.
- `notebooks/02_idealista_API_processing/idealistaAPI_raw_to_preprocess.ipynb`: JSON raw -> CSV total por operacion.
- `notebooks/02_idealista_API_processing/idealistaAPI_data.ipynb`: procesamiento unificado de `rent` o `sale` con trigger `OPERATION`.

## 10. Troubleshooting

- `ModuleNotFoundError: No module named 'src'`:
  - Define `PYTHONPATH=.`.
- `Auth fallida (401/403)`:
  - Revisa `IDEALISTA_CLIENT_ID` y `IDEALISTA_CLIENT_SECRET`.
- El run para antes de `max-requests`:
  - Revisar `summary.json` para distinguir agotamiento de paginas vs cupo API.
- No aparece el CSV tras un run:
  - Es esperado; ahora la limpieza se hace manualmente desde notebook.
