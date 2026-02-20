# Idealista API User Guide

Guia operativa para ejecutar los runners de Idealista en este repositorio.

## 1. Requisitos

- Python 3.10+
- Credenciales de API Idealista:
  - `IDEALISTA_CLIENT_ID`
  - `IDEALISTA_CLIENT_SECRET`

## 2. Variables de entorno

### Windows (PowerShell)

```powershell
$env:IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
$env:IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"
$env:PYTHONPATH="."
```

### macOS / Linux

```bash
export IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
export IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"
export PYTHONPATH="."
```

## 3. Test de conectividad

```powershell
python -m src.ingestion.test_one_request
```

Si falla auth, revisa credenciales. Si falla import de `src`, revisa `PYTHONPATH`.

## 4. Ejecucion de runners

### Venta (run nuevo)

```powershell
python src/ingestion/run_sale_requests.py --max-requests 100
```

### Alquiler (run nuevo)

```powershell
python src/ingestion/run_rent_requests.py --max-requests 100
```

### Resume alquiler (ultimo run)

```powershell
python src/ingestion/resume_rent_requests.py --max-requests 100
```

## 5. Parametros comunes

- `--max-requests`: tope total de requests en el run.
- `--max-pages-per-circle`: paginas maximas por circulo.
- `--output-csv`: nombre del CSV final.
- `--no-adaptive-pages`: no cortar por pagina parcial.

Ejemplo:

```powershell
python src/ingestion/run_rent_requests.py --max-requests 100 --max-pages-per-circle 25 --output-csv rent_custom.csv
```

## 6. Estructura de salida

Para cada run:

- `data/raw/idealista/<operation>_homes_run_<timestamp>/manifest.json`
- `data/raw/idealista/<operation>_homes_run_<timestamp>/reqXXX__<circle>__pYYY.json`
- `data/processed/idealista/<operation>_homes_run_<timestamp>/<output_csv>.csv`
- `data/processed/idealista/<operation>_homes_run_<timestamp>/summary.json`

## 7. Regla de parada por cupo

Cuando la API devuelve error de cupo/rate limit:

1. Se detiene el runner.
2. Se guarda `reqXXX__STOP_QUOTA.json` en `raw`.
3. Se vuelca estado final en `summary.json`.

Esto evita gastar reintentos inutiles cuando no hay cuota disponible.

## 8. Arquitectura actual de ingestion

- `src/idealistaAPI/config/idealista.py`: defaults centralizados.
- `src/idealistaAPI/ingestion/api_types.py`: tipos de respuesta.
- `src/idealistaAPI/ingestion/client.py`: OAuth + `search`.
- `src/idealistaAPI/ingestion/services/request_service.py`: logica de run y resume.
- `src/idealistaAPI/ingestion/run_*_requests.py`: CLIs.

## 9. Troubleshooting rapido

- `ModuleNotFoundError: No module named 'src'`
  - Define `PYTHONPATH=.`
- `Auth fallida (401/403)`
  - Credenciales invalidas o expiradas.
- Stop temprano sin llegar a max requests
  - Puede ser agotamiento real de paginas o cupo API (ver `summary.json`).
