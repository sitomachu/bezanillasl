# BezanillaSL --> Módulo de conexión con API Idealista

Módulo para ingesta de datos inmobiliarios de Idealista orientado a casos de negocio en Cantabria.

## Setup rapido

1. Crea y activa entorno virtual.
2. Instala dependencias del modulo:

```powershell
pip install -r src/idealistaAPI/requirements.txt
```

3. Define credenciales de Idealista:

```powershell
$env:IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
$env:IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"
$env:PYTHONPATH="."
```

## Estructura relevante

```text
├── data/
│   ├── raw/
│   │   └── idealista/                      # JSON crudos versionados por ejecución.
│   │
│   └── processed/
│       └── idealista/                      # CSV consolidados + summary.json por run.
│
├── src/
│   └── idealistaAPI/                       # Módulo completo de integración con API Idealista.
│       │
│       ├── config/                         # Configuración central del módulo.
│       │   └── idealista.py                # Paths, max_items, definición de círculos geográficos.
│       │
│       ├── ingestion/                      # Capa de ingesta (descarga desde API).
│       │   ├── client.py                   # Cliente OAuth2 + conexión con Search API.
│       │   ├── api_types.py                # TypedDicts de respuestas API.
│       │   │
│       │   ├── services/
│       │   │   └── request_service.py      # Lógica de ejecución, paginación y resume.
│       │   │
│       │   ├── run_sale_requests.py        # CLI para descarga mercado de venta.
│       │   ├── run_rent_requests.py        # CLI para descarga mercado de alquiler.
│       │   └── resume_rent_requests.py     # CLI para reanudar última ejecución alquiler.
│       │
│       └── processing/                     # Capa de transformación.
│           └── clean_idealista.py          # Conversión JSON raw → CSV estructurado.
```

## Flujos principales

### 1) Run nuevo venta

```powershell
python src/ingestion/run_sale_requests.py --max-requests 100
```

### 2) Run nuevo alquiler

```powershell
python src/ingestion/run_rent_requests.py --max-requests 100
```

### 3) Resume alquiler (ultimo run)

```powershell
python src/ingestion/resume_rent_requests.py --max-requests 100
```

El resume:
- Lee `data/raw/idealista/rent_homes_run_*/manifest.json` del ultimo run.
- Detecta requests ya hechas por nombre de fichero.
- Continua solo con llamadas nuevas hasta `max-requests`.

## Salidas por run

- Configuracion del run: `data/raw/idealista/<run>/manifest.json`
- Respuestas API: `data/raw/idealista/<run>/reqXXX__<circle>__pYYY.json`
- CSV limpio: `data/processed/idealista/<run>/<output_csv>.csv`
- Metricas: `data/processed/idealista/<run>/summary.json`

## Corte automatico por cupo

Si Idealista devuelve error de cupo/rate-limit, los runners:
- paran inmediatamente,
- guardan `reqXXX__STOP_QUOTA.json` en raw,
- escriben estado final en `summary.json`.

## Opciones utiles

Todos los runners principales soportan:

```text
--max-requests
--max-pages-per-circle
--output-csv
--no-adaptive-pages
```

## Nota

Hay un shim de compatibilidad en `src/ingestion/request_runner.py` que reexporta funciones hacia `services/request_service.py`.
