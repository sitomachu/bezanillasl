# BezanillaSL --> Módulo de conexión con API Idealista

Módulo para ingesta de datos inmobiliarios de Idealista orientado a casos de negocio en Cantabria.

## Setup rapido

1. Crea y activa entorno virtual.
2. Instala dependencias del modulo:

```bash
python -m pip install -r requirements.txt
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
│   │   └── idealistaAPI/
│   │       ├── raw/                        # JSON crudos versionados por ejecución.
│   │       └── preprocess/                 # CSV consolidados + summary.json por run.
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
│       │
│       └── processing/                     # Capa de transformación.
│           └── clean_idealista.py          # Conversión JSON raw → CSV estructurado.
```

## Flujos principales

### 1) Run nuevo venta

```powershell
python src/idealistaAPI/ingestion/run_sale_requests.py --max-requests 100
```

### 2) Run nuevo alquiler

```powershell
python src/idealistaAPI/ingestion/run_rent_requests.py --max-requests 100
```

### 2.b) Reanudar el ultimo run de alquiler sin rehacer las requests ya hechas

```bash
python src/idealistaAPI/ingestion/run_resume_rent_requests.py --max-requests 100
```

### 3) Test rápido de conectividad

```powershell
python -m src.idealistaAPI.ingestion.test_one_request
```

## Salidas por run

- Configuracion del run: `data/raw/idealistaAPI/raw/<run>/manifest.json`
- Respuestas API: `data/raw/idealistaAPI/raw/<run>/reqXXX__<circle>__pYYY.json`
- CSV limpio: `data/raw/idealistaAPI/preprocess/<run>/<output_csv>.csv`
- Metricas: `data/raw/idealistaAPI/preprocess/<run>/summary.json`

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
