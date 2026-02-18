# User Guide · Uso de la API de Idealista (macOS y Windows)

Este documento describe cómo configurar el entorno y ejecutar una petición de prueba a la API de Idealista dentro del proyecto **BezanillaSL**.

# 1. Requisitos Previos

- Python 3.10 o superior
- Conexión a internet sin bloqueo SSL (o correctamente configurada)
- Credenciales de acceso a la API:
  - `client_id` (Apikey)
  - `client_secret` (Secret)


## 2. Configuración de Credenciales (IMPORTANTE)

> **ADVERTENCIA:** No introducir credenciales directamente en el código.
> Se deben definir estrictamente como **variables de entorno**.

---

## 3. Configuración en macOS / Linux

Ejecutar desde la raíz del proyecto para cargar las credenciales en la sesión actual de la shell:

```bash
export IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
export IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"
```

echo $IDEALISTA_CLIENT_ID
echo $IDEALISTA_CLIENT_SECRET

## 4. Configuración en Windows (PowerShell)

Ejecutar desde la raíz del proyecto para cargar las credenciales en la sesión actual:

```powershell
$env:IDEALISTA_CLIENT_ID="TU_CLIENT_ID"
$env:IDEALISTA_CLIENT_SECRET="TU_CLIENT_SECRET"

echo $env:IDEALISTA_CLIENT_ID
echo $env:IDEALISTA_CLIENT_SECRET
```

## 5. Ejecución de la Petición de Prueba
macOS / Linux
```bash
python -m src.ingestion.test_one_request
```

Windows
```powershell
python -m src.ingestion.test_one_request
```

## 6. Descarga Automática 100 Requests (Alquiler)

Script:
```bash
src/ingestion/run_rent_100.py
```

Este script:
- Ejecuta hasta --max-requests
- Prioriza zonas tipo Santa Cruz de Bezana
- Guarda JSON en data/raw/idealista/...
- Llama automáticamente a clean_idealista.py
- Genera CSV final en data/processed/idealista/...

Ejecutar 100 requests (Alquiler)

macOS / Linux:
```bash
python -m src.ingestion.run_rent_100 --max-requests 100
```

Windows:
```powershell
python -m src.ingestion.run_rent_100 --max-requests 100
```

Estructura de salida
JSON crudos:
```
data/raw/idealista/rent_homes_run_YYYYMMDD_HHMMSS/
```

CSV procesado automáticamente:
```
data/processed/idealista/rent_homes_run_YYYYMMDD_HHMMSS/rent_homes_cantabria_bezana_like_raw.csv
```

summary.json con métricas:
```
data/processed/idealista/rent_homes_run_YYYYMMDD_HHMMSS/summary.json
```

## 7. Descarga Automática 100 Requests (Venta)

Script:
```bash
src/ingestion/run_sale_100.py
```

Este script:
- Ejecuta hasta --max-requests
- Prioriza zonas tipo Santa Cruz de Bezana
- Guarda JSON en data/raw/idealista/...
- Llama automáticamente a clean_idealista.py
- Genera CSV final en data/processed/idealista/...

Ejecutar 100 requests (Venta)

macOS / Linux:
```bash
python -m src.ingestion.run_sale_100 --max-requests 100
```

Windows:
```powershell
python -m src.ingestion.run_sale_100 --max-requests 100
```

Estructura de salida
JSON crudos:
```
data/raw/idealista/sale_homes_run_YYYYMMDD_HHMMSS/
```

CSV procesado automáticamente:
```
data/processed/idealista/sale_homes_run_YYYYMMDD_HHMMSS/sale_homes_cantabria_bezana_like_raw.csv
```

summary.json con métricas:
```
data/processed/idealista/sale_homes_run_YYYYMMDD_HHMMSS/summary.json
```

## 8. Qué Hace Internamente run_*_100
- Recorre círculos geográficos (priorizando Bezana-like)
- Descarga hasta 50 anuncios por request
- Controla páginas adaptativamente
- Guarda cada respuesta en JSON
- Llama a:
```
clean_json_run(...)
```
- Genera CSV final

## 9. Arquitectura Final

**run_rent_100.py**  
**run_sale_100.py**  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
**JSON** en `data/raw/`  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
**clean_idealista.py**  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
**CSV** en `data/processed/`
