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