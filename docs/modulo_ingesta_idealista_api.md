# Módulo de Ingesta desde la API de Idealista

## Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Arquitectura del Módulo](#arquitectura-del-módulo)
3. [Estructura de Archivos](#estructura-de-archivos)
4. [Componentes del Módulo](#componentes-del-módulo)
   - [Configuración](#configuración-srcidealista-apiconfig)
   - [Cliente API (OAuth2)](#cliente-api-srcidealistaapiclientpy)
   - [Tipos de la API](#tipos-de-la-api-srcidealistaapiapiTypespy)
   - [Servicio de Peticiones (Orquestador)](#servicio-de-peticiones-srcidealistaapi-ingestionservicesrequest_servicepy)
   - [Procesamiento y Deduplicación](#procesamiento-y-deduplicación-srcidealistaapiprocessingclean_idealistepy)
   - [Scripts de Entrada (Runners)](#scripts-de-entrada-runners)
5. [Flujo Completo de Datos](#flujo-completo-de-datos)
6. [Fase 1: Llamada a la API y Almacenamiento Raw](#fase-1-llamada-a-la-api-y-almacenamiento-raw)
   - [Autenticación OAuth2](#autenticación-oauth2)
   - [Estrategia de Localización Geográfica](#estrategia-de-localización-geográfica)
   - [Planificación Round-Robin y Paginación](#planificación-round-robin-y-paginación)
   - [Gestión de Errores y Resiliencia](#gestión-de-errores-y-resiliencia)
   - [Estructura de archivos raw por ejecución](#estructura-de-archivos-raw-por-ejecución)
7. [Fase 2: Deduplicación Intra-Run y Generación de CSV por Ejecución](#fase-2-deduplicación-intra-run-y-generación-de-csv-por-ejecución)
   - [Lógica de deduplicación (dos niveles)](#lógica-de-deduplicación-dos-niveles)
   - [Estructura de archivos preprocess por ejecución](#estructura-de-archivos-preprocess-por-ejecución)
8. [Fase 3: Consolidación de Múltiples Ejecuciones y CSV Final](#fase-3-consolidación-de-múltiples-ejecuciones-y-csv-final)
   - [Notebook de consolidación](#notebook-de-consolidación)
   - [Deduplicación Inter-Run](#deduplicación-inter-run)
   - [Archivos de salida finales](#archivos-de-salida-finales)
9. [Ejecuciones Reales Registradas](#ejecuciones-reales-registradas)
10. [Estadísticas de Deduplicación Observadas](#estadísticas-de-deduplicación-observadas)
11. [Referencia Rápida de Parámetros CLI](#referencia-rápida-de-parámetros-cli)
12. [Diagrama de Flujo Completo](#diagrama-de-flujo-completo)
13. [Campos del CSV de Salida](#campos-del-csv-de-salida)

---

## Visión General

El módulo de ingesta desde la API de Idealista es el sistema responsable de extraer, almacenar y preprocesar los datos de anuncios inmobiliarios (alquiler y venta) en Cantabria, España. Su diseño responde a las restricciones de la API de Idealista: cuotas de peticiones limitadas, paginación (máximo 50 resultados por página), y la necesidad de lanzar múltiples ejecuciones en momentos diferentes para construir un corpus de datos representativo.

El módulo implementa tres fases claramente diferenciadas:

1. **Ingesta raw**: Llamadas a la API de Idealista con autenticación OAuth2, almacenando las respuestas JSON originales intactas, una por petición.
2. **Deduplicación intra-run**: Conversión de los JSON raw de cada ejecución a un único CSV deduplicado para esa ejecución.
3. **Consolidación inter-run**: Fusión de los CSV de múltiples ejecuciones en dos archivos maestros (`total_rent_cantabria.csv` y `total_sales_cantabria.csv`), con deduplicación a nivel global priorizando los registros de las ejecuciones más recientes.

El resultado final son dos archivos CSV con cobertura temporal extendida: cada propiedad aparece exactamente una vez, reflejando sus datos en la ejecución más reciente en que fue observada.

---

## Arquitectura del Módulo

```
src/idealistaAPI/
├── config/
│   └── idealista.py               # Constantes globales, localizaciones, pool geográfico
├── ingestion/
│   ├── client.py                  # Cliente OAuth2 + método search()
│   ├── api_types.py               # TypedDicts: PropertyItem, SearchResponse
│   ├── services/
│   │   └── request_service.py     # Orquestador principal: run_new()
│   ├── run_rent_requests.py       # Runner CLI para alquiler
│   ├── run_sale_requests.py       # Runner CLI para venta
│   ├── run_extended_rent_requests.py  # Runner CLI alquiler extendido (localidades adicionales)
│   └── test_one_request.py        # Test de conectividad (una sola petición)
└── processing/
    └── clean_idealista.py         # JSON → CSV con deduplicación intra-run

notebooks/02_idealista_API_processing/
└── idealistaAPI_raw_to_preprocess.ipynb  # Consolidación multi-run → CSV total
```

---

## Estructura de Archivos

El módulo opera sobre dos directorios de datos bien diferenciados:

```
data/raw/idealistaAPI/
├── raw/                                         # Respuestas JSON originales
│   └── {operation}_homes_run_{TIMESTAMP}/       # Una carpeta por ejecución
│       ├── manifest.json                        # Config de la ejecución
│       ├── req001__{LocationName}__p001.json    # Respuesta de la API (página 1 de LocationName)
│       ├── req002__{LocationName}__p002.json    # Respuesta de la API (página 2 de LocationName)
│       ├── req003__{OtraLocation}__p001.json
│       ├── ...
│       └── req###__STOP_QUOTA.json              # (opcional) si se agotó la cuota
│
└── preprocess/                                  # CSVs procesados
    ├── total_rent_cantabria.csv                 # CSV maestro consolidado de alquiler
    ├── total_rent_cantabria_summary.json        # Metadatos de la consolidación de alquiler
    ├── total_sales_cantabria.csv                # CSV maestro consolidado de venta
    ├── total_sales_cantabria_summary.json       # Metadatos de la consolidación de venta
    └── {operation}_homes_run_{TIMESTAMP}/       # Una carpeta por ejecución
        ├── summary.json                         # Resumen de ejecución
        └── {operation}_homes_cantabria_bezana_like_raw.csv  # CSV deduplicado del run
```

---

## Componentes del Módulo

### Configuración (`src/idealista-api/config/`)

**Archivo:** [src/idealistaAPI/config/idealista.py](../src/idealistaAPI/config/idealista.py)

Define todas las constantes globales y las localizaciones geográficas de trabajo.

**Constantes principales:**

| Constante | Valor | Descripción |
|---|---|---|
| `RAW_BASE` | `data/raw/idealistaAPI/raw` | Directorio de almacenamiento de JSON raw |
| `PROCESSED_BASE` | `data/raw/idealistaAPI/preprocess` | Directorio de CSVs procesados |
| `MAX_ITEMS` | 50 | Máximo de resultados por página (límite de la API) |
| `SLEEP_S` | 0.5 | Pausa entre peticiones para no saturar la API |

**Localizaciones por defecto (`DEFAULT_LOCATIONS`):**

Son 10 municipios del área de Cantabria centrados alrededor de Santa Cruz de Bezana, cada uno con:

- `name`: Nombre del municipio (ej. `"SantaCruzDeBezana"`)
- `location_id`: Código en formato INE de Idealista (`"0-EU-ES-39-{código}"`) — referencia a un área geográfica sin solapamiento
- `fallback_center`: Coordenadas `lat,lon` para búsqueda radial como alternativa
- `fallback_distance_m`: Radio en metros de la búsqueda radial (típicamente 4.000-7.000 m)

Las 10 localizaciones por defecto son: SantaCruzDeBezana, Camargo, Pielagos, Santander, MarinaDeCudeyo, Miengo, RibamontanAlMar, Suances, Laredo y CastroUrdiales.

**Pool de expansión (`_CANTABRIA_POOL`):**

Cuando el presupuesto de peticiones no se ha agotado pero los municipios por defecto ya están exhaustos, el sistema puede cargar automáticamente hasta 41 municipios adicionales de Cantabria, ordenados por similitud geográfica con Santa Cruz de Bezana:

- **Batch 1** (10 mun.): Municipios costeros core (Bareyo, Arnuero, Ribamontán al Mar ampliado, etc.)
- **Batch 2** (10 mun.): Áreas suburbanas de la bahía de Santander
- **Batch 3** (11 mun.): Costa este y oeste
- **Batch 4** (10 mun.): Expansión adicional interior

---

### Cliente API (`src/idealistaAPI/client.py`)

**Archivo:** [src/idealistaAPI/ingestion/client.py](../src/idealistaAPI/ingestion/client.py) (~278 líneas)

Implementa la autenticación OAuth2 y la interfaz de búsqueda contra la API de Idealista.

**Clases de error:**

- `IdealistaAuthError(RuntimeError)`: Fallos de autenticación (HTTP 401/403)
- `IdealistaAPIError(RuntimeError)`: Errores de API, timeouts, rate limits

**Clase principal `IdealistaClient`:**

Se inicializa leyendo `IDEALISTA_CLIENT_ID` e `IDEALISTA_CLIENT_SECRET` del entorno. Parámetros configurables:

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `timeout_s` | 30 | Timeout por petición HTTP |
| `max_retries` | 4 | Número máximo de reintentos por petición |
| `backoff_s` | 1.2 | Base del backoff exponencial |

**Métodos clave:**

- **`_request_token()`**: Ejecuta el flujo OAuth2 `client_credentials`. POST a `/oauth/token` con cabecera Basic Auth. Devuelve `(token, expires_in_segundos)`.
- **`get_access_token()`**: Devuelve el token cacheado si sigue siendo válido (con margen de 60 segundos). Si ha expirado o está próximo a expirar, lo renueva automáticamente.
- **`_sleep_backoff(attempt)`**: Calcula el tiempo de espera exponencial: `0.5s × 2^(attempt-1)`. El intento 1 espera ~1.2s, el intento 2 ~2.4s, etc.
- **`_request_with_retries(...)`**: Ejecuta la petición HTTP con reintentos automáticos en caso de errores 429 (rate limit), 5xx (errores de servidor) o excepciones de red. Lanza excepción si todos los reintentos fallan.
- **`search(...)`** → `SearchResponse`: El método principal de búsqueda. Requiere `operation` ("rent"/"sale"), `property_type` ("homes"), `num_page`, `max_items`. Acepta localización por `location_id` O por `center + distance`. Renueva el token automáticamente si recibe un 401 durante la petición.

---

### Tipos de la API (`src/idealistaAPI/api_types.py`)

**Archivo:** [src/idealistaAPI/ingestion/api_types.py](../src/idealistaAPI/ingestion/api_types.py) (~23 líneas)

Define los `TypedDict` que describen la estructura de las respuestas de la API.

**`PropertyItem`**: Representa un inmueble individual con campos como `propertyCode` (ID único), `price`, `size` (m²), `latitude`, `longitude`, `address`, `province`, `rooms`, `bathrooms`, `district`, y más de 20 campos adicionales.

**`SearchResponse`**: Envuelve la respuesta de una página de búsqueda:
- `elementList`: Lista de 0 a 50 `PropertyItem`
- `totalPages`: Número total de páginas disponibles para esa búsqueda
- `actualPage`: Número de página actual
- `total`: Total de inmuebles en todos los resultados

---

### Servicio de Peticiones (`src/idealista-api-ingestion/services/request_service.py`)

**Archivo:** [src/idealistaAPI/ingestion/services/request_service.py](../src/idealistaAPI/ingestion/services/request_service.py) (~489 líneas)

Este es el orquestador central del módulo. Gestiona toda la lógica de ejecución: scheduling, paginación, fallbacks geográficos, manejo de cuotas y escritura de resultados.

#### Data Classes internas

**`Location`** (frozen dataclass): Representa un municipio con sus cuatro atributos (`name`, `location_id`, `fallback_center`, `fallback_distance_m`).

**`CircleState`**: Registra el estado de paginación en curso para cada localización:

| Campo | Descripción |
|---|---|
| `location` | Objeto `Location` asociado |
| `next_page` | Próxima página a pedir (empieza en 1) |
| `exhausted` | `True` cuando no quedan más páginas que pedir |
| `requests` | Número de peticiones ya realizadas para esta localización |
| `consecutive_errors` | Contador de errores consecutivos (reinicia en éxito) |
| `total_pages` | Total de páginas disponibles (se rellena tras la primera petición exitosa) |

#### Función principal: `run_new(...)`

Esta función ejecuta un run completo de ingesta y devuelve el path del directorio preprocess generado. Sus parámetros son:

| Parámetro | Descripción |
|---|---|
| `operation` | `"rent"` o `"sale"` |
| `max_requests` | Presupuesto total de peticiones API para el run |
| `max_pages_per_circle` | Máximo de páginas a pedir por localización |
| `output_csv_name` | Nombre sugerido para el CSV de salida |
| `no_adaptive_pages` | Si `True`, no para al detectar páginas parciales |
| `force_max_requests` | Si `True`, consume todas las peticiones aunque no haya datos nuevos |
| `locations` | Lista personalizada de localizaciones (por defecto `DEFAULT_LOCATIONS`) |
| `allow_pool_expansion` | Si `True`, expande automáticamente al pool de Cantabria cuando se agotan las localizaciones iniciales |

---

## Flujo Completo de Datos

La siguiente descripción sigue el ciclo de vida de los datos desde la primera llamada a la API hasta el CSV final consolidado.

---

## Fase 1: Llamada a la API y Almacenamiento Raw

### Autenticación OAuth2

Antes de cualquier petición, el cliente obtiene un token de acceso OAuth2 usando el flujo `client_credentials`:

1. POST a `https://api.idealista.com/oauth/token` con las credenciales en cabecera Basic Auth (Base64 de `CLIENT_ID:CLIENT_SECRET`)
2. La API devuelve un `access_token` con una vigencia de ~30 minutos
3. El token se cachea en memoria. El cliente lo renueva automáticamente 60 segundos antes de que expire

En cada petición de búsqueda, el token se incluye como `Authorization: Bearer {token}`. Si la API devuelve un 401, el sistema intenta renovar el token una vez antes de lanzar `IdealistaAuthError`.

### Estrategia de Localización Geográfica

La API de Idealista permite dos modos de búsqueda geográfica:

**Modo 1 — location_id (INE):** Se usa el código de municipio en formato INE (`"0-EU-ES-39-{código}"`). Este modo busca exactamente dentro de los límites administrativos del municipio. No hay solapamiento geográfico entre municipios distintos. Es el modo preferido.

**Modo 2 — center + distance (radial):** Se especifica un punto central (latitud, longitud) y un radio en metros. Se usa como fallback automático si el `location_id` devuelve error 404 (municipio no indexado por Idealista). El sistema activa este modo sin intervención manual, registrando el aviso en los logs.

La activación del fallback es por localización: si `SantaCruzDeBezana` falla con 404, el resto de municipios siguen usando sus `location_id` habituales.

### Planificación Round-Robin y Paginación

El sistema no procesa un municipio completo antes de pasar al siguiente. Usa una estrategia **round-robin** (`_pick_state()`) que distribuye equitativamente las peticiones:

**Criterio de selección en cada iteración:**

1. Localización con menor número de peticiones totales realizadas
2. En caso de empate: la que tiene el `next_page` más bajo
3. En caso de doble empate: orden alfabético

Esto garantiza que ningún municipio monopoliza el presupuesto de peticiones y que el conjunto de datos final tiene representación geográfica equilibrada.

**Reglas de parada por localización:**

Una localización se marca como `exhausted=True` cuando:
- La respuesta devuelve `elementList` vacío (sin resultados)
- `next_page > totalPages` (ya se han pedido todas las páginas disponibles)
- Se detectan 3 errores consecutivos para esa localización
- La página devuelta tiene menos de 50 resultados (página parcial) y no hay información de `totalPages` disponible — salvo que se use `--no-adaptive-pages`

Cuando todas las localizaciones activas están exhaustas, el sistema puede expandir al pool de municipios adicionales (si `allow_pool_expansion=True`) o finalizar el run.

### Gestión de Errores y Resiliencia

**Rate limiting y cuota:**

Si la API devuelve cualquier respuesta que contenga las palabras clave `"quota"`, `"rate limit"`, `"too many requests"`, `"limit exceeded"`, `"cupo"`, o el código HTTP 429/403, el sistema interpreta que se ha agotado la cuota disponible y **detiene inmediatamente el run**. Se guarda un archivo `req{NNN}__STOP_QUOTA.json` en el directorio raw de la ejecución y el `summary.json` se marca con `stopped_by_quota: true`.

**Errores no críticos:**

Para errores que no son de cuota (timeouts, errores 5xx puntuales, errores de red), el sistema:
1. Aplica backoff exponencial con hasta 4 reintentos por petición
2. Si la petición sigue fallando, guarda un `req{NNN}__ERROR.json` con el detalle del error
3. Incrementa `consecutive_errors` para esa localización
4. Continúa con la siguiente localización del round-robin
5. Tras 3 errores consecutivos en la misma localización, la marca como `exhausted`

**Backoff exponencial:** Esperas de ~1.2s, ~2.4s, ~4.8s y ~9.6s entre reintentos.

### Estructura de archivos raw por ejecución

Cada ejecución genera una carpeta con timestamp:

```
data/raw/idealistaAPI/raw/rent_homes_run_20260405_140420/
├── manifest.json
├── req001__SantaCruzDeBezana__p001.json
├── req002__Camargo__p001.json
├── req003__Santander__p001.json
├── req004__Pielagos__p001.json
...
├── req020__Santander__p002.json
├── req021__Camargo__p002.json
...
└── req106__Suances__p005.json
```

**Nomenclatura de archivos:**
- `req{NNN}`: Número de petición con padding de 3 dígitos (001, 002, ..., 999)
- `{LocationName}`: Nombre exacto del municipio tal como está en el objeto `Location`
- `p{PAGE}`: Número de página con padding de 3 dígitos

**`manifest.json`** almacena la configuración completa de la ejecución:

```json
{
  "run_id": "20260405_140420",
  "operation": "rent",
  "property_type": "homes",
  "max_requests": 106,
  "max_pages_per_location": 20,
  "max_items": 50,
  "locations_effective": [...],
  "pool_expansion_enabled": true,
  "strategy": "fair_round_robin_by_locationid_with_center_fallback_no_pool_expansion"
}
```

Cada archivo de respuesta `req{NNN}__{Loc}__p{PAGE}.json` contiene la respuesta completa de la API tal cual la devuelve Idealista: campos `elementList`, `totalPages`, `actualPage`, `total`, y `summary`.

---

## Fase 2: Deduplicación Intra-Run y Generación de CSV por Ejecución

**Archivo:** [src/idealistaAPI/processing/clean_idealista.py](../src/idealistaAPI/processing/clean_idealista.py) (~95 líneas)

Una vez completado el run (o al finalizar `run_new()` en su bloque `finally`), se invoca `clean_json_run(input_dir, output_filename)`. Esta función:

1. **Extrae todos los inmuebles** de todos los archivos `.json` del directorio raw de la ejecución (omite `manifest.json` y archivos `*__ERROR.json` y `*__STOP_QUOTA.json`)
2. **Normaliza a DataFrame** usando `pd.json_normalize()`, que aplana los campos JSON anidados (ej. `priceInfo.price.amount` se convierte en columna `priceInfo.price.amount`)
3. **Deduplica** usando una clave compuesta de dos niveles
4. **Escribe el CSV** resultante en el directorio preprocess correspondiente

### Lógica de deduplicación (dos niveles)

La función `_build_dedupe_key(df)` genera una clave única por inmueble:

```python
# Nivel 1: propertyCode (si está presente y no vacío)
property_code = df['propertyCode'].fillna("").astype(str).str.strip()

# Nivel 2: clave compuesta como fallback
fallback = (
    df['price'].astype(str) + "|" +
    df['size'].astype(str) + "|" +
    df['latitude'].astype(str) + "|" +
    df['longitude'].astype(str) + "|" +
    df['address'].str.lower().str.strip()
)

dedupe_key = property_code.where(property_code != "", fallback)
df = df.drop_duplicates(subset="_dedupe_key", keep="first")
```

**Nivel 1 — `propertyCode`**: Idealista asigna un código único y estable a cada inmueble. Si está presente y no está vacío, es el identificador definitivo. Este es el caso habitual.

**Nivel 2 — clave compuesta** (fallback): Si `propertyCode` está ausente o vacío, se construye una clave con precio + superficie + coordenadas geográficas + dirección normalizada. Esto cubre casos excepcionales de JSON incompleto.

**Política de retención:** `keep="first"` — si el mismo `propertyCode` aparece en varias páginas del mismo run (lo que ocurre cuando las páginas se solapan por cambios en el índice de la API durante la ejecución), se conserva la primera ocurrencia.

**Por qué puede haber duplicados intra-run:** La API de Idealista no garantiza orden estable entre páginas. Si durante el transcurso de un run se publican o eliminan anuncios, las páginas sucesivas pueden incluir resultados ya vistos en páginas anteriores. La deduplicación intra-run elimina estos solapamientos.

### Estructura de archivos preprocess por ejecución

```
data/raw/idealistaAPI/preprocess/rent_homes_run_20260405_140420/
├── summary.json
└── rent_homes_cantabria_bezana_like_raw.csv
```

**`summary.json`** contiene el resumen de ejecución:

```json
{
  "used_requests": 106,
  "stopped_by_quota": false,
  "location_states": {
    "SantaCruzDeBezana": {"requests": 1, "exhausted": true, "total_pages": 1},
    "Santander": {"requests": 6, "exhausted": true, "total_pages": 6},
    "RibamontanAlMar": {"requests": 10, "exhausted": false, "total_pages": 5}
  },
  "csv_path": "data/raw/idealistaAPI/preprocess/rent_homes_run_20260405_140420/rent_homes_cantabria_bezana_like_raw.csv"
}
```

El CSV resultante (`rent_homes_cantabria_bezana_like_raw.csv` o `sale_homes_cantabria_bezana_like_raw.csv`) contiene todos los inmuebles únicos encontrados durante esa ejecución, con los campos JSON aplanados y sin duplicados.

---

## Fase 3: Consolidación de Múltiples Ejecuciones y CSV Final

Las Fases 1 y 2 se ejecutan cada vez que se lanza un runner. Con el paso del tiempo, se acumulan múltiples ejecuciones en `data/raw/idealistaAPI/preprocess/`, cada una correspondiente a una sesión de llamadas realizada en un momento diferente. La Fase 3 fusiona todas estas ejecuciones en un único archivo CSV maestro.

### Notebook de consolidación

**Archivo:** [notebooks/02_idealista_API_processing/idealistaAPI_raw_to_preprocess.ipynb](../notebooks/02_idealista_API_processing/idealistaAPI_raw_to_preprocess.ipynb)

Este notebook se ejecuta manualmente para regenerar los CSVs totales. Tiene una única variable de configuración:

```python
OPERATION = "rent"  # cambiar a "sale" para el otro mercado
```

**Pasos del notebook:**

**1. Detección de runs disponibles:**

```python
operation_runs = sorted(
    [path for path in PREPROCESS_BASE.glob(f"{OPERATION}_homes_run_*") if path.is_dir()],
    reverse=True,   # orden descendente: más recientes primero
)
```

Se detectan automáticamente todas las carpetas `{operation}_homes_run_*` en el directorio preprocess. El orden descendente es clave: las ejecuciones más recientes se cargan primero, lo que determina qué versión de un inmueble se conserva en caso de duplicado (la más reciente).

**2. Carga con estadísticas por run:**

Para cada run se carga su CSV, se calcula la clave de deduplicación y se registran estadísticas:

```python
{
  "run_name": "rent_homes_run_20260405_140420",
  "rows_total": 486,
  "rows_unique": 486,
  "rows_duplicated": 0
}
```

También se añade una columna `source_run` con el nombre del run de origen, que queda en el CSV final permitiendo trazabilidad.

**3. Concatenación:**

Se concatenan todos los DataFrames en uno solo (`df_raw`). En este momento hay duplicados esperados: un mismo inmueble que estuvo publicado durante varios meses aparecerá una vez por cada run en que fue capturado.

**4. Deduplicación inter-run:**

```python
df_raw["_dedupe_key"] = build_property_dedupe_key(df_raw)
df_preprocess = df_raw.drop_duplicates(subset="_dedupe_key", keep="first")
```

Usando exactamente la misma lógica de dos niveles que en la Fase 2, se eliminan todos los duplicados. Como los runs están ordenados de más reciente a más antiguo, `keep="first"` garantiza que para cada inmueble se conserva la versión de la ejecución más reciente en que fue observado.

### Deduplicación Inter-Run

Este es el mecanismo más importante del pipeline completo. Dado que las ejecuciones se realizan en fechas diferentes (por ejemplo, 20/02/2026, 10/03/2026, 01/04/2026 y 05/04/2026), un inmueble que lleve publicado varios meses habrá sido capturado en múltiples runs. Sin deduplicación inter-run, ese inmueble aparecería múltiples veces en el dataset final.

**Ejemplo real (alquiler):**

```
Ejecuciones detectadas (ordenadas de más reciente a más antigua):
  rent_homes_run_20260405_140420: 486 filas (todas únicas)
  rent_homes_run_20260401_135939: 488 filas (todas únicas)
  rent_homes_run_20260310_171627: 524 filas (todas únicas)
  rent_homes_run_20260220_111903: 2365 filas (493 únicas, 1872 duplicados internos)

Total filas tras concatenar: 3863
Duplicados inter-run eliminados: 2988
Filas finales en total_rent_cantabria.csv: 875
```

Las 875 filas del CSV final representan 875 inmuebles distintos que han estado publicados en algún momento entre febrero y abril de 2026 en los municipios cubiertos, cada uno representado por sus datos más recientes.

**Por qué la primera ejecución (20260220) tiene tantos duplicados internos:** La ejecución del 20 de febrero utilizó una configuración anterior con mayor volumen de páginas y localizaciones que producía solapamientos internos. La deduplicación intra-run (Fase 2) ya los redujo de 2365 a 493 únicos, pero al consolidar con runs posteriores, esos 493 aparecen también en runs más recientes, siendo eliminados en la deduplicación inter-run.

### Archivos de salida finales

Los dos archivos maestros se guardan directamente en la raíz del directorio preprocess:

| Archivo | Descripción |
|---|---|
| `data/raw/idealistaAPI/preprocess/total_rent_cantabria.csv` | Alquileres consolidados de Cantabria |
| `data/raw/idealistaAPI/preprocess/total_rent_cantabria_summary.json` | Metadatos de la consolidación de alquiler |
| `data/raw/idealistaAPI/preprocess/total_sales_cantabria.csv` | Ventas consolidadas de Cantabria |
| `data/raw/idealistaAPI/preprocess/total_sales_cantabria_summary.json` | Metadatos de la consolidación de venta |

El `summary.json` de consolidación registra, entre otros campos:
- `runs_included`: Lista de nombres de runs incorporados
- `rows_raw`: Total de filas antes de deduplicar
- `rows_output`: Filas finales tras deduplicar
- `duplicates_removed`: Número de duplicados eliminados
- `priority_rule`: `"latest_run_first"` — la versión más reciente prevalece
- `run_stats`: Estadísticas individuales por run (total, únicos, duplicados)

---

## Ejecuciones Reales Registradas

Las siguientes ejecuciones han sido efectivamente realizadas y están presentes en el repositorio:

### Alquiler (`rent`)

| Run | Fecha | Peticiones | Filas en CSV del run | Notas |
|---|---|---|---|---|
| `rent_homes_run_20260220_111903` | 20/02/2026 | ~106 | 493 (deduplicado) | Primera ejecución grande |
| `rent_homes_run_20260310_171627` | 10/03/2026 | ~106 | 524 | Segunda ejecución |
| `rent_homes_run_20260401_135939` | 01/04/2026 | ~106 | 488 | Tercera ejecución |
| `rent_homes_run_20260405_140420` | 05/04/2026 | 106 | 486 | Cuarta ejecución |

**CSV final:** `total_rent_cantabria.csv` → **875 inmuebles únicos**

### Venta (`sale`)

| Run | Fecha | Peticiones | Filas brutas (raw) | Filas únicas (intra-run) | Notas |
|---|---|---|---|---|---|
| `sale_homes_run_20260218_173035` | 18/02/2026 | 99 | 4.950 | 604 | Primera ejecución; localizaciones basadas en búsqueda radial (centro + radio), diferente estrategia geográfica |
| `sale_homes_run_20260331_174125` | 31/03/2026 | 97 | 4.547 | 2.469 | Segunda ejecución; localizaciones por `location_id` INE (10 municipios estándar) |

**CSV final:** `total_sales_cantabria.csv` → **2.851 inmuebles únicos**

> **Nota sobre el primer run de venta:** La ejecución del 18/02/2026 utilizó una estrategia geográfica distinta basada en búsquedas radiales (centro + radio en metros) con nombres de zona propios (`SotoDeLaMarina`, `Liencres`, `Pielagos_Boo`, `Camargo_Muriedas`, `Somo`, etc.) en lugar de los códigos INE municipales. Esto explica la alta tasa de duplicados intra-run (4.950 → 604): las zonas radiales se solapan geográficamente entre sí, generando el mismo inmueble en varias respuestas. A partir de la segunda ejecución, se adoptó la estrategia definitiva basada en `location_id` INE, que evita estos solapamientos.

---

## Estadísticas de Deduplicación Observadas

### Alquiler (`total_rent_cantabria.csv`)

```
Ejecuciones incluidas:              4 runs (20/02 → 10/03 → 01/04 → 05/04/2026)
Filas brutas antes de consolidar:   3.863
Duplicados eliminados:              2.988  (77,3%)
Inmuebles únicos finales:             875  (22,7%)
```

El alto porcentaje de duplicados (~77%) es esperable y deseado: refleja que la mayoría de los inmuebles activos en el mercado estuvieron publicados durante varias semanas y fueron capturados en múltiples ejecuciones. No es ruido — es la superposición temporal de cuatro snapshots del mercado de alquiler.

### Venta (`total_sales_cantabria.csv`)

```
Ejecuciones incluidas:              2 runs (18/02 → 31/03/2026)
Filas brutas antes de consolidar:   3.073  (604 del run de feb + 2.469 del run de marzo)
Duplicados inter-run eliminados:      222  (7,2%)
Inmuebles únicos finales:           2.851  (92,8%)
```

El bajo porcentaje de duplicados inter-run (~7%) en venta se explica por dos factores combinados: solo hay dos ejecuciones (menor superposición temporal) y la primera ejecución ya entregó muy pocos únicos (604) tras su deduplicación intra-run, por lo que el solapamiento real con la segunda es pequeño. Los datos de venta tienen además una rotación más lenta que el alquiler: los inmuebles en venta permanecen publicados más tiempo, pero el mayor volumen de propiedades distintas en el mercado de compraventa (más de 2.800 únicos frente a 875 en alquiler) refleja la diferencia estructural entre ambos mercados en Cantabria.

### Resumen comparativo

| Dataset | Runs | Filas brutas | Duplicados eliminados | Únicos finales |
|---|---|---|---|---|
| `total_rent_cantabria.csv` | 4 | 3.863 | 2.988 (77,3%) | 875 |
| `total_sales_cantabria.csv` | 2 | 3.073 | 222 (7,2%) | 2.851 |

---

## Referencia Rápida de Parámetros CLI

Todos los runners comparten los mismos argumentos CLI:

```bash
python src/idealistaAPI/ingestion/run_rent_requests.py \
  --max-requests 106 \
  --max-pages-per-circle 20 \
  --output-csv "rent_homes_cantabria_bezana_like_raw.csv"

python src/idealistaAPI/ingestion/run_sale_requests.py \
  --max-requests 106

python src/idealistaAPI/ingestion/run_extended_rent_requests.py \
  --max-requests 100
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--max-requests` | 100 | Presupuesto total de peticiones API |
| `--max-pages-per-circle` | 20 | Máximo de páginas por municipio |
| `--output-csv` | (según runner) | Nombre del CSV de salida en preprocess |
| `--no-adaptive-pages` | False | No parar al detectar páginas parciales |
| `--force-max-requests` | False | Consumir todas las peticiones aunque no haya datos |

El runner extendido (`run_extended_rent_requests.py`) añade 10 municipios costeros adicionales (Bareyo, Arnuero, Comillas, San Vicente de la Barquera, etc.) a las localizaciones por defecto.

---

## Diagrama de Flujo Completo

```
╔══════════════════════════════════════════════════════════════════════╗
║                    EJECUCIÓN DE UN RUN                               ║
╚══════════════════════════════════════════════════════════════════════╝

[Inicio: run_rent_requests.py o run_sale_requests.py]
          │
          │ --max-requests, --max-pages-per-circle
          ▼
[run_new(operation, max_requests, ...)]
          │
          │ Crea directorio:
          │ data/raw/idealistaAPI/raw/{op}_homes_run_{TIMESTAMP}/
          │
          │ Inicializa CircleState para cada Location
          ▼
┌─────────────────────────────────────────────────────┐
│              BUCLE PRINCIPAL (round-robin)           │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ _pick_state() → selecciona Location con      │   │
│  │ menos peticiones (desempate: menor next_page) │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │ _search_one(location, page)                  │   │
│  │  1. Prueba location_id (INE)                 │   │
│  │  2. Si 404 → activa fallback center+distance │   │
│  │  3. IdealistaClient.search()                 │   │
│  │     - OAuth2 token refresh si necesario      │   │
│  │     - POST /3.5/es/search                    │   │
│  │     - Retry con backoff en 429/5xx/red       │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│      ┌──────────┴──────────┐                        │
│      │                     │                        │
│   Éxito                  Error                      │
│      │                     │                        │
│      ▼                     ▼                        │
│  Guarda JSON          ¿Es cuota?                    │
│  req{NNN}__           │        │                    │
│  {Loc}__p{P}.json    Sí       No                    │
│      │                │        │                    │
│  Actualiza         STOP     Guarda                  │
│  CircleState      QUOTA    ERROR.json               │
│  (next_page++)      │      incrementa               │
│      │              │      consecutive_             │
│  ¿exhausted?        │      errors                   │
│      │              │                               │
│      ▼              │                               │
│  Continúa ←─────────┘                              │
│  round-robin                                        │
│                                                     │
│  Cuando: requests >= max_requests                   │
│        O todas las locations exhausted              │
│        O stop_quota                                 │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
              Escribe manifest.json
                        │
                        ▼
╔══════════════════════════════════════════════════════════════════════╗
║             FASE 2: DEDUPLICACIÓN INTRA-RUN                          ║
╚══════════════════════════════════════════════════════════════════════╝
                        │
                        ▼
         [clean_json_run(raw_dir, output_csv_name)]
                        │
              Itera todos los *.json del run
              Extrae elementList de cada uno
              Concatena en un único DataFrame
                        │
                        ▼
              _build_dedupe_key():
              ┌─ propertyCode (si presente)
              └─ fallback: precio|tamaño|lat|lon|dirección
                        │
              drop_duplicates(keep="first")
                        │
              pd.json_normalize() → aplana anidados
                        │
              Escribe CSV:
              preprocess/{run}/
              {op}_homes_cantabria_bezana_like_raw.csv
                        │
              Escribe summary.json


╔══════════════════════════════════════════════════════════════════════╗
║   FASE 3: CONSOLIDACIÓN INTER-RUN (Notebook)                         ║
║   Ejecutada manualmente después de uno o varios runs                 ║
╚══════════════════════════════════════════════════════════════════════╝

[idealistaAPI_raw_to_preprocess.ipynb]
OPERATION = "rent" | "sale"
          │
          ▼
Detecta todos los runs: preprocess/{op}_homes_run_*/
Ordena de más reciente a más antiguo (reverse=True)
          │
          ▼
Para cada run:
  - Carga CSV
  - Genera _dedupe_key
  - Registra estadísticas (total, únicos, duplicados)
  - Añade columna source_run
          │
          ▼
pd.concat(todos los DataFrames)
→ df_raw: N filas brutas (con duplicados inter-run esperados)
          │
          ▼
_build_dedupe_key(df_raw)
drop_duplicates(keep="first")
→ df_preprocess: M filas únicas (M << N)
          │
          ▼
Exporta:
  preprocess/total_{op}_cantabria.csv
  preprocess/total_{op}_cantabria_summary.json
```

---

## Campos del CSV de Salida

Los CSVs finales (`total_rent_cantabria.csv` y `total_sales_cantabria.csv`) contienen los siguientes campos (los campos con punto corresponden a campos JSON anidados que fueron aplanados por `pd.json_normalize()`):

| Campo | Tipo | Descripción |
|---|---|---|
| `propertyCode` | str | Identificador único del inmueble en Idealista |
| `thumbnail` | str | URL de la imagen miniatura |
| `externalReference` | str | Referencia del anunciante |
| `numPhotos` | int | Número de fotografías |
| `floor` | str | Planta del inmueble |
| `price` | float | Precio (€/mes en alquiler, € en venta) |
| `propertyType` | str | Tipo de propiedad (`flat`, `house`, etc.) |
| `operation` | str | `"rent"` o `"sale"` |
| `size` | float | Superficie en m² |
| `exterior` | bool | Si el piso es exterior |
| `rooms` | int | Número de habitaciones |
| `bathrooms` | int | Número de baños |
| `address` | str | Dirección completa |
| `province` | str | Provincia (`"Cantabria"`) |
| `municipality` | str | Municipio |
| `district` | str | Barrio/distrito |
| `country` | str | País (`"es"`) |
| `latitude` | float | Latitud del inmueble |
| `longitude` | float | Longitud del inmueble |
| `showAddress` | bool | Si se muestra la dirección exacta |
| `url` | str | URL del anuncio en idealista.com |
| `distance` | float | Distancia al centro de búsqueda (m) |
| `description` | str | Texto descriptivo del anuncio |
| `hasVideo` | bool | Si tiene vídeo |
| `status` | str | Estado de conservación |
| `newDevelopment` | bool | Si es obra nueva |
| `hasLift` | bool | Si tiene ascensor |
| `priceByArea` | float | Precio por m² |
| `hasPlan` | bool | Si tiene plano |
| `has3DTour` | bool | Si tiene tour 3D |
| `has360` | bool | Si tiene fotografía 360° |
| `hasStaging` | bool | Si tiene staging virtual |
| `notes` | list | Notas adicionales |
| `topNewDevelopment` | bool | Destacado como obra nueva |
| `newDevelopmentHighlight` | bool | Destacado de obra nueva |
| `topPlus` | bool | Anuncio destacado Plus |
| `priceInfo.price.amount` | float | Precio (campo anidado normalizado) |
| `priceInfo.price.currencySuffix` | str | Sufijo de moneda (`"€/mes"`, `"€"`) |
| `detailedType.typology` | str | Tipología detallada |
| `suggestedTexts.subtitle` | str | Subtítulo sugerido (zona + municipio) |
| `suggestedTexts.title` | str | Título sugerido del anuncio |
| `parkingSpace.hasParkingSpace` | bool | Si tiene plaza de garaje |
| `parkingSpace.isParkingSpaceIncludedInPrice` | bool | Si el garaje está incluido en precio |
| `parkingSpace.parkingSpacePrice` | float | Precio del garaje (si es separado) |
| `detailedType.subTypology` | str | Subtipología detallada |
| `priceInfo.price.priceDropInfo.formerPrice` | float | Precio anterior (si ha bajado) |
| `priceInfo.price.priceDropInfo.priceDropValue` | float | Reducción en € |
| `priceInfo.price.priceDropInfo.priceDropPercentage` | float | Reducción en % |
| `source_run` | str | Nombre del run en que fue capturado (ej. `rent_homes_run_20260405_140420`) |

> **Nota:** No todos los campos están presentes en todos los registros. Campos como `parkingSpace.*` o `priceInfo.price.priceDropInfo.*` son opcionales y aparecen como `NaN` cuando no aplican.
