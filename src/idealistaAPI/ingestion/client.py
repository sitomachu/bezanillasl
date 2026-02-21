from __future__ import annotations

import os
import time
import typing as t
import requests

from src.idealistaAPI.ingestion.api_types import SearchResponse

class IdealistaAuthError(RuntimeError):
    """Errores relacionados con credenciales / autenticación."""
    pass


class IdealistaAPIError(RuntimeError):
    """Errores de API (HTTP no OK, timeouts, rate limit persistente, etc.)."""
    pass


class IdealistaClient:
    """
    Cliente mínimo y robusto para Idealista.

    Responsabilidades (y SOLO estas):
    1) Obtener y cachear un access_token OAuth2 (client_credentials)
    2) Ejecutar búsquedas contra /3.5/{country}/search

    Buenas prácticas incorporadas:
    - NO hardcodear credenciales: se leen de variables de entorno.
    - Caché de token con margen (renueva antes de caducar).
    - Reintentos con backoff ante 429/5xx y errores de red.
    - User-Agent identificable.

    Importante:
    - El endpoint de token y el formato de auth pueden variar según el contrato.
      Por defecto se usa:
        POST https://api.idealista.com/oauth/token
        grant_type=client_credentials
        auth=(client_id, client_secret)  # Basic Auth
      Si tu documentación dice otra cosa, ajusta TOKEN_URL y/o _request_token().
    """

    DEFAULT_BASE_URL = "https://api.idealista.com"
    DEFAULT_TOKEN_URL = "https://api.idealista.com/oauth/token"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        base_url: str | None = None,
        token_url: str | None = None,
        timeout_s: int = 30,
        max_retries: int = 4,
        backoff_s: float = 1.2,
        user_agent: str = "tfm-idealista-client/1.0",
    ) -> None:
        # 1) Credenciales: primero parámetros explícitos, luego entorno.
        self.client_id = client_id or os.environ.get("IDEALISTA_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("IDEALISTA_CLIENT_SECRET", "")

        # Si no hay credenciales, fallamos rápido y claro.
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Faltan credenciales. Define IDEALISTA_CLIENT_ID e IDEALISTA_CLIENT_SECRET en el entorno "
                "o pásalas al constructor."
            )

        # 2) URLs configurables por entorno (útil si Idealista te da otra base).
        self.base_url = (base_url or os.environ.get("IDEALISTA_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.token_url = token_url or os.environ.get("IDEALISTA_TOKEN_URL") or self.DEFAULT_TOKEN_URL

        # 3) Parámetros operativos
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.user_agent = user_agent

        # 4) Cache del token
        self._access_token: str | None = None
        self._token_expiry_epoch: float = 0.0

        # 5) Sesión HTTP reutilizable (mejor rendimiento y headers comunes)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    def _sleep_backoff(self, attempt: int) -> None:
        """
        Backoff exponencial suave: si attempt=1 duerme ~1.2s, attempt=2 ~2.4s, etc.
        Evita spamear la API si hay rate limiting o errores temporales.
        """
        time.sleep(self.backoff_s * (2 ** max(0, attempt - 1)))

    def _request_token(self) -> tuple[str, int]:
        """
        Solicita token OAuth2 con grant_type=client_credentials.
        Devuelve: (token, expires_in_seconds)

        Ajusta este método si tu documentación de Idealista exige otro flujo/formato.
        """
        data = {"grant_type": "client_credentials"}

        r = self._session.post(
            self.token_url,
            data=data,
            auth=(self.client_id, self.client_secret),  # Basic Auth
            timeout=self.timeout_s,
        )

        # Credenciales mal: no tiene sentido reintentar.
        if r.status_code in (401, 403):
            raise IdealistaAuthError(f"Auth fallida ({r.status_code}): {r.text}")

        if not r.ok:
            raise IdealistaAPIError(f"Error token ({r.status_code}): {r.text}")

        payload = r.json()
        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 0) or 0)

        if not token:
            raise IdealistaAuthError(f"Respuesta token sin access_token: {payload}")

        # Si no viene expires_in (raro pero posible), aplica TTL conservador para no romper.
        if expires_in <= 0:
            expires_in = 1800  # 30 min

        return token, expires_in

    def get_access_token(self) -> str:
        """
        Devuelve un token válido.
        Renueva si está caducado o a <60s de caducar (margen de seguridad).
        """
        now = time.time()

        if self._access_token and (now < (self._token_expiry_epoch - 60)):
            return self._access_token

        token, expires_in = self._request_token()
        self._access_token = token
        self._token_expiry_epoch = now + expires_in
        return token

    def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, t.Any] | None = None,
        json: dict[str, t.Any] | None = None,
    ) -> requests.Response:
        """
        Envoltorio HTTP con reintentos. Reintenta ante:
        - rate limit (429)
        - errores temporales (5xx)
        - errores de red (timeout, DNS, conexión)
        """
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json,
                    timeout=self.timeout_s,
                )

                # Si rate limit o 5xx, esperamos y reintentamos
                if r.status_code in (429, 500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue

                return r

            except requests.RequestException as e:
                last_err = e
                self._sleep_backoff(attempt)

        raise IdealistaAPIError(f"Fallo tras reintentos llamando {url}: {last_err}")

    def search(
        self,
        *,
        country: str = "es",
        operation: str = "rent",
        property_type: str = "homes",
        num_page: int = 1,
        max_items: int = 50,
        center: str | None = None,
        distance: int | None = None,
        location_id: str | None = None,
        extra_params: dict[str, t.Any] | None = None,
    ) -> SearchResponse:
        """
        Ejecuta una búsqueda Idealista.

        Reglas clave:
        - Debes indicar (center + distance) o locationId.
        - maxItems máximo 50 (si te pasas, se recorta a 50).
        """
        # Validación de parámetros de área
        if not location_id:
            if not center or distance is None:
                raise ValueError("Debes pasar location_id o bien center y distance.")
        else:
            # Si hay location_id, ignoramos center/distance aunque estén
            center, distance = None, None

        # Guardrails en max_items
        max_items = int(max_items)
        if max_items > 50:
            max_items = 50
        if max_items < 1:
            max_items = 1

        url = f"{self.base_url}/3.5/{country}/search"

        # Token y header Authorization
        token = self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        # Payload base requerido por la API
        payload: dict[str, t.Any] = {
            "country": country,
            "operation": operation,
            "propertyType": property_type,
            "numPage": int(num_page),
            "maxItems": max_items,
        }

        # Área de búsqueda
        if location_id:
            payload["locationId"] = location_id
        else:
            payload["center"] = center
            payload["distance"] = int(distance)

        # Filtros extra (precio, tamaño, orden, etc.)
        if extra_params:
            payload.update(extra_params)

        # Request con reintentos
        r = self._request_with_retries("POST", url, headers=headers, data=payload)

        # Si token expiró/invalidó, reintenta una vez forzando refresh
        if r.status_code == 401:
            self._access_token = None
            token = self.get_access_token()
            headers = {"Authorization": f"Bearer {token}"}
            r = self._request_with_retries("POST", url, headers=headers, data=payload)

        if not r.ok:
            raise IdealistaAPIError(f"Search error ({r.status_code}): {r.text}")

        payload: SearchResponse = t.cast(SearchResponse, r.json())
        return payload
