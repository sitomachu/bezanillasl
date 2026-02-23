from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def nearest_point(
    lat: float,
    lon: float,
    points: Iterable[Tuple[str, float, float]],
) -> Optional[Tuple[str, float]]:
    nearest_name: Optional[str] = None
    nearest_distance: Optional[float] = None

    for name, plat, plon in points:
        d = haversine_m(lat, lon, plat, plon)
        if nearest_distance is None or d < nearest_distance:
            nearest_name = name
            nearest_distance = d

    if nearest_name is None or nearest_distance is None:
        return None
    return nearest_name, nearest_distance
