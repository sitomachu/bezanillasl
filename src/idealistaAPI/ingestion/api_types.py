from __future__ import annotations

from typing import Any, Dict, List, NotRequired, TypedDict


class PropertyItem(TypedDict, total=False):
    propertyCode: str
    price: float
    size: float
    latitude: float
    longitude: float
    address: str
    province: str


class SearchResponse(TypedDict, total=False):
    elementList: List[PropertyItem]
    totalPages: int
    actualPage: int
    total: int
    summary: str
    extra: NotRequired[Dict[str, Any]]
