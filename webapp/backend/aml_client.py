"""Thin client for the AML managed online endpoint (the forecasting model)."""
from __future__ import annotations

from typing import Any

import requests

from config import settings


def get_forecast(
    start: str | None = None,
    horizon_hours: int = 24,
    demand_multiplier: float = 1.0,
    temperature_offset_c: float = 0.0,
) -> dict[str, Any]:
    """Call the deployed model to get a price forecast."""
    if not settings.aml_endpoint_url:
        raise RuntimeError("AML_ENDPOINT_URL is not configured.")

    payload: dict[str, Any] = {"horizon_hours": int(horizon_hours)}
    if start:
        payload["start"] = start
    if demand_multiplier and demand_multiplier != 1.0:
        payload["demand_multiplier"] = float(demand_multiplier)
    if temperature_offset_c:
        payload["temperature_offset_c"] = float(temperature_offset_c)

    headers = {"Content-Type": "application/json"}
    if settings.aml_endpoint_key:
        headers["Authorization"] = f"Bearer {settings.aml_endpoint_key}"

    resp = requests.post(settings.aml_endpoint_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()
