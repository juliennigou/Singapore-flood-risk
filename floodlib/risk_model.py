from __future__ import annotations

import hashlib
import math

MISSING_FEATURE_COLOR = [160, 160, 160, 110]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.5
    return clamp((value - lower) / (upper - lower), 0.0, 1.0)


def default_feature_row() -> dict[str, float | None]:
    return {
        "rainfall_mm_5min": 0.0,
        "rainfall_mm_15min": 0.0,
        "rainfall_mm_30min": 0.0,
        "rainfall_mm_60min": 0.0,
        "air_temp_c": None,
        "humidity_pct": None,
        "lightning_count_5min": 0.0,
        "lightning_count_sg_5min": 0.0,
        "flood_alert_count_sg_5min": 0.0,
        "forecast_rainy_fraction_2h": None,
        "forecast_thundery_fraction_2h": None,
    }


def average_or_none(values: list[float] | None) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def deterministic_noise(subzone_upper: str, timestamp: str) -> float:
    payload = f"{subzone_upper}|{timestamp}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    raw = int(digest[:8], 16)
    return raw / 0xFFFFFFFF


def predict_flood_risk(
    row: dict[str, float | None],
    subzone_upper: str,
    timestamp: str,
    w_r60: float,
    w_r15: float,
    w_humidity: float,
    w_lightning: float,
    w_forecast: float,
    w_flood_now: float,
    synthetic_factor: float,
) -> float:
    r60 = float(row.get("rainfall_mm_60min") or 0.0)
    r15 = float(row.get("rainfall_mm_15min") or 0.0)
    humidity = float(row.get("humidity_pct") or 70.0)
    lightning_local = float(row.get("lightning_count_5min") or 0.0)
    lightning_sg = float(row.get("lightning_count_sg_5min") or 0.0)
    forecast_rain = float(row.get("forecast_rainy_fraction_2h") or 0.0)
    flood_now = float(row.get("flood_alert_count_sg_5min") or 0.0)

    rain_60_n = normalize(r60, 0.0, 80.0)
    rain_15_n = normalize(r15, 0.0, 30.0)
    humidity_n = normalize(humidity, 55.0, 100.0)
    lightning_proxy = lightning_local if lightning_local > 0 else lightning_sg / 80.0
    lightning_n = normalize(lightning_proxy, 0.0, 8.0)
    forecast_n = normalize(forecast_rain, 0.0, 1.0)
    flood_now_n = normalize(flood_now, 0.0, 4.0)

    signal = (
        w_r60 * rain_60_n
        + w_r15 * rain_15_n
        + w_humidity * humidity_n
        + w_lightning * lightning_n
        + w_forecast * forecast_n
        + w_flood_now * flood_now_n
    )

    risk = 1.0 / (1.0 + math.exp(-3.5 * (signal - 0.55)))

    if synthetic_factor > 0:
        fake = deterministic_noise(subzone_upper=subzone_upper, timestamp=timestamp)
        risk = (1.0 - synthetic_factor) * risk + synthetic_factor * fake

    return clamp(risk, 0.0, 1.0)


def risk_level(value: float) -> str:
    if value >= 0.8:
        return "Very High"
    if value >= 0.6:
        return "High"
    if value >= 0.35:
        return "Medium"
    return "Low"


def risk_color(value: float) -> list[int]:
    v = clamp(value, 0.0, 1.0)
    return [int(35 + 220 * v), int(170 - 140 * v), int(70 - 55 * v), 185]


def feature_color(value: float) -> list[int]:
    v = clamp(value, 0.0, 1.0)
    red = int(30 + 225 * v)
    green = int(80 + 130 * (1.0 - abs(v - 0.5) * 2.0))
    blue = int(230 - 210 * v)
    return [red, green, blue, 180]


def feature_fill_color(value: float | None, lower: float, upper: float) -> list[int]:
    if value is None:
        return MISSING_FEATURE_COLOR
    return feature_color(normalize(float(value), lower, upper))
