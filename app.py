from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pydeck as pdk
import requests
import pandas as pd
import streamlit as st

BOUNDARY_PATH = Path("data/MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson")
PROCESSED_FEATURES_PATH = Path("data/processed/subzone_weather_features.csv")
ENV_PATH = Path(".env")

SG_TZ = timezone(timedelta(hours=8))

FEATURE_META: dict[str, dict[str, str]] = {
    "rainfall_mm_5min": {"label": "Rainfall (5 min, mm)", "fmt": "{:.2f}"},
    "rainfall_mm_15min": {"label": "Rainfall (15 min, mm)", "fmt": "{:.2f}"},
    "rainfall_mm_30min": {"label": "Rainfall (30 min, mm)", "fmt": "{:.2f}"},
    "rainfall_mm_60min": {"label": "Rainfall (60 min, mm)", "fmt": "{:.2f}"},
    "air_temp_c": {"label": "Air Temperature (C)", "fmt": "{:.2f}"},
    "humidity_pct": {"label": "Humidity (%)", "fmt": "{:.1f}"},
    "lightning_count_5min": {"label": "Lightning (5 min, local count)", "fmt": "{:.0f}"},
    "lightning_count_sg_5min": {"label": "Lightning (5 min, SG count)", "fmt": "{:.0f}"},
    "flood_alert_count_sg_5min": {"label": "Flood alerts (5 min, SG count)", "fmt": "{:.0f}"},
    "forecast_rainy_fraction_2h": {"label": "2h Forecast Rainy Fraction", "fmt": "{:.2f}"},
    "forecast_thundery_fraction_2h": {"label": "2h Forecast Thundery Fraction", "fmt": "{:.2f}"},
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.5
    return clamp((value - lower) / (upper - lower), 0.0, 1.0)


def parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SG_TZ)
    return dt.astimezone(SG_TZ)


def floor_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def to_float(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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


@st.cache_data
def load_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_processed_table(path: Path) -> tuple[list[str], dict[str, dict[str, dict[str, float | None]]]]:
    if not path.exists():
        return [], {}

    by_ts: dict[str, dict[str, dict[str, float | None]]] = {}
    timestamps: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ts = str(row.get("timestamp_5min", "")).strip()
            subzone = str(row.get("subzone", "")).strip().upper()
            if not ts or not subzone:
                continue

            parsed = default_feature_row()
            for key in FEATURE_META:
                parsed[key] = to_float(row.get(key))

            by_ts.setdefault(ts, {})[subzone] = parsed
            timestamps.add(ts)

    sorted_ts = sorted(timestamps, key=parse_ts)
    return sorted_ts, by_ts


def extract_coords(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list) and node:
            if isinstance(node[0], (int, float)) and len(node) >= 2:
                points.append((float(node[0]), float(node[1])))
                return
            for inner in node:
                walk(inner)

    walk(geometry.get("coordinates", []))
    return points


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1

    for i in range(n):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_polygon(lon: float, lat: float, polygon_coords: list[Any]) -> bool:
    if not polygon_coords:
        return False
    if not point_in_ring(lon, lat, polygon_coords[0]):
        return False
    for hole in polygon_coords[1:]:
        if point_in_ring(lon, lat, hole):
            return False
    return True


def geometry_contains_point(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        return point_in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        return any(point_in_polygon(lon, lat, poly) for poly in coords)
    return False


@st.cache_data
def build_subzone_index(path: Path) -> list[dict[str, Any]]:
    geojson = load_geojson(path)
    out: list[dict[str, Any]] = []

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        subzone = str(props.get("SUBZONE_N", "")).strip()
        if not subzone:
            continue

        pts = extract_coords(geom)
        if pts:
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            centroid_lon = sum(lons) / len(lons)
            centroid_lat = sum(lats) / len(lats)
            bbox = (min(lons), min(lats), max(lons), max(lats))
        else:
            centroid_lon, centroid_lat = 103.8198, 1.3521
            bbox = (103.5, 1.2, 104.1, 1.5)

        out.append(
            {
                "subzone": subzone,
                "subzone_upper": subzone.upper(),
                "planning_area": props.get("PLN_AREA_N"),
                "region": props.get("REGION_N"),
                "geometry": geom,
                "bbox": bbox,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
            }
        )

    return out


def map_point_to_subzone(lon: float, lat: float, subzones: list[dict[str, Any]]) -> str:
    for sub in subzones:
        min_lon, min_lat, max_lon, max_lat = sub["bbox"]
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        if geometry_contains_point(sub["geometry"], lon, lat):
            return str(sub["subzone_upper"])

    best_name = ""
    best_dist = float("inf")
    for sub in subzones:
        dx = lon - float(sub["centroid_lon"])
        dy = lat - float(sub["centroid_lat"])
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_name = str(sub["subzone_upper"])
    return best_name


def extract_lat_lon(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        for lat_key, lon_key in (("latitude", "longitude"), ("lat", "lon"), ("lat", "lng")):
            if lat_key in value and lon_key in value:
                try:
                    return float(value[lat_key]), float(value[lon_key])
                except (TypeError, ValueError):
                    pass
        for inner in value.values():
            lat, lon = extract_lat_lon(inner)
            if lat is not None and lon is not None:
                return lat, lon
    elif isinstance(value, list):
        for inner in value:
            lat, lon = extract_lat_lon(inner)
            if lat is not None and lon is not None:
                return lat, lon
    return None, None


def parse_api_key(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().lower() == "api-key" and value.strip():
            return value.strip()
    return None


def api_get_json(url: str, api_key: str) -> dict[str, Any]:
    response = requests.get(url, headers={"X-Api-Key": api_key}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    code = payload.get("code")
    if code not in (0, "0", None):
        raise RuntimeError(f"API error for {url}: {payload.get('errorMsg')}")
    return payload


@st.cache_data(ttl=90)
def fetch_live_snapshot(api_key: str, refresh_nonce: int) -> tuple[str, dict[str, dict[str, float | None]], str]:
    _ = refresh_nonce
    subzones = build_subzone_index(BOUNDARY_PATH)
    snapshot: dict[str, dict[str, float | None]] = {s["subzone_upper"]: default_feature_row() for s in subzones}

    # Use lists for averaging station-based features.
    rain_values: dict[str, list[float]] = defaultdict(list)
    temp_values: dict[str, list[float]] = defaultdict(list)
    humidity_values: dict[str, list[float]] = defaultdict(list)
    forecast_rainy: dict[str, list[float]] = defaultdict(list)
    forecast_thundery: dict[str, list[float]] = defaultdict(list)
    lightning_local: dict[str, float] = defaultdict(float)

    lightning_sg = 0.0
    flood_sg = 0.0

    timestamps: list[datetime] = []
    notes: list[str] = []

    # Rainfall / Air Temp / Humidity (station-based)
    for url, value_key, sink in (
        ("https://api-open.data.gov.sg/v2/real-time/api/rainfall", "rainfall_mm_5min", rain_values),
        ("https://api-open.data.gov.sg/v2/real-time/api/air-temperature", "air_temp_c", temp_values),
        ("https://api-open.data.gov.sg/v2/real-time/api/relative-humidity", "humidity_pct", humidity_values),
    ):
        payload = api_get_json(url, api_key)
        data = payload.get("data", {})
        stations = {str(s.get("id", "")): s for s in data.get("stations", [])}
        readings = data.get("readings", [])
        if not readings:
            continue

        latest = max(readings, key=lambda x: x.get("timestamp", ""))
        ts = latest.get("timestamp")
        if ts:
            timestamps.append(parse_ts(ts))

        for measurement in latest.get("data", []):
            station_id = str(measurement.get("stationId", ""))
            value = to_float(measurement.get("value"))
            station = stations.get(station_id, {})
            location = station.get("location", {}) if isinstance(station, dict) else {}
            lat = to_float(location.get("latitude"))
            lon = to_float(location.get("longitude"))
            if value is None or lat is None or lon is None:
                continue
            subzone = map_point_to_subzone(lon=lon, lat=lat, subzones=subzones)
            if subzone:
                sink[subzone].append(value)

    # 2h Forecast
    forecast_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast", api_key)
    forecast_data = forecast_payload.get("data", {})
    area_meta = {str(a.get("name", "")): a for a in forecast_data.get("area_metadata", [])}
    forecast_items = forecast_data.get("items", [])
    if forecast_items:
        latest_item = max(forecast_items, key=lambda x: x.get("timestamp", ""))
        ts = latest_item.get("timestamp")
        if ts:
            timestamps.append(parse_ts(ts))

        for fcast in latest_item.get("forecasts", []):
            area_name = str(fcast.get("area", "")).strip()
            text = str(fcast.get("forecast", "")).lower()
            area = area_meta.get(area_name, {})
            label_location = area.get("label_location", {}) if isinstance(area, dict) else {}
            lat = to_float(label_location.get("latitude"))
            lon = to_float(label_location.get("longitude"))
            if lat is None or lon is None:
                continue

            rainy = 1.0 if any(k in text for k in ("rain", "showers", "drizzle", "thunder")) else 0.0
            thundery = 1.0 if "thunder" in text else 0.0
            subzone = map_point_to_subzone(lon=lon, lat=lat, subzones=subzones)
            if subzone:
                forecast_rainy[subzone].append(rainy)
                forecast_thundery[subzone].append(thundery)

    # Flood alerts
    flood_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/weather/flood-alerts", api_key)
    flood_records = flood_payload.get("data", {}).get("records", [])
    if flood_records:
        latest_record = max(flood_records, key=lambda x: x.get("datetime", ""))
        ts = latest_record.get("datetime")
        if ts:
            timestamps.append(parse_ts(ts))
        readings = ((latest_record.get("item") or {}).get("readings") or [])
        flood_sg = float(len(readings))

    # Lightning
    lightning_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/weather?api=lightning", api_key)
    lightning_records = lightning_payload.get("data", {}).get("records", [])
    if lightning_records:
        latest_record = max(lightning_records, key=lambda x: x.get("datetime", ""))
        ts = latest_record.get("datetime")
        if ts:
            timestamps.append(parse_ts(ts))

        readings = ((latest_record.get("item") or {}).get("readings") or [])
        lightning_sg = float(len(readings))
        for reading in readings:
            lat, lon = extract_lat_lon(reading)
            if lat is None or lon is None:
                continue
            subzone = map_point_to_subzone(lon=lon, lat=lat, subzones=subzones)
            if subzone:
                lightning_local[subzone] += 1.0

    for sub in subzones:
        name = sub["subzone_upper"]
        row = snapshot[name]

        r5 = average_or_none(rain_values.get(name))
        t = average_or_none(temp_values.get(name))
        h = average_or_none(humidity_values.get(name))
        fr = average_or_none(forecast_rainy.get(name))
        ft = average_or_none(forecast_thundery.get(name))

        row["rainfall_mm_5min"] = r5 if r5 is not None else 0.0
        # No labeled historical model yet: estimate 1h context from latest intensity.
        row["rainfall_mm_15min"] = (row["rainfall_mm_5min"] or 0.0) * 3.0
        row["rainfall_mm_30min"] = (row["rainfall_mm_5min"] or 0.0) * 6.0
        row["rainfall_mm_60min"] = (row["rainfall_mm_5min"] or 0.0) * 12.0

        row["air_temp_c"] = t
        row["humidity_pct"] = h
        row["forecast_rainy_fraction_2h"] = fr
        row["forecast_thundery_fraction_2h"] = ft

        row["lightning_count_5min"] = lightning_local.get(name, 0.0)
        row["lightning_count_sg_5min"] = lightning_sg
        row["flood_alert_count_sg_5min"] = flood_sg

    snapshot_ts = max(timestamps).isoformat() if timestamps else datetime.now(tz=SG_TZ).isoformat()
    notes.append("Live mode estimates 15/30/60 min rainfall from latest 5-min reading (proxy).")

    return snapshot_ts, snapshot, " ".join(notes)


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

    # Rule-based proxy model while flood labels are unavailable.
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
    # Blue -> cyan -> yellow -> red
    red = int(30 + 225 * v)
    green = int(80 + 130 * (1.0 - abs(v - 0.5) * 2.0))
    blue = int(230 - 210 * v)
    return [red, green, blue, 180]


def fmt_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "NA"
    return fmt.format(value)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    return ordered[idx]


def geojson_center(features: list[dict[str, Any]]) -> tuple[float, float]:
    lons: list[float] = []
    lats: list[float] = []
    for f in features:
        for lon, lat in extract_coords(f.get("geometry", {})):
            lons.append(lon)
            lats.append(lat)
    if not lons or not lats:
        return 103.8198, 1.3521
    return (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2


def main() -> None:
    st.set_page_config(page_title="Singapore Flood Risk Map", layout="wide")
    st.title("Singapore Flood Risk Map")
    st.caption(
        "Interactive feature map + rule-based flood risk proxy. "
        "Risk uses the last 1h weather context (real model can replace proxy later)."
    )

    if not BOUNDARY_PATH.exists():
        st.error(f"Missing boundary file: {BOUNDARY_PATH}")
        st.stop()

    geojson = load_geojson(BOUNDARY_PATH)
    all_features = geojson.get("features", [])

    regions = sorted({f.get("properties", {}).get("REGION_N", "Unknown") for f in all_features})

    with st.sidebar:
        st.header("Data Source")
        data_mode = st.radio("Mode", ["Historical Replay", "Live API Snapshot"], index=0)

        st.header("Area Filters")
        region = st.selectbox("Region", ["All", *regions], index=0)
        planning_areas = sorted(
            {
                f.get("properties", {}).get("PLN_AREA_N", "Unknown")
                for f in all_features
                if region == "All" or f.get("properties", {}).get("REGION_N") == region
            }
        )
        planning_area = st.selectbox("Planning area", ["All", *planning_areas], index=0)

        st.header("Visualization")
        viz_mode = st.radio("Layer", ["Single Feature", "Combined Flood Risk"], index=1)

        selected_feature = st.selectbox(
            "Feature",
            list(FEATURE_META.keys()),
            format_func=lambda k: FEATURE_META[k]["label"],
            index=3,
            disabled=viz_mode != "Single Feature",
        )

        st.header("Risk Formula (No Labels Yet)")
        w_r60 = st.slider("Weight: Rainfall 60min", 0.0, 2.0, 1.0, 0.05)
        w_r15 = st.slider("Weight: Rainfall 15min", 0.0, 2.0, 0.6, 0.05)
        w_humidity = st.slider("Weight: Humidity", 0.0, 2.0, 0.3, 0.05)
        w_lightning = st.slider("Weight: Lightning", 0.0, 2.0, 0.7, 0.05)
        w_forecast = st.slider("Weight: Forecast Rain", 0.0, 2.0, 0.5, 0.05)
        w_flood_now = st.slider("Weight: Ongoing Flood Alerts", 0.0, 2.0, 1.0, 0.05)
        synthetic_factor = st.slider("Synthetic/Fake Factor", 0.0, 0.6, 0.4, 0.01)

    filtered = [
        f
        for f in all_features
        if (region == "All" or f.get("properties", {}).get("REGION_N") == region)
        and (planning_area == "All" or f.get("properties", {}).get("PLN_AREA_N") == planning_area)
    ]

    snapshot_by_subzone: dict[str, dict[str, float | None]] = {}
    selected_timestamp: str = ""
    live_note = ""
    all_timestamps: list[str] = []
    historical_by_ts: dict[str, dict[str, dict[str, float | None]]] = {}

    if data_mode == "Historical Replay":
        timestamps, by_ts = load_processed_table(PROCESSED_FEATURES_PATH)
        if not timestamps:
            st.error(
                f"No historical feature table found at {PROCESSED_FEATURES_PATH}. "
                "Run your dataset pipeline first."
            )
            st.stop()

        all_timestamps = timestamps
        historical_by_ts = by_ts

        with st.sidebar:
            selected_timestamp = st.select_slider("Timestamp (5 min)", options=timestamps, value=timestamps[-1])

        snapshot_by_subzone = {
            str(k).upper(): v for k, v in (by_ts.get(selected_timestamp) or {}).items()
        }

    else:
        api_key = parse_api_key(ENV_PATH)
        if not api_key:
            st.error("Live mode requires api-key in .env")
            st.stop()

        if "live_refresh_nonce" not in st.session_state:
            st.session_state.live_refresh_nonce = 0

        with st.sidebar:
            if st.button("Refresh Live Data"):
                st.session_state.live_refresh_nonce += 1

        try:
            selected_timestamp, snapshot_by_subzone, live_note = fetch_live_snapshot(
                api_key=api_key,
                refresh_nonce=int(st.session_state.live_refresh_nonce),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live API fetch failed: {exc}")
            st.stop()

    records: list[dict[str, Any]] = []
    raw_feature_values: list[float] = []

    for feature in filtered:
        props = feature.get("properties", {})
        subzone_name = str(props.get("SUBZONE_N", "")).strip()
        subzone_upper = subzone_name.upper()

        row = dict(default_feature_row())
        row.update(snapshot_by_subzone.get(subzone_upper, {}))

        risk = predict_flood_risk(
            row=row,
            subzone_upper=subzone_upper,
            timestamp=selected_timestamp,
            w_r60=w_r60,
            w_r15=w_r15,
            w_humidity=w_humidity,
            w_lightning=w_lightning,
            w_forecast=w_forecast,
            w_flood_now=w_flood_now,
            synthetic_factor=synthetic_factor,
        )

        if viz_mode == "Single Feature":
            feature_value = to_float(row.get(selected_feature))
            if feature_value is not None:
                raw_feature_values.append(feature_value)
        else:
            feature_value = risk
            raw_feature_values.append(feature_value)

        records.append(
            {
                "feature": feature,
                "subzone": subzone_name,
                "row": row,
                "risk": risk,
                "feature_value": feature_value,
            }
        )

    if not records:
        st.warning("No subzones match current filters.")
        st.stop()

    if raw_feature_values:
        vmin = percentile(raw_feature_values, 0.05)
        vmax = percentile(raw_feature_values, 0.95)
        if vmax <= vmin:
            vmax = vmin + 1e-9
    else:
        vmin, vmax = 0.0, 1.0

    out_features: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []

    for rec in records:
        feature = rec["feature"]
        props = dict(feature.get("properties", {}))
        row = rec["row"]
        risk = float(rec["risk"])

        if viz_mode == "Single Feature":
            value = rec["feature_value"]
            label = FEATURE_META[selected_feature]["label"]
            fmt = FEATURE_META[selected_feature]["fmt"]
            value_num = float(value) if value is not None else 0.0
            norm = normalize(value_num, vmin, vmax)
            fill_color = feature_color(norm)
            value_text = fmt_value(value, fmt)
        else:
            label = "Combined Flood Risk"
            value_num = risk
            fill_color = risk_color(risk)
            value_text = f"{risk * 100:.1f}%"

        props["FEATURE_LABEL"] = label
        props["FEATURE_VALUE"] = value_text
        props["RISK_SCORE"] = round(risk, 4)
        props["RISK_PCT"] = round(risk * 100.0, 1)
        props["RISK_LEVEL"] = risk_level(risk)
        props["RISK_COLOR"] = fill_color

        out_features.append(
            {
                "type": feature.get("type", "Feature"),
                "geometry": feature.get("geometry", {}),
                "properties": props,
            }
        )

        ranking.append(
            {
                "Subzone": props.get("SUBZONE_N", "Unknown"),
                "Planning Area": props.get("PLN_AREA_N", "Unknown"),
                "Feature": value_text,
                "Risk %": round(risk * 100.0, 1),
                "Risk Level": props["RISK_LEVEL"],
            }
        )

    ranking.sort(key=lambda x: x["Risk %"], reverse=True)

    def feature_values(key: str) -> list[float]:
        vals: list[float] = []
        for rec in records:
            v = to_float(rec["row"].get(key))
            if v is not None:
                vals.append(v)
        return vals

    risk_values = [float(rec["risk"]) for rec in records]
    avg_risk = (sum(risk_values) / len(risk_values)) * 100.0
    p90_risk = percentile(risk_values, 0.90) * 100.0
    high_risk = sum(1 for v in risk_values if v >= 0.60)
    very_high_risk = sum(1 for v in risk_values if v >= 0.80)

    rain_60_vals = feature_values("rainfall_mm_60min")
    humidity_vals = feature_values("humidity_pct")
    forecast_rain_vals = feature_values("forecast_rainy_fraction_2h")
    lightning_sg_vals = feature_values("lightning_count_sg_5min")
    flood_alert_vals = feature_values("flood_alert_count_sg_5min")

    mean_rain_60 = sum(rain_60_vals) / len(rain_60_vals) if rain_60_vals else 0.0
    max_rain_60 = max(rain_60_vals) if rain_60_vals else 0.0
    mean_humidity = sum(humidity_vals) / len(humidity_vals) if humidity_vals else None
    lightning_sg_now = max(lightning_sg_vals) if lightning_sg_vals else 0.0
    flood_alert_now = max(flood_alert_vals) if flood_alert_vals else 0.0
    mean_forecast_rain = sum(forecast_rain_vals) / len(forecast_rain_vals) if forecast_rain_vals else None

    selected_non_missing = sum(
        1 for rec in records if to_float(rec["row"].get(selected_feature)) is not None
    )
    selected_coverage_pct = 100.0 * selected_non_missing / len(records)

    feature_summary: list[dict[str, str]] = []
    for key, meta in FEATURE_META.items():
        vals = feature_values(key)
        coverage = 100.0 * len(vals) / len(records)
        if vals:
            mean_val = sum(vals) / len(vals)
            p90_val = percentile(vals, 0.90)
            max_val = max(vals)
            fmt = meta["fmt"]
            mean_text = fmt_value(mean_val, fmt)
            p90_text = fmt_value(p90_val, fmt)
            max_text = fmt_value(max_val, fmt)
        else:
            mean_text = "NA"
            p90_text = "NA"
            max_text = "NA"

        feature_summary.append(
            {
                "Feature": meta["label"],
                "Coverage %": round(coverage, 1),
                "Mean": mean_text,
                "P90": p90_text,
                "Max": max_text,
            }
        )

    layer_geojson = {"type": "FeatureCollection", "features": out_features}
    lon, lat = geojson_center(out_features)

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=layer_geojson,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="properties.RISK_COLOR",
        get_line_color=[30, 30, 30, 210],
        line_width_min_pixels=1,
    )

    view = pdk.ViewState(longitude=lon, latitude=lat, zoom=10, min_zoom=9, max_zoom=15, pitch=0, bearing=0)

    tooltip = {
        "html": "<b>Subzone:</b> {SUBZONE_N}<br/>"
        "<b>Planning Area:</b> {PLN_AREA_N}<br/>"
        "<b>Display:</b> {FEATURE_LABEL}<br/>"
        "<b>Value:</b> {FEATURE_VALUE}<br/>"
        "<b>Predicted Flood Risk:</b> {RISK_PCT}% ({RISK_LEVEL})",
        "style": {"backgroundColor": "#1f2937", "color": "white"},
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))

    if data_mode == "Live API Snapshot":
        st.info(f"Live snapshot timestamp: {selected_timestamp}. {live_note}")
    else:
        st.info(f"Replay timestamp: {selected_timestamp}")

    # Charts and plots
    chart_rows: list[dict[str, Any]] = []
    for rec in records:
        feature_val = to_float(rec["row"].get(selected_feature))
        chart_rows.append(
            {
                "subzone": rec["subzone"],
                "region": rec["feature"].get("properties", {}).get("REGION_N", "Unknown"),
                "risk_pct": float(rec["risk"]) * 100.0,
                "feature_value": feature_val,
            }
        )

    chart_df = pd.DataFrame(chart_rows)

    bins = list(range(0, 101, 10))
    hist_rows: list[dict[str, Any]] = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            count = int(((chart_df["risk_pct"] >= low) & (chart_df["risk_pct"] <= high)).sum())
        else:
            count = int(((chart_df["risk_pct"] >= low) & (chart_df["risk_pct"] < high)).sum())
        hist_rows.append({"risk_bin": f"{low}-{high}", "count": count})
    risk_hist_df = pd.DataFrame(hist_rows).set_index("risk_bin")
    top_risk_df = pd.DataFrame(ranking[:15])[["Subzone", "Risk %"]].set_index("Subzone")
    scatter_df = chart_df.dropna(subset=["feature_value"])[["feature_value", "risk_pct"]]
    region_df = chart_df.groupby("region", as_index=False)["risk_pct"].mean().sort_values("risk_pct", ascending=False)

    trend_df: pd.DataFrame | None = None
    if data_mode == "Historical Replay" and all_timestamps and historical_by_ts:
        filtered_subzones = {
            str(f.get("properties", {}).get("SUBZONE_N", "")).strip().upper()
            for f in filtered
            if str(f.get("properties", {}).get("SUBZONE_N", "")).strip()
        }
        ts_cutoff = parse_ts(selected_timestamp)
        trend_points: list[dict[str, Any]] = []
        eligible_ts = [ts for ts in all_timestamps if parse_ts(ts) <= ts_cutoff]
        for ts in eligible_ts[-24:]:
            rows_at_ts = historical_by_ts.get(ts, {})
            risks: list[float] = []
            rains: list[float] = []
            lightning_sg_vals_ts: list[float] = []
            for subzone in filtered_subzones:
                row = dict(default_feature_row())
                row.update(rows_at_ts.get(subzone, {}))
                risk = predict_flood_risk(
                    row=row,
                    subzone_upper=subzone,
                    timestamp=ts,
                    w_r60=w_r60,
                    w_r15=w_r15,
                    w_humidity=w_humidity,
                    w_lightning=w_lightning,
                    w_forecast=w_forecast,
                    w_flood_now=w_flood_now,
                    synthetic_factor=synthetic_factor,
                )
                risks.append(risk * 100.0)
                rains.append(float(row.get("rainfall_mm_60min") or 0.0))
                lightning_sg_vals_ts.append(float(row.get("lightning_count_sg_5min") or 0.0))

            if risks:
                trend_points.append(
                    {
                        "timestamp": ts,
                        "avg_risk_pct": sum(risks) / len(risks),
                        "mean_rain_60": sum(rains) / len(rains),
                        "lightning_sg_5min": sum(lightning_sg_vals_ts) / len(lightning_sg_vals_ts),
                    }
                )
        if trend_points:
            trend_df = pd.DataFrame(trend_points).set_index("timestamp")

    feature_summary_df = pd.DataFrame(feature_summary)
    ranking_df = pd.DataFrame(ranking[:30])

    st.markdown("---")
    st.subheader("Analytics Dashboard")
    st.caption(
        f"Mode: {data_mode} | Region: {region} | Planning Area: {planning_area} | "
        f"Layer: {viz_mode} | Feature: {FEATURE_META[selected_feature]['label']}"
    )

    tab_overview, tab_charts, tab_tables = st.tabs(["Overview", "Charts", "Tables"])

    with tab_overview:
        row1 = st.columns(4)
        row1[0].metric("Avg Predicted Risk", f"{avg_risk:.1f}%")
        row1[1].metric("P90 Predicted Risk", f"{p90_risk:.1f}%")
        row1[2].metric("High Risk (>=60%)", high_risk)
        row1[3].metric("Very High Risk (>=80%)", very_high_risk)

        row2 = st.columns(4)
        row2[0].metric("Mean Rainfall 60min", f"{mean_rain_60:.2f} mm")
        row2[1].metric("Max Rainfall 60min", f"{max_rain_60:.2f} mm")
        row2[2].metric("Lightning SG (5min)", f"{lightning_sg_now:.0f}")
        row2[3].metric("Flood Alerts SG (5min)", f"{flood_alert_now:.0f}")

        row3 = st.columns(4)
        row3[0].metric("Displayed Subzones", len(ranking))
        row3[1].metric("Selected Feature Coverage", f"{selected_coverage_pct:.1f}%")
        row3[2].metric("Mean Humidity", f"{mean_humidity:.1f}%" if mean_humidity is not None else "NA")
        row3[3].metric(
            "Mean Forecast Rain",
            f"{mean_forecast_rain:.2f}" if mean_forecast_rain is not None else "NA",
        )

    with tab_charts:
        ch1, ch2 = st.columns(2)
        with ch1:
            st.caption("Risk Distribution")
            st.bar_chart(risk_hist_df)
        with ch2:
            st.caption("Top 15 Predicted Risk Subzones")
            st.bar_chart(top_risk_df)

        ch3, ch4 = st.columns(2)
        with ch3:
            st.caption(f"{FEATURE_META[selected_feature]['label']} vs Predicted Risk")
            if not scatter_df.empty:
                st.scatter_chart(scatter_df, x="feature_value", y="risk_pct")
            else:
                st.write("No values available for this feature at current selection.")
        with ch4:
            st.caption("Average Predicted Risk by Region")
            st.bar_chart(region_df.set_index("region"))

        if trend_df is not None:
            st.caption("Trend (Last 24 Timestamps)")
            st.line_chart(trend_df[["avg_risk_pct", "mean_rain_60", "lightning_sg_5min"]])

    with tab_tables:
        t1, t2 = st.columns(2)
        with t1:
            st.caption("Feature Summary")
            st.dataframe(
                feature_summary_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Coverage %": st.column_config.ProgressColumn(
                        "Coverage %",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f%%",
                    )
                },
            )
        with t2:
            st.caption("Top Risk Subzones")
            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Risk %": st.column_config.ProgressColumn(
                        "Risk %",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f%%",
                    )
                },
            )


if __name__ == "__main__":
    main()
