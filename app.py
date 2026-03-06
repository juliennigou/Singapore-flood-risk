from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from floodlib.common import (
    SG_TZ,
    extract_coordinates,
    extract_lat_lon,
    floor_5min,
    load_subzones,
    map_point_to_subzone,
    parse_api_key,
    parse_timestamp as parse_ts,
    to_float,
)
from floodlib.risk_model import (
    average_or_none,
    default_feature_row,
    feature_fill_color,
    predict_flood_risk,
    risk_color,
    risk_level,
)

BOUNDARY_PATH = Path("data/MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson")
PROCESSED_FEATURES_PATH = Path("data/processed/subzone_weather_features.csv")
ENV_PATH = Path(".env")

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


@st.cache_data
def build_subzone_index(path: Path) -> list[dict[str, Any]]:
    return load_subzones(path)


def map_point_to_subzone_upper(lon: float, lat: float, subzones: list[dict[str, Any]]) -> str:
    return map_point_to_subzone(lon=lon, lat=lat, subzones=subzones).upper()


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("apikey")
        if key:
            return str(key)
    except Exception:
        pass
    return parse_api_key(ENV_PATH)


def api_get_json(url: str, api_key: str) -> dict[str, Any]:
    response = requests.get(url, headers={"X-Api-Key": api_key}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    code = payload.get("code")
    if code not in (0, "0", None):
        raise RuntimeError(f"API error for {url}: {payload.get('errorMsg')}")
    return payload


def latest_station_snapshot(
    payload: dict[str, Any],
    subzones: list[dict[str, Any]],
) -> tuple[datetime | None, dict[str, float | None]]:
    data = payload.get("data", {})
    stations = {str(station.get("id", "")): station for station in data.get("stations", [])}
    readings = data.get("readings", [])
    if not readings:
        return None, {}

    latest = max(readings, key=lambda item: item.get("timestamp", ""))
    ts_raw = latest.get("timestamp")
    values_by_subzone: dict[str, list[float]] = defaultdict(list)

    for measurement in latest.get("data", []):
        station_id = str(measurement.get("stationId", ""))
        value = to_float(measurement.get("value"))
        station = stations.get(station_id, {})
        location = station.get("location", {}) if isinstance(station, dict) else {}
        lat = to_float(location.get("latitude"))
        lon = to_float(location.get("longitude"))
        if value is None or lat is None or lon is None:
            continue
        subzone = map_point_to_subzone_upper(lon=lon, lat=lat, subzones=subzones)
        if subzone:
            values_by_subzone[subzone].append(value)

    return (
        parse_ts(ts_raw) if ts_raw else None,
        {subzone: average_or_none(values) for subzone, values in values_by_subzone.items()},
    )


def aggregate_live_rainfall(
    payload: dict[str, Any],
    subzones: list[dict[str, Any]],
) -> tuple[datetime | None, dict[str, dict[str, float]]]:
    data = payload.get("data", {})
    stations = {str(station.get("id", "")): station for station in data.get("stations", [])}
    readings = data.get("readings", [])
    rainfall_by_subzone_ts: dict[tuple[str, str], list[float]] = defaultdict(list)
    timestamps: set[datetime] = set()

    for reading in readings:
        ts_raw = reading.get("timestamp")
        if not ts_raw:
            continue
        dt = floor_5min(parse_ts(ts_raw))
        ts = dt.isoformat()
        timestamps.add(dt)
        for measurement in reading.get("data", []):
            station_id = str(measurement.get("stationId", ""))
            value = to_float(measurement.get("value"))
            station = stations.get(station_id, {})
            location = station.get("location", {}) if isinstance(station, dict) else {}
            lat = to_float(location.get("latitude"))
            lon = to_float(location.get("longitude"))
            if value is None or lat is None or lon is None:
                continue
            subzone = map_point_to_subzone_upper(lon=lon, lat=lat, subzones=subzones)
            if subzone:
                rainfall_by_subzone_ts[(subzone, ts)].append(value)

    latest_dt = max(timestamps) if timestamps else None
    rainfall_windows = {
        subzone["subzone_upper"]: {
            "rainfall_mm_5min": 0.0,
            "rainfall_mm_15min": 0.0,
            "rainfall_mm_30min": 0.0,
            "rainfall_mm_60min": 0.0,
        }
        for subzone in subzones
    }
    if latest_dt is None:
        return None, rainfall_windows

    window_points = {
        "rainfall_mm_5min": 1,
        "rainfall_mm_15min": 3,
        "rainfall_mm_30min": 6,
        "rainfall_mm_60min": 12,
    }

    for subzone in rainfall_windows:
        for key, points in window_points.items():
            total = 0.0
            has_value = False
            for step in range(points):
                dt = latest_dt - timedelta(minutes=5 * step)
                values = rainfall_by_subzone_ts.get((subzone, dt.isoformat()))
                if not values:
                    continue
                total += sum(values) / len(values)
                has_value = True
            rainfall_windows[subzone][key] = round(total, 4) if has_value else 0.0

    return latest_dt, rainfall_windows


def fetch_live_snapshot(api_key: str) -> tuple[str, dict[str, dict[str, float | None]], str]:
    subzones = build_subzone_index(BOUNDARY_PATH)
    snapshot: dict[str, dict[str, float | None]] = {s["subzone_upper"]: default_feature_row() for s in subzones}
    timestamps: list[datetime] = []
    notes: list[str] = []

    rainfall_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/rainfall", api_key)
    rainfall_ts, rainfall_windows = aggregate_live_rainfall(rainfall_payload, subzones)
    if rainfall_ts is not None:
        timestamps.append(rainfall_ts)
        notes.append("Live rainfall windows use the timestamps currently returned by the rainfall API.")
    for subzone, windows in rainfall_windows.items():
        snapshot[subzone].update(windows)

    for url, value_key in (
        ("https://api-open.data.gov.sg/v2/real-time/api/air-temperature", "air_temp_c"),
        ("https://api-open.data.gov.sg/v2/real-time/api/relative-humidity", "humidity_pct"),
    ):
        payload = api_get_json(url, api_key)
        ts, values = latest_station_snapshot(payload, subzones)
        if ts is not None:
            timestamps.append(ts)
        for subzone, value in values.items():
            snapshot[subzone][value_key] = value

    forecast_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast", api_key)
    forecast_data = forecast_payload.get("data", {})
    area_meta = {str(area.get("name", "")): area for area in forecast_data.get("area_metadata", [])}
    forecast_items = forecast_data.get("items", [])
    if forecast_items:
        latest_item = max(forecast_items, key=lambda item: item.get("timestamp", ""))
        ts = latest_item.get("timestamp")
        if ts:
            timestamps.append(parse_ts(ts))

        forecast_rainy: dict[str, list[float]] = defaultdict(list)
        forecast_thundery: dict[str, list[float]] = defaultdict(list)
        for forecast in latest_item.get("forecasts", []):
            area_name = str(forecast.get("area", "")).strip()
            text = str(forecast.get("forecast", "")).lower()
            area = area_meta.get(area_name, {})
            label_location = area.get("label_location", {}) if isinstance(area, dict) else {}
            lat = to_float(label_location.get("latitude"))
            lon = to_float(label_location.get("longitude"))
            if lat is None or lon is None:
                continue

            subzone = map_point_to_subzone_upper(lon=lon, lat=lat, subzones=subzones)
            if not subzone:
                continue
            forecast_rainy[subzone].append(
                1.0 if any(token in text for token in ("rain", "showers", "drizzle", "thunder")) else 0.0
            )
            forecast_thundery[subzone].append(1.0 if "thunder" in text else 0.0)

        for subzone in snapshot:
            snapshot[subzone]["forecast_rainy_fraction_2h"] = average_or_none(forecast_rainy.get(subzone))
            snapshot[subzone]["forecast_thundery_fraction_2h"] = average_or_none(forecast_thundery.get(subzone))

    flood_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/weather/flood-alerts", api_key)
    flood_records = flood_payload.get("data", {}).get("records", [])
    flood_sg = 0.0
    if flood_records:
        latest_record = max(flood_records, key=lambda item: item.get("datetime", ""))
        ts = latest_record.get("datetime")
        if ts:
            timestamps.append(parse_ts(ts))
        readings = ((latest_record.get("item") or {}).get("readings") or [])
        flood_sg = float(len(readings))

    lightning_payload = api_get_json("https://api-open.data.gov.sg/v2/real-time/api/weather?api=lightning", api_key)
    lightning_records = lightning_payload.get("data", {}).get("records", [])
    lightning_sg = 0.0
    lightning_local: dict[str, float] = defaultdict(float)
    if lightning_records:
        latest_record = max(lightning_records, key=lambda item: item.get("datetime", ""))
        ts = latest_record.get("datetime")
        if ts:
            timestamps.append(parse_ts(ts))

        readings = ((latest_record.get("item") or {}).get("readings") or [])
        lightning_sg = float(len(readings))
        for reading in readings:
            lat, lon = extract_lat_lon(reading)
            if lat is None or lon is None:
                continue
            subzone = map_point_to_subzone_upper(lon=lon, lat=lat, subzones=subzones)
            if subzone:
                lightning_local[subzone] += 1.0

    for subzone in snapshot:
        snapshot[subzone]["lightning_count_5min"] = lightning_local.get(subzone, 0.0)
        snapshot[subzone]["lightning_count_sg_5min"] = lightning_sg
        snapshot[subzone]["flood_alert_count_sg_5min"] = flood_sg

    snapshot_ts = max(timestamps).isoformat() if timestamps else datetime.now(tz=SG_TZ).isoformat()
    return snapshot_ts, snapshot, " ".join(notes)


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
    for feature in features:
        for lon, lat in extract_coordinates(feature.get("geometry", {})):
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
        api_key = get_api_key()
        if not api_key:
            st.error("Live mode requires `apikey` in Streamlit secrets or `api-key` in local .env")
            st.stop()

        if "live_refresh_nonce" not in st.session_state:
            st.session_state.live_refresh_nonce = 0
        if "live_snapshot_nonce" not in st.session_state:
            st.session_state.live_snapshot_nonce = -1
        if "live_snapshot_payload" not in st.session_state:
            st.session_state.live_snapshot_payload = None

        with st.sidebar:
            if st.button("Refresh Live Data"):
                st.session_state.live_refresh_nonce += 1

        if (
            st.session_state.live_snapshot_payload is None
            or st.session_state.live_snapshot_nonce != st.session_state.live_refresh_nonce
        ):
            try:
                st.session_state.live_snapshot_payload = fetch_live_snapshot(api_key=api_key)
                st.session_state.live_snapshot_nonce = st.session_state.live_refresh_nonce
            except Exception as exc:  # noqa: BLE001
                st.error(f"Live API fetch failed: {exc}")
                st.stop()

        selected_timestamp, snapshot_by_subzone, live_note = st.session_state.live_snapshot_payload

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
            fill_color = feature_fill_color(value, vmin, vmax)
            value_text = fmt_value(value, fmt)
        else:
            label = "Combined Flood Risk"
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
