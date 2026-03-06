from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from floodlib.common import datetime_iso, floor_5min, load_subzones, map_point_to_subzone, parse_timestamp, read_numeric

PROCESSED_FIELDS = [
    "timestamp_5min",
    "subzone",
    "planning_area",
    "region",
    "rainfall_mm_5min",
    "rainfall_mm_15min",
    "rainfall_mm_30min",
    "rainfall_mm_60min",
    "air_temp_c",
    "humidity_pct",
    "lightning_count_5min",
    "lightning_count_sg_5min",
    "flood_alert_count_sg_5min",
    "forecast_rainy_fraction_2h",
    "forecast_thundery_fraction_2h",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_station_to_subzone(subzones: list[dict[str, Any]], csv_paths: list[Path]) -> dict[str, str]:
    coords: dict[str, tuple[float, float]] = {}
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                sid = (row.get("station_id") or "").strip()
                if not sid or sid in coords:
                    continue
                lat = read_numeric(row.get("latitude"))
                lon = read_numeric(row.get("longitude"))
                if lat is None or lon is None:
                    continue
                coords[sid] = (lon, lat)

    mapping: dict[str, str] = {}
    for sid, (lon, lat) in coords.items():
        mapping[sid] = map_point_to_subzone(lon=lon, lat=lat, subzones=subzones)
    return mapping


def build_processed_table(raw_dir: Path, output_path: Path, subzone_geojson_path: Path) -> None:
    subzones = load_subzones(subzone_geojson_path)
    if not subzones:
        raise RuntimeError("No subzones found")

    station_to_subzone = load_station_to_subzone(
        subzones=subzones,
        csv_paths=[raw_dir / "rainfall.csv", raw_dir / "air_temp.csv", raw_dir / "humidity.csv"],
    )

    rain_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    temp_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    hum_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    lightning_local: dict[tuple[str, str], int] = defaultdict(int)
    lightning_global: dict[str, int] = defaultdict(int)
    flood_global: dict[str, int] = defaultdict(int)
    forecast_rainy: dict[tuple[str, str], list[int]] = defaultdict(list)
    forecast_thundery: dict[tuple[str, str], list[int]] = defaultdict(list)
    timestamps: set[datetime] = set()

    def ingest_station(path: Path, value_col: str, target: dict[tuple[str, str], list[float]]) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                ts_raw = row.get("timestamp")
                sid = (row.get("station_id") or "").strip()
                if not ts_raw or not sid:
                    continue
                val = read_numeric(row.get(value_col))
                if val is None:
                    continue
                subzone = station_to_subzone.get(sid)
                if not subzone:
                    continue
                dt = floor_5min(parse_timestamp(ts_raw))
                ts = datetime_iso(dt)
                timestamps.add(dt)
                target[(subzone, ts)].append(val)

    ingest_station(raw_dir / "rainfall.csv", "rainfall_mm", rain_values)
    ingest_station(raw_dir / "air_temp.csv", "air_temp_c", temp_values)
    ingest_station(raw_dir / "humidity.csv", "humidity_pct", hum_values)

    for path, is_flood in ((raw_dir / "lightning.csv", False), (raw_dir / "flood_alerts.csv", True)):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                ts_raw = row.get("timestamp")
                if not ts_raw:
                    continue
                dt = floor_5min(parse_timestamp(ts_raw))
                ts = datetime_iso(dt)
                timestamps.add(dt)
                has_reading = int(row.get("has_reading") or 0)
                if has_reading <= 0:
                    continue
                if is_flood:
                    flood_global[ts] += 1
                    continue

                lightning_global[ts] += 1
                lat = read_numeric(row.get("latitude"))
                lon = read_numeric(row.get("longitude"))
                if lat is None or lon is None:
                    continue
                subzone = map_point_to_subzone(lon=lon, lat=lat, subzones=subzones)
                lightning_local[(subzone, ts)] += 1

    forecast_path = raw_dir / "forecast_2h.csv"
    if forecast_path.exists():
        with forecast_path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                ts_raw = row.get("timestamp")
                lat = read_numeric(row.get("latitude"))
                lon = read_numeric(row.get("longitude"))
                if not ts_raw or lat is None or lon is None:
                    continue
                dt = floor_5min(parse_timestamp(ts_raw))
                ts = datetime_iso(dt)
                timestamps.add(dt)
                text = (row.get("forecast") or "").lower()
                subzone = map_point_to_subzone(lon=lon, lat=lat, subzones=subzones)
                forecast_rainy[(subzone, ts)].append(
                    int(any(token in text for token in ("rain", "showers", "drizzle", "thunder")))
                )
                forecast_thundery[(subzone, ts)].append(int("thunder" in text))

    if not timestamps:
        raise RuntimeError("No timestamps found in raw data")

    min_ts = min(timestamps)
    max_ts = max(timestamps)
    grid: list[datetime] = []
    cur = min_ts
    while cur <= max_ts:
        grid.append(cur)
        cur += timedelta(minutes=5)

    subzone_meta = {str(item["subzone"]): item for item in subzones}
    subzone_names = sorted(subzone_meta.keys())

    rows: list[dict[str, Any]] = []
    idx_by_subzone: dict[str, list[int]] = defaultdict(list)

    for subzone in subzone_names:
        for dt in grid:
            ts = datetime_iso(dt)
            key = (subzone, ts)
            rain = rain_values.get(key)
            temp = temp_values.get(key)
            hum = hum_values.get(key)
            rain_frac = forecast_rainy.get(key)
            thunder_frac = forecast_thundery.get(key)
            row = {
                "timestamp_5min": ts,
                "subzone": subzone,
                "planning_area": subzone_meta[subzone].get("planning_area"),
                "region": subzone_meta[subzone].get("region"),
                "rainfall_mm_5min": round(sum(rain) / len(rain), 4) if rain else 0.0,
                "rainfall_mm_15min": 0.0,
                "rainfall_mm_30min": 0.0,
                "rainfall_mm_60min": 0.0,
                "air_temp_c": round(sum(temp) / len(temp), 4) if temp else "",
                "humidity_pct": round(sum(hum) / len(hum), 4) if hum else "",
                "lightning_count_5min": lightning_local.get(key, 0),
                "lightning_count_sg_5min": lightning_global.get(ts, 0),
                "flood_alert_count_sg_5min": flood_global.get(ts, 0),
                "forecast_rainy_fraction_2h": round(sum(rain_frac) / len(rain_frac), 4) if rain_frac else "",
                "forecast_thundery_fraction_2h": (
                    round(sum(thunder_frac) / len(thunder_frac), 4) if thunder_frac else ""
                ),
            }
            rows.append(row)
            idx_by_subzone[subzone].append(len(rows) - 1)

    for indices in idx_by_subzone.values():
        prefix: list[float] = [0.0]
        for idx in indices:
            prefix.append(prefix[-1] + float(rows[idx]["rainfall_mm_5min"]))

        def rolling_sum(i: int, points: int) -> float:
            right = i + 1
            left = max(0, right - points)
            return prefix[right] - prefix[left]

        for i, idx in enumerate(indices):
            rows[idx]["rainfall_mm_15min"] = round(rolling_sum(i, 3), 4)
            rows[idx]["rainfall_mm_30min"] = round(rolling_sum(i, 6), 4)
            rows[idx]["rainfall_mm_60min"] = round(rolling_sum(i, 12), 4)

    write_csv(output_path, rows, PROCESSED_FIELDS)
