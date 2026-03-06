from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SG_TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]
SUBZONE_GEOJSON = ROOT / "data" / "MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"


def parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SG_TZ)
    return dt.astimezone(SG_TZ)


def floor_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def datetime_iso(dt: datetime) -> str:
    return dt.isoformat()


def extract_coordinates(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list) and node:
            if isinstance(node[0], (int, float)) and len(node) >= 2:
                points.append((float(node[0]), float(node[1])))
                return
            for x in node:
                walk(x)

    walk(geometry.get("coordinates", []))
    return points


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        cross = ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi)
        if cross:
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


def geometry_contains_point(geom: dict[str, Any], lon: float, lat: float) -> bool:
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        return point_in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        return any(point_in_polygon(lon, lat, poly) for poly in coords)
    return False


def load_subzones() -> list[dict[str, Any]]:
    payload = json.loads(SUBZONE_GEOJSON.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for f in payload.get("features", []):
        p = f.get("properties") or {}
        g = f.get("geometry") or {}
        name = str(p.get("SUBZONE_N", "")).strip()
        if not name:
            continue
        pts = extract_coordinates(g)
        if pts:
            lon = sum(x for x, _ in pts) / len(pts)
            lat = sum(y for _, y in pts) / len(pts)
        else:
            lon, lat = 103.8198, 1.3521
        out.append(
            {
                "subzone": name,
                "planning_area": p.get("PLN_AREA_N"),
                "region": p.get("REGION_N"),
                "geometry": g,
                "centroid_lon": lon,
                "centroid_lat": lat,
            }
        )
    return out


def nearest_subzone(lon: float, lat: float, subzones: list[dict[str, Any]]) -> str:
    best = ""
    best_d = float("inf")
    for s in subzones:
        dx = lon - float(s["centroid_lon"])
        dy = lat - float(s["centroid_lat"])
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best = str(s["subzone"])
    return best


def map_point_to_subzone(lon: float, lat: float, subzones: list[dict[str, Any]]) -> str:
    for s in subzones:
        if geometry_contains_point(s["geometry"], lon, lat):
            return str(s["subzone"])
    return nearest_subzone(lon, lat, subzones)


def read_numeric(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_station_map(raw_dir: Path, subzones: list[dict[str, Any]]) -> dict[str, str]:
    station_coords: dict[str, tuple[float, float]] = {}
    for name in ("rainfall.csv", "air_temp.csv", "humidity.csv"):
        path = raw_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                sid = (row.get("station_id") or "").strip()
                if not sid or sid in station_coords:
                    continue
                lat = read_numeric(row.get("latitude"))
                lon = read_numeric(row.get("longitude"))
                if lat is None or lon is None:
                    continue
                station_coords[sid] = (lon, lat)

    mapping: dict[str, str] = {}
    for sid, (lon, lat) in station_coords.items():
        mapping[sid] = map_point_to_subzone(lon, lat, subzones)
    return mapping


def build(raw_dir: Path, output_path: Path) -> None:
    subzones = load_subzones()
    station_map = load_station_map(raw_dir, subzones)

    rain: dict[tuple[str, str], list[float]] = defaultdict(list)
    temp: dict[tuple[str, str], list[float]] = defaultdict(list)
    hum: dict[tuple[str, str], list[float]] = defaultdict(list)
    lightning_sg: dict[str, int] = defaultdict(int)
    flood_sg: dict[str, int] = defaultdict(int)
    forecast_rain: dict[str, list[int]] = defaultdict(list)
    forecast_thunder: dict[str, list[int]] = defaultdict(list)
    timestamps: set[datetime] = set()

    def ingest_station(name: str, value_col: str, target: dict[tuple[str, str], list[float]]) -> None:
        path = raw_dir / name
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                sid = (row.get("station_id") or "").strip()
                ts_raw = row.get("timestamp")
                if not sid or not ts_raw:
                    continue
                subzone = station_map.get(sid)
                if not subzone:
                    continue
                val = read_numeric(row.get(value_col))
                if val is None:
                    continue
                dt = floor_5min(parse_timestamp(ts_raw))
                ts = datetime_iso(dt)
                timestamps.add(dt)
                target[(subzone, ts)].append(val)

    ingest_station("rainfall.csv", "rainfall_mm", rain)
    ingest_station("air_temp.csv", "air_temp_c", temp)
    ingest_station("humidity.csv", "humidity_pct", hum)

    for filename, target in (("lightning.csv", lightning_sg), ("flood_alerts.csv", flood_sg)):
        path = raw_dir / filename
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
                if int(row.get("has_reading") or 0) > 0:
                    target[ts] += 1

    path = raw_dir / "forecast_2h.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                ts_raw = row.get("timestamp")
                if not ts_raw:
                    continue
                dt = floor_5min(parse_timestamp(ts_raw))
                ts = datetime_iso(dt)
                timestamps.add(dt)
                text = (row.get("forecast") or "").lower()
                forecast_rain[ts].append(int(any(k in text for k in ("rain", "showers", "drizzle", "thunder"))))
                forecast_thunder[ts].append(int("thunder" in text))

    if not timestamps:
        raise RuntimeError("No timestamps in raw data")

    min_ts = min(timestamps)
    max_ts = max(timestamps)
    grid: list[datetime] = []
    cur = min_ts
    while cur <= max_ts:
        grid.append(cur)
        cur += timedelta(minutes=5)

    meta = {str(s["subzone"]): s for s in subzones}
    names = sorted(meta)

    rows: list[dict[str, Any]] = []
    idx_by_sub: dict[str, list[int]] = defaultdict(list)
    for subzone in names:
        for dt in grid:
            ts = datetime_iso(dt)
            key = (subzone, ts)
            r = rain.get(key)
            t = temp.get(key)
            h = hum.get(key)
            rf = forecast_rain.get(ts)
            tf = forecast_thunder.get(ts)
            row = {
                "timestamp_5min": ts,
                "subzone": subzone,
                "planning_area": meta[subzone].get("planning_area"),
                "region": meta[subzone].get("region"),
                "rainfall_mm_5min": round(sum(r) / len(r), 4) if r else 0.0,
                "rainfall_mm_15min": 0.0,
                "rainfall_mm_30min": 0.0,
                "rainfall_mm_60min": 0.0,
                "air_temp_c": round(sum(t) / len(t), 4) if t else "",
                "humidity_pct": round(sum(h) / len(h), 4) if h else "",
                "lightning_count_5min": 0,
                "lightning_count_sg_5min": lightning_sg.get(ts, 0),
                "flood_alert_count_sg_5min": flood_sg.get(ts, 0),
                "forecast_rainy_fraction_2h": round(sum(rf) / len(rf), 4) if rf else "",
                "forecast_thundery_fraction_2h": round(sum(tf) / len(tf), 4) if tf else "",
            }
            rows.append(row)
            idx_by_sub[subzone].append(len(rows) - 1)

    for subzone, idxs in idx_by_sub.items():
        prefix = [0.0]
        for idx in idxs:
            prefix.append(prefix[-1] + float(rows[idx]["rainfall_mm_5min"]))

        def wsum(i: int, pts: int) -> float:
            r = i + 1
            l = max(0, r - pts)
            return prefix[r] - prefix[l]

        for i, idx in enumerate(idxs):
            rows[idx]["rainfall_mm_15min"] = round(wsum(i, 3), 4)
            rows[idx]["rainfall_mm_30min"] = round(wsum(i, 6), 4)
            rows[idx]["rainfall_mm_60min"] = round(wsum(i, 12), 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build(Path(args.raw_dir), Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
