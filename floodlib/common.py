from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SG_TZ = timezone(timedelta(hours=8))


def parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SG_TZ)
    return dt.astimezone(SG_TZ)


def floor_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def datetime_iso(dt: datetime) -> str:
    return dt.isoformat()


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


def read_numeric(value: str | int | float | None) -> float | None:
    return to_float(value)


def parse_api_key(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key in {"api-key", "apikey"} and value.strip():
            return value.strip()
    return None


def load_api_key(path: Path) -> str:
    api_key = parse_api_key(path)
    if api_key:
        return api_key
    raise ValueError(f"No api-key found in {path}")


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


def extract_coordinates(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    coords = geometry.get("coordinates", [])
    points: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list) and node:
            if isinstance(node[0], (int, float)) and len(node) >= 2:
                points.append((float(node[0]), float(node[1])))
                return
            for item in node:
                walk(item)

    walk(coords)
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


def geometry_contains_point(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        return point_in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        return any(point_in_polygon(lon, lat, poly) for poly in coords)
    return False


def load_subzones(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        name = str(props.get("SUBZONE_N", "")).strip()
        if not name:
            continue

        points = extract_coordinates(geometry)
        if points:
            lons = [point[0] for point in points]
            lats = [point[1] for point in points]
            centroid_lon = sum(lons) / len(lons)
            centroid_lat = sum(lats) / len(lats)
            bbox = (min(lons), min(lats), max(lons), max(lats))
        else:
            centroid_lon, centroid_lat = 103.8198, 1.3521
            bbox = (103.5, 1.2, 104.1, 1.5)

        out.append(
            {
                "subzone": name,
                "subzone_upper": name.upper(),
                "planning_area": props.get("PLN_AREA_N"),
                "region": props.get("REGION_N"),
                "geometry": geometry,
                "bbox": bbox,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
            }
        )
    return out


def map_point_to_subzone(lon: float, lat: float, subzones: list[dict[str, Any]]) -> str:
    for subzone in subzones:
        min_lon, min_lat, max_lon, max_lat = subzone["bbox"]
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        if geometry_contains_point(subzone["geometry"], lon, lat):
            return str(subzone["subzone"])

    best_name = ""
    best_dist = float("inf")
    for subzone in subzones:
        dx = lon - float(subzone["centroid_lon"])
        dy = lat - float(subzone["centroid_lat"])
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_name = str(subzone["subzone"])
    return best_name
