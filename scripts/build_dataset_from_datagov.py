from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

SG_TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DEFAULT_RAW_DIR = DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"
SUBZONE_GEOJSON = DATA_DIR / "MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"

DATASETS: list[dict[str, Any]] = [
    {
        "name": "flood_alerts",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/weather/flood-alerts",
        "kind": "weather_records",
        "output": "flood_alerts.csv",
    },
    {
        "name": "rainfall",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/rainfall",
        "kind": "station_readings",
        "value_column": "rainfall_mm",
        "output": "rainfall.csv",
    },
    {
        "name": "air_temp",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/air-temperature",
        "kind": "station_readings",
        "value_column": "air_temp_c",
        "output": "air_temp.csv",
    },
    {
        "name": "humidity",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity",
        "kind": "station_readings",
        "value_column": "humidity_pct",
        "output": "humidity.csv",
    },
    {
        "name": "lightning",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/weather?api=lightning",
        "kind": "weather_records",
        "output": "lightning.csv",
    },
    {
        "name": "forecast_2h",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast",
        "kind": "forecast",
        "output": "forecast_2h.csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data.gov.sg weather/flood datasets and build subzone+timestamp_5min feature table."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--env-file", default=str(ROOT / ".env"), help="Path to .env with api-key")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Raw output directory")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=8, help="Retries for HTTP 429/5xx")
    parser.add_argument("--max-pages-per-day", type=int, default=120, help="Pagination safety cap")
    parser.add_argument("--sleep-between-requests", type=float, default=0.10, help="Throttle requests")
    parser.add_argument(
        "--output-processed",
        default=str(DEFAULT_PROCESSED_DIR / "subzone_weather_features.csv"),
        help="Processed feature table path",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_api_key(env_path: Path) -> str:
    from floodlib.common import load_api_key as shared_load_api_key

    return shared_load_api_key(env_path)


def iter_days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SG_TZ)
    return dt.astimezone(SG_TZ)


def floor_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def datetime_iso(dt: datetime) -> str:
    return dt.isoformat()


def request_json(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    timeout: int,
    max_retries: int,
    sleep_between_requests: float,
) -> dict[str, Any]:
    backoff = 0.7
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(backoff * (2**attempt), 20.0))
                continue
            response.raise_for_status()
            payload = response.json()
            code = payload.get("code")
            if code not in (0, "0", None):
                raise RuntimeError(
                    f"API error code={code}, name={payload.get('name')}, error={payload.get('errorMsg')}"
                )
            if sleep_between_requests > 0:
                time.sleep(sleep_between_requests)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(backoff * (2**attempt), 20.0))
    raise RuntimeError(f"Failed request {url} params={params}: {last_error}")


def extract_page_datetimes(kind: str, data: dict[str, Any]) -> list[datetime]:
    out: list[datetime] = []
    if kind == "station_readings":
        for reading in data.get("readings", []):
            ts = reading.get("timestamp")
            if ts:
                out.append(parse_timestamp(ts))
    elif kind == "forecast":
        for item in data.get("items", []):
            ts = item.get("timestamp")
            if ts:
                out.append(parse_timestamp(ts))
    else:
        for rec in data.get("records", []):
            ts = rec.get("datetime")
            if ts:
                out.append(parse_timestamp(ts))
    return out


def fetch_paginated(
    url: str,
    kind: str,
    day: date,
    headers: dict[str, str],
    timeout: int,
    max_retries: int,
    max_pages_per_day: int,
    sleep_between_requests: float,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()

    for _ in range(max_pages_per_day):
        params: dict[str, Any] = {"date": day.isoformat()}
        if token:
            params["paginationToken"] = token

        payload = request_json(
            url=url,
            headers=headers,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            sleep_between_requests=sleep_between_requests,
        )
        data = payload.get("data") or {}
        pages.append(data)

        page_times = extract_page_datetimes(kind=kind, data=data)
        if page_times and min(t.date() for t in page_times) < day:
            # Token stream crossed into older day; stop here.
            break

        next_token = data.get("paginationToken")
        if not next_token:
            break
        next_token = str(next_token)
        if next_token in seen_tokens:
            break

        seen_tokens.add(next_token)
        token = next_token

    return pages


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


def flatten_station_pages(
    pages: list[dict[str, Any]],
    target_day: date,
    value_column: str,
    dedupe: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    stations: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for page in pages:
        for station in page.get("stations", []):
            sid = str(station.get("id", ""))
            if not sid:
                continue
            loc = station.get("location") or {}
            stations[sid] = {
                "station_id": sid,
                "device_id": station.get("deviceId"),
                "station_name": station.get("name"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
            }

    for page in pages:
        for reading in page.get("readings", []):
            ts_raw = reading.get("timestamp")
            if not ts_raw:
                continue
            dt = floor_5min(parse_timestamp(ts_raw))
            if dt.date() != target_day:
                continue
            ts = datetime_iso(dt)
            for m in reading.get("data", []):
                sid = str(m.get("stationId", ""))
                if not sid:
                    continue
                key = (ts, sid)
                if key in dedupe:
                    continue
                dedupe.add(key)
                meta = stations.get(
                    sid,
                    {
                        "station_id": sid,
                        "device_id": None,
                        "station_name": None,
                        "latitude": None,
                        "longitude": None,
                    },
                )
                rows.append({"timestamp": ts, **meta, value_column: m.get("value")})
    return rows


def flatten_forecast_pages(
    pages: list[dict[str, Any]],
    target_day: date,
    dedupe: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    area_lookup: dict[str, tuple[float | None, float | None]] = {}

    for page in pages:
        for area in page.get("area_metadata", []):
            name = str(area.get("name", "")).strip()
            if not name:
                continue
            loc = area.get("label_location") or {}
            area_lookup[name] = (loc.get("latitude"), loc.get("longitude"))

    for page in pages:
        for item in page.get("items", []):
            ts_raw = item.get("timestamp")
            if not ts_raw:
                continue
            dt = floor_5min(parse_timestamp(ts_raw))
            if dt.date() != target_day:
                continue
            ts = datetime_iso(dt)
            valid = item.get("valid_period") or {}
            for fc in item.get("forecasts", []):
                area = str(fc.get("area", "")).strip()
                if not area:
                    continue
                key = (ts, area)
                if key in dedupe:
                    continue
                dedupe.add(key)
                lat, lon = area_lookup.get(area, (None, None))
                rows.append(
                    {
                        "timestamp": ts,
                        "update_timestamp": item.get("update_timestamp"),
                        "valid_start": valid.get("start"),
                        "valid_end": valid.get("end"),
                        "area": area,
                        "forecast": fc.get("forecast"),
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
    return rows


def flatten_weather_record_pages(
    pages: list[dict[str, Any]],
    target_day: date,
    dedupe: set[tuple[str, int, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for page in pages:
        for rec in page.get("records", []):
            ts_raw = rec.get("datetime")
            if not ts_raw:
                continue
            dt = floor_5min(parse_timestamp(ts_raw))
            if dt.date() != target_day:
                continue
            ts = datetime_iso(dt)

            item = rec.get("item") or {}
            readings = item.get("readings") or []
            if not readings:
                key = (ts, -1, "")
                if key in dedupe:
                    continue
                dedupe.add(key)
                rows.append(
                    {
                        "timestamp": ts,
                        "updated_timestamp": rec.get("updatedTimestamp"),
                        "item_type": item.get("type"),
                        "is_station_data": item.get("isStationData"),
                        "reading_index": "",
                        "reading_json": "",
                        "latitude": "",
                        "longitude": "",
                        "has_reading": 0,
                    }
                )
                continue

            for idx, reading in enumerate(readings):
                reading_json = json.dumps(reading, sort_keys=True, separators=(",", ":"))
                key = (ts, idx, reading_json)
                if key in dedupe:
                    continue
                dedupe.add(key)
                lat, lon = extract_lat_lon(reading)
                rows.append(
                    {
                        "timestamp": ts,
                        "updated_timestamp": rec.get("updatedTimestamp"),
                        "item_type": item.get("type"),
                        "is_station_data": item.get("isStationData"),
                        "reading_index": idx,
                        "reading_json": reading_json,
                        "latitude": lat if lat is not None else "",
                        "longitude": lon if lon is not None else "",
                        "has_reading": 1,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_processed_table(raw_dir: Path, output_path: Path) -> None:
    from floodlib.feature_pipeline import build_processed_table as build_canonical_processed_table

    build_canonical_processed_table(
        raw_dir=raw_dir,
        output_path=output_path,
        subzone_geojson_path=SUBZONE_GEOJSON,
    )


def download_raw_csvs(
    raw_dir: Path,
    start: date,
    end: date,
    api_key: str,
    timeout: int,
    max_retries: int,
    max_pages_per_day: int,
    sleep_between_requests: float,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {"X-Api-Key": api_key}

    for cfg in DATASETS:
        name = cfg["name"]
        kind = cfg["kind"]
        rows: list[dict[str, Any]] = []
        output = raw_dir / cfg["output"]

        if kind == "station_readings":
            dedupe_station: set[tuple[str, str]] = set()
        elif kind == "forecast":
            dedupe_forecast: set[tuple[str, str]] = set()
        else:
            dedupe_weather: set[tuple[str, int, str]] = set()

        print(f"\\n[{name}] {start.isoformat()} -> {end.isoformat()}", flush=True)
        for day in iter_days(start, end):
            pages = fetch_paginated(
                url=cfg["url"],
                kind=kind,
                day=day,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                max_pages_per_day=max_pages_per_day,
                sleep_between_requests=sleep_between_requests,
            )

            if kind == "station_readings":
                day_rows = flatten_station_pages(
                    pages=pages,
                    target_day=day,
                    value_column=cfg["value_column"],
                    dedupe=dedupe_station,
                )
            elif kind == "forecast":
                day_rows = flatten_forecast_pages(pages=pages, target_day=day, dedupe=dedupe_forecast)
            else:
                day_rows = flatten_weather_record_pages(pages=pages, target_day=day, dedupe=dedupe_weather)

            rows.extend(day_rows)
            print(f"  {day.isoformat()}: pages={len(pages)} rows_added={len(day_rows)}", flush=True)

        if kind == "station_readings":
            fields = [
                "timestamp",
                "station_id",
                "device_id",
                "station_name",
                "latitude",
                "longitude",
                cfg["value_column"],
            ]
        elif kind == "forecast":
            fields = [
                "timestamp",
                "update_timestamp",
                "valid_start",
                "valid_end",
                "area",
                "forecast",
                "latitude",
                "longitude",
            ]
        else:
            fields = [
                "timestamp",
                "updated_timestamp",
                "item_type",
                "is_station_data",
                "reading_index",
                "reading_json",
                "latitude",
                "longitude",
                "has_reading",
            ]

        rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in fields[:3]))
        write_csv(output, rows, fields)
        print(f"[{name}] saved {len(rows)} rows -> {output}", flush=True)


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise ValueError("--end-date must be >= --start-date")

    api_key = load_api_key(Path(args.env_file))
    raw_dir = Path(args.raw_dir)
    output_processed = Path(args.output_processed)
    output_processed.parent.mkdir(parents=True, exist_ok=True)

    download_raw_csvs(
        raw_dir=raw_dir,
        start=start,
        end=end,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_pages_per_day=args.max_pages_per_day,
        sleep_between_requests=args.sleep_between_requests,
    )

    build_processed_table(raw_dir=raw_dir, output_path=output_processed)
    print(f"\\nProcessed table saved -> {output_processed}", flush=True)


if __name__ == "__main__":
    main()
