from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
DEFAULT_ENV_PATH = Path('.env')
DEFAULT_OUT = Path('data/raw/rainfall_history.csv')


def load_api_key(env_path: Path) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f"Missing env file: {env_path}")

    for raw in env_path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key.strip().lower() == 'api-key' and value.strip():
            return value.strip()

    raise ValueError("No 'api-key' entry found in .env")


def parse_date(value: str) -> date:
    return datetime.strptime(value, '%Y-%m-%d').date()


def daterange(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def fetch_day(day: date, api_key: str, timeout: int) -> dict[str, Any]:
    response = requests.get(
        API_URL,
        headers={'X-Api-Key': api_key},
        params={'date': day.isoformat()},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('code') not in (0, '0', None):
        raise RuntimeError(f"API error for {day}: {payload.get('errorMsg')}")
    return payload


def flatten_payload(day_payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = day_payload.get('data', {})
    stations = data.get('stations', [])
    readings = data.get('readings', [])

    station_lookup: dict[str, dict[str, Any]] = {}
    for station in stations:
        station_id = station.get('id')
        location = station.get('location') or {}
        if not station_id:
            continue
        station_lookup[station_id] = {
            'station_id': station_id,
            'device_id': station.get('deviceId'),
            'station_name': station.get('name'),
            'latitude': location.get('latitude'),
            'longitude': location.get('longitude'),
        }

    rows: list[dict[str, Any]] = []
    for reading in readings:
        timestamp = reading.get('timestamp')
        for measurement in reading.get('data', []):
            station_id = measurement.get('stationId')
            meta = station_lookup.get(
                station_id,
                {
                    'station_id': station_id,
                    'device_id': None,
                    'station_name': None,
                    'latitude': None,
                    'longitude': None,
                },
            )
            rows.append(
                {
                    'timestamp': timestamp,
                    **meta,
                    'rainfall_mm': measurement.get('value'),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'timestamp',
        'station_id',
        'device_id',
        'station_name',
        'latitude',
        'longitude',
        'rainfall_mm',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build rainfall history CSV from data.gov.sg real-time API.')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD).')
    parser.add_argument('--env-file', default=str(DEFAULT_ENV_PATH), help='Path to .env with api-key.')
    parser.add_argument('--output', default=str(DEFAULT_OUT), help='Output CSV path.')
    parser.add_argument('--timeout', type=int, default=30, help='HTTP timeout in seconds.')
    args = parser.parse_args()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise ValueError('--end-date must be >= --start-date')

    api_key = load_api_key(Path(args.env_file))

    all_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for day in daterange(start, end):
        payload = fetch_day(day, api_key=api_key, timeout=args.timeout)
        rows = flatten_payload(payload)
        for row in rows:
            key = (row.get('timestamp'), row.get('station_id'))
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(row)
        print(f"{day.isoformat()}: fetched {len(rows)} rows")

    all_rows.sort(key=lambda r: (str(r.get('timestamp')), str(r.get('station_id'))))
    out = Path(args.output)
    write_csv(all_rows, out)

    print(f"Saved {len(all_rows)} rows to {out}")


if __name__ == '__main__':
    main()
