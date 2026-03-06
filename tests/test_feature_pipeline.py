from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from floodlib.feature_pipeline import build_processed_table


class FeaturePipelineTest(unittest.TestCase):
    def test_build_processed_table_localizes_forecast_and_lightning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            output_path = root / "processed" / "features.csv"
            subzones_path = root / "subzones.geojson"

            subzones_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "SUBZONE_N": "ALPHA",
                                    "PLN_AREA_N": "AREA_A",
                                    "REGION_N": "REGION_A",
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [103.0, 1.0],
                                            [103.1, 1.0],
                                            [103.1, 1.1],
                                            [103.0, 1.1],
                                            [103.0, 1.0],
                                        ]
                                    ],
                                },
                            },
                            {
                                "type": "Feature",
                                "properties": {
                                    "SUBZONE_N": "BETA",
                                    "PLN_AREA_N": "AREA_B",
                                    "REGION_N": "REGION_B",
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [103.2, 1.0],
                                            [103.3, 1.0],
                                            [103.3, 1.1],
                                            [103.2, 1.1],
                                            [103.2, 1.0],
                                        ]
                                    ],
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (raw_dir / "forecast_2h.csv").open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "timestamp",
                        "update_timestamp",
                        "valid_start",
                        "valid_end",
                        "area",
                        "forecast",
                        "latitude",
                        "longitude",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "timestamp": "2026-03-05T00:00:00+08:00",
                            "update_timestamp": "",
                            "valid_start": "",
                            "valid_end": "",
                            "area": "North",
                            "forecast": "Thundery Showers",
                            "latitude": "1.05",
                            "longitude": "103.05",
                        },
                        {
                            "timestamp": "2026-03-05T00:00:00+08:00",
                            "update_timestamp": "",
                            "valid_start": "",
                            "valid_end": "",
                            "area": "East",
                            "forecast": "Fair",
                            "latitude": "1.05",
                            "longitude": "103.25",
                        },
                    ]
                )

            with (raw_dir / "lightning.csv").open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "timestamp",
                        "updated_timestamp",
                        "item_type",
                        "is_station_data",
                        "reading_index",
                        "reading_json",
                        "latitude",
                        "longitude",
                        "has_reading",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "2026-03-05T00:00:00+08:00",
                        "updated_timestamp": "",
                        "item_type": "lightning",
                        "is_station_data": "",
                        "reading_index": "0",
                        "reading_json": "{}",
                        "latitude": "1.05",
                        "longitude": "103.05",
                        "has_reading": "1",
                    }
                )

            build_processed_table(
                raw_dir=raw_dir,
                output_path=output_path,
                subzone_geojson_path=subzones_path,
            )

            with output_path.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

            by_subzone = {row["subzone"]: row for row in rows}
            self.assertEqual(set(by_subzone), {"ALPHA", "BETA"})

            self.assertEqual(by_subzone["ALPHA"]["lightning_count_5min"], "1")
            self.assertEqual(by_subzone["BETA"]["lightning_count_5min"], "0")
            self.assertEqual(by_subzone["ALPHA"]["forecast_rainy_fraction_2h"], "1.0")
            self.assertEqual(by_subzone["ALPHA"]["forecast_thundery_fraction_2h"], "1.0")
            self.assertEqual(by_subzone["BETA"]["forecast_rainy_fraction_2h"], "0.0")
            self.assertEqual(by_subzone["BETA"]["forecast_thundery_fraction_2h"], "0.0")


if __name__ == "__main__":
    unittest.main()
