from __future__ import annotations

import unittest

from floodlib.risk_model import MISSING_FEATURE_COLOR, feature_fill_color, predict_flood_risk


class RiskModelTest(unittest.TestCase):
    def test_feature_fill_color_uses_neutral_color_for_missing_values(self) -> None:
        self.assertEqual(feature_fill_color(None, 0.0, 1.0), MISSING_FEATURE_COLOR)
        self.assertNotEqual(feature_fill_color(0.5, 0.0, 1.0), MISSING_FEATURE_COLOR)

    def test_predict_flood_risk_increases_with_stronger_signal(self) -> None:
        low_signal = predict_flood_risk(
            row={
                "rainfall_mm_15min": 0.0,
                "rainfall_mm_60min": 0.0,
                "humidity_pct": 60.0,
                "lightning_count_5min": 0.0,
                "lightning_count_sg_5min": 0.0,
                "forecast_rainy_fraction_2h": 0.0,
                "flood_alert_count_sg_5min": 0.0,
            },
            subzone_upper="ALPHA",
            timestamp="2026-03-05T00:00:00+08:00",
            w_r60=1.0,
            w_r15=0.6,
            w_humidity=0.3,
            w_lightning=0.7,
            w_forecast=0.5,
            w_flood_now=1.0,
            synthetic_factor=0.0,
        )
        high_signal = predict_flood_risk(
            row={
                "rainfall_mm_15min": 20.0,
                "rainfall_mm_60min": 70.0,
                "humidity_pct": 95.0,
                "lightning_count_5min": 3.0,
                "lightning_count_sg_5min": 20.0,
                "forecast_rainy_fraction_2h": 1.0,
                "flood_alert_count_sg_5min": 2.0,
            },
            subzone_upper="ALPHA",
            timestamp="2026-03-05T00:00:00+08:00",
            w_r60=1.0,
            w_r15=0.6,
            w_humidity=0.3,
            w_lightning=0.7,
            w_forecast=0.5,
            w_flood_now=1.0,
            synthetic_factor=0.0,
        )

        self.assertGreater(high_signal, low_signal)
        self.assertGreaterEqual(low_signal, 0.0)
        self.assertLessEqual(high_signal, 1.0)


if __name__ == "__main__":
    unittest.main()
