from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBZONE_GEOJSON = ROOT / "data" / "MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"


def main() -> None:
    import argparse
    from floodlib.feature_pipeline import build_processed_table

    parser = argparse.ArgumentParser(
        description="Build the canonical processed subzone feature table from raw CSV exports."
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_processed_table(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        subzone_geojson_path=SUBZONE_GEOJSON,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
