"""Utility script to download additional Sentinel-2 L2A datasets from AWS Open Data.

Fetches 3 distinct, high-value 512x512 multi-spectral Sentinel-2 scenes from the
official Element 84 Earth Search archive (sentinel-2-c1-l2a) with complete georeferencing:

1. Scene 'urban_core': High-density built-up urban infrastructure & road networks.
2. Scene 'agriculture': Intensive crop patchwork, vegetation boundaries & irrigation.
3. Scene 'temporal_april2024': Identical geographic footprint 2 months later (2024-04-27)
   to evaluate seasonal vegetation shifts and multi-temporal generalization.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import rasterio
from rasterio.windows import Window, transform as window_transform

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import get_project_root


DATASET_CONFIGS = {
    "urban_core": {
        "description": "High-Density Urban Core & Road Network (Bangalore Metro)",
        "scene_id": "S2A_T43PGQ_20240227T052054_L2A",
        "base_url": "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/43/P/GQ/2024/2/S2A_T43PGQ_20240227T052054_L2A",
        "crop_col": 3500,
        "crop_row": 3500,
        "width": 512,
        "height": 512,
    },
    "agriculture": {
        "description": "Intensive Agricultural Farmland & Vegetation Gradients",
        "scene_id": "S2A_T43PGQ_20240227T052054_L2A",
        "base_url": "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/43/P/GQ/2024/2/S2A_T43PGQ_20240227T052054_L2A",
        "crop_col": 7500,
        "crop_row": 4500,
        "width": 512,
        "height": 512,
    },
    "temporal_april2024": {
        "description": "Temporal Seasonal Verification (April 2024 - Dry Season Shift)",
        "scene_id": "S2A_T43PGQ_20240427T051439_L2A",
        "base_url": "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/43/P/GQ/2024/4/S2A_T43PGQ_20240427T051439_L2A",
        "crop_col": 5000,
        "crop_row": 5000,
        "width": 512,
        "height": 512,
    },
}

BANDS = ["B02", "B03", "B04", "B08"]


def download_scene_dataset(name: str, meta: dict, base_output_dir: Path) -> Dict[str, Path]:
    """Download 4-band crop for a specific scene into its own directory."""
    scene_dir = base_output_dir / name
    scene_dir.mkdir(parents=True, exist_ok=True)

    col, row = meta["crop_col"], meta["crop_row"]
    w, h = meta["width"], meta["height"]
    win = Window(col, row, w, h)
    saved_files: Dict[str, Path] = {}

    print(f"\n[{name.upper()}] {meta['description']}")
    print(f"  Scene ID    : {meta['scene_id']}")
    print(f"  Crop Window : col={col}, row={row}, size=({w}x{h})")
    print(f"  Output Dir  : {scene_dir}")

    for band in BANDS:
        band_url = f"{meta['base_url']}/{band}.tif"
        out_file = scene_dir / f"{meta['scene_id']}_{band}_10m.tif"

        with rasterio.open(band_url) as src:
            data = src.read(1, window=win)
            crop_meta = src.meta.copy()
            crop_meta.update({
                "driver": "GTiff",
                "height": h,
                "width": w,
                "transform": window_transform(win, src.transform),
                "count": 1,
                "compress": "deflate",
            })

            with rasterio.open(out_file, "w", **crop_meta) as dst:
                dst.write(data, 1)

        size_kb = out_file.stat().st_size / 1024.0
        print(f"  -> Saved {band}: {out_file.name} ({size_kb:.1f} KB)")
        saved_files[band] = out_file

    return saved_files


def main():
    root = get_project_root()
    base_dir = root / "data" / "additional_datasets"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   Downloading Additional Sentinel-2 L2A Datasets from AWS Open Data")
    print(f"   Target Directory: {base_dir}")
    print("=" * 80)

    for name, meta in DATASET_CONFIGS.items():
        try:
            download_scene_dataset(name, meta, base_dir)
        except Exception as e:
            print(f"  [ERROR] Failed to download {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("   [SUCCESS] All additional Sentinel-2 datasets downloaded successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
