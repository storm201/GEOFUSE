"""Download and package diverse real Sentinel-2 Level-2A test scenes into 'new data/'.

Streams windowed crops from AWS Open Data (Element 84 Earth Search archive)
and packages them into:
1. Single 4-band GeoTIFFs (B02, B03, B04, B08) for 1-click drag-and-drop testing.
2. Separate 4-band folders for multi-file upload testing.
"""

from pathlib import Path
import sys
import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "new data"

SCENE_ID = "S2A_T43PGQ_20240227T052054_L2A"
BASE_URL = f"https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/43/P/GQ/2024/2/{SCENE_ID}"

TEST_SCENES = {
    "test_scene_01_water_reservoir": {
        "title": "Water Reservoir & Shoreline Wetland",
        "description": "Thippagondanahalli / Arkavathi water reservoir basin with water-land boundaries.",
        "col": 1800,
        "row": 1800,
        "width": 512,
        "height": 512,
    },
    "test_scene_02_airport_industrial": {
        "title": "International Airport & Industrial Logistics Hub",
        "description": "Kempegowda airport runway complexes, terminals, logistics warehouses, and road interchanges.",
        "col": 5500,
        "row": 1000,
        "width": 512,
        "height": 512,
    },
    "test_scene_03_forested_hills": {
        "title": "Forested Mountain Ridges & Granite Outcrops",
        "description": "Dense tropical dry deciduous canopy, rocky hill slopes, and steep elevation shadows.",
        "col": 8500,
        "row": 8000,
        "width": 512,
        "height": 512,
    },
}

BANDS = ["B02", "B03", "B04", "B08"]


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    print("===============================================================================")
    print("  GeoFUSE SentinelGuard — Fetching New Test Imagery from AWS Open Data")
    print("===============================================================================")
    print(f"Target Directory: {OUTPUT_BASE.resolve()}\n")

    summary_records = []

    for scene_key, info in TEST_SCENES.items():
        print(f"\n[*] Processing [{scene_key}]: {info['title']}")
        print(f"    Crop: col={info['col']}, row={info['row']}, size={info['width']}x{info['height']}")

        # Prepare separate bands folder
        separate_dir = OUTPUT_BASE / f"{scene_key}_separate_bands"
        separate_dir.mkdir(parents=True, exist_ok=True)

        win = Window(info["col"], info["row"], info["width"], info["height"])
        band_arrays = []
        ref_meta = None

        for band in BANDS:
            band_url = f"{BASE_URL}/{band}.tif"
            sep_file = separate_dir / f"{SCENE_ID}_{band}_10m.tif"

            print(f"    -> Streaming {band} from AWS Open Data COG...")
            with rasterio.open(band_url) as src:
                arr = src.read(1, window=win)
                band_arrays.append(arr)

                if ref_meta is None:
                    ref_meta = src.meta.copy()
                    ref_meta.update({
                        "driver": "GTiff",
                        "height": info["height"],
                        "width": info["width"],
                        "transform": window_transform(win, src.transform),
                        "compress": "deflate",
                    })

                # Write separate band file
                single_meta = ref_meta.copy()
                single_meta.update({"count": 1, "dtype": arr.dtype})
                with rasterio.open(sep_file, "w", **single_meta) as dst:
                    dst.write(arr, 1)

        # Write combined 4-band GeoTIFF
        combined_file = OUTPUT_BASE / f"{scene_key}_4band.tif"
        combined_meta = ref_meta.copy()
        combined_meta.update({
            "count": 4,
            "dtype": band_arrays[0].dtype,
        })

        with rasterio.open(combined_file, "w", **combined_meta) as dst:
            for b_idx, arr in enumerate(band_arrays, start=1):
                dst.write(arr, b_idx)
                dst.set_band_description(b_idx, BANDS[b_idx - 1])

        c_size_mb = combined_file.stat().st_size / (1024 * 1024)
        print(f"    [OK] Created 4-band GeoTIFF: {combined_file.name} ({c_size_mb:.2f} MB)")
        print(f"    [OK] Created 4 separate band files in: {separate_dir.name}/")

        summary_records.append({
            "key": scene_key,
            "title": info["title"],
            "description": info["description"],
            "combined": combined_file.name,
            "separate_folder": separate_dir.name,
            "size_mb": c_size_mb,
        })

    # Write a clean README in "new data"
    readme_path = OUTPUT_BASE / "README.md"
    readme_content = f"""# New Test Datasets for GeoFUSE SentinelGuard

Real Sentinel-2 Level-2A multi-spectral test scenes sourced directly from the **AWS Open Data Program (Element 84 Earth Search Archive)**.

- **Satellite**: ESA Sentinel-2A
- **Acquisition**: Level-2A Bottom-Of-Atmosphere (BOA) Reflectance
- **GSD**: 10.0m native spatial resolution
- **Dimensions**: 512 × 512 pixels (~5.12 km × 5.12 km ground footprint)
- **Spectral Bands**:
  - Band 1: **B02** (Blue — 490 nm)
  - Band 2: **B03** (Green — 560 nm)
  - Band 3: **B04** (Red — 665 nm)
  - Band 4: **B08** (Near-Infrared / NIR — 842 nm)

---

## Included Test Datasets

### 1. Water Reservoir & Shoreline Wetland
- **File**: `test_scene_01_water_reservoir_4band.tif`
- **Separate Bands Folder**: `test_scene_01_water_reservoir_separate_bands/`
- **Characteristics**: Open water body with clear land-water boundaries, wetlands, and aquatic vegetation.
- **What to Observe**:
  - Strong NIR (B08) water absorption (appears dark in CIR false color).
  - High NDVI along irrigated shoreline vs. near-zero NDVI over open water.
  - Razor-sharp shoreline demarcation at 4m super-resolution.

### 2. International Airport & Industrial Logistics Hub
- **File**: `test_scene_02_airport_industrial_4band.tif`
- **Separate Bands Folder**: `test_scene_02_airport_industrial_separate_bands/`
- **Characteristics**: Runway strips, airport aprons, aircraft hangars, warehouse roofs, and highway grids.
- **What to Observe**:
  - Linear road and runway edge synthesis with no stair-stepping.
  - Building Footprint Extraction mask highlighting rectangular roof geometries.
  - Epistemic uncertainty maps showing low variance along tarmac and high confidence.

### 3. Forested Mountain Ridges & Granite Outcrops
- **File**: `test_scene_03_forested_hills_4band.tif`
- **Separate Bands Folder**: `test_scene_03_forested_hills_separate_bands/`
- **Characteristics**: Mountainous topography, rocky hill summits, dense forest canopy, and shadow gradients.
- **What to Observe**:
  - High NDVI vegetation response (> 0.70) across dense canopy.
  - Stability evaluation on terrain shadows.
  - Rich natural texture recovery without hallucinated artifacts.

---

## How to Test in GeoFUSE

### Option A: 1-Click Single File Upload (Recommended)
1. Start GeoFUSE via `run me.bat` or `python scripts/launch_server.py`.
2. In the web interface, click the **"User Input"** tab in the sidebar.
3. Drag and drop any `*_4band.tif` file directly into the dropzone.
4. The system validates the raster, displays the macro overview, and partitions it into an interactive **5×5 spatial tile grid** (25 tiles).
5. Select any tile (e.g. Tile #07 or #12) and click **"Run Super-Resolution Inference"**!

### Option B: Multi-File Separate Band Upload
1. In the **"User Input"** tab, open any of the `*_separate_bands` folders.
2. Select all 4 files (`*_B02_10m.tif`, `*_B03_10m.tif`, `*_B04_10m.tif`, `*_B08_10m.tif`) and drag them together into the dropzone.
3. GeoFUSE automatically binds the bands into a multi-spectral stack and builds the tile grid.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"\n[OK] Generated documentation: {readme_path.name}")
    print("\n===============================================================================")
    print("  All test datasets successfully downloaded and packaged in 'new data/'")
    print("===============================================================================\n")


if __name__ == "__main__":
    main()
