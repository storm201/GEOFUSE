# Dataset Setup & Provenance Guide — GeoFUSE SentinelGuard

This document details the source, licensing, retrieval procedure, and format of the Sentinel-2 L2A satellite imagery used in this project.

---

## 1. Dataset Provenance & Metadata

| Attribute | Specification |
| :--- | :--- |
| **Data Provider** | European Space Agency (ESA) Copernicus Programme via **AWS Open Data** |
| **Catalog / Archive** | Element 84 Earth Search (`sentinel-2-c1-l2a` STAC collection) |
| **Scene Identifier** | `S2A_T43PGQ_20240227T052054_L2A` |
| **Acquisition Date** | 2024-02-27 05:20:54 UTC |
| **MGRS Tile** | `43PGQ` (Karnataka / Bangalore Region, India) |
| **Processing Level** | Level-2A (Bottom-of-Atmosphere / Surface Reflectance) |
| **Coordinate System** | `EPSG:32643` (WGS 84 / UTM zone 43N) |
| **Spatial Resolution**| 10.0 meters per pixel (for 10m bands) |
| **Cloud Cover** | < 1% over ROI |

---

## 2. Bands Obtained

GeoFUSE utilizes the standard 10m multi-spectral bands:

| Band Name | Description | Central Wavelength | Spatial Resolution | File Name in `data/raw/` |
| :--- | :--- | :--- | :--- | :--- |
| **B02** | Blue | 490 nm | 10 m | `S2A_T43PGQ_20240227T052054_L2A_B02_10m.tif` |
| **B03** | Green | 560 nm | 10 m | `S2A_T43PGQ_20240227T052054_L2A_B03_10m.tif` |
| **B04** | Red | 665 nm | 10 m | `S2A_T43PGQ_20240227T052054_L2A_B04_10m.tif` |
| **B08** | Near-Infrared (NIR) | 842 nm | 10 m | `S2A_T43PGQ_20240227T052054_L2A_B08_10m.tif` |

---

## 3. Spatial Crop & Local Directory

To respect hardware constraints (RTX 4060 Laptop GPU) and project guidelines without storing unneeded multi-gigabyte full tiles, a representative **512 × 512 pixel window** (\(\approx 5.12\text{ km} \times 5.12\text{ km}\)) was extracted:
- **Pixel Offset**: Column `5000`, Row `5000`
- **Total Download Size**: \(\approx 1.56\text{ MB}\) across all 4 bands
- **Local Directory**: `data/raw/` (configured via `paths.raw_data_dir` in `config.yaml`)

All spatial georeferencing metadata (CRS `EPSG:32643`, geotransform affine matrix, nodata tag) is preserved.

---

## 4. Automated Retrieval & Inspection

To reproduce or re-download the sample scene, run:
```bash
python scripts/download_sample_data.py
```

To run spatial integrity checks, compute radiometric statistics, and render the RGB preview:
```bash
python scripts/inspect_sentinel2.py
```

---

## 5. Additional Datasets (`data/additional_datasets/`)

To support multi-landscape generalizability and seasonal drift testing, 3 additional 4-band scenes (512x512 pixels each) are available in `data/additional_datasets/`:

| Directory | Landscape Description | Acquisition Date | Scene ID | Crop Window |
| :--- | :--- | :---: | :--- | :--- |
| `urban_core/` | High-Density Built-Up Urban Core & Roads | 2024-02-27 | `S2A_T43PGQ_20240227T052054_L2A` | `col=3500, row=3500` |
| `agriculture/` | Intensive Agricultural Farmland & Canals | 2024-02-27 | `S2A_T43PGQ_20240227T052054_L2A` | `col=7500, row=4500` |
| `temporal_april2024/` | Dry Season Shift (Same Coordinates, +2 Months) | 2024-04-27 | `S2A_T43PGQ_20240427T051439_L2A` | `col=5000, row=5000` |

To re-fetch all additional datasets at any time:
```bash
python scripts/download_additional_datasets.py
```

---

## 6. Access Limitations & Licensing

- **Authentication / API Keys**: None required. Data is streamed directly via HTTPS from the publicly accessible AWS Open Data Registry.
- **License**: Copernicus Sentinel Data is governed by the **Copernicus Open Access Policy**:
  - Free, full, and open access for research, educational, and commercial uses.
  - Attribution: *"Contains modified Copernicus Sentinel data [2024], processed by ESA / Element 84."*
