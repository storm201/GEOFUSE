"""Sentinel-2 L2A Data Inspection and Integrity Validation Module.

Loads Sentinel-2 L2A bands (B02, B03, B04, B08) using Rasterio from paths specified
in config.yaml, validates spatial alignment and CRS, computes reflectance statistics,
and generates RGB visual composites.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image


class SpatialAlignmentError(Exception):
    """Raised when Sentinel-2 bands have mismatched CRS, shape, or resolution."""
    pass


class MissingBandError(Exception):
    """Raised when required Sentinel-2 bands are missing from the data path."""
    pass


def find_band_files(data_dir: Path, target_bands: List[str]) -> Dict[str, Path]:
    """Locate GeoTIFF files corresponding to target Sentinel-2 bands.

    Supports common Sentinel-2 naming conventions (e.g. *B02*.tif, *B02_10m.tif, etc.).

    Args:
        data_dir: Directory containing band files.
        target_bands: List of band identifiers (e.g., ['B02', 'B03', 'B04', 'B08']).

    Returns:
        Dict[str, Path]: Mapping from band identifier to resolved file path.

    Raises:
        MissingBandError: If one or more target bands cannot be found.
    """
    data_dir = Path(data_dir)
    found_bands: Dict[str, Path] = {}
    extensions = ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.jp2")

    all_files: List[Path] = []
    for ext in extensions:
        all_files.extend(data_dir.glob(ext))
        # Also check subdirectories (e.g. GRANULE/L2A_.../IMG_DATA/R10m)
        all_files.extend(data_dir.glob(f"**/{ext}"))

    # Remove duplicates while preserving Path objects
    all_files = list({f.resolve(): f for f in all_files}.values())

    for band in target_bands:
        band_matches = []
        for file_path in all_files:
            stem = file_path.stem.upper()
            # Match band name ensuring word/token boundary (e.g. '_B02_', '_B02', 'B02.')
            # Avoid partial matches like B02 matching B02_something_else inappropriately
            tokens = stem.replace("-", "_").split("_")
            if band.upper() in tokens or any(t.startswith(band.upper()) for t in tokens):
                band_matches.append(file_path)

        if band_matches:
            # Prefer 10m resolution files if multiple matches exist
            best_match = next((f for f in band_matches if "10M" in f.name.upper()), band_matches[0])
            found_bands[band] = best_match

    missing = [b for b in target_bands if b not in found_bands]
    if missing:
        available_files = [f.name for f in all_files[:10]]
        raise MissingBandError(
            f"Could not find band file(s) for {missing} in '{data_dir.resolve()}'. "
            f"Available candidate files in directory: {available_files}"
        )

    return found_bands


def check_spatial_consistency(band_datasets: Dict[str, rasterio.DatasetReader]) -> Dict[str, Any]:
    """Verify that all band rasters share the exact same CRS, shape, transform, and resolution.

    Args:
        band_datasets: Dictionary of open Rasterio dataset readers for each band.

    Returns:
        Dict[str, Any]: Metadata summary of the aligned scene.

    Raises:
        SpatialAlignmentError: If CRS, shape, or transform mismatch between bands.
    """
    first_band, reference_ds = next(iter(band_datasets.items()))
    ref_crs = reference_ds.crs
    ref_shape = (reference_ds.height, reference_ds.width)
    ref_transform = reference_ds.transform
    ref_res = reference_ds.res

    mismatches = []
    for band_name, ds in band_datasets.items():
        if ds.crs != ref_crs:
            mismatches.append(f"Band {band_name} CRS ({ds.crs}) != Reference {first_band} CRS ({ref_crs})")
        if (ds.height, ds.width) != ref_shape:
            mismatches.append(f"Band {band_name} shape ({ds.height}, {ds.width}) != Reference {first_band} shape {ref_shape}")
        if ds.res != ref_res:
            mismatches.append(f"Band {band_name} resolution {ds.res} != Reference {first_band} resolution {ref_res}")
        if ds.transform != ref_transform:
            # Check transform proximity within floating point tolerance
            t_diff = [abs(a - b) for a, b in zip(ds.transform, ref_transform)]
            if max(t_diff) > 1e-4:
                mismatches.append(f"Band {band_name} affine transform differs from Reference {first_band}")

    if mismatches:
        err_msg = "Spatial alignment check failed across bands:\n" + "\n".join(f"  - {m}" for m in mismatches)
        raise SpatialAlignmentError(err_msg)

    return {
        "crs": str(ref_crs),
        "shape": ref_shape,
        "transform": ref_transform,
        "resolution": ref_res,
        "bounds": reference_ds.bounds,
    }


def compute_band_stats(
    band_name: str,
    data: np.ndarray,
    nodata_val: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute radiometric statistics for a single band.

    Args:
        band_name: Identifier for the band (e.g. 'B02').
        data: 2D numpy array of band values.
        nodata_val: Optional nodata value specified in raster metadata.

    Returns:
        Dict[str, Any]: Statistics including min, max, mean, std, nodata percentage,
                        and surface reflectance scaling.
    """
    total_pixels = data.size

    # Detect nodata and invalid values
    invalid_mask = np.isnan(data) | np.isinf(data)
    if nodata_val is not None:
        invalid_mask = invalid_mask | (data == nodata_val)
    # Sentinel-2 L2A valid pixels are strictly positive; 0 is conventionally nodata in ESA L2A
    invalid_mask = invalid_mask | (data <= 0)

    invalid_count = int(np.count_nonzero(invalid_mask))
    valid_count = total_pixels - invalid_count
    nodata_pct = (invalid_count / total_pixels) * 100.0

    if valid_count == 0:
        return {
            "band": band_name,
            "total_pixels": total_pixels,
            "valid_pixels": 0,
            "nodata_percentage": 100.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "is_scaled_dn": False,
            "physically_plausible": False,
        }

    valid_data = data[~invalid_mask].astype(np.float64)
    b_min = float(np.min(valid_data))
    b_max = float(np.max(valid_data))
    b_mean = float(np.mean(valid_data))
    b_std = float(np.std(valid_data))

    # Determine scaling: Sentinel-2 L2A raw DN is 0-10000 (scale factor 10000 for surface reflectance 0-1)
    is_scaled_dn = b_max > 10.0
    if is_scaled_dn:
        refl_min = b_min / 10000.0
        refl_max = b_max / 10000.0
        refl_mean = b_mean / 10000.0
    else:
        refl_min, refl_max, refl_mean = b_min, b_max, b_mean

    # Physically plausible: surface reflectance is generally between 0.0 and 1.0 (with slight specular tolerances up to ~1.2)
    physically_plausible = (0.0 <= refl_min <= 1.2) and (0.0 <= refl_mean <= 1.0) and (0.0 < refl_max <= 2.0)

    return {
        "band": band_name,
        "total_pixels": total_pixels,
        "valid_pixels": valid_count,
        "nodata_percentage": round(nodata_pct, 2),
        "raw_min": round(b_min, 4),
        "raw_max": round(b_max, 4),
        "raw_mean": round(b_mean, 4),
        "raw_std": round(b_std, 4),
        "reflectance_min": round(refl_min, 4),
        "reflectance_max": round(refl_max, 4),
        "reflectance_mean": round(refl_mean, 4),
        "is_scaled_dn": is_scaled_dn,
        "physically_plausible": physically_plausible,
    }


def generate_rgb_preview(
    r_band: np.ndarray,
    g_band: np.ndarray,
    b_band: np.ndarray,
    output_path: Path,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> Path:
    """Generate and save a normalized true-color RGB composite image.

    Applies robust 2%-98% percentile stretching per channel for natural visual contrast.

    Args:
        r_band: Red band array (B04).
        g_band: Green band array (B03).
        b_band: Blue band array (B02).
        output_path: Target PNG file path.
        p_low: Lower percentile for contrast stretching.
        p_high: Upper percentile for contrast stretching.

    Returns:
        Path: Path to saved preview PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = [r_band, g_band, b_band]
    stretched_channels = []

    for ch in channels:
        ch_valid = ch[ch > 0]
        if ch_valid.size > 0:
            vmin = np.percentile(ch_valid, p_low)
            vmax = np.percentile(ch_valid, p_high)
            if vmax > vmin:
                clipped = np.clip(ch, vmin, vmax)
                norm = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(ch, dtype=np.uint8)
        else:
            norm = np.zeros_like(ch, dtype=np.uint8)
        stretched_channels.append(norm)

    rgb_array = np.stack(stretched_channels, axis=-1)
    img = Image.fromarray(rgb_array, mode="RGB")
    img.save(output_path, format="PNG")
    return output_path


def inspect_sentinel2_data(
    data_dir_or_file: Path,
    output_preview_path: Optional[Path] = None,
    target_bands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Inspect and validate Sentinel-2 dataset from a path.

    Args:
        data_dir_or_file: Path to directory containing band GeoTIFFs, or a multi-band GeoTIFF.
        output_preview_path: Destination path for RGB preview PNG.
        target_bands: List of band identifiers, defaults to ['B02', 'B03', 'B04', 'B08'].

    Returns:
        Dict[str, Any]: Complete inspection report dictionary.
    """
    if target_bands is None:
        target_bands = ["B02", "B03", "B04", "B08"]

    if not data_dir_or_file.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {data_dir_or_file.resolve()}. "
            "Please provide a valid dataset path in config.yaml."
        )

    band_datasets: Dict[str, rasterio.DatasetReader] = {}
    band_arrays: Dict[str, np.ndarray] = {}
    nodata_values: Dict[str, Optional[float]] = {}

    # Case 1: Directory containing individual band files
    if data_dir_or_file.is_dir():
        band_files = find_band_files(data_dir_or_file, target_bands)
        for band, path in band_files.items():
            ds = rasterio.open(path)
            band_datasets[band] = ds
            band_arrays[band] = ds.read(1)
            nodata_values[band] = ds.nodata

    # Case 2: A single multi-band GeoTIFF file
    elif data_dir_or_file.is_file():
        ds = rasterio.open(data_dir_or_file)
        if ds.count < len(target_bands):
            raise ValueError(
                f"Multi-band file '{data_dir_or_file.name}' contains {ds.count} bands, "
                f"but {len(target_bands)} are required: {target_bands}."
            )
        for idx, band in enumerate(target_bands, start=1):
            band_datasets[band] = ds
            band_arrays[band] = ds.read(idx)
            nodata_values[band] = ds.nodata

    # Spatial consistency verification
    spatial_meta = check_spatial_consistency(band_datasets)

    # Statistical computation per band
    stats_summary = {}
    for band in target_bands:
        stats_summary[band] = compute_band_stats(
            band_name=band,
            data=band_arrays[band],
            nodata_val=nodata_values.get(band),
        )

    # Generate RGB true-color preview (Red: B04, Green: B03, Blue: B02)
    preview_file = None
    if output_preview_path and all(b in band_arrays for b in ("B04", "B03", "B02")):
        preview_file = generate_rgb_preview(
            r_band=band_arrays["B04"],
            g_band=band_arrays["B03"],
            b_band=band_arrays["B02"],
            output_path=output_preview_path,
        )

    # Clean up dataset handles
    for ds in set(band_datasets.values()):
        ds.close()

    return {
        "spatial_metadata": spatial_meta,
        "band_statistics": stats_summary,
        "preview_path": str(preview_file) if preview_file else None,
        "all_bands_plausible": all(s["physically_plausible"] for s in stats_summary.values()),
    }
