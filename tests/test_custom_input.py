"""Unit tests for Custom User Input Degradation and Super-Resolution Pipeline."""

from pathlib import Path
import pytest
import numpy as np

from scripts.run_custom_input import load_user_image, run_custom_pipeline
from src.utils.config import get_project_root


def test_load_user_image_geotiff():
    """Verify load_user_image properly decodes a multi-band GeoTIFF."""
    root = get_project_root()
    tif_path = root / "data" / "raw" / "S2A_T43PGQ_20240227T052054_L2A_B04_10m.tif"
    assert tif_path.exists()

    arr, meta = load_user_image(tif_path, target_size=128)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (128, 128, 4)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() <= arr.max() <= 1.5


def test_load_user_image_rgb_png():
    """Verify load_user_image properly decodes a 3-channel RGB PNG and synthesizes NIR."""
    root = get_project_root()
    png_path = root / "outputs" / "presentation" / "01_side_by_side_super_resolution.png"
    assert png_path.exists()

    arr, meta = load_user_image(png_path, target_size=128)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (128, 128, 4)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() <= arr.max() <= 1.0


def test_run_custom_pipeline_end_to_end(tmp_path):
    """Verify that run_custom_pipeline executes cleanly in benchmark mode."""
    root = get_project_root()
    tif_path = root / "data" / "raw" / "S2A_T43PGQ_20240227T052054_L2A_B04_10m.tif"
    assert tif_path.exists()

    results = run_custom_pipeline(
        input_image_path=tif_path,
        output_dir=tmp_path,
        mode="benchmark",
        save_plot=True,
        save_receipt=True,
    )

    assert isinstance(results, dict)
    assert "trust_score_pct" in results
    assert 0.0 <= results["trust_score_pct"] <= 100.0
    assert "status" in results
    assert "sr_psnr" in results
    assert "bicubic_psnr" in results
    assert "hr_tile" in results
    assert "sr_tile" in results
    assert "lr_tile" in results
    assert results["lr_tile"].shape in [(64, 64, 4), (51, 51, 4)]
    assert results["sr_tile"].shape == (128, 128, 4)
    assert "fusion_result" in results
    assert "receipt" in results
    assert results["synthetic_degradation_used"] is True

    # Check saved outputs
    plot_file = Path(results["plot_path"])
    receipt_file = Path(results["receipt_path"])
    assert plot_file.exists()
    assert receipt_file.exists()


def test_load_user_image_4_band_list():
    """Verify load_user_image properly stacks 4 separate Sentinel-2 band files."""
    root = get_project_root()
    urban_dir = root / "data" / "additional_datasets" / "urban_core"
    band_files = list(urban_dir.glob("*.tif"))
    assert len(band_files) == 4

    arr, meta = load_user_image(band_files, target_size=128)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (128, 128, 4)
    assert arr.dtype == np.float32
    assert meta.get("is_multi_band_stack") is True
    assert set(meta.get("matched_bands", {}).keys()) == {"B02", "B03", "B04", "B08"}


def test_load_user_image_4_band_directory():
    """Verify load_user_image properly processes a directory containing 4 band GeoTIFFs."""
    root = get_project_root()
    agri_dir = root / "data" / "additional_datasets" / "agriculture"
    assert agri_dir.is_dir()

    arr, meta = load_user_image(agri_dir, target_size=128)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (128, 128, 4)
    assert meta.get("is_multi_band_stack") is True
    assert set(meta.get("matched_bands", {}).keys()) == {"B02", "B03", "B04", "B08"}


def test_run_custom_pipeline_4_band_stack(tmp_path):
    """Verify end-to-end execution in benchmark mode with 4 uploaded band files."""
    root = get_project_root()
    urban_dir = root / "data" / "additional_datasets" / "urban_core"
    band_files = list(urban_dir.glob("*.tif"))
    assert len(band_files) == 4

    results = run_custom_pipeline(
        input_image_path=band_files,
        output_dir=tmp_path,
        mode="benchmark",
        save_plot=True,
        save_receipt=True,
    )

    assert isinstance(results, dict)
    assert "trust_score_pct" in results
    assert results["trust_score_pct"] >= 80.0
    assert results["is_trusted"] is True
    assert results["hr_tile"].shape == (128, 128, 4)
    assert results["sr_tile"].shape == (128, 128, 4)
    assert results["input_meta"].get("is_multi_band_stack") is True
    assert Path(results["receipt_path"]).exists()


def test_run_custom_pipeline_real_mode(tmp_path):
    """Verify end-to-end execution in real mode with direct 2.5x learned SR and zero synthetic degradation."""
    root = get_project_root()
    urban_dir = root / "data" / "additional_datasets" / "urban_core"

    results = run_custom_pipeline(
        input_image_path=urban_dir,
        output_dir=tmp_path,
        mode="real",
        save_plot=True,
        save_receipt=True,
    )

    assert isinstance(results, dict)
    assert results["mode"] == "real"
    assert results["synthetic_degradation_used"] is False
    assert results["sr_psnr"] == "N/A (Real Scene)"
    assert results["bicubic_psnr"] == "N/A (Real Scene)"
    assert "sr_4m" in results
    # 512x512 * 2.5 = 1280x1280
    assert results["sr_4m"].shape == (1280, 1280, 4)
    assert results["input_10m"].shape == (512, 512, 4)
    assert "trust_score_pct" in results
    assert 0.0 <= results["trust_score_pct"] <= 100.0

    # Verify receipt provenance
    receipt = results["receipt"]
    assert receipt["provenance"]["synthetic_degradation_used_for_inference"] is False
    assert receipt["provenance"]["input_type"] == "real_sentinel2"
    assert receipt["resolution"]["input_gsd_m"] == 10.0
    assert receipt["resolution"]["output_gsd_m"] == 4.0
    assert receipt["resolution"]["scale_factor"] == 2.5

    # Check exported GeoTIFF exists and has correct 4m dimensions
    exported = results["exported_paths"]
    assert "sr_4m" in exported
    sr_tif = Path(exported["sr_4m"])
    assert sr_tif.exists()


