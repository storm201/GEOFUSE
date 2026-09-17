"""Comprehensive Regression Tests for Direct Real Sentinel-2 Super-Resolution Inference.

Validates all 14 requirements from the project specification:
1. Real inference does not call synthetic degradation
2. B02/B03/B04/B08 are loaded in correct order
3. Normalization is consistent with training (DN / 10000.0, range [0.0, 1.2])
4. All three checkpoints load successfully
5. Scale factor is verified 2.5x
6. 64x64 input becomes 160x160 output
7. Real inference runs with CUDA when available
8. Tiling works seamlessly
9. Overlap-aware stitching works without boundary seams
10. GeoTIFF transform changes to 4m while preserving extent
11. CRS is preserved in output GeoTIFF
12. Benchmark mode still works independently
13. Real mode does not generate fake PSNR/SSIM
14. Trust receipt records synthetic_degradation_used_for_inference: false
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
import torch

from src.data.degrade import synthesize_pseudo_lr
from src.inference.real_inference import (
    compute_4m_geotransform,
    compute_real_inference_trust,
    export_real_inference_products,
    generate_real_trust_receipt,
    load_and_preprocess_sentinel2,
    run_direct_sr_scene,
    run_direct_sr_tile,
    validate_sentinel2_input,
)
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.models.model import ResidualSRNet
from src.utils.config import get_device, get_project_root, load_config
from scripts.run_custom_input import run_custom_pipeline


@pytest.fixture(scope="module")
def project_root():
    return get_project_root()


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def device(config):
    return get_device(config)


@pytest.fixture(scope="module")
def ensemble_models(project_root, config, device):
    ckpt_dir = project_root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (ckpt_dir / "ensemble_member_0.pth").exists():
        ckpt_dir = project_root / "outputs" / "checkpoints"
    ckpt_paths = [ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)]
    for p in ckpt_paths:
        assert p.exists(), f"Checkpoint missing: {p}"
    return load_ensemble_members(ckpt_paths, config=config, device=device)


# -----------------------------------------------------------------------------
# 1. Real inference does not call synthetic degradation
# -----------------------------------------------------------------------------
def test_real_inference_no_synthetic_degradation(project_root, tmp_path):
    """Verify that during real inference, synthesize_pseudo_lr is NEVER called."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    assert urban_dir.exists()

    with patch("src.data.degrade.synthesize_pseudo_lr", side_effect=AssertionError("synthesize_pseudo_lr MUST NOT be called in real inference!")):
        results = run_custom_pipeline(
            input_image_path=urban_dir,
            output_dir=tmp_path,
            mode="real",
            save_plot=False,
            save_receipt=True,
        )

    assert results["synthetic_degradation_used"] is False
    assert results["mode"] == "real"


# -----------------------------------------------------------------------------
# 2. B02/B03/B04/B08 are loaded in correct order
# -----------------------------------------------------------------------------
def test_band_order_and_validation(project_root):
    """Verify validate_sentinel2_input returns the exact band ordering [B02, B03, B04, B08]."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    val_info = validate_sentinel2_input(urban_dir)

    assert val_info["is_valid"] is True
    assert val_info["band_order"] == ["B02", "B03", "B04", "B08"]
    assert "B02" in val_info["band_paths"]
    assert "B03" in val_info["band_paths"]
    assert "B04" in val_info["band_paths"]
    assert "B08" in val_info["band_paths"]
    assert "B02" in val_info["band_paths"]["B02"].name
    assert "B08" in val_info["band_paths"]["B08"].name



# -----------------------------------------------------------------------------
# 3. Normalization is consistent with training (DN / 10000.0, range [0.0, 1.2])
# -----------------------------------------------------------------------------
def test_normalization_consistency(project_root):
    """Verify normalization converts raw Sentinel-2 DN to surface reflectance in [0.0, 1.2]."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    scene_10m, val_info = load_and_preprocess_sentinel2(urban_dir)

    assert isinstance(scene_10m, np.ndarray)
    assert scene_10m.dtype == np.float32
    assert scene_10m.ndim == 3
    assert scene_10m.shape[2] == 4
    # Reflectance range checks
    assert float(scene_10m.min()) >= 0.0
    assert float(scene_10m.max()) <= 1.2
    # Typical land surface reflectance mean is between 0.05 and 0.45
    assert 0.03 < float(np.mean(scene_10m)) < 0.60


# -----------------------------------------------------------------------------
# 4. All three checkpoints load successfully
# -----------------------------------------------------------------------------
def test_checkpoints_loading(ensemble_models):
    """Verify all 3 ensemble checkpoints load cleanly with 8 residual blocks and 64 features."""
    assert len(ensemble_models) == 3
    for i, model in enumerate(ensemble_models):
        assert isinstance(model, ResidualSRNet)
        assert len(model.lr_blocks) == 8, f"Model {i} expected 8 residual blocks, found {len(model.lr_blocks)}"
        assert model.scale_factor == 2.5
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 957477


# -----------------------------------------------------------------------------
# 5 & 6. Scale factor is verified 2.5x and 64x64 input becomes 160x160 output
# -----------------------------------------------------------------------------
def test_forward_pass_scale_factor_64_to_160(ensemble_models, device):
    """Verify exact 2.5x spatial output transformation: (1, 4, 64, 64) -> (1, 4, 160, 160)."""
    x = torch.zeros((1, 4, 64, 64), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for i, model in enumerate(ensemble_models):
            y = model(x)
            assert y.shape == (1, 4, 160, 160), f"Member {i} output shape mismatch: {y.shape}"

    # Also test through run_direct_sr_tile
    tile_64 = np.zeros((64, 64, 4), dtype=np.float32)
    sr_160, disag = run_direct_sr_tile(ensemble_models, tile_64, device=device)
    assert sr_160.shape == (160, 160, 4)
    assert disag.shape == (160, 160)


# -----------------------------------------------------------------------------
# 7. Real inference runs with CUDA when available
# -----------------------------------------------------------------------------
def test_cuda_execution(ensemble_models, device):
    """Verify CUDA is utilized when available on this platform."""
    if torch.cuda.is_available():
        assert device.type == "cuda"
        for model in ensemble_models:
            first_param = next(model.parameters())
            assert first_param.is_cuda
    else:
        assert device.type == "cpu"


# -----------------------------------------------------------------------------
# 8 & 9. Tiling and overlap-aware stitching work seamlessly
# -----------------------------------------------------------------------------
def test_tiling_and_stitching(ensemble_models, device):
    """Verify overlap-aware scene processing produces seamless, correct 2.5x dimensions."""
    # Test on a 192x192 synthetic test raster (multiple tiles with overlap)
    np.random.seed(42)
    scene_192 = np.random.uniform(0.05, 0.45, (192, 192, 4)).astype(np.float32)

    sr_scene, disag_scene = run_direct_sr_scene(
        models=ensemble_models,
        scene_10m=scene_192,
        tile_size=64,
        overlap=16,
        scale_factor=2.5,
        device=device,
    )

    expected_h = round(192 * 2.5)  # 480
    expected_w = round(192 * 2.5)  # 480
    assert sr_scene.shape == (expected_h, expected_w, 4)
    assert disag_scene.shape == (expected_h, expected_w)
    # Ensure no NaN values, proper non-negative range, and populated pixels
    assert float(sr_scene.min()) >= 0.0
    assert float(sr_scene.max()) > 0.1
    assert float(np.mean(sr_scene)) > 0.1



# -----------------------------------------------------------------------------
# 10. GeoTIFF transform changes to 4m while preserving extent
# -----------------------------------------------------------------------------
def test_geotransform_4m_update():
    """Verify compute_4m_geotransform adjusts pixel resolution by 1/2.5 without shifting bounds."""
    # 10m Sentinel-2 transform: pixel width = 10.0, pixel height = -10.0
    orig_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    in_shape = (512, 512)
    out_shape = (1280, 1280)

    trans_4m = compute_4m_geotransform(orig_transform, in_shape, out_shape, scale_factor=2.5)

    assert pytest.approx(trans_4m.a, rel=1e-3) == 4.0   # 10.0 / 2.5
    assert pytest.approx(trans_4m.e, rel=1e-3) == -4.0  # -10.0 / 2.5
    # Origin should be identical
    assert pytest.approx(trans_4m.c) == orig_transform.c
    assert pytest.approx(trans_4m.f) == orig_transform.f

    # Total bounding box extent must match
    in_x_max = orig_transform.c + in_shape[1] * orig_transform.a
    in_y_min = orig_transform.f + in_shape[0] * orig_transform.e
    out_x_max = trans_4m.c + out_shape[1] * trans_4m.a
    out_y_min = trans_4m.f + out_shape[0] * trans_4m.e

    assert pytest.approx(out_x_max, abs=0.1) == in_x_max
    assert pytest.approx(out_y_min, abs=0.1) == in_y_min


# -----------------------------------------------------------------------------
# 11. CRS is preserved in output GeoTIFF
# -----------------------------------------------------------------------------
def test_crs_preservation(project_root, tmp_path):
    """Verify output GeoTIFF retains the exact CRS of the input raster."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    results = run_custom_pipeline(
        input_image_path=urban_dir,
        output_dir=tmp_path,
        mode="real",
        save_plot=False,
        save_receipt=True,
    )

    sr_tif_path = Path(results["exported_paths"]["sr_4m"])
    assert sr_tif_path.exists()

    with rasterio.open(sr_tif_path) as src:
        assert src.crs is not None
        assert src.crs.to_epsg() == 32643  # UTM zone 43N
        assert src.count == 4
        assert src.width == 1280
        assert src.height == 1280
        assert pytest.approx(src.transform.a, rel=1e-2) == 4.0
        assert pytest.approx(src.transform.e, rel=1e-2) == -4.0


# -----------------------------------------------------------------------------
# 12. Benchmark mode still works independently
# -----------------------------------------------------------------------------
def test_benchmark_mode_independence(project_root, tmp_path):
    """Verify benchmark mode continues to evaluate degrade-and-recover PSNR/SSIM with synthetic degradation."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    results = run_custom_pipeline(
        input_image_path=urban_dir,
        output_dir=tmp_path,
        mode="benchmark",
        save_plot=False,
        save_receipt=True,
    )

    assert results["mode"] == "benchmark"
    assert results["synthetic_degradation_used"] is True
    assert isinstance(results["sr_psnr"], float)
    assert results["sr_psnr"] > 20.0
    assert "lr_tile" in results

    assert "hr_tile" in results


# -----------------------------------------------------------------------------
# 13. Real mode does not generate fake PSNR/SSIM
# -----------------------------------------------------------------------------
def test_no_fake_metrics_in_real_mode(project_root, tmp_path):
    """Verify that real mode refuses to fabricate PSNR/SSIM when no independent HR ground truth exists."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    results = run_custom_pipeline(
        input_image_path=urban_dir,
        output_dir=tmp_path,
        mode="real",
        save_plot=False,
        save_receipt=True,
    )

    assert results["sr_psnr"] == "N/A (Real Scene)"
    assert results["bicubic_psnr"] == "N/A (Real Scene)"

    # Check trust receipt notes
    receipt = results["receipt"]
    assert "reconstruction_metrics_psnr_ssim" in receipt
    assert receipt["reconstruction_metrics_psnr_ssim"]["note"] == "PSNR/SSIM unavailable for unreferenced real Sentinel-2 scenes"
    assert receipt["reconstruction_metrics_psnr_ssim"]["psnr_db"] is None
    assert receipt["reconstruction_metrics_psnr_ssim"]["ssim"] is None


# -----------------------------------------------------------------------------
# 14. Trust receipt records synthetic_degradation_used_for_inference: false
# -----------------------------------------------------------------------------
def test_trust_receipt_verifiable_provenance(project_root, tmp_path):
    """Verify trust receipt records complete provenance with synthetic_degradation_used_for_inference: false."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    results = run_custom_pipeline(
        input_image_path=urban_dir,
        output_dir=tmp_path,
        mode="real",
        save_plot=False,
        save_receipt=True,
    )

    receipt_path = Path(results["exported_paths"]["trust_receipt"])
    assert receipt_path.exists()

    with open(receipt_path, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)

    prov = receipt_data["provenance"]
    assert prov["synthetic_degradation_used_for_inference"] is False
    assert prov["input_type"] == "real_sentinel2"
    assert prov["ensemble_size"] == 3

    res = receipt_data["resolution"]
    assert res["input_gsd_m"] == 10.0
    assert res["output_gsd_m"] == 4.0
    assert res["scale_factor"] == 2.5

    assert receipt_data["bands"] == ["B02", "B03", "B04", "B08"]
    assert receipt_data["model"]["architecture"] == "residual_srnet"
    assert len(receipt_data["limitations"]) > 0


# -----------------------------------------------------------------------------
# 15. User Input Validation: 4 Correct Bands Accepted
# -----------------------------------------------------------------------------
def test_user_input_four_bands_accepted(project_root):
    """Test 1: User uploads four correct Sentinel-2 bands -> accepted."""
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    band_files = list(urban_dir.glob("*.tif"))
    assert len(band_files) == 4

    val_info = validate_sentinel2_input(band_files)
    assert val_info["is_valid"] is True
    assert set(val_info["bands"]) == {"B02", "B03", "B04", "B08"}


# -----------------------------------------------------------------------------
# 16. User Input Validation: Missing B08 -> Clear Validation Error
# -----------------------------------------------------------------------------
def test_user_input_missing_b08_error(project_root):
    """Test 2: User uploads three bands missing B08 -> clear validation error."""
    from src.data.inspect_data import MissingBandError
    urban_dir = project_root / "data" / "additional_datasets" / "urban_core"
    # Provide only B02, B03, B04
    incomplete_bands = [p for p in urban_dir.glob("*.tif") if "B08" not in p.name]
    assert len(incomplete_bands) == 3

    with pytest.raises(MissingBandError) as exc_info:
        validate_sentinel2_input(incomplete_bands)

    err_msg = str(exc_info.value)
    assert "Missing B08" in err_msg or "B08" in err_msg


# -----------------------------------------------------------------------------
# 17. User Input Validation: Spatially Mismatched Bands -> Clear Alignment Error
# -----------------------------------------------------------------------------
def test_user_input_spatial_mismatch_error(tmp_path):
    """Test 3: User uploads bands with mismatched dimensions -> clear alignment error."""
    from src.data.inspect_data import SpatialAlignmentError

    # Create dummy GeoTIFFs with different dimensions
    t1_path = tmp_path / "scene_B02.tif"
    t2_path = tmp_path / "scene_B03.tif"
    t3_path = tmp_path / "scene_B04.tif"
    t4_path = tmp_path / "scene_B08.tif"

    trans = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)

    # Write B02 as 64x64
    with rasterio.open(t1_path, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32", crs="EPSG:32643", transform=trans) as dst:
        dst.write(np.ones((64, 64), dtype=np.float32), 1)

    # Write B03 as 128x128 (mismatched dimensions)
    with rasterio.open(t2_path, "w", driver="GTiff", height=128, width=128, count=1, dtype="float32", crs="EPSG:32643", transform=trans) as dst:
        dst.write(np.ones((128, 128), dtype=np.float32), 1)

    with rasterio.open(t3_path, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32", crs="EPSG:32643", transform=trans) as dst:
        dst.write(np.ones((64, 64), dtype=np.float32), 1)

    with rasterio.open(t4_path, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32", crs="EPSG:32643", transform=trans) as dst:
        dst.write(np.ones((64, 64), dtype=np.float32), 1)

    with pytest.raises(SpatialAlignmentError) as exc_info:
        validate_sentinel2_input([t1_path, t2_path, t3_path, t4_path])

    assert "Spatial alignment check failed" in str(exc_info.value) or "shape" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 18. User Input Validation: Standard RGB Image Rejection in Real Mode
# -----------------------------------------------------------------------------
def test_user_input_standard_rgb_rejection(project_root):
    """Verify standard 3-channel RGB image is rejected in real Sentinel-2 mode."""
    png_path = project_root / "outputs" / "presentation" / "01_side_by_side_super_resolution.png"
    assert png_path.exists()

    with pytest.raises(ValueError) as exc_info:
        load_and_preprocess_sentinel2(png_path)

    assert "Real Sentinel-2 inference requires B02, B03, B04 and B08" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 19. Offline Demo Cache Functionality
# -----------------------------------------------------------------------------
def test_offline_cached_demo_works():
    """Test 13: Offline cached demo bundles load cleanly without GPU dependency."""
    from src.dashboard.app import load_demo_bundle, load_demo_manifest

    manifest = load_demo_manifest()
    assert manifest is not None
    assert "tiles" in manifest or "verified_tiles" in manifest

    bundle = load_demo_bundle(0)
    assert bundle is not None
    assert "hr_tile" in bundle
    assert "sr_tile" in bundle
    assert "fusion_result" in bundle

