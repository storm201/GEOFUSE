"""Phase 6 Final QA, Validation Consistency, and Deployment Freeze Test Suite."""

import io
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from starlette.datastructures import UploadFile

from src.api.routes.inference import infer_tile, validate_custom_geotiff
from src.api.routes.system import get_system_status
from src.api.schemas.payloads import TileInferenceRequest
from src.api.services.cleanup_service import cleanup_service
from src.api.services.scene_service import scene_service


def _create_synthetic_tiff_bytes(height: int, width: int) -> bytes:
    """Generate in-memory 4-band GeoTIFF with specified spatial dimensions."""
    buf = io.BytesIO()
    transform = from_origin(500000.0, 5800000.0, 10.0, 10.0)
    data = (np.random.rand(4, height, width) * 2000).astype(np.uint16)
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=data.dtype,
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.getvalue()


@pytest.mark.anyio
async def test_dimension_validation_consistency_rejects_sub_128():
    """Verify that rasters below 128x128 are rejected with an explicit error specifying the 128x128 minimum."""
    # 64x64 is below the 128x128 scientific threshold for 5x5 tiling and 2.5x SR
    small_tiff_bytes = _create_synthetic_tiff_bytes(64, 64)
    upload_file = UploadFile(file=io.BytesIO(small_tiff_bytes), filename="small_raster.tif")

    with pytest.raises(Exception) as excinfo:
        await validate_custom_geotiff([upload_file])

    err_msg = str(excinfo.value).lower()
    assert "128x128" in err_msg or "too small" in err_msg


@pytest.mark.anyio
async def test_dimension_validation_accepts_128_and_above():
    """Verify that a 128x128 raster satisfies the scientific pipeline requirements."""
    valid_tiff_bytes = _create_synthetic_tiff_bytes(128, 128)
    upload_file = UploadFile(file=io.BytesIO(valid_tiff_bytes), filename="valid_128.tif")

    resp = await validate_custom_geotiff([upload_file])
    assert resp.is_valid is True
    assert resp.shape == [128, 128]
    assert resp.grid is not None
    assert resp.grid.total_tiles == 25


@pytest.mark.anyio
async def test_provenance_isolation_between_distinct_uploads():
    """Verify that two distinct uploads have isolated namespaces, run IDs, and receipt IDs."""
    bytes_a = _create_synthetic_tiff_bytes(140, 140)
    bytes_b = _create_synthetic_tiff_bytes(150, 150)

    val_a = await validate_custom_geotiff([UploadFile(file=io.BytesIO(bytes_a), filename="raster_a.tif")])
    val_b = await validate_custom_geotiff([UploadFile(file=io.BytesIO(bytes_b), filename="raster_b.tif")])

    assert val_a.scene_id != val_b.scene_id

    resp_a = await infer_tile(TileInferenceRequest(scene_id=val_a.scene_id, tile_id=7, force_live=False))
    resp_b = await infer_tile(TileInferenceRequest(scene_id=val_b.scene_id, tile_id=7, force_live=False))

    assert resp_a.run_id != resp_b.run_id
    assert resp_a.receipt_id != resp_b.receipt_id
    assert resp_a.assets.rgb_sr != resp_b.assets.rgb_sr


@pytest.mark.anyio
async def test_tile_switching_updates_all_assets():
    """Verify switching from Tile 07 to Tile 14 on a custom scene updates all visualization assets."""
    tiff_bytes = _create_synthetic_tiff_bytes(160, 160)
    val = await validate_custom_geotiff([UploadFile(file=io.BytesIO(tiff_bytes), filename="raster_switch.tif")])

    resp_07 = await infer_tile(TileInferenceRequest(scene_id=val.scene_id, tile_id=7, force_live=False))
    resp_14 = await infer_tile(TileInferenceRequest(scene_id=val.scene_id, tile_id=14, force_live=False))

    assert resp_07.run_id != resp_14.run_id
    assert resp_07.assets.rgb_sr != resp_14.assets.rgb_sr
    assert resp_07.assets.rgb_input != resp_14.assets.rgb_input
    assert resp_07.receipt_id != resp_14.receipt_id


@pytest.mark.anyio
async def test_force_live_executes_real_inference_and_bypasses_cache():
    """Verify force_live=True generates a fresh live_ run ID and real execution latency."""
    # Preloaded Scene urban_core tile 0
    resp_cached = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False))
    assert resp_cached.source == "demo_cache"

    resp_live = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=True))
    assert resp_live.source == "live_inference"
    assert "live_" in resp_live.run_id
    assert resp_live.run_id != resp_cached.run_id
    assert resp_live.latency_ms > 0.0


@pytest.mark.anyio
async def test_system_telemetry_dynamically_reports_four_scenes():
    """Verify system status reports exactly 4 detected scenes without hardcoding."""
    status = await get_system_status()
    assert status.active_dataset_count == 4
    assert len(scene_service.list_scenes()) == 4
    assert len(status.device_name) > 0


def test_cleanup_policy_never_deletes_demo_or_raw_assets():
    """Verify cleanup_service never purges preloaded demo cache or raw baseline data."""
    # Run cleanup with aggressive 0-day TTL to test boundary conditions
    res = cleanup_service.run_cleanup(max_age_days=0.0, max_disk_bytes=100, dry_run=False)
    assert res.freed_bytes >= 0

    # Baseline datasets and demo cache must be completely intact
    assert Path("data/raw").exists()
    assert Path("data/additional_datasets/urban_core").exists()
    assert Path("data/additional_datasets/agriculture").exists()
    assert Path("data/additional_datasets/temporal_april2024").exists()
    assert Path("outputs/demo_cache").exists()
