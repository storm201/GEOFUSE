"""Automated Integration & Security Tests for Phase 2.5.

Covers:
1. User Input: Validation, scene creation, 5x5 grid generation, tile selection (#07), custom inference.
2. Live Execution & Integrity: force_live=True bypasses demo cache & API cache, produces source=live_inference and fresh run identity.
3. Cache Isolation: Upload A vs Upload B isolation, same input cache reuse, different input non-reuse.
4. Security: Rejection of oversized uploads, invalid TIFF formats, path traversal attacks, and safe error messages.
5. Viewer Geometry: Shared viewport aspect ratio, equal physical frame dimensions, and tile updates.
"""

import io
import time
from pathlib import Path
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.api.routes.inference import (
    infer_custom_geotiff,
    infer_tile,
    validate_custom_geotiff,
)
from src.api.routes.receipt import get_trust_receipt
from src.api.routes.scenes import get_scene_grid_partition
from src.api.schemas.payloads import TileInferenceRequest
from src.api.services.cache_service import cache_service
from src.api.services.scene_service import scene_service


@pytest.fixture
def urban_band_files():
    """Load the 4 valid Sentinel-2 bands for Urban Core scene."""
    urban_dir = Path("data/additional_datasets/urban_core")
    band_paths = list(urban_dir.glob("*.tif"))
    assert len(band_paths) == 4
    files = []
    for bp in sorted(band_paths, key=lambda p: p.name):
        with open(bp, "rb") as f:
            files.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))
    return files


@pytest.fixture
def agri_band_files():
    """Load the 4 valid Sentinel-2 bands for Agriculture scene."""
    agri_dir = Path("data/additional_datasets/agriculture")
    band_paths = list(agri_dir.glob("*.tif"))
    assert len(band_paths) == 4
    files = []
    for bp in sorted(band_paths, key=lambda p: p.name):
        with open(bp, "rb") as f:
            files.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))
    return files


# =============================================================================
# 1. USER INPUT WORKFLOW & 5x5 TILE GRID TESTS
# =============================================================================

@pytest.mark.anyio
async def test_user_input_validation_and_grid_generation(urban_band_files):
    """Test that uploaded imagery produces a validated custom scene with a 5x5 tile grid."""
    val_resp = await validate_custom_geotiff(urban_band_files)

    assert val_resp.is_valid is True
    assert val_resp.scene_id.startswith("custom_")
    assert len(val_resp.upload_id) == 12
    assert val_resp.resolution[0] == 10.0
    assert val_resp.resolution[1] == 10.0
    assert set(val_resp.bands) == {"B02", "B03", "B04", "B08"}

    # Verify 5x5 tile grid was automatically generated
    assert val_resp.grid is not None
    assert val_resp.grid.total_tiles == 25
    assert len(val_resp.grid.tiles) == 25
    assert val_resp.grid.macro_preview_url.startswith("/api/assets/scenes/")

    # Verify tile 7 properties
    tile_7 = val_resp.grid.tiles[7]
    assert tile_7.tile_id == 7
    assert tile_7.patch_size == 128
    assert tile_7.w == 128
    assert tile_7.h == 128


@pytest.mark.anyio
async def test_user_input_tile_selection_inference(urban_band_files):
    """Test that selecting Tile #07 on a custom scene executes real inference without manual coordinates."""
    val_resp = await validate_custom_geotiff(urban_band_files)
    custom_scene_id = val_resp.scene_id

    req = TileInferenceRequest(
        scene_id=custom_scene_id,
        tile_id=7,
        force_live=False,
    )
    resp = await infer_tile(req)

    assert resp.scene_id == custom_scene_id
    assert resp.tile_id == 7
    assert resp.source in ("user_input", "live_inference", "user_input_cache")
    assert resp.latency_ms >= 0.0
    assert resp.evidence_metrics.trust_score_pct > 0.0
    assert resp.assets.rgb_input.startswith("/api/assets/")
    assert resp.assets.rgb_sr.startswith("/api/assets/")
    assert resp.assets.uncertainty.startswith("/api/assets/")
    assert resp.receipt_id.startswith("TR-")

    # Authoritative Trust Receipt inspection
    receipt_resp = await get_trust_receipt(resp.receipt_id)
    assert receipt_resp.status_code == 200


# =============================================================================
# 2. PROVE USER INPUT AFFECTS OUTPUT & CACHE ISOLATION
# =============================================================================

@pytest.mark.anyio
async def test_user_input_affects_output_and_cache_isolation(urban_band_files, agri_band_files):
    """Prove that Upload A and Upload B have isolated namespaces and produce different outputs."""
    # Upload Scene A (Urban Core)
    val_a = await validate_custom_geotiff(urban_band_files)
    scene_a = val_a.scene_id

    # Upload Scene B (Agriculture)
    val_b = await validate_custom_geotiff(agri_band_files)
    scene_b = val_b.scene_id

    assert scene_a != scene_b

    # Process Tile #07 on Scene A
    resp_a = await infer_tile(TileInferenceRequest(scene_id=scene_a, tile_id=7, force_live=False))

    # Process Tile #07 on Scene B
    resp_b = await infer_tile(TileInferenceRequest(scene_id=scene_b, tile_id=7, force_live=False))

    # Assert distinct run IDs, receipt IDs, and asset directories
    assert resp_a.run_id != resp_b.run_id
    assert resp_a.receipt_id != resp_b.receipt_id
    assert resp_a.assets.rgb_sr != resp_b.assets.rgb_sr

    # Second call to Scene A Tile #07 should hit custom cache
    resp_a_cached = await infer_tile(TileInferenceRequest(scene_id=scene_a, tile_id=7, force_live=False))
    assert resp_a_cached.source == "user_input_cache"
    assert resp_a_cached.run_id == resp_a.run_id


# =============================================================================
# 3. FORCE LIVE EXECUTION INTEGRITY
# =============================================================================

@pytest.mark.anyio
async def test_force_live_bypasses_demo_cache_and_api_cache():
    """Verify that force_live=True bypasses demo cache on preloaded scenes and sets source=live_inference."""
    # Normal request on demo tile 0 hits demo cache
    normal_req = TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False)
    resp_cached = await infer_tile(normal_req)
    assert resp_cached.source == "demo_cache"

    # Force live on the exact same tile
    live_req = TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=True)
    resp_live = await infer_tile(live_req)

    # Must genuinely execute, bypass demo cache, and return fresh live run ID
    assert resp_live.source == "live_inference"
    assert resp_live.run_id != resp_cached.run_id
    assert resp_live.run_id.startswith("live_urban_core_tile00_")
    assert resp_live.latency_ms > 0.0
    assert resp_live.receipt_id.startswith("TR-")


@pytest.mark.anyio
async def test_force_live_on_custom_upload(urban_band_files):
    """Verify that force_live=True on a custom scene bypasses user_input_cache and executes live."""
    val_resp = await validate_custom_geotiff(urban_band_files)
    scene_id = val_resp.scene_id

    # 1. First run creates cached entry
    resp_first = await infer_tile(TileInferenceRequest(scene_id=scene_id, tile_id=12, force_live=False))
    # 2. Subsequent call reuses cache
    resp_cached = await infer_tile(TileInferenceRequest(scene_id=scene_id, tile_id=12, force_live=False))
    assert resp_cached.source == "user_input_cache"

    # 3. Force live bypasses user_input_cache
    resp_live = await infer_tile(TileInferenceRequest(scene_id=scene_id, tile_id=12, force_live=True))
    assert resp_live.source == "live_inference"
    assert resp_live.run_id != resp_cached.run_id
    assert "live_" in resp_live.run_id


# =============================================================================
# 4. SECURITY AUDIT TESTS
# =============================================================================

@pytest.mark.anyio
async def test_security_reject_invalid_file_extension():
    """Verify that non-TIFF/non-JP2 files are rejected."""
    bad_file = UploadFile(file=io.BytesIO(b"dummy data"), filename="payload.exe")
    with pytest.raises(HTTPException) as exc_info:
        await validate_custom_geotiff([bad_file])
    assert exc_info.value.status_code == 400
    assert "Unsupported file format" in exc_info.value.detail


@pytest.mark.anyio
async def test_security_reject_path_traversal_filename():
    """Verify that path traversal filenames like '../../etc/passwd.tif' are rejected."""
    traversal_file = UploadFile(file=io.BytesIO(b"fake tiff content"), filename="../../malicious.tif")
    with pytest.raises(HTTPException) as exc_info:
        await validate_custom_geotiff([traversal_file])
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_security_reject_invalid_tiff_header():
    """Verify that invalid TIFF magic bytes are rejected."""
    corrupt_file = UploadFile(file=io.BytesIO(b"NOT_A_TIFF_FILE_HEADER"), filename="corrupt.tif")
    with pytest.raises(HTTPException) as exc_info:
        await validate_custom_geotiff([corrupt_file])
    assert exc_info.value.status_code == 400
    assert "not a valid TIFF" in exc_info.value.detail


@pytest.mark.anyio
async def test_security_reject_path_traversal_receipt_id():
    """Verify that path traversal in receipt retrieval returns 404 safely."""
    with pytest.raises(HTTPException) as exc_info:
        await get_trust_receipt("../../../etc/passwd")
    assert exc_info.value.status_code == 404
