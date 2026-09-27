"""Automated Verification Tests for Phase 3:
- GPU Concurrency & Safety (Semaphore serialization, queue depth, VRAM management)
- Spatial Viewport Parity & Layer Alignment (10m vs 4m, building mask isolation)
- Tile Switching & Fresh Asset Generation (stale asset prevention)
- Preloaded vs User Input Workflow Parity
"""

import asyncio
import io
import time
from pathlib import Path
import pytest
from starlette.datastructures import UploadFile

from src.api.routes.inference import infer_tile, validate_custom_geotiff
from src.api.routes.scenes import get_scene_grid_partition
from src.api.routes.system import get_system_status
from src.api.schemas.payloads import TileInferenceRequest
from src.api.services.model_service import model_service
from src.api.services.scene_service import scene_service


@pytest.mark.anyio
async def test_gpu_concurrency_serialization():
    """Verify that model_service.acquire_gpu_slot() strictly serializes concurrent execution."""
    execution_order = []
    max_concurrent = 0

    async def worker(worker_id: int, hold_duration: float):
        nonlocal max_concurrent
        async with model_service.acquire_gpu_slot(timeout_seconds=5.0):
            current_active = model_service.active_inferences
            if current_active > max_concurrent:
                max_concurrent = current_active
            execution_order.append(f"start_{worker_id}")
            await asyncio.sleep(hold_duration)
            execution_order.append(f"end_{worker_id}")

    # Launch two workers simultaneously
    await asyncio.gather(worker(1, 0.1), worker(2, 0.1))

    # Concurrency must never exceed 1 slot
    assert max_concurrent == 1
    # Worker 1 must start and finish before Worker 2 starts (or vice versa)
    assert execution_order in (
        ["start_1", "end_1", "start_2", "end_2"],
        ["start_2", "end_2", "start_1", "end_1"],
    )
    # Queue depth and active inference counters must return to 0
    assert model_service.active_inferences == 0
    assert model_service.queue_depth == 0


@pytest.mark.anyio
async def test_gpu_queue_timeout_handling():
    """Verify that acquiring a GPU slot times out if held longer than timeout."""
    async with model_service.acquire_gpu_slot(timeout_seconds=2.0):
        # Attempt to acquire while slot is held, with short timeout
        with pytest.raises(TimeoutError, match="GPU inference queue timeout"):
            async with model_service.acquire_gpu_slot(timeout_seconds=0.1):
                pass

    # Ensure queue depth reset after timeout
    assert model_service.queue_depth == 0


@pytest.mark.anyio
async def test_system_status_reports_gpu_queue():
    """Verify that /api/system/status exposes GPU queue telemetry."""
    status = await get_system_status()
    assert hasattr(status, "gpu_queue_active")
    assert hasattr(status, "gpu_queue_depth")
    assert isinstance(status.gpu_queue_active, int)
    assert isinstance(status.gpu_queue_depth, int)
    assert status.gpu_queue_active >= 0
    assert status.gpu_queue_depth >= 0


@pytest.mark.anyio
async def test_force_live_fresh_asset_prevents_stale_browser_cache():
    """Verify that force_live=True generates fresh timestamped run IDs and asset URLs."""
    req1 = TileInferenceRequest(scene_id="urban_core", tile_id=7, force_live=True)
    resp1 = await infer_tile(req1)

    # Slight pause to ensure millisecond timestamp changes
    await asyncio.sleep(0.01)

    req2 = TileInferenceRequest(scene_id="urban_core", tile_id=7, force_live=True)
    resp2 = await infer_tile(req2)

    # Both must be live_inference
    assert resp1.source == "live_inference"
    assert resp2.source == "live_inference"

    # Run IDs and asset URLs must be distinct so the browser never serves stale cached images
    assert resp1.run_id != resp2.run_id
    assert resp1.assets.rgb_sr != resp2.assets.rgb_sr
    assert resp1.run_id.startswith("live_urban_core_tile07_")
    assert resp2.run_id.startswith("live_urban_core_tile07_")


@pytest.mark.anyio
async def test_tile_switching_produces_isolated_assets():
    """Verify that selecting tile #07 vs tile #14 yields completely isolated assets."""
    req_tile7 = TileInferenceRequest(scene_id="urban_core", tile_id=7, force_live=False)
    resp7 = await infer_tile(req_tile7)

    req_tile14 = TileInferenceRequest(scene_id="urban_core", tile_id=14, force_live=False)
    resp14 = await infer_tile(req_tile14)

    assert resp7.tile_id == 7
    assert resp14.tile_id == 14
    assert resp7.run_id != resp14.run_id
    assert resp7.assets.rgb_input != resp14.assets.rgb_input
    assert resp7.assets.rgb_sr != resp14.assets.rgb_sr
    assert resp7.receipt_id != resp14.receipt_id


@pytest.mark.anyio
async def test_spatial_viewport_parity_and_resolution():
    """Verify that 10m LR and 4m SR layers maintain strict 2.5x spatial ratio."""
    req = TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False)
    resp = await infer_tile(req)

    assert resp.input_resolution_m == 10.0
    assert resp.output_resolution_m == 4.0
    assert resp.scale_factor == 2.5

    # All multi-band and diagnostic asset URLs must be populated
    assert resp.assets.rgb_input is not None
    assert resp.assets.rgb_sr is not None
    assert resp.assets.cir_input is not None
    assert resp.assets.cir_sr is not None
    assert resp.assets.ndvi_input is not None
    assert resp.assets.ndvi_sr is not None
    assert resp.assets.uncertainty is not None
    assert resp.assets.trust_map is not None


@pytest.mark.anyio
async def test_building_footprint_tile_isolation():
    """Verify building mask is bound strictly to the active tile."""
    resp7 = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=7, force_live=False))
    resp8 = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=8, force_live=False))

    if resp7.assets.building_mask and resp8.assets.building_mask:
        assert resp7.assets.building_mask != resp8.assets.building_mask
        assert "tile07" in resp7.assets.building_mask or resp7.run_id in resp7.assets.building_mask
        assert "tile08" in resp8.assets.building_mask or resp8.run_id in resp8.assets.building_mask


@pytest.mark.anyio
async def test_preloaded_and_user_input_workflow_parity():
    """Verify both workflows adhere to the exact same response schema and metadata."""
    # 1. Preloaded Scene
    preloaded_resp = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False))

    # 2. User Input Scene
    urban_dir = Path("data/additional_datasets/urban_core")
    files = []
    for bp in sorted(urban_dir.glob("*.tif")):
        with open(bp, "rb") as f:
            files.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))
    val_resp = await validate_custom_geotiff(files)

    custom_resp = await infer_tile(TileInferenceRequest(scene_id=val_resp.scene_id, tile_id=0, force_live=False))

    # Common schema verification
    for r in (preloaded_resp, custom_resp):
        assert r.run_id is not None
        assert r.scene_id is not None
        assert r.tile_id == 0
        assert r.scale_factor == 2.5
        assert r.input_resolution_m == 10.0
        assert r.output_resolution_m == 4.0
        assert r.evidence_metrics.trust_score_pct >= 0.0
        assert r.evidence_metrics.status_label in ("NOMINAL_HIGH_TRUST", "WARNING_LOW_TRUST")
        assert r.assets.rgb_input is not None
        assert r.assets.rgb_sr is not None
        assert r.receipt_id.startswith("TR-")
