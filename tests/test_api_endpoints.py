"""Automated Integration Tests for Phase 1: FastAPI Backend Services & Endpoints."""

import asyncio
from pathlib import Path
import pytest
from fastapi import HTTPException

from src.api.routes.benchmark import get_benchmark_summary
from src.api.routes.inference import infer_tile
from src.api.routes.receipt import get_trust_receipt
from src.api.routes.scenes import get_scene_grid_partition, list_available_scenes
from src.api.schemas.payloads import TileInferenceRequest
from src.api.services.cache_service import cache_service
from src.api.services.scene_service import scene_service


@pytest.mark.anyio
async def test_api_scenes_list():
    """Verify that /api/scenes/list returns real local Sentinel-2 scenes."""
    resp = await list_available_scenes()
    assert len(resp.scenes) >= 1
    assert resp.default_scene_id == "urban_core"
    scene_ids = [s.scene_id for s in resp.scenes]
    assert "urban_core" in scene_ids
    assert "agriculture" in scene_ids


@pytest.mark.anyio
async def test_api_scene_grid():
    """Verify that /api/scenes/urban_core/grid returns 25 tiles and valid preview URL."""
    resp = await get_scene_grid_partition("urban_core")
    assert resp.scene_id == "urban_core"
    assert resp.total_tiles == 25
    assert len(resp.tiles) == 25
    assert resp.macro_preview_url.startswith("/api/assets/scenes/")
    assert resp.dimensions == [512, 512]

    # Verify tile 0 properties
    t0 = resp.tiles[0]
    assert t0.tile_id == 0
    assert t0.row == 1 and t0.col == 1
    assert t0.has_demo_cache is True


@pytest.mark.anyio
async def test_api_scene_grid_invalid_scene():
    """Verify that non-existent scene returns 404."""
    with pytest.raises(HTTPException) as exc_info:
        await get_scene_grid_partition("non_existent_scene_xyz")
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_api_inference_demo_cache_hit():
    """Verify that requesting demo tile 0 uses the precomputed demo cache."""
    req = TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False)
    resp = await infer_tile(req)

    assert resp.scene_id == "urban_core"
    assert resp.tile_id == 0
    assert resp.source == "demo_cache"
    assert resp.evidence_metrics.trust_score_pct > 80.0
    assert resp.assets.rgb_input.startswith("/api/assets/")
    assert resp.assets.rgb_sr.startswith("/api/assets/")
    assert resp.assets.cir_input.startswith("/api/assets/")
    assert resp.assets.cir_sr.startswith("/api/assets/")
    assert resp.assets.ndvi_input.startswith("/api/assets/")
    assert resp.assets.ndvi_sr.startswith("/api/assets/")
    assert resp.assets.uncertainty.startswith("/api/assets/")
    assert resp.assets.trust_map.startswith("/api/assets/")
    assert resp.receipt_id.startswith("TR-")


from pydantic import ValidationError


@pytest.mark.anyio
async def test_api_inference_invalid_tile():
    """Verify that invalid tile index returns validation or bad request error."""
    with pytest.raises((HTTPException, ValidationError)):
        req = TileInferenceRequest(scene_id="urban_core", tile_id=99)
        await infer_tile(req)


@pytest.mark.anyio
async def test_api_receipt_lookup():
    """Verify that /api/receipt/{id} fetches the authoritative JSON receipt."""
    # First ensure tile 0 is cached
    req = TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False)
    infer_res = await infer_tile(req)
    receipt_id = infer_res.receipt_id

    resp = await get_trust_receipt(receipt_id=receipt_id)
    assert resp.status_code == 200
    # Response content is in body
    import json
    data = json.loads(resp.body.decode("utf-8"))
    assert data["receipt_id"] == receipt_id
    assert "tile_metadata" in data
    assert "evidence_metrics" in data


@pytest.mark.anyio
async def test_api_receipt_not_found():
    """Verify that non-existent receipt ID returns 404."""
    with pytest.raises(HTTPException) as exc_info:
        await get_trust_receipt(receipt_id="TR-NON-EXISTENT-XYZ-9999")
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_api_benchmark_summary():
    """Verify that /api/benchmark/summary returns hold-out benchmark results."""
    resp = await get_benchmark_summary()
    assert resp.benchmark_type == "synthetic_degrade_and_recover"
    assert len(resp.baseline_methods) >= 2
    assert "scientific_disclaimer" in resp.model_dump()
    # Check ResidualSRNet is present
    method_names = [m["name"] for m in resp.baseline_methods]
    assert any("ResidualSRNet" in n for n in method_names)


import io
from starlette.datastructures import UploadFile
from src.api.routes.inference import infer_custom_geotiff, validate_custom_geotiff


@pytest.mark.anyio
async def test_api_custom_validation_valid():
    """Verify that /api/inference/validate accepts 4 matching Sentinel-2 bands."""
    urban_dir = Path("data/additional_datasets/urban_core")
    band_paths = list(urban_dir.glob("*.tif"))
    assert len(band_paths) == 4

    upload_files = []
    for bp in band_paths:
        with open(bp, "rb") as f:
            upload_files.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))

    resp = await validate_custom_geotiff(upload_files)
    assert resp.is_valid is True
    assert resp.format == "separate_bands"
    assert resp.files_count == 4
    assert set(resp.bands) == {"B02", "B03", "B04", "B08"}
    assert resp.resolution[0] == 10.0
    assert resp.resolution[1] == 10.0
    assert len(resp.upload_id) == 12


@pytest.mark.anyio
async def test_api_custom_validation_invalid():
    """Verify that /api/inference/validate rejects invalid/incomplete files."""
    dummy_file = UploadFile(file=io.BytesIO(b"not a geotiff"), filename="invalid.txt")
    with pytest.raises(HTTPException) as exc_info:
        await validate_custom_geotiff([dummy_file])
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_api_custom_inference_end_to_end():
    """Verify that /api/inference/custom executes the complete GeoFUSE pipeline on uploaded files."""
    urban_dir = Path("data/additional_datasets/urban_core")
    band_paths = list(urban_dir.glob("*.tif"))
    assert len(band_paths) == 4

    upload_files = []
    for bp in band_paths:
        with open(bp, "rb") as f:
            upload_files.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))

    # 1. Validate
    val_resp = await validate_custom_geotiff(upload_files)
    upload_id = val_resp.upload_id

    # 2. Run Custom Inference using staged upload_id
    infer_resp = await infer_custom_geotiff(
        upload_id=upload_id,
        tile_size=128,
        force_live=True,
    )

    assert infer_resp.source == "user_input"
    assert infer_resp.scene_id == "user_input"
    assert infer_resp.tile_id == 0
    assert infer_resp.evidence_metrics.trust_score_pct > 0.0
    assert infer_resp.assets.rgb_input.startswith("/api/assets/")
    assert infer_resp.assets.rgb_sr.startswith("/api/assets/")
    assert infer_resp.assets.uncertainty.startswith("/api/assets/")
    assert infer_resp.assets.trust_map.startswith("/api/assets/")
    assert infer_resp.receipt_id.startswith("TR-")

    # 3. Verify receipt can be fetched
    receipt_resp = await get_trust_receipt(receipt_id=infer_resp.receipt_id)
    assert receipt_resp.status_code == 200

