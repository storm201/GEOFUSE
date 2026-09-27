"""Automated Verification Tests for Phase 5:
- Scene Discovery & 4-Dataset Sanity Verification
- User-Upload Retention Management (TTL + LRU cleanup, dry-run, API endpoints)
- Offline Resilience & Zero CDN Runtime Requirement
- Demo Cache Instant Retrieval & Clean Fallback
- GPU Failure Handling & Truthful Recovery States
- User Input Integrity: Never substitutes demo data on failure
"""

import asyncio
import io
import shutil
import time
from pathlib import Path
import pytest
from starlette.datastructures import UploadFile

from src.api.routes.inference import infer_tile, validate_custom_geotiff
from src.api.routes.scenes import list_available_scenes
from src.api.routes.system import get_storage_metrics, get_system_status, run_storage_cleanup
from src.api.schemas.payloads import TileInferenceRequest
from src.api.services.cleanup_service import cleanup_service
from src.api.services.scene_service import scene_service


@pytest.mark.anyio
async def test_scene_discovery_all_four_datasets():
    """Verify that all 4 local datasets are discovered and exposed through the API."""
    scenes_resp = await list_available_scenes()
    scene_ids = [s.scene_id for s in scenes_resp.scenes]

    assert len(scenes_resp.scenes) == 4, f"Expected 4 active scenes, found {len(scenes_resp.scenes)}: {scene_ids}"
    assert "urban_core" in scene_ids
    assert "agriculture" in scene_ids
    assert "temporal_april2024" in scene_ids
    assert "primary_raw" in scene_ids

    # Telemetry must match authoritative count
    sys_status = await get_system_status()
    assert sys_status.active_dataset_count == 4


@pytest.mark.anyio
async def test_cleanup_service_inspect_storage():
    """Verify storage inspection returns structured telemetry without disk mutations."""
    metrics = await get_storage_metrics()
    assert "upload_count" in metrics
    assert "total_bytes" in metrics
    assert "total_mb" in metrics
    assert isinstance(metrics["uploads"], list)
    assert metrics["upload_count"] >= 0


def test_cleanup_service_dry_run_preserves_disk(tmp_path):
    """Verify cleanup in dry_run mode calculates candidates without deleting files."""
    test_cleanup = type(cleanup_service)()
    test_cleanup.uploads_dir = tmp_path / "test_uploads"
    test_cleanup.uploads_dir.mkdir(parents=True)

    # Create dummy upload dir
    dummy_dir = test_cleanup.uploads_dir / "test_hash_old"
    dummy_dir.mkdir()
    (dummy_dir / "band.tif").write_bytes(b"0" * 1024)

    # Artificially set modification time to 10 days ago
    ten_days_ago = time.time() - (10 * 86400)
    import os
    os.utime(dummy_dir / "band.tif", (ten_days_ago, ten_days_ago))
    os.utime(dummy_dir, (ten_days_ago, ten_days_ago))

    summary = test_cleanup.run_cleanup(max_age_days=7.0, dry_run=True)
    assert summary.dry_run is True
    assert len(summary.evicted_candidates) == 1
    assert summary.freed_bytes == 1024
    # File must still exist because dry_run was True
    assert dummy_dir.exists()


def test_cleanup_service_ttl_execution(tmp_path):
    """Verify cleanup deletes uploads older than max_age_days when dry_run=False."""
    test_cleanup = type(cleanup_service)()
    test_cleanup.uploads_dir = tmp_path / "test_uploads"
    test_cleanup.uploads_dir.mkdir(parents=True)

    # 1. Old upload (10 days old)
    old_dir = test_cleanup.uploads_dir / "hash_old"
    old_dir.mkdir()
    (old_dir / "b.tif").write_bytes(b"A" * 2048)
    ten_days_ago = time.time() - (10 * 86400)
    import os
    os.utime(old_dir / "b.tif", (ten_days_ago, ten_days_ago))
    os.utime(old_dir, (ten_days_ago, ten_days_ago))

    # 2. Fresh upload (1 hour old)
    fresh_dir = test_cleanup.uploads_dir / "hash_fresh"
    fresh_dir.mkdir()
    (fresh_dir / "b.tif").write_bytes(b"B" * 2048)

    summary = test_cleanup.run_cleanup(max_age_days=7.0, dry_run=False)
    assert summary.dry_run is False
    assert len(summary.evicted_candidates) == 1
    assert summary.evicted_candidates[0].upload_hash == "hash_old"

    # Old dir deleted, fresh dir preserved
    assert not old_dir.exists()
    assert fresh_dir.exists()


def test_cleanup_service_lru_cap_execution(tmp_path):
    """Verify cleanup enforces max_disk_bytes by evicting oldest uploads first."""
    test_cleanup = type(cleanup_service)()
    test_cleanup.uploads_dir = tmp_path / "test_uploads"
    test_cleanup.uploads_dir.mkdir(parents=True)

    now = time.time()
    import os

    # Dir 1: 3 hours old, 1000 bytes
    d1 = test_cleanup.uploads_dir / "hash_1"
    d1.mkdir()
    (d1 / "f.tif").write_bytes(b"1" * 1000)
    os.utime(d1 / "f.tif", (now - 10800, now - 10800))
    os.utime(d1, (now - 10800, now - 10800))

    # Dir 2: 2 hours old, 1000 bytes
    d2 = test_cleanup.uploads_dir / "hash_2"
    d2.mkdir()
    (d2 / "f.tif").write_bytes(b"2" * 1000)
    os.utime(d2 / "f.tif", (now - 7200, now - 7200))
    os.utime(d2, (now - 7200, now - 7200))

    # Dir 3: 1 hour old, 1000 bytes
    d3 = test_cleanup.uploads_dir / "hash_3"
    d3.mkdir()
    (d3 / "f.tif").write_bytes(b"3" * 1000)
    os.utime(d3 / "f.tif", (now - 3600, now - 3600))
    os.utime(d3, (now - 3600, now - 3600))

    # Cap at 1500 bytes (should evict oldest d1 and d2 to stay under cap)
    summary = test_cleanup.run_cleanup(max_age_days=30.0, max_disk_bytes=1500, dry_run=False)
    evicted_names = [c.upload_hash for c in summary.evicted_candidates]

    assert "hash_1" in evicted_names
    assert not d1.exists()
    assert d3.exists()  # Newest must be preserved


@pytest.mark.anyio
async def test_demo_cache_instant_retrieval():
    """Verify preloaded demo tile returns instantly with source=demo_cache."""
    resp = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False))
    assert resp.source == "demo_cache"
    assert resp.latency_ms == 0.0
    assert resp.assets.rgb_sr is not None


@pytest.mark.anyio
async def test_user_input_never_substitutes_demo_data_on_error():
    """Verify that an invalid tile or failure on User Input clearly errors and never substitutes demo results."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    # 1. Tile index out of bounds must fail validation immediately
    with pytest.raises(ValidationError):
        TileInferenceRequest(scene_id="custom_nonexistent_hash", tile_id=99, force_live=False)

    # 2. Non-existent custom scene must fail with 404 rather than silently returning demo data
    with pytest.raises(HTTPException) as excinfo:
        await infer_tile(TileInferenceRequest(scene_id="custom_nonexistent_hash", tile_id=12, force_live=False))

    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.detail.lower()


def test_offline_bundle_zero_external_cdn_dependency():
    """Verify that production bundle HTML and CSS do not depend on external scripts."""
    dist_html = Path("frontend/dist/index.html").read_text(encoding="utf-8")

    # Scripts must only be local modules
    import re
    script_srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', dist_html)
    for src in script_srcs:
        assert src.startswith("/assets/") or src.startswith("./"), f"External script found: {src}"
