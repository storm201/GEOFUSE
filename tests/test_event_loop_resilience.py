"""Automated Tests for FastAPI Event-Loop Resilience, GPU Serialization, and Stale Request Protection."""

import asyncio
import time
import pytest
import torch

from src.api.main import health_check
from src.api.routes.inference import infer_tile
from src.api.schemas.payloads import TileInferenceRequest
from src.api.services.model_service import model_service


@pytest.mark.anyio
async def test_event_loop_unblocked_during_live_inference():
    """Verify that the FastAPI asyncio event loop remains fully responsive while GPU inference is executing."""
    # Start live inference task
    inf_task = asyncio.create_task(
        infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=7, force_live=True))
    )

    # Ping health endpoint concurrently
    ping_latencies = []
    for _ in range(4):
        await asyncio.sleep(0.01)
        t0 = time.perf_counter()
        resp = await health_check()
        dt = (time.perf_counter() - t0) * 1000.0
        ping_latencies.append(dt)
        assert resp.status == "ok"

    res = await inf_task
    assert res.source == "live_inference"

    # Max health ping must be under 25ms (proves event loop was never blocked by PyTorch/WebP encoding)
    assert max(ping_latencies) < 25.0


@pytest.mark.anyio
async def test_concurrent_gpu_inference_serialization():
    """Verify concurrent tile inference requests are serialized safely without CUDA crashes."""
    tasks = [
        asyncio.create_task(
            infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=i, force_live=True))
        )
        for i in range(3)
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    run_ids = {r.run_id for r in results}
    assert len(run_ids) == 3  # All run IDs must be unique
    for r in results:
        assert r.source == "live_inference"


@pytest.mark.anyio
async def test_gpu_queue_timeout_handling():
    """Verify that acquire_gpu_slot raises TimeoutError if GPU is held beyond timeout threshold."""
    # Acquire the single GPU slot
    async with model_service.acquire_gpu_slot():
        # Attempt to acquire another slot with ultra-short 0.05s timeout
        with pytest.raises(TimeoutError) as excinfo:
            async with model_service.acquire_gpu_slot(timeout_seconds=0.05):
                pass
        assert "timeout" in str(excinfo.value).lower()


@pytest.mark.anyio
async def test_rapid_tile_switching_integrity():
    """Verify that rapidly requesting multiple tiles returns distinct, correct tile results."""
    # Rapid sequence: Tile 07 -> Tile 12 -> Tile 19
    t7 = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=7, force_live=False))
    t12 = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=12, force_live=False))
    t19 = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=19, force_live=False))

    assert t7.tile_id == 7
    assert t12.tile_id == 12
    assert t19.tile_id == 19
    assert t7.assets.rgb_sr != t12.assets.rgb_sr
    assert t12.assets.rgb_sr != t19.assets.rgb_sr


@pytest.mark.anyio
async def test_repeated_inference_vram_stability():
    """Verify that 10 consecutive live inference calls do not grow allocated VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Warmup
    await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=True))
    torch.cuda.empty_cache()
    vram_start = torch.cuda.memory_allocated()

    for i in range(10):
        await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=i % 25, force_live=True))

    torch.cuda.empty_cache()
    vram_end = torch.cuda.memory_allocated()
    # VRAM allocated should not grow
    delta_mb = (vram_end - vram_start) / (1024 ** 2)
    assert delta_mb < 2.0, f"VRAM grew by {delta_mb:.2f} MB"
