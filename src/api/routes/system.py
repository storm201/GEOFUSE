"""System Telemetry and Hardware Inspection Route."""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
import torch

from src.api.config import (
    CHECKPOINTS_DIR,
    CORE_CONFIG,
    DEMO_CACHE_DIR,
    PROJECT_ROOT,
)
from src.api.schemas.payloads import SystemStatusResponse
from src.api.services.model_service import model_service
from src.api.services.scene_service import scene_service

router = APIRouter(prefix="/api/system", tags=["System Telemetry"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """Query verifiable local environment, GPU hardware, and model readiness status.

    SCIENTIFIC HONESTY:
    All values are directly introspected from the local runtime environment (torch.cuda, sys, filesystem).
    No fabricated telemetry is returned.
    """
    cuda_avail = torch.cuda.is_available()
    dev_name = "CPU"
    vram_total = None
    vram_allocated = None
    vram_reserved = None

    if cuda_avail:
        try:
            dev_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total = round(props.total_memory / (1024**3), 2)
            vram_allocated = round(torch.cuda.memory_allocated(0) / (1024**3), 3)
            vram_reserved = round(torch.cuda.memory_reserved(0) / (1024**3), 3)
        except Exception:
            dev_name = "CUDA (Unidentified)"

    # Inspect checkpoint presence
    ckpt_files = list(CHECKPOINTS_DIR.glob("ensemble_member_*.pth"))
    ckpt_count = len(ckpt_files)

    # Inspect demo cache readiness
    demo_manifest = DEMO_CACHE_DIR / "manifest.json"
    demo_cache_ok = demo_manifest.exists()

    # Inspect local scene count dynamically from authoritative scene service
    active_scenes = scene_service.list_scenes()
    scene_count = len(active_scenes)

    return SystemStatusResponse(
        status="operational",
        project_name=CORE_CONFIG.get("project", {}).get("name", "GeoFUSE SentinelGuard"),
        version=CORE_CONFIG.get("project", {}).get("version", "0.1.0"),
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        cuda_available=cuda_avail,
        device_name=dev_name,
        vram_total_gb=vram_total,
        vram_allocated_gb=vram_allocated,
        vram_reserved_gb=vram_reserved,
        ensemble_checkpoints_found=ckpt_count,
        demo_cache_available=demo_cache_ok,
        active_dataset_count=scene_count,
        gpu_queue_active=model_service.active_inferences,
        gpu_queue_depth=model_service.queue_depth,
    )


class CleanupRequest(BaseModel):
    """Parameters for deterministic upload storage cleanup."""
    max_age_days: float = 7.0
    max_disk_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB
    dry_run: bool = False


@router.get("/storage")
async def get_storage_metrics():
    """Inspect user upload storage metrics without modifying disk."""
    from src.api.services.cleanup_service import cleanup_service
    return cleanup_service.inspect_storage()


@router.post("/cleanup")
async def run_storage_cleanup(req: Optional[CleanupRequest] = None):
    """Execute deterministic TTL + LRU storage cleanup on user uploads."""
    from src.api.services.cleanup_service import cleanup_service
    if req is None:
        req = CleanupRequest()
    return cleanup_service.run_cleanup(
        max_age_days=req.max_age_days,
        max_disk_bytes=req.max_disk_bytes,
        dry_run=req.dry_run,
    )

