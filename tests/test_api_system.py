"""Unit tests for Phase 0: FastAPI Initialization and System Telemetry."""

import asyncio
import pytest

from src.api.main import app
from src.api.routes.system import get_system_status


def test_fastapi_app_initialization():
    """Verify that the FastAPI application initializes with correct metadata."""
    assert app.title == "GeoFUSE SentinelGuard Intelligence API"
    assert app.version == "0.1.0"


def test_system_status_telemetry():
    """Verify that system status returns genuine, non-fabricated hardware telemetry."""
    status = asyncio.run(get_system_status())

    assert status.status == "operational"
    assert status.project_name == "GeoFUSE SentinelGuard"
    assert status.python_version.startswith("3.")
    assert "2." in status.pytorch_version  # PyTorch 2.x
    assert status.ensemble_checkpoints_found == 3
    assert status.demo_cache_available is True
    assert status.active_dataset_count >= 1

    if status.cuda_available:
        assert status.vram_total_gb is not None and status.vram_total_gb > 0
        assert "RTX" in status.device_name or "GeForce" in status.device_name or "NVIDIA" in status.device_name
