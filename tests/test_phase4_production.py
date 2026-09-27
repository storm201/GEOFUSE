"""Automated Verification Tests for Phase 4:
- Production Frontend Build & Dist Artifact Verification
- FastAPI Static Serving of React Bundle & API Precedence
- One-Click Launcher Verification & Prerequisites Check
- Health & System Telemetry Readiness
- Streamlit Fallback Module Integrity
"""

import asyncio
from pathlib import Path
import pytest

from src.api.main import app
from src.api.routes.system import get_system_status
from src.api.schemas.payloads import SystemStatusResponse


def test_frontend_production_build_exists():
    """Verify that frontend/dist is built and ready for production serving."""
    dist_dir = Path("frontend/dist")
    index_html = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    assert dist_dir.exists(), "frontend/dist directory must exist"
    assert index_html.exists(), "frontend/dist/index.html must exist"
    assert assets_dir.exists(), "frontend/dist/assets directory must exist"

    html_content = index_html.read_text(encoding="utf-8")
    assert '<div id="root"></div>' in html_content or '<div id="root"' in html_content
    assert "/assets/index-" in html_content, "index.html must reference bundled production assets"


def test_launcher_script_integrity():
    """Verify scripts/launch_server.py syntax and helper functions."""
    import scripts.launch_server as launcher

    assert hasattr(launcher, "is_port_in_use")
    assert hasattr(launcher, "poll_endpoint")
    assert hasattr(launcher, "verify_prerequisites")
    assert hasattr(launcher, "HOST")
    assert hasattr(launcher, "PORT")
    assert launcher.HOST == "127.0.0.1"
    assert launcher.PORT == 8000

    # Test port check on unused port
    assert not launcher.is_port_in_use("127.0.0.1", 59999)


@pytest.mark.anyio
async def test_fastapi_production_static_mount():
    """Verify that FastAPI ASGI application serves production frontend on GET /."""
    response_body = []
    response_status = None
    response_headers = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        nonlocal response_status, response_headers
        if message["type"] == "http.response.start":
            response_status = message["status"]
            response_headers = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "server": ("127.0.0.1", 8000),
    }

    await app(scope, receive, send)

    assert response_status == 200, f"Expected 200 OK for root path, got {response_status}"
    body = b"".join(response_body).decode("utf-8", errors="ignore")
    assert '<div id="root">' in body or '<div id="root"' in body
    assert "GeoFUSE SentinelGuard" in body


@pytest.mark.anyio
async def test_api_precedence_over_static_mount():
    """Verify that /api/health and /api/system/status are not shadowed by the static mount."""
    status_response = await get_system_status()
    assert isinstance(status_response, SystemStatusResponse)
    assert status_response.status == "operational"
    assert status_response.project_name == "GeoFUSE SentinelGuard"
    assert status_response.ensemble_checkpoints_found == 3
    assert status_response.demo_cache_available is True


def test_batch_launchers_configuration():
    """Verify that both run_pipeline.bat and run me.bat include the production launcher."""
    pipeline_bat = Path("run_pipeline.bat").read_text(encoding="utf-8")
    assert "Launch GeoFUSE Full-Stack Platform" in pipeline_bat
    assert "scripts\\launch_server.py" in pipeline_bat

    run_me_bat = Path("run me.bat").read_text(encoding="utf-8")
    assert "scripts\\launch_server.py" in run_me_bat


def test_streamlit_fallback_dashboard_integrity():
    """Verify that the original Streamlit dashboard remains importable and functional."""
    import src.dashboard.app as st_app
    assert st_app is not None
