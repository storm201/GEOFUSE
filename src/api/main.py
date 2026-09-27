"""GeoFUSE SentinelGuard — FastAPI Application Gateway.

Central entry point for the REST API serving the modern React UI.
Coordinates GPU inference, raster caching, system telemetry, and auditable trust receipts.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CACHE_API_DIR,
    CORS_ORIGINS,
)
from src.api.routes import benchmark, inference, receipt, scenes, system
from src.api.schemas.payloads import HealthResponse
from src.api.services.model_service import model_service
from src.api.services.scene_service import scene_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for model warm-up and resource cleanup."""
    # Ensure cache directory is ready
    CACHE_API_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_API_DIR / "scenes").mkdir(parents=True, exist_ok=True)

    # 1. Warm up models in memory
    try:
        model_service.load_models()
    except Exception as e:
        print(f"[Lifespan Warning] Could not preload ensemble models on startup: {e}")

    # 2. Warm up primary scene overview preview
    try:
        scene_service.get_scene_grid("urban_core")
    except Exception as e:
        print(f"[Lifespan Warning] Could not preload default scene grid: {e}")

    yield
    # Cleanup logic on shutdown if needed


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount generated static raster assets directly for zero-overhead browser HTTP caching
app.mount("/api/assets", StaticFiles(directory=str(CACHE_API_DIR)), name="assets")

# Include route modules
app.include_router(system.router)
app.include_router(scenes.router)
app.include_router(inference.router)
app.include_router(receipt.router)
app.include_router(benchmark.router)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Mount frontend production build if available for zero-dependency all-in-one serving
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

