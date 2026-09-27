"""GeoFUSE SentinelGuard — API Configuration and Path Resolution.

All paths are resolved relative to the project root using centralized configuration.
No machine-specific absolute paths are used.
"""

from pathlib import Path
from typing import List

from src.utils.config import get_project_root, load_config

# Project root and core configuration
PROJECT_ROOT: Path = get_project_root()
CORE_CONFIG = load_config()

# API Metadata
API_TITLE: str = "GeoFUSE SentinelGuard Intelligence API"
API_VERSION: str = "0.1.0"
API_DESCRIPTION: str = (
    "Production REST API for trust-aware Sentinel-2 super-resolution (10m -> 4m) "
    "with empirical evidence receipts, uncertainty proxies, and multi-band analytics."
)

# Host and Port Configuration
API_HOST: str = "127.0.0.1"
API_PORT: int = 8000

# CORS Allowed Origins for Local Development
CORS_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Dataset Directories
RAW_DATA_DIR: Path = PROJECT_ROOT / CORE_CONFIG.get("paths", {}).get("raw_data_dir", "data/raw")
ADDITIONAL_DATASETS_DIR: Path = PROJECT_ROOT / "data" / "additional_datasets"

# Cache & Storage Directories
OUTPUTS_DIR: Path = PROJECT_ROOT / CORE_CONFIG.get("paths", {}).get("outputs_dir", "outputs")
CACHE_API_DIR: Path = OUTPUTS_DIR / "cache_api"
DEMO_CACHE_DIR: Path = OUTPUTS_DIR / "demo_cache"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / CORE_CONFIG.get("paths", {}).get("checkpoints_dir", "checkpoints")

# Ensure required runtime cache directories exist
CACHE_API_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_API_DIR / "scenes").mkdir(parents=True, exist_ok=True)
