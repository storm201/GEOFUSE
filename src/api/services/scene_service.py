"""Scene Service — Discovery, Metadata, and Tile Grid Partitioning.

Reuses the existing GeoFUSE scene utilities and tile grid partitioning logic
from src.utils.scene_visualizer and src.data.tiling.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

from src.api.config import (
    ADDITIONAL_DATASETS_DIR,
    CACHE_API_DIR,
    CORE_CONFIG,
    DEMO_CACHE_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from src.api.schemas.payloads import (
    SceneGridResponse,
    SceneItem,
    TileMetadataItem,
)
from src.data.inspect_data import find_band_files
from src.inference.real_inference import load_and_preprocess_sentinel2, validate_sentinel2_input
from src.utils.scene_visualizer import get_tile_metadata_grid, to_display_rgb

# Standard demo tile indices with precomputed offline caches
DEMO_TILES = {0, 8, 16, 24}

# Scene Registry mapping logical IDs to local directories and descriptions
KNOWN_SCENES: Dict[str, Dict[str, str]] = {
    "urban_core": {
        "name": "Urban Core (Hyderabad Sector)",
        "description": "Dense metropolitan landscape with high-contrast structural and road features (S2A 10m L2A).",
        "dir_rel": "data/additional_datasets/urban_core",
    },
    "agriculture": {
        "name": "Agricultural & River Valley",
        "description": "Heterogeneous farmland plots, field boundaries, and riparian vegetation corridors (S2A 10m L2A).",
        "dir_rel": "data/additional_datasets/agriculture",
    },
    "temporal_april2024": {
        "name": "Temporal Revisit (April 2024)",
        "description": "Multi-temporal revisit scene of the study area captured during dry seasonal transition.",
        "dir_rel": "data/additional_datasets/temporal_april2024",
    },
    "primary_raw": {
        "name": "Primary Reference Scene",
        "description": "Original raw Sentinel-2 4-band MSI dataset used for research hold-out evaluation.",
        "dir_rel": "data/raw",
    },
}


class SceneService:
    """Manages scene discovery, spatial metadata extraction, and tile partition layouts."""

    def __init__(self):
        self._scenes_dir = CACHE_API_DIR / "scenes"
        self._scenes_dir.mkdir(parents=True, exist_ok=True)
        self._scene_cache: Dict[str, np.ndarray] = {}

    def get_scene_path(self, scene_id: str) -> Path:
        """Resolve directory path for a given scene ID."""
        if scene_id in KNOWN_SCENES:
            path = PROJECT_ROOT / KNOWN_SCENES[scene_id]["dir_rel"]
            if path.exists() and list(path.glob("*.tif")):
                return path

        # Fallback check in additional_datasets
        candidate = ADDITIONAL_DATASETS_DIR / scene_id
        if candidate.exists() and candidate.is_dir() and list(candidate.glob("*.tif")):
            return candidate

        # Support user-uploaded custom scenes
        if scene_id.startswith("custom_"):
            upload_hash = scene_id[len("custom_"):]
            custom_dir = OUTPUTS_DIR / "user_uploads" / upload_hash
            if custom_dir.exists() and custom_dir.is_dir():
                return custom_dir

        raise FileNotFoundError(f"Scene directory for '{scene_id}' not found.")

    def list_scenes(self) -> List[SceneItem]:
        """Discover and return all active Sentinel-2 scenes in the local filesystem."""
        items: List[SceneItem] = []

        for sid, meta in KNOWN_SCENES.items():
            dir_path = PROJECT_ROOT / meta["dir_rel"]
            if not dir_path.exists():
                continue

            tif_files = list(dir_path.glob("*.tif"))
            if not tif_files:
                continue

            try:
                bands = find_band_files(dir_path)
                band_keys = list(bands.keys())
            except Exception:
                band_keys = ["B02", "B03", "B04", "B08"]

            items.append(
                SceneItem(
                    scene_id=sid,
                    name=meta["name"],
                    description=meta["description"],
                    path=meta["dir_rel"],
                    bands=band_keys,
                    resolution_meters=10.0,
                    crs="EPSG:32643 (UTM zone 43N)",
                    dimensions_pixels=[512, 512],
                )
            )

        return items

    def load_scene_stack(self, scene_id: str) -> np.ndarray:
        """Load and cache (H, W, 4) multi-band reflectance array for a scene."""
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        path = self.get_scene_path(scene_id)
        stack, _ = load_and_preprocess_sentinel2(path)
        self._scene_cache[scene_id] = stack
        return stack

    def get_scene_grid(self, scene_id: str) -> SceneGridResponse:
        """Return 5x5 tile grid metadata and macro overview preview asset URL."""
        scene_path = self.get_scene_path(scene_id)
        if scene_id in KNOWN_SCENES:
            meta = KNOWN_SCENES[scene_id]
            scene_name = meta["name"]
        elif scene_id.startswith("custom_"):
            upload_hash = scene_id[len("custom_"):]
            scene_name = f"User Input Scene ({upload_hash[:8]})"
        else:
            scene_name = scene_id.replace("_", " ").title()

        stack = self.load_scene_stack(scene_id)
        h, w = stack.shape[:2]

        if h < 128 or w < 128:
            raise ValueError(
                f"Raster dimensions ({h}x{w}) are too small for Sentinel-2 super-resolution. "
                "Minimum required size is 128x128 pixels."
            )

        # 1. Ensure macro preview image is rendered & saved as asset
        preview_filename = f"{scene_id}_macro.png"
        preview_path = self._scenes_dir / preview_filename
        if not preview_path.exists():
            rgb = to_display_rgb(stack, false_color=False)
            img = Image.fromarray(rgb)
            img.save(preview_path, format="PNG", optimize=True)

        preview_url = f"/api/assets/scenes/{preview_filename}"

        # 2. Extract 25-tile grid partition metadata
        # Generate 5x5 partition using 5 uniform steps along each axis
        patch_size = 128
        y_steps = np.linspace(0, h - patch_size, 5).astype(int).tolist()
        x_steps = np.linspace(0, w - patch_size, 5).astype(int).tolist()

        quadrant_names = {
            (0, 0): "NW Sector (Top-Left Settlement)",
            (0, 4): "NE Sector (North-East Border)",
            (1, 1): "Agricultural & Rural Roads",
            (2, 2): "Central Settlement & Urban Core",
            (3, 1): "Rural River Corridor",
            (4, 0): "SW Sector (South-West Plains)",
            (4, 4): "SE Quadrant (Hold-out Transition)",
        }

        tile_items: List[TileMetadataItem] = []
        is_primary = scene_id in ("urban_core", "primary_raw")
        demo_manifest_ok = (DEMO_CACHE_DIR / "manifest.json").exists()

        tile_id = 0
        for r_idx, y in enumerate(y_steps):
            for c_idx, x in enumerate(x_steps):
                pos_key = (r_idx, c_idx)
                if pos_key in quadrant_names:
                    desc = quadrant_names[pos_key]
                elif r_idx == 0:
                    desc = f"Northern Perimeter (Col {c_idx + 1})"
                elif r_idx == 4:
                    desc = f"Southern Perimeter (Col {c_idx + 1})"
                elif c_idx == 0:
                    desc = f"Western Margin (Row {r_idx + 1})"
                elif c_idx == 4:
                    desc = f"Eastern Margin (Row {r_idx + 1})"
                else:
                    desc = f"Interior Landscape (Row {r_idx + 1}, Col {c_idx + 1})"

                has_demo = is_primary and demo_manifest_ok and (tile_id in DEMO_TILES)

                tile_items.append(
                    TileMetadataItem(
                        tile_id=tile_id,
                        row=r_idx + 1,
                        col=c_idx + 1,
                        x=int(x),
                        y=int(y),
                        w=patch_size,
                        h=patch_size,
                        patch_size=patch_size,
                        description=desc,
                        label=f"Tile #{tile_id:02d} [R{r_idx+1}:C{c_idx+1}] — {desc} (X:{x}, Y:{y})",
                        has_demo_cache=has_demo,
                    )
                )
                tile_id += 1

        return SceneGridResponse(
            scene_id=scene_id,
            scene_name=scene_name,
            dimensions=[h, w],
            total_tiles=len(tile_items),
            macro_preview_url=preview_url,
            tiles=tile_items,
        )


# Singleton scene service instance
scene_service = SceneService()
