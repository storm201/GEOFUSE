"""Unit tests for Interactive Tile Selection & Exploration Feature."""

from pathlib import Path
import pytest
import numpy as np

from src.utils.scene_visualizer import get_tile_metadata_grid
from src.dashboard.app import (
    run_cached_pipeline,
    run_cached_crop_pipeline,
    load_cached_scene,
)
from scripts.generate_trust_receipts import main as generate_receipts_main
from scripts.run_custom_input import load_user_image, run_custom_pipeline
from src.utils.config import get_project_root


def test_get_tile_metadata_grid():
    """Verify get_tile_metadata_grid produces 25 correctly mapped tiles for a 512x512 scene."""
    tiles = get_tile_metadata_grid(h=512, w=512, patch_size=128, stride=96)
    assert len(tiles) == 25

    # Check first and last tiles
    t0 = tiles[0]
    assert t0["tile_id"] == 0
    assert t0["row"] == 1
    assert t0["col"] == 1
    assert t0["x"] == 0
    assert t0["y"] == 0
    assert t0["w"] == 128
    assert t0["h"] == 128
    assert "NW Sector" in t0["description"] or "Top-Left" in t0["description"]

    t24 = tiles[24]
    assert t24["tile_id"] == 24
    assert t24["row"] == 5
    assert t24["col"] == 5
    assert t24["x"] == 384
    assert t24["y"] == 384
    assert "Hold-out" in t24["description"] or "SE Quadrant" in t24["description"]

    # Verify tile 12 (Center Core)
    t12 = tiles[12]
    assert t12["tile_id"] == 12
    assert t12["row"] == 3
    assert t12["col"] == 3
    assert t12["x"] == 192
    assert t12["y"] == 192


def test_run_cached_pipeline_arbitrary_tile():
    """Verify run_cached_pipeline executes live and returns complete bundle on non-demo tiles."""
    # Tile #12 is the center core (not in the 4 offline precomputed demo cache tiles)
    data = run_cached_pipeline(12)
    assert data is not None
    assert "hr_tile" in data
    assert "sr_tile" in data
    assert "fusion_result" in data
    assert "receipt" in data
    assert data["hr_tile"].shape == (128, 128, 4)
    assert data["sr_tile"].shape == (128, 128, 4)
    assert 0.0 <= data["fusion_result"]["trust_score_pct"] <= 100.0


def test_run_cached_crop_pipeline_coordinates():
    """Verify run_cached_crop_pipeline extracts custom coordinate bounding box."""
    data = run_cached_crop_pipeline(crop_x=150, crop_y=120)
    assert data is not None
    assert "hr_tile" in data
    assert "crop_box" in data
    assert data["crop_box"]["x"] == 150
    assert data["crop_box"]["y"] == 120
    assert data["crop_box"]["w"] == 128
    assert data["crop_box"]["h"] == 128
    assert data["hr_tile"].shape == (128, 128, 4)
    assert data["sr_tile"].shape == (128, 128, 4)


def test_cli_generate_trust_receipt_single_tile(tmp_path):
    """Verify generate_trust_receipts.py CLI accepts --tile argument for a single tile."""
    ret = generate_receipts_main(["--tile", "12"])
    assert ret == 0

    root = get_project_root()
    receipt_file = root / "outputs" / "receipts" / "trust_receipt_tile_12.json"
    assert receipt_file.exists()


def test_custom_input_tile_and_crop_extraction():
    """Verify load_user_image and run_custom_pipeline with tile_idx and crop_coords."""
    root = get_project_root()
    urban_dir = root / "data" / "additional_datasets" / "urban_core"
    assert urban_dir.is_dir()

    # 1. Test tile_idx extraction
    arr_t5, meta_t5 = load_user_image(urban_dir, target_size=128, tile_idx=5)
    assert arr_t5.shape == (128, 128, 4)
    assert meta_t5["crop_box"]["x"] == 0
    assert meta_t5["crop_box"]["y"] == 96
    assert meta_t5.get("tile_idx") == 5

    # 2. Test custom crop_coords extraction
    arr_c, meta_c = load_user_image(urban_dir, target_size=128, crop_coords=(120, 80))
    assert arr_c.shape == (128, 128, 4)
    assert meta_c["crop_box"]["x"] == 120
    assert meta_c["crop_box"]["y"] == 80
