"""Automated Tests for GeoFUSE SentinelGuard Tile Insight & Grounded Explainability Layer.

Validates:
1. TileStatistics schema and integration into TileInferenceResponse
2. Tile-specific spectral metrics (NDVI, vegetation %, built %, building counts) vary authentically across tiles
3. No hardcoded tile descriptions or canned tile-to-text lookup tables
4. Zero geographic fabrication for unknown user uploads
5. Color guides strictly match backend OpenCV colormaps (SUMMER, MAGMA, TURBO, tactical cyan)
6. Graceful handling of missing or partial telemetry
7. Distinct explanations across all 6 diagnostic visualization modes
"""

import asyncio
from pathlib import Path
import numpy as np
import pytest

from src.api.schemas.payloads import (
    EvidenceMetricsSummary,
    InferenceAssetUrls,
    TileInferenceRequest,
    TileInferenceResponse,
    TileStatistics,
)
from src.api.services.cache_service import cache_service
from src.api.services.inference_service import inference_service


def test_tile_statistics_schema():
    """Verify TileStatistics schema fields and validation constraints."""
    stats = TileStatistics(
        mean_ndvi=0.366,
        vegetation_pct=61.9,
        water_pct=0.3,
        built_pct=37.7,
        building_count=19,
        dominant_feature="Dense vegetation canopy",
    )
    assert stats.mean_ndvi == 0.366
    assert stats.vegetation_pct == 61.9
    assert stats.water_pct == 0.3
    assert stats.built_pct == 37.7
    assert stats.building_count == 19
    assert "vegetation" in stats.dominant_feature.lower()

    # Serialization test
    dumped = stats.model_dump()
    assert dumped["building_count"] == 19
    loaded = TileStatistics(**dumped)
    assert loaded == stats


def test_tile_inference_response_accepts_optional_stats():
    """Verify TileInferenceResponse accepts both populated and None tile_stats for backward compatibility."""
    assets = InferenceAssetUrls(
        rgb_input="/api/assets/test/rgb_in.webp",
        rgb_sr="/api/assets/test/rgb_sr.webp",
        cir_input="/api/assets/test/cir_in.webp",
        cir_sr="/api/assets/test/cir_sr.webp",
        ndvi_input="/api/assets/test/ndvi_in.webp",
        ndvi_sr="/api/assets/test/ndvi_sr.webp",
        uncertainty="/api/assets/test/uncertainty.webp",
        trust_map="/api/assets/test/trust.webp",
        building_mask=None,
    )
    metrics = EvidenceMetricsSummary(
        trust_score_pct=91.8,
        is_trusted=True,
        status_label="NOMINAL_HIGH_TRUST",
        mean_disagreement=0.0012,
        delta_ndvi_mean=0.015,
        pct_spectral_consistent=96.5,
        edge_iou=0.45,
        gradient_correlation=0.82,
        warnings=[],
    )

    # 1. Without tile_stats (legacy cache compatibility)
    resp_none = TileInferenceResponse(
        run_id="test_run_none",
        scene_id="urban_core",
        tile_id=0,
        source="demo_cache",
        latency_ms=1.5,
        assets=assets,
        evidence_metrics=metrics,
        receipt_id="TR-TEST-001",
        receipt_url="/api/receipt/TR-TEST-001",
        tile_stats=None,
    )
    assert resp_none.tile_stats is None

    # 2. With authentic tile_stats
    stats = TileStatistics(
        mean_ndvi=0.42,
        vegetation_pct=70.0,
        water_pct=0.0,
        built_pct=30.0,
        building_count=15,
        dominant_feature="Dense vegetation canopy",
    )
    resp_with_stats = TileInferenceResponse(
        run_id="test_run_stats",
        scene_id="urban_core",
        tile_id=0,
        source="demo_cache",
        latency_ms=1.5,
        assets=assets,
        evidence_metrics=metrics,
        receipt_id="TR-TEST-002",
        receipt_url="/api/receipt/TR-TEST-002",
        tile_stats=stats,
    )
    assert resp_with_stats.tile_stats is not None
    assert resp_with_stats.tile_stats.building_count == 15


@pytest.mark.anyio
async def test_tile_stats_vary_across_scene_tiles():
    """Verify that different tiles return distinct, authentic computed statistics rather than identical canned values."""
    # Test preloaded demo tiles
    req_0 = TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False)
    req_8 = TileInferenceRequest(scene_id="urban_core", tile_id=8, force_live=False)

    resp_0 = await inference_service.process_tile_inference(req_0)
    resp_8 = await inference_service.process_tile_inference(req_8)

    assert resp_0.tile_stats is not None, "Tile #00 must contain authentic tile_stats"
    assert resp_8.tile_stats is not None, "Tile #08 must contain authentic tile_stats"

    # Verify tile 0 and tile 8 have distinct physical NDVI and built-up values
    assert resp_0.tile_stats.mean_ndvi != resp_8.tile_stats.mean_ndvi
    assert resp_0.tile_stats.vegetation_pct != resp_8.tile_stats.vegetation_pct
    assert resp_0.tile_stats.built_pct != resp_8.tile_stats.built_pct


def test_no_hardcoded_tile_lookup_in_frontend():
    """Perform a code audit on TileInsight.jsx to verify zero hardcoded tile-to-description dictionaries."""
    insight_file = Path("frontend/src/components/TileInsight.jsx")
    assert insight_file.exists(), "TileInsight.jsx must exist"
    code = insight_file.read_text(encoding="utf-8")

    # Forbidden hardcoded patterns:
    # e.g.: tile_00 = "urban", case 0: return "urban", etc.
    forbidden_patterns = [
        'case 0: return "urban"',
        'case 7: return',
        'Tile #00: "urban',
        'Tile #01: "vegetation',
        'tile_descriptions = {',
        'const tileDescriptions = {',
    ]
    for pattern in forbidden_patterns:
        assert pattern not in code, f"Found forbidden hardcoded tile description pattern: {pattern}"


def test_no_geographic_hallucination_for_custom_uploads():
    """Verify that custom uploaded scenes do not inject fabricated city/neighborhood names."""
    insight_file = Path("frontend/src/components/TileInsight.jsx")
    code = insight_file.read_text(encoding="utf-8")

    # In custom scene branch, it must use strictly generic, evidence-grounded phrasing
    assert 'scene_id.startsWith("custom_")' in code or "isCustom" in code
    assert "User-Uploaded Sentinel-2" in code

    # Verify no fabricated city names are present as fixed custom claims
    assert "residential district in Bengaluru" not in code
    assert "downtown Mumbai" not in code
    assert "commercial hub of New Delhi" not in code


def test_color_guides_match_backend_colormaps():
    """Verify that TileInsight color guides match OpenCV colormaps used in cache_service.py."""
    insight_file = Path("frontend/src/components/TileInsight.jsx")
    code = insight_file.read_text(encoding="utf-8")

    # 1. NDVI must reference SUMMER colormap (dark teal to bright yellow)
    assert "summer" in code.lower()
    assert "#008066" in code  # Teal (low NDVI in COLORMAP_SUMMER)
    assert "#ffff66" in code  # Yellow (high NDVI in COLORMAP_SUMMER)

    # 2. Uncertainty must reference MAGMA colormap (black to pale yellow)
    assert "magma" in code.lower()
    assert "#000004" in code  # Black (min in COLORMAP_MAGMA)
    assert "#fcfdbf" in code  # Pale yellow (max in COLORMAP_MAGMA)

    # 3. Trust / Risk must reference TURBO colormap (blue to red)
    assert "turbo" in code.lower()
    assert "#30123b" in code  # Dark blue (min in COLORMAP_TURBO)
    assert "#7a0403" in code  # Deep red (max in COLORMAP_TURBO)

    # 4. Building footprints must reference bright tactical cyan outline (RGB 0, 240, 255)
    assert "#00f0ff" in code or "(0, 240, 255)" in code


def test_scientific_integrity_rules_respected():
    """Verify that uncertainty is never called 'error' and trust is never called 'accuracy'."""
    insight_file = Path("frontend/src/components/TileInsight.jsx")
    code = insight_file.read_text(encoding="utf-8")

    # Uncertainty must explicitly clarify it is model disagreement, NOT measured error
    assert "Uncertainty is model disagreement, NOT measured error" in code

    # Trust must explicitly clarify it is an empirical heuristic, NOT a guarantee of ground-truth perfection
    assert "empirical multi-criteria heuristic" in code
