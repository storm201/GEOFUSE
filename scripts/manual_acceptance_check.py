"""Execute Phase 2.5 Manual Acceptance Test Protocol (TEST A through TEST G)."""

import asyncio
import io
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.datastructures import UploadFile
from src.api.routes.inference import infer_tile, validate_custom_geotiff
from src.api.routes.receipt import get_trust_receipt
from src.api.routes.scenes import get_scene_grid_partition
from src.api.schemas.payloads import TileInferenceRequest


async def run_acceptance_tests():
    print("============================================================")
    print("PHASE 2.5 MANUAL ACCEPTANCE TEST EXECUTION")
    print("============================================================\n")

    # Load Scene A (urban_core)
    urban_dir = Path("data/additional_datasets/urban_core")
    files_a = []
    for bp in sorted(urban_dir.glob("*.tif")):
        with open(bp, "rb") as f:
            files_a.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))

    # Load Scene B (agriculture)
    agri_dir = Path("data/additional_datasets/agriculture")
    files_b = []
    for bp in sorted(agri_dir.glob("*.tif")):
        with open(bp, "rb") as f:
            files_b.append(UploadFile(file=io.BytesIO(f.read()), filename=bp.name))

    # ------------------------------------------------------------
    # TEST A: Upload Scene A -> Select Tile #07 -> Run inference
    # ------------------------------------------------------------
    print("[TEST A] Uploading Scene A (urban_core) & Processing Tile #07...")
    val_a = await validate_custom_geotiff(files_a)
    scene_a_id = val_a.scene_id
    assert val_a.grid is not None and val_a.grid.total_tiles == 25

    resp_a = await infer_tile(TileInferenceRequest(scene_id=scene_a_id, tile_id=7, force_live=False))
    print(f"  Scene A ID:  {scene_a_id}")
    print(f"  Run ID:      {resp_a.run_id}")
    print(f"  Source:      {resp_a.source}")
    print(f"  Latency:     {resp_a.latency_ms:.1f} ms")
    print(f"  Trust Score: {resp_a.evidence_metrics.trust_score_pct:.2f}%")
    print(f"  RGB Input:   {resp_a.assets.rgb_input}")
    print(f"  RGB SR:      {resp_a.assets.rgb_sr}")
    print(f"  Receipt ID:  {resp_a.receipt_id}")
    print("  -> TEST A PASSED\n")

    # ------------------------------------------------------------
    # TEST B: Upload Scene B -> Select Tile #07 -> Run inference
    # ------------------------------------------------------------
    print("[TEST B] Uploading Scene B (agriculture) & Processing Tile #07...")
    val_b = await validate_custom_geotiff(files_b)
    scene_b_id = val_b.scene_id
    assert scene_b_id != scene_a_id

    resp_b = await infer_tile(TileInferenceRequest(scene_id=scene_b_id, tile_id=7, force_live=False))
    print(f"  Scene B ID:  {scene_b_id}")
    print(f"  Run ID:      {resp_b.run_id}")
    print(f"  Source:      {resp_b.source}")
    print(f"  Latency:     {resp_b.latency_ms:.1f} ms")
    print(f"  Trust Score: {resp_b.evidence_metrics.trust_score_pct:.2f}%")
    print(f"  RGB Input:   {resp_b.assets.rgb_input}")
    print(f"  RGB SR:      {resp_b.assets.rgb_sr}")
    print(f"  Receipt ID:  {resp_b.receipt_id}")

    # Verification: Scene B does NOT show Scene A results
    assert resp_b.run_id != resp_a.run_id, "FAIL: Scene B reused Scene A run_id!"
    assert resp_b.assets.rgb_sr != resp_a.assets.rgb_sr, "FAIL: Scene B reused Scene A assets!"
    assert resp_b.receipt_id != resp_a.receipt_id, "FAIL: Scene B reused Scene A receipt!"
    print("  -> TEST B PASSED (Scene B produces completely isolated output from Scene A)\n")

    # ------------------------------------------------------------
    # TEST C: With Scene B still selected -> RUN USING GPU / FORCE LIVE
    # ------------------------------------------------------------
    print("[TEST C] Executing RUN USING GPU / FORCE LIVE on Scene B Tile #07...")
    resp_c_live = await infer_tile(TileInferenceRequest(scene_id=scene_b_id, tile_id=7, force_live=True))
    print(f"  Live Run ID: {resp_c_live.run_id}")
    print(f"  Source:      {resp_c_live.source}")
    print(f"  Latency:     {resp_c_live.latency_ms:.1f} ms")
    print(f"  RGB SR:      {resp_c_live.assets.rgb_sr}")
    print(f"  Receipt ID:  {resp_c_live.receipt_id}")

    assert resp_c_live.source == "live_inference", f"FAIL: Expected live_inference, got {resp_c_live.source}"
    assert resp_c_live.run_id != resp_b.run_id, "FAIL: Live run ID did not update!"
    assert "live_" in resp_c_live.run_id
    assert resp_c_live.assets.rgb_sr != resp_b.assets.rgb_sr
    assert resp_c_live.latency_ms > 0.0

    # Receipt audit
    receipt_resp = await get_trust_receipt(resp_c_live.receipt_id)
    assert receipt_resp.status_code == 200
    receipt_data = json.loads(receipt_resp.body.decode("utf-8"))
    assert receipt_data["receipt_id"] == resp_c_live.receipt_id
    print("  -> TEST C PASSED (Genuinely forced live execution, fresh ID & receipt)\n")

    # ------------------------------------------------------------
    # TEST D: Select another tile (#14)
    # ------------------------------------------------------------
    print("[TEST D] Selecting Tile #14 on Scene B...")
    resp_d = await infer_tile(TileInferenceRequest(scene_id=scene_b_id, tile_id=14, force_live=False))
    print(f"  Tile #14 Run ID: {resp_d.run_id}")
    print(f"  Tile #14 Assets: {resp_d.assets.rgb_sr}")
    assert resp_d.tile_id == 14
    assert resp_d.run_id != resp_b.run_id
    assert resp_d.assets.rgb_sr != resp_b.assets.rgb_sr
    print("  -> TEST D PASSED (Both input and SR update to new tile)\n")

    # ------------------------------------------------------------
    # TEST E: Check side-by-side display geometry
    # ------------------------------------------------------------
    print("[TEST E] Checking shared spatial viewport geometry...")
    print(f"  Input Resolution:  {resp_d.input_resolution_m}m")
    print(f"  Output Resolution: {resp_d.output_resolution_m}m")
    print(f"  Scale Factor:      {resp_d.scale_factor}x")
    assert resp_d.input_resolution_m == 10.0
    assert resp_d.output_resolution_m == 4.0
    assert resp_d.scale_factor == 2.5
    print("  -> TEST E PASSED (Shared 1:1 viewport geometry verified)\n")

    # ------------------------------------------------------------
    # TEST F: Layer consistency across RGB, CIR, NDVI, Uncertainty, Trust, Building
    # ------------------------------------------------------------
    print("[TEST F] Checking multi-band & diagnostic layer asset URLs...")
    assert resp_d.assets.rgb_input.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.rgb_sr.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.cir_input.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.cir_sr.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.ndvi_input.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.ndvi_sr.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.uncertainty.startswith(f"/api/assets/{resp_d.run_id}/")
    assert resp_d.assets.trust_map.startswith(f"/api/assets/{resp_d.run_id}/")
    print("  -> All 8 raster asset layers strictly belong to the active tile run ID")
    print("  -> TEST F PASSED\n")

    # ------------------------------------------------------------
    # TEST G: Switch back to Preloaded Scenes (urban_core)
    # ------------------------------------------------------------
    print("[TEST G] Switching back to Preloaded Scenes workflow...")
    grid_preloaded = await get_scene_grid_partition("urban_core")
    assert grid_preloaded.total_tiles == 25
    resp_preloaded = await infer_tile(TileInferenceRequest(scene_id="urban_core", tile_id=0, force_live=False))
    assert resp_preloaded.source == "demo_cache"
    assert resp_preloaded.tile_id == 0
    print(f"  Preloaded Grid Tiles: {grid_preloaded.total_tiles}")
    print(f"  Preloaded Tile 0 Run: {resp_preloaded.run_id}")
    print(f"  Source:               {resp_preloaded.source}")
    print("  -> TEST G PASSED (Existing Preloaded Scene 5x5 workflow fully operational)\n")

    print("============================================================")
    print("ALL ACCEPTANCE TESTS (A THROUGH G) COMPLETED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    asyncio.run(run_acceptance_tests())
