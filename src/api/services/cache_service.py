"""Cache & Asset Service — Deterministic Raster Asset Persistence and Retrieval.

Saves processed tile rasters (RGB, CIR, NDVI, Uncertainty, Trust, Building Masks)
to disk under outputs/cache_api/{run_id}/ as lightweight WebP/PNG assets served
directly via static HTTP URLs (/api/assets/...), eliminating giant Base64 payloads.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from src.api.config import (
    CACHE_API_DIR,
    CORE_CONFIG,
    DEMO_CACHE_DIR,
    OUTPUTS_DIR,
)
from src.api.schemas.payloads import (
    EvidenceMetricsSummary,
    InferenceAssetUrls,
    TileInferenceResponse,
    TileStatistics,
)
from src.inference.real_inference import colorize_ndvi
from src.utils.scene_visualizer import to_display_rgb


def _colorize_single_channel(arr: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Colorize a 2D float or uint8 array using an OpenCV colormap."""
    if arr is None or arr.size == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    arr_clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    p_min, p_max = float(np.min(arr_clean)), float(np.max(arr_clean))
    if p_max > p_min:
        norm = np.clip((arr_clean - p_min) / (p_max - p_min), 0.0, 1.0)
    else:
        norm = np.zeros_like(arr_clean)

    u8 = (norm * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(u8, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _render_building_overlay(base_rgb: np.ndarray, mask: Any) -> np.ndarray:
    """Render bright cyan building footprint outlines onto an RGB image."""
    overlay = base_rgb.copy()
    if mask is None:
        return overlay
    if isinstance(mask, dict):
        mask = mask.get("mask")
    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return overlay

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if mask_u8.shape[:2] != overlay.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Bright tactical cyan contour outline: RGB (0, 240, 255)
    cv2.drawContours(overlay, contours, -1, (0, 240, 255), 1, lineType=cv2.LINE_AA)
    return overlay


class CacheService:
    """Manages deterministic asset caching, demo bundle unpacking, and metadata persistence."""

    def __init__(self):
        self.base_dir = CACHE_API_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_run_id(self, scene_id: str, tile_id: int, scale_factor: float = 2.5) -> str:
        """Create a human-readable deterministic run key."""
        return f"run_{scene_id}_tile{tile_id:02d}_s{int(scale_factor*10)}"

    def get_run_dir(self, run_id: str) -> Path:
        """Return directory path for a given run ID."""
        return self.base_dir / run_id

    def has_cached_run(self, run_id: str) -> bool:
        """Check if complete cached assets and metadata exist on disk."""
        meta_file = self.get_run_dir(run_id) / "meta.json"
        return meta_file.exists()

    def load_cached_run(self, run_id: str) -> Optional[TileInferenceResponse]:
        """Load and deserialize a cached TileInferenceResponse from disk."""
        run_dir = self.get_run_dir(run_id)
        meta_file = run_dir / "meta.json"
        if not meta_file.exists():
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            resp = TileInferenceResponse(**data)

            # If tile_stats is missing in older cache, compute dynamically from cached assets
            if resp.tile_stats is None:
                cir_file = run_dir / "cir_sr.webp"
                bldg_file = run_dir / "building_mask.webp"
                if cir_file.exists():
                    try:
                        cir_img = np.array(Image.open(cir_file).convert("RGB"))
                        # In False-Color CIR: R=NIR (B08), G=Red (B04), B=Green (B03)
                        nir = cir_img[:, :, 0].astype(np.float32) / 255.0
                        red = cir_img[:, :, 1].astype(np.float32) / 255.0
                        ndvi_raw = (nir - red) / (nir + red + 1e-8)
                        mean_ndvi = float(np.mean(ndvi_raw))
                        veg_pct = float(np.mean(ndvi_raw > 0.25) * 100.0)
                        water_pct = float(np.mean((ndvi_raw < 0.0) & (nir < 0.15)) * 100.0)
                        built_pct = float(np.mean((ndvi_raw >= -0.05) & (ndvi_raw <= 0.25) & (red > 0.1)) * 100.0)

                        bldg_cnt = 0
                        if bldg_file.exists() and resp.assets.building_mask is not None:
                            bldg_img = np.array(Image.open(bldg_file).convert("RGB"))
                            cyan_mask = (bldg_img[:, :, 0] < 50) & (bldg_img[:, :, 1] > 200) & (bldg_img[:, :, 2] > 200)
                            if np.any(cyan_mask):
                                cnts, _ = cv2.findContours(cyan_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                bldg_cnt = len(cnts)

                        if water_pct > 30.0:
                            dominant = "Water surface / aquatic body"
                        elif veg_pct > 55.0:
                            dominant = "Dense vegetation canopy"
                        elif veg_pct > 25.0 and built_pct > 25.0:
                            dominant = "Mixed vegetation and built structures"
                        elif built_pct > 35.0:
                            dominant = "Predominantly built-up / impervious surface"
                        elif veg_pct > 20.0:
                            dominant = "Moderate vegetation with open terrain"
                        else:
                            dominant = "Open terrain / mixed ground cover"

                        resp.tile_stats = TileStatistics(
                            mean_ndvi=round(mean_ndvi, 3),
                            vegetation_pct=round(veg_pct, 1),
                            water_pct=round(water_pct, 1),
                            built_pct=round(built_pct, 1),
                            building_count=bldg_cnt,
                            dominant_feature=dominant,
                        )
                        # Persist back into meta.json so it's cached for subsequent loads
                        try:
                            with open(meta_file, "w", encoding="utf-8") as wf:
                                json.dump(resp.model_dump(), wf, indent=2)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[Warning] Failed to dynamically derive tile_stats for {run_id}: {e}")

            return resp
        except Exception:
            return None

    def unpack_demo_cache(self, tile_id: int, scene_id: str = "urban_core") -> Optional[TileInferenceResponse]:
        """Check and unpack precomputed demo cache bundle from outputs/demo_cache/."""
        demo_bundle_path = DEMO_CACHE_DIR / f"demo_tile_{tile_id}.pkl"
        demo_receipt_path = DEMO_CACHE_DIR / f"demo_tile_{tile_id}_receipt.json"

        if not demo_bundle_path.exists():
            return None

        run_id = f"demo_{scene_id}_tile{tile_id:02d}"
        run_dir = self.get_run_dir(run_id)

        # If already unpacked into cache_api, return directly
        if (run_dir / "meta.json").exists():
            return self.load_cached_run(run_id)

        try:
            with open(demo_bundle_path, "rb") as f:
                data = pickle.load(f)

            receipt = {}
            if demo_receipt_path.exists():
                with open(demo_receipt_path, "r", encoding="utf-8") as rf:
                    receipt = json.load(rf)
            elif "receipt" in data:
                receipt = data["receipt"]

            hr_tile = data.get("hr_tile")
            sr_tile = data.get("sr_tile")
            bic_tile = data.get("bicubic_tile")
            disag = data.get("disagreement_map")
            fusion = data.get("fusion_result", {})
            foot_sr = data.get("foot_sr")

            return self.save_inference_products(
                run_id=run_id,
                scene_id=scene_id,
                tile_id=tile_id,
                source="demo_cache",
                latency_ms=0.0,
                input_10m=hr_tile,
                sr_4m=sr_tile,
                disagreement_map=disag,
                fusion_result=fusion,
                foot_mask_sr=foot_sr,
                receipt=receipt,
            )
        except Exception as e:
            print(f"[Warning] Failed to unpack demo bundle for tile {tile_id}: {e}")
            return None

    def save_inference_products(
        self,
        run_id: str,
        scene_id: str,
        tile_id: int,
        source: str,
        latency_ms: float,
        input_10m: np.ndarray,
        sr_4m: np.ndarray,
        disagreement_map: Optional[np.ndarray],
        fusion_result: Dict[str, Any],
        foot_mask_sr: Optional[np.ndarray],
        receipt: Dict[str, Any],
    ) -> TileInferenceResponse:
        """Persist raster assets, receipt, and metadata JSON for a run."""
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        asset_urls = {}

        # 1. Natural RGB (B4, B3, B2)
        rgb_in = to_display_rgb(input_10m, false_color=False)
        rgb_sr = to_display_rgb(sr_4m, false_color=False)
        Image.fromarray(rgb_in).save(run_dir / "rgb_input.webp", "WEBP", quality=92)
        Image.fromarray(rgb_sr).save(run_dir / "rgb_sr.webp", "WEBP", quality=92)
        asset_urls["rgb_input"] = f"/api/assets/{run_id}/rgb_input.webp"
        asset_urls["rgb_sr"] = f"/api/assets/{run_id}/rgb_sr.webp"

        # 2. False-Color Infrared CIR (B8, B4, B3)
        cir_in = to_display_rgb(input_10m, false_color=True)
        cir_sr = to_display_rgb(sr_4m, false_color=True)
        Image.fromarray(cir_in).save(run_dir / "cir_input.webp", "WEBP", quality=92)
        Image.fromarray(cir_sr).save(run_dir / "cir_sr.webp", "WEBP", quality=92)
        asset_urls["cir_input"] = f"/api/assets/{run_id}/cir_input.webp"
        asset_urls["cir_sr"] = f"/api/assets/{run_id}/cir_sr.webp"

        # 3. NDVI Vegetation Colormaps
        # Calculate NDVI: (NIR - Red) / (NIR + Red + 1e-8) -> NIR=idx 3, Red=idx 2
        ndvi_in_raw = (input_10m[:, :, 3] - input_10m[:, :, 2]) / (input_10m[:, :, 3] + input_10m[:, :, 2] + 1e-8)
        ndvi_sr_raw = (sr_4m[:, :, 3] - sr_4m[:, :, 2]) / (sr_4m[:, :, 3] + sr_4m[:, :, 2] + 1e-8)
        ndvi_in_disp = colorize_ndvi(ndvi_in_raw)
        ndvi_sr_disp = colorize_ndvi(ndvi_sr_raw)
        Image.fromarray(ndvi_in_disp).save(run_dir / "ndvi_input.webp", "WEBP", quality=92)
        Image.fromarray(ndvi_sr_disp).save(run_dir / "ndvi_sr.webp", "WEBP", quality=92)
        asset_urls["ndvi_input"] = f"/api/assets/{run_id}/ndvi_input.webp"
        asset_urls["ndvi_sr"] = f"/api/assets/{run_id}/ndvi_sr.webp"

        # 4. Uncertainty Map (Ensemble Standard Deviation)
        if disagreement_map is not None:
            disag_disp = _colorize_single_channel(disagreement_map, cv2.COLORMAP_MAGMA)
        else:
            disag_disp = np.zeros_like(rgb_sr)
        Image.fromarray(disag_disp).save(run_dir / "uncertainty.webp", "WEBP", quality=90)
        asset_urls["uncertainty"] = f"/api/assets/{run_id}/uncertainty.webp"

        # 5. Composite Trust / Risk Map
        risk_map = fusion_result.get("risk_map")
        if risk_map is not None:
            risk_disp = _colorize_single_channel(risk_map, cv2.COLORMAP_TURBO)
        else:
            risk_disp = np.zeros_like(rgb_sr)
        Image.fromarray(risk_disp).save(run_dir / "trust_map.webp", "WEBP", quality=90)
        asset_urls["trust_map"] = f"/api/assets/{run_id}/trust_map.webp"

        # 6. Downstream Building Footprint Overlay
        if foot_mask_sr is not None:
            building_overlay = _render_building_overlay(rgb_sr, foot_mask_sr)
            Image.fromarray(building_overlay).save(run_dir / "building_mask.webp", "WEBP", quality=90)
            asset_urls["building_mask"] = f"/api/assets/{run_id}/building_mask.webp"
        else:
            asset_urls["building_mask"] = None

        # 7. Save Trust Receipt JSON
        receipt_path = run_dir / "receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as rf:
            json.dump(receipt, rf, indent=2)

        receipt_id = receipt.get("receipt_id", f"TR-{run_id}")

        # 8. Extract clean evidence metrics (strictly authentic, zero hardcoded production results)
        comp_stats = fusion_result.get("component_stats", {})
        trust_score_pct = float(fusion_result.get("trust_score_pct", 0.0))
        if trust_score_pct == 0.0 and "mean_trust_score" in fusion_result:
            trust_score_pct = float(fusion_result["mean_trust_score"]) * 100.0

        is_trusted = trust_score_pct >= float(CORE_CONFIG.get("downstream", {}).get("trust_partition_threshold", 0.85) * 100.0)

        spec = comp_stats.get("spectral", {})
        edge = comp_stats.get("structural", {})
        disag_stat = comp_stats.get("disagreement", {})

        trust_eval = receipt.get("trust_evaluation", {})
        warnings = trust_eval.get("warnings_and_advisories", [])

        mean_disag = float(disag_stat.get("raw_mean", np.mean(disagreement_map) if disagreement_map is not None else 0.0))
        delta_ndvi_val = float(spec.get("mean_delta_ndvi", 0.0))
        pct_consistent = float(100.0 - spec.get("pct_inconsistent_pixels", 0.0))
        edge_iou_val = float(edge.get("edge_iou", 0.0))
        grad_corr_val = float(edge.get("gradient_correlation_r", 0.0))

        metrics = EvidenceMetricsSummary(
            trust_score_pct=round(trust_score_pct, 2),
            is_trusted=is_trusted,
            status_label="NOMINAL_HIGH_TRUST" if is_trusted else "WARNING_LOW_TRUST",
            mean_disagreement=round(mean_disag, 4),
            delta_ndvi_mean=round(delta_ndvi_val, 4),
            pct_spectral_consistent=round(pct_consistent, 1),
            edge_iou=round(edge_iou_val, 4),
            gradient_correlation=round(grad_corr_val, 4),
            warnings=warnings,
        )

        # 8b. Compute authentic tile-level spectral distribution and feature proxies from sr_4m
        # Band index 3: NIR, Band index 2: Red, Band index 1: Green, Band index 0: Blue
        ndvi_arr = (sr_4m[:, :, 3] - sr_4m[:, :, 2]) / (sr_4m[:, :, 3] + sr_4m[:, :, 2] + 1e-8)
        mean_ndvi_val = float(np.mean(ndvi_arr))
        veg_pct_val = float(np.mean(ndvi_arr > 0.25) * 100.0)
        water_pct_val = float(np.mean((ndvi_arr < 0.0) & (sr_4m[:, :, 3] < 0.12)) * 100.0)
        built_pct_val = float(np.mean((ndvi_arr >= -0.05) & (ndvi_arr <= 0.28) & (sr_4m[:, :, 0] > 0.06)) * 100.0)

        bldg_count = 0
        if foot_mask_sr is not None:
            f_mask = foot_mask_sr.get("mask") if isinstance(foot_mask_sr, dict) else foot_mask_sr
            if f_mask is not None and isinstance(f_mask, np.ndarray) and f_mask.size > 0:
                f_u8 = (f_mask > 0).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(f_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bldg_count = len(cnts)

        if water_pct_val > 30.0:
            dominant = "Water surface / aquatic body"
        elif veg_pct_val > 55.0:
            dominant = "Dense vegetation canopy"
        elif veg_pct_val > 25.0 and built_pct_val > 25.0:
            dominant = "Mixed vegetation and built structures"
        elif built_pct_val > 35.0:
            dominant = "Predominantly built-up / impervious surface"
        elif veg_pct_val > 20.0:
            dominant = "Moderate vegetation with open terrain"
        else:
            dominant = "Open terrain / mixed ground cover"

        stats = TileStatistics(
            mean_ndvi=round(mean_ndvi_val, 3),
            vegetation_pct=round(veg_pct_val, 1),
            water_pct=round(water_pct_val, 1),
            built_pct=round(built_pct_val, 1),
            building_count=bldg_count,
            dominant_feature=dominant,
        )

        response = TileInferenceResponse(
            run_id=run_id,
            scene_id=scene_id,
            tile_id=tile_id,
            source=source,
            latency_ms=round(latency_ms, 2),
            input_resolution_m=10.0,
            output_resolution_m=4.0,
            scale_factor=2.5,
            assets=InferenceAssetUrls(**asset_urls),
            evidence_metrics=metrics,
            receipt_id=receipt_id,
            receipt_url=f"/api/receipt/{receipt_id}",
            tile_stats=stats,
        )

        # 9. Save response metadata for future cache hits
        with open(run_dir / "meta.json", "w", encoding="utf-8") as mf:
            json.dump(response.model_dump(), mf, indent=2)

        return response


# Singleton cache service instance
cache_service = CacheService()
