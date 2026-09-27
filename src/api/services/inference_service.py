"""Inference Service — Direct Sentinel-2 10m -> 4m Super-Resolution Orchestrator.

Orchestrates tile extraction, GPU ensemble forward pass (zero synthetic degradation),
empirical trust verification, building footprint analysis, and receipt generation.
Wraps the battle-tested logic in src.inference.real_inference and src.evaluation.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.api.config import CORE_CONFIG
from src.api.schemas.payloads import TileInferenceRequest, TileInferenceResponse
from src.api.services.cache_service import cache_service
from src.api.services.model_service import model_service
from src.api.services.scene_service import scene_service
from src.evaluation.downstream_eval import extract_building_footprints
from src.inference.real_inference import (
    compute_real_inference_trust,
    generate_real_trust_receipt,
    load_and_preprocess_sentinel2,
    run_direct_sr_tile,
    validate_sentinel2_input,
)
from src.utils.scene_visualizer import get_tile_metadata_grid


class InferenceService:
    """Coordinates cached and live GPU super-resolution inference on Sentinel-2 tiles."""

    async def process_tile_inference(self, request: TileInferenceRequest) -> TileInferenceResponse:
        """Execute or retrieve cached super-resolution products for a requested tile."""
        scene_id = request.scene_id
        tile_id = request.tile_id
        force_live = request.force_live
        is_custom = scene_id.startswith("custom_")

        # Validate tile index
        if tile_id < 0 or tile_id > 24:
            raise ValueError(f"Tile index {tile_id} out of bounds (must be 0..24).")

        scale_factor = float(CORE_CONFIG.get("model", {}).get("scale_factor", 2.5))

        # 1. Determine Run ID: Fresh timestamped identity if force_live, else deterministic cache ID
        if force_live:
            timestamp_ms = int(time.time() * 1000)
            run_id = f"live_{scene_id}_tile{tile_id:02d}_{timestamp_ms}"
        else:
            run_id = cache_service.generate_run_id(scene_id, tile_id, scale_factor)

        # 2. Check Offline Demo Cache first (if not forced live and not a custom upload)
        if not force_live and not is_custom and scene_id in ("urban_core", "primary_raw") and tile_id in (0, 8, 16, 24):
            demo_resp = cache_service.unpack_demo_cache(tile_id, scene_id)
            if demo_resp is not None:
                return demo_resp

        # 3. Check API Cache (if not forced live)
        if not force_live and cache_service.has_cached_run(run_id):
            cached_resp = cache_service.load_cached_run(run_id)
            if cached_resp is not None:
                cached_resp.source = "user_input_cache" if is_custom else "api_cache"
                return cached_resp

        # 4. Live GPU/Compute Forward Pass (Cache Miss or Forced Live)
        start_time = time.perf_counter()

        # Load scene multi-band stack and resolve grid coordinates
        scene_stack = scene_service.load_scene_stack(scene_id)
        grid_resp = scene_service.get_scene_grid(scene_id)

        if tile_id >= len(grid_resp.tiles):
            raise ValueError(f"Tile index {tile_id} not available in scene grid (total: {len(grid_resp.tiles)}).")

        t_info = grid_resp.tiles[tile_id]
        sx = int(t_info.x)
        sy = int(t_info.y)
        ps = int(t_info.patch_size)

        sub_10m = scene_stack[sy : sy + ps, sx : sx + ps, :]

        # Acquire GPU inference slot to prevent concurrent VRAM over-allocation
        async with model_service.acquire_gpu_slot():
            # Offload heavy synchronous PyTorch execution, verification, and WebP encoding
            # to a worker thread so the FastAPI asyncio event loop remains unblocked and responsive!
            return await asyncio.to_thread(
                self._sync_execute_tile_inference,
                scene_id=scene_id,
                tile_id=tile_id,
                run_id=run_id,
                sub_10m=sub_10m,
                force_live=force_live,
                is_custom=is_custom,
                start_time=start_time,
            )

    def _sync_execute_tile_inference(
        self,
        scene_id: str,
        tile_id: int,
        run_id: str,
        sub_10m: np.ndarray,
        force_live: bool,
        is_custom: bool,
        start_time: float,
    ) -> TileInferenceResponse:
        """Synchronously execute PyTorch forward pass, trust verification, building extraction, and WebP generation on worker thread."""
        models = model_service.get_models()
        device = model_service.get_device()

        with torch.inference_mode():
            # Direct 2.5x learned SR (ZERO synthetic degradation)
            sr_4m, disag_map = run_direct_sr_tile(models, sub_10m, device=device)

        # Empirical Trust Evaluation (no fake ground truth)
        trust_data = compute_real_inference_trust(
            sr_4m=sr_4m,
            lr_10m=sub_10m,
            disagreement_map=disag_map,
            models=models,
            device=device,
            config=CORE_CONFIG,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Downstream building footprint extraction
        morph_cfg = CORE_CONFIG.get("downstream", {}).get("morphology", {})
        foot_sr = extract_building_footprints(
            sr_4m,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=2,
            nir_idx=3,
        )

        # Build auditable trust receipt
        scene_path = scene_service.get_scene_path(scene_id)
        try:
            val_info = validate_sentinel2_input(scene_path)
        except Exception:
            val_info = {"platform": "Sentinel-2", "native_gsd_meters": 10.0}

        dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        receipt = generate_real_trust_receipt(
            scene_name=f"{scene_id}_tile{tile_id:02d}",
            val_info=val_info,
            trust_data=trust_data,
            config=CORE_CONFIG,
            device_name=dev_name,
        )

        # Explicitly assign truthful source label
        if force_live:
            source_label = "live_inference"
        elif is_custom:
            source_label = "user_input"
        else:
            source_label = "live_inference"

        # Save all raster visual assets & metadata
        return cache_service.save_inference_products(
            run_id=run_id,
            scene_id=scene_id,
            tile_id=tile_id,
            source=source_label,
            latency_ms=latency_ms,
            input_10m=sub_10m,
            sr_4m=sr_4m,
            disagreement_map=disag_map,
            fusion_result=trust_data["fusion_result"],
            foot_mask_sr=foot_sr,
            receipt=receipt,
        )

    async def process_custom_inference(
        self,
        saved_files: List[Path],
        tile_size: int = 128,
        crop_coords: Optional[Tuple[int, int]] = None,
        force_live: bool = False,
    ) -> TileInferenceResponse:
        """Execute end-to-end direct 2.5x super-resolution and verification on user-uploaded Sentinel-2 files."""
        import hashlib
        start_time = time.perf_counter()

        # 1. Compute deterministic hash from files
        h = hashlib.sha256()
        for p in sorted(saved_files, key=lambda x: x.name):
            h.update(p.name.encode("utf-8"))
            h.update(str(p.stat().st_size).encode("utf-8"))
        upload_hash = h.hexdigest()[:12]

        # 2. Validate Sentinel-2 input structure
        upload_source = saved_files[0] if (len(saved_files) == 1 and saved_files[0].is_file()) else saved_files[0].parent
        val_info = validate_sentinel2_input(upload_source)

        # 3. Load & preprocess multi-band reflectance
        stack_10m, _ = load_and_preprocess_sentinel2(upload_source)
        h0, w0 = stack_10m.shape[:2]

        # 4. Extract target sub-tile (center crop by default, or specific coords)
        if h0 <= tile_size and w0 <= tile_size:
            sub_10m = stack_10m
            sx, sy = 0, 0
        elif crop_coords is not None:
            cx, cy = crop_coords
            sx = max(0, min(int(cx), max(0, w0 - tile_size)))
            sy = max(0, min(int(cy), max(0, h0 - tile_size)))
            sub_10m = stack_10m[sy : sy + tile_size, sx : sx + tile_size, :]
        else:
            sx = max(0, (w0 - tile_size) // 2)
            sy = max(0, (h0 - tile_size) // 2)
            sub_10m = stack_10m[sy : sy + tile_size, sx : sx + tile_size, :]

        run_id = f"custom_{upload_hash}_{sx}_{sy}"

        # 5. Check if already processed in API cache
        if not force_live and cache_service.has_cached_run(run_id):
            cached = cache_service.load_cached_run(run_id)
            if cached is not None:
                cached.source = "user_input_cache"
                return cached

        # 6. Acquire GPU inference slot and execute forward pass off the event loop
        async with model_service.acquire_gpu_slot():
            return await asyncio.to_thread(
                self._sync_execute_custom_inference,
                sub_10m=sub_10m,
                upload_hash=upload_hash,
                val_info=val_info,
                run_id=run_id,
                start_time=start_time,
            )

    def _sync_execute_custom_inference(
        self,
        sub_10m: np.ndarray,
        upload_hash: str,
        val_info: Dict[str, Any],
        run_id: str,
        start_time: float,
    ) -> TileInferenceResponse:
        """Synchronously execute custom uploaded imagery forward pass and asset saving on worker thread."""
        models = model_service.get_models()
        device = model_service.get_device()

        with torch.inference_mode():
            sr_4m, disag_map = run_direct_sr_tile(models, sub_10m, device=device)

        # Empirical Trust Evaluation
        trust_data = compute_real_inference_trust(
            sr_4m=sr_4m,
            lr_10m=sub_10m,
            disagreement_map=disag_map,
            models=models,
            device=device,
            config=CORE_CONFIG,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Downstream building footprint extraction
        morph_cfg = CORE_CONFIG.get("downstream", {}).get("morphology", {})
        foot_sr = extract_building_footprints(
            sr_4m,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=2,
            nir_idx=3,
        )

        # Build auditable trust receipt
        dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        receipt = generate_real_trust_receipt(
            scene_name=f"user_input_{upload_hash[:8]}",
            val_info=val_info,
            trust_data=trust_data,
            config=CORE_CONFIG,
            device_name=dev_name,
        )

        # Save assets and metadata
        return cache_service.save_inference_products(
            run_id=run_id,
            scene_id="user_input",
            tile_id=0,
            source="user_input",
            latency_ms=latency_ms,
            input_10m=sub_10m,
            sr_4m=sr_4m,
            disagreement_map=disag_map,
            fusion_result=trust_data["fusion_result"],
            foot_mask_sr=foot_sr,
            receipt=receipt,
        )


# Singleton instance
inference_service = InferenceService()

