import hashlib
from pathlib import Path
import shutil
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.config import OUTPUTS_DIR
from src.api.schemas.payloads import (
    CustomValidationResponse,
    TileInferenceRequest,
    TileInferenceResponse,
)
from src.api.services.inference_service import inference_service
from src.api.services.scene_service import scene_service
from src.inference.real_inference import validate_sentinel2_input

router = APIRouter(prefix="/api/inference", tags=["Inference & Verification"])

# Security and resource exhaustion limits
MAX_UPLOAD_BYTES = 150 * 1024 * 1024  # 150 MB total
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jp2"}
TIFF_MAGIC_LE = b"II*\x00"
TIFF_MAGIC_BE = b"MM\x00*"
JP2_MAGIC = b"\x00\x00\x00\x0c"


@router.post("/tile", response_model=TileInferenceResponse)
async def infer_tile(request: TileInferenceRequest) -> TileInferenceResponse:
    """Execute direct 10m -> 4m learned super-resolution on a Sentinel-2 scene tile.

    Checks offline demo cache and API cache first. If cache miss, executes live forward
    pass through the 3-member ResidualSRNet ensemble on GPU (zero synthetic degradation).
    """
    try:
        return await inference_service.process_tile_inference(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")


@router.post("/validate", response_model=CustomValidationResponse)
async def validate_custom_geotiff(
    files: List[UploadFile] = File(..., description="1 four-band GeoTIFF or 4 individual band TIFFs (B02, B03, B04, B08)")
) -> CustomValidationResponse:
    """Validate and stage user-uploaded Sentinel-2 GeoTIFF files before execution.

    Inspects raster structure, band coverage, resolution (~10m GSD), and CRS consistency.
    Enforces upload size limits, extension whitelisting, and path traversal sanitization.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for validation.")

    # 1. Enforce upload security: extension whitelist and total size limits
    total_size = 0
    file_bytes_map = {}
    h = hashlib.sha256()

    for f in sorted(files, key=lambda x: x.filename or ""):
        raw_fname = f.filename or "unknown.tif"
        # Sanitize against path traversal attacks (../, ..\, absolute, UNC)
        clean_name = Path(raw_fname).name
        if ".." in clean_name or "/" in clean_name or "\\" in clean_name:
            raise HTTPException(status_code=400, detail=f"Unsafe filename rejected: '{clean_name}'.")

        ext = Path(clean_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Only GeoTIFF (.tif, .tiff) and JP2 (.jp2) are accepted.",
            )

        content = await f.read()
        total_size += len(content)
        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload size exceeds maximum allowed limit ({MAX_UPLOAD_BYTES // (1024*1024)} MB).",
            )

        # Basic magic bytes sanity check
        if ext in {".tif", ".tiff"}:
            if len(content) < 4 or (content[:4] != TIFF_MAGIC_LE and content[:4] != TIFF_MAGIC_BE):
                raise HTTPException(status_code=400, detail=f"File '{clean_name}' is not a valid TIFF image.")
        elif ext == ".jp2":
            if len(content) < 4 or content[:4] != JP2_MAGIC:
                raise HTTPException(status_code=400, detail=f"File '{clean_name}' is not a valid JPEG 2000 image.")

        file_bytes_map[clean_name] = content
        h.update(clean_name.encode("utf-8"))
        h.update(content)
        await f.seek(0)

    upload_hash = h.hexdigest()[:12]
    scene_id = f"custom_{upload_hash}"
    upload_dir = OUTPUTS_DIR / "user_uploads" / upload_hash
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    try:
        for clean_name, content in file_bytes_map.items():
            target_path = upload_dir / clean_name
            with open(target_path, "wb") as buffer:
                buffer.write(content)
            saved_paths.append(target_path)

        # 2. Strict Sentinel-2 validation
        val_source = saved_paths[0] if (len(saved_paths) == 1 and saved_paths[0].is_file()) else upload_dir
        val_info = validate_sentinel2_input(val_source)

        shape = list(val_info.get("shape", []))
        if shape and (shape[0] < 128 or shape[1] < 128):
            raise ValueError(
                f"Uploaded raster dimensions ({shape[0]}x{shape[1]}) are too small. "
                "Minimum required size is 128x128 pixels for learned Sentinel-2 super-resolution."
            )

        res_x = float(val_info.get("resolution", (10.0, 10.0))[0])
        res_y = float(val_info.get("resolution", (10.0, 10.0))[1])

        # 3. Generate 5x5 tile grid partition & macro overview preview for user navigation
        scene_grid = scene_service.get_scene_grid(scene_id)

        return CustomValidationResponse(
            is_valid=True,
            upload_id=upload_hash,
            scene_id=scene_id,
            message="Custom Sentinel-2 imagery successfully validated and partitioned into 5×5 tile grid.",
            format=str(val_info.get("format", "unknown")),
            crs=str(val_info.get("crs", "EPSG:4326")),
            shape=shape,
            resolution=[round(res_x, 2), round(res_y, 2)],
            bands=list(val_info.get("bands", [])),
            files_count=len(saved_paths),
            grid=scene_grid,
        )
    except Exception as e:
        # Clean up failed upload directory
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        detail_msg = str(e)
        # Avoid leaking raw internal file system paths to the browser
        for path_str in [str(upload_dir), str(OUTPUTS_DIR)]:
            detail_msg = detail_msg.replace(path_str, "[internal_storage]")
        raise HTTPException(status_code=400, detail=f"Invalid Sentinel-2 GeoTIFF input: {detail_msg}")


@router.post("/custom", response_model=TileInferenceResponse)
async def infer_custom_geotiff(
    files: Optional[List[UploadFile]] = File(None, description="Uploaded Sentinel-2 GeoTIFF files"),
    upload_id: Optional[str] = Form(None, description="Pre-validated upload session hash"),
    tile_id: Optional[int] = Form(None, description="Optional target tile index (0..24)"),
    crop_x: Optional[int] = Form(None, description="Crop origin X coordinate"),
    crop_y: Optional[int] = Form(None, description="Crop origin Y coordinate"),
    tile_size: int = Form(128, description="Target crop tile size in pixels (default: 128)"),
    force_live: bool = Form(False, description="Bypass cache and force GPU re-execution"),
) -> TileInferenceResponse:
    """Execute end-to-end direct 2.5x learned super-resolution and trust evaluation on custom input.

    Reuses the authentic GeoFUSE scientific pipeline: ResidualSRNet ensemble forward pass,
    epistemic disagreement uncertainty, spectral delta-NDVI, structural edge IoU,
    morphological building footprint extraction, and auditable Trust Receipt.
    """
    upload_dir: Optional[Path] = None

    has_files = isinstance(files, (list, tuple)) and len(files) > 0
    has_upload_id = isinstance(upload_id, str) and bool(upload_id.strip())

    # Option A: Direct upload with files
    if has_files:
        h = hashlib.sha256()
        file_bytes_map = {}
        for f in sorted(files, key=lambda x: x.filename or ""):
            content = await f.read()
            clean_name = Path(f.filename or "unknown.tif").name
            file_bytes_map[clean_name] = content
            h.update(clean_name.encode("utf-8"))
            h.update(content)

        upload_hash = h.hexdigest()[:12]
        upload_dir = OUTPUTS_DIR / "user_uploads" / upload_hash
        upload_dir.mkdir(parents=True, exist_ok=True)

        for fname, content in file_bytes_map.items():
            with open(upload_dir / fname, "wb") as buffer:
                buffer.write(content)

    # Option B: Reusing existing staged upload_id
    elif has_upload_id:
        clean_id = upload_id.strip()
        # Security: sanitize against path traversal
        clean_id = Path(clean_id).name
        candidate_dir = OUTPUTS_DIR / "user_uploads" / clean_id
        if candidate_dir.exists() and candidate_dir.is_dir():
            upload_dir = candidate_dir
            upload_hash = clean_id
        else:
            raise HTTPException(status_code=404, detail=f"Upload session '{clean_id}' not found or expired.")

    else:
        raise HTTPException(status_code=400, detail="Must provide either 'files' or a valid 'upload_id'.")

    # If tile_id is specified, route directly through the unified tile pipeline
    if tile_id is not None and isinstance(tile_id, int):
        scene_id = f"custom_{upload_hash}"
        req = TileInferenceRequest(
            scene_id=scene_id,
            tile_id=tile_id,
            force_live=bool(force_live),
        )
        return await inference_service.process_tile_inference(req)

    # Discover staged files
    raster_exts = {".tif", ".tiff", ".jp2", ".TIF", ".TIFF"}
    saved_paths = [p for p in upload_dir.glob("*") if p.suffix in raster_exts]
    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid GeoTIFF files found in upload session.")

    actual_tile_size = tile_size if isinstance(tile_size, int) else 128
    actual_force_live = force_live if isinstance(force_live, bool) else False

    crop_coords = None
    if isinstance(crop_x, int) and isinstance(crop_y, int):
        crop_coords = (crop_x, crop_y)

    try:
        return await inference_service.process_custom_inference(
            saved_files=saved_paths,
            tile_size=actual_tile_size,
            crop_coords=crop_coords,
            force_live=actual_force_live,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom pipeline execution failed: {e}")

