"""Pydantic Models and Schemas for GeoFUSE API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    """System hardware and runtime status telemetry schema."""

    status: str = Field(default="operational", description="Backend service status")
    project_name: str = Field(..., description="Project name")
    version: str = Field(..., description="Project version")
    python_version: str = Field(..., description="Active Python version")
    pytorch_version: str = Field(..., description="Active PyTorch version")
    cuda_available: bool = Field(..., description="Whether CUDA acceleration is available")
    device_name: str = Field(..., description="Active compute device name")
    vram_total_gb: Optional[float] = Field(None, description="Total GPU VRAM in GB")
    vram_allocated_gb: Optional[float] = Field(None, description="Currently allocated VRAM in GB")
    vram_reserved_gb: Optional[float] = Field(None, description="Currently reserved VRAM in GB")
    ensemble_checkpoints_found: int = Field(..., description="Number of valid ensemble member checkpoints detected")
    demo_cache_available: bool = Field(..., description="Whether offline precomputed demo cache is present")
    active_dataset_count: int = Field(..., description="Number of detected Sentinel-2 scene directories")
    gpu_queue_active: int = Field(default=0, description="Active inferences currently executing on GPU")
    gpu_queue_depth: int = Field(default=0, description="Number of requests waiting in the GPU queue")


class HealthResponse(BaseModel):
    """Basic health check schema."""

    status: str = "ok"
    timestamp: str


class SceneItem(BaseModel):
    """Metadata summary for a single Sentinel-2 scene."""

    scene_id: str = Field(..., description="Unique scene identifier")
    name: str = Field(..., description="Display name for UI")
    description: str = Field(..., description="Geographical or temporal description")
    path: str = Field(..., description="Relative scene directory path")
    bands: List[str] = Field(..., description="Available spectral bands")
    resolution_meters: float = Field(default=10.0, description="Native ground sample distance in meters")
    crs: str = Field(default="UTM", description="Coordinate Reference System")
    dimensions_pixels: List[int] = Field(..., description="[Height, Width] of scene in pixels")


class SceneListResponse(BaseModel):
    """List of all available local Sentinel-2 scenes."""

    scenes: List[SceneItem]
    default_scene_id: str


class TileMetadataItem(BaseModel):
    """Partition tile coordinates and spatial bounds within a scene."""

    tile_id: int = Field(..., description="Zero-indexed tile ID (0..24)")
    row: int = Field(..., description="1-indexed grid row")
    col: int = Field(..., description="1-indexed grid column")
    x: int = Field(..., description="X pixel offset in scene")
    y: int = Field(..., description="Y pixel offset in scene")
    w: int = Field(..., description="Tile width in pixels")
    h: int = Field(..., description="Tile height in pixels")
    patch_size: int = Field(..., description="Square patch size")
    description: str = Field(..., description="Quadrant / region description")
    label: str = Field(..., description="Formatted UI badge label")
    has_demo_cache: bool = Field(default=False, description="Whether precomputed offline demo cache is available")


class SceneGridResponse(BaseModel):
    """5x5 scene grid partition metadata and macro overview image."""

    scene_id: str
    scene_name: str
    dimensions: List[int]
    total_tiles: int
    macro_preview_url: str = Field(..., description="URL to high-resolution scene overview asset")
    tiles: List[TileMetadataItem]


class TileInferenceRequest(BaseModel):
    """Request payload to execute or retrieve inference for a single tile."""

    scene_id: str = Field(default="urban_core", description="Identifier of the target scene")
    tile_id: int = Field(default=0, ge=0, le=24, description="Target tile index (0..24)")
    force_live: bool = Field(default=False, description="If True, bypasses precomputed demo cache and forces GPU inference")


class InferenceAssetUrls(BaseModel):
    """URLs for raster visualization assets (avoiding Base64 JSON payloads)."""

    rgb_input: str = Field(..., description="10m Original Sentinel-2 Natural RGB (B4-B3-B2)")
    rgb_sr: str = Field(..., description="4m GeoFUSE Super-Resolution Natural RGB")
    cir_input: str = Field(..., description="10m False-Color Infrared CIR (B8-B4-B3)")
    cir_sr: str = Field(..., description="4m False-Color Infrared CIR")
    ndvi_input: str = Field(..., description="10m NDVI Vegetation Index colormap")
    ndvi_sr: str = Field(..., description="4m NDVI Vegetation Index colormap")
    uncertainty: str = Field(..., description="Ensemble epistemic disagreement map")
    trust_map: str = Field(..., description="Multi-criteria composite trust / risk map")
    building_mask: Optional[str] = Field(None, description="Downstream building footprint extraction overlay")


class EvidenceMetricsSummary(BaseModel):
    """Empirical reliability metrics calculated without ground truth."""

    trust_score_pct: float = Field(..., description="Composite empirical Trust Score (0-100%)")
    is_trusted: bool = Field(..., description="Whether trust score meets operational threshold (86.5%)")
    status_label: str = Field(..., description="NOMINAL_HIGH_TRUST or WARNING_LOW_TRUST")
    mean_disagreement: float = Field(..., description="Mean ensemble standard deviation")
    delta_ndvi_mean: float = Field(..., description="Mean radiometric spectral delta-NDVI")
    pct_spectral_consistent: float = Field(..., description="Percentage of pixels within delta-NDVI tolerance")
    edge_iou: float = Field(..., description="Structural edge IoU agreement")
    gradient_correlation: float = Field(..., description="Sobel gradient Pearson correlation r")
    warnings: List[str] = Field(default_factory=list, description="Plain-language advisories or warnings")


class TileStatistics(BaseModel):
    """Authentic tile-level spectral distribution, land-cover proxies, and feature counts."""

    mean_ndvi: float = Field(..., description="Mean NDVI vegetation index of the tile")
    vegetation_pct: float = Field(..., description="Estimated percentage of vegetated surfaces (NDVI > 0.25)")
    water_pct: float = Field(..., description="Estimated percentage of water-like surfaces (negative NDVI, low NIR)")
    built_pct: float = Field(..., description="Estimated percentage of built/impervious surfaces")
    building_count: int = Field(0, description="Detected building footprint contours in tile")
    dominant_feature: str = Field(..., description="Dominant surface feature based on spectral response")


class TileInferenceResponse(BaseModel):
    """Complete inference and evidence response for a tile."""

    run_id: str = Field(..., description="Unique deterministic execution or cache ID")
    scene_id: str
    tile_id: int
    source: str = Field(..., description="'demo_cache', 'api_cache', 'live_inference', 'user_input', or 'user_input_cache'")
    latency_ms: float = Field(..., description="Execution time in milliseconds")
    input_resolution_m: float = 10.0
    output_resolution_m: float = 4.0
    scale_factor: float = 2.5
    assets: InferenceAssetUrls
    evidence_metrics: EvidenceMetricsSummary
    receipt_id: str = Field(..., description="Cryptographic receipt ID for full audit")
    receipt_url: str = Field(..., description="Direct URL to fetch full JSON Trust Receipt")
    tile_stats: Optional[TileStatistics] = Field(None, description="Authentic computed tile spectral statistics and feature summary")


class CustomValidationResponse(BaseModel):
    """Validation metadata returned upon checking user-uploaded Sentinel-2 files."""

    is_valid: bool = Field(..., description="Whether the uploaded imagery strictly satisfies Sentinel-2 Level-2A requirements")
    upload_id: str = Field(..., description="Deterministic upload session hash")
    scene_id: str = Field(..., description="Canonical custom scene identifier for tile navigation")
    message: str = Field(..., description="Status summary or advisory")
    format: str = Field(..., description="'single_multiband' or 'separate_bands'")
    crs: str = Field(..., description="Detected Coordinate Reference System (e.g., EPSG:32643)")
    shape: List[int] = Field(..., description="Native dimensions [height, width] in pixels")
    resolution: List[float] = Field(..., description="Native spatial resolution [res_x, res_y] in meters (~10m)")
    bands: List[str] = Field(..., description="Identified Sentinel-2 bands (B02, B03, B04, B08)")
    files_count: int = Field(..., description="Number of staged files")
    grid: Optional[SceneGridResponse] = Field(None, description="Pre-computed 5x5 tile grid partition layout")



class BenchmarkSummaryResponse(BaseModel):
    """Historical synthetic degrade-and-recover benchmark results."""

    benchmark_type: str = "synthetic_degrade_and_recover"
    description: str
    baseline_methods: List[Dict[str, Any]]
    evaluation_dataset: Dict[str, Any]
    structural_metrics: Dict[str, Any]
    radiometric_metrics: Dict[str, Any]
    scientific_disclaimer: str

