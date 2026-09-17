"""GeoFUSE SentinelGuard — Inference Package.

Exposes direct real-image super-resolution and validation functions.
"""

from .real_inference import (
    BAND_DESCRIPTIONS,
    BAND_NAMES,
    compute_4m_geotransform,
    compute_real_inference_trust,
    execute_real_sentinel2_sr,
    export_real_inference_products,
    generate_real_trust_receipt,
    load_and_preprocess_sentinel2,
    run_direct_sr_scene,
    run_direct_sr_tile,
    validate_sentinel2_input,
)

__all__ = [
    "BAND_NAMES",
    "BAND_DESCRIPTIONS",
    "validate_sentinel2_input",
    "load_and_preprocess_sentinel2",
    "run_direct_sr_tile",
    "run_direct_sr_scene",
    "compute_4m_geotransform",
    "compute_real_inference_trust",
    "generate_real_trust_receipt",
    "export_real_inference_products",
    "execute_real_sentinel2_sr",
]
