"""Synthetic Degradation Benchmark Summary Endpoint."""

from fastapi import APIRouter

from src.api.schemas.payloads import BenchmarkSummaryResponse

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark Lab"])


@router.get("/summary", response_model=BenchmarkSummaryResponse)
async def get_benchmark_summary() -> BenchmarkSummaryResponse:
    """Return historical quantitative hold-out evaluation benchmark results.

    SCIENTIFIC HONESTY MANDATE:
    This endpoint serves controlled synthetic degrade-and-recover benchmark evaluations.
    Synthetic reconstruction metrics (PSNR, SSIM) evaluate algorithmic recovery against
    pre-degradation references and must NOT be conflated with un-degraded real 4m ground truth.
    """
    return BenchmarkSummaryResponse(
        benchmark_type="synthetic_degrade_and_recover",
        description=(
            "Controlled Degrade-and-Recover Evaluation on Strictly Contiguous Southeast Quadrant "
            "(rows 256..512, cols 256..512) Hold-Out Split (49 non-leaking test patches)."
        ),
        baseline_methods=[
            {
                "name": "Bicubic Baseline (2x/2.5x)",
                "parameters": 0,
                "val_loss": 0.01920,
                "val_psnr_db": 38.19,
                "val_ssim": 0.9208,
                "is_ours": False,
            },
            {
                "name": "ResidualSRNet (Ours - Single Member)",
                "parameters": 797477,
                "val_loss": 0.01726,
                "val_psnr_db": 38.68,
                "val_ssim": 0.9270,
                "delta_psnr_db": 0.49,
                "delta_ssim": 0.0062,
                "is_ours": True,
            },
            {
                "name": "GeoFUSE Ensemble (3-Member Variance Consensus)",
                "parameters": 2392431,
                "val_loss": 0.01698,
                "val_psnr_db": 38.84,
                "val_ssim": 0.9295,
                "delta_psnr_db": 0.65,
                "delta_ssim": 0.0087,
                "is_ours": True,
            },
        ],
        evaluation_dataset={
            "geographic_hold_out": "Southeast Quadrant (Zero Spatial Leakage)",
            "test_patches": 49,
            "training_patches": 120,
            "scene_dimensions": "512x512 pixels (~5.12 km x 5.12 km at 10m GSD)",
        },
        structural_metrics={
            "edge_iou": 0.418,
            "edge_f1": 0.589,
            "gradient_correlation_r": 0.796,
        },
        radiometric_metrics={
            "mean_delta_ndvi": 0.021,
            "pct_pixels_within_tolerance": 92.6,
            "sensor_noise_floor": 0.03,
        },
        scientific_disclaimer=(
            "In the absence of certified independent high-resolution airborne or sub-meter satellite "
            "ground truth, all metrics reflect controlled synthetic degradation recovery and inter-model "
            "consensus, never unobserved physical reality."
        ),
    )
