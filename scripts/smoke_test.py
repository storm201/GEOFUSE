"""Smoke test for GeoFUSE SentinelGuard.

Verifies:
1. All core dependencies import cleanly with reported versions.
2. Configuration loading from config.yaml.
3. Compute device detection (CUDA RTX 4060 or graceful CPU fallback with clear warning).
4. Basic PyTorch tensor operations and synthetic array handling.
5. Integrity of project directory layout.
"""

import sys
import traceback
from pathlib import Path


def main() -> int:
    print("=" * 70)
    print("   GeoFUSE SentinelGuard — Smoke Test & Verification Suite")
    print("=" * 70)

    # 1. Dependency Imports & Version Reporting
    print("\n[Step 1/5] Verifying dependency imports...")
    dependencies = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("rasterio", "rasterio"),
        ("cv2 (opencv)", "cv2"),
        ("skimage (scikit-image)", "skimage"),
        ("matplotlib", "matplotlib"),
        ("yaml (pyyaml)", "yaml"),
        ("streamlit", "streamlit"),
        ("tqdm", "tqdm"),
    ]

    import_failures = []
    for display_name, mod_name in dependencies:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {display_name:<26} version: {ver}")
        except ImportError as e:
            print(f"  [FAIL] {display_name:<26} ERROR: {e}")
            import_failures.append((display_name, str(e)))

    if import_failures:
        print(f"\n[ERROR] Failed to import {len(import_failures)} dependencies:")
        for name, err in import_failures:
            print(f"  - {name}: {err}")
        return 1

    # 2. Config Loading Verification
    print("\n[Step 2/5] Testing config.yaml loading...")
    try:
        # Add project root to sys.path to allow src imports
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.utils.config import load_config, get_device, ensure_directories

        config = load_config()
        project_name = config.get("project", {}).get("name", "Unknown")
        print(f"  [OK] Successfully parsed config for: '{project_name}'")
        print(f"       Scale factor: {config.get('model', {}).get('scale_factor')}")
        print(f"       Patch size:   {config.get('preprocessing', {}).get('patch_size')}")
        print(f"       Bands:        {config.get('preprocessing', {}).get('bands')}")
    except Exception as e:
        print(f"  [FAIL] Failed to load configuration: {e}")
        traceback.print_exc()
        return 1

    # 3. Directory Structure Verification
    print("\n[Step 3/5] Verifying directory structure...")
    try:
        ensure_directories(config)
        for key, rel_path in config.get("paths", {}).items():
            if key.endswith("_dir"):
                p = project_root / rel_path
                exists = p.exists() and p.is_dir()
                status = "OK" if exists else "MISSING"
                print(f"  [{status}] {key:<20} -> {p.relative_to(project_root)}")
                if not exists:
                    return 1
    except Exception as e:
        print(f"  [FAIL] Directory verification failed: {e}")
        return 1

    # 4. Device & Hardware Verification
    print("\n[Step 4/5] Checking hardware and compute device...")
    import torch
    device = get_device(config)
    print(f"  Resolved execution device: {device}")
    if device.type == "cuda":
        print(f"  Device Name:  {torch.cuda.get_device_name(0)}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"  Total VRAM:   {vram_mb:.0f} MB")
    else:
        print("  [Notice] Running with CPU fallback mode active.")

    # 5. PyTorch & Array Sanity Check
    print("\n[Step 5/5] Running tensor & array sanity checks...")
    try:
        import numpy as np
        # Simulate a 4-band Sentinel-2 patch: (Batch, Channels=4, H=64, W=64)
        patch_size = config.get("preprocessing", {}).get("patch_size", 64)
        dummy_input = torch.randn(2, 4, patch_size, patch_size, dtype=torch.float32, device=device)
        dummy_mean = dummy_input.mean().item()
        
        # NumPy conversion
        np_arr = dummy_input.cpu().numpy()
        assert np_arr.shape == (2, 4, patch_size, patch_size)
        print(f"  [OK] Successfully allocated and operated on tensor of shape {tuple(dummy_input.shape)}")
        print(f"       Tensor device: {dummy_input.device}, Mean: {dummy_mean:.4f}")
    except Exception as e:
        print(f"  [FAIL] Sanity check failed: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("   [SUCCESS] GeoFUSE SentinelGuard Phase 0 Smoke Test Passed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
