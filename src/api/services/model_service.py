"""Model Service — Singleton Ensemble Model Lifecycle and GPU Concurrency Guard.

Loads the 3-member ResidualSRNet ensemble once during FastAPI application lifespan.
Reuses existing checkpoint loading from src.models.ensemble without duplicating
architecture definitions or reloading weights per request.

Includes an explicit async execution lock to serialize GPU inference requests on the
8GB RTX 4060 Laptop GPU to prevent VRAM over-allocation.
"""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import threading
from typing import List, Optional
import torch
import torch.nn as nn

from src.api.config import CHECKPOINTS_DIR, CORE_CONFIG, PROJECT_ROOT
from src.models.ensemble import load_ensemble_members
from src.utils.config import get_device


class ModelService:
    """Manages GPU ensemble checkpoints lifecycle and inference execution locks."""

    def __init__(self):
        self._models: Optional[List[nn.Module]] = None
        self._device: Optional[torch.device] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._gpu_thread_lock = threading.Lock()
        self._active_inferences: int = 0
        self._queue_depth: int = 0
        self._loaded: bool = False

    @property
    def gpu_thread_lock(self) -> threading.Lock:
        """OS thread-level lock serializing concurrent CUDA model executions."""
        return self._gpu_thread_lock

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy-initialize asyncio semaphore (slot=1) bound to the current running event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)
        elif current_loop is not None and getattr(self._semaphore, "_loop", None) not in (None, current_loop):
            self._semaphore = asyncio.Semaphore(1)
            self._active_inferences = 0
            self._queue_depth = 0
        return self._semaphore

    @property
    def lock(self) -> asyncio.Semaphore:
        """Backwards compatibility alias for the GPU execution semaphore."""
        return self.semaphore

    @property
    def active_inferences(self) -> int:
        """Number of inference requests currently executing forward pass on GPU."""
        return self._active_inferences

    @property
    def queue_depth(self) -> int:
        """Number of inference requests currently waiting in the GPU queue."""
        return self._queue_depth

    @asynccontextmanager
    async def acquire_gpu_slot(self, timeout_seconds: float = 60.0):
        """Acquire a single GPU inference slot with concurrency safety and VRAM cleanup.
        
        Guarantees that on consumer GPUs (e.g. 8GB RTX 4060), concurrent requests are
        safely queued rather than crashing with CUDA Out Of Memory.
        """
        self._queue_depth += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(self.semaphore.acquire(), timeout=timeout_seconds)
                acquired = True
            except asyncio.TimeoutError:
                raise TimeoutError(f"GPU inference queue timeout after {timeout_seconds}s (server busy).")

            self._queue_depth = max(0, self._queue_depth - 1)
            self._active_inferences += 1
            try:
                yield
            finally:
                self._active_inferences = max(0, self._active_inferences - 1)
                # Cleanup GPU VRAM allocations to avoid memory fragmentation
                if self._device is not None and self._device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        finally:
            if acquired:
                self.semaphore.release()
            else:
                self._queue_depth = max(0, self._queue_depth - 1)


    def load_models(self) -> List[nn.Module]:
        """Load ensemble members into GPU VRAM once during server startup."""
        if self._loaded and self._models is not None:
            return self._models

        self._device = get_device(CORE_CONFIG)
        ckpt_dir = CHECKPOINTS_DIR
        ckpt_paths = [ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)]

        # Verify all checkpoints exist
        for cp in ckpt_paths:
            if not cp.exists():
                raise FileNotFoundError(f"Missing required ensemble checkpoint: {cp.resolve()}")

        print(f"[ModelService] Loading 3-member ResidualSRNet ensemble onto {self._device}...")
        self._models = load_ensemble_members(ckpt_paths, config=CORE_CONFIG, device=self._device)
        for m in self._models:
            m.eval()

        self._loaded = True
        print(f"[ModelService] Successfully loaded ensemble ({len(self._models)} members) into memory.")
        return self._models

    def get_models(self) -> List[nn.Module]:
        """Return the preloaded ensemble models, initializing if needed."""
        if not self._loaded or self._models is None:
            return self.load_models()
        return self._models

    def get_device(self) -> torch.device:
        """Return the active compute device (CUDA / CPU)."""
        if self._device is None:
            self._device = get_device(CORE_CONFIG)
        return self._device

    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return self._loaded and self._models is not None


# Singleton instance
model_service = ModelService()
