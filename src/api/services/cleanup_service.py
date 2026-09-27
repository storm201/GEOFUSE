"""Cleanup Service — Deterministic LRU & TTL Storage Management for User Uploads.

Provides safe, audited retention management for custom uploads:
1. Time-To-Live (TTL): Removes user uploads exceeding max_age_days (default 7 days).
2. Storage Cap (LRU): Enforces max_disk_bytes (default 5GB) by evicting oldest uploads first.
3. Cache Cascade: Cleans corresponding raster visual assets in outputs/cache_api/.
4. Safety Guarantees: Never touches preloaded demo datasets or system checkpoints.
5. Dry-Run Mode: Simulates cleanup and returns exact audit telemetry without deletion.
"""

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.api.config import CACHE_API_DIR, OUTPUTS_DIR, PROJECT_ROOT


class CleanupCandidate(BaseModel):
    """Metadata for an identified upload directory candidate for cleanup."""
    upload_hash: str
    path: str
    size_bytes: int
    size_mb: float
    age_hours: float
    mtime: float
    reason: str


class CleanupSummary(BaseModel):
    """Audit summary of cleanup execution."""
    dry_run: bool
    initial_upload_count: int
    initial_total_bytes: int
    initial_total_mb: float
    freed_bytes: int
    freed_mb: float
    remaining_upload_count: int
    remaining_total_bytes: int
    remaining_total_mb: float
    evicted_candidates: List[CleanupCandidate] = Field(default_factory=list)
    timestamp: float


def get_dir_size_and_mtime(directory: Path) -> Tuple[int, float]:
    """Calculate recursive byte size and latest modification time for a directory."""
    total_size = 0
    latest_mtime = directory.stat().st_mtime

    for p in directory.rglob("*"):
        if p.is_file():
            stat = p.stat()
            total_size += stat.st_size
            if stat.st_mtime > latest_mtime:
                latest_mtime = stat.st_mtime

    return total_size, latest_mtime


class CleanupService:
    """Manages disk retention policies and storage reclamation for user uploads."""

    def __init__(self):
        self.uploads_dir = OUTPUTS_DIR / "user_uploads"
        self.cache_api_dir = CACHE_API_DIR

    def inspect_storage(self) -> Dict[str, Any]:
        """Inspect current user upload storage metrics without modifying disk."""
        if not self.uploads_dir.exists():
            return {
                "upload_count": 0,
                "total_bytes": 0,
                "total_mb": 0.0,
                "uploads": [],
            }

        upload_items = []
        total_bytes = 0

        for d in self.uploads_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                size_bytes, mtime = get_dir_size_and_mtime(d)
                total_bytes += size_bytes
                age_hours = (time.time() - mtime) / 3600.0
                upload_items.append({
                    "upload_hash": d.name,
                    "path": str(d),
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "age_hours": round(age_hours, 2),
                    "mtime": mtime,
                })

        return {
            "upload_count": len(upload_items),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "uploads": sorted(upload_items, key=lambda x: x["mtime"]),
        }

    def run_cleanup(
        self,
        max_age_days: float = 7.0,
        max_disk_bytes: int = 5 * 1024 * 1024 * 1024,  # 5 GB
        dry_run: bool = False,
    ) -> CleanupSummary:
        """Execute deterministic TTL + LRU cleanup on user uploads."""
        now = time.time()
        max_age_seconds = max_age_days * 86400.0

        if not self.uploads_dir.exists():
            return CleanupSummary(
                dry_run=dry_run,
                initial_upload_count=0,
                initial_total_bytes=0,
                initial_total_mb=0.0,
                freed_bytes=0,
                freed_mb=0.0,
                remaining_upload_count=0,
                remaining_total_bytes=0,
                remaining_total_mb=0.0,
                evicted_candidates=[],
                timestamp=now,
            )

        # 1. Gather all upload directories
        candidates = []
        initial_total_bytes = 0

        for d in self.uploads_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                size_bytes, mtime = get_dir_size_and_mtime(d)
                initial_total_bytes += size_bytes
                age_hours = (now - mtime) / 3600.0
                candidates.append({
                    "upload_hash": d.name,
                    "dir_path": d,
                    "size_bytes": size_bytes,
                    "mtime": mtime,
                    "age_hours": age_hours,
                })

        initial_count = len(candidates)
        # Sort oldest first (LRU order)
        candidates.sort(key=lambda x: x["mtime"])

        evicted_records: List[CleanupCandidate] = []
        evicted_hashes = set()
        freed_bytes = 0
        current_total_bytes = initial_total_bytes

        # 2. TTL Pass: Evict uploads older than max_age_days
        for c in candidates:
            age_sec = now - c["mtime"]
            if age_sec > max_age_seconds:
                evicted_hashes.add(c["upload_hash"])
                freed_bytes += c["size_bytes"]
                current_total_bytes -= c["size_bytes"]
                evicted_records.append(
                    CleanupCandidate(
                        upload_hash=c["upload_hash"],
                        path=str(c["dir_path"]),
                        size_bytes=c["size_bytes"],
                        size_mb=round(c["size_bytes"] / (1024 * 1024), 2),
                        age_hours=round(c["age_hours"], 2),
                        mtime=c["mtime"],
                        reason=f"TTL expired (> {max_age_days} days)",
                    )
                )

        # 3. LRU Storage Cap Pass: If still exceeding max_disk_bytes, evict oldest
        if current_total_bytes > max_disk_bytes:
            for c in candidates:
                if c["upload_hash"] in evicted_hashes:
                    continue
                if current_total_bytes <= max_disk_bytes:
                    break

                evicted_hashes.add(c["upload_hash"])
                freed_bytes += c["size_bytes"]
                current_total_bytes -= c["size_bytes"]
                evicted_records.append(
                    CleanupCandidate(
                        upload_hash=c["upload_hash"],
                        path=str(c["dir_path"]),
                        size_bytes=c["size_bytes"],
                        size_mb=round(c["size_bytes"] / (1024 * 1024), 2),
                        age_hours=round(c["age_hours"], 2),
                        mtime=c["mtime"],
                        reason=f"Disk cap exceeded ({round(max_disk_bytes / (1024*1024*1024), 1)}GB cap, LRU eviction)",
                    )
                )

        # 4. Perform actual deletion if not dry_run
        if not dry_run:
            for rec in evicted_records:
                target_dir = Path(rec.path)
                try:
                    if target_dir.exists() and target_dir.is_dir():
                        shutil.rmtree(target_dir)

                    # Cascade delete associated cached outputs
                    self._cascade_cleanup_cache(rec.upload_hash)
                except Exception as e:
                    print(f"[Cleanup Error] Failed to delete {rec.upload_hash}: {e}")

        remaining_count = initial_count - len(evicted_records)
        remaining_bytes = initial_total_bytes - freed_bytes

        return CleanupSummary(
            dry_run=dry_run,
            initial_upload_count=initial_count,
            initial_total_bytes=initial_total_bytes,
            initial_total_mb=round(initial_total_bytes / (1024 * 1024), 2),
            freed_bytes=freed_bytes,
            freed_mb=round(freed_bytes / (1024 * 1024), 2),
            remaining_upload_count=remaining_count,
            remaining_total_bytes=remaining_bytes,
            remaining_total_mb=round(remaining_bytes / (1024 * 1024), 2),
            evicted_candidates=evicted_records,
            timestamp=now,
        )

    def _cascade_cleanup_cache(self, upload_hash: str):
        """Remove cached visual products associated with an evicted upload hash."""
        # 1. Macro scene preview
        scene_preview = self.cache_api_dir / "scenes" / f"custom_{upload_hash}_preview.webp"
        if scene_preview.exists():
            try:
                scene_preview.unlink()
            except Exception:
                pass

        # 2. Tile inference product directories: run_custom_{upload_hash}_* and live_custom_{upload_hash}_*
        for d in self.cache_api_dir.glob(f"*custom_{upload_hash}_*"):
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass


# Singleton instance
cleanup_service = CleanupService()
