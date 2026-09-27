"""Scene Discovery and Tile Grid Layout Endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.schemas.payloads import SceneGridResponse, SceneListResponse
from src.api.services.scene_service import scene_service

router = APIRouter(prefix="/api/scenes", tags=["Scene & Grid Explorer"])


@router.get("/list", response_model=SceneListResponse)
async def list_available_scenes() -> SceneListResponse:
    """Return all verified Sentinel-2 scenes present in the local filesystem."""
    scenes = scene_service.list_scenes()
    if not scenes:
        raise HTTPException(status_code=404, detail="No valid Sentinel-2 scenes found.")
    return SceneListResponse(
        scenes=scenes,
        default_scene_id="urban_core" if any(s.scene_id == "urban_core" for s in scenes) else scenes[0].scene_id,
    )


@router.get("/{scene_id}/grid", response_model=SceneGridResponse)
async def get_scene_grid_partition(scene_id: str) -> SceneGridResponse:
    """Return 5x5 tile grid partition coordinates and macro overview preview URL."""
    try:
        return scene_service.get_scene_grid(scene_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scene grid: {e}")
