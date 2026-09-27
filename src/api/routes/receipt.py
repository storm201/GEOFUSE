"""Trust Receipt Endpoints — Authoritative Verification Certificate Retrieval."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.services.receipt_service import receipt_service

router = APIRouter(prefix="/api/receipt", tags=["Trust Receipt & Audit"])


@router.get("/{receipt_id}")
async def get_trust_receipt(
    receipt_id: str,
    download: bool = Query(False, description="If True, sets download attachment header"),
) -> Any:
    """Retrieve the authoritative, machine-readable JSON Trust Receipt.

    Contains full scientific provenance: model hyperparameters, checkpoints,
    GeoTIFF acquisition tags, empirical reliability metrics, and audit warnings.
    """
    receipt = receipt_service.get_receipt(receipt_id)
    if receipt is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trust Receipt '{receipt_id}' not found. Please verify the receipt ID.",
        )

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{receipt_id}.json"'

    return JSONResponse(content=receipt, headers=headers)
