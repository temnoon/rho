"""
Database management routes for the Rho Quantum Narrative System.

This module provides endpoints for database state inspection, cleanup operations,
and narrative management across density matrices.
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.quantum_state import diagnostics
from .matrix_routes import STATE
from .povm_routes import PACKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["database"])

# Configuration
DATA_DIR = "data"
COMPOSITE_MATRIX_ID = "global_consciousness"


class CleanupRequest(BaseModel):
    max_narratives: Optional[int] = None


class DatabaseStateResponse(BaseModel):
    total_matrices: int
    total_narratives: int
    total_packs: int
    matrices: List[Dict]
    data_dir: str


@router.get("/state", response_model=DatabaseStateResponse)
def database_state():
    """Get comprehensive database state information"""
    rho_info = []
    total_narratives = 0
    
    for rho_id, item in STATE.items():
        narratives = item.get("narratives", [])
        narrative_count = len(narratives)
        
        # For composite matrices, aggregate count from metadata
        if narrative_count == 0 and (rho_id == COMPOSITE_MATRIX_ID or rho_id == "global_consciousness"):
            meta = item.get("meta", {})
            book_titles = meta.get("book_titles", [])
            # Try to count narratives from component matrices
            for book_title in book_titles:
                book_rho_id = book_title.replace("Book_", "")
                if book_rho_id in STATE:
                    book_item = STATE[book_rho_id]
                    narrative_count += len(book_item.get("narratives", []))
        
        total_narratives += narrative_count
        
        ops = item.get("ops", [])
        
        rho_info.append({
            "rho_id": rho_id,
            "narrative_count": narrative_count,
            "operations": len(ops),
            "last_operation": ops[-1] if ops else None,
            "diagnostics": diagnostics(item["rho"]),
            "meta": item.get("meta", {})
        })
    
    return DatabaseStateResponse(
        total_matrices=len(STATE),
        total_narratives=total_narratives,
        total_packs=len(PACKS),
        matrices=rho_info,
        data_dir=DATA_DIR
    )


@router.get("/narratives/{rho_id}")
def get_narratives(rho_id: str):
    """Get all narratives for a specific density matrix"""
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    item = STATE[rho_id]
    narratives = item.get("narratives", [])
    
    return {
        "rho_id": rho_id,
        "narratives": narratives,
        "count": len(narratives)
    }


@router.post("/cleanup")
def cleanup_database(request: CleanupRequest):
    """Clean up database by removing matrices with few narratives"""
    deleted_count = 0
    deleted_matrices = []
    
    if request.max_narratives is None:
        return {
            "deleted_count": 0,
            "deleted_matrices": [],
            "message": "No cleanup criteria specified"
        }
    
    matrices_to_delete = []
    
    for rho_id, item in STATE.items():
        # Skip special matrices
        if rho_id in ['global_consciousness', 'composite_matrix', 'preview_matrix']:
            continue
            
        narratives = item.get("narratives", [])
        if len(narratives) <= request.max_narratives:
            matrices_to_delete.append(rho_id)
    
    for rho_id in matrices_to_delete:
        del STATE[rho_id]
        deleted_matrices.append(rho_id)
        deleted_count += 1
    
    logger.info(f"Cleaned up {deleted_count} matrices with ≤{request.max_narratives} narratives")
    
    return {
        "deleted_count": deleted_count,
        "deleted_matrices": deleted_matrices,
        "criteria": f"matrices with ≤{request.max_narratives} narratives"
    }


@router.post("/cleanup/duplicates")
def cleanup_duplicates():
    """Remove duplicate density matrices based on narrative content"""
    deleted_count = 0
    deleted_matrices = []
    duplicate_groups_found = 0
    
    # Group matrices by their narrative content
    narrative_groups = {}
    
    for rho_id, item in STATE.items():
        # Skip special matrices
        if rho_id in ['global_consciousness', 'composite_matrix', 'preview_matrix']:
            continue
            
        narratives = item.get("narratives", [])
        if not narratives:
            continue
            
        # Create a signature from narrative texts
        narrative_texts = [n.get("text", "") for n in narratives]
        signature = hash(tuple(sorted(narrative_texts)))
        
        if signature not in narrative_groups:
            narrative_groups[signature] = []
        narrative_groups[signature].append(rho_id)
    
    # Remove duplicates (keep the first one in each group)
    for signature, rho_ids in narrative_groups.items():
        if len(rho_ids) > 1:
            duplicate_groups_found += 1
            # Keep the first, delete the rest
            for rho_id in rho_ids[1:]:
                del STATE[rho_id]
                deleted_matrices.append(rho_id)
                deleted_count += 1
    
    logger.info(f"Removed {deleted_count} duplicate matrices from {duplicate_groups_found} groups")
    
    return {
        "deleted_count": deleted_count,
        "deleted_matrices": deleted_matrices,
        "duplicate_groups_found": duplicate_groups_found
    }


@router.delete("/matrix/{rho_id}")
def delete_matrix(rho_id: str):
    """Delete a specific density matrix"""
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # Prevent deletion of special matrices
    if rho_id in ['global_consciousness', 'composite_matrix', 'preview_matrix']:
        raise HTTPException(status_code=400, detail="Cannot delete special matrices")
    
    del STATE[rho_id]
    logger.info(f"Deleted matrix {rho_id}")
    
    return {
        "deleted": rho_id,
        "success": True
    }