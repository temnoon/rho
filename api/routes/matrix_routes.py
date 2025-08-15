"""
Basic density matrix operations routes.

This module handles fundamental operations on quantum density matrices:
creation, reading, measurement, and basic state management.
"""

import uuid
import logging
import numpy as np
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException

from core.quantum_state import (
    create_maximally_mixed_state,
    blend_states,
    diagnostics,
    rho_matrix_to_list,
    psd_project,
    apply_text_channel
)
from core.embedding import text_to_rho
from models.requests import (
    ReadReq, 
    MeasurementResponse,
    MatrixResponse,
    DiagnosticsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rho", tags=["matrix"])

# Global state storage (in production, use proper database)
STATE: Dict[str, Dict[str, Any]] = {}


def save_json_atomic(path: str, obj: Any) -> None:
    """Atomic JSON save operation."""
    import json
    import os
    
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.rename(tmp, path)


@router.post("/init")
def rho_init(seed_text: Optional[str] = None, label: Optional[str] = None):
    """
    Create a new rho (maximally mixed or seeded from seed_text).
    Returns diagnostics.
    """
    rho_id = str(uuid.uuid4())
    
    if seed_text and seed_text.strip():
        # Create pure state from text
        rho = text_to_rho(seed_text.strip())
        logger.info(f"Created seeded rho {rho_id} from text: {seed_text[:50]}...")
    else:
        # Create maximally mixed state
        rho = create_maximally_mixed_state()
        logger.info(f"Created maximally mixed rho {rho_id}")
    
    # Store state
    STATE[rho_id] = {
        "rho": rho,
        "ops": [],  # Operation history
        "narratives": [],  # Stored narratives
        "label": label or f"rho_{rho_id[:8]}",
        "created_at": None  # TODO: Add timestamp
    }
    
    # Return diagnostics
    diag = diagnostics(rho)
    
    return {
        "rho_id": rho_id,
        "created": True,
        "seeded": bool(seed_text),
        "diagnostics": diag
    }


@router.get("/{rho_id}")
def rho_get(rho_id: str):
    """Get current rho matrix and diagnostics."""
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    rho = item["rho"]
    diag = diagnostics(rho)
    
    return MatrixResponse(
        rho_id=rho_id,
        matrix=rho_matrix_to_list(rho),
        diagnostics=DiagnosticsResponse(**diag),
        label=item.get("label")
    )


@router.post("/{rho_id}/read")
def rho_read(rho_id: str, req: ReadReq):
    """Read text and update rho matrix via blending."""
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # Get text to read
    if req.raw_text:
        text = req.raw_text.strip()
    elif req.text_id:
        # TODO: Implement text_id lookup
        raise HTTPException(status_code=400, detail="text_id lookup not implemented")
    else:
        raise HTTPException(status_code=400, detail="Must provide raw_text or text_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    # Convert text to density matrix
    text_rho = text_to_rho(text)
    
    # Blend with current state
    current_rho = item["rho"]
    new_rho = blend_states(current_rho, text_rho, req.alpha)
    
    # Update state
    item["rho"] = new_rho
    item["ops"].append({
        "op": "read",
        "text": text[:100] + "..." if len(text) > 100 else text,
        "alpha": req.alpha,
        "summary": f"Read {len(text)} chars with alpha={req.alpha}"
    })
    item["narratives"].append({
        "text": text,
        "alpha": req.alpha,
        "length": len(text)
    })
    
    logger.info(f"Read {len(text)} chars into rho {rho_id} with alpha={req.alpha}")
    
    # Return updated diagnostics
    diag = diagnostics(new_rho)
    
    return {
        "rho_id": rho_id,
        "read": True,
        "text_length": len(text),
        "alpha": req.alpha,
        "diagnostics": diag
    }


@router.post("/{rho_id}/read_channel")
def rho_read_with_channel(rho_id: str, req: ReadReq, channel_type: str = "rank_one_update"):
    """
    🚀 ENHANCED: Read text using proper quantum channels (CPTP evolution).
    
    This is the NEW channel-based reading that implements proper quantum
    information theory instead of simple convex combinations.
    
    Channel types:
    - "rank_one_update": Standard blending with CPTP guarantees
    - "coherent_rotation": Unitary perspective shift (entropy-preserving)
    - "dephasing_mixture": Ambiguous text with multiple interpretations
    """
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # Get text to read
    if req.raw_text:
        text = req.raw_text.strip()
    elif req.text_id:
        raise HTTPException(status_code=400, detail="text_id lookup not implemented")
    else:
        raise HTTPException(status_code=400, detail="Must provide raw_text or text_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        # Convert text to embedding vector (not density matrix)
        from core.embedding import text_to_embedding_vector
        text_embedding = text_to_embedding_vector(text)
        
        # Apply proper quantum channel evolution
        current_rho = item["rho"]
        new_rho = apply_text_channel(current_rho, text_embedding, req.alpha, channel_type)
        
        # Channel audit (in debug mode)
        channel_audit = None
        if __debug__:
            try:
                from core.text_channels import text_to_channel, audit_channel_properties
                test_channel = text_to_channel(text_embedding, req.alpha, channel_type)
                channel_audit = audit_channel_properties(test_channel, current_rho)
            except Exception as audit_error:
                logger.warning(f"Channel audit failed: {audit_error}")
                channel_audit = {"error": str(audit_error)}
        
        # Update state with channel information
        item["rho"] = new_rho
        item["ops"].append({
            "op": "read_channel",
            "channel_type": channel_type,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "alpha": req.alpha,
            "summary": f"Channel read {len(text)} chars via {channel_type}",
            "channel_audit": channel_audit
        })
        item["narratives"].append({
            "text": text,
            "alpha": req.alpha,
            "length": len(text),
            "channel_type": channel_type
        })
        
        logger.info(f"Channel read {len(text)} chars into rho {rho_id} via {channel_type}")
        
        # Return enhanced diagnostics with channel info
        diag = diagnostics(new_rho)
        
        # Ensure all data is JSON serializable
        from utils.persistence import numpy_to_json_serializable
        
        return {
            "rho_id": rho_id,
            "read": True,
            "method": "quantum_channel",
            "channel_type": channel_type,
            "text_length": len(text),
            "alpha": req.alpha,
            "diagnostics": numpy_to_json_serializable(diag),
            "channel_audit": numpy_to_json_serializable(channel_audit) if channel_audit else None,
            "quantum_effects": {
                "trace_preservation": float(abs(diag["trace"] - 1.0)),
                "psd_compliance": bool(min(diag["eigenvals"]) >= -1e-10),
                "entropy_change": float(diag.get("entropy", 0) - diagnostics(current_rho).get("entropy", 0))
            }
        }
        
    except Exception as e:
        logger.error(f"Channel reading failed for {rho_id}: {e}")
        logger.error(f"Channel failure details: {type(e).__name__}: {str(e)}")
        
        # Maintain quantum channel integrity - return proper error instead of degraded fallback
        raise HTTPException(
            status_code=422, 
            detail=f"Quantum channel evolution failed: {str(e)}. "
                   f"The density matrix state cannot be safely updated. "
                   f"Channel type '{channel_type}' may be incompatible with current state."
        )


@router.post("/{rho_id}/reset")
def rho_reset(rho_id: str, seed_text: Optional[str] = None):
    """Reset rho matrix to maximally mixed or seeded state."""
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    if seed_text and seed_text.strip():
        # Reset to seeded state
        rho = text_to_rho(seed_text.strip())
        reset_type = "seeded"
    else:
        # Reset to maximally mixed
        rho = create_maximally_mixed_state()
        reset_type = "maximally_mixed"
    
    # Update state
    item["rho"] = rho
    item["ops"].append({
        "op": "reset",
        "type": reset_type,
        "seed_text": seed_text[:50] + "..." if seed_text and len(seed_text) > 50 else seed_text
    })
    # Keep narratives but clear operation history for new session
    item["ops"] = item["ops"][-1:]  # Keep only the reset operation
    
    logger.info(f"Reset rho {rho_id} to {reset_type}")
    
    # Return diagnostics
    diag = diagnostics(rho)
    
    return {
        "rho_id": rho_id,
        "reset": True,
        "type": reset_type,
        "diagnostics": diag
    }


@router.post("/math/eig")
def math_eig(rho_id: str):
    """Eigendecomposition diagnostics for a given rho."""
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    rho = item["rho"]
    
    # Full eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(rho)
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    return {
        "rho_id": rho_id,
        "eigenvals": eigenvals.tolist(),
        "eigenvecs": eigenvecs.tolist(),
        "rank": int(np.sum(eigenvals > 1e-10)),
        "trace": float(np.sum(eigenvals))
    }


@router.get("/list")
def list_matrices():
    """List all available density matrices."""
    matrices = []
    for rho_id, item in STATE.items():
        diag = diagnostics(item["rho"])
        matrices.append({
            "rho_id": rho_id,
            "label": item.get("label", f"rho_{rho_id[:8]}"),
            "narratives_count": len(item.get("narratives", [])),
            "operations_count": len(item.get("ops", [])),
            "purity": diag["purity"],
            "entropy": diag["entropy"]
        })
    
    return {"matrices": matrices}


@router.delete("/{rho_id}")
def delete_matrix(rho_id: str):
    """Delete a density matrix."""
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    deleted_info = {
        "rho_id": rho_id,
        "label": STATE[rho_id].get("label"),
        "narratives_count": len(STATE[rho_id].get("narratives", []))
    }
    
    del STATE[rho_id]
    logger.info(f"Deleted matrix {rho_id}")
    
    return {"deleted": True, "matrix_info": deleted_info}


# State management utilities
def get_state_dict() -> Dict[str, Dict[str, Any]]:
    """Get current state dictionary."""
    return STATE


def clear_all_state():
    """Clear all state (for testing/admin)."""
    global STATE
    STATE.clear()
    logger.info("Cleared all matrix state")


def get_matrix_count() -> int:
    """Get number of matrices in memory."""
    return len(STATE)


@router.post("/dual/init")
def dual_matrix_init():
    """Initialize dual matrix system for comparative analysis"""
    # Create two matrices for dual comparison
    rho_id_1 = str(uuid.uuid4())
    rho_id_2 = str(uuid.uuid4())
    
    # Create maximally mixed states
    rho_1 = create_maximally_mixed_state()
    rho_2 = create_maximally_mixed_state()
    
    # Store both states
    STATE[rho_id_1] = {
        "rho": rho_1,
        "ops": [],
        "narratives": [],
        "label": f"dual_A_{rho_id_1[:8]}",
        "created_at": None,
        "dual_pair": rho_id_2
    }
    
    STATE[rho_id_2] = {
        "rho": rho_2,
        "ops": [],
        "narratives": [],
        "label": f"dual_B_{rho_id_2[:8]}",
        "created_at": None,
        "dual_pair": rho_id_1
    }
    
    logger.info(f"Created dual matrix system: {rho_id_1} <-> {rho_id_2}")
    
    return {
        "dual_system_created": True,
        "matrix_a": rho_id_1,
        "matrix_b": rho_id_2,
        "diagnostics_a": diagnostics(rho_1),
        "diagnostics_b": diagnostics(rho_2)
    }


@router.get("/global/status")
def get_global_status():
    """Get global status of all matrices."""
    if not STATE:
        return {
            "has_matrices": False,
            "matrix_count": 0,
            "matrices": [],
            "meta": {
                "books_processed": 0,
                "total_operations": 0,
                "system_status": "empty"
            }
        }
    
    matrices = []
    # Create a copy to avoid race condition
    state_copy = dict(STATE)
    for rho_id, item in state_copy.items():
        try:
            diag = diagnostics(item["rho"])
            matrices.append({
                "rho_id": rho_id,
                "label": item.get("label", f"rho_{rho_id[:8]}"),
                "narratives_count": len(item.get("narratives", [])),
                "operations_count": len(item.get("ops", [])),
                "purity": diag["purity"],
                "entropy": diag["entropy"],
                "eigenvals": diag["eigenvals"][:5]  # Top 5 eigenvalues
            })
        except Exception as e:
            logger.error(f"Error processing rho {rho_id}: {e}")
            continue
    
    # Calculate composite matrix state from all matrices
    total_narratives = sum(len(item.get("narratives", [])) for item in state_copy.values())
    total_operations = sum(len(item.get("ops", [])) for item in state_copy.values())
    
    # Composite statistics from all matrices
    if matrices:
        avg_purity = sum(m["purity"] for m in matrices) / len(matrices)
        avg_entropy = sum(m["entropy"] for m in matrices) / len(matrices)
        # Get eigenvalues from the first matrix as representative
        composite_eigenvals = matrices[0]["eigenvals"] if matrices else []
    else:
        avg_purity = 0.0
        avg_entropy = 0.0
        composite_eigenvals = []
    
    return {
        "has_matrices": True,
        "matrix_count": len(STATE),
        "matrices": matrices,
        "meta": {
            "books_processed": total_narratives,
            "total_operations": total_operations,
            "total_chunks": total_narratives,  # For compatibility
            "total_tokens": total_narratives * 100,  # Estimate
            "system_status": "operational"
        },
        "matrix_state": {
            "purity": avg_purity,
            "entropy": avg_entropy,
            "eigenvals": composite_eigenvals
        },
        "processing_queue": {
            "processing": 0,
            "queued": 0
        }
    }