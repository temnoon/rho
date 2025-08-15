"""
POVM measurement and pack management routes.

This module handles POVM pack creation, management, and measurement operations
for extracting interpretable attributes from quantum density matrices.
"""

import logging
import numpy as np
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from core.povm_operations import (
    create_binary_povm,
    create_multiclass_povm,
    measure_povm,
    create_coverage_povm,
    create_attribute_povm
)
from models.requests import (
    MeasureReq,
    PackModel,
    PackAxis,
    MeasurementResponse,
    DiagnosticsResponse,
    PackListResponse,
    PackInfo
)
from core.quantum_state import diagnostics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/packs", tags=["povm"])

# Global POVM pack storage
PACKS: Dict[str, Dict[str, Any]] = {}

# Import STATE from matrix routes for measurements
from .matrix_routes import STATE

DIM = 64  # Hilbert space dimension


def init_demo_packs() -> None:
    """Initialize demonstration POVM packs."""
    if PACKS:
        return  # Already initialized
    
    # Create demo axes pack
    axes = []
    
    # Binary attributes with basis vectors
    binary_attributes = [
        ("reliability", ["unreliable", "reliable"]),
        ("formality", ["informal", "formal"]),
        ("certainty", ["uncertain", "certain"]),
        ("personal_focus", ["impersonal", "personal"]),
        ("emotional_intensity", ["subdued", "intense"]),
        ("temporal_focus", ["timeless", "time-bound"])
    ]
    
    for i, (attr_name, labels) in enumerate(binary_attributes):
        # Create basis vector for this attribute
        basis_vector = np.zeros(DIM)
        basis_vector[i] = 1.0  # Use orthogonal basis
        
        axes.append({
            "id": attr_name,
            "labels": labels,
            "basis_vector": basis_vector.tolist(),
            "type": "binary"
        })
    
    demo_pack = {
        "pack_id": "demo_binary_pack",
        "axes": axes,
        "description": "Demonstration binary attribute pack"
    }
    PACKS[demo_pack["pack_id"]] = demo_pack
    
    # Create coverage pack for general exploration
    coverage_effects = create_coverage_povm(n_effects=16)
    coverage_axes = []
    
    for i, effect in enumerate(coverage_effects):
        coverage_axes.append({
            "id": f"probe_{i:02d}",
            "labels": [f"low_{i}", f"high_{i}"],
            "effect_matrix": effect.tolist(),
            "type": "coverage"
        })
    
    coverage_pack = {
        "pack_id": "coverage_pack",
        "axes": coverage_axes,
        "description": "Coverage pack for space exploration"
    }
    PACKS[coverage_pack["pack_id"]] = coverage_pack
    
    logger.info(f"Initialized {len(PACKS)} demo POVM packs")


@router.get("")
def packs_list():
    """List all available POVM packs."""
    init_demo_packs()
    
    pack_info = []
    for pack_id, pack_data in PACKS.items():
        pack_info.append(PackInfo(
            pack_id=pack_id,
            description=pack_data.get("description", ""),
            num_axes=len(pack_data.get("axes", [])),
            type=pack_data.get("type", "custom")
        ))
    
    return PackListResponse(packs=pack_info)


@router.post("")
def packs_add(pack: PackModel):
    """Add a new POVM pack."""
    if pack.pack_id in PACKS:
        raise HTTPException(status_code=400, detail="pack_id already exists")
    
    # Validate and convert pack model
    axes = []
    for axis in pack.axes:
        axis_data = {
            "id": axis.id,
            "labels": axis.labels,
            "type": "custom"
        }
        
        # For now, create simple binary POVM for each axis
        if len(axis.labels) == 2:
            # Binary measurement
            basis_vector = np.random.randn(DIM)
            basis_vector = basis_vector / np.linalg.norm(basis_vector)
            
            E_pos, E_neg = create_binary_povm(basis_vector)
            axis_data["effects"] = [E_pos.tolist(), E_neg.tolist()]
            axis_data["type"] = "binary"
        else:
            # Multi-class measurement
            basis_vectors = []
            for _ in range(len(axis.labels)):
                v = np.random.randn(DIM)
                v = v / np.linalg.norm(v)
                basis_vectors.append(v)
            
            effects = create_multiclass_povm(basis_vectors)
            axis_data["effects"] = [E.tolist() for E in effects]
            axis_data["type"] = "multiclass"
        
        axes.append(axis_data)
    
    pack_data = {
        "pack_id": pack.pack_id,
        "axes": axes,
        "description": f"Custom pack with {len(axes)} axes",
        "type": "custom"
    }
    
    PACKS[pack.pack_id] = pack_data
    logger.info(f"Added new POVM pack: {pack.pack_id}")
    
    return {"created": True, "pack_id": pack.pack_id, "num_axes": len(axes)}


@router.post("/measure/{rho_id}")
def measure_rho(rho_id: str, req: MeasureReq):
    """Apply POVM measurements to a density matrix."""
    # Check if rho exists
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # Check if pack exists
    if req.pack_id not in PACKS:
        init_demo_packs()  # Try to initialize demo packs
        if req.pack_id not in PACKS:
            raise HTTPException(status_code=404, detail="pack_id not found")
    
    rho = STATE[rho_id]["rho"]
    pack_data = PACKS[req.pack_id]
    
    # Perform measurements
    measurements = {}
    
    for axis in pack_data["axes"]:
        axis_id = axis["id"]
        labels = axis["labels"]
        
        if "effects" in axis:
            # Use pre-computed effects
            effects = [np.array(E) for E in axis["effects"]]
        elif "basis_vector" in axis and axis.get("type") == "binary":
            # Compute binary POVM on the fly
            basis_vector = np.array(axis["basis_vector"])
            E_pos, E_neg = create_binary_povm(basis_vector)
            effects = [E_pos, E_neg]
        else:
            # Fallback: create random measurement
            v = np.random.randn(DIM)
            v = v / np.linalg.norm(v)
            E_pos, E_neg = create_binary_povm(v)
            effects = [E_pos, E_neg]
        
        # Measure and store results
        probs = measure_povm(rho, effects, labels)
        # Flatten the probability dictionary into individual measurements
        for label, prob in probs.items():
            measurements[f"{axis_id}_{label}"] = prob
    
    # Update operation history
    # Ensure ops key exists for legacy matrices
    if "ops" not in STATE[rho_id]:
        STATE[rho_id]["ops"] = []
    
    STATE[rho_id]["ops"].append({
        "op": "measure",
        "pack_id": req.pack_id,
        "summary": f"Measured POVM pack {req.pack_id}",
        "math": "for each axis u: p_plus = Tr(|u><u| rho); p_minus = 1 - p_plus"
    })
    
    # Get diagnostics
    diag = diagnostics(rho)
    
    logger.info(f"Applied POVM pack {req.pack_id} to rho {rho_id}")
    
    return MeasurementResponse(
        measurements=measurements,
        diagnostics=DiagnosticsResponse(**diag)
    )


@router.get("/{pack_id}")
def get_pack_details(pack_id: str):
    """Get detailed information about a specific POVM pack."""
    init_demo_packs()
    
    if pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="pack_id not found")
    
    pack_data = PACKS[pack_id]
    
    # Return pack info without large matrices
    pack_info = {
        "pack_id": pack_id,
        "description": pack_data.get("description", ""),
        "type": pack_data.get("type", "custom"),
        "axes": []
    }
    
    for axis in pack_data["axes"]:
        axis_info = {
            "id": axis["id"],
            "labels": axis["labels"],
            "type": axis.get("type", "unknown"),
            "num_effects": len(axis.get("effects", []))
        }
        pack_info["axes"].append(axis_info)
    
    return pack_info


@router.post("/create_attribute")
def create_attribute_pack(
    pack_id: str,
    attribute_name: str,
    positive_texts: List[str],
    negative_texts: List[str] = None
):
    """Create a POVM pack for a specific attribute from example texts."""
    if pack_id in PACKS:
        raise HTTPException(status_code=400, detail="pack_id already exists")
    
    # Convert texts to embedding vectors
    from ..core.embedding import embed, project_to_local
    
    positive_vectors = []
    for text in positive_texts:
        x = embed(text)
        v = project_to_local(x)
        positive_vectors.append(v)
    
    negative_vectors = []
    if negative_texts:
        for text in negative_texts:
            x = embed(text)
            v = project_to_local(x)
            negative_vectors.append(v)
    
    # Create attribute POVM
    E_pos, E_neg = create_attribute_povm(positive_vectors, negative_vectors)
    
    # Create pack
    axis_data = {
        "id": attribute_name,
        "labels": ["low", "high"],
        "effects": [E_neg.tolist(), E_pos.tolist()],
        "type": "learned_attribute"
    }
    
    pack_data = {
        "pack_id": pack_id,
        "axes": [axis_data],
        "description": f"Learned attribute pack for '{attribute_name}'",
        "type": "learned",
        "training_data": {
            "positive_texts": positive_texts,
            "negative_texts": negative_texts or []
        }
    }
    
    PACKS[pack_id] = pack_data
    logger.info(f"Created learned attribute pack {pack_id} for '{attribute_name}'")
    
    return {
        "created": True,
        "pack_id": pack_id,
        "attribute_name": attribute_name,
        "positive_examples": len(positive_texts),
        "negative_examples": len(negative_texts) if negative_texts else 0
    }


@router.delete("/{pack_id}")
def delete_pack(pack_id: str):
    """Delete a POVM pack."""
    if pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="pack_id not found")
    
    # Don't allow deletion of demo packs
    if pack_id in ["demo_binary_pack", "coverage_pack"]:
        raise HTTPException(status_code=400, detail="Cannot delete demo packs")
    
    pack_info = {
        "pack_id": pack_id,
        "description": PACKS[pack_id].get("description", ""),
        "num_axes": len(PACKS[pack_id].get("axes", []))
    }
    
    del PACKS[pack_id]
    logger.info(f"Deleted POVM pack {pack_id}")
    
    return {"deleted": True, "pack_info": pack_info}


# Utility functions
def get_packs_dict() -> Dict[str, Dict[str, Any]]:
    """Get current packs dictionary."""
    init_demo_packs()
    return PACKS


def clear_all_packs():
    """Clear all packs (for testing/admin)."""
    global PACKS
    PACKS.clear()
    logger.info("Cleared all POVM packs")


def get_pack_count() -> int:
    """Get number of packs in memory."""
    return len(PACKS)