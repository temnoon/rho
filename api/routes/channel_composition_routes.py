"""
Channel Composition API Routes

Provides REST endpoints for composing quantum channels in complex ways.
Supports sequential, convex, conditional, and interpolated compositions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional, Any
import logging

from core.channel_composition import (
    CHANNEL_COMPOSER, 
    create_composition_from_texts,
    CompositionType
)

router = APIRouter(prefix="/composition", tags=["channel-composition"])
logger = logging.getLogger(__name__)


class RegisterChannelRequest(BaseModel):
    """Request to register a base channel."""
    channel_id: str
    text: str
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('channel_type')
    @classmethod
    def validate_channel_type(cls, v):
        valid_types = ["rank_one_update", "coherent_rotation", "dephasing_mixture"]
        if v not in valid_types:
            raise ValueError(f"Channel type must be one of: {valid_types}")
        return v
    
    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        return v


class SequentialCompositionRequest(BaseModel):
    """Request to create sequential composition."""
    channel_ids: List[str]
    composition_id: Optional[str] = None
    
    @field_validator('channel_ids')
    @classmethod
    def validate_channel_ids(cls, v):
        if len(v) < 2:
            raise ValueError("Sequential composition requires at least 2 channels")
        return v


class ConvexCompositionRequest(BaseModel):
    """Request to create convex combination."""
    channel_weights: Dict[str, float]
    composition_id: Optional[str] = None
    
    @field_validator('channel_weights')
    @classmethod
    def validate_weights(cls, v):
        if len(v) < 2:
            raise ValueError("Convex combination requires at least 2 channels")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total}, will be normalized")
        return v


class ConditionalCompositionRequest(BaseModel):
    """Request to create conditional composition."""
    condition_channel_id: str
    true_channel_id: str
    false_channel_id: str
    threshold: float = 0.5
    composition_id: Optional[str] = None
    
    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


class InterpolationRequest(BaseModel):
    """Request to create interpolated composition."""
    channel_id_1: str
    channel_id_2: str
    interpolation_parameter: float
    composition_id: Optional[str] = None
    
    @field_validator('interpolation_parameter')
    @classmethod
    def validate_interpolation(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Interpolation parameter must be between 0.0 and 1.0")
        return v


class CompositionFromTextsRequest(BaseModel):
    """Request to create composition from texts."""
    texts: List[str]
    composition_type: str = "sequential"
    weights: Optional[List[float]] = None
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 texts for composition")
        return [t.strip() for t in v if t.strip()]
    
    @field_validator('composition_type')
    @classmethod
    def validate_composition_type(cls, v):
        if v not in ["sequential", "convex"]:
            raise ValueError("Composition type must be 'sequential' or 'convex'")
        return v


class ApplyCompositionRequest(BaseModel):
    """Request to apply composed channel to a quantum state."""
    composition_id: str
    rho_id: str  # ID of quantum state to apply to


@router.post("/register_channel")
async def register_base_channel(request: RegisterChannelRequest):
    """
    Register a base channel in the composition library.
    """
    try:
        node = CHANNEL_COMPOSER.register_base_channel(
            channel_id=request.channel_id,
            text=request.text,
            channel_type=request.channel_type,
            alpha=request.alpha,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "channel_id": node.node_id,
            "channel_type": node.channel_type,
            "parameters": node.parameters,
            "created_at": node.created_at,
            "message": f"Registered base channel {request.channel_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to register channel {request.channel_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/sequential")
async def create_sequential_composition(request: SequentialCompositionRequest):
    """
    Create a sequential composition: Φₙ ∘ ... ∘ Φ₂ ∘ Φ₁
    """
    try:
        composed = CHANNEL_COMPOSER.create_sequential_composition(
            channel_ids=request.channel_ids,
            composition_id=request.composition_id
        )
        
        return {
            "success": True,
            "composition_id": composed.channel_id,
            "composition_type": "sequential",
            "num_channels": len(request.channel_ids),
            "input_channels": request.channel_ids,
            "composition_graph": composed.composition_graph,
            "metadata": composed.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to create sequential composition: {e}")
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@router.post("/convex")
async def create_convex_combination(request: ConvexCompositionRequest):
    """
    Create a convex combination: Σᵢ λᵢ Φᵢ where Σᵢ λᵢ = 1
    """
    try:
        composed = CHANNEL_COMPOSER.create_convex_combination(
            channel_weights=request.channel_weights,
            composition_id=request.composition_id
        )
        
        return {
            "success": True,
            "composition_id": composed.channel_id,
            "composition_type": "convex",
            "num_channels": len(request.channel_weights),
            "channel_weights": request.channel_weights,
            "composition_graph": composed.composition_graph,
            "metadata": composed.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to create convex combination: {e}")
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@router.post("/conditional")
async def create_conditional_composition(request: ConditionalCompositionRequest):
    """
    Create a conditional composition based on measurement outcome.
    """
    try:
        composed = CHANNEL_COMPOSER.create_conditional_composition(
            condition_channel_id=request.condition_channel_id,
            true_channel_id=request.true_channel_id,
            false_channel_id=request.false_channel_id,
            threshold=request.threshold,
            composition_id=request.composition_id
        )
        
        return {
            "success": True,
            "composition_id": composed.channel_id,
            "composition_type": "conditional",
            "condition_channel": request.condition_channel_id,
            "true_branch": request.true_channel_id,
            "false_branch": request.false_channel_id,
            "threshold": request.threshold,
            "composition_graph": composed.composition_graph,
            "metadata": composed.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to create conditional composition: {e}")
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@router.post("/interpolate")
async def create_interpolated_composition(request: InterpolationRequest):
    """
    Create an interpolated composition between two channels.
    """
    try:
        composed = CHANNEL_COMPOSER.create_interpolated_composition(
            channel_id_1=request.channel_id_1,
            channel_id_2=request.channel_id_2,
            interpolation_parameter=request.interpolation_parameter,
            composition_id=request.composition_id
        )
        
        return {
            "success": True,
            "composition_id": composed.channel_id,
            "composition_type": "interpolated",
            "channel_1": request.channel_id_1,
            "channel_2": request.channel_id_2,
            "interpolation_parameter": request.interpolation_parameter,
            "composition_graph": composed.composition_graph,
            "metadata": composed.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to create interpolated composition: {e}")
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@router.post("/from_texts")
async def create_composition_from_texts_endpoint(request: CompositionFromTextsRequest):
    """
    Create a channel composition directly from a list of texts.
    """
    try:
        composition_id = create_composition_from_texts(
            texts=request.texts,
            composition_type=request.composition_type,
            weights=request.weights,
            channel_type=request.channel_type,
            alpha=request.alpha
        )
        
        # Get composition info
        info = CHANNEL_COMPOSER.get_composition_info(composition_id)
        
        return {
            "success": True,
            "composition_id": composition_id,
            "composition_type": request.composition_type,
            "num_texts": len(request.texts),
            "text_previews": [text[:50] + "..." if len(text) > 50 else text for text in request.texts],
            "composition_info": info
        }
        
    except Exception as e:
        logger.error(f"Failed to create composition from texts: {e}")
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@router.post("/apply/{composition_id}")
async def apply_composed_channel(composition_id: str, request: ApplyCompositionRequest):
    """
    Apply a composed channel to a quantum state.
    """
    try:
        # Get the quantum state
        from routes.matrix_routes import STATE
        
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail=f"Quantum state {request.rho_id} not found")
        
        current_state = STATE[request.rho_id]["rho"]
        
        # Apply the composed channel
        new_state = CHANNEL_COMPOSER.apply_composed_channel(composition_id, current_state)
        
        # Update the quantum state
        STATE[request.rho_id]["rho"] = new_state
        STATE[request.rho_id]["ops"].append({
            "op": "apply_composed_channel",
            "composition_id": composition_id,
            "timestamp": time.time(),
            "summary": f"Applied composed channel {composition_id}"
        })
        
        # Get diagnostics
        from core.quantum_state import diagnostics
        diag = diagnostics(new_state)
        
        return {
            "success": True,
            "composition_id": composition_id,
            "rho_id": request.rho_id,
            "applied": True,
            "diagnostics": diag,
            "operation_recorded": True
        }
        
    except Exception as e:
        logger.error(f"Failed to apply composed channel {composition_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Application failed: {str(e)}")


@router.post("/audit/{composition_id}")
async def audit_composed_channel(composition_id: str):
    """
    Audit a composed channel for CPTP properties and correctness.
    """
    try:
        audit_results = CHANNEL_COMPOSER.audit_composed_channel(composition_id)
        
        return {
            "success": True,
            "composition_id": composition_id,
            "audit_results": audit_results,
            "passes_audit": audit_results.get("passes_audit", False),
            "recommendations": audit_results.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to audit composed channel {composition_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@router.get("/info/{composition_id}")
async def get_composition_info(composition_id: str):
    """
    Get detailed information about a composed channel.
    """
    try:
        info = CHANNEL_COMPOSER.get_composition_info(composition_id)
        return {
            "success": True,
            "composition_info": info
        }
        
    except Exception as e:
        logger.error(f"Failed to get composition info for {composition_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Composition not found: {str(e)}")


@router.get("/list")
async def list_compositions():
    """
    List all composed channels.
    """
    try:
        compositions = CHANNEL_COMPOSER.list_compositions()
        base_channels = CHANNEL_COMPOSER.list_base_channels()
        
        return {
            "success": True,
            "compositions": compositions,
            "base_channels": base_channels,
            "composition_count": len(compositions),
            "base_channel_count": len(base_channels)
        }
        
    except Exception as e:
        logger.error(f"Failed to list compositions: {e}")
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.get("/history")
async def get_composition_history(limit: int = 20):
    """
    Get composition operation history.
    """
    try:
        history = CHANNEL_COMPOSER.composition_history[-limit:] if CHANNEL_COMPOSER.composition_history else []
        
        return {
            "success": True,
            "history": history,
            "total_operations": len(CHANNEL_COMPOSER.composition_history),
            "returned_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get composition history: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@router.delete("/channel/{channel_id}")
async def delete_base_channel(channel_id: str):
    """
    Delete a base channel from the library.
    """
    try:
        if channel_id not in CHANNEL_COMPOSER.channel_library:
            raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")
        
        # Check if channel is used in any compositions
        used_in_compositions = []
        for comp_id, comp in CHANNEL_COMPOSER.composed_channels.items():
            if channel_id in comp.channel_nodes:
                used_in_compositions.append(comp_id)
        
        if used_in_compositions:
            return {
                "success": False,
                "error": "Channel is used in active compositions",
                "used_in_compositions": used_in_compositions,
                "message": "Delete compositions first or force delete"
            }
        
        # Delete the channel
        del CHANNEL_COMPOSER.channel_library[channel_id]
        
        return {
            "success": True,
            "deleted_channel": channel_id,
            "message": f"Base channel {channel_id} deleted"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete channel {channel_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.delete("/composition/{composition_id}")
async def delete_composition(composition_id: str):
    """
    Delete a composed channel.
    """
    try:
        if composition_id not in CHANNEL_COMPOSER.composed_channels:
            raise HTTPException(status_code=404, detail=f"Composition {composition_id} not found")
        
        # Get info before deletion
        info = CHANNEL_COMPOSER.get_composition_info(composition_id)
        
        # Delete the composition
        del CHANNEL_COMPOSER.composed_channels[composition_id]
        
        return {
            "success": True,
            "deleted_composition": composition_id,
            "composition_info": info,
            "message": f"Composed channel {composition_id} deleted"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete composition {composition_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


# Import time for operations that need timestamps
import time