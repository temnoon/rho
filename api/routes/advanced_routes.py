"""
Advanced quantum operations routes.

This module provides API endpoints for sophisticated quantum operations:
unitary steering, max-entropy projection, dynamic POVM generation,
and quantum channels for precise narrative control.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException

from core.quantum_operations import (
    unitary_steering,
    max_entropy_projection,
    depolarizing_channel,
    dephasing_channel,
    commutant_flow,
    style_channel
)
from core.dynamic_povm import (
    analyze_narrative_space,
    generate_optimal_povm_pack,
    optimize_povm_for_discrimination,
    evaluate_povm_efficiency
)
from core.quantum_state import diagnostics, psd_project
from models.requests import DiagnosticsResponse
from pydantic import BaseModel
from .matrix_routes import STATE
from .povm_routes import PACKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advanced", tags=["advanced"])


# === Request Models ===

class SteerRequest(BaseModel):
    target_attributes: Dict[str, float]
    attribute_pack_id: str = "demo_binary_pack"
    max_iterations: int = 20
    step_size: float = 0.1

class StyleChannelRequest(BaseModel):
    style_name: str
    strength: float = 1.0

class GenerateNarrativePOVMRequest(BaseModel):
    rho_id: str
    pack_name: str = "narrative_optimized"
    n_measurements: int = 12
    min_variance_threshold: float = 0.95


# === Dynamic POVM Generation ===

@router.post("/povm/generate_from_narrative")
def generate_narrative_povm(request: GenerateNarrativePOVMRequest):
    """
    Generate an optimal POVM pack based on the narratives in a density matrix.
    
    This implements the key insight that POVMs should be created AFTER narrative
    ingestion to ensure efficient use of the semantic space.
    """
    if request.rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    item = STATE[request.rho_id]
    narratives = item.get("narratives", [])
    
    if not narratives:
        raise HTTPException(status_code=400, detail="No narratives found for POVM generation")
    
    # Extract text samples from narratives
    texts = [narrative["text"] for narrative in narratives if narrative.get("text")]
    
    if len(texts) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 text samples for meaningful POVM generation")
    
    try:
        # Analyze the narrative space
        analysis = analyze_narrative_space(texts, request.min_variance_threshold)
        
        # Generate optimal POVM pack
        pack_data = generate_optimal_povm_pack(
            analysis, 
            pack_name=request.pack_name,
            n_measurements=request.n_measurements
        )
        
        # Store the pack
        PACKS[request.pack_name] = pack_data
        
        logger.info(f"Generated narrative-optimized POVM pack '{request.pack_name}' for rho {request.rho_id}")
        
        return {
            "created": True,
            "pack_id": request.pack_name,
            "rho_id": request.rho_id,
            "analysis_summary": {
                "n_texts": len(texts),
                "n_principal_components": len(analysis.principal_components),
                "variance_explained": analysis.variance_explained.tolist(),
                "semantic_clusters": {k: len(v) for k, v in analysis.semantic_clusters.items()},
                "n_measurements": len(pack_data["axes"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate narrative POVM: {e}")
        raise HTTPException(status_code=500, detail=f"POVM generation failed: {str(e)}")


@router.post("/povm/optimize_discrimination")
def optimize_discrimination_povm(
    rho_ids: List[str],
    pack_name: str = "discrimination_optimized",
    max_effects: int = 16
):
    """
    Create a POVM pack optimized for discriminating between specific density matrices.
    """
    if len(rho_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 matrices for discrimination")
    
    # Collect density matrices
    rho_states = []
    for rho_id in rho_ids:
        if rho_id not in STATE:
            raise HTTPException(status_code=404, detail=f"rho_id {rho_id} not found")
        rho_states.append(STATE[rho_id]["rho"])
    
    try:
        # Generate discriminating POVM
        effects = optimize_povm_for_discrimination(rho_states, max_effects=max_effects)
        
        if not effects:
            raise HTTPException(status_code=500, detail="Failed to generate discriminating POVM")
        
        # Create pack structure
        axes = []
        for i in range(0, len(effects), 2):
            if i + 1 < len(effects):
                axis = {
                    "id": f"discriminator_{i//2}",
                    "labels": [f"state_set_A", f"state_set_B"],
                    "effects": [effects[i].tolist(), effects[i+1].tolist()],
                    "type": "discrimination_optimized",
                    "description": f"Optimized for discriminating between density matrix sets"
                }
                axes.append(axis)
        
        pack_data = {
            "pack_id": pack_name,
            "axes": axes,
            "type": "discrimination_optimized",
            "description": f"POVM pack optimized for discriminating between {len(rho_ids)} density matrices",
            "source_matrices": rho_ids
        }
        
        PACKS[pack_name] = pack_data
        
        logger.info(f"Generated discrimination POVM pack '{pack_name}' for {len(rho_ids)} matrices")
        
        return {
            "created": True,
            "pack_id": pack_name,
            "source_matrices": rho_ids,
            "n_measurements": len(axes),
            "n_effects": len(effects)
        }
        
    except Exception as e:
        logger.error(f"Failed to generate discrimination POVM: {e}")
        raise HTTPException(status_code=500, detail=f"Discrimination POVM generation failed: {str(e)}")


@router.post("/povm/evaluate_efficiency")
def evaluate_povm_pack_efficiency(pack_id: str, test_rho_ids: List[str]):
    """
    Evaluate the efficiency of a POVM pack on test density matrices.
    """
    if pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="pack_id not found")
    
    # Collect test matrices
    test_states = []
    for rho_id in test_rho_ids:
        if rho_id not in STATE:
            raise HTTPException(status_code=404, detail=f"rho_id {rho_id} not found")
        test_states.append(STATE[rho_id]["rho"])
    
    # Extract effects from pack
    pack_data = PACKS[pack_id]
    effects = []
    
    for axis in pack_data["axes"]:
        if "effects" in axis:
            for effect_data in axis["effects"]:
                effects.append(np.array(effect_data))
    
    if not effects:
        raise HTTPException(status_code=400, detail="No effects found in POVM pack")
    
    # Evaluate efficiency
    efficiency_metrics = evaluate_povm_efficiency(effects, test_states)
    
    return {
        "pack_id": pack_id,
        "test_matrices": test_rho_ids,
        "efficiency_metrics": efficiency_metrics
    }


# === Unitary Steering ===

@router.post("/steer/{rho_id}")
def steer_density_matrix(rho_id: str, request: SteerRequest):
    """
    Apply unitary steering to adjust attribute expectations while preserving essence.
    
    This implements meaning-preserving retells that hit specific attribute targets.
    """
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    if request.attribute_pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="attribute_pack_id not found")
    
    current_rho = STATE[rho_id]["rho"]
    pack_data = PACKS[request.attribute_pack_id]
    
    # Convert POVM pack to attribute operators
    # For binary POVMs: A = E_+ - E_- (gives expectation difference)
    attribute_operators = {}
    
    for axis in pack_data["axes"]:
        axis_id = axis["id"]
        if axis_id in request.target_attributes and "effects" in axis:
            effects = axis["effects"]
            if len(effects) >= 2:
                E_neg = np.array(effects[0])
                E_pos = np.array(effects[1])
                # Attribute operator: difference between positive and negative
                A = E_pos - E_neg
                attribute_operators[axis_id] = A
    
    if not attribute_operators:
        raise HTTPException(status_code=400, detail="No valid attribute operators found")
    
    try:
        # Apply unitary steering
        steered_rho, final_attributes = unitary_steering(
            current_rho,
            request.target_attributes,
            attribute_operators,
            max_iterations=request.max_iterations,
            step_size=request.step_size
        )
        
        # Update stored matrix
        STATE[rho_id]["rho"] = steered_rho
        
        # Ensure ops key exists for legacy matrices
        if "ops" not in STATE[rho_id]:
            STATE[rho_id]["ops"] = []
        
        STATE[rho_id]["ops"].append({
            "op": "unitary_steer",
            "target_attributes": request.target_attributes,
            "final_attributes": final_attributes,
            "pack_id": request.attribute_pack_id,
            "summary": f"Unitary steering with {len(request.target_attributes)} attribute targets"
        })
        
        # Get diagnostics
        diag = diagnostics(steered_rho)
        
        logger.info(f"Applied unitary steering to rho {rho_id}")
        
        return {
            "success": True,
            "rho_id": rho_id,
            "target_attributes": request.target_attributes,
            "final_attributes": final_attributes,
            "diagnostics": DiagnosticsResponse(**diag)
        }
        
    except Exception as e:
        logger.error(f"Unitary steering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Unitary steering failed: {str(e)}")


# === Max-Entropy Projection ===

@router.post("/project_maxent/{rho_id}")
def project_max_entropy(
    rho_id: str,
    constraints: List[Dict[str, float]],  # [{"attribute": "reliability", "value": 0.8}, ...]
    attribute_pack_id: str = "demo_binary_pack",
    max_iterations: int = 50
):
    """
    Project density matrix to maximum entropy subject to attribute constraints.
    
    This finds the least-biased matrix that satisfies specific attribute requirements.
    """
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    if attribute_pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="attribute_pack_id not found")
    
    current_rho = STATE[rho_id]["rho"]
    pack_data = PACKS[attribute_pack_id]
    
    # Convert constraints to (operator, value) tuples
    constraint_tuples = []
    
    for constraint in constraints:
        if "attribute" not in constraint or "value" not in constraint:
            continue
            
        attr_name = constraint["attribute"]
        target_value = constraint["value"]
        
        # Find corresponding operator in pack
        for axis in pack_data["axes"]:
            if axis["id"] == attr_name and "effects" in axis:
                effects = axis["effects"]
                if len(effects) >= 2:
                    E_neg = np.array(effects[0])
                    E_pos = np.array(effects[1])
                    A = E_pos - E_neg  # Attribute operator
                    constraint_tuples.append((A, target_value))
                    break
    
    if not constraint_tuples:
        raise HTTPException(status_code=400, detail="No valid constraints found")
    
    try:
        # Apply max-entropy projection
        projected_rho = max_entropy_projection(
            current_rho,
            constraint_tuples,
            max_iterations=max_iterations
        )
        
        # Update stored matrix
        STATE[rho_id]["rho"] = projected_rho
        
        # Ensure ops key exists for legacy matrices
        if "ops" not in STATE[rho_id]:
            STATE[rho_id]["ops"] = []
        
        STATE[rho_id]["ops"].append({
            "op": "max_entropy_projection",
            "constraints": constraints,
            "pack_id": attribute_pack_id,
            "summary": f"Max-entropy projection with {len(constraints)} constraints"
        })
        
        # Verify final constraint satisfaction
        final_values = {}
        for constraint in constraints:
            attr_name = constraint["attribute"]
            for axis in pack_data["axes"]:
                if axis["id"] == attr_name and "effects" in axis:
                    effects = axis["effects"]
                    if len(effects) >= 2:
                        E_neg = np.array(effects[0])
                        E_pos = np.array(effects[1])
                        A = E_pos - E_neg
                        final_values[attr_name] = float(np.real(np.trace(projected_rho @ A)))
                        break
        
        diag = diagnostics(projected_rho)
        
        logger.info(f"Applied max-entropy projection to rho {rho_id}")
        
        return {
            "success": True,
            "rho_id": rho_id,
            "constraints": constraints,
            "final_values": final_values,
            "diagnostics": DiagnosticsResponse(**diag)
        }
        
    except Exception as e:
        logger.error(f"Max-entropy projection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Max-entropy projection failed: {str(e)}")


# === Quantum Channels ===

@router.post("/channel/depolarize/{rho_id}")
def apply_depolarizing_channel(
    rho_id: str,
    strength: float,
    preserve_attributes: Optional[List[str]] = None,
    attribute_pack_id: str = "demo_binary_pack"
):
    """
    Apply depolarizing channel to increase entropy and avoid ruts.
    
    Optionally preserve specific attributes by restricting to commutant subspace.
    """
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    strength = max(0.0, min(1.0, strength))
    current_rho = STATE[rho_id]["rho"]
    
    # Get attribute operators if preservation is requested
    preserve_ops = None
    if preserve_attributes and attribute_pack_id in PACKS:
        preserve_ops = []
        pack_data = PACKS[attribute_pack_id]
        
        for attr_name in preserve_attributes:
            for axis in pack_data["axes"]:
                if axis["id"] == attr_name and "effects" in axis:
                    effects = axis["effects"]
                    if len(effects) >= 2:
                        E_neg = np.array(effects[0])
                        E_pos = np.array(effects[1])
                        A = E_pos - E_neg
                        preserve_ops.append(A)
                        break
    
    try:
        # Apply depolarizing channel
        depolarized_rho = depolarizing_channel(current_rho, strength, preserve_ops)
        
        # Update stored matrix
        STATE[rho_id]["rho"] = depolarized_rho
        
        # Ensure ops key exists for legacy matrices
        if "ops" not in STATE[rho_id]:
            STATE[rho_id]["ops"] = []
        
        STATE[rho_id]["ops"].append({
            "op": "depolarizing_channel",
            "strength": strength,
            "preserved_attributes": preserve_attributes or [],
            "summary": f"Depolarization (p={strength:.3f})"
        })
        
        diag = diagnostics(depolarized_rho)
        
        logger.info(f"Applied depolarizing channel to rho {rho_id}")
        
        return {
            "success": True,
            "rho_id": rho_id,
            "strength": strength,
            "preserved_attributes": preserve_attributes or [],
            "diagnostics": DiagnosticsResponse(**diag)
        }
        
    except Exception as e:
        logger.error(f"Depolarizing channel failed: {e}")
        raise HTTPException(status_code=500, detail=f"Depolarizing channel failed: {str(e)}")


@router.post("/channel/style/{rho_id}")
def apply_style_channel(rho_id: str, request: StyleChannelRequest):
    """
    Apply a learned style transformation channel.
    
    Available styles: noir, romantic, minimalist
    """
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    strength = max(0.0, min(1.0, request.strength))
    current_rho = STATE[rho_id]["rho"]
    
    valid_styles = ["noir", "romantic", "minimalist"]
    if request.style_name not in valid_styles:
        raise HTTPException(status_code=400, detail=f"Invalid style. Available: {valid_styles}")
    
    try:
        # Apply style channel
        styled_rho = style_channel(current_rho, request.style_name, strength)
        
        # Update stored matrix
        STATE[rho_id]["rho"] = styled_rho
        
        # Ensure ops key exists for legacy matrices
        if "ops" not in STATE[rho_id]:
            STATE[rho_id]["ops"] = []
        
        STATE[rho_id]["ops"].append({
            "op": "style_channel",
            "style_name": request.style_name,
            "strength": strength,
            "summary": f"Applied {request.style_name} style (strength={strength:.3f})"
        })
        
        diag = diagnostics(styled_rho)
        
        logger.info(f"Applied {request.style_name} style channel to rho {rho_id}")
        
        return {
            "success": True,
            "rho_id": rho_id,
            "style_name": request.style_name,
            "strength": strength,
            "diagnostics": DiagnosticsResponse(**diag)
        }
        
    except Exception as e:
        logger.error(f"Style channel failed: {e}")
        raise HTTPException(status_code=500, detail=f"Style channel failed: {str(e)}")


# === POVM Creation ===

@router.post("/create-povm")
def create_custom_povm(
    pack_name: str,
    dialectical_concept: str,
    description: str = "",
    povm_type: str = "dialectical",
    n_measurements: int = 8
):
    """
    Create a custom POVM based on dialectical concepts.
    
    This endpoint allows users to define new POVMs that capture specific
    narrative dimensions expressed as dialectical tensions.
    """
    if pack_name in PACKS:
        raise HTTPException(status_code=400, detail=f"POVM pack '{pack_name}' already exists")
    
    try:
        from core.povm_operations import create_dialectical_povm
        
        # Parse dialectical concept to extract poles
        if '⟷' in dialectical_concept:
            left_pole, right_pole = dialectical_concept.split('⟷')
            left_pole = left_pole.strip()
            right_pole = right_pole.strip()
        elif '/' in dialectical_concept:
            left_pole, right_pole = dialectical_concept.split('/')
            left_pole = left_pole.strip()
            right_pole = right_pole.strip()
        else:
            # Default dialectical structure
            left_pole = f"Low {dialectical_concept}"
            right_pole = f"High {dialectical_concept}"
        
        # Create the POVM pack based on type
        if povm_type == "dialectical":
            pack_data = create_dialectical_povm(
                pack_name=pack_name,
                left_pole=left_pole,
                right_pole=right_pole,
                description=description,
                n_measurements=n_measurements
            )
        else:
            # Use generic POVM creation for other types
            from core.povm_operations import create_multiclass_povm, create_coverage_povm, create_attribute_povm
            
            if povm_type == "multiclass":
                pack_data = create_multiclass_povm(pack_name, n_measurements, description)
            elif povm_type == "coverage":
                pack_data = create_coverage_povm(pack_name, description)
            elif povm_type == "attribute":
                pack_data = create_attribute_povm(pack_name, dialectical_concept, description)
            else:
                raise ValueError(f"Unknown POVM type: {povm_type}")
        
        # Store the pack
        PACKS[pack_name] = pack_data
        
        logger.info(f"Created custom POVM pack '{pack_name}' of type '{povm_type}'")
        
        return {
            "created": True,
            "pack_id": pack_name,
            "pack_type": povm_type,
            "dialectical_concept": dialectical_concept,
            "left_pole": left_pole,
            "right_pole": right_pole,
            "description": description,
            "n_measurements": len(pack_data["axes"]),
            "pack_data": pack_data
        }
        
    except Exception as e:
        logger.error(f"Failed to create custom POVM: {e}")
        raise HTTPException(status_code=500, detail=f"POVM creation failed: {str(e)}")


@router.post("/generate-optimal-povm")
def generate_optimal_povm_endpoint(
    pack_name: str,
    description: str = "",
    n_measurements: int = 8,
    rho_id: Optional[str] = None
):
    """
    Generate an optimal POVM based on current narrative content.
    
    This uses the dynamic POVM generation system to create measurements
    optimally suited to the loaded narrative content.
    """
    if pack_name in PACKS:
        raise HTTPException(status_code=400, detail=f"POVM pack '{pack_name}' already exists")
    
    try:
        # If rho_id provided, use its narratives
        if rho_id and rho_id in STATE:
            item = STATE[rho_id]
            narratives = item.get("narratives", [])
            texts = [narrative["text"] for narrative in narratives if narrative.get("text")]
        else:
            # Use narratives from all loaded matrices
            texts = []
            for item in STATE.values():
                narratives = item.get("narratives", [])
                texts.extend([narrative["text"] for narrative in narratives if narrative.get("text")])
        
        if len(texts) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 text samples for optimal POVM generation")
        
        # Analyze narrative space and generate optimal POVM
        analysis = analyze_narrative_space(texts, min_variance_threshold=0.90)
        pack_data = generate_optimal_povm_pack(
            analysis,
            pack_name=pack_name,
            n_measurements=n_measurements
        )
        
        # Store the pack
        PACKS[pack_name] = pack_data
        
        logger.info(f"Generated optimal POVM pack '{pack_name}' from {len(texts)} texts")
        
        return {
            "created": True,
            "pack_id": pack_name,
            "pack_type": "optimal",
            "description": description,
            "n_texts_analyzed": len(texts),
            "n_measurements": len(pack_data["axes"]),
            "analysis_summary": {
                "n_principal_components": len(analysis.principal_components),
                "variance_explained": analysis.variance_explained.tolist()[:5],  # First 5 components
                "semantic_clusters": {k: len(v) for k, v in analysis.semantic_clusters.items()}
            },
            "pack_data": pack_data
        }
        
    except Exception as e:
        logger.error(f"Failed to generate optimal POVM: {e}")
        raise HTTPException(status_code=500, detail=f"Optimal POVM generation failed: {str(e)}")


# === Utility Functions ===

@router.get("/capabilities")
def get_advanced_capabilities():
    """Get information about available advanced operations."""
    return {
        "dynamic_povm": {
            "description": "Generate POVMs optimized for specific narrative content",
            "operations": ["generate_from_narrative", "optimize_discrimination", "evaluate_efficiency"]
        },
        "unitary_steering": {
            "description": "Meaning-preserving adjustments to hit attribute targets",
            "operations": ["steer"]
        },
        "max_entropy": {
            "description": "Find least-biased matrix satisfying constraints",
            "operations": ["project_maxent"]
        },
        "quantum_channels": {
            "description": "Transformation channels for style and decoherence",
            "operations": ["depolarize", "style"],
            "available_styles": ["noir", "romantic", "minimalist"]
        }
    }


class RegenerateNarrativeRequest(BaseModel):
    original_text: str
    adjusted_rho_id: str


@router.post("/regenerate_narrative")
def regenerate_narrative(request: RegenerateNarrativeRequest):
    """Regenerate narrative based on current quantum state measurements."""
    
    if not request.original_text.strip():
        raise HTTPException(status_code=400, detail="No original text provided")
    
    if not request.adjusted_rho_id or request.adjusted_rho_id not in STATE:
        raise HTTPException(status_code=400, detail="Invalid or missing rho_id")
    
    # Import LLM integration
    try:
        from core.llm_integration import get_narrative_transformer, QuantumNarrativeContext
        from core.quantum_state import diagnostics
        
        # Get the best available LLM client
        try:
            transformer = get_narrative_transformer()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"No LLM service available: {str(e)}"
            )
        
        # Get quantum state and measurements
        rho_data = STATE[request.adjusted_rho_id]
        rho_matrix = rho_data["rho"]
        
        # Get current POVM measurements - always get fresh measurements for accurate transformation
        current_measurements = {}
        used_pack_id = None
        
        # First try to determine which pack was used from recent operations
        if "ops" in rho_data:
            for op in reversed(rho_data["ops"]):
                if op.get("op") == "measure" and "pack_id" in op:
                    used_pack_id = op["pack_id"]
                    break
        
        # If no pack ID found, use preferred packs
        if not used_pack_id:
            from .povm_routes import PACKS
            if not PACKS:
                from .povm_routes import init_demo_packs
                init_demo_packs()
            
            # Use the first available pack, preferring narrative-related ones
            preferred_packs = ["advanced_narrative_pack", "demo_binary_pack"]
            for pack_id in preferred_packs:
                if pack_id in PACKS:
                    used_pack_id = pack_id
                    break
            
            if not used_pack_id and PACKS:
                used_pack_id = list(PACKS.keys())[0]
        
        # Always get fresh measurements to ensure we have the latest state
        if used_pack_id:
            from .povm_routes import measure_rho
            from models.requests import MeasureReq
            measurements_result = measure_rho(request.adjusted_rho_id, MeasureReq(pack_id=used_pack_id))
            current_measurements = measurements_result.measurements
            logger.info(f"Retrieved fresh measurements for rho {request.adjusted_rho_id} using pack {used_pack_id}: {len(current_measurements)} measurements")
        else:
            logger.warning(f"No POVM pack available for measurements in rho {request.adjusted_rho_id}")
            current_measurements = {}
        
        # Get quantum diagnostics
        quantum_diagnostics = diagnostics(rho_matrix)
        
        # Extract target attributes from recent operations
        target_attributes = {}
        if "ops" in rho_data:
            # Look for the most recent unitary steering operation
            for op in reversed(rho_data["ops"]):  # Check recent operations first
                if op.get("op") == "unitary_steer" and "target_attributes" in op:
                    target_attributes = op["target_attributes"]
                    break
        
        # Create context for LLM generation
        context = QuantumNarrativeContext(
            original_text=request.original_text,
            target_attributes=target_attributes,
            current_measurements=current_measurements,
            quantum_diagnostics=quantum_diagnostics
        )
        
        # Generate transformed narrative
        try:
            transformed_text = transformer.generate_narrative_transformation(context, used_pack_id)
            
            logger.info(f"Generated narrative transformation for rho {request.adjusted_rho_id}")
            
            # Get model name/info for response
            model_info = getattr(transformer, 'model', transformer.__class__.__name__)
            
            return {
                "success": True,
                "transformed_text": transformed_text,
                "original_text": request.original_text,
                "rho_id": request.adjusted_rho_id,
                "measurements_used": current_measurements,
                "model": model_info
            }
            
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate narrative transformation: {str(e)}"
            )
        
    except ImportError as e:
        logger.error(f"LLM integration import failed: {e}")
        raise HTTPException(
            status_code=501,
            detail="LLM integration module not available. Please ensure all dependencies are installed."
        )