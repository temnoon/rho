"""
Unified POVM-Attribute System for the Rho Quantum Narrative System.

This module treats POVMs as attributes, creating a coherent system where:
- Each POVM axis defines an attribute with dialectical poles
- Users can adjust along these dimensions via sliders
- An LLM agent translates natural language requests to POVM adjustments
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from routes.povm_routes import PACKS
from routes.matrix_routes import STATE
from core.llm_integration import generate_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/povm-attributes", tags=["povm-attributes"])


class AttributeAdjustmentRequest(BaseModel):
    """Request to adjust attributes via natural language or direct values"""
    rho_id: str
    natural_language_request: Optional[str] = None  # e.g., "make this more formal and distant"
    direct_adjustments: Optional[Dict[str, float]] = None  # e.g., {"formality": 0.8, "narrative_distance": 0.6}
    strength: float = 0.5  # How strong the adjustment should be (0-1)


class AttributeState(BaseModel):
    """Current state of all attributes for a given rho"""
    rho_id: str
    attributes: Dict[str, float]  # attribute_name -> current_value (0-1 scale)
    measurements: Dict[str, Any]  # Raw POVM measurement results


@router.get("/available")
def get_available_attributes():
    """Get all available attributes derived from POVM packs"""
    attributes = {}
    
    for pack_id, pack_data in PACKS.items():
        pack_attributes = []
        
        for axis in pack_data.get("axes", []):
            axis_id = axis.get("id")
            if axis_id:
                # Create attribute from POVM axis
                attribute = {
                    "id": axis_id,
                    "name": axis_id.replace("_", " ").title(),
                    "description": axis.get("description", ""),
                    "pack_id": pack_id,
                    "poles": axis.get("labels", ["low", "high"]),
                    "category": axis.get("category", "uncategorized"),
                    "dialectic": {
                        "negative_pole": axis.get("labels", ["low", "high"])[0],
                        "positive_pole": axis.get("labels", ["low", "high"])[1],
                        "dimension": axis_id
                    }
                }
                pack_attributes.append(attribute)
        
        if pack_attributes:
            attributes[pack_id] = {
                "pack_name": pack_data.get("description", pack_id),
                "attributes": pack_attributes,
                "count": len(pack_attributes)
            }
    
    return {
        "attribute_packs": attributes,
        "total_attributes": sum(len(pack["attributes"]) for pack in attributes.values())
    }


@router.get("/{rho_id}/state")
def get_attribute_state(rho_id: str):
    """Get current attribute state for a rho by measuring all POVMs"""
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # Measure all POVM packs to get current attribute values
    from routes.povm_routes import measure_rho
    
    all_measurements = {}
    all_attributes = {}
    
    for pack_id in PACKS.keys():
        try:
            # Measure this POVM pack
            measurement_result = measure_rho(rho_id, pack_id)
            all_measurements[pack_id] = measurement_result
            
            # Extract attribute values from measurements
            if "measurements" in measurement_result:
                for axis_id, measurement in measurement_result["measurements"].items():
                    # Convert measurement probabilities to attribute value (0-1 scale)
                    if len(measurement) >= 2:
                        # Use the positive pole probability as the attribute value
                        all_attributes[axis_id] = float(measurement[1])
                    else:
                        all_attributes[axis_id] = 0.5  # Neutral
                        
        except Exception as e:
            logger.error(f"Failed to measure pack {pack_id} for rho {rho_id}: {e}")
            continue
    
    return AttributeState(
        rho_id=rho_id,
        attributes=all_attributes,
        measurements=all_measurements
    )


@router.post("/adjust")
async def adjust_attributes(request: AttributeAdjustmentRequest):
    """Adjust attributes via natural language or direct values"""
    if request.rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # If natural language request, use LLM to translate to adjustments
    if request.natural_language_request:
        adjustments = await translate_natural_language_to_adjustments(
            request.natural_language_request, 
            request.rho_id
        )
    elif request.direct_adjustments:
        adjustments = request.direct_adjustments
    else:
        raise HTTPException(status_code=400, detail="Must provide either natural_language_request or direct_adjustments")
    
    # Apply adjustments by steering the quantum state
    results = {}
    
    for attribute_id, target_value in adjustments.items():
        try:
            # Find which POVM pack contains this attribute
            pack_id = None
            for pid, pack_data in PACKS.items():
                for axis in pack_data.get("axes", []):
                    if axis.get("id") == attribute_id:
                        pack_id = pid
                        break
                if pack_id:
                    break
            
            if not pack_id:
                logger.warning(f"Attribute {attribute_id} not found in any POVM pack")
                continue
            
            # Create steering target for this attribute
            steering_target = {attribute_id: target_value}
            
            # Apply quantum steering to adjust this attribute
            from routes.advanced_routes import steer_rho
            steering_result = steer_rho(
                request.rho_id, 
                {"target_measurements": steering_target, "strength": request.strength}
            )
            
            results[attribute_id] = {
                "target_value": target_value,
                "steering_applied": True,
                "rmse": steering_result.get("rmse", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to adjust attribute {attribute_id}: {e}")
            results[attribute_id] = {"error": str(e)}
    
    # Get updated attribute state
    updated_state = get_attribute_state(request.rho_id)
    
    return {
        "rho_id": request.rho_id,
        "adjustment_results": results,
        "updated_attributes": updated_state.attributes,
        "natural_language_request": request.natural_language_request
    }


async def translate_natural_language_to_adjustments(request: str, rho_id: str) -> Dict[str, float]:
    """Use LLM to translate natural language requests to specific attribute adjustments"""
    
    # Get current attribute state for context
    try:
        current_state = get_attribute_state(rho_id)
        current_attributes = current_state.attributes
    except:
        current_attributes = {}
    
    # Get available attributes for context
    available_attrs = get_available_attributes()
    
    # Build context for the LLM
    attribute_descriptions = []
    for pack in available_attrs["attribute_packs"].values():
        for attr in pack["attributes"]:
            current_val = current_attributes.get(attr["id"], 0.5)
            attribute_descriptions.append(
                f"- {attr['name']} ({attr['id']}): {attr['description']} "
                f"[{attr['dialectic']['negative_pole']} ← {current_val:.2f} → {attr['dialectic']['positive_pole']}]"
            )
    
    # Create prompt for LLM
    prompt = f"""You are an expert at translating natural language requests into specific attribute adjustments for a quantum narrative system.

Current Narrative Attributes:
{chr(10).join(attribute_descriptions)}

User Request: "{request}"

Based on this request, provide specific attribute adjustments as a JSON object where:
- Keys are attribute IDs (like "formality", "narrative_distance", etc.)
- Values are target levels from 0.0 to 1.0 (where 0.0 = negative pole, 1.0 = positive pole)
- Only include attributes that should be adjusted based on the request
- Consider the current values when deciding on targets

Response format (JSON only, no explanation):
{{"attribute_id": target_value, "another_attribute": target_value}}"""
    
    try:
        # Generate response using existing LLM integration
        llm_response = await generate_text(prompt, max_tokens=200)
        
        # Parse JSON response
        import json
        adjustments = json.loads(llm_response.strip())
        
        # Validate adjustments
        validated_adjustments = {}
        for attr_id, value in adjustments.items():
            if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                validated_adjustments[attr_id] = float(value)
            else:
                logger.warning(f"Invalid adjustment value for {attr_id}: {value}")
        
        return validated_adjustments
        
    except Exception as e:
        logger.error(f"Failed to translate natural language request: {e}")
        # Fallback: provide reasonable defaults based on common requests
        return parse_common_requests(request)


def parse_common_requests(request: str) -> Dict[str, float]:
    """Fallback parser for common attribute adjustment requests"""
    request_lower = request.lower()
    adjustments = {}
    
    # Common patterns and their attribute mappings
    patterns = {
        "more formal": {"formality": 0.8},
        "less formal": {"formality": 0.2},
        "more distant": {"narrative_distance": 0.8},
        "less distant": {"narrative_distance": 0.2},
        "more emotional": {"affect": 0.8},
        "less emotional": {"affect": 0.2},
        "more elaborate": {"elaboration": 0.8},
        "simpler": {"elaboration": 0.2},
        "more certain": {"certainty": 0.8},
        "less certain": {"certainty": 0.2}
    }
    
    for pattern, attrs in patterns.items():
        if pattern in request_lower:
            adjustments.update(attrs)
    
    return adjustments


@router.get("/{rho_id}/impact_analysis")
def analyze_attribute_impact(rho_id: str):
    """Analyze which attributes have the most impact on the current quantum state"""
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    # Get current state
    current_rho = STATE[rho_id]["rho"]
    
    # Test small perturbations for each attribute to measure sensitivity
    impact_scores = {}
    
    # Get available attributes
    available = get_available_attributes()
    all_attributes = []
    for pack in available["attribute_packs"].values():
        for attr in pack["attributes"]:
            all_attributes.append(attr["id"])
    
    for attr_id in all_attributes:
        try:
            # Create small perturbation (±0.1)
            test_adjustments = {attr_id: 0.1}
            
            # Test how much the quantum state would change
            # For now, use a simple heuristic based on the attribute category
            base_impact = calculate_base_impact(attr_id)
            
            # TODO: In a more sophisticated implementation, we would:
            # 1. Apply small steering operation
            # 2. Measure the resulting change in quantum state
            # 3. Calculate the magnitude of change (Frobenius norm, trace distance, etc.)
            
            impact_scores[attr_id] = base_impact
            
        except Exception as e:
            logger.warning(f"Failed to analyze impact for {attr_id}: {e}")
            impact_scores[attr_id] = 0.5  # Default medium impact
    
    # Normalize scores to 0-1 range
    if impact_scores:
        max_score = max(impact_scores.values())
        min_score = min(impact_scores.values())
        if max_score > min_score:
            for attr_id in impact_scores:
                impact_scores[attr_id] = (impact_scores[attr_id] - min_score) / (max_score - min_score)
    
    # Categorize impact levels
    impact_categories = {}
    for attr_id, score in impact_scores.items():
        if score >= 0.7:
            impact_categories[attr_id] = {"level": "high", "color": "#ff4444", "priority": 1}
        elif score >= 0.4:
            impact_categories[attr_id] = {"level": "medium", "color": "#ff9800", "priority": 2}
        else:
            impact_categories[attr_id] = {"level": "low", "color": "#9e9e9e", "priority": 3}
    
    return {
        "rho_id": rho_id,
        "impact_scores": impact_scores,
        "impact_categories": impact_categories,
        "analysis_method": "base_heuristic",  # Will be "quantum_perturbation" when fully implemented
        "recommendations": generate_impact_recommendations(impact_categories)
    }


def calculate_base_impact(attr_id: str) -> float:
    """Calculate base impact score for an attribute based on linguistic research"""
    
    # High-impact attributes that significantly affect narrative style
    high_impact_attrs = {
        "involved_production": 0.9,  # Personal vs impersonal - major style difference
        "tenor_formality": 0.85,     # Formal vs informal - very noticeable
        "tenor_affect": 0.8,         # Emotional intensity - significant impact
        "narrative_concerns": 0.75,  # Narrative vs non-narrative - structural change
    }
    
    # Medium-impact attributes  
    medium_impact_attrs = {
        "elaborated_reference": 0.6,  # Detailed vs minimal reference
        "temporal_perspective": 0.55, # Past vs present perspective
    }
    
    # Check for matches
    if attr_id in high_impact_attrs:
        return high_impact_attrs[attr_id]
    elif attr_id in medium_impact_attrs:
        return medium_impact_attrs[attr_id]
    else:
        # Default for unknown attributes - assume medium impact
        return 0.5


def generate_impact_recommendations(impact_categories: Dict[str, Dict]) -> List[str]:
    """Generate recommendations based on impact analysis"""
    recommendations = []
    
    high_impact = [attr for attr, data in impact_categories.items() if data["level"] == "high"]
    medium_impact = [attr for attr, data in impact_categories.items() if data["level"] == "medium"]
    low_impact = [attr for attr, data in impact_categories.items() if data["level"] == "low"]
    
    if high_impact:
        recommendations.append(f"🔴 Focus on {', '.join(high_impact[:3])} - these have the strongest effect on narrative style")
    
    if medium_impact:
        recommendations.append(f"🟡 {', '.join(medium_impact[:3])} provide moderate stylistic changes")
    
    if low_impact:
        recommendations.append(f"⚪ {', '.join(low_impact[:3])} have subtle effects - use for fine-tuning")
    
    recommendations.append("💡 Try adjusting high-impact attributes first for more noticeable transformations")
    
    return recommendations


@router.post("/{rho_id}/regenerate")
async def regenerate_with_attributes(rho_id: str, request: AttributeAdjustmentRequest):
    """Adjust attributes and regenerate narrative in one operation"""
    
    # First adjust the attributes
    adjustment_result = await adjust_attributes(request)
    
    # Then regenerate the narrative
    from routes.advanced_routes import regenerate_narrative
    
    # Get the current narrative from the rho
    rho_state = STATE.get(rho_id)
    if not rho_state or not rho_state.get("narratives"):
        raise HTTPException(status_code=400, detail="No narrative found to regenerate")
    
    # Use the most recent narrative
    current_narrative = rho_state["narratives"][-1]["text"]
    
    # Regenerate with the adjusted quantum state
    regeneration_result = await regenerate_narrative({
        "rho_id": rho_id,
        "original_text": current_narrative,
        "preserve_meaning": True,
        "temperature": 0.7
    })
    
    return {
        "rho_id": rho_id,
        "attribute_adjustments": adjustment_result["adjustment_results"],
        "updated_attributes": adjustment_result["updated_attributes"],
        "original_narrative": current_narrative,
        "regenerated_narrative": regeneration_result.get("generated_text", ""),
        "natural_language_request": request.natural_language_request
    }