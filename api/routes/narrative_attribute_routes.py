"""
API routes for Narrative Attribute Algebra system.

Provides endpoints for extracting, manipulating, and applying narrative attributes
through the quantum density matrix framework.
"""

import logging
import json
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.narrative_attributes import (
    NarrativeAttributeManager,
    NarrativeAttributeTaxonomy, 
    NarrativeSignature
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/narrative-attributes", tags=["narrative-attributes"])

# Global manager instance - initialized with Groq client when available
manager = None

def initialize_manager():
    """Initialize the narrative attribute manager with Groq client"""
    global manager
    if manager is None:
        try:
            from core.llm_integration import groq_client
            manager = NarrativeAttributeManager(groq_client)
            logger.info("Initialized NarrativeAttributeManager with Groq client")
        except Exception as e:
            logger.warning(f"Failed to initialize with Groq client: {e}")
            manager = NarrativeAttributeManager()
            logger.info("Initialized NarrativeAttributeManager without LLM client")

# Request/Response Models
class ExtractSignatureRequest(BaseModel):
    text: str
    signature_name: Optional[str] = None
    use_llm: bool = True

class ApplySignatureRequest(BaseModel):
    text: str
    signature_name: str
    strength: float = 1.0
    auto_balance: bool = True

class CreateSignatureRequest(BaseModel):
    name: str
    namespace: Dict[str, float]
    style: Dict[str, float]
    persona: Dict[str, float]
    description: str = ""

class SignatureOperationRequest(BaseModel):
    base_signature: str
    modifier_signature: str
    operation: str  # "add", "subtract", "blend"
    strength: float = 1.0
    blend_ratio: Optional[float] = 0.5  # for blend operation

# === Core Endpoints ===

@router.post("/transform-style")
async def transform_text_style(request: dict):
    """Transform text style using Groq while preserving meaning"""
    
    text = request.get("text", "")
    target_style = request.get("target_style", "formal academic")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get Groq client
        from core.llm_integration import groq_client
        
        style_prompt = f"""Transform the following text into {target_style} style while preserving all meaning, concepts, and factual content. Do not add new information or change the core ideas. Only adjust the language style and presentation.

Original text:
{text}

Transformed text:"""

        # Use the GroqClient's generate_text method
        transformed_text = await groq_client.generate_text(style_prompt, max_tokens=2000)
        
        return {
            "success": True,
            "original_text": text,
            "transformed_text": transformed_text,
            "target_style": target_style,
            "model": "openai/gpt-oss-120b"
        }
        
    except Exception as e:
        logger.error(f"Style transformation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Style transformation failed: {str(e)}")

@router.post("/find-similar-literature")
async def find_similar_literature(request: dict):
    """Find similar passages from Project Gutenberg based on narrative attributes"""
    
    attributes = request.get("attributes", {})
    max_results = request.get("max_results", 3)
    
    if not attributes:
        raise HTTPException(status_code=400, detail="No attributes provided")
    
    try:
        from core.gutenberg_integration import search_similar_literature
        
        passages = search_similar_literature(attributes, max_results)
        
        return {
            "success": True,
            "passages": passages,
            "total_found": len(passages),
            "search_attributes": list(attributes.keys())[:5]  # Show first 5 attributes used
        }
        
    except Exception as e:
        logger.error(f"Similar literature search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Literature search failed: {str(e)}")

@router.post("/extract")
async def extract_signature(request: ExtractSignatureRequest):
    """Extract narrative signature from text"""
    initialize_manager()
    
    try:
        if request.signature_name:
            # Extract and save
            signature = await manager.extract_and_save(
                request.text, 
                request.signature_name
            )
        else:
            # Just extract
            signature = await manager.extractor.extract_signature(
                request.text, 
                use_llm=request.use_llm
            )
        
        return {
            "success": True,
            "signature": {
                "namespace": signature.namespace,
                "style": signature.style,
                "persona": signature.persona,
                "metadata": signature.metadata
            },
            "saved_as": request.signature_name
        }
        
    except Exception as e:
        logger.error(f"Failed to extract signature: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.post("/apply")
async def apply_signature(request: ApplySignatureRequest):
    """Apply saved signature to new text"""
    initialize_manager()
    
    try:
        result = await manager.apply_signature(
            request.text,
            request.signature_name,
            request.strength
        )
        
        # Return the signature data that can be used for quantum steering
        response = {
            "success": True,
            "original_text": request.text,
            "signature_applied": request.signature_name,
            "strength": request.strength,
            "quantum_steering_data": {
                "target_attributes": {},
                "measurement_adjustments": {}
            }
        }
        
        # Convert signature to quantum steering format
        balanced_sig = result["balanced_signature"]
        
        # Combine all attributes into a flat dictionary for quantum steering
        all_attributes = {}
        all_attributes.update(balanced_sig.namespace)
        all_attributes.update(balanced_sig.style)
        all_attributes.update(balanced_sig.persona)
        
        # Map to POVM measurement format
        for attr_name, weight in all_attributes.items():
            if weight > 0.1:  # Only include significant attributes
                # Convert to POVM naming convention
                povm_name = f"{attr_name}_high"  # Could be more sophisticated mapping
                response["quantum_steering_data"]["target_attributes"][povm_name] = weight
        
        if request.auto_balance:
            response["balancing_suggestions"] = result["balancing_suggestions"]
            response["balanced_signature"] = {
                "namespace": balanced_sig.namespace,
                "style": balanced_sig.style,
                "persona": balanced_sig.persona
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to apply signature: {e}")
        raise HTTPException(status_code=500, detail=f"Application failed: {str(e)}")

@router.post("/create")
async def create_signature(request: CreateSignatureRequest):
    """Create and save a custom signature"""
    initialize_manager()
    
    try:
        signature = NarrativeSignature(
            namespace=request.namespace,
            style=request.style,
            persona=request.persona,
            metadata={"description": request.description, "custom": True}
        )
        
        manager.saved_signatures[request.name] = signature
        
        return {
            "success": True,
            "signature_name": request.name,
            "signature": {
                "namespace": signature.namespace,
                "style": signature.style,
                "persona": signature.persona,
                "metadata": signature.metadata
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create signature: {e}")
        raise HTTPException(status_code=500, detail=f"Creation failed: {str(e)}")

@router.post("/operations")
async def signature_operation(request: SignatureOperationRequest):
    """Perform mathematical operations on signatures"""
    initialize_manager()
    
    try:
        base_sig = manager.get_signature(request.base_signature)
        modifier_sig = manager.get_signature(request.modifier_signature)
        
        if not base_sig:
            raise HTTPException(status_code=404, detail=f"Base signature '{request.base_signature}' not found")
        if not modifier_sig:
            raise HTTPException(status_code=404, detail=f"Modifier signature '{request.modifier_signature}' not found")
        
        if request.operation == "add":
            result_sig = manager.operator.add_signatures(base_sig, modifier_sig, request.strength)
        elif request.operation == "subtract":
            result_sig = manager.operator.subtract_signatures(base_sig, modifier_sig, request.strength)
        elif request.operation == "blend":
            result_sig = manager.operator.blend_signatures(base_sig, modifier_sig, request.blend_ratio)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
        
        return {
            "success": True,
            "operation": request.operation,
            "result_signature": {
                "namespace": result_sig.namespace,
                "style": result_sig.style,
                "persona": result_sig.persona,
                "metadata": result_sig.metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signature operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")

# === Management Endpoints ===

@router.get("/signatures")
async def list_signatures():
    """List all saved signatures"""
    initialize_manager()
    
    signatures_info = []
    for name, signature in manager.saved_signatures.items():
        signatures_info.append({
            "name": name,
            "namespace_attributes": len(signature.namespace),
            "style_attributes": len(signature.style),
            "persona_attributes": len(signature.persona),
            "metadata": signature.metadata
        })
    
    return {
        "signatures": signatures_info,
        "count": len(signatures_info)
    }

@router.get("/signatures/{signature_name}")
async def get_signature(signature_name: str):
    """Get detailed information about a specific signature"""
    initialize_manager()
    
    signature = manager.get_signature(signature_name)
    if not signature:
        raise HTTPException(status_code=404, detail=f"Signature '{signature_name}' not found")
    
    return {
        "name": signature_name,
        "signature": {
            "namespace": signature.namespace,
            "style": signature.style,
            "persona": signature.persona,
            "metadata": signature.metadata
        }
    }

@router.delete("/signatures/{signature_name}")
async def delete_signature(signature_name: str):
    """Delete a saved signature"""
    initialize_manager()
    
    if signature_name not in manager.saved_signatures:
        raise HTTPException(status_code=404, detail=f"Signature '{signature_name}' not found")
    
    del manager.saved_signatures[signature_name]
    
    return {
        "success": True,
        "deleted": signature_name
    }

@router.post("/balance")
async def get_balancing_suggestions(signature_data: Dict):
    """Get AI-assisted balancing suggestions for a signature"""
    initialize_manager()
    
    try:
        # Convert input to NarrativeSignature
        signature = NarrativeSignature(
            namespace=signature_data.get("namespace", {}),
            style=signature_data.get("style", {}),
            persona=signature_data.get("persona", {}),
            metadata=signature_data.get("metadata", {})
        )
        
        balancing_result = await manager.balancer.suggest_balancing(signature)
        
        return {
            "success": True,
            "original_signature": {
                "namespace": signature.namespace,
                "style": signature.style,
                "persona": signature.persona
            },
            "issues": balancing_result["issues"],
            "suggestions": balancing_result["suggestions"],
            "balanced_signature": {
                "namespace": balancing_result["balanced_signature"].namespace,
                "style": balancing_result["balanced_signature"].style,
                "persona": balancing_result["balanced_signature"].persona
            }
        }
        
    except Exception as e:
        logger.error(f"Balancing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Balancing failed: {str(e)}")

# === Taxonomy and Reference ===

@router.get("/taxonomy")
async def get_taxonomy():
    """Get the complete narrative attribute taxonomy"""
    taxonomy = NarrativeAttributeTaxonomy.get_all_attributes()
    
    return {
        "taxonomy": taxonomy,
        "categories": list(taxonomy.keys()),
        "total_attributes": sum(len(attrs) for attrs in taxonomy.values())
    }

@router.get("/taxonomy/{category}")
async def get_category_attributes(category: str):
    """Get attributes for a specific category"""
    taxonomy = NarrativeAttributeTaxonomy.get_all_attributes()
    
    if category not in taxonomy:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    return {
        "category": category,
        "attributes": taxonomy[category],
        "count": len(taxonomy[category])
    }

# === File Management ===

@router.post("/save")
async def save_signatures_to_file(filepath: str = "data/narrative_signatures.json"):
    """Save all signatures to file"""
    initialize_manager()
    
    try:
        manager.save_signatures_to_file(filepath)
        return {
            "success": True,
            "saved_to": filepath,
            "signature_count": len(manager.saved_signatures)
        }
    except Exception as e:
        logger.error(f"Failed to save signatures: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@router.post("/load")
async def load_signatures_from_file(filepath: str = "data/narrative_signatures.json"):
    """Load signatures from file"""
    initialize_manager()
    
    try:
        manager.load_signatures_from_file(filepath)
        return {
            "success": True,
            "loaded_from": filepath,
            "signature_count": len(manager.saved_signatures)
        }
    except Exception as e:
        logger.error(f"Failed to load signatures: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

# === Integration with Quantum System ===

@router.post("/quantum-integration/{signature_name}")
async def prepare_quantum_steering(signature_name: str, text: str, strength: float = 1.0):
    """Prepare signature data for quantum steering operations"""
    initialize_manager()
    
    signature = manager.get_signature(signature_name)
    if not signature:
        raise HTTPException(status_code=404, detail=f"Signature '{signature_name}' not found")
    
    try:
        # Convert signature to quantum steering format
        target_attributes = {}
        
        # Map narrative attributes to POVM measurements
        # This would need to be customized based on available POVM packs
        for category, attributes in [
            ("namespace", signature.namespace),
            ("style", signature.style),
            ("persona", signature.persona)
        ]:
            for attr_name, weight in attributes.items():
                if weight > 0.1:
                    # Convert to POVM format - this mapping would be more sophisticated
                    # in a real implementation, connecting to actual POVM axes
                    povm_attr = f"{attr_name}_{category}"
                    target_attributes[povm_attr] = min(0.99, weight * strength)
        
        return {
            "success": True,
            "signature_name": signature_name,
            "original_text": text,
            "quantum_steering_ready": True,
            "target_attributes": target_attributes,
            "recommended_pack": "advanced_narrative_pack",  # Could be dynamic
            "strength": strength
        }
        
    except Exception as e:
        logger.error(f"Quantum integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integration failed: {str(e)}")

# Manager will be initialized on first use via initialize_manager() calls