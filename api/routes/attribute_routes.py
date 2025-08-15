"""
Attribute management routes for the Rho Quantum Narrative System.

This module provides endpoints for attribute extraction, manipulation,
and narrative generation based on attribute adjustments.
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attributes", tags=["attributes"])

# Mock attribute mapping - in a real system this would be loaded from a database
ATTRIBUTE_MAPPING = {
    "formality": {
        "description": "Level of formal vs informal language",
        "category": "register",
        "basis_vectors": ["formal", "informal"]
    },
    "narrative_distance": {
        "description": "Narrator's emotional/temporal distance from events",
        "category": "narrative",
        "basis_vectors": ["close", "distant"]
    },
    "affect": {
        "description": "Emotional intensity and expression",
        "category": "emotion",
        "basis_vectors": ["low_affect", "high_affect"]
    },
    "elaboration": {
        "description": "Level of detail and elaboration",
        "category": "style",
        "basis_vectors": ["simple", "elaborate"]
    },
    "temporal_perspective": {
        "description": "Temporal orientation of the narrative",
        "category": "narrative",
        "basis_vectors": ["present", "past"]
    }
}


class ExtractRequest(BaseModel):
    text: str
    rho_id: Optional[str] = None


class AttributeAdjustmentRequest(BaseModel):
    text: str
    adjustments: Dict[str, float]
    rho_id: Optional[str] = None


@router.get("/list")
def list_attributes():
    """List all available attributes grouped by category"""
    categories = {}
    for attr_name, attr_config in ATTRIBUTE_MAPPING.items():
        category = attr_config.get("category", "other")
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "name": attr_name,
            "description": attr_config["description"],
            "basis_vectors": attr_config["basis_vectors"],
            "dimension_count": len(attr_config["basis_vectors"])
        })
    
    return {
        "categories": categories,
        "total_attributes": len(ATTRIBUTE_MAPPING),
        "category_counts": {cat: len(attrs) for cat, attrs in categories.items()}
    }


@router.post("/extract")
def extract_attributes(request: ExtractRequest):
    """Extract attributes from text (placeholder implementation)"""
    # This is a placeholder - in a real system this would analyze the text
    # and return actual attribute measurements
    mock_attributes = {}
    for attr_name in ATTRIBUTE_MAPPING.keys():
        mock_attributes[attr_name] = 0.5  # Neutral value
    
    return {
        "text": request.text,
        "attributes": mock_attributes,
        "rho_id": request.rho_id,
        "method": "mock_extraction"
    }


@router.get("/collections")
def get_collections():
    """Get attribute collections"""
    # Convert attribute names to full attribute objects
    def get_attribute_by_name(name):
        if name in ATTRIBUTE_MAPPING:
            attr = ATTRIBUTE_MAPPING[name].copy()
            attr["name"] = name
            return attr
        return {"name": name, "description": f"Description for {name}", "basis_vectors": ["low", "high"]}
    
    return {
        "collections": [
            {
                "id": "narrative_basics",
                "name": "Narrative Basics",
                "description": "Essential narrative attributes",
                "attributes": [
                    get_attribute_by_name("narrative_distance"), 
                    get_attribute_by_name("temporal_perspective")
                ]
            },
            {
                "id": "style_features", 
                "name": "Style Features",
                "description": "Stylistic and register features",
                "attributes": [
                    get_attribute_by_name("formality"),
                    get_attribute_by_name("elaboration"), 
                    get_attribute_by_name("affect")
                ]
            }
        ]
    }


@router.get("/favorites")
def get_favorites():
    """Get favorite attributes"""
    # Convert attribute names to full attribute objects
    def get_attribute_by_name(name):
        if name in ATTRIBUTE_MAPPING:
            attr = ATTRIBUTE_MAPPING[name].copy()
            attr["name"] = name
            return attr
        return {"name": name, "description": f"Description for {name}", "basis_vectors": ["low", "high"]}
    
    favorite_names = ["formality", "narrative_distance", "affect"]
    return {
        "favorites": [get_attribute_by_name(name) for name in favorite_names]
    }


@router.get("/suggestions/{category}")
def get_suggestions(category: str):
    """Get attribute suggestions for a category"""
    suggestions_by_category = {
        "namespace": {
            "basic_stance": [
                {"name": "politeness", "description": "Level of politeness in language", "category": "register"},
                {"name": "certainty", "description": "Degree of certainty or hedging", "category": "stance"}
            ],
            "advanced_stance": [
                {"name": "objectivity", "description": "Objective vs subjective perspective", "category": "stance"}
            ]
        },
        "persona": {
            "authority": [
                {"name": "authoritativeness", "description": "Level of authoritative tone", "category": "register"}
            ],
            "emotion": [
                {"name": "empathy", "description": "Degree of empathetic expression", "category": "emotion"},
                {"name": "warmth", "description": "Emotional warmth in expression", "category": "emotion"}
            ]
        },
        "style": {
            "register": [
                {"name": "technicality", "description": "Use of technical terminology", "category": "style"},
                {"name": "intimacy", "description": "Level of personal/intimate language", "category": "register"}
            ],
            "complexity": [
                {"name": "elaboration", "description": "Level of detail and elaboration", "category": "style"}
            ]
        }
    }
    
    return {
        "suggestions": suggestions_by_category.get(category, {})
    }


@router.post("/search")
def search_attributes(request: dict):
    """Search attributes by query"""
    query = request.get("query", "").lower()
    category = request.get("category")
    limit = request.get("limit", 20)
    
    # Mock search through attributes
    results = []
    for attr_name, attr_config in ATTRIBUTE_MAPPING.items():
        if query in attr_name.lower() or query in attr_config["description"].lower():
            if not category or attr_config.get("category") == category:
                attr = attr_config.copy()
                attr["name"] = attr_name
                results.append(attr)
    
    return {"attributes": results[:limit]}


@router.post("/favorites/add")
def add_favorite(request: dict):
    """Add attribute to favorites"""
    # In a real system, this would store to database
    return {"success": True, "attribute_name": request.get("attribute_name")}


@router.post("/favorites/remove")
def remove_favorite(request: dict):
    """Remove attribute from favorites"""
    # In a real system, this would remove from database
    return {"success": True, "attribute_name": request.get("attribute_name")}


@router.post("/regenerate_narrative")
def regenerate_narrative(request: AttributeAdjustmentRequest):
    """Regenerate narrative with attribute adjustments (placeholder)"""
    # This is a placeholder - would integrate with LLM for actual regeneration
    modified_text = f"[MODIFIED: {request.text[:100]}...]"
    
    return {
        "original_text": request.text,
        "modified_text": modified_text,
        "adjustments": request.adjustments,
        "rho_id": request.rho_id,
        "method": "mock_regeneration"
    }