"""
API routes for Un-Earthify functionality - systematic transformation
of Earth-based narratives into alien contexts while preserving ρ-space essence.
"""

import logging
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.un_earthify import un_earthify_text, get_alien_dictionary, un_earthify_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/un-earthify", tags=["un-earthify"])

# Request/Response Models
class UnEarthifyRequest(BaseModel):
    text: str
    preserve_rho_essence: bool = True
    context: str = ""

class AlienDictionaryResponse(BaseModel):
    alien_term: str
    earth_term: str
    category: str
    usage_count: int
    context_examples: List[str]

@router.post("/transform")
async def transform_text(request: UnEarthifyRequest):
    """Transform Earth-based text into alien equivalent"""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = un_earthify_text(request.text)
        
        return {
            "success": True,
            "original_text": result["original_text"],
            "transformed_text": result["un_earthified_text"],
            "transformations": result["transformations"],
            "earth_terms_found": result["earth_terms_found"],
            "rho_preservation_verified": result["rho_preservation"],
            "transformation_summary": {
                "total_replacements": len(result["transformations"]),
                "categories_affected": list(set(t["category"] for t in result["transformations"])),
                "new_alien_terms": len([t for t in result["transformations"] if t.get("is_new", False)])
            }
        }
        
    except Exception as e:
        logger.error(f"Un-earthify transformation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

@router.get("/dictionary")
async def get_dictionary():
    """Get the complete alien-to-earth dictionary"""
    
    try:
        dictionary = get_alien_dictionary()
        
        alien_entries = []
        for earth_term, mapping in dictionary.items():
            alien_entries.append({
                "alien_term": mapping.alien_term,
                "earth_term": mapping.earth_term,
                "category": mapping.category,
                "usage_count": mapping.usage_count,
                "context_examples": mapping.context_examples,
                "phonetic_pattern": mapping.phonetic_pattern
            })
        
        # Sort by usage count (most used first)
        alien_entries.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "dictionary": alien_entries,
            "total_mappings": len(alien_entries),
            "categories": list(set(entry["category"] for entry in alien_entries)),
            "most_used_terms": alien_entries[:10]  # Top 10 most used
        }
        
    except Exception as e:
        logger.error(f"Failed to get alien dictionary: {e}")
        raise HTTPException(status_code=500, detail=f"Dictionary access failed: {str(e)}")

@router.get("/dictionary/{category}")
async def get_dictionary_by_category(category: str):
    """Get alien dictionary entries for a specific category"""
    
    try:
        dictionary = get_alien_dictionary()
        
        category_entries = []
        for earth_term, mapping in dictionary.items():
            if mapping.category == category:
                category_entries.append({
                    "alien_term": mapping.alien_term,
                    "earth_term": mapping.earth_term,
                    "usage_count": mapping.usage_count,
                    "context_examples": mapping.context_examples
                })
        
        if not category_entries:
            raise HTTPException(status_code=404, detail=f"No entries found for category: {category}")
        
        category_entries.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "category": category,
            "entries": category_entries,
            "count": len(category_entries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get category dictionary: {e}")
        raise HTTPException(status_code=500, detail=f"Category lookup failed: {str(e)}")

@router.post("/reverse")
async def reverse_transform(alien_text: str):
    """Convert alien text back to Earth terms (for debugging/verification)"""
    
    try:
        earth_text = un_earthify_engine.reverse_mapping(alien_text)
        
        return {
            "alien_text": alien_text,
            "earth_text": earth_text,
            "transformations_reversed": alien_text != earth_text
        }
        
    except Exception as e:
        logger.error(f"Reverse transformation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reverse transformation failed: {str(e)}")

@router.get("/categories")
async def get_categories():
    """Get all available Earth term categories"""
    
    return {
        "categories": [
            {
                "name": "planets",
                "description": "Celestial bodies and astronomical references",
                "examples": ["earth", "mars", "venus", "jupiter"]
            },
            {
                "name": "animals", 
                "description": "Earth fauna and biological life forms",
                "examples": ["dog", "cat", "horse", "elephant"]
            },
            {
                "name": "foods",
                "description": "Earth-specific consumables and cuisine", 
                "examples": ["coffee", "tea", "bread", "wine"]
            },
            {
                "name": "places",
                "description": "Earth locations, countries, and regions",
                "examples": ["america", "europe", "asia", "california"]
            },
            {
                "name": "technologies",
                "description": "Earth-specific technologies and inventions",
                "examples": ["internet", "smartphone", "airplane", "television"]
            },
            {
                "name": "materials",
                "description": "Earth-specific materials and substances",
                "examples": ["steel", "gold", "plastic", "wood"]
            },
            {
                "name": "concepts",
                "description": "Earth-specific ideologies and belief systems",
                "examples": ["democracy", "capitalism", "christianity", "buddhism"]
            },
            {
                "name": "organizations",
                "description": "Earth-specific institutions and groups",
                "examples": ["nasa", "un", "fbi", "nato"]
            }
        ]
    }

@router.post("/identify-terms")
async def identify_earth_terms(text: str):
    """Identify Earth-specific terms in text without transformation"""
    
    try:
        earth_terms = un_earthify_engine.identify_earth_terms(text)
        
        categorized_terms = {}
        for term in earth_terms:
            category = un_earthify_engine.categorizer.categorize_term(term)
            if category not in categorized_terms:
                categorized_terms[category] = []
            categorized_terms[category].append(term)
        
        return {
            "text": text,
            "earth_terms_found": earth_terms,
            "total_terms": len(earth_terms),
            "categorized_terms": categorized_terms,
            "categories_found": list(categorized_terms.keys())
        }
        
    except Exception as e:
        logger.error(f"Earth term identification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Term identification failed: {str(e)}")

@router.get("/stats")
async def get_transformation_stats():
    """Get statistics about the un-earthify system"""
    
    try:
        dictionary = get_alien_dictionary()
        
        if not dictionary:
            return {
                "total_mappings": 0,
                "categories": {},
                "usage_stats": {},
                "most_active_categories": []
            }
        
        # Category statistics
        category_counts = {}
        category_usage = {}
        
        for mapping in dictionary.values():
            category = mapping.category
            category_counts[category] = category_counts.get(category, 0) + 1
            category_usage[category] = category_usage.get(category, 0) + mapping.usage_count
        
        # Sort categories by usage
        most_active = sorted(category_usage.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_mappings": len(dictionary),
            "categories": category_counts,
            "usage_stats": {
                "total_usages": sum(m.usage_count for m in dictionary.values()),
                "average_usage_per_term": sum(m.usage_count for m in dictionary.values()) / len(dictionary) if dictionary else 0,
                "category_usage": category_usage
            },
            "most_active_categories": most_active[:5],
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Failed to get transformation stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.post("/test-generation")
async def test_word_generation(earth_term: str, category: str = None):
    """Test alien word generation for a specific Earth term"""
    
    try:
        if category is None:
            category = un_earthify_engine.categorizer.categorize_term(earth_term)
        
        # Generate multiple options
        alien_options = []
        for i in range(5):
            alien_word = un_earthify_engine.generator.generate_alien_word(earth_term, category)
            alien_options.append(alien_word)
        
        return {
            "earth_term": earth_term,
            "detected_category": category,
            "alien_options": alien_options,
            "phonetic_patterns": un_earthify_engine.generator.PHONEME_PATTERNS.get(category, []),
            "suffix_patterns": un_earthify_engine.generator.SUFFIXES.get(category, [])
        }
        
    except Exception as e:
        logger.error(f"Word generation test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation test failed: {str(e)}")

@router.post("/purge-dictionary")
async def purge_alien_dictionary():
    """Purge the entire alien dictionary to start fresh with new word generation rules"""
    
    try:
        # Clear all mappings
        un_earthify_engine.mappings.clear()
        un_earthify_engine.alien_dictionary.clear()
        
        # Save the empty state
        un_earthify_engine._save_mappings()
        
        logger.info("Purged alien dictionary - all mappings cleared")
        
        return {
            "success": True,
            "message": "Alien dictionary purged successfully",
            "mappings_cleared": True,
            "total_mappings": 0,
            "note": "New transformations will use updated word generation rules"
        }
        
    except Exception as e:
        logger.error(f"Dictionary purge failed: {e}")
        raise HTTPException(status_code=500, detail=f"Purge failed: {str(e)}")

@router.delete("/dictionary/{earth_term}")
async def delete_specific_mapping(earth_term: str):
    """Delete a specific Earth term mapping from the alien dictionary"""
    
    try:
        if earth_term not in un_earthify_engine.mappings:
            raise HTTPException(status_code=404, detail=f"No mapping found for: {earth_term}")
        
        # Get the alien term before deletion
        mapping = un_earthify_engine.mappings[earth_term]
        alien_term = mapping.alien_term
        
        # Remove from both dictionaries
        del un_earthify_engine.mappings[earth_term]
        if alien_term.lower() in un_earthify_engine.alien_dictionary:
            del un_earthify_engine.alien_dictionary[alien_term.lower()]
        
        # Save updated mappings
        un_earthify_engine._save_mappings()
        
        logger.info(f"Deleted mapping: {earth_term} -> {alien_term}")
        
        return {
            "success": True,
            "deleted_mapping": {
                "earth_term": earth_term,
                "alien_term": alien_term,
                "category": mapping.category
            },
            "remaining_mappings": len(un_earthify_engine.mappings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mapping deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")