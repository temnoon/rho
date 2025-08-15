"""
Matrix Library API Routes

Provides REST endpoints for advanced density matrix management, analysis,
and creative synthesis capabilities.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import numpy as np
from datetime import datetime

from core.matrix_library import matrix_library, MatrixMetadata, QualityAssessment
from routes.matrix_routes import STATE

router = APIRouter(prefix="/matrix-library", tags=["matrix-library"])
logger = logging.getLogger(__name__)


class MatrixRegistrationRequest(BaseModel):
    """Request to register a matrix in the library."""
    rho_id: str
    label: Optional[str] = None
    source_type: str = "unknown"
    content_preview: str = ""
    content_length: int = 0
    tags: List[str] = Field(default_factory=list)
    
class SynthesisRequest(BaseModel):
    """Request to synthesize multiple matrices."""
    matrix_ids: List[str]
    method: str = "convex_combination"
    weights: Optional[List[float]] = None
    label: Optional[str] = None
    
class QualityAssessmentRequest(BaseModel):
    """Request for quality assessment."""
    rho_id: str
    content_metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/register")
async def register_matrix(request: MatrixRegistrationRequest):
    """Register a matrix in the library with comprehensive metadata."""
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail=f"Matrix {request.rho_id} not found")
        
        matrix = STATE[request.rho_id]['rho']
        
        source_info = {
            'label': request.label or f"Matrix_{request.rho_id[:8]}",
            'type': request.source_type,
            'preview': request.content_preview,
            'content_length': request.content_length,
            'tags': request.tags,
            'reading_history': STATE[request.rho_id].get('ops', [])
        }
        
        metadata = matrix_library.register_matrix(request.rho_id, matrix, source_info)
        
        return {
            "success": True,
            "rho_id": request.rho_id,
            "metadata": {
                "label": metadata.label,
                "source_type": metadata.source_type,
                "creation_date": metadata.creation_date.isoformat(),
                "content_length": metadata.content_length,
                "tags": list(metadata.tags),
                "quality_metrics": metadata.quality_metrics
            },
            "message": "Matrix registered successfully in library"
        }
        
    except Exception as e:
        logger.error(f"Failed to register matrix: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/analyze")
async def analyze_matrix_collection():
    """Perform comprehensive analysis of the entire matrix collection."""
    try:
        # Get all matrices from STATE that are registered in library
        registered_matrices = {}
        for rho_id in matrix_library.metadata_cache.keys():
            if rho_id in STATE:
                registered_matrices[rho_id] = STATE[rho_id]['rho']
        
        if len(registered_matrices) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 registered matrices for collection analysis"
            )
        
        analysis = matrix_library.analyze_matrix_collection(registered_matrices)
        
        # Add metadata to the analysis
        for i, matrix_id in enumerate(analysis['matrix_ids']):
            metadata = matrix_library.metadata_cache.get(matrix_id)
            if metadata:
                analysis[f'metadata_{i}'] = {
                    'label': metadata.label,
                    'source_type': metadata.source_type,
                    'tags': list(metadata.tags),
                    'content_preview': metadata.content_preview[:100]
                }
        
        return {
            "success": True,
            "analysis": analysis,
            "collection_size": len(registered_matrices),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collection analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/assess-quality")
async def assess_matrix_quality(request: QualityAssessmentRequest):
    """Perform comprehensive quality assessment for a matrix."""
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail=f"Matrix {request.rho_id} not found")
        
        matrix = STATE[request.rho_id]['rho']
        
        # Use provided metadata or extract from STATE
        content_metadata = request.content_metadata
        if not content_metadata:
            state_data = STATE[request.rho_id]
            content_metadata = {
                'content_length': len(str(state_data.get('narratives', []))),
                'reading_history': state_data.get('ops', [])
            }
        
        assessment = matrix_library.assess_matrix_quality(
            request.rho_id, matrix, content_metadata
        )
        
        return {
            "success": True,
            "assessment": {
                "rho_id": assessment.rho_id,
                "overall_score": assessment.overall_score,
                "complexity_score": assessment.complexity_score,
                "coherence_score": assessment.coherence_score,
                "novelty_score": assessment.novelty_score,
                "depth_score": assessment.depth_score,
                "emotional_resonance": assessment.emotional_resonance,
                "technical_merit": assessment.technical_merit,
                "assessment_rationale": assessment.assessment_rationale
            },
            "recommendations": _generate_improvement_recommendations(assessment)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.get("/best-work")
async def find_best_work(
    criteria: str = Query("overall", description="Criteria for ranking: overall, complexity, novelty, depth, resonance"),
    top_k: int = Query(5, description="Number of top works to return")
):
    """Find your best work based on quality assessments."""
    try:
        # Get matrices for registered items
        registered_matrices = {}
        for rho_id in matrix_library.metadata_cache.keys():
            if rho_id in STATE:
                registered_matrices[rho_id] = STATE[rho_id]['rho']
        
        if not registered_matrices:
            raise HTTPException(status_code=400, detail="No registered matrices found")
        
        best_work = matrix_library.find_best_work(registered_matrices, criteria, top_k)
        
        # Enrich with metadata
        enriched_results = []
        for rho_id, score in best_work:
            metadata = matrix_library.metadata_cache.get(rho_id)
            enriched_results.append({
                "rho_id": rho_id,
                "score": score,
                "label": metadata.label if metadata else f"Matrix_{rho_id[:8]}",
                "source_type": metadata.source_type if metadata else "unknown",
                "content_preview": metadata.content_preview if metadata else "",
                "tags": list(metadata.tags) if metadata else [],
                "creation_date": metadata.creation_date.isoformat() if metadata else None
            })
        
        return {
            "success": True,
            "criteria": criteria,
            "best_work": enriched_results,
            "total_assessed": len(matrix_library.quality_assessments)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Best work search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/synthesize")
async def synthesize_matrices(request: SynthesisRequest):
    """Synthesize multiple matrices to create new work."""
    try:
        # Validate all matrices exist
        missing_matrices = [mid for mid in request.matrix_ids if mid not in STATE]
        if missing_matrices:
            raise HTTPException(
                status_code=404, 
                detail=f"Matrices not found: {missing_matrices}"
            )
        
        # Get matrices
        matrices = {mid: STATE[mid]['rho'] for mid in request.matrix_ids}
        
        # Perform synthesis
        synthesized_matrix, synthesis_metadata = matrix_library.synthesize_matrices(
            request.matrix_ids, matrices, request.method, request.weights
        )
        
        # Create new matrix entry in STATE
        from routes.matrix_routes import create_new_rho_id
        new_rho_id = create_new_rho_id()
        
        # Store synthesized matrix
        STATE[new_rho_id] = {
            'rho': synthesized_matrix,
            'label': request.label or synthesis_metadata['expected_properties']['recommended_label'],
            'created_at': datetime.now().isoformat(),
            'ops': [],
            'narratives': [],
            'synthesis_info': synthesis_metadata
        }
        
        # Register in library
        source_info = {
            'label': request.label or synthesis_metadata['expected_properties']['recommended_label'],
            'type': 'synthesis',
            'preview': f"Synthesis of {len(request.matrix_ids)} matrices using {request.method}",
            'content_length': 0,
            'tags': synthesis_metadata['expected_properties'].get('predicted_tags', []),
            'parent_matrices': request.matrix_ids,
            'synthesis_method': request.method
        }
        
        matrix_library.register_matrix(new_rho_id, synthesized_matrix, source_info)
        
        return {
            "success": True,
            "new_rho_id": new_rho_id,
            "synthesis_metadata": synthesis_metadata,
            "parent_matrices": request.matrix_ids,
            "method": request.method,
            "message": "Matrix synthesis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matrix synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.get("/recommendations")
async def get_synthesis_recommendations(
    target_theme: Optional[str] = Query(None, description="Optional target theme for recommendations"),
    max_recommendations: int = Query(10, description="Maximum number of recommendations")
):
    """Get recommendations for interesting matrix combinations."""
    try:
        # Get matrices for registered items
        registered_matrices = {}
        for rho_id in matrix_library.metadata_cache.keys():
            if rho_id in STATE:
                registered_matrices[rho_id] = STATE[rho_id]['rho']
        
        if len(registered_matrices) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 registered matrices for recommendations"
            )
        
        recommendations = matrix_library.generate_synthesis_recommendations(
            registered_matrices, target_theme, max_recommendations
        )
        
        # Enrich recommendations with metadata
        enriched_recommendations = []
        for rec in recommendations:
            enriched_rec = dict(rec)
            enriched_rec['matrices_info'] = []
            
            for matrix_id in rec['matrices']:
                metadata = matrix_library.metadata_cache.get(matrix_id)
                enriched_rec['matrices_info'].append({
                    'rho_id': matrix_id,
                    'label': metadata.label if metadata else f"Matrix_{matrix_id[:8]}",
                    'source_type': metadata.source_type if metadata else "unknown",
                    'tags': list(metadata.tags) if metadata else []
                })
            
            enriched_recommendations.append(enriched_rec)
        
        return {
            "success": True,
            "recommendations": enriched_recommendations,
            "total_matrices": len(registered_matrices),
            "target_theme": target_theme
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@router.get("/similarity/{rho_id_a}/{rho_id_b}")
async def calculate_matrix_similarity(
    rho_id_a: str, 
    rho_id_b: str,
    method: str = Query("bures", description="Similarity method: bures, trace, fidelity, eigenspace")
):
    """Calculate similarity between two specific matrices."""
    try:
        if rho_id_a not in STATE:
            raise HTTPException(status_code=404, detail=f"Matrix {rho_id_a} not found")
        if rho_id_b not in STATE:
            raise HTTPException(status_code=404, detail=f"Matrix {rho_id_b} not found")
        
        matrix_a = STATE[rho_id_a]['rho']
        matrix_b = STATE[rho_id_b]['rho']
        
        similarity = matrix_library.calculate_matrix_similarity(matrix_a, matrix_b, method)
        
        # Get metadata for context
        metadata_a = matrix_library.metadata_cache.get(rho_id_a)
        metadata_b = matrix_library.metadata_cache.get(rho_id_b)
        
        return {
            "success": True,
            "rho_id_a": rho_id_a,
            "rho_id_b": rho_id_b,
            "similarity": similarity,
            "distance": 1.0 - similarity,
            "method": method,
            "matrix_a_info": {
                "label": metadata_a.label if metadata_a else f"Matrix_{rho_id_a[:8]}",
                "source_type": metadata_a.source_type if metadata_a else "unknown"
            },
            "matrix_b_info": {
                "label": metadata_b.label if metadata_b else f"Matrix_{rho_id_b[:8]}",
                "source_type": metadata_b.source_type if metadata_b else "unknown"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")


@router.get("/library/status")
async def get_library_status():
    """Get status and statistics of the matrix library."""
    try:
        # Count matrices by source type
        source_counts = {}
        quality_stats = {
            'assessed': 0,
            'avg_overall_score': 0,
            'top_score': 0,
            'score_distribution': {}
        }
        
        for metadata in matrix_library.metadata_cache.values():
            source_type = metadata.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        # Quality assessment statistics
        if matrix_library.quality_assessments:
            scores = [a.overall_score for a in matrix_library.quality_assessments.values()]
            quality_stats['assessed'] = len(scores)
            quality_stats['avg_overall_score'] = np.mean(scores)
            quality_stats['top_score'] = np.max(scores)
            
            # Score distribution
            for score in scores:
                bucket_low = int(score * 10) * 10
                bucket_high = bucket_low + 10
                bucket = f"{bucket_low}-{bucket_high}%"
                quality_stats['score_distribution'][bucket] = quality_stats['score_distribution'].get(bucket, 0) + 1
        
        # Synthesis statistics
        synthesis_count = sum(1 for m in matrix_library.metadata_cache.values() 
                            if m.source_type == 'synthesis')
        
        return {
            "success": True,
            "library_stats": {
                "total_matrices": len(matrix_library.metadata_cache),
                "source_type_distribution": source_counts,
                "quality_assessments": quality_stats,
                "synthesis_count": synthesis_count,
                "similarity_cache_size": len(matrix_library.distance_cache)
            },
            "capabilities": {
                "similarity_methods": ["bures", "trace", "fidelity", "eigenspace"],
                "synthesis_methods": ["convex_combination", "geometric_mean", "coherent_superposition", "interference_pattern"],
                "quality_criteria": ["overall", "complexity", "novelty", "depth", "resonance", "technical"]
            }
        }
        
    except Exception as e:
        logger.error(f"Library status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/metadata/{rho_id}")
async def get_matrix_metadata(rho_id: str):
    """Get comprehensive metadata for a specific matrix."""
    try:
        metadata = matrix_library.metadata_cache.get(rho_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Matrix {rho_id} not registered in library")
        
        # Get quality assessment if available
        assessment = matrix_library.quality_assessments.get(rho_id)
        
        result = {
            "success": True,
            "rho_id": rho_id,
            "metadata": {
                "label": metadata.label,
                "creation_date": metadata.creation_date.isoformat(),
                "source_type": metadata.source_type,
                "content_preview": metadata.content_preview,
                "content_length": metadata.content_length,
                "tags": list(metadata.tags),
                "parent_matrices": metadata.parent_matrices,
                "synthesis_method": metadata.synthesis_method,
                "quality_metrics": metadata.quality_metrics
            }
        }
        
        if assessment:
            result["quality_assessment"] = {
                "overall_score": assessment.overall_score,
                "complexity_score": assessment.complexity_score,
                "coherence_score": assessment.coherence_score,
                "novelty_score": assessment.novelty_score,
                "depth_score": assessment.depth_score,
                "emotional_resonance": assessment.emotional_resonance,
                "technical_merit": assessment.technical_merit,
                "assessment_rationale": assessment.assessment_rationale
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metadata retrieval failed: {str(e)}")


def _generate_improvement_recommendations(assessment: QualityAssessment) -> List[str]:
    """Generate recommendations for improving work based on assessment."""
    recommendations = []
    
    if assessment.complexity_score < 0.4:
        recommendations.append("Consider adding more conceptual layers or exploring multiple perspectives")
    
    if assessment.coherence_score < 0.4:
        recommendations.append("Work on creating stronger thematic connections and internal consistency")
    
    if assessment.novelty_score < 0.4:
        recommendations.append("Explore more unusual combinations or unique angles on familiar themes")
    
    if assessment.depth_score < 0.4:
        recommendations.append("Invest more time in developing ideas and exploring implications")
    
    if assessment.emotional_resonance < 0.4:
        recommendations.append("Consider incorporating more varied emotional elements or personal stakes")
    
    if assessment.technical_merit < 0.4:
        recommendations.append("Focus on improving structural foundation and technical execution")
    
    if not recommendations:
        recommendations.append("Excellent work! Consider synthesizing with other pieces to explore new directions")
    
    return recommendations