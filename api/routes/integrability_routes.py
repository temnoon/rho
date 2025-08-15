"""
Integrability Testing API Routes

Provides REST endpoints for testing integrability in quantum narrative operations.
These routes implement the mathematical framework for verifying that different
text segmentations yield equivalent quantum states.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import logging

from core.integrability_testing import (
    IntegrabilityTester, 
    test_text_integrability,
    quick_integrability_check
)

router = APIRouter(prefix="/integrability", tags=["integrability-testing"])
logger = logging.getLogger(__name__)

# Global integrability tester instance
TESTER = IntegrabilityTester(tolerance=1e-6)


class IntegrabilityTestRequest(BaseModel):
    """Request for integrability testing."""
    segments_a: List[str]
    segments_b: List[str]
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    tolerance: Optional[float] = None
    
    @field_validator('segments_a', 'segments_b')
    @classmethod
    def validate_segments(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Segments list cannot be empty")
        return [seg.strip() for seg in v if seg.strip()]
    
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


class QuickIntegrabilityRequest(BaseModel):
    """Request for quick integrability check."""
    text: str
    alpha: float = 0.3
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")
        return v.strip()


class CompositionTestRequest(BaseModel):
    """Request for compositional consistency testing."""
    text_pieces: List[str]
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    
    @field_validator('text_pieces')
    @classmethod
    def validate_text_pieces(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Must provide at least 2 text pieces")
        return [piece.strip() for piece in v if piece.strip()]


class PathSensitivityRequest(BaseModel):
    """Request for path sensitivity analysis."""
    text: str
    num_segmentations: int = 5
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    
    @field_validator('num_segmentations')
    @classmethod
    def validate_num_segmentations(cls, v):
        if not 2 <= v <= 20:
            raise ValueError("Number of segmentations must be between 2 and 20")
        return v


@router.post("/test_segmentations")
async def test_segmentation_integrability(request: IntegrabilityTestRequest):
    """
    Test if two different segmentations of text yield equivalent quantum states.
    
    This is the core integrability test that verifies the path-independence
    of quantum narrative evolution.
    """
    try:
        # Use provided tolerance or global default
        if request.tolerance is not None:
            tester = IntegrabilityTester(tolerance=request.tolerance)
        else:
            tester = TESTER
        
        result = tester.test_segmentation_integrability(
            segments_a=request.segments_a,
            segments_b=request.segments_b,
            channel_type=request.channel_type,
            alpha=request.alpha
        )
        
        # Convert numpy arrays to lists for JSON serialization
        return {
            "test_type": "segmentation_integrability",
            "segments_a": result.segments_a,
            "segments_b": result.segments_b,
            "segments_a_count": len(result.segments_a),
            "segments_b_count": len(result.segments_b),
            "bures_distance": float(result.bures_distance),
            "trace_distance": float(result.trace_distance),
            "fidelity": float(result.fidelity),
            "passes_test": result.passes_test,
            "tolerance": float(result.tolerance),
            "channel_logs": result.channel_logs,
            "recommendations": result.recommendations,
            "quantum_metrics": {
                "final_state_a_trace": float(result.final_state_a.trace()),
                "final_state_b_trace": float(result.final_state_b.trace()),
                "final_state_a_purity": float((result.final_state_a @ result.final_state_a).trace()),
                "final_state_b_purity": float((result.final_state_b @ result.final_state_b).trace())
            }
        }
        
    except Exception as e:
        logger.error(f"Integrability test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integrability test failed: {str(e)}")


@router.post("/quick_check")
async def quick_integrability_check_endpoint(request: QuickIntegrabilityRequest):
    """
    Perform a quick integrability check by comparing different automatic segmentations.
    
    This endpoint automatically generates sentence-level and word-level segmentations
    and tests their integrability.
    """
    try:
        result = quick_integrability_check(
            text=request.text,
            alpha=request.alpha
        )
        
        return {
            "success": True,
            "test_type": "quick_integrability_check",
            **result
        }
        
    except Exception as e:
        logger.error(f"Quick integrability check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick check failed: {str(e)}")


@router.post("/test_composition")
async def test_compositional_consistency(request: CompositionTestRequest):
    """
    Test compositional consistency: whether reading pieces sequentially equals reading concatenated text.
    
    This tests the fundamental property that ρ(AB) = Φ_B(Φ_A(ρ_0)).
    """
    try:
        result = TESTER.test_compositional_consistency(
            text_pieces=request.text_pieces,
            channel_type=request.channel_type,
            alpha=request.alpha
        )
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Compositional consistency test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Composition test failed: {str(e)}")


@router.post("/analyze_path_sensitivity")
async def analyze_path_sensitivity(request: PathSensitivityRequest):
    """
    Analyze how sensitive quantum evolution is to different segmentation strategies.
    
    This generates multiple random segmentations and measures variance in final states.
    """
    try:
        result = TESTER.analyze_path_sensitivity(
            base_text=request.text,
            num_segmentations=request.num_segmentations,
            channel_type=request.channel_type,
            alpha=request.alpha
        )
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Path sensitivity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Path sensitivity analysis failed: {str(e)}")


@router.get("/test_history")
async def get_test_history(limit: int = 10):
    """
    Get recent integrability test history.
    
    Args:
        limit: Maximum number of tests to return
    """
    try:
        history = TESTER.test_history[-limit:] if TESTER.test_history else []
        
        # Convert to serializable format
        serialized_history = []
        for test in history:
            serialized_history.append({
                "segments_a_count": len(test.segments_a),
                "segments_b_count": len(test.segments_b),
                "bures_distance": float(test.bures_distance),
                "trace_distance": float(test.trace_distance),
                "fidelity": float(test.fidelity),
                "passes_test": test.passes_test,
                "tolerance": float(test.tolerance),
                "recommendations_count": len(test.recommendations),
                "channel_logs_count": len(test.channel_logs)
            })
        
        return {
            "test_history": serialized_history,
            "total_tests": len(TESTER.test_history),
            "returned_count": len(serialized_history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get test history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/batch_test")
async def batch_integrability_test(texts: List[str], alpha: float = 0.3):
    """
    Run integrability tests on multiple texts in batch.
    
    This is useful for analyzing integrability across a corpus of texts.
    """
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="Must provide at least one text")
    
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")
    
    try:
        results = []
        for i, text in enumerate(texts):
            try:
                result = quick_integrability_check(text, alpha=alpha)
                results.append({
                    "text_index": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "success": True,
                    **result
                })
            except Exception as e:
                results.append({
                    "text_index": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "success": False,
                    "error": str(e)
                })
        
        # Summary statistics
        successful_tests = [r for r in results if r["success"]]
        if successful_tests:
            avg_distance = sum(r["bures_distance"] for r in successful_tests) / len(successful_tests)
            pass_rate = sum(1 for r in successful_tests if r["passes_test"]) / len(successful_tests)
        else:
            avg_distance = None
            pass_rate = 0.0
        
        return {
            "batch_results": results,
            "summary": {
                "total_texts": len(texts),
                "successful_tests": len(successful_tests),
                "failed_tests": len(texts) - len(successful_tests),
                "pass_rate": pass_rate,
                "average_bures_distance": avg_distance
            }
        }
        
    except Exception as e:
        logger.error(f"Batch integrability test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch test failed: {str(e)}")


@router.get("/metrics")
async def get_integrability_metrics():
    """
    Get global integrability testing metrics and statistics.
    """
    try:
        history = TESTER.test_history
        
        if not history:
            return {
                "total_tests": 0,
                "pass_rate": 0.0,
                "average_bures_distance": None,
                "metrics": "No tests performed yet"
            }
        
        # Calculate metrics
        total_tests = len(history)
        passed_tests = sum(1 for test in history if test.passes_test)
        pass_rate = passed_tests / total_tests
        
        bures_distances = [test.bures_distance for test in history]
        avg_bures = sum(bures_distances) / len(bures_distances)
        min_bures = min(bures_distances)
        max_bures = max(bures_distances)
        
        fidelities = [test.fidelity for test in history]
        avg_fidelity = sum(fidelities) / len(fidelities)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": pass_rate,
            "bures_distance_stats": {
                "average": avg_bures,
                "minimum": min_bures,
                "maximum": max_bures
            },
            "average_fidelity": avg_fidelity,
            "tolerance": TESTER.tolerance,
            "interpretation": {
                "pass_rate_quality": "excellent" if pass_rate > 0.9 else "good" if pass_rate > 0.7 else "needs_attention",
                "distance_quality": "excellent" if avg_bures < 1e-4 else "good" if avg_bures < 1e-3 else "needs_attention"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get integrability metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/configure")
async def configure_integrability_tester(tolerance: float = 1e-6):
    """
    Configure the global integrability tester settings.
    
    Args:
        tolerance: New tolerance for integrability tests
    """
    if not 1e-12 <= tolerance <= 1e-2:
        raise HTTPException(status_code=400, detail="Tolerance must be between 1e-12 and 1e-2")
    
    try:
        global TESTER
        TESTER = IntegrabilityTester(tolerance=tolerance)
        
        return {
            "success": True,
            "new_tolerance": tolerance,
            "message": "Integrability tester reconfigured"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure integrability tester: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.delete("/clear_history")
async def clear_test_history():
    """
    Clear the integrability test history.
    """
    try:
        old_count = len(TESTER.test_history)
        TESTER.test_history.clear()
        
        return {
            "success": True,
            "cleared_tests": old_count,
            "message": "Test history cleared"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear test history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")