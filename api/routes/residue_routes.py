"""
Residue/Holonomy Detection Routes (APLG Claim Set C)

Implements detection of interpretive residue through paraphrase loops for identifying
irony, narrative twists, and "sticky" moments in text. This corresponds to the 
mathematical framework where non-zero residue Δ = Φ_γ - I indicates path-dependent
interpretation effects.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import uuid
from datetime import datetime

from routes.matrix_routes import STATE, rho_init, rho_read_with_channel
from models.requests import ReadReq

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/residue", tags=["residue-holonomy"])

# Request Models
class ResidueLoopRequest(BaseModel):
    """Request for residue loop analysis."""
    base_text: str
    variants: List[str] = Field(min_items=2, max_items=10)
    rho0: Optional[str] = None  # Starting rho_id, or will create one
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    basis_pack_id: str = "advanced_narrative_pack"

class ParaphraseVariant(BaseModel):
    """A single paraphrase variant."""
    text: str
    variant_type: str = "paraphrase"  # paraphrase, framing, layout
    metadata: Optional[Dict[str, Any]] = None

class ResidueAnalysisRequest(BaseModel):
    """Request for comprehensive residue analysis."""
    base_passage: str
    variant_groups: List[List[ParaphraseVariant]]
    analysis_depth: str = "standard"  # standard, deep, minimal

# Response Models
class ResidueResult(BaseModel):
    """Result of residue loop analysis."""
    residue_norm: float
    principal_axes: List[str]
    variant_ids: List[str]
    report_id: str
    loop_data: Dict[str, Any]
    interpretation: str

class ResidueReport(BaseModel):
    """Comprehensive residue analysis report."""
    base_text: str
    variants: List[str]
    residue_results: List[ResidueResult]
    holonomy_analysis: Dict[str, Any]
    irony_indicators: Dict[str, float]
    created_at: str

# In-memory storage for residue reports
RESIDUE_REPORTS: Dict[str, ResidueReport] = {}

def compute_matrix_residue(rho_initial: np.ndarray, rho_final: np.ndarray) -> float:
    """
    Compute the residue norm ||Φ_γ - I|| where Φ_γ is the composition of channels
    around the loop and I is the identity.
    """
    try:
        # The residue operator is the difference from identity transformation
        # In practice, we compute ||ρ_final - ρ_initial|| as proxy for channel residue
        residue_matrix = rho_final - rho_initial
        
        # Use Frobenius norm as matrix norm
        residue_norm = float(np.linalg.norm(residue_matrix, 'fro'))
        
        return residue_norm
        
    except Exception as e:
        logger.warning(f"Residue computation failed: {e}")
        return 0.0

def analyze_residue_axes(rho_initial: np.ndarray, rho_final: np.ndarray) -> List[str]:
    """
    Determine which axes (semantic dimensions) are most affected by the residue.
    """
    try:
        residue_matrix = rho_final - rho_initial
        
        # Get eigendecomposition to find principal directions
        eigenvals, eigenvecs = np.linalg.eig(residue_matrix)
        
        # Sort by magnitude of eigenvalues
        idx = np.argsort(np.abs(eigenvals))[::-1]
        
        # Map to semantic axis names (simplified mapping)
        axis_names = [
            "narrator_reliability", "narrative_distance", "temporal_perspective",
            "agency_locus", "causal_grain", "affect_valence", "genre_conformity",
            "discourse_coherence", "cognitive_load", "reader_engagement"
        ]
        
        # Return top 3 most affected axes
        principal_axes = []
        for i in range(min(3, len(idx))):
            axis_idx = idx[i] % len(axis_names)
            principal_axes.append(axis_names[axis_idx])
            
        return principal_axes
        
    except Exception as e:
        logger.warning(f"Axis analysis failed: {e}")
        return ["unknown_axis"]

def detect_irony_indicators(variants: List[str], residue_norm: float) -> Dict[str, float]:
    """
    Detect linguistic indicators of irony based on residue and text analysis.
    """
    indicators = {
        "residue_magnitude": residue_norm,
        "lexical_contrast": 0.0,
        "semantic_incongruity": 0.0,
        "punctuation_signals": 0.0,
        "framing_shift": 0.0
    }
    
    try:
        base_text = variants[0] if variants else ""
        
        # Simple lexical contrast detection
        contrast_words = ["but", "however", "yet", "although", "despite", "ironically"]
        contrast_count = sum(1 for word in contrast_words if word in base_text.lower())
        indicators["lexical_contrast"] = min(contrast_count / 3.0, 1.0)
        
        # Punctuation signals (quotes, question marks, exclamation)
        punct_signals = base_text.count('"') + base_text.count("'") + base_text.count("?") + base_text.count("!")
        indicators["punctuation_signals"] = min(punct_signals / 5.0, 1.0)
        
        # Framing shift between variants
        if len(variants) > 1:
            variant_lengths = [len(v) for v in variants]
            length_variance = np.var(variant_lengths) if len(variant_lengths) > 1 else 0
            indicators["framing_shift"] = min(length_variance / 1000.0, 1.0)
        
        # Semantic incongruity proxy from residue
        indicators["semantic_incongruity"] = min(residue_norm * 10, 1.0)
        
    except Exception as e:
        logger.warning(f"Irony detection failed: {e}")
    
    return indicators

@router.post("/analyze_loop")
async def analyze_residue_loop(request: ResidueLoopRequest) -> ResidueResult:
    """
    Analyze residue in a paraphrase/framing loop.
    
    This implements the core of APLG Claim Set C: detect interpretive residue
    by running a loop of licensed variants and computing Δ = Φ_γ - I.
    """
    try:
        # Initialize or get starting quantum state
        if request.rho0 and request.rho0 in STATE:
            rho_id = request.rho0
            initial_state = STATE[rho_id]["rho"]
        else:
            # Create new quantum state
            init_result = rho_init(seed_text=request.base_text[:100], label=f"Residue_Analysis_{datetime.now().strftime('%H%M%S')}")
            rho_id = init_result["rho_id"]
            initial_state = STATE[rho_id]["rho"]
        
        # Store initial state
        rho_initial = np.array(initial_state) if initial_state is not None else np.eye(64) / 64
        
        # Process each variant in the loop
        current_rho_id = rho_id
        variant_ids = []
        
        for i, variant_text in enumerate(request.variants):
            logger.info(f"Processing variant {i+1}/{len(request.variants)}: {variant_text[:50]}...")
            
            # Apply channel for this variant
            read_req = ReadReq(raw_text=variant_text, alpha=request.alpha)
            result = rho_read_with_channel(current_rho_id, read_req, request.channel_type)
            
            variant_ids.append(f"variant_{i}_{variant_text[:20].replace(' ', '_')}")
        
        # Get final state after loop
        final_state = STATE[current_rho_id]["rho"]
        rho_final = np.array(final_state) if final_state is not None else np.eye(64) / 64
        
        # Compute residue
        residue_norm = compute_matrix_residue(rho_initial, rho_final)
        principal_axes = analyze_residue_axes(rho_initial, rho_final)
        
        # Generate interpretation
        if residue_norm < 0.01:
            interpretation = "Low residue: Variants are semantically equivalent with minimal path-dependence."
        elif residue_norm < 0.05:
            interpretation = "Moderate residue: Some interpretive sensitivity detected, possible subtle framing effects."
        elif residue_norm < 0.15:
            interpretation = "High residue: Significant path-dependence detected, likely irony or ambiguity."
        else:
            interpretation = "Very high residue: Strong interpretive instability, potential narrative twist or deep irony."
        
        # Create report
        report_id = str(uuid.uuid4())
        
        result = ResidueResult(
            residue_norm=residue_norm,
            principal_axes=principal_axes,
            variant_ids=variant_ids,
            report_id=report_id,
            loop_data={
                "base_text": request.base_text,
                "variants": request.variants,
                "channel_type": request.channel_type,
                "alpha": request.alpha,
                "initial_rho_id": rho_id,
                "final_rho_id": current_rho_id
            },
            interpretation=interpretation
        )
        
        # Store report for later retrieval
        RESIDUE_REPORTS[report_id] = ResidueReport(
            base_text=request.base_text,
            variants=request.variants,
            residue_results=[result],
            holonomy_analysis={
                "loop_closure_error": residue_norm,
                "dominant_modes": principal_axes,
                "geometric_interpretation": interpretation
            },
            irony_indicators=detect_irony_indicators(request.variants, residue_norm),
            created_at=datetime.now().isoformat()
        )
        
        logger.info(f"Residue analysis complete: norm={residue_norm:.4f}, axes={principal_axes}")
        return result
        
    except Exception as e:
        logger.error(f"Residue loop analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Residue analysis failed: {str(e)}")

@router.get("/report/{report_id}")
async def get_residue_report(report_id: str) -> ResidueReport:
    """Retrieve a stored residue analysis report."""
    if report_id not in RESIDUE_REPORTS:
        raise HTTPException(status_code=404, detail="Residue report not found")
    
    return RESIDUE_REPORTS[report_id]

@router.post("/batch_analyze")
async def batch_residue_analysis(request: ResidueAnalysisRequest):
    """
    Perform comprehensive residue analysis with multiple variant groups.
    
    This allows testing multiple paraphrase/framing loops on the same base passage
    to build a complete picture of interpretive instabilities.
    """
    try:
        results = []
        
        for group_idx, variant_group in enumerate(request.variant_groups):
            # Convert variant group to simple strings
            variant_texts = [v.text for v in variant_group]
            
            # Create residue request
            loop_request = ResidueLoopRequest(
                base_text=request.base_passage,
                variants=variant_texts,
                channel_type="rank_one_update",
                alpha=0.3
            )
            
            # Analyze this group
            group_result = await analyze_residue_loop(loop_request)
            group_result.loop_data["group_index"] = group_idx
            group_result.loop_data["variant_types"] = [v.variant_type for v in variant_group]
            
            results.append(group_result)
        
        return {
            "base_passage": request.base_passage,
            "total_groups": len(request.variant_groups),
            "results": results,
            "analysis_summary": {
                "max_residue": max(r.residue_norm for r in results),
                "avg_residue": sum(r.residue_norm for r in results) / len(results),
                "affected_axes": list(set(axis for r in results for axis in r.principal_axes))
            }
        }
        
    except Exception as e:
        logger.error(f"Batch residue analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/list_reports")
async def list_residue_reports():
    """List all stored residue reports."""
    reports_summary = []
    
    for report_id, report in RESIDUE_REPORTS.items():
        reports_summary.append({
            "report_id": report_id,
            "base_text_preview": report.base_text[:100] + "..." if len(report.base_text) > 100 else report.base_text,
            "variant_count": len(report.variants),
            "max_residue": max(r.residue_norm for r in report.residue_results),
            "created_at": report.created_at,
            "irony_score": report.irony_indicators.get("semantic_incongruity", 0.0)
        })
    
    return {
        "total_reports": len(RESIDUE_REPORTS),
        "reports": reports_summary
    }

@router.delete("/clear_reports")
async def clear_residue_reports():
    """Clear all stored residue reports."""
    count = len(RESIDUE_REPORTS)
    RESIDUE_REPORTS.clear()
    
    return {
        "cleared": True,
        "reports_removed": count
    }

@router.get("/test_irony")
async def test_irony_detection():
    """
    Test endpoint with known ironic passages to validate residue detection.
    """
    test_cases = [
        {
            "name": "Oscar Wilde Irony",
            "base_text": "I can resist everything except temptation.",
            "variants": [
                "I can resist everything but temptation.",
                "I resist all things save temptation.",
                "Everything can be resisted by me, except temptation."
            ]
        },
        {
            "name": "Situational Irony", 
            "base_text": "The fire station burned down while the firefighters were out on a call.",
            "variants": [
                "The fire station caught fire when firefighters were away responding to an emergency.",
                "While firefighters attended another emergency, their own station was destroyed by fire.",
                "Firefighters returned from a call to find their station had burned down."
            ]
        },
        {
            "name": "Neutral Control",
            "base_text": "The weather is nice today.",
            "variants": [
                "Today's weather is pleasant.",
                "It's a nice day weather-wise.",
                "The weather today is good."
            ]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        request = ResidueLoopRequest(
            base_text=test_case["base_text"],
            variants=test_case["variants"]
        )
        
        result = await analyze_residue_loop(request)
        results.append({
            "test_name": test_case["name"],
            "residue_norm": result.residue_norm,
            "interpretation": result.interpretation,
            "expected_irony": test_case["name"] != "Neutral Control"
        })
    
    return {
        "test_results": results,
        "validation_summary": {
            "ironic_cases_detected": sum(1 for r in results if r["expected_irony"] and r["residue_norm"] > 0.05),
            "neutral_cases_correct": sum(1 for r in results if not r["expected_irony"] and r["residue_norm"] < 0.05),
            "overall_accuracy": "See individual results"
        }
    }