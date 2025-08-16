"""
Invariant-Preserving Editor Routes (APLG Claim Set D)

Implements content transformation operations that preserve specified narrative
invariants while achieving targeted changes. This provides sophisticated text
editing capabilities that maintain quantum state properties like purity,
specific attribute measurements, or narrative coherence measures.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import uuid
from datetime import datetime

from routes.matrix_routes import STATE, rho_read_with_channel
from routes.povm_routes import measure_rho
from models.requests import ReadReq

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/invariant-editor", tags=["invariant-editor"])

# Request Models
class InvariantSpec(BaseModel):
    """Specification of an invariant to preserve."""
    type: str  # "purity", "measurement", "entropy", "eigenvalue_span", "custom"
    name: str
    target_value: Optional[float] = None
    tolerance: float = 0.05
    measurement_pack: Optional[str] = None
    measurement_axis: Optional[str] = None

class TransformationTarget(BaseModel):
    """Target for text transformation."""
    type: str  # "style", "tone", "perspective", "content", "length"
    description: str
    target_value: Optional[str] = None
    intensity: float = 0.5  # 0-1 scale

class InvariantEditRequest(BaseModel):
    """Request for invariant-preserving text editing."""
    rho_id: str
    original_text: str
    invariants: List[InvariantSpec]
    targets: List[TransformationTarget]
    max_iterations: int = 20
    convergence_threshold: float = 0.01
    preserve_meaning: bool = True

class EditCandidate(BaseModel):
    """A candidate text edit."""
    text: str
    invariant_violations: Dict[str, float]
    target_achievement: Dict[str, float]
    overall_score: float
    quantum_effects: Dict[str, float]

class InvariantEditResult(BaseModel):
    """Result of invariant-preserving editing."""
    success: bool
    original_text: str
    edited_text: str
    iterations: int
    invariant_preservation: Dict[str, float]
    target_achievement: Dict[str, float]
    final_score: float
    candidates_explored: int
    editing_log: List[str]

# In-memory storage for editing sessions
EDITING_SESSIONS: Dict[str, Dict[str, Any]] = {}

def measure_invariant(rho: np.ndarray, invariant: InvariantSpec) -> float:
    """
    Measure the current value of a specified invariant.
    """
    try:
        if invariant.type == "purity":
            # Calculate purity: Tr(ρ²)
            return float(np.real(np.trace(rho @ rho)))
            
        elif invariant.type == "entropy":
            # Calculate von Neumann entropy
            eigenvals = np.linalg.eigvals(rho)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Filter near-zero
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
            return float(entropy)
            
        elif invariant.type == "eigenvalue_span":
            # Calculate span of significant eigenvalues
            eigenvals = np.linalg.eigvals(rho)
            eigenvals = np.sort(eigenvals[eigenvals > 1e-10])[::-1]
            if len(eigenvals) > 1:
                return float(eigenvals[0] - eigenvals[-1])
            return float(eigenvals[0]) if len(eigenvals) > 0 else 0.0
            
        elif invariant.type == "measurement":
            # Measurement-based invariant (simplified)
            # In production, this would use actual POVM measurements
            if invariant.measurement_pack and invariant.measurement_axis:
                # Placeholder: simulate measurement
                from routes.povm_routes import PACKS
                if invariant.measurement_pack in PACKS:
                    # Simple projection measurement simulation
                    trace_val = float(np.real(np.trace(rho)))
                    return trace_val * 0.5  # Placeholder calculation
            return 0.5
            
        elif invariant.type == "custom":
            # Custom invariant (placeholder)
            return float(np.real(np.trace(rho)))
            
        else:
            logger.warning(f"Unknown invariant type: {invariant.type}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Failed to measure invariant {invariant.name}: {e}")
        return 0.0

def evaluate_target_achievement(original_text: str, candidate_text: str, target: TransformationTarget) -> float:
    """
    Evaluate how well a candidate text achieves the transformation target.
    """
    try:
        if target.type == "style":
            # Style change evaluation (simplified heuristics)
            if "formal" in target.description.lower():
                formal_indicators = ["therefore", "furthermore", "consequently", "moreover"]
                score = sum(1 for indicator in formal_indicators if indicator in candidate_text.lower())
                return min(score / 3.0, 1.0)
            elif "casual" in target.description.lower():
                casual_indicators = ["hey", "yeah", "like", "kinda", "pretty much"]
                score = sum(1 for indicator in casual_indicators if indicator in candidate_text.lower())
                return min(score / 3.0, 1.0)
                
        elif target.type == "tone":
            # Tone change evaluation
            if "positive" in target.description.lower():
                positive_words = ["good", "great", "wonderful", "excellent", "amazing", "fantastic"]
                score = sum(1 for word in positive_words if word in candidate_text.lower())
                return min(score / 2.0, 1.0)
            elif "negative" in target.description.lower():
                negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
                score = sum(1 for word in negative_words if word in candidate_text.lower())
                return min(score / 2.0, 1.0)
                
        elif target.type == "perspective":
            # Perspective change evaluation
            if "first_person" in target.description.lower():
                first_person = ["I", "me", "my", "mine", "myself"]
                score = sum(candidate_text.count(word) for word in first_person)
                return min(score / 5.0, 1.0)
            elif "third_person" in target.description.lower():
                third_person = ["he", "she", "they", "them", "his", "her", "their"]
                score = sum(candidate_text.count(word) for word in third_person)
                return min(score / 5.0, 1.0)
                
        elif target.type == "length":
            # Length change evaluation
            original_len = len(original_text)
            candidate_len = len(candidate_text)
            if "shorter" in target.description.lower():
                reduction = max(0, original_len - candidate_len) / original_len
                return min(reduction * 2, 1.0)
            elif "longer" in target.description.lower():
                expansion = max(0, candidate_len - original_len) / original_len
                return min(expansion, 1.0)
                
        elif target.type == "content":
            # Content change evaluation (keyword-based)
            if target.target_value:
                keywords = target.target_value.lower().split()
                score = sum(1 for keyword in keywords if keyword in candidate_text.lower())
                return min(score / len(keywords), 1.0) if keywords else 0.0
        
        # Default scoring based on text similarity
        if len(candidate_text) > 0:
            return 0.5  # Neutral score for unrecognized targets
        return 0.0
        
    except Exception as e:
        logger.error(f"Failed to evaluate target {target.type}: {e}")
        return 0.0

def generate_text_variants(text: str, targets: List[TransformationTarget], iteration: int) -> List[str]:
    """
    Generate variant texts attempting to achieve the specified targets.
    """
    variants = []
    
    try:
        # Strategy 1: Lexical substitution
        if any(t.type == "tone" for t in targets):
            positive_target = next((t for t in targets if t.type == "tone" and "positive" in t.description.lower()), None)
            if positive_target:
                # Simple positive word substitution
                positive_subs = {
                    "bad": "good", "terrible": "wonderful", "awful": "amazing",
                    "hate": "love", "difficult": "manageable", "problem": "challenge"
                }
                variant = text
                for neg, pos in positive_subs.items():
                    variant = variant.replace(neg, pos)
                if variant != text:
                    variants.append(variant)
        
        # Strategy 2: Perspective shift
        if any(t.type == "perspective" for t in targets):
            first_person_target = next((t for t in targets if t.type == "perspective" and "first_person" in t.description.lower()), None)
            if first_person_target:
                # Convert to first person
                variant = text.replace("you", "I").replace("your", "my").replace("You", "I").replace("Your", "My")
                if variant != text:
                    variants.append(variant)
        
        # Strategy 3: Style formalization
        if any(t.type == "style" for t in targets):
            formal_target = next((t for t in targets if t.type == "style" and "formal" in t.description.lower()), None)
            if formal_target:
                # Add formal connectors
                formal_phrases = ["Furthermore, ", "In addition, ", "Moreover, ", "Consequently, "]
                if iteration < len(formal_phrases):
                    variant = formal_phrases[iteration] + text
                    variants.append(variant)
        
        # Strategy 4: Length adjustment
        if any(t.type == "length" for t in targets):
            shorter_target = next((t for t in targets if t.type == "length" and "shorter" in t.description.lower()), None)
            if shorter_target:
                # Remove adjectives and adverbs (simplified)
                words = text.split()
                if len(words) > 3:
                    # Remove every 3rd word as approximation
                    shortened = [word for i, word in enumerate(words) if i % 3 != 2]
                    variant = " ".join(shortened)
                    variants.append(variant)
            
            longer_target = next((t for t in targets if t.type == "length" and "longer" in t.description.lower()), None)
            if longer_target:
                # Add descriptive phrases
                descriptive_additions = [" with great care", " in a thoughtful manner", " with considerable attention", " through careful consideration"]
                if iteration < len(descriptive_additions):
                    variant = text + descriptive_additions[iteration]
                    variants.append(variant)
        
        # Strategy 5: Content augmentation
        if any(t.type == "content" for t in targets):
            content_target = next((t for t in targets if t.type == "content" and t.target_value), None)
            if content_target and content_target.target_value:
                # Add target content
                variant = text + f" {content_target.target_value}"
                variants.append(variant)
        
        # If no variants generated, create minor variations
        if not variants:
            # Minor rephrasing
            basic_variants = [
                text + ".",
                text.replace(".", ","),
                text.replace(" and ", " as well as "),
                text.replace(" but ", " however ")
            ]
            variants.extend([v for v in basic_variants if v != text])
    
    except Exception as e:
        logger.error(f"Failed to generate text variants: {e}")
        variants = [text]  # Fallback to original
    
    return variants[:5]  # Limit to 5 variants per iteration

def score_candidate(rho_id: str, candidate_text: str, original_text: str, 
                   invariants: List[InvariantSpec], targets: List[TransformationTarget]) -> EditCandidate:
    """
    Score a candidate text edit based on invariant preservation and target achievement.
    """
    try:
        # Simulate applying the candidate text to get quantum effects
        from core.embedding import text_to_embedding_vector
        from core.quantum_state import apply_text_channel
        
        current_rho = STATE[rho_id]["rho"]
        candidate_embedding = text_to_embedding_vector(candidate_text)
        candidate_rho = apply_text_channel(current_rho, candidate_embedding, 0.3, "rank_one_update")
        
        # Measure invariant violations
        invariant_violations = {}
        for invariant in invariants:
            current_value = measure_invariant(candidate_rho, invariant)
            if invariant.target_value is not None:
                violation = abs(current_value - invariant.target_value) / (invariant.tolerance + 1e-10)
            else:
                # For invariants without target, measure preservation from original
                original_value = measure_invariant(current_rho, invariant)
                violation = abs(current_value - original_value) / (invariant.tolerance + 1e-10)
            invariant_violations[invariant.name] = min(violation, 10.0)  # Cap violations
        
        # Measure target achievement
        target_achievement = {}
        for target in targets:
            achievement = evaluate_target_achievement(original_text, candidate_text, target)
            target_achievement[target.description] = achievement
        
        # Calculate overall score
        avg_violation = np.mean(list(invariant_violations.values())) if invariant_violations else 0.0
        avg_achievement = np.mean(list(target_achievement.values())) if target_achievement else 0.0
        
        # Score formula: prioritize invariant preservation, then target achievement
        violation_penalty = min(avg_violation, 5.0)  # Cap penalty
        overall_score = max(0, 1.0 - violation_penalty * 0.5) * (0.7 + 0.3 * avg_achievement)
        
        # Calculate quantum effects
        purity_change = float(np.real(np.trace(candidate_rho @ candidate_rho) - np.trace(current_rho @ current_rho)))
        quantum_effects = {
            "purity_change": purity_change,
            "trace_preservation": float(abs(np.trace(candidate_rho) - 1.0))
        }
        
        return EditCandidate(
            text=candidate_text,
            invariant_violations=invariant_violations,
            target_achievement=target_achievement,
            overall_score=overall_score,
            quantum_effects=quantum_effects
        )
        
    except Exception as e:
        logger.error(f"Failed to score candidate: {e}")
        return EditCandidate(
            text=candidate_text,
            invariant_violations={inv.name: 10.0 for inv in invariants},
            target_achievement={tgt.description: 0.0 for tgt in targets},
            overall_score=0.0,
            quantum_effects={"error": str(e)}
        )

@router.post("/edit")
async def invariant_preserving_edit(request: InvariantEditRequest) -> InvariantEditResult:
    """
    Perform invariant-preserving text editing.
    
    This implements APLG Claim Set D: transform text while preserving
    specified quantum state invariants.
    """
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        # Initialize editing session
        session_id = str(uuid.uuid4())
        editing_log = [f"Starting invariant-preserving edit session {session_id}"]
        
        current_text = request.original_text
        best_candidate = None
        candidates_explored = 0
        
        # Iterative improvement process
        for iteration in range(request.max_iterations):
            editing_log.append(f"Iteration {iteration + 1}: Generating variants")
            
            # Generate candidate variants
            variants = generate_text_variants(current_text, request.targets, iteration)
            
            # Score each candidate
            iteration_candidates = []
            for variant in variants:
                candidate = score_candidate(request.rho_id, variant, request.original_text, 
                                          request.invariants, request.targets)
                iteration_candidates.append(candidate)
                candidates_explored += 1
            
            # Select best candidate from this iteration
            if iteration_candidates:
                iteration_best = max(iteration_candidates, key=lambda c: c.overall_score)
                
                if best_candidate is None or iteration_best.overall_score > best_candidate.overall_score:
                    best_candidate = iteration_best
                    current_text = iteration_best.text
                    editing_log.append(f"New best candidate found: score={iteration_best.overall_score:.3f}")
                else:
                    editing_log.append(f"No improvement in iteration {iteration + 1}")
            
            # Check convergence
            if best_candidate and best_candidate.overall_score > (1.0 - request.convergence_threshold):
                editing_log.append(f"Converged after {iteration + 1} iterations")
                break
            
            # Early stopping if no viable candidates
            if not iteration_candidates or max(c.overall_score for c in iteration_candidates) < 0.1:
                editing_log.append(f"Stopping early: no viable candidates")
                break
        
        # Determine success
        success = best_candidate is not None and best_candidate.overall_score > 0.3
        
        # Prepare result
        final_text = best_candidate.text if best_candidate else request.original_text
        invariant_preservation = best_candidate.invariant_violations if best_candidate else {}
        target_achievement = best_candidate.target_achievement if best_candidate else {}
        final_score = best_candidate.overall_score if best_candidate else 0.0
        
        editing_log.append(f"Edit session complete: success={success}, final_score={final_score:.3f}")
        
        # Store session results
        EDITING_SESSIONS[session_id] = {
            "request": request.dict(),
            "result": {
                "success": success,
                "iterations": iteration + 1,
                "candidates_explored": candidates_explored,
                "final_score": final_score
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return InvariantEditResult(
            success=success,
            original_text=request.original_text,
            edited_text=final_text,
            iterations=iteration + 1,
            invariant_preservation=invariant_preservation,
            target_achievement=target_achievement,
            final_score=final_score,
            candidates_explored=candidates_explored,
            editing_log=editing_log
        )
        
    except Exception as e:
        logger.error(f"Invariant-preserving edit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Edit operation failed: {str(e)}")

@router.post("/preview_edit")
async def preview_invariant_edit(request: InvariantEditRequest):
    """
    Preview the effects of invariant-preserving editing without applying changes.
    """
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        # Generate a few candidate variants
        variants = generate_text_variants(request.original_text, request.targets, 0)[:3]
        
        # Score each variant
        candidate_previews = []
        for variant in variants:
            candidate = score_candidate(request.rho_id, variant, request.original_text, 
                                      request.invariants, request.targets)
            candidate_previews.append({
                "text": candidate.text,
                "score": candidate.overall_score,
                "invariant_violations": candidate.invariant_violations,
                "target_achievement": candidate.target_achievement,
                "quantum_effects": candidate.quantum_effects
            })
        
        return {
            "original_text": request.original_text,
            "candidate_previews": candidate_previews,
            "invariant_specs": [inv.dict() for inv in request.invariants],
            "target_specs": [tgt.dict() for tgt in request.targets],
            "preview_note": "These are preliminary candidates. Actual editing may produce different results."
        }
        
    except Exception as e:
        logger.error(f"Edit preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@router.get("/sessions")
async def list_editing_sessions():
    """
    List all stored editing sessions.
    """
    sessions = []
    for session_id, session_data in EDITING_SESSIONS.items():
        sessions.append({
            "session_id": session_id,
            "timestamp": session_data["timestamp"],
            "success": session_data["result"]["success"],
            "iterations": session_data["result"]["iterations"],
            "candidates_explored": session_data["result"]["candidates_explored"],
            "final_score": session_data["result"]["final_score"],
            "original_text_preview": session_data["request"]["original_text"][:100] + "..." if len(session_data["request"]["original_text"]) > 100 else session_data["request"]["original_text"]
        })
    
    return {
        "total_sessions": len(EDITING_SESSIONS),
        "sessions": sessions
    }

@router.get("/session/{session_id}")
async def get_editing_session(session_id: str):
    """
    Retrieve detailed information about a specific editing session.
    """
    if session_id not in EDITING_SESSIONS:
        raise HTTPException(status_code=404, detail="Editing session not found")
    
    return EDITING_SESSIONS[session_id]

@router.delete("/clear_sessions")
async def clear_editing_sessions():
    """
    Clear all stored editing sessions.
    """
    count = len(EDITING_SESSIONS)
    EDITING_SESSIONS.clear()
    
    return {
        "cleared": True,
        "sessions_removed": count
    }

@router.post("/test_invariant_preservation")
async def test_invariant_preservation():
    """
    Test endpoint to validate invariant preservation capabilities.
    """
    from routes.matrix_routes import rho_init
    
    # Create test matrix
    test_init = rho_init(seed_text="Test matrix for invariant preservation")
    test_rho_id = test_init["rho_id"]
    
    try:
        test_cases = [
            {
                "name": "Purity Preservation with Style Change",
                "original_text": "The weather is bad today.",
                "invariants": [
                    InvariantSpec(type="purity", name="quantum_purity", tolerance=0.05)
                ],
                "targets": [
                    TransformationTarget(type="tone", description="positive tone", intensity=0.7)
                ]
            },
            {
                "name": "Entropy Preservation with Length Change",
                "original_text": "This is a simple sentence about nothing in particular.",
                "invariants": [
                    InvariantSpec(type="entropy", name="state_entropy", tolerance=0.1)
                ],
                "targets": [
                    TransformationTarget(type="length", description="shorter text", intensity=0.5)
                ]
            }
        ]
        
        results = []
        for test_case in test_cases:
            request = InvariantEditRequest(
                rho_id=test_rho_id,
                original_text=test_case["original_text"],
                invariants=test_case["invariants"],
                targets=test_case["targets"],
                max_iterations=5
            )
            
            result = await invariant_preserving_edit(request)
            results.append({
                "test_name": test_case["name"],
                "success": result.success,
                "original_text": result.original_text,
                "edited_text": result.edited_text,
                "invariant_preservation": result.invariant_preservation,
                "target_achievement": result.target_achievement,
                "final_score": result.final_score
            })
        
        return {
            "test_results": results,
            "validation_summary": {
                "successful_edits": sum(1 for r in results if r["success"]),
                "total_tests": len(results),
                "average_score": sum(r["final_score"] for r in results) / len(results) if results else 0
            }
        }
        
    finally:
        # Clean up test matrix
        if test_rho_id in STATE:
            del STATE[test_rho_id]