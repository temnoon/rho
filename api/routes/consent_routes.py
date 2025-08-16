"""
Consent/Agency Risk Gating Routes (APLG Claim Set G)

Implements pre-application risk assessment for potentially persuasive or manipulative
content with user consent mechanisms and rollback capabilities. This provides
safeguards for quantum narrative operations that could significantly alter user
perspective or emotional state.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import uuid
from datetime import datetime

from routes.matrix_routes import STATE, rho_read_with_channel
from models.requests import ReadReq

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/consent", tags=["consent-gating"])

# Request Models
class RiskAssessmentRequest(BaseModel):
    """Request for pre-application risk assessment."""
    rho_id: str
    content: str
    channel_type: str = "rank_one_update"
    alpha: float = 0.3
    user_profile: Optional[Dict[str, Any]] = None

class ConsentRequest(BaseModel):
    """Request for user consent with risk disclosure."""
    assessment_id: str
    user_consent: bool
    consent_scope: List[str] = Field(default_factory=list)  # specific risks consented to
    session_context: Optional[Dict[str, str]] = None

class RollbackRequest(BaseModel):
    """Request to rollback to pre-application state."""
    rho_id: str
    checkpoint_id: str
    reason: str = "user_request"

# Response Models
class RiskAssessment(BaseModel):
    """Risk assessment result."""
    assessment_id: str
    rho_id: str
    content_preview: str
    risk_level: str  # low, moderate, high, critical
    risk_factors: List[str]
    predicted_effects: Dict[str, float]
    requires_consent: bool
    consent_questions: List[str]
    estimated_duration: str  # how long effects might last

class ConsentResponse(BaseModel):
    """Response to consent request."""
    assessment_id: str
    consent_granted: bool
    proceed_with_application: bool
    checkpoint_created: bool
    checkpoint_id: Optional[str]

# In-memory storage for assessments and checkpoints
RISK_ASSESSMENTS: Dict[str, RiskAssessment] = {}
CONSENT_RECORDS: Dict[str, ConsentResponse] = {}
STATE_CHECKPOINTS: Dict[str, Dict[str, Any]] = {}

def analyze_content_risks(content: str, rho_current: np.ndarray) -> Dict[str, Any]:
    """
    Analyze content for potential psychological/persuasive risks.
    """
    risks = {
        "emotional_manipulation": 0.0,
        "cognitive_bias_exploitation": 0.0,
        "narrative_hijacking": 0.0,
        "identity_destabilization": 0.0,
        "false_authority": 0.0,
        "temporal_disorientation": 0.0
    }
    
    try:
        content_lower = content.lower()
        
        # Emotional manipulation indicators
        emotion_triggers = ["must", "urgent", "crisis", "fear", "panic", "desperate", "final chance"]
        emotion_score = sum(1 for trigger in emotion_triggers if trigger in content_lower)
        risks["emotional_manipulation"] = min(emotion_score / 3.0, 1.0)
        
        # Cognitive bias exploitation
        bias_patterns = ["everyone knows", "experts agree", "studies show", "proven fact", "undeniable"]
        bias_score = sum(1 for pattern in bias_patterns if pattern in content_lower)
        risks["cognitive_bias_exploitation"] = min(bias_score / 2.0, 1.0)
        
        # Narrative hijacking (first/second person, commands)
        hijack_patterns = ["you should", "you must", "you are", "imagine you", "what if you"]
        hijack_score = sum(1 for pattern in hijack_patterns if pattern in content_lower)
        risks["narrative_hijacking"] = min(hijack_score / 2.0, 1.0)
        
        # Identity destabilization
        identity_triggers = ["who you really are", "your true self", "you've been wrong", "everything you know"]
        identity_score = sum(1 for trigger in identity_triggers if trigger in content_lower)
        risks["identity_destabilization"] = min(identity_score / 1.0, 1.0)
        
        # False authority
        authority_claims = ["trust me", "i know", "believe me", "take my word", "as an expert"]
        authority_score = sum(1 for claim in authority_claims if claim in content_lower)
        risks["false_authority"] = min(authority_score / 2.0, 1.0)
        
        # Content length risk (very long content can be overwhelming)
        length_risk = min(len(content) / 5000.0, 1.0)
        risks["temporal_disorientation"] = length_risk
        
    except Exception as e:
        logger.warning(f"Content risk analysis failed: {e}")
    
    return risks

def predict_quantum_effects(rho_current: np.ndarray, content: str, alpha: float) -> Dict[str, float]:
    """
    Predict quantum state changes from applying the content.
    """
    try:
        from core.embedding import text_to_embedding_vector
        from core.quantum_state import apply_text_channel
        
        # Simulate the application without actually applying it
        text_embedding = text_to_embedding_vector(content)
        predicted_rho = apply_text_channel(rho_current, text_embedding, alpha, "rank_one_update")
        
        # Calculate predicted changes
        current_purity = float(np.real(np.trace(rho_current @ rho_current)))
        predicted_purity = float(np.real(np.trace(predicted_rho @ predicted_rho)))
        
        current_eigs = np.linalg.eigvals(rho_current)
        predicted_eigs = np.linalg.eigvals(predicted_rho)
        
        current_entropy = -np.sum(current_eigs * np.log2(current_eigs + 1e-10))
        predicted_entropy = -np.sum(predicted_eigs * np.log2(predicted_eigs + 1e-10))
        
        return {
            "purity_change": float(predicted_purity - current_purity),
            "entropy_change": float(predicted_entropy - current_entropy), 
            "eigenvalue_shift": float(np.linalg.norm(predicted_eigs - current_eigs)),
            "trace_preservation": float(abs(np.trace(predicted_rho) - 1.0))
        }
        
    except Exception as e:
        logger.warning(f"Quantum effect prediction failed: {e}")
        return {
            "purity_change": 0.0,
            "entropy_change": 0.0,
            "eigenvalue_shift": 0.0,
            "trace_preservation": 0.0
        }

def determine_risk_level(risk_factors: Dict[str, float], quantum_effects: Dict[str, float]) -> str:
    """
    Determine overall risk level from individual factors.
    """
    # Calculate weighted risk score
    content_risk = max(risk_factors.values())
    quantum_risk = max(
        abs(quantum_effects.get("purity_change", 0)),
        abs(quantum_effects.get("entropy_change", 0)),
        quantum_effects.get("eigenvalue_shift", 0) / 10.0
    )
    
    overall_risk = max(content_risk, quantum_risk)
    
    if overall_risk < 0.2:
        return "low"
    elif overall_risk < 0.5:
        return "moderate"
    elif overall_risk < 0.8:
        return "high"
    else:
        return "critical"

def generate_consent_questions(risk_factors: Dict[str, float], risk_level: str) -> List[str]:
    """
    Generate appropriate consent questions based on identified risks.
    """
    questions = []
    
    if risk_level in ["high", "critical"]:
        questions.append("Do you understand that this content may significantly alter your perspective?")
    
    if risk_factors.get("emotional_manipulation", 0) > 0.3:
        questions.append("This content contains emotionally charged language. Do you consent to potential emotional impact?")
    
    if risk_factors.get("narrative_hijacking", 0) > 0.3:
        questions.append("This content uses direct address and commands. Do you consent to perspective-taking?")
        
    if risk_factors.get("identity_destabilization", 0) > 0.3:
        questions.append("This content may challenge core beliefs. Do you consent to identity questioning?")
        
    if risk_factors.get("cognitive_bias_exploitation", 0) > 0.3:
        questions.append("This content makes authoritative claims. Do you consent to persuasive arguments?")
    
    if not questions:
        questions.append("Do you consent to applying this content to your narrative state?")
    
    return questions

@router.post("/assess")
async def assess_content_risk(request: RiskAssessmentRequest) -> RiskAssessment:
    """
    Assess risk of applying content before actual application.
    
    This implements APLG Claim Set G: pre-application risk assessment
    with user consent mechanisms.
    """
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        current_rho = STATE[request.rho_id]["rho"]
        
        # Analyze content risks
        risk_factors = analyze_content_risks(request.content, current_rho)
        
        # Predict quantum effects
        quantum_effects = predict_quantum_effects(current_rho, request.content, request.alpha)
        
        # Determine overall risk level
        risk_level = determine_risk_level(risk_factors, quantum_effects)
        
        # Generate consent questions
        consent_questions = generate_consent_questions(risk_factors, risk_level)
        
        # Create assessment
        assessment_id = str(uuid.uuid4())
        assessment = RiskAssessment(
            assessment_id=assessment_id,
            rho_id=request.rho_id,
            content_preview=request.content[:200] + "..." if len(request.content) > 200 else request.content,
            risk_level=risk_level,
            risk_factors=[f"{k}: {v:.2f}" for k, v in risk_factors.items() if v > 0.1],
            predicted_effects=quantum_effects,
            requires_consent=risk_level in ["moderate", "high", "critical"],
            consent_questions=consent_questions,
            estimated_duration="temporary" if risk_level == "low" else "session" if risk_level == "moderate" else "persistent"
        )
        
        # Store assessment
        RISK_ASSESSMENTS[assessment_id] = assessment
        
        logger.info(f"Risk assessment {assessment_id}: {risk_level} risk for rho {request.rho_id}")
        return assessment
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.post("/consent")
async def process_consent(request: ConsentRequest) -> ConsentResponse:
    """
    Process user consent and create checkpoint if proceeding.
    """
    try:
        if request.assessment_id not in RISK_ASSESSMENTS:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        assessment = RISK_ASSESSMENTS[request.assessment_id]
        
        if not request.user_consent and assessment.requires_consent:
            # User declined consent
            response = ConsentResponse(
                assessment_id=request.assessment_id,
                consent_granted=False,
                proceed_with_application=False,
                checkpoint_created=False,
                checkpoint_id=None
            )
            CONSENT_RECORDS[request.assessment_id] = response
            return response
        
        # User consented or no consent required
        checkpoint_id = None
        checkpoint_created = False
        
        # Create checkpoint for rollback capability
        if assessment.risk_level in ["moderate", "high", "critical"]:
            checkpoint_id = str(uuid.uuid4())
            rho_id = assessment.rho_id
            
            if rho_id in STATE:
                # Deep copy current state for checkpoint
                current_state = STATE[rho_id]
                checkpoint_data = {
                    "rho": np.array(current_state["rho"]),
                    "ops": list(current_state["ops"]),
                    "narratives": list(current_state["narratives"]),
                    "label": current_state["label"],
                    "created_at": datetime.now().isoformat(),
                    "checkpoint_reason": f"Pre-risk-application checkpoint for assessment {request.assessment_id}"
                }
                STATE_CHECKPOINTS[checkpoint_id] = checkpoint_data
                checkpoint_created = True
                logger.info(f"Created checkpoint {checkpoint_id} for rho {rho_id}")
        
        response = ConsentResponse(
            assessment_id=request.assessment_id,
            consent_granted=request.user_consent,
            proceed_with_application=True,
            checkpoint_created=checkpoint_created,
            checkpoint_id=checkpoint_id
        )
        
        CONSENT_RECORDS[request.assessment_id] = response
        return response
        
    except Exception as e:
        logger.error(f"Consent processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consent processing failed: {str(e)}")

@router.post("/rollback")
async def rollback_to_checkpoint(request: RollbackRequest):
    """
    Rollback quantum state to a previous checkpoint.
    """
    try:
        if request.checkpoint_id not in STATE_CHECKPOINTS:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        # Restore state from checkpoint
        checkpoint_data = STATE_CHECKPOINTS[request.checkpoint_id]
        
        STATE[request.rho_id] = {
            "rho": np.array(checkpoint_data["rho"]),
            "ops": list(checkpoint_data["ops"]),
            "narratives": list(checkpoint_data["narratives"]),
            "label": checkpoint_data["label"]
        }
        
        # Add rollback operation to history
        STATE[request.rho_id]["ops"].append({
            "op": "rollback",
            "checkpoint_id": request.checkpoint_id,
            "reason": request.reason,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Rolled back rho {request.rho_id} to checkpoint {request.checkpoint_id}")
        
        return {
            "success": True,
            "rho_id": request.rho_id,
            "checkpoint_id": request.checkpoint_id,
            "rollback_reason": request.reason,
            "state_restored": True
        }
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")

@router.get("/checkpoints/{rho_id}")
async def list_checkpoints(rho_id: str):
    """
    List available checkpoints for a matrix.
    """
    checkpoints = []
    
    for checkpoint_id, checkpoint_data in STATE_CHECKPOINTS.items():
        # Check if this checkpoint relates to the requested rho_id
        # (We'd need to track this more systematically in production)
        checkpoints.append({
            "checkpoint_id": checkpoint_id,
            "created_at": checkpoint_data.get("created_at"),
            "reason": checkpoint_data.get("checkpoint_reason", "Unknown"),
            "operations_count": len(checkpoint_data.get("ops", [])),
            "narratives_count": len(checkpoint_data.get("narratives", []))
        })
    
    return {
        "rho_id": rho_id,
        "available_checkpoints": checkpoints,
        "total_checkpoints": len(checkpoints)
    }

@router.get("/assessment/{assessment_id}")
async def get_assessment(assessment_id: str) -> RiskAssessment:
    """
    Retrieve a stored risk assessment.
    """
    if assessment_id not in RISK_ASSESSMENTS:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    return RISK_ASSESSMENTS[assessment_id]

@router.delete("/clear_assessments")
async def clear_assessments():
    """
    Clear all stored assessments and consent records.
    """
    assessment_count = len(RISK_ASSESSMENTS)
    consent_count = len(CONSENT_RECORDS)
    checkpoint_count = len(STATE_CHECKPOINTS)
    
    RISK_ASSESSMENTS.clear()
    CONSENT_RECORDS.clear()
    STATE_CHECKPOINTS.clear()
    
    return {
        "cleared": True,
        "assessments_removed": assessment_count,
        "consent_records_removed": consent_count,
        "checkpoints_removed": checkpoint_count
    }

@router.get("/test_risk_detection")
async def test_risk_detection():
    """
    Test endpoint with known risky content to validate detection.
    """
    test_cases = [
        {
            "name": "Emotional Manipulation",
            "content": "You MUST act now! This is your FINAL CHANCE to avoid disaster! Don't let fear control you - but you should be very afraid if you don't follow my advice immediately!"
        },
        {
            "name": "Identity Destabilization", 
            "content": "Everything you know about yourself is wrong. Your true self has been hidden from you. Who you really are is completely different from who you think you are."
        },
        {
            "name": "Cognitive Bias Exploitation",
            "content": "Studies show that experts agree this is a proven fact. Everyone knows this undeniable truth that scientists have confirmed beyond doubt."
        },
        {
            "name": "Neutral Control",
            "content": "This is a simple factual statement about the weather being pleasant today."
        }
    ]
    
    # Create a test matrix for risk assessment
    from routes.matrix_routes import rho_init
    test_init = rho_init(label="Risk_Assessment_Test")
    test_rho_id = test_init["rho_id"]
    
    results = []
    
    for test_case in test_cases:
        request = RiskAssessmentRequest(
            rho_id=test_rho_id,
            content=test_case["content"],
            alpha=0.3
        )
        
        assessment = await assess_content_risk(request)
        results.append({
            "test_name": test_case["name"],
            "risk_level": assessment.risk_level,
            "requires_consent": assessment.requires_consent,
            "risk_factors": assessment.risk_factors,
            "expected_risky": test_case["name"] != "Neutral Control"
        })
    
    # Clean up test matrix
    if test_rho_id in STATE:
        del STATE[test_rho_id]
    
    return {
        "test_results": results,
        "validation_summary": {
            "risky_cases_detected": sum(1 for r in results if r["expected_risky"] and r["requires_consent"]),
            "neutral_cases_correct": sum(1 for r in results if not r["expected_risky"] and not r["requires_consent"]),
            "overall_detection_rate": "See individual results"
        }
    }