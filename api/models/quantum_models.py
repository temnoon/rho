"""
Standardized Quantum Data Models for Rho Narrative System

This module defines unified data structures for quantum operations across
all components (core math, API routes, frontend). It enforces consistency
in quantum state representation and eliminates data structure mismatches.

Key Principles:
- All quantum matrices use consistent diagnostic structure
- POVM measurements follow standardized format
- Complex quantum results include validation metadata
- Frontend-backend data compatibility guaranteed
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

# ============================================================================
# CORE QUANTUM DATA STRUCTURES
# ============================================================================

class QuantumDiagnostics(BaseModel):
    """Standardized quantum state diagnostics used across all components."""
    trace: float = Field(description="Matrix trace (should be 1.0)")
    purity: float = Field(description="Quantum purity Tr(ρ²)")
    entropy: float = Field(description="Von Neumann entropy -Tr(ρ log ρ)")
    eigenvals: List[float] = Field(description="Eigenvalues (descending order)")
    effective_rank: int = Field(description="Number of significant eigenvalues")
    condition_number: float = Field(description="max(eigenval)/min(eigenval)")
    
    @field_validator('trace')
    @classmethod
    def validate_trace(cls, v):
        if not 0.99 <= v <= 1.01:  # Allow small numerical errors
            raise ValueError(f"Trace must be approximately 1.0, got {v}")
        return v
    
    @field_validator('purity')
    @classmethod
    def validate_purity(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Purity must be between 0 and 1, got {v}")
        return v
    
    @field_validator('entropy')
    @classmethod
    def validate_entropy(cls, v):
        if v < 0:
            raise ValueError(f"Entropy must be non-negative, got {v}")
        return v


class POVMMeasurements(BaseModel):
    """Standardized POVM measurement results."""
    measurements: Dict[str, float] = Field(description="Measurement outcome probabilities")
    pack_id: str = Field(description="ID of the POVM pack used")
    measurement_timestamp: Optional[datetime] = Field(default=None, description="When measurements were taken")
    
    @field_validator('measurements')
    @classmethod
    def validate_measurement_probabilities(cls, v):
        for key, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Measurement {key} must be probability in [0,1], got {value}")
        
        # Check if probabilities sum to approximately 1 for each axis pair
        # This is more complex validation that could be added based on POVM structure
        return v


class QuantumChannelAudit(BaseModel):
    """Audit results for quantum channel operations."""
    trace_preservation: float = Field(description="Trace preservation error")
    psd_check: float = Field(description="Positive semidefinite compliance")
    input_trace: float = Field(description="Input state trace")
    output_trace: float = Field(description="Output state trace")
    kraus_operators: int = Field(description="Number of Kraus operators")
    passes_audit: bool = Field(description="Whether channel passes quantum constraints")


class QuantumState(BaseModel):
    """Complete quantum state representation."""
    rho_id: str = Field(description="Unique quantum state identifier")
    diagnostics: QuantumDiagnostics = Field(description="Quantum diagnostics")
    matrix: Optional[List[List[float]]] = Field(default=None, description="64x64 density matrix")
    label: Optional[str] = Field(default=None, description="Human-readable label")
    creation_timestamp: Optional[datetime] = Field(default=None, description="Creation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class QuantumOperation(BaseModel):
    """Record of a quantum operation applied to a state."""
    operation_id: str = Field(description="Unique operation identifier")
    operation_type: str = Field(description="Type of operation (read, measure, transform)")
    rho_id: str = Field(description="Target quantum state ID")
    parameters: Dict[str, Any] = Field(description="Operation parameters")
    result: Dict[str, Any] = Field(description="Operation results")
    timestamp: datetime = Field(description="Operation timestamp")
    quantum_effects: Optional[Dict[str, float]] = Field(default=None, description="Quantum mechanical effects")

# ============================================================================
# INTEGRABILITY AND ADVANCED QUANTUM STRUCTURES
# ============================================================================

@dataclass
class IntegrabilityTestResult:
    """Results from quantum integrability testing - matches core implementation."""
    segments_a: List[str]
    segments_b: List[str]
    final_state_a: np.ndarray
    final_state_b: np.ndarray
    bures_distance: float
    trace_distance: float
    fidelity: float
    passes_test: bool
    tolerance: float
    channel_logs: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class ResidueAnalysisResult:
    """Results from residue computation analysis - matches core implementation."""
    loop_sequence: List[str]
    initial_state: np.ndarray
    final_state: np.ndarray
    residue_value: complex
    phase_accumulation: float
    monodromy_matrix: np.ndarray
    loop_fidelity: float
    semantic_coherence: float
    singularities_detected: List[Dict[str, Any]]
    recommendations: List[str]


class BuresGeometry(BaseModel):
    """Bures geometric measurements between quantum states."""
    state_a_id: str
    state_b_id: str
    bures_distance: float = Field(description="Bures distance between states")
    quantum_fidelity: float = Field(description="Quantum fidelity")
    trace_distance: float = Field(description="Trace distance")
    geometric_phase: Optional[float] = Field(default=None, description="Geometric phase if applicable")


# ============================================================================
# MATRIX LIBRARY AND QUALITY STRUCTURES
# ============================================================================

class MatrixQuality(BaseModel):
    """Quality assessment for quantum matrices."""
    rho_id: str
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    coherence_score: float = Field(description="Narrative coherence")
    complexity_score: float = Field(description="Complexity appropriateness")
    diversity_score: float = Field(description="Semantic diversity")
    stability_score: float = Field(description="Numerical stability")
    interpretation_confidence: float = Field(description="Interpretation reliability")
    recommendations: List[str] = Field(description="Quality improvement suggestions")


class MatrixMetadata(BaseModel):
    """Comprehensive metadata for quantum matrices."""
    rho_id: str
    label: str
    creation_date: datetime
    source_type: str = Field(description="book, narrative, conversation, synthesis, etc.")
    content_preview: str = Field(max_length=200, description="Preview of source content")
    content_length: int = Field(description="Length of source content")
    reading_history: List[QuantumOperation] = Field(description="Operations applied to this matrix")
    quality_metrics: MatrixQuality = Field(description="Quality assessment")
    tags: Set[str] = Field(description="Categorization tags")
    parent_matrices: List[str] = Field(default=[], description="Parent matrices for synthesis")
    synthesis_method: Optional[str] = Field(default=None, description="Method used for synthesis")


# ============================================================================
# CHANNEL AND TRANSFORMATION STRUCTURES  
# ============================================================================

class ChannelType(str, Enum):
    """Supported quantum channel types."""
    RANK_ONE_UPDATE = "rank_one_update"
    COHERENT_ROTATION = "coherent_rotation"
    DEPHASING_MIXTURE = "dephasing_mixture"
    CUSTOM_CPTP = "custom_cptp"


class TextChannelOperation(BaseModel):
    """Text-to-quantum channel operation record."""
    operation_id: str
    rho_id: str
    text_content: str = Field(description="Text that was processed")
    channel_type: ChannelType
    alpha: float = Field(ge=0.0, le=1.0, description="Blending parameter")
    embedding_dimension: int = Field(description="Embedding space dimension")
    projection_matrix_shape: tuple = Field(description="Shape of projection matrix W")
    channel_audit: QuantumChannelAudit = Field(description="Channel validation results")
    quantum_effects: Dict[str, float] = Field(description="Measured quantum effects")


# ============================================================================
# APLG COMPATIBILITY STRUCTURES
# ============================================================================

class APLGClaimResult(BaseModel):
    """Results from APLG claim validation."""
    claim_id: str = Field(description="APLG claim identifier (A-I)")
    claim_description: str = Field(description="Human-readable claim description")
    test_passed: bool = Field(description="Whether claim validation passed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in result")
    evidence: Dict[str, Any] = Field(description="Supporting evidence")
    quantum_metrics: Dict[str, float] = Field(description="Relevant quantum measurements")


class APLGOperationRequest(BaseModel):
    """Standardized request for APLG operations."""
    operation_type: str = Field(description="APLG operation type")
    rho_id: Optional[str] = Field(default=None, description="Target quantum state")
    parameters: Dict[str, Any] = Field(description="Operation-specific parameters")
    validation_required: bool = Field(default=True, description="Whether to validate quantum constraints")


# ============================================================================
# FRONTEND COMPATIBILITY STRUCTURES
# ============================================================================

class WorkflowStateSnapshot(BaseModel):
    """Snapshot of frontend workflow state for backend processing."""
    current_stage: str = Field(description="Current workflow stage")
    narrative_text: str = Field(description="Input narrative text")
    current_rho_id: Optional[str] = Field(default=None, description="Active quantum state")
    quantum_diagnostics: Optional[QuantumDiagnostics] = Field(default=None)
    povm_measurements: Optional[POVMMeasurements] = Field(default=None)
    transformations: List[Dict[str, Any]] = Field(default=[], description="Applied transformations")
    session_metadata: Dict[str, Any] = Field(description="Session tracking data")


class VisualizationData(BaseModel):
    """Data structure for quantum visualization components."""
    quantum_state: QuantumState
    povm_results: Optional[POVMMeasurements] = None
    trajectory_points: List[Dict[str, float]] = Field(default=[], description="Trajectory visualization data")
    geometric_embedding: Optional[List[List[float]]] = Field(default=None, description="2D/3D embedding coordinates")
    interaction_metadata: Dict[str, Any] = Field(default={}, description="UI interaction data")


# ============================================================================
# ERROR AND VALIDATION STRUCTURES
# ============================================================================

class QuantumValidationError(BaseModel):
    """Quantum constraint violation details."""
    error_type: str = Field(description="Type of quantum violation")
    description: str = Field(description="Human-readable error description")
    measured_value: float = Field(description="Actual measured value")
    expected_range: tuple = Field(description="Expected value range")
    severity: str = Field(description="error, warning, info")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested correction")


class QuantumOperationResponse(BaseModel):
    """Standardized response for all quantum operations."""
    success: bool = Field(description="Whether operation succeeded")
    result: Dict[str, Any] = Field(description="Operation-specific results")
    quantum_state: Optional[QuantumState] = Field(default=None, description="Resulting quantum state")
    validation_errors: List[QuantumValidationError] = Field(default=[], description="Quantum constraint violations")
    performance_metrics: Dict[str, float] = Field(default={}, description="Operation timing and resource usage")
    operation_id: str = Field(description="Unique operation identifier")
    timestamp: datetime = Field(description="Operation completion time")


# ============================================================================
# UTILITY FUNCTIONS FOR DATA CONVERSION
# ============================================================================

def numpy_to_quantum_diagnostics(matrix: np.ndarray) -> QuantumDiagnostics:
    """Convert numpy density matrix to standardized diagnostics."""
    trace = float(np.trace(matrix))
    purity = float(np.trace(matrix @ matrix))
    
    # Calculate entropy
    eigenvals = np.linalg.eigvals(matrix)
    eigenvals = eigenvals[eigenvals > 1e-12]  # Filter numerical zeros
    entropy = -float(np.sum(eigenvals * np.log(eigenvals + 1e-12)))
    
    # Sort eigenvalues descending
    sorted_eigenvals = sorted(eigenvals, reverse=True)
    effective_rank = int(np.sum(eigenvals > 1e-6))
    condition_number = float(max(eigenvals) / max(min(eigenvals), 1e-12))
    
    return QuantumDiagnostics(
        trace=trace,
        purity=purity,
        entropy=entropy,
        eigenvals=sorted_eigenvals[:8],  # Top 8 eigenvalues for display
        effective_rank=effective_rank,
        condition_number=condition_number
    )


def core_result_to_standard(core_result: Any, result_type: str) -> Dict[str, Any]:
    """Convert core quantum results to standardized format."""
    if result_type == "integrability":
        return {
            "segments_a": core_result.segments_a,
            "segments_b": core_result.segments_b,
            "bures_distance": float(core_result.bures_distance),
            "trace_distance": float(core_result.trace_distance),
            "fidelity": float(core_result.fidelity),
            "passes_test": bool(core_result.passes_test),
            "tolerance": float(core_result.tolerance),
            "recommendations": core_result.recommendations
        }
    elif result_type == "residue":
        return {
            "residue_value": complex(core_result.residue_value),
            "phase_accumulation": float(core_result.phase_accumulation),
            "loop_fidelity": float(core_result.loop_fidelity),
            "semantic_coherence": float(core_result.semantic_coherence),
            "recommendations": core_result.recommendations
        }
    else:
        raise ValueError(f"Unknown result type: {result_type}")


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_quantum_constraints(state: QuantumState) -> List[QuantumValidationError]:
    """Validate quantum mechanical constraints on a state."""
    errors = []
    
    # Check trace
    if not 0.99 <= state.diagnostics.trace <= 1.01:
        errors.append(QuantumValidationError(
            error_type="trace_violation",
            description=f"Trace should be 1.0, got {state.diagnostics.trace}",
            measured_value=state.diagnostics.trace,
            expected_range=(0.99, 1.01),
            severity="error",
            suggested_fix="Apply trace normalization"
        ))
    
    # Check purity bounds
    if not 0.0 <= state.diagnostics.purity <= 1.0:
        errors.append(QuantumValidationError(
            error_type="purity_violation", 
            description=f"Purity must be in [0,1], got {state.diagnostics.purity}",
            measured_value=state.diagnostics.purity,
            expected_range=(0.0, 1.0),
            severity="error",
            suggested_fix="Apply PSD projection"
        ))
    
    # Check entropy
    if state.diagnostics.entropy < 0:
        errors.append(QuantumValidationError(
            error_type="entropy_violation",
            description=f"Entropy must be non-negative, got {state.diagnostics.entropy}",
            measured_value=state.diagnostics.entropy,
            expected_range=(0.0, float('inf')),
            severity="error",
            suggested_fix="Check eigenvalue computation"
        ))
    
    return errors