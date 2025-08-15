"""
Pydantic models for API request/response schemas.

This module defines all the data models used in the API endpoints
for type validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class ReadReq(BaseModel):
    """Request model for reading text into a density matrix."""
    raw_text: Optional[str] = None
    text_id: Optional[str] = None
    alpha: float = Field(default=0.2, ge=0.0, le=1.0, description="Blending parameter")


class MeasureReq(BaseModel):
    """Request model for POVM measurements."""
    pack_id: str = Field(description="ID of the POVM pack to apply")


class PackAxis(BaseModel):
    """Model for a single axis in a POVM pack."""
    id: str = Field(description="Unique identifier for the axis")
    labels: List[str] = Field(description="Labels for the measurement outcomes")


class PackModel(BaseModel):
    """Model for a complete POVM pack."""
    pack_id: str = Field(description="Unique identifier for the pack")
    axes: List[PackAxis] = Field(description="List of measurement axes")


class ExplainReq(BaseModel):
    """Request model for operation explanations."""
    rho_id: str = Field(description="ID of the density matrix")
    last_n: int = Field(default=1, ge=1, description="Number of recent operations to explain")


class JobStatus(str, Enum):
    """Enumeration of job processing states."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Enumeration of job priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BookIngestionReq(BaseModel):
    """Request model for book ingestion."""
    gutenberg_id: str = Field(description="Project Gutenberg book ID")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Characters per chunk")
    reading_alpha: float = Field(default=0.1, ge=0.01, le=1.0, description="Reading blending parameter")


class BatchJobRequest(BaseModel):
    """Request model for batch job submission."""
    search_queries: List[str] = Field(description="List of search queries for books")
    instructions: str = Field(
        default="Process books with standard settings for comprehensive analysis",
        description="Processing instructions"
    )
    priority: JobPriority = Field(default=JobPriority.MEDIUM, description="Job priority")


class QueuedJob(BaseModel):
    """Model for a queued job."""
    job_id: str = Field(description="Unique job identifier")
    book_title: str = Field(description="Title of the book to process")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Completion progress")


class PreviewReq(BaseModel):
    """Request model for narrative preview."""
    text: str = Field(description="Text to preview")
    alpha: float = Field(default=0.2, ge=0.0, le=1.0, description="Blending parameter")


class QueryReq(BaseModel):
    """Request model for matrix queries."""
    query: str = Field(description="Query text")
    rho_id: Optional[str] = Field(default=None, description="Target density matrix ID")


class AttributeExtractionReq(BaseModel):
    """Request model for attribute extraction."""
    text: str = Field(description="Text to analyze")
    rho_id: Optional[str] = Field(default=None, description="Optional density matrix ID")


class AttributeAdjustmentReq(BaseModel):
    """Request model for attribute adjustments."""
    persona_strength: float = Field(default=0.0, ge=-1.0, le=1.0, description="Persona adjustment")
    namespace_strength: float = Field(default=0.0, ge=-1.0, le=1.0, description="Namespace adjustment")
    style_strength: float = Field(default=0.0, ge=-1.0, le=1.0, description="Style adjustment")
    adjusted_rho_id: str = Field(description="ID for the adjusted matrix")


class NarrativeRegenerationReq(BaseModel):
    """Request model for narrative regeneration."""
    original_text: str = Field(description="Original text to transform")
    adjusted_rho_id: str = Field(description="ID of the adjusted density matrix")


class CustomAttributeReq(BaseModel):
    """Request model for custom attribute creation."""
    name: str = Field(description="Name of the custom attribute")
    strength: float = Field(default=1.0, ge=0.0, le=2.0, description="Attribute strength")
    text: str = Field(description="Example text for the attribute")


class SentencePreviewReq(BaseModel):
    """Request model for sentence-level previews."""
    text: str = Field(description="Text to preview")
    attribute_adjustments: Dict[str, float] = Field(
        description="Attribute adjustments (-1.0 to 1.0)"
    )


class BatchAttributeExtractionReq(BaseModel):
    """Request model for batch attribute extraction."""
    texts: List[str] = Field(description="List of texts to analyze")
    attributes_filter: Optional[List[str]] = Field(
        default=None, 
        description="Only extract specific attributes"
    )


class CleanupCriteria(BaseModel):
    """Request model for database cleanup."""
    max_narratives: Optional[int] = Field(default=None, ge=0, description="Maximum narratives to keep")
    min_narratives: Optional[int] = Field(default=None, ge=0, description="Minimum narratives required")


class BatchImportReq(BaseModel):
    """Request model for batch text import."""
    texts: List[str] = Field(description="List of texts to import")
    alpha: float = Field(default=0.2, ge=0.0, le=1.0, description="Blending parameter")
    rho_id: str = Field(description="Target density matrix ID")


class TransformationRequest(BaseModel):
    """Request model for text transformations."""
    text: str = Field(description="Text to transform")
    transformation_name: str = Field(description="Name of the transformation to apply")


class LearnTransformationRequest(BaseModel):
    """Request model for learning new transformations."""
    name: str = Field(description="Name of the new transformation")
    description: str = Field(description="Description of the transformation")
    examples: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Example input/output pairs"
    )


class AttributeSearchRequest(BaseModel):
    """Request model for attribute search."""
    query: str = Field(default="", description="Search query")
    category: Optional[str] = Field(default=None, description="Attribute category filter")
    tags: Optional[List[str]] = Field(default=None, description="Tag filters")


class AttributeFavoriteRequest(BaseModel):
    """Request model for attribute favorites."""
    attribute_name: str = Field(description="Name of the attribute")


class CustomAttributeRequest(BaseModel):
    """Request model for custom attribute creation."""
    name: str = Field(description="Attribute name")
    category: str = Field(description="Attribute category")
    description: Optional[str] = Field(default=None, description="Attribute description")
    examples: Optional[List[str]] = Field(default=None, description="Example texts")


# Response models

class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool = True
    dim: int = Field(description="Hilbert space dimension")
    packs: int = Field(description="Number of POVM packs")
    rhos: int = Field(description="Number of density matrices")


class DiagnosticsResponse(BaseModel):
    """Density matrix diagnostics response."""
    trace: float = Field(description="Matrix trace")
    purity: float = Field(description="Quantum purity Tr(ρ²)")
    entropy: float = Field(description="Von Neumann entropy")
    eigenvals: List[float] = Field(description="Top eigenvalues")
    effective_rank: int = Field(description="Number of significant eigenvalues")
    condition_number: float = Field(description="Condition number")


class MeasurementResponse(BaseModel):
    """POVM measurement response."""
    measurements: Dict[str, float] = Field(description="Measurement probabilities")
    diagnostics: DiagnosticsResponse = Field(description="Matrix diagnostics")


class MatrixResponse(BaseModel):
    """General matrix response."""
    rho_id: str = Field(description="Matrix identifier")
    matrix: List[List[float]] = Field(description="Matrix entries as nested list")
    diagnostics: DiagnosticsResponse = Field(description="Matrix diagnostics")
    label: Optional[str] = Field(default=None, description="Matrix label")


class PackInfo(BaseModel):
    """POVM pack information."""
    pack_id: str = Field(description="Pack identifier")
    description: str = Field(description="Pack description")
    num_axes: int = Field(description="Number of measurement axes")
    type: str = Field(description="Pack type")

class PackListResponse(BaseModel):
    """POVM pack list response."""
    packs: List[PackInfo] = Field(description="Available pack information")


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embedding: List[float] = Field(description="Embedding vector")
    dimension: int = Field(description="Embedding dimension")


class ProjectionResponse(BaseModel):
    """Projection response."""
    local_vector: List[float] = Field(description="64D local vector")
    norm: float = Field(description="Vector norm")


class ExplanationResponse(BaseModel):
    """Operation explanation response."""
    explanations: List[Dict[str, Any]] = Field(description="List of operation explanations")
    rho_id: str = Field(description="Matrix identifier")