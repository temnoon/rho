"""
Core quantum state operations for density matrices.

This module provides the fundamental operations for managing quantum density matrices (ρ),
including state creation, validation, and basic measurements.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

DIM = 64  # Default Hilbert space dimension


def psd_project(A: np.ndarray) -> np.ndarray:
    """
    Make symmetric, clip negative eigenvalues, reconstruct, normalize trace to 1.
    
    Args:
        A: Matrix to project to positive semidefinite
        
    Returns:
        Positive semidefinite matrix with trace = 1
    """
    # Ensure Hermitian symmetry
    A_sym = 0.5 * (A + A.T)
    
    # Eigendecomposition
    w, V = np.linalg.eigh(A_sym)
    
    # Clip negative eigenvalues
    w_clipped = np.maximum(w, 1e-12)
    
    # Reconstruct matrix
    result = V @ np.diag(w_clipped) @ V.T
    
    # Normalize trace to 1
    trace = np.trace(result)
    if trace > 1e-12:
        result = result / trace
    else:
        # Fallback to maximally mixed state
        result = np.eye(A.shape[0]) / A.shape[0]
    
    return result


def diagnostics(rho: np.ndarray, top_k: int = 8) -> Dict[str, Any]:
    """
    Return comprehensive diagnostics for a density matrix.
    
    Args:
        rho: Density matrix to analyze
        top_k: Number of top eigenvalues to return
        
    Returns:
        Dictionary with trace, purity, entropy, and eigenvalues
    """
    rho_sym = 0.5 * (rho + rho.T)
    
    # Ensure numeric stability
    w, _ = np.linalg.eigh(rho_sym)
    w = np.maximum(w, 1e-15)  # Prevent log(0) in entropy
    
    trace = float(np.trace(rho_sym))
    purity = float(np.sum(w**2))  # Tr(ρ²)
    
    # Von Neumann entropy: S = -Tr(ρ log ρ)
    entropy = -float(np.sum(w * np.log(w + 1e-15)))
    
    # Sort eigenvalues in descending order
    w_sorted = np.sort(w)[::-1]
    
    return {
        "trace": trace,
        "purity": purity,
        "entropy": entropy,
        "eigenvals": w_sorted[:top_k].tolist(),
        "effective_rank": int(np.sum(w > 1e-6)),
        "condition_number": float(w_sorted[0] / (w_sorted[-1] + 1e-15))
    }


def create_pure_state(v: np.ndarray) -> np.ndarray:
    """
    Create a pure state density matrix from a state vector.
    
    Args:
        v: State vector (will be normalized)
        
    Returns:
        Pure state density matrix |v⟩⟨v|
    """
    # Normalize the vector
    v_norm = v / (np.linalg.norm(v) + 1e-15)
    
    # Create pure state ρ = |v⟩⟨v|
    rho = np.outer(v_norm, v_norm.conj())
    
    return rho.real  # Ensure real for numerical stability


def create_maximally_mixed_state(dim: int = DIM) -> np.ndarray:
    """
    Create a maximally mixed state I/d.
    
    Args:
        dim: Dimension of the Hilbert space
        
    Returns:
        Maximally mixed density matrix
    """
    return np.eye(dim) / dim


def blend_states(rho1: np.ndarray, rho2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two density matrices with exponential moving average.
    
    DEPRECATED: Use text_channels.text_to_channel() for proper CPTP evolution.
    This function is kept for backward compatibility but should be replaced
    with proper quantum channels for text processing.
    
    Args:
        rho1: Current density matrix
        rho2: New density matrix to blend in
        alpha: Blending parameter (0 = keep rho1, 1 = use rho2)
        
    Returns:
        Blended density matrix
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    blended = (1 - alpha) * rho1 + alpha * rho2
    return psd_project(blended)


def apply_text_channel(rho: np.ndarray, 
                      text_embedding: np.ndarray, 
                      alpha: float,
                      channel_type: str = "rank_one_update") -> np.ndarray:
    """
    Apply text as a proper quantum channel (CPTP map).
    
    This replaces simple convex combinations with proper channel evolution:
    ρ' = Φ_text(ρ) where Φ is completely positive and trace-preserving.
    
    Args:
        rho: Current density matrix
        text_embedding: Embedded text vector (should be DIM-dimensional)
        alpha: Blending strength for the channel
        channel_type: Type of channel ("rank_one_update", "coherent_rotation", "dephasing_mixture")
        
    Returns:
        Updated density matrix via proper channel evolution
    """
    try:
        from .text_channels import text_to_channel, audit_channel_properties
        
        # Create proper quantum channel from text
        text_channel = text_to_channel(text_embedding, alpha, channel_type)
        
        # Apply channel: ρ' = Φ(ρ)
        rho_new = text_channel.apply(rho)
        
        # Optional: audit channel properties (can be disabled in production)
        if __debug__:  # Only in debug mode
            audit = audit_channel_properties(text_channel, rho)
            if not audit["passes_audit"]:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Channel audit failed: {audit}")
        
        return rho_new
        
    except ImportError:
        # Fallback to old method if text_channels not available
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("text_channels module not available, using deprecated blend_states")
        
        # Convert embedding to pure state for fallback
        psi = text_embedding / (np.linalg.norm(text_embedding) + 1e-12)
        pure_state = np.outer(psi, psi)
        return blend_states(rho, pure_state, alpha)


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute trace distance between two density matrices.
    
    Args:
        rho1, rho2: Density matrices
        
    Returns:
        Trace distance (0 = identical, 1 = orthogonal)
    """
    diff = rho1 - rho2
    eigenvals = np.linalg.eigvals(diff)
    return 0.5 * np.sum(np.abs(eigenvals))


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute quantum fidelity between two density matrices.
    
    Args:
        rho1, rho2: Density matrices
        
    Returns:
        Fidelity (1 = identical, 0 = orthogonal)
    """
    sqrt_rho1 = np.linalg.matrix_power(rho1, 0.5)
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    eigenvals = np.linalg.eigvals(product)
    eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
    return float(np.sum(np.sqrt(eigenvals))**2)


def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is a valid density matrix.
    
    Args:
        rho: Matrix to check
        tol: Numerical tolerance
        
    Returns:
        True if valid density matrix
    """
    # Check Hermiticity
    if not np.allclose(rho, rho.T.conj(), atol=tol):
        return False
    
    # Check positive semidefinite
    eigenvals = np.linalg.eigvals(rho)
    if np.any(eigenvals < -tol):
        return False
    
    # Check trace = 1
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=tol):
        return False
    
    return True


def rho_matrix_to_list(rho: np.ndarray) -> list:
    """Convert numpy array to nested list for JSON serialization."""
    return rho.tolist()