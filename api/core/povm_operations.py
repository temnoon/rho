"""
POVM (Positive Operator-Valued Measure) operations for quantum measurements.

This module implements POVM measurements for extracting interpretable attributes
from density matrices, following the mathematical framework for quantum measurements.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DIM = 64  # Default Hilbert space dimension


def create_binary_povm(basis_vector: np.ndarray, lambda_param: float = 0.8, 
                      epsilon: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a binary POVM for measuring an attribute.
    
    Following the framework: E_+ = λ * Π + ε * I/d, E_- = I - E_+
    where Π is a projector onto the attribute subspace.
    
    Args:
        basis_vector: Vector defining the attribute direction
        lambda_param: Strength parameter (0 < λ < 1)
        epsilon: Coverage parameter for numerical stability
        
    Returns:
        Tuple of (E_positive, E_negative) effect operators
    """
    # Normalize basis vector
    v = basis_vector / (np.linalg.norm(basis_vector) + 1e-15)
    
    # Create rank-1 projector Π = |v⟩⟨v|
    Pi = np.outer(v, v)
    
    # Identity matrix
    I = np.eye(DIM)
    
    # Positive effect: E_+ = λ * Π + ε * I/d
    E_positive = lambda_param * Pi + epsilon * I / DIM
    
    # Negative effect: E_- = I - E_+
    E_negative = I - E_positive
    
    # Ensure positive semidefinite (numerical stability)
    E_positive = project_to_psd(E_positive)
    E_negative = project_to_psd(E_negative)
    
    return E_positive, E_negative


def create_multiclass_povm(basis_vectors: List[np.ndarray], 
                          alpha_params: Optional[List[float]] = None) -> List[np.ndarray]:
    """
    Create a multi-class POVM for measuring categorical attributes.
    
    Args:
        basis_vectors: List of vectors defining each class
        alpha_params: Optional weights for each class
        
    Returns:
        List of effect operators that sum to identity
    """
    n_classes = len(basis_vectors)
    if alpha_params is None:
        alpha_params = [1.0] * n_classes
    
    # Normalize basis vectors and create projectors
    projectors = []
    for v in basis_vectors:
        v_norm = v / (np.linalg.norm(v) + 1e-15)
        Pi = np.outer(v_norm, v_norm)
        projectors.append(Pi)
    
    # Create unnormalized effects
    effects = []
    for i, (Pi, alpha) in enumerate(zip(projectors, alpha_params)):
        E_i = alpha * Pi
        effects.append(E_i)
    
    # Normalize to ensure sum = I (using least-squares with PSD constraint)
    effects = normalize_povm_effects(effects)
    
    return effects


def normalize_povm_effects(effects: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize a list of effects to ensure they sum to identity while maintaining PSD.
    
    Args:
        effects: List of effect operators
        
    Returns:
        Normalized effects that sum to identity
    """
    # Compute current sum
    current_sum = sum(effects)
    
    # Simple scaling approach (more sophisticated methods available)
    trace_sum = np.trace(current_sum)
    if trace_sum > 1e-10:
        scale = DIM / trace_sum
        effects = [scale * E for E in effects]
    
    # Project each effect to PSD
    effects = [project_to_psd(E) for E in effects]
    
    # Final adjustment to ensure exact sum = I
    actual_sum = sum(effects)
    residual = np.eye(DIM) - actual_sum
    
    # Distribute residual equally (simple approach)
    correction = residual / len(effects)
    effects = [E + correction for E in effects]
    
    # Final PSD projection
    effects = [project_to_psd(E) for E in effects]
    
    return effects


def project_to_psd(A: np.ndarray) -> np.ndarray:
    """
    Project matrix to positive semidefinite while preserving trace.
    
    Args:
        A: Matrix to project
        
    Returns:
        Positive semidefinite matrix
    """
    # Ensure Hermitian
    A_sym = 0.5 * (A + A.T)
    
    # Eigendecomposition
    w, V = np.linalg.eigh(A_sym)
    
    # Clip negative eigenvalues
    w_pos = np.maximum(w, 1e-12)
    
    # Reconstruct
    return V @ np.diag(w_pos) @ V.T


def measure_povm(rho: np.ndarray, effects: List[np.ndarray], 
                labels: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Perform POVM measurement on a density matrix.
    
    Args:
        rho: Density matrix to measure
        effects: List of POVM effect operators
        labels: Optional labels for each effect
        
    Returns:
        Dictionary of measurement probabilities
    """
    if labels is None:
        labels = [f"outcome_{i}" for i in range(len(effects))]
    
    probabilities = {}
    for i, (effect, label) in enumerate(zip(effects, labels)):
        # Probability = Tr(E_i * ρ)
        prob = float(np.trace(effect @ rho))
        prob = max(0.0, min(1.0, prob))  # Clamp to [0,1]
        probabilities[label] = prob
    
    return probabilities


def create_attribute_povm(positive_examples: List[np.ndarray], 
                         negative_examples: List[np.ndarray],
                         rank: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn a POVM for an attribute from positive/negative examples.
    
    Args:
        positive_examples: List of vectors representing positive examples
        negative_examples: List of vectors representing negative examples
        rank: Rank of the projector (number of principal components)
        
    Returns:
        Tuple of (E_positive, E_negative) learned from examples
    """
    if not positive_examples:
        # Fallback to random basis vector
        v = np.random.randn(DIM)
        return create_binary_povm(v)
    
    # Stack examples
    pos_matrix = np.array(positive_examples).T  # Shape: (DIM, n_pos)
    
    # Compute principal subspace via SVD
    if pos_matrix.shape[1] == 1:
        # Single example - use as-is
        v = pos_matrix[:, 0]
        v = v / (np.linalg.norm(v) + 1e-15)
        Pi = np.outer(v, v)
    else:
        # Multiple examples - find principal subspace
        U, s, Vt = np.linalg.svd(pos_matrix, full_matrices=False)
        
        # Take top 'rank' components
        U_reduced = U[:, :min(rank, U.shape[1])]
        Pi = U_reduced @ U_reduced.T
    
    # Create POVM effects
    I = np.eye(DIM)
    lambda_param = 0.8
    epsilon = 0.01
    
    E_positive = lambda_param * Pi + epsilon * I / DIM
    E_negative = I - E_positive
    
    # Ensure PSD
    E_positive = project_to_psd(E_positive)
    E_negative = project_to_psd(E_negative)
    
    return E_positive, E_negative


def create_coverage_povm(n_effects: int = 16) -> List[np.ndarray]:
    """
    Create a coverage POVM for general space exploration.
    
    Uses random orthogonal vectors to create a informationally diverse set
    of measurements that probe different aspects of the quantum state.
    
    Args:
        n_effects: Number of POVM effects to create
        
    Returns:
        List of POVM effects providing good coverage
    """
    # Generate random orthogonal directions
    directions = []
    for _ in range(n_effects):
        v = np.random.randn(DIM)
        v = v / np.linalg.norm(v)
        directions.append(v)
    
    # Create rank-1 projectors
    projectors = [np.outer(v, v) for v in directions]
    
    # Create uniform POVM
    effects = [(I / n_effects) for I in projectors]
    
    # Normalize to sum to identity
    effects = normalize_povm_effects(effects)
    
    return effects


def povm_fisher_information(rho: np.ndarray, effects: List[np.ndarray]) -> np.ndarray:
    """
    Compute Fisher information matrix for a POVM on a quantum state.
    
    This quantifies how much information the POVM extracts about
    small perturbations to the quantum state.
    
    Args:
        rho: Quantum state
        effects: POVM effects
        
    Returns:
        Fisher information matrix
    """
    n_effects = len(effects)
    fisher_matrix = np.zeros((n_effects, n_effects))
    
    # Compute probabilities
    probs = [max(np.trace(E @ rho), 1e-15) for E in effects]
    
    # Fisher information matrix elements
    for i in range(n_effects):
        for j in range(n_effects):
            if i == j:
                # Diagonal terms
                fisher_matrix[i, j] = np.trace(effects[i] @ effects[i]) / probs[i]
            else:
                # Off-diagonal terms
                fisher_matrix[i, j] = np.trace(effects[i] @ effects[j]) / np.sqrt(probs[i] * probs[j])
    
    return fisher_matrix


def optimize_povm_for_discrimination(states: List[np.ndarray], 
                                   n_effects: int = 8) -> List[np.ndarray]:
    """
    Optimize a POVM for discriminating between a set of quantum states.
    
    Args:
        states: List of density matrices to discriminate
        n_effects: Number of POVM effects
        
    Returns:
        Optimized POVM effects
    """
    # Simple heuristic: use the principal differences between states
    if len(states) < 2:
        return create_coverage_povm(n_effects)
    
    # Compute pairwise differences
    differences = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            diff = states[i] - states[j]
            differences.append(diff)
    
    # Stack differences and compute SVD
    if differences:
        diff_stack = np.array([diff.flatten() for diff in differences])
        U, s, Vt = np.linalg.svd(diff_stack, full_matrices=False)
        
        # Use top components as measurement directions
        n_components = min(n_effects, len(s))
        effects = []
        
        for i in range(n_components):
            direction = Vt[i].reshape((DIM, DIM))
            direction = 0.5 * (direction + direction.T)  # Symmetrize
            direction = project_to_psd(direction)
            effects.append(direction)
        
        # Normalize
        effects = normalize_povm_effects(effects)
        
        return effects
    
    # Fallback
    return create_coverage_povm(n_effects)


def create_dialectical_povm(pack_name: str, 
                          left_pole: str, 
                          right_pole: str, 
                          description: str = "", 
                          n_measurements: int = 8) -> Dict[str, Any]:
    """
    Create a POVM pack based on dialectical concepts.
    
    Args:
        pack_name: Name for the POVM pack
        left_pole: Left side of the dialectic (e.g., "Order")
        right_pole: Right side of the dialectic (e.g., "Chaos")
        description: Optional description
        n_measurements: Number of measurement effects
        
    Returns:
        POVM pack data structure
    """
    # Create binary opposition POVM
    effects = create_binary_povm(left_pole, right_pole, n_measurements)
    
    # Create measurement axes
    axes = []
    
    # Primary dialectical axis
    primary_axis = {
        "id": f"{pack_name}_primary",
        "label": f"{left_pole} ⟷ {right_pole}",
        "description": description or f"Dialectical measurement between {left_pole} and {right_pole}",
        "type": "binary",
        "effects": [effects[0].tolist(), effects[1].tolist()],
        "categories": [left_pole, right_pole],
        "dialectical_poles": {
            "left": left_pole,
            "right": right_pole
        }
    }
    axes.append(primary_axis)
    
    # Add nuanced measurements if we have more effects
    if len(effects) > 2:
        # Create gradient measurements
        for i in range(2, min(len(effects), n_measurements)):
            weight = (i - 1) / (n_measurements - 2) if n_measurements > 2 else 0.5
            axis_id = f"{pack_name}_gradient_{i-1}"
            
            axes.append({
                "id": axis_id,
                "label": f"{left_pole}-{right_pole} Gradient {i-1}",
                "description": f"Nuanced measurement along {left_pole}-{right_pole} spectrum",
                "type": "binary",
                "effects": [effects[0].tolist(), effects[i].tolist()],
                "categories": [f"Weak {left_pole}", f"Strong {right_pole}"],
                "gradient_weight": weight
            })
    
    # Create the pack structure
    pack_data = {
        "pack_id": pack_name,
        "name": pack_name,
        "description": description or f"Dialectical POVM measuring {left_pole} ⟷ {right_pole}",
        "type": "dialectical",
        "dialectical_concept": f"{left_pole} ⟷ {right_pole}",
        "axes": axes,
        "metadata": {
            "created_by": "dialectical_generator",
            "left_pole": left_pole,
            "right_pole": right_pole,
            "n_measurements": len(axes),
            "total_effects": len(effects)
        }
    }
    
    return pack_data