"""
Advanced quantum operations for narrative manipulation.

This module implements the sophisticated quantum operations from the mathematical
framework: unitary steering, max-entropy projection, quantum channels, and
commutant flows for precise narrative control.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable
from scipy.optimize import minimize, line_search
from scipy.linalg import expm

from .quantum_state import psd_project, diagnostics

logger = logging.getLogger(__name__)

DIM = 64
TOLERANCE = 1e-8


def unitary_steering(rho: np.ndarray,
                   target_attributes: Dict[str, float],
                   attribute_operators: Dict[str, np.ndarray],
                   max_iterations: int = 20,
                   step_size: float = 0.1) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform unitary steering to adjust attribute expectations while preserving essence.
    
    Implements the framework's gradient-based approach:
    U(η) = exp(-iηG) where G = Σ γₖ Aₖ
    
    Args:
        rho: Current density matrix
        target_attributes: Dict of {attribute_name: target_value}
        attribute_operators: Dict of {attribute_name: hermitian_operator}
        max_iterations: Maximum optimization iterations
        step_size: Initial step size for line search
        
    Returns:
        Tuple of (steered_rho, final_attributes)
    """
    if not target_attributes or not attribute_operators:
        return rho, {}
    
    # Validate operators are Hermitian
    for name, A in attribute_operators.items():
        if not np.allclose(A, A.T.conj(), atol=TOLERANCE):
            logger.warning(f"Operator {name} is not Hermitian, symmetrizing")
            attribute_operators[name] = 0.5 * (A + A.T.conj())
    
    current_rho = rho.copy()
    iteration_history = []
    
    for iteration in range(max_iterations):
        # Compute current attribute values
        current_attrs = {}
        for name, A in attribute_operators.items():
            current_attrs[name] = float(np.real(np.trace(current_rho @ A)))
        
        # Compute gradients: ∂/∂η Tr(U ρ U† A) = i Tr(ρ [G, A])
        gradient_coeffs = {}
        total_error = 0.0
        
        for name, target in target_attributes.items():
            if name not in attribute_operators:
                continue
                
            A = attribute_operators[name]
            current_val = current_attrs[name]
            error = target - current_val
            total_error += error**2
            
            # Gradient coefficient for this attribute
            gradient_coeffs[name] = error
        
        # Check convergence
        rmse = np.sqrt(total_error / len(target_attributes))
        iteration_history.append(rmse)
        
        if rmse < TOLERANCE:
            logger.info(f"Unitary steering converged after {iteration} iterations")
            break
        
        # Construct generator G = Σ γₖ Aₖ
        G = np.zeros((DIM, DIM), dtype=complex)
        for name, coeff in gradient_coeffs.items():
            A = attribute_operators[name]
            G += coeff * A
        
        # Ensure G is anti-Hermitian for unitary evolution
        G = 0.5 * (G - G.T.conj())
        
        # Line search for optimal step size
        def objective(eta):
            U = expm(-1j * eta * G)
            new_rho = U @ current_rho @ U.T.conj()
            new_rho = psd_project(new_rho.real)  # Ensure real and PSD
            
            error = 0.0
            for name, target in target_attributes.items():
                if name in attribute_operators:
                    A = attribute_operators[name]
                    current_val = float(np.real(np.trace(new_rho @ A)))
                    error += (target - current_val)**2
            return error
        
        # Find optimal step size
        try:
            result = minimize(objective, step_size, method='Brent',
                            options={'xtol': TOLERANCE})
            optimal_eta = result.x
        except:
            optimal_eta = step_size * 0.5
        
        # Apply unitary evolution
        U = expm(-1j * optimal_eta * G)
        current_rho = U @ current_rho @ U.T.conj()
        current_rho = psd_project(current_rho.real)
        
        # Adaptive step size
        if iteration > 0 and iteration_history[-1] >= iteration_history[-2]:
            step_size *= 0.7  # Reduce step size if not improving
        else:
            step_size *= 1.1  # Increase step size if improving
        
        step_size = np.clip(step_size, 0.01, 1.0)
    
    # Compute final attributes
    final_attrs = {}
    for name, A in attribute_operators.items():
        final_attrs[name] = float(np.real(np.trace(current_rho @ A)))
    
    logger.info(f"Unitary steering completed: RMSE {rmse:.6f}")
    return current_rho, final_attrs


def max_entropy_projection(rho: np.ndarray,
                         constraints: List[Tuple[np.ndarray, float]],
                         max_iterations: int = 50,
                         learning_rate: float = 0.1) -> np.ndarray:
    """
    Project density matrix to maximum entropy subject to constraints.
    
    Solves: max S(σ) = -Tr(σ log σ) subject to Tr(σ Aₖ) = tₖ
    Solution: σ ∝ exp(Σ λₖ Aₖ + log ρ) (with ρ-prior)
    
    Args:
        rho: Prior density matrix
        constraints: List of (operator, target_value) tuples
        max_iterations: Maximum iterations for optimization
        learning_rate: Learning rate for Lagrange multipliers
        
    Returns:
        Maximum entropy density matrix satisfying constraints
    """
    if not constraints:
        return rho
    
    n_constraints = len(constraints)
    lambdas = np.zeros(n_constraints)  # Lagrange multipliers
    
    def compute_sigma(lambdas_current):
        """Compute σ for given Lagrange multipliers."""
        # Build exponent: Σ λₖ Aₖ + log ρ
        log_rho = _matrix_log(rho)
        exponent = log_rho.copy()
        
        for i, (A, _) in enumerate(constraints):
            exponent += lambdas_current[i] * A
        
        # Compute σ = exp(exponent)
        sigma = _matrix_exp(exponent)
        
        # Ensure PSD and normalized
        sigma = psd_project(sigma)
        trace = np.trace(sigma)
        if trace > TOLERANCE:
            sigma = sigma / trace
        
        return sigma
    
    # Newton-Raphson optimization for Lagrange multipliers
    for iteration in range(max_iterations):
        sigma = compute_sigma(lambdas)
        
        # Compute constraint violations
        violations = []
        for A, target in constraints:
            current_val = float(np.real(np.trace(sigma @ A)))
            violations.append(current_val - target)
        
        violations = np.array(violations)
        
        # Check convergence
        if np.linalg.norm(violations) < TOLERANCE:
            logger.info(f"Max-entropy projection converged after {iteration} iterations")
            break
        
        # Compute gradient and Hessian for Newton step
        gradients = []
        hessian = np.zeros((n_constraints, n_constraints))
        
        for i, (A_i, _) in enumerate(constraints):
            # Gradient: ∂/∂λᵢ Tr(σ Aⱼ) = Tr(σ [Aᵢ, Aⱼ])
            grad_i = []
            for j, (A_j, _) in enumerate(constraints):
                if i == j:
                    # Diagonal: variance
                    mean_val = np.real(np.trace(sigma @ A_i))
                    variance = np.real(np.trace(sigma @ A_i @ A_i)) - mean_val**2
                    grad_i.append(variance)
                else:
                    # Off-diagonal: covariance
                    mean_i = np.real(np.trace(sigma @ A_i))
                    mean_j = np.real(np.trace(sigma @ A_j))
                    covariance = np.real(np.trace(sigma @ A_i @ A_j)) - mean_i * mean_j
                    grad_i.append(covariance)
                    
                hessian[i, j] = grad_i[j]
        
        # Newton step with regularization
        try:
            # Add small regularization for numerical stability
            hessian_reg = hessian + TOLERANCE * np.eye(n_constraints)
            delta_lambdas = np.linalg.solve(hessian_reg, -violations)
        except:
            # Fallback to gradient descent
            delta_lambdas = -learning_rate * violations
        
        # Update with line search
        alpha = 1.0
        for _ in range(5):  # Simple backtracking
            new_lambdas = lambdas + alpha * delta_lambdas
            try:
                new_sigma = compute_sigma(new_lambdas)
                new_violations = []
                for A, target in constraints:
                    current_val = float(np.real(np.trace(new_sigma @ A)))
                    new_violations.append(current_val - target)
                
                if np.linalg.norm(new_violations) < np.linalg.norm(violations):
                    lambdas = new_lambdas
                    break
                alpha *= 0.5
            except:
                alpha *= 0.5
        else:
            # If line search fails, use small gradient step
            lambdas += 0.01 * delta_lambdas
    
    final_sigma = compute_sigma(lambdas)
    logger.info(f"Max-entropy projection: final constraint violation {np.linalg.norm(violations):.6f}")
    
    return final_sigma


def _matrix_log(A: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm with numerical stability."""
    try:
        eigenvals, eigenvecs = np.linalg.eigh(A)
        eigenvals = np.maximum(eigenvals, TOLERANCE)  # Avoid log(0)
        log_eigenvals = np.log(eigenvals)
        return eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T
    except:
        # Fallback: use scipy if available
        from scipy.linalg import logm
        return logm(A).real


def _matrix_exp(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential."""
    try:
        return expm(A).real
    except:
        # Fallback: eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(A)
        exp_eigenvals = np.exp(eigenvals)
        return eigenvecs @ np.diag(exp_eigenvals) @ eigenvecs.T


def depolarizing_channel(rho: np.ndarray, 
                        p: float,
                        preserve_attributes: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Apply depolarizing channel: Φ(ρ) = (1-p)ρ + p*I/d
    
    Args:
        rho: Input density matrix
        p: Depolarization parameter (0 = no change, 1 = maximally mixed)
        preserve_attributes: Optional list of operators to preserve
        
    Returns:
        Depolarized density matrix
    """
    p = np.clip(p, 0.0, 1.0)
    I = np.eye(DIM) / DIM
    
    if preserve_attributes:
        # Commutant depolarization: preserve specified attributes
        result = rho.copy()
        
        for A in preserve_attributes:
            # Project to commutant subspace
            commutant_part = _project_to_commutant(rho, A)
            non_commutant_part = rho - commutant_part
            
            # Depolarize only the non-commutant part
            depolarized_part = (1 - p) * non_commutant_part + p * _project_to_commutant(I, A)
            result = commutant_part + depolarized_part
    else:
        # Standard depolarization
        result = (1 - p) * rho + p * I
    
    return psd_project(result)


def dephasing_channel(rho: np.ndarray,
                     basis_operators: List[np.ndarray],
                     strength: float = 1.0) -> np.ndarray:
    """
    Apply dephasing in a specific basis to commit to attribute values.
    
    Φ(ρ) = Σᵢ Pᵢ ρ Pᵢ where {Pᵢ} are projectors onto the basis
    
    Args:
        rho: Input density matrix
        basis_operators: List of operators defining the dephasing basis
        strength: Dephasing strength (1.0 = complete dephasing)
        
    Returns:
        Dephased density matrix
    """
    if not basis_operators:
        return rho
    
    strength = np.clip(strength, 0.0, 1.0)
    
    # Find common eigenbasis of the operators
    try:
        # For simplicity, use the first operator's eigenbasis
        eigenvals, eigenvecs = np.linalg.eigh(basis_operators[0])
        
        # Create projectors
        projectors = []
        for i in range(DIM):
            P_i = np.outer(eigenvecs[:, i], eigenvecs[:, i].conj())
            projectors.append(P_i)
        
        # Apply dephasing: Σᵢ Pᵢ ρ Pᵢ
        dephased = np.zeros_like(rho)
        for P_i in projectors:
            dephased += P_i @ rho @ P_i
        
        # Blend with original
        result = (1 - strength) * rho + strength * dephased
        
    except:
        logger.warning("Dephasing failed, returning original matrix")
        result = rho
    
    return psd_project(result)


def _project_to_commutant(rho: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project density matrix to the commutant of operator A."""
    # Find joint eigenbasis of ρ and A
    try:
        # Simplified: project to A's eigenspaces
        eigenvals, eigenvecs = np.linalg.eigh(A)
        
        # Group eigenvectors by eigenvalue
        eigenval_groups = {}
        for i, val in enumerate(eigenvals):
            val_key = round(val, 10)  # Numerical tolerance
            if val_key not in eigenval_groups:
                eigenval_groups[val_key] = []
            eigenval_groups[val_key].append(i)
        
        # Project ρ within each eigenspace
        projected = np.zeros_like(rho)
        
        for indices in eigenval_groups.values():
            if len(indices) == 1:
                # Single eigenvalue - preserve component
                i = indices[0]
                P_i = np.outer(eigenvecs[:, i], eigenvecs[:, i].conj())
                projected += P_i @ rho @ P_i
            else:
                # Degenerate eigenspace - project block
                subspace_vecs = eigenvecs[:, indices]
                P_subspace = subspace_vecs @ subspace_vecs.T.conj()
                projected += P_subspace @ rho @ P_subspace
        
        return projected.real
        
    except:
        # Fallback: return original
        return rho


def commutant_flow(rho: np.ndarray,
                  preserved_operators: List[np.ndarray],
                  target_generator: np.ndarray,
                  evolution_time: float = 0.1) -> np.ndarray:
    """
    Apply commutant flow: change essence while preserving specified attributes.
    
    Find G such that [G, Aₖ] = 0 for all preserved operators, then apply U = exp(-iG).
    
    Args:
        rho: Current density matrix
        preserved_operators: List of operators to preserve
        target_generator: Desired generator (will be projected to commutant)
        evolution_time: Evolution parameter
        
    Returns:
        Evolved density matrix with preserved attributes
    """
    if not preserved_operators:
        # No constraints - apply target generator directly
        U = expm(-1j * evolution_time * target_generator)
        return psd_project((U @ rho @ U.T.conj()).real)
    
    # Project target generator to commutant subspace
    G = target_generator.copy()
    
    for A in preserved_operators:
        # Remove non-commuting part: G → G - [G,A]†A / ||A||²
        commutator = G @ A - A @ G
        A_norm_sq = np.trace(A.T.conj() @ A)
        
        if A_norm_sq > TOLERANCE:
            correction = (commutator.T.conj() @ A) / A_norm_sq
            G = G - correction
    
    # Ensure G is anti-Hermitian
    G = 0.5 * (G - G.T.conj())
    
    # Apply evolution
    U = expm(-1j * evolution_time * G)
    evolved_rho = U @ rho @ U.T.conj()
    
    return psd_project(evolved_rho.real)


def style_channel(rho: np.ndarray,
                 style_name: str,
                 strength: float = 1.0) -> np.ndarray:
    """
    Apply a learned style transformation channel.
    
    This is a placeholder for style-specific Kraus operators.
    In practice, these would be learned from style examples.
    
    Args:
        rho: Input density matrix
        style_name: Name of the style to apply
        strength: Transformation strength
        
    Returns:
        Style-transformed density matrix
    """
    # Placeholder implementation - would use learned Kraus operators
    style_transformations = {
        "noir": _noir_style_channel,
        "romantic": _romantic_style_channel,
        "minimalist": _minimalist_style_channel
    }
    
    if style_name in style_transformations:
        return style_transformations[style_name](rho, strength)
    else:
        logger.warning(f"Unknown style: {style_name}")
        return rho


def _noir_style_channel(rho: np.ndarray, strength: float) -> np.ndarray:
    """Apply noir style transformation."""
    # Increase contrast, reduce brightness
    contrast_op = np.random.randn(DIM, DIM)
    contrast_op = 0.5 * (contrast_op + contrast_op.T)
    
    # Simple channel: mix with style operator
    style_rho = psd_project(contrast_op)
    return (1 - strength) * rho + strength * style_rho


def _romantic_style_channel(rho: np.ndarray, strength: float) -> np.ndarray:
    """Apply romantic style transformation."""
    # Increase emotional components
    romantic_basis = np.eye(DIM)
    romantic_basis[:5, :5] *= 1.5  # Boost emotional dimensions
    
    transformed = romantic_basis @ rho @ romantic_basis.T
    return psd_project((1 - strength) * rho + strength * transformed)


def _minimalist_style_channel(rho: np.ndarray, strength: float) -> np.ndarray:
    """Apply minimalist style transformation."""
    # Reduce complexity, focus on main components
    eigenvals, eigenvecs = np.linalg.eigh(rho)
    
    # Keep only top components
    n_keep = max(1, DIM // 4)
    simplified_eigenvals = eigenvals.copy()
    simplified_eigenvals[:-n_keep] *= (1 - strength)
    
    simplified_rho = eigenvecs @ np.diag(simplified_eigenvals) @ eigenvecs.T
    return psd_project(simplified_rho)