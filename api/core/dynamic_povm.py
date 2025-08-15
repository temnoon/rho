"""
Dynamic POVM generation based on narrative content.

This module implements the key insight that POVMs should be generated AFTER
narrative ingestion to optimize the measurement basis for the specific content,
following the mathematical framework for efficient space utilization.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .quantum_state import diagnostics
from .povm_operations import (
    create_binary_povm, 
    create_multiclass_povm,
    normalize_povm_effects,
    povm_fisher_information,
    project_to_psd
)
from .embedding import embed, project_to_local

logger = logging.getLogger(__name__)

DIM = 64


@dataclass
class NarrativeAnalysis:
    """Analysis of narrative content for POVM optimization."""
    text_samples: List[str]
    embedding_vectors: List[np.ndarray]
    local_vectors: List[np.ndarray]
    principal_components: np.ndarray
    variance_explained: np.ndarray
    semantic_clusters: Dict[str, List[int]]
    attribute_directions: Dict[str, np.ndarray]


def analyze_narrative_space(texts: List[str], 
                          min_variance_threshold: float = 0.95) -> NarrativeAnalysis:
    """
    Analyze the semantic space spanned by narrative texts.
    
    This implements the framework's recommendation to analyze the narrative
    content first, then design POVMs that efficiently probe the occupied subspace.
    
    Args:
        texts: List of text samples from the narrative
        min_variance_threshold: Minimum cumulative variance to retain
        
    Returns:
        NarrativeAnalysis with space characterization
    """
    if not texts:
        raise ValueError("No texts provided for analysis")
    
    # Embed all texts
    embeddings = []
    local_vectors = []
    
    for text in texts:
        emb = embed(text)
        local_vec = project_to_local(emb)
        embeddings.append(emb)
        local_vectors.append(local_vec)
    
    # Stack into matrix for analysis
    X = np.array(local_vectors)  # Shape: (n_samples, 64)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Principal Component Analysis
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Determine how many components to keep
    variance_explained = s**2 / np.sum(s**2)
    cumulative_variance = np.cumsum(variance_explained)
    n_components = np.argmax(cumulative_variance >= min_variance_threshold) + 1
    n_components = max(n_components, 3)  # Keep at least 3 components
    
    principal_components = Vt[:n_components]
    
    # Semantic clustering (simple k-means-style)
    semantic_clusters = _cluster_semantic_content(texts, local_vectors)
    
    # Extract attribute directions
    attribute_directions = _extract_attribute_directions(
        texts, local_vectors, principal_components
    )
    
    return NarrativeAnalysis(
        text_samples=texts,
        embedding_vectors=embeddings,
        local_vectors=local_vectors,
        principal_components=principal_components,
        variance_explained=variance_explained[:n_components],
        semantic_clusters=semantic_clusters,
        attribute_directions=attribute_directions
    )


def _cluster_semantic_content(texts: List[str], 
                            vectors: List[np.ndarray],
                            n_clusters: int = 5) -> Dict[str, List[int]]:
    """Simple semantic clustering based on content analysis."""
    clusters = {
        "dialogue": [],
        "description": [], 
        "action": [],
        "emotion": [],
        "other": []
    }
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        # Simple heuristic clustering
        if '"' in text or "'" in text or any(word in text_lower for word in ["said", "asked", "replied"]):
            clusters["dialogue"].append(i)
        elif any(word in text_lower for word in ["felt", "emotion", "happy", "sad", "angry", "love"]):
            clusters["emotion"].append(i)
        elif any(word in text_lower for word in ["ran", "walked", "moved", "jumped", "grabbed"]):
            clusters["action"].append(i)
        elif any(word in text_lower for word in ["looked", "appeared", "seemed", "beautiful", "dark"]):
            clusters["description"].append(i)
        else:
            clusters["other"].append(i)
    
    return clusters


def _extract_attribute_directions(texts: List[str],
                                vectors: List[np.ndarray],
                                principal_components: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract meaningful attribute directions from the narrative space."""
    attributes = {}
    
    # Use principal components as base directions
    for i, pc in enumerate(principal_components[:6]):  # Top 6 components
        attributes[f"narrative_axis_{i}"] = pc
    
    # Extract polarity directions
    positive_indices = []
    negative_indices = []
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        positive_words = ["good", "happy", "bright", "hope", "love", "joy", "beautiful"]
        negative_words = ["bad", "sad", "dark", "fear", "hate", "despair", "terrible"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count and pos_count > 0:
            positive_indices.append(i)
        elif neg_count > pos_count and neg_count > 0:
            negative_indices.append(i)
    
    # Create polarity direction if we have examples
    if positive_indices and negative_indices:
        pos_vectors = np.array([vectors[i] for i in positive_indices])
        neg_vectors = np.array([vectors[i] for i in negative_indices])
        
        pos_mean = np.mean(pos_vectors, axis=0)
        neg_mean = np.mean(neg_vectors, axis=0)
        
        polarity_direction = pos_mean - neg_mean
        polarity_direction = polarity_direction / (np.linalg.norm(polarity_direction) + 1e-15)
        attributes["emotional_polarity"] = polarity_direction
    
    return attributes


def generate_optimal_povm_pack(analysis: NarrativeAnalysis,
                             pack_name: str = "narrative_optimized",
                             n_measurements: int = 12) -> Dict[str, Any]:
    """
    Generate an optimal POVM pack based on narrative analysis.
    
    Following the framework: use Fisher information optimization to select
    measurements that maximally discriminate within the occupied subspace.
    
    Args:
        analysis: Narrative space analysis
        pack_name: Name for the generated pack
        n_measurements: Number of measurements to include
        
    Returns:
        POVM pack dictionary ready for use
    """
    effects = []
    labels = []
    descriptions = []
    
    # 1. Principal component measurements
    for i, pc in enumerate(analysis.principal_components[:6]):
        E_pos, E_neg = create_binary_povm(pc, lambda_param=0.85, epsilon=0.02)
        effects.extend([E_neg, E_pos])
        labels.extend([f"pc{i}_low", f"pc{i}_high"])
        descriptions.extend([
            f"Low projection on narrative principal component {i}",
            f"High projection on narrative principal component {i}"
        ])
    
    # 2. Attribute-specific measurements
    for attr_name, direction in analysis.attribute_directions.items():
        if len(effects) >= n_measurements * 2:
            break
            
        E_pos, E_neg = create_binary_povm(direction, lambda_param=0.8, epsilon=0.02)
        effects.extend([E_neg, E_pos])
        labels.extend([f"{attr_name}_low", f"{attr_name}_high"])
        descriptions.extend([
            f"Low {attr_name.replace('_', ' ')}",
            f"High {attr_name.replace('_', ' ')}"
        ])
    
    # 3. Cluster discrimination measurements
    cluster_directions = _compute_cluster_directions(analysis)
    for cluster_name, direction in cluster_directions.items():
        if len(effects) >= n_measurements * 2:
            break
            
        E_pos, E_neg = create_binary_povm(direction, lambda_param=0.75, epsilon=0.03)
        effects.extend([E_neg, E_pos])
        labels.extend([f"{cluster_name}_low", f"{cluster_name}_high"])
        descriptions.extend([
            f"Low {cluster_name} content",
            f"High {cluster_name} content"
        ])
    
    # Normalize effects to ensure they sum to identity
    effects = normalize_povm_effects(effects)
    
    # Group into axes (pairs of effects)
    axes = []
    for i in range(0, len(effects), 2):
        if i + 1 < len(effects):
            axis = {
                "id": f"axis_{i//2}",
                "labels": [labels[i], labels[i+1]],
                "effects": [effects[i].tolist(), effects[i+1].tolist()],
                "type": "narrative_optimized",
                "description": f"Optimized measurement: {descriptions[i]} vs {descriptions[i+1]}"
            }
            axes.append(axis)
    
    pack_data = {
        "pack_id": pack_name,
        "axes": axes,
        "type": "narrative_optimized",
        "description": f"Dynamically generated POVM pack optimized for narrative content",
        "generation_metadata": {
            "n_text_samples": len(analysis.text_samples),
            "n_principal_components": len(analysis.principal_components),
            "variance_explained": analysis.variance_explained.tolist(),
            "semantic_clusters": {k: len(v) for k, v in analysis.semantic_clusters.items()},
            "n_measurements": len(axes)
        }
    }
    
    return pack_data


def _compute_cluster_directions(analysis: NarrativeAnalysis) -> Dict[str, np.ndarray]:
    """Compute characteristic directions for semantic clusters."""
    directions = {}
    
    for cluster_name, indices in analysis.semantic_clusters.items():
        if len(indices) < 2:
            continue
            
        # Compute cluster centroid
        cluster_vectors = [analysis.local_vectors[i] for i in indices]
        centroid = np.mean(cluster_vectors, axis=0)
        
        # Compare with overall centroid
        overall_centroid = np.mean(analysis.local_vectors, axis=0)
        direction = centroid - overall_centroid
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            directions[cluster_name] = direction / norm
    
    return directions


def optimize_povm_for_discrimination(rho_states: List[np.ndarray],
                                  target_attributes: List[str] = None,
                                  max_effects: int = 16) -> List[np.ndarray]:
    """
    Optimize POVM for discriminating between quantum states.
    
    Implements the framework's Fisher information maximization approach.
    
    Args:
        rho_states: List of density matrices to discriminate
        target_attributes: Optional list of attribute names to focus on
        max_effects: Maximum number of POVM effects
        
    Returns:
        List of optimized POVM effects
    """
    if len(rho_states) < 2:
        logger.warning("Need at least 2 states for discrimination optimization")
        return []
    
    # Compute state differences for discrimination
    differences = []
    for i in range(len(rho_states)):
        for j in range(i + 1, len(rho_states)):
            diff = rho_states[i] - rho_states[j]
            differences.append(diff)
    
    if not differences:
        return []
    
    # Stack differences and perform SVD
    diff_matrices = np.array([diff.flatten() for diff in differences])
    U, s, Vt = np.linalg.svd(diff_matrices, full_matrices=False)
    
    # Select top discriminating directions
    n_directions = min(max_effects // 2, len(s), 8)
    effects = []
    
    for i in range(n_directions):
        # Reshape back to matrix form
        direction_matrix = Vt[i].reshape((DIM, DIM))
        
        # Symmetrize and project to PSD
        direction_matrix = 0.5 * (direction_matrix + direction_matrix.T)
        direction_matrix = project_to_psd(direction_matrix)
        
        # Create binary measurement
        trace = np.trace(direction_matrix)
        if trace > 1e-10:
            # Normalize to unit trace for probability interpretation
            effect = direction_matrix / trace
            complement = np.eye(DIM) - effect
            complement = project_to_psd(complement)
            
            effects.extend([effect, complement])
    
    # Final normalization
    effects = normalize_povm_effects(effects)
    
    logger.info(f"Generated {len(effects)} optimized POVM effects for state discrimination")
    return effects


def evaluate_povm_efficiency(effects: List[np.ndarray], 
                           test_states: List[np.ndarray]) -> Dict[str, float]:
    """
    Evaluate the efficiency of a POVM pack on test states.
    
    Args:
        effects: POVM effects to evaluate
        test_states: Test density matrices
        
    Returns:
        Dictionary with efficiency metrics
    """
    if not effects or not test_states:
        return {"error": "No effects or test states provided"}
    
    # Compute Fisher information
    fisher_infos = []
    for rho in test_states:
        fisher_matrix = povm_fisher_information(rho, effects)
        fisher_infos.append(np.trace(fisher_matrix))
    
    # Compute discrimination power
    discrimination_scores = []
    for i in range(len(test_states)):
        for j in range(i + 1, len(test_states)):
            # Compute probability distance
            probs_i = [max(np.trace(E @ test_states[i]), 0) for E in effects]
            probs_j = [max(np.trace(E @ test_states[j]), 0) for E in effects]
            
            # Total variation distance
            tv_distance = 0.5 * sum(abs(pi - pj) for pi, pj in zip(probs_i, probs_j))
            discrimination_scores.append(tv_distance)
    
    return {
        "mean_fisher_info": float(np.mean(fisher_infos)),
        "std_fisher_info": float(np.std(fisher_infos)),
        "mean_discrimination": float(np.mean(discrimination_scores)),
        "std_discrimination": float(np.std(discrimination_scores)),
        "n_effects": len(effects),
        "n_test_states": len(test_states)
    }