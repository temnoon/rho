"""
Density Matrix Library Management System

This module provides advanced management, analysis, and creative synthesis
capabilities for density matrices. It transforms accumulated matrices from
byproducts into powerful tools for:

1. Creative Synthesis - Combine disparate ideas into new work
2. Quality Assessment - Identify your best work through quantum metrics
3. Similarity Analysis - Find connections between different pieces
4. Matrix Archaeology - Explore the evolution of your thinking
5. Compositional Intelligence - Generate new works from matrix combinations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

logger = logging.getLogger(__name__)

@dataclass
class MatrixMetadata:
    """Comprehensive metadata for a density matrix."""
    rho_id: str
    label: str
    creation_date: datetime
    source_type: str  # 'book', 'narrative', 'conversation', 'personal_file', 'synthesis'
    content_preview: str
    content_length: int
    reading_history: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    tags: Set[str]
    parent_matrices: List[str]  # For synthesized matrices
    synthesis_method: Optional[str]

@dataclass
class MatrixSimilarity:
    """Similarity relationship between two matrices."""
    matrix_a: str
    matrix_b: str
    distance: float
    similarity_type: str  # 'bures', 'trace', 'fidelity', 'eigenspace'
    conceptual_overlap: float
    shared_themes: List[str]

@dataclass
class QualityAssessment:
    """Quality assessment for a density matrix."""
    rho_id: str
    overall_score: float
    complexity_score: float
    coherence_score: float
    novelty_score: float
    depth_score: float
    emotional_resonance: float
    technical_merit: float
    assessment_rationale: str

class MatrixLibraryManager:
    """Advanced management system for density matrix collections."""
    
    def __init__(self):
        self.metadata_cache: Dict[str, MatrixMetadata] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.distance_cache: Dict[Tuple[str, str], float] = {}
        self.quality_assessments: Dict[str, QualityAssessment] = {}
        self.clusters: Dict[str, List[str]] = {}
        self.synthesis_recipes: Dict[str, Dict[str, Any]] = {}
        
    def register_matrix(self, rho_id: str, matrix: np.ndarray, 
                       source_info: Dict[str, Any]) -> MatrixMetadata:
        """Register a new matrix in the library with comprehensive metadata."""
        
        # Extract metadata from source info
        metadata = MatrixMetadata(
            rho_id=rho_id,
            label=source_info.get('label', f'Matrix_{rho_id[:8]}'),
            creation_date=datetime.now(),
            source_type=source_info.get('type', 'unknown'),
            content_preview=source_info.get('preview', '')[:200],
            content_length=source_info.get('content_length', 0),
            reading_history=source_info.get('reading_history', []),
            quality_metrics=self._calculate_intrinsic_metrics(matrix),
            tags=set(source_info.get('tags', [])),
            parent_matrices=source_info.get('parent_matrices', []),
            synthesis_method=source_info.get('synthesis_method')
        )
        
        self.metadata_cache[rho_id] = metadata
        
        # Invalidate cached calculations
        self.similarity_matrix = None
        
        logger.info(f"Registered matrix {rho_id} in library: {metadata.label}")
        return metadata
    
    def _calculate_intrinsic_metrics(self, matrix: np.ndarray) -> Dict[str, float]:
        """Calculate intrinsic quality metrics for a matrix."""
        try:
            # Eigenvalue analysis
            eigenvals = np.linalg.eigvals(matrix)
            eigenvals = np.real(eigenvals[eigenvals.real > 1e-10])
            eigenvals = eigenvals / np.sum(eigenvals)  # Normalize
            
            # Complexity metrics
            von_neumann_entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
            participation_ratio = 1.0 / np.sum(eigenvals**2)
            purity = np.real(np.trace(matrix @ matrix))
            
            # Coherence metrics
            off_diagonal_strength = np.sum(np.abs(matrix - np.diag(np.diag(matrix))))
            coherence = off_diagonal_strength / (np.sum(np.abs(matrix)) + 1e-10)
            
            # Information metrics
            effective_rank = np.exp(von_neumann_entropy)
            spectral_gap = np.max(eigenvals) - np.median(eigenvals)
            
            # Structural metrics
            matrix_norm = np.linalg.norm(matrix, 'fro')
            condition_number = np.linalg.cond(matrix)
            
            return {
                'von_neumann_entropy': float(von_neumann_entropy),
                'participation_ratio': float(participation_ratio),
                'purity': float(purity),
                'coherence': float(coherence),
                'effective_rank': float(effective_rank),
                'spectral_gap': float(spectral_gap),
                'matrix_norm': float(matrix_norm),
                'condition_number': float(np.log10(condition_number + 1))
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate intrinsic metrics: {e}")
            return {}
    
    def calculate_matrix_similarity(self, matrix_a: np.ndarray, matrix_b: np.ndarray,
                                  method: str = 'bures') -> float:
        """Calculate similarity between two density matrices."""
        try:
            if method == 'bures':
                # Bures distance (quantum metric)
                sqrt_a = self._matrix_sqrt(matrix_a)
                inner = sqrt_a @ matrix_b @ sqrt_a
                sqrt_inner = self._matrix_sqrt(inner)
                fidelity = np.real(np.trace(sqrt_inner))
                distance = np.sqrt(2 * (1 - fidelity))
                return 1.0 / (1.0 + distance)  # Convert to similarity
                
            elif method == 'trace':
                # Trace distance
                diff = matrix_a - matrix_b
                trace_norm = np.sum(np.abs(np.linalg.eigvals(diff)))
                return 1.0 - trace_norm / 2.0
                
            elif method == 'fidelity':
                # Quantum fidelity
                sqrt_a = self._matrix_sqrt(matrix_a)
                inner = sqrt_a @ matrix_b @ sqrt_a
                sqrt_inner = self._matrix_sqrt(inner)
                fidelity = np.real(np.trace(sqrt_inner))
                return fidelity
                
            elif method == 'eigenspace':
                # Eigenspace overlap
                evals_a, evecs_a = np.linalg.eigh(matrix_a)
                evals_b, evecs_b = np.linalg.eigh(matrix_b)
                
                # Weight by eigenvalues and calculate overlap
                overlap = 0.0
                for i, eval_a in enumerate(evals_a):
                    for j, eval_b in enumerate(evals_b):
                        vec_overlap = np.abs(np.dot(evecs_a[:, i], evecs_b[:, j]))**2
                        overlap += eval_a * eval_b * vec_overlap
                
                return overlap
                
        except Exception as e:
            logger.error(f"Failed to calculate {method} similarity: {e}")
            return 0.0
    
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix square root using eigendecomposition."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        sqrt_eigenvals = np.sqrt(eigenvals)
        return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
    
    def analyze_matrix_collection(self, matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive analysis of a matrix collection."""
        
        # Calculate all pairwise similarities
        matrix_ids = list(matrices.keys())
        n_matrices = len(matrix_ids)
        
        similarity_matrix = np.zeros((n_matrices, n_matrices))
        similarities = []
        
        for i in range(n_matrices):
            for j in range(i + 1, n_matrices):
                id_a, id_b = matrix_ids[i], matrix_ids[j]
                similarity = self.calculate_matrix_similarity(
                    matrices[id_a], matrices[id_b], method='bures'
                )
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                
                similarities.append(MatrixSimilarity(
                    matrix_a=id_a,
                    matrix_b=id_b,
                    distance=1.0 - similarity,
                    similarity_type='bures',
                    conceptual_overlap=similarity,
                    shared_themes=self._extract_shared_themes(id_a, id_b)
                ))
        
        # Hierarchical clustering
        if n_matrices > 2:
            condensed_distances = squareform(1.0 - similarity_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Create clusters at different levels
            clusters = {}
            for n_clusters in [2, 3, 5, min(8, n_matrices)]:
                if n_clusters < n_matrices:
                    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    clusters[f'{n_clusters}_clusters'] = {
                        f'cluster_{i}': [matrix_ids[j] for j, label in enumerate(cluster_labels) if label == i]
                        for i in range(1, n_clusters + 1)
                    }
        
        # Dimensionality reduction for visualization
        if n_matrices > 3:
            try:
                # PCA
                pca = PCA(n_components=min(3, n_matrices-1))
                pca_coords = pca.fit_transform(similarity_matrix)
                
                # t-SNE for non-linear embedding
                if n_matrices > 5:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_coords = tsne.fit_transform(similarity_matrix)
                else:
                    tsne_coords = None
                    
            except Exception as e:
                logger.warning(f"Dimensionality reduction failed: {e}")
                pca_coords = tsne_coords = None
        else:
            pca_coords = tsne_coords = None
        
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'matrix_ids': matrix_ids,
            'similarities': similarities,
            'clusters': clusters,
            'pca_coordinates': pca_coords.tolist() if pca_coords is not None else None,
            'tsne_coordinates': tsne_coords.tolist() if tsne_coords is not None else None,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _extract_shared_themes(self, rho_id_a: str, rho_id_b: str) -> List[str]:
        """Extract shared themes between two matrices based on metadata."""
        metadata_a = self.metadata_cache.get(rho_id_a)
        metadata_b = self.metadata_cache.get(rho_id_b)
        
        if not metadata_a or not metadata_b:
            return []
        
        # Find overlapping tags
        shared_tags = metadata_a.tags.intersection(metadata_b.tags)
        
        # Add source type if same
        if metadata_a.source_type == metadata_b.source_type:
            shared_tags.add(f"source:{metadata_a.source_type}")
        
        return list(shared_tags)
    
    def assess_matrix_quality(self, rho_id: str, matrix: np.ndarray,
                            content_metadata: Dict[str, Any]) -> QualityAssessment:
        """Comprehensive quality assessment for a density matrix."""
        
        metadata = self.metadata_cache.get(rho_id)
        if not metadata:
            logger.warning(f"No metadata found for matrix {rho_id}")
            
        intrinsic_metrics = metadata.quality_metrics if metadata else {}
        
        # Complexity assessment
        complexity_score = self._assess_complexity(intrinsic_metrics, content_metadata)
        
        # Coherence assessment  
        coherence_score = self._assess_coherence(intrinsic_metrics, matrix)
        
        # Novelty assessment
        novelty_score = self._assess_novelty(rho_id, matrix)
        
        # Depth assessment
        depth_score = self._assess_depth(intrinsic_metrics, content_metadata)
        
        # Emotional resonance (estimated from matrix properties)
        emotional_resonance = self._assess_emotional_resonance(matrix, content_metadata)
        
        # Technical merit
        technical_merit = self._assess_technical_merit(intrinsic_metrics)
        
        # Overall score (weighted combination)
        overall_score = (
            0.2 * complexity_score +
            0.2 * coherence_score +
            0.15 * novelty_score +
            0.15 * depth_score +
            0.15 * emotional_resonance +
            0.15 * technical_merit
        )
        
        assessment = QualityAssessment(
            rho_id=rho_id,
            overall_score=overall_score,
            complexity_score=complexity_score,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            depth_score=depth_score,
            emotional_resonance=emotional_resonance,
            technical_merit=technical_merit,
            assessment_rationale=self._generate_assessment_rationale(
                complexity_score, coherence_score, novelty_score, 
                depth_score, emotional_resonance, technical_merit
            )
        )
        
        self.quality_assessments[rho_id] = assessment
        return assessment
    
    def _assess_complexity(self, metrics: Dict[str, float], 
                          content: Dict[str, Any]) -> float:
        """Assess complexity based on entropy and content characteristics."""
        entropy = metrics.get('von_neumann_entropy', 0)
        participation = metrics.get('participation_ratio', 1)
        content_length = content.get('content_length', 0)
        
        # Normalize entropy (log2(64) ≈ 6 is max for 64x64 matrix)
        entropy_score = min(entropy / 6.0, 1.0)
        
        # Participation ratio score (higher is more complex)
        participation_score = min(participation / 32.0, 1.0)  # 32 is half of 64
        
        # Content length factor
        length_score = min(content_length / 10000, 1.0)  # Normalize to 10k chars
        
        return 0.4 * entropy_score + 0.4 * participation_score + 0.2 * length_score
    
    def _assess_coherence(self, metrics: Dict[str, float], matrix: np.ndarray) -> float:
        """Assess coherence based on off-diagonal structure and purity."""
        coherence = metrics.get('coherence', 0)
        purity = metrics.get('purity', 0)
        
        # Coherence indicates quantum superposition
        coherence_score = min(coherence * 2, 1.0)  # Scale appropriately
        
        # Medium purity indicates balanced mixing
        optimal_purity = 0.3  # Neither pure nor maximally mixed
        purity_score = 1.0 - abs(purity - optimal_purity) / optimal_purity
        
        return 0.6 * coherence_score + 0.4 * purity_score
    
    def _assess_novelty(self, rho_id: str, matrix: np.ndarray) -> float:
        """Assess novelty by comparing to existing matrices."""
        if len(self.metadata_cache) <= 1:
            return 1.0  # First or only matrix is novel
        
        # Calculate average similarity to all other matrices
        similarities = []
        for other_id, other_metadata in self.metadata_cache.items():
            if other_id != rho_id:
                # This would need access to other matrices - simplified for now
                similarities.append(0.5)  # Placeholder
        
        if similarities:
            avg_similarity = np.mean(similarities)
            novelty_score = 1.0 - avg_similarity  # Lower similarity = higher novelty
        else:
            novelty_score = 1.0
        
        return novelty_score
    
    def _assess_depth(self, metrics: Dict[str, float], 
                     content: Dict[str, Any]) -> float:
        """Assess depth based on effective rank and reading complexity."""
        effective_rank = metrics.get('effective_rank', 1)
        content_length = content.get('content_length', 0)
        reading_steps = len(content.get('reading_history', []))
        
        # Effective rank indicates dimensionality of meaning
        rank_score = min(effective_rank / 20.0, 1.0)  # Normalize to reasonable range
        
        # Length and reading steps indicate investment
        investment_score = min((content_length * reading_steps) / 50000, 1.0)
        
        return 0.7 * rank_score + 0.3 * investment_score
    
    def _assess_emotional_resonance(self, matrix: np.ndarray, 
                                   content: Dict[str, Any]) -> float:
        """Estimate emotional resonance from matrix spectral properties."""
        try:
            eigenvals = np.linalg.eigvals(matrix)
            eigenvals = np.real(eigenvals[eigenvals.real > 1e-10])
            eigenvals = eigenvals / np.sum(eigenvals)
            
            # Spectral spread indicates emotional range
            spectral_spread = np.std(eigenvals)
            
            # Asymmetry in eigenvalue distribution
            spectral_skew = abs(np.mean(eigenvals) - np.median(eigenvals))
            
            # Combine into resonance score
            resonance_score = min(spectral_spread * 10 + spectral_skew * 20, 1.0)
            
            return resonance_score
        except:
            return 0.5  # Default moderate score
    
    def _assess_technical_merit(self, metrics: Dict[str, float]) -> float:
        """Assess technical merit based on matrix properties."""
        condition_number = metrics.get('condition_number', 0)
        matrix_norm = metrics.get('matrix_norm', 0)
        
        # Well-conditioned matrices have lower condition numbers
        conditioning_score = max(0, 1.0 - condition_number / 10.0)
        
        # Appropriate norm indicates well-scaled representation
        norm_score = min(matrix_norm / 10.0, 1.0)
        
        return 0.6 * conditioning_score + 0.4 * norm_score
    
    def _generate_assessment_rationale(self, complexity: float, coherence: float,
                                     novelty: float, depth: float, 
                                     resonance: float, technical: float) -> str:
        """Generate human-readable assessment rationale."""
        
        components = []
        
        if complexity > 0.7:
            components.append("high conceptual complexity")
        elif complexity > 0.4:
            components.append("moderate complexity")
        else:
            components.append("relatively simple structure")
            
        if coherence > 0.7:
            components.append("strong internal coherence")
        elif coherence > 0.4:
            components.append("adequate coherence")
        else:
            components.append("fragmented structure")
            
        if novelty > 0.7:
            components.append("highly novel content")
        elif novelty > 0.4:
            components.append("some novel elements")
        else:
            components.append("familiar themes")
            
        if depth > 0.7:
            components.append("substantial depth")
        elif depth > 0.4:
            components.append("moderate depth")
        else:
            components.append("surface-level treatment")
            
        return f"This work demonstrates {', '.join(components)}."
    
    def synthesize_matrices(self, matrix_ids: List[str], matrices: Dict[str, np.ndarray],
                          method: str = 'convex_combination', 
                          weights: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Synthesize multiple matrices to create new work."""
        
        if not matrix_ids or len(matrix_ids) < 2:
            raise ValueError("Need at least 2 matrices for synthesis")
        
        selected_matrices = [matrices[rid] for rid in matrix_ids if rid in matrices]
        
        if len(selected_matrices) != len(matrix_ids):
            raise ValueError("Some matrix IDs not found in collection")
        
        if weights is None:
            weights = [1.0 / len(selected_matrices)] * len(selected_matrices)
        elif len(weights) != len(selected_matrices):
            raise ValueError("Number of weights must match number of matrices")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        if method == 'convex_combination':
            # Standard convex combination
            synthesized = np.zeros_like(selected_matrices[0])
            for matrix, weight in zip(selected_matrices, weights):
                synthesized += weight * matrix
                
        elif method == 'geometric_mean':
            # Quantum geometric mean (more complex)
            synthesized = self._geometric_mean_matrices(selected_matrices, weights)
            
        elif method == 'coherent_superposition':
            # Create coherent superposition in eigenspace
            synthesized = self._coherent_superposition(selected_matrices, weights)
            
        elif method == 'interference_pattern':
            # Create interference patterns between matrices
            synthesized = self._interference_synthesis(selected_matrices, weights)
            
        else:
            raise ValueError(f"Unknown synthesis method: {method}")
        
        # Ensure result is a valid density matrix
        synthesized = self._project_to_density_matrix(synthesized)
        
        # Generate synthesis metadata
        synthesis_metadata = {
            'parent_matrices': matrix_ids,
            'method': method,
            'weights': weights.tolist(),
            'synthesis_timestamp': datetime.now().isoformat(),
            'expected_properties': self._predict_synthesis_properties(
                matrix_ids, method, weights
            )
        }
        
        return synthesized, synthesis_metadata
    
    def _geometric_mean_matrices(self, matrices: List[np.ndarray], 
                               weights: np.ndarray) -> np.ndarray:
        """Compute weighted geometric mean of density matrices."""
        # Simplified geometric mean - use log space
        log_sum = np.zeros_like(matrices[0])
        
        for matrix, weight in zip(matrices, weights):
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-10)  # Avoid log(0)
            log_matrix = eigenvecs @ np.diag(np.log(eigenvals)) @ eigenvecs.T
            log_sum += weight * log_matrix
        
        # Exponentiate back
        eigenvals, eigenvecs = np.linalg.eigh(log_sum)
        result = eigenvecs @ np.diag(np.exp(eigenvals)) @ eigenvecs.T
        
        return result
    
    def _coherent_superposition(self, matrices: List[np.ndarray], 
                              weights: np.ndarray) -> np.ndarray:
        """Create coherent superposition in eigenspace."""
        # Decompose each matrix and create superposition of eigenvectors
        combined_evecs = []
        combined_evals = []
        
        for matrix, weight in zip(matrices, weights):
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            # Weight the contribution of each matrix
            combined_evecs.append(np.sqrt(weight) * eigenvecs)
            combined_evals.append(weight * eigenvals)
        
        # Create interference pattern
        superposed_evecs = np.sum(combined_evecs, axis=0)
        superposed_evals = np.sum(combined_evals, axis=0)
        
        # Normalize and reconstruct
        superposed_evals = superposed_evals / np.sum(superposed_evals)
        result = superposed_evecs @ np.diag(superposed_evals) @ superposed_evecs.T
        
        return result
    
    def _interference_synthesis(self, matrices: List[np.ndarray], 
                              weights: np.ndarray) -> np.ndarray:
        """Create interference patterns between matrices."""
        n = len(matrices)
        result = np.zeros_like(matrices[0])
        
        # Add individual contributions
        for matrix, weight in zip(matrices, weights):
            result += weight * matrix
        
        # Add interference terms (cross products)
        for i in range(n):
            for j in range(i + 1, n):
                # Geometric mean of off-diagonal pairs
                cross_term = self._matrix_sqrt(matrices[i] @ matrices[j])
                interference_weight = 2 * np.sqrt(weights[i] * weights[j])
                result += 0.1 * interference_weight * cross_term  # Weak interference
        
        return result
    
    def _project_to_density_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Project matrix to valid density matrix (positive semidefinite, trace 1)."""
        # Ensure Hermitian
        matrix = (matrix + matrix.T.conj()) / 2
        
        # Ensure positive semidefinite
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 0)
        matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure trace 1
        trace = np.trace(matrix)
        if abs(trace) > 1e-10:
            matrix = matrix / trace
        else:
            # Fallback to maximally mixed state
            matrix = np.eye(matrix.shape[0]) / matrix.shape[0]
        
        return matrix
    
    def _predict_synthesis_properties(self, matrix_ids: List[str], 
                                    method: str, weights: np.ndarray) -> Dict[str, Any]:
        """Predict properties of synthesized matrix."""
        
        parent_metadata = [self.metadata_cache.get(mid) for mid in matrix_ids]
        parent_metadata = [m for m in parent_metadata if m is not None]
        
        if not parent_metadata:
            return {}
        
        # Predict combined tags
        all_tags = set()
        for metadata in parent_metadata:
            all_tags.update(metadata.tags)
        
        # Predict source types
        source_types = [m.source_type for m in parent_metadata]
        dominant_source = max(set(source_types), key=source_types.count)
        
        # Predict complexity (weighted average)
        complexities = [m.quality_metrics.get('von_neumann_entropy', 0) 
                       for m in parent_metadata]
        expected_complexity = np.average(complexities, weights=weights[:len(complexities)])
        
        return {
            'predicted_tags': list(all_tags),
            'dominant_source_type': dominant_source,
            'expected_complexity': float(expected_complexity),
            'synthesis_novelty': 'high',  # Synthesis typically creates novel combinations
            'recommended_label': f"Synthesis_{method}_{len(matrix_ids)}matrices"
        }
    
    def find_best_work(self, matrices: Dict[str, np.ndarray], 
                      criteria: str = 'overall', top_k: int = 5) -> List[Tuple[str, float]]:
        """Find best work based on quality assessments."""
        
        if not self.quality_assessments:
            # Generate assessments for all matrices
            for rho_id, matrix in matrices.items():
                metadata = self.metadata_cache.get(rho_id, {})
                content_info = {
                    'content_length': getattr(metadata, 'content_length', 0),
                    'reading_history': getattr(metadata, 'reading_history', [])
                }
                self.assess_matrix_quality(rho_id, matrix, content_info)
        
        # Sort by specified criteria
        assessments = list(self.quality_assessments.values())
        
        if criteria == 'overall':
            sorted_work = sorted(assessments, key=lambda x: x.overall_score, reverse=True)
        elif criteria == 'complexity':
            sorted_work = sorted(assessments, key=lambda x: x.complexity_score, reverse=True)
        elif criteria == 'novelty':
            sorted_work = sorted(assessments, key=lambda x: x.novelty_score, reverse=True)
        elif criteria == 'depth':
            sorted_work = sorted(assessments, key=lambda x: x.depth_score, reverse=True)
        elif criteria == 'resonance':
            sorted_work = sorted(assessments, key=lambda x: x.emotional_resonance, reverse=True)
        else:
            sorted_work = sorted(assessments, key=lambda x: x.overall_score, reverse=True)
        
        # Return top k with scores
        return [(work.rho_id, getattr(work, f'{criteria}_score')) 
                for work in sorted_work[:top_k]]
    
    def generate_synthesis_recommendations(self, matrices: Dict[str, np.ndarray],
                                         target_theme: Optional[str] = None,
                                         max_combinations: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations for interesting matrix combinations."""
        
        recommendations = []
        matrix_ids = list(matrices.keys())
        
        # Analyze current collection
        analysis = self.analyze_matrix_collection(matrices)
        
        # Recommendation 1: Combine most similar matrices for refinement
        similarities = analysis['similarities']
        if similarities:
            most_similar = max(similarities, key=lambda x: x.conceptual_overlap)
            recommendations.append({
                'type': 'refinement',
                'matrices': [most_similar.matrix_a, most_similar.matrix_b],
                'method': 'geometric_mean',
                'rationale': f"Refine shared themes between similar works (similarity: {most_similar.conceptual_overlap:.3f})",
                'expected_outcome': 'distilled essence of common elements'
            })
        
        # Recommendation 2: Combine most dissimilar for innovation
        if similarities:
            most_dissimilar = min(similarities, key=lambda x: x.conceptual_overlap)
            recommendations.append({
                'type': 'innovation',
                'matrices': [most_dissimilar.matrix_a, most_dissimilar.matrix_b],
                'method': 'interference_pattern',
                'rationale': f"Create novel synthesis from contrasting works (similarity: {most_dissimilar.conceptual_overlap:.3f})",
                'expected_outcome': 'breakthrough creative combination'
            })
        
        # Recommendation 3: Best work amplification
        best_work = self.find_best_work(matrices, criteria='overall', top_k=3)
        if len(best_work) >= 2:
            best_ids = [work[0] for work in best_work[:2]]
            recommendations.append({
                'type': 'amplification',
                'matrices': best_ids,
                'method': 'coherent_superposition',
                'rationale': f"Amplify strengths of your best work (scores: {', '.join(f'{w[1]:.3f}' for w in best_work[:2])})",
                'expected_outcome': 'enhanced version of your strongest ideas'
            })
        
        # Recommendation 4: Thematic clustering
        clusters = analysis.get('clusters', {})
        for cluster_name, cluster_data in clusters.items():
            for cluster_id, cluster_matrices in cluster_data.items():
                if len(cluster_matrices) >= 3:
                    recommendations.append({
                        'type': 'thematic_synthesis',
                        'matrices': cluster_matrices[:3],  # Limit to 3 for manageable synthesis
                        'method': 'convex_combination',
                        'rationale': f"Synthesize thematic cluster {cluster_id} with {len(cluster_matrices)} related works",
                        'expected_outcome': 'comprehensive treatment of shared themes'
                    })
        
        # Recommendation 5: Temporal synthesis (recent + historical)
        if len(matrix_ids) >= 4:
            # Sort by creation date if available
            dated_matrices = []
            for rho_id in matrix_ids:
                metadata = self.metadata_cache.get(rho_id)
                if metadata:
                    dated_matrices.append((rho_id, metadata.creation_date))
            
            if len(dated_matrices) >= 4:
                dated_matrices.sort(key=lambda x: x[1])
                recent = [dated_matrices[-1][0], dated_matrices[-2][0]]  # 2 most recent
                historical = [dated_matrices[0][0], dated_matrices[1][0]]  # 2 oldest
                
                recommendations.append({
                    'type': 'temporal_bridge',
                    'matrices': recent + historical,
                    'method': 'convex_combination',
                    'weights': [0.3, 0.3, 0.2, 0.2],  # Favor recent work
                    'rationale': "Bridge your early and recent work to see evolution",
                    'expected_outcome': 'synthesis spanning your creative development'
                })
        
        return recommendations[:max_combinations]

# Global library manager instance
matrix_library = MatrixLibraryManager()