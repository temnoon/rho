"""
Integrability Testing System for Quantum Narrative Operations

This module implements the mathematical framework for testing integrability
in the Analytic Post-Lexical Grammatology system. It verifies that different
text segmentations yield equivalent quantum states, ensuring that the quantum
evolution is independent of arbitrary textual boundaries.

Key Concepts:
- Integrability: Different segmentations of the same text should yield the same final quantum state
- Path Independence: The route through text space should not affect the destination
- Compositional Consistency: ρ(AB) should equal the composition of ρ(A) and ρ(B)
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from core.quantum_state import create_maximally_mixed_state, apply_text_channel, diagnostics
from core.embedding import text_to_embedding_vector
from core.text_channels import text_to_channel, TextChannel

logger = logging.getLogger(__name__)

@dataclass
class IntegrabilityTestResult:
    """Results from an integrability test."""
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
class SegmentationPath:
    """Represents a path through text via different segmentations."""
    segments: List[str]
    channel_type: str
    alpha: float
    final_state: Optional[np.ndarray] = None
    intermediate_states: List[np.ndarray] = None
    applied_channels: List[TextChannel] = None


class IntegrabilityTester:
    """
    Core class for testing integrability of quantum narrative operations.
    
    The tester verifies that the quantum evolution of reading text is independent
    of how the text is segmented, which is a fundamental requirement for the 
    consistency of the Post-Lexical Grammatological framework.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the integrability tester.
        
        Args:
            tolerance: Maximum allowed distance between equivalent paths
        """
        self.tolerance = tolerance
        self.test_history = []
    
    def test_segmentation_integrability(
        self, 
        segments_a: List[str], 
        segments_b: List[str],
        channel_type: str = "rank_one_update",
        alpha: float = 0.3,
        initial_state: Optional[np.ndarray] = None
    ) -> IntegrabilityTestResult:
        """
        Test if two different segmentations of the same text yield equivalent quantum states.
        
        This is the core integrability test. It takes the same conceptual text content
        segmented in two different ways and verifies that the final quantum states
        are equivalent within tolerance.
        
        Args:
            segments_a: First segmentation of the text
            segments_b: Second segmentation of the text  
            channel_type: Type of quantum channel to use
            alpha: Blending parameter for channel operations
            initial_state: Starting quantum state (defaults to maximally mixed)
            
        Returns:
            IntegrabilityTestResult with detailed analysis
        """
        if initial_state is None:
            initial_state = create_maximally_mixed_state()
        
        logger.info(f"Testing integrability: {len(segments_a)} vs {len(segments_b)} segments")
        
        # Apply first segmentation path
        path_a = self._apply_segmentation_path(
            segments_a, initial_state.copy(), channel_type, alpha
        )
        
        # Apply second segmentation path  
        path_b = self._apply_segmentation_path(
            segments_b, initial_state.copy(), channel_type, alpha
        )
        
        # Calculate distance metrics between final states
        bures_dist = self._bures_distance(path_a.final_state, path_b.final_state)
        trace_dist = self._trace_distance(path_a.final_state, path_b.final_state)
        fidelity = self._quantum_fidelity(path_a.final_state, path_b.final_state)
        
        # Determine if test passes
        passes_test = bures_dist < self.tolerance
        
        # Generate recommendations
        recommendations = self._generate_integrability_recommendations(
            bures_dist, trace_dist, fidelity, passes_test, path_a, path_b
        )
        
        # Collect channel logs
        channel_logs = []
        for i, channel in enumerate(path_a.applied_channels):
            channel_logs.append({
                "path": "A",
                "segment_index": i,
                "segment_text": segments_a[i][:50] + "..." if len(segments_a[i]) > 50 else segments_a[i],
                "channel_type": channel_type
            })
        for i, channel in enumerate(path_b.applied_channels):
            channel_logs.append({
                "path": "B", 
                "segment_index": i,
                "segment_text": segments_b[i][:50] + "..." if len(segments_b[i]) > 50 else segments_b[i],
                "channel_type": channel_type
            })
        
        result = IntegrabilityTestResult(
            segments_a=segments_a,
            segments_b=segments_b,
            final_state_a=path_a.final_state,
            final_state_b=path_b.final_state,
            bures_distance=bures_dist,
            trace_distance=trace_dist,
            fidelity=fidelity,
            passes_test=passes_test,
            tolerance=self.tolerance,
            channel_logs=channel_logs,
            recommendations=recommendations
        )
        
        # Store in test history
        self.test_history.append(result)
        
        logger.info(f"Integrability test {'PASSED' if passes_test else 'FAILED'}: "
                   f"Bures distance = {bures_dist:.2e}")
        
        return result
    
    def test_compositional_consistency(
        self,
        text_pieces: List[str],
        channel_type: str = "rank_one_update", 
        alpha: float = 0.3
    ) -> Dict[str, Any]:
        """
        Test if reading text pieces individually vs. concatenated yields same result.
        
        This tests the compositional property: ρ(AB) = Φ_B(Φ_A(ρ_0))
        where Φ_A and Φ_B are the quantum channels for text pieces A and B.
        
        Args:
            text_pieces: List of text segments to test
            channel_type: Type of quantum channel
            alpha: Channel parameter
            
        Returns:
            Dict with compositional consistency results
        """
        initial_state = create_maximally_mixed_state()
        
        # Path 1: Read pieces sequentially
        state_sequential = initial_state.copy()
        for piece in text_pieces:
            embedding = text_to_embedding_vector(piece)
            state_sequential = apply_text_channel(state_sequential, embedding, alpha, channel_type)
        
        # Path 2: Read concatenated text at once
        concatenated_text = " ".join(text_pieces)
        embedding_concat = text_to_embedding_vector(concatenated_text)
        state_concatenated = apply_text_channel(initial_state.copy(), embedding_concat, alpha, channel_type)
        
        # Calculate distances
        bures_dist = self._bures_distance(state_sequential, state_concatenated)
        trace_dist = self._trace_distance(state_sequential, state_concatenated)
        fidelity = self._quantum_fidelity(state_sequential, state_concatenated)
        
        passes_test = bures_dist < self.tolerance
        
        return {
            "test_type": "compositional_consistency",
            "text_pieces": [piece[:30] + "..." if len(piece) > 30 else piece for piece in text_pieces],
            "concatenated_length": len(concatenated_text),
            "bures_distance": bures_dist,
            "trace_distance": trace_dist,
            "fidelity": fidelity,
            "passes_test": passes_test,
            "tolerance": self.tolerance,
            "interpretation": "Sequential vs concatenated reading equivalence test"
        }
    
    def analyze_path_sensitivity(
        self,
        base_text: str,
        num_segmentations: int = 5,
        channel_type: str = "rank_one_update",
        alpha: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze how sensitive the quantum evolution is to different segmentation strategies.
        
        This generates multiple random segmentations of the same text and measures
        the variance in final quantum states.
        
        Args:
            base_text: Text to segment in different ways
            num_segmentations: Number of different segmentations to test
            channel_type: Type of quantum channel
            alpha: Channel parameter
            
        Returns:
            Analysis of path sensitivity
        """
        if len(base_text.split()) < 4:
            logger.warning("Text too short for meaningful segmentation analysis")
            return {"error": "Text too short for analysis"}
        
        words = base_text.split()
        segmentations = []
        final_states = []
        
        initial_state = create_maximally_mixed_state()
        
        # Generate different segmentations
        for i in range(num_segmentations):
            # Create random segmentation by choosing split points
            if len(words) > 2:
                num_splits = np.random.randint(1, min(len(words), 5))
                split_points = sorted(np.random.choice(range(1, len(words)), num_splits, replace=False))
                split_points = [0] + list(split_points) + [len(words)]
                
                segments = []
                for j in range(len(split_points) - 1):
                    segment = " ".join(words[split_points[j]:split_points[j+1]])
                    segments.append(segment)
                
                segmentations.append(segments)
                
                # Apply this segmentation
                path = self._apply_segmentation_path(segments, initial_state.copy(), channel_type, alpha)
                final_states.append(path.final_state)
        
        if len(final_states) < 2:
            return {"error": "Could not generate enough segmentations"}
        
        # Calculate pairwise distances
        pairwise_distances = []
        for i in range(len(final_states)):
            for j in range(i + 1, len(final_states)):
                dist = self._bures_distance(final_states[i], final_states[j])
                pairwise_distances.append(dist)
        
        mean_distance = np.mean(pairwise_distances)
        max_distance = np.max(pairwise_distances)
        std_distance = np.std(pairwise_distances)
        
        return {
            "test_type": "path_sensitivity",
            "base_text_length": len(base_text),
            "num_segmentations": len(segmentations),
            "segmentations": [[seg[:20] + "..." if len(seg) > 20 else seg for seg in segs] for segs in segmentations],
            "mean_pairwise_distance": mean_distance,
            "max_pairwise_distance": max_distance,
            "std_pairwise_distance": std_distance,
            "passes_stability_test": max_distance < self.tolerance * 10,  # More lenient for path sensitivity
            "interpretation": "Low variance indicates good integrability; high variance suggests path sensitivity"
        }
    
    def _apply_segmentation_path(
        self, 
        segments: List[str], 
        initial_state: np.ndarray,
        channel_type: str,
        alpha: float
    ) -> SegmentationPath:
        """Apply a sequence of text segments as quantum channels."""
        current_state = initial_state.copy()
        intermediate_states = [current_state.copy()]
        applied_channels = []
        
        for segment in segments:
            if segment.strip():  # Skip empty segments
                embedding = text_to_embedding_vector(segment)
                channel = text_to_channel(embedding, alpha, channel_type)
                current_state = channel.apply(current_state)
                
                intermediate_states.append(current_state.copy())
                applied_channels.append(channel)
        
        return SegmentationPath(
            segments=segments,
            channel_type=channel_type,
            alpha=alpha,
            final_state=current_state,
            intermediate_states=intermediate_states,
            applied_channels=applied_channels
        )
    
    def _bures_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Calculate Bures distance between two density matrices."""
        try:
            # Bures distance: d_B(ρ₁, ρ₂) = √(2(1 - √F(ρ₁, ρ₂)))
            # where F is the quantum fidelity
            fidelity = self._quantum_fidelity(rho1, rho2)
            bures_dist = np.sqrt(2 * (1 - np.sqrt(fidelity)))
            return float(bures_dist)
        except Exception as e:
            logger.warning(f"Bures distance calculation failed: {e}")
            return float('inf')
    
    def _trace_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Calculate trace distance between two density matrices."""
        try:
            # Trace distance: ||ρ₁ - ρ₂||₁ / 2
            diff = rho1 - rho2
            eigenvals = np.linalg.eigvals(diff)
            trace_dist = 0.5 * np.sum(np.abs(eigenvals))
            return float(trace_dist)
        except Exception as e:
            logger.warning(f"Trace distance calculation failed: {e}")
            return float('inf')
    
    def _quantum_fidelity(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Calculate quantum fidelity between two density matrices."""
        try:
            # Quantum fidelity: F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))²
            sqrt_rho1 = self._matrix_sqrt(rho1)
            temp = sqrt_rho1 @ rho2 @ sqrt_rho1
            sqrt_temp = self._matrix_sqrt(temp)
            fidelity = np.trace(sqrt_temp) ** 2
            return float(fidelity.real)
        except Exception as e:
            logger.warning(f"Quantum fidelity calculation failed: {e}")
            return 0.0
    
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix square root using eigendecomposition."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        sqrt_eigenvals = np.sqrt(eigenvals)
        return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
    
    def _generate_integrability_recommendations(
        self,
        bures_dist: float,
        trace_dist: float, 
        fidelity: float,
        passes_test: bool,
        path_a: SegmentationPath,
        path_b: SegmentationPath
    ) -> List[str]:
        """Generate human-readable recommendations based on integrability test results."""
        recommendations = []
        
        if passes_test:
            recommendations.append("✓ Integrability test PASSED - segmentations yield equivalent quantum states")
            if bures_dist < self.tolerance / 10:
                recommendations.append("⭐ Excellent integrability - states are nearly identical")
        else:
            recommendations.append("⚠️ Integrability test FAILED - segmentations yield different quantum states")
            
            if bures_dist > self.tolerance * 10:
                recommendations.append("🚨 Large segmentation sensitivity detected")
                recommendations.append("Consider using smaller alpha values or different channel types")
            
            if fidelity < 0.9:
                recommendations.append("Low quantum fidelity suggests significant state differences")
        
        # Analyze segmentation patterns
        len_diff = abs(len(path_a.segments) - len(path_b.segments))
        if len_diff > 3:
            recommendations.append(f"Large difference in segment count ({len(path_a.segments)} vs {len(path_b.segments)})")
        
        # Channel-specific recommendations
        if path_a.channel_type == "coherent_rotation" and not passes_test:
            recommendations.append("Consider rank_one_update channels for better integrability")
        elif path_a.channel_type == "dephasing_mixture" and bures_dist > self.tolerance * 5:
            recommendations.append("Dephasing channels may be too sensitive to segmentation")
        
        return recommendations


# Convenience functions for direct usage
def test_text_integrability(
    segments_a: List[str],
    segments_b: List[str], 
    channel_type: str = "rank_one_update",
    alpha: float = 0.3,
    tolerance: float = 1e-6
) -> IntegrabilityTestResult:
    """
    Convenience function to test integrability of two segmentations.
    
    Args:
        segments_a: First segmentation
        segments_b: Second segmentation  
        channel_type: Type of quantum channel
        alpha: Channel parameter
        tolerance: Test tolerance
        
    Returns:
        IntegrabilityTestResult
    """
    tester = IntegrabilityTester(tolerance=tolerance)
    return tester.test_segmentation_integrability(segments_a, segments_b, channel_type, alpha)


def quick_integrability_check(text: str, alpha: float = 0.3) -> Dict[str, Any]:
    """
    Quick integrability check by comparing sentence-level vs word-level segmentation.
    
    Args:
        text: Text to analyze
        alpha: Channel parameter
        
    Returns:
        Quick integrability analysis
    """
    # Sentence-level segmentation
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 2:
        sentences = [text]  # Fallback if no sentence boundaries
    
    # Word-level segmentation (group words into chunks)
    words = text.split()
    if len(words) > 4:
        chunk_size = max(2, len(words) // len(sentences))
        word_segments = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            word_segments.append(chunk)
    else:
        word_segments = words
    
    # Test integrability
    result = test_text_integrability(sentences, word_segments, alpha=alpha)
    
    return {
        "test_type": "quick_integrability_check",
        "text_length": len(text),
        "sentence_segments": len(sentences),
        "word_segments": len(word_segments),
        "bures_distance": result.bures_distance,
        "passes_test": result.passes_test,
        "recommendations": result.recommendations[:3]  # Top 3 recommendations
    }