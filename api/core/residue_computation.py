"""
Residue Computation Engine for Quantum Narrative Operations

This module implements the mathematical framework for computing residues in the
context of quantum narrative analysis. In the Analytic Post-Lexical Grammatology
framework, residues arise from circular references, loops, and non-commuting
operations in text space.

Key Mathematical Concepts:
- Residue Theorem: Analysis of poles and singularities in the complex quantum evolution
- Narrative Loops: Text sequences that return to similar semantic states
- Circular References: Self-referential or mutually referential text patterns
- Phase Accumulation: Complex phase gathered during narrative circuits
- Monodromy: How quantum states transform under loop operations

This engine provides tools for detecting and analyzing these phenomena in quantum
narrative systems, enabling deeper understanding of textual coherence and meaning.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable, Complex
from dataclasses import dataclass
from core.quantum_state import apply_text_channel, diagnostics, create_maximally_mixed_state
from core.embedding import text_to_embedding_vector
from core.text_channels import text_to_channel, TextChannel
from scipy.linalg import logm, expm
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class ResidueResult:
    """Results from residue computation analysis."""
    loop_sequence: List[str]
    initial_state: np.ndarray
    final_state: np.ndarray
    residue_value: Complex
    phase_accumulation: float
    monodromy_matrix: np.ndarray
    loop_fidelity: float
    semantic_coherence: float
    singularities_detected: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class NarrativeLoop:
    """Represents a detected narrative loop."""
    loop_id: str
    text_segments: List[str]
    start_embedding: np.ndarray
    end_embedding: np.ndarray
    embedding_distance: float
    semantic_similarity: float
    loop_type: str  # 'circular', 'spiral', 'convergent', 'divergent'
    complexity: float

@dataclass
class CircularReference:
    """Represents a circular reference in text."""
    reference_id: str
    source_segment: str
    target_segment: str
    reference_strength: float
    semantic_closure: float
    path_segments: List[str]


class ResidueComputationEngine:
    """
    Core engine for computing residues in quantum narrative operations.
    
    The engine analyzes how quantum states evolve under loops and circular
    references, computing the residues that arise from these operations.
    This provides insight into the coherence and self-consistency of narratives.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, residue_tolerance: float = 1e-6):
        """
        Initialize the residue computation engine.
        
        Args:
            similarity_threshold: Threshold for detecting narrative loops
            residue_tolerance: Tolerance for residue calculations
        """
        self.similarity_threshold = similarity_threshold
        self.residue_tolerance = residue_tolerance
        self.detected_loops = []
        self.circular_references = []
        self.residue_history = []
    
    def detect_narrative_loops(
        self, 
        text_segments: List[str],
        min_loop_length: int = 3,
        max_loop_length: int = 20
    ) -> List[NarrativeLoop]:
        """
        Detect narrative loops in a sequence of text segments.
        
        A narrative loop occurs when the semantic content returns to a similar
        state after a sequence of transformations. This is detected by comparing
        embeddings of text segments.
        
        Args:
            text_segments: Sequence of text segments to analyze
            min_loop_length: Minimum length for a valid loop
            max_loop_length: Maximum length for a valid loop
            
        Returns:
            List of detected narrative loops
        """
        if len(text_segments) < min_loop_length * 2:
            logger.warning("Text too short for meaningful loop detection")
            return []
        
        # Compute embeddings for all segments
        embeddings = []
        for segment in text_segments:
            try:
                embedding = text_to_embedding_vector(segment)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed segment: {e}")
                embeddings.append(np.zeros(64))  # Fallback
        
        detected_loops = []
        
        # Look for loops of different lengths
        for loop_length in range(min_loop_length, min(max_loop_length + 1, len(text_segments) // 2)):
            for start_idx in range(len(text_segments) - loop_length):
                end_idx = start_idx + loop_length
                
                # Check if we can find a similar segment later in the text
                start_embedding = embeddings[start_idx]
                
                for potential_end in range(end_idx, min(len(text_segments), start_idx + max_loop_length)):
                    end_embedding = embeddings[potential_end]
                    
                    # Calculate semantic similarity
                    similarity = self._calculate_embedding_similarity(start_embedding, end_embedding)
                    
                    if similarity > self.similarity_threshold:
                        # Found a potential loop
                        loop_segments = text_segments[start_idx:potential_end + 1]
                        embedding_distance = np.linalg.norm(end_embedding - start_embedding)
                        
                        # Determine loop type based on trajectory analysis
                        loop_type = self._classify_loop_type(
                            embeddings[start_idx:potential_end + 1], 
                            start_embedding, 
                            end_embedding
                        )
                        
                        # Calculate complexity metric
                        complexity = self._calculate_loop_complexity(embeddings[start_idx:potential_end + 1])
                        
                        loop = NarrativeLoop(
                            loop_id=f"loop_{start_idx}_{potential_end}_{len(detected_loops)}",
                            text_segments=loop_segments,
                            start_embedding=start_embedding,
                            end_embedding=end_embedding,
                            embedding_distance=embedding_distance,
                            semantic_similarity=similarity,
                            loop_type=loop_type,
                            complexity=complexity
                        )
                        
                        detected_loops.append(loop)
                        logger.info(f"Detected {loop_type} loop of length {len(loop_segments)} "
                                  f"with similarity {similarity:.3f}")
        
        # Filter overlapping loops and keep the most significant ones
        filtered_loops = self._filter_overlapping_loops(detected_loops)
        
        # Store results
        self.detected_loops.extend(filtered_loops)
        
        return filtered_loops
    
    def compute_loop_residue(
        self,
        loop: NarrativeLoop,
        channel_type: str = "rank_one_update",
        alpha: float = 0.3,
        initial_state: Optional[np.ndarray] = None
    ) -> ResidueResult:
        """
        Compute the quantum residue for a given narrative loop.
        
        The residue quantifies the complex phase and transformation that
        accumulates when traversing a closed loop in the narrative space.
        
        Args:
            loop: Narrative loop to analyze
            channel_type: Type of quantum channel to use
            alpha: Channel parameter
            initial_state: Starting quantum state
            
        Returns:
            ResidueResult with detailed analysis
        """
        if initial_state is None:
            initial_state = create_maximally_mixed_state()
        
        logger.info(f"Computing residue for loop {loop.loop_id} with {len(loop.text_segments)} segments")
        
        # Apply quantum channels around the loop
        current_state = initial_state.copy()
        intermediate_states = [current_state.copy()]
        applied_channels = []
        
        for i, segment in enumerate(loop.text_segments):
            try:
                embedding = text_to_embedding_vector(segment)
                channel = text_to_channel(embedding, alpha, channel_type)
                current_state = channel.apply(current_state)
                
                intermediate_states.append(current_state.copy())
                applied_channels.append(channel)
            except Exception as e:
                logger.warning(f"Failed to apply channel for segment {i}: {e}")
        
        final_state = current_state
        
        # Compute the monodromy matrix (how state transforms under the loop)
        monodromy_matrix = self._compute_monodromy_matrix(applied_channels)
        
        # Calculate loop fidelity (how well the loop closes)
        loop_fidelity = self._calculate_loop_fidelity(initial_state, final_state)
        
        # Compute complex residue value
        residue_value = self._compute_complex_residue(initial_state, final_state, monodromy_matrix)
        
        # Calculate phase accumulation
        phase_accumulation = self._calculate_phase_accumulation(intermediate_states)
        
        # Analyze semantic coherence of the loop
        semantic_coherence = self._analyze_semantic_coherence(loop.text_segments)
        
        # Detect singularities in the loop
        singularities = self._detect_singularities(intermediate_states, applied_channels)
        
        # Generate recommendations
        recommendations = self._generate_residue_recommendations(
            residue_value, phase_accumulation, loop_fidelity, semantic_coherence, singularities
        )
        
        result = ResidueResult(
            loop_sequence=loop.text_segments,
            initial_state=initial_state,
            final_state=final_state,
            residue_value=residue_value,
            phase_accumulation=phase_accumulation,
            monodromy_matrix=monodromy_matrix,
            loop_fidelity=loop_fidelity,
            semantic_coherence=semantic_coherence,
            singularities_detected=singularities,
            recommendations=recommendations
        )
        
        # Store in residue history
        self.residue_history.append(result)
        
        logger.info(f"Computed residue: {residue_value}, phase: {phase_accumulation:.3f}, "
                   f"fidelity: {loop_fidelity:.3f}")
        
        return result
    
    def analyze_circular_references(
        self,
        text_segments: List[str],
        reference_patterns: Optional[List[str]] = None
    ) -> List[CircularReference]:
        """
        Analyze circular references in text segments.
        
        Circular references occur when text segments refer back to earlier
        content, creating loops in the semantic dependency graph.
        
        Args:
            text_segments: Text segments to analyze
            reference_patterns: Optional patterns to look for (pronouns, etc.)
            
        Returns:
            List of detected circular references
        """
        if reference_patterns is None:
            reference_patterns = [
                "this", "that", "it", "they", "these", "those",
                "aforementioned", "previously", "earlier", "above",
                "said", "such", "same", "former", "latter"
            ]
        
        detected_references = []
        
        # Build semantic similarity matrix
        embeddings = []
        for segment in text_segments:
            embedding = text_to_embedding_vector(segment)
            embeddings.append(embedding)
        
        # Look for potential circular references
        for i, current_segment in enumerate(text_segments):
            # Check if current segment contains reference patterns
            current_lower = current_segment.lower()
            has_reference_pattern = any(pattern in current_lower for pattern in reference_patterns)
            
            if has_reference_pattern:
                # Look for potential targets in earlier segments
                for j in range(i):
                    target_segment = text_segments[j]
                    
                    # Calculate reference strength based on semantic similarity
                    similarity = self._calculate_embedding_similarity(embeddings[i], embeddings[j])
                    
                    if similarity > 0.6:  # Threshold for reference detection
                        # Calculate semantic closure (how well the reference closes the loop)
                        path_segments = text_segments[j:i+1]
                        semantic_closure = self._calculate_semantic_closure(path_segments)
                        
                        reference = CircularReference(
                            reference_id=f"ref_{j}_{i}_{len(detected_references)}",
                            source_segment=current_segment,
                            target_segment=target_segment,
                            reference_strength=similarity,
                            semantic_closure=semantic_closure,
                            path_segments=path_segments
                        )
                        
                        detected_references.append(reference)
                        logger.debug(f"Detected circular reference from segment {i} to {j} "
                                   f"with strength {similarity:.3f}")
        
        # Store results
        self.circular_references.extend(detected_references)
        
        return detected_references
    
    def build_narrative_graph(
        self,
        text_segments: List[str],
        loops: Optional[List[NarrativeLoop]] = None,
        references: Optional[List[CircularReference]] = None
    ) -> nx.DiGraph:
        """
        Build a directed graph representing the narrative structure.
        
        The graph shows how text segments connect through loops and references,
        providing a topological view of the narrative structure.
        
        Args:
            text_segments: Text segments to analyze
            loops: Detected narrative loops
            references: Detected circular references
            
        Returns:
            NetworkX directed graph representing narrative structure
        """
        graph = nx.DiGraph()
        
        # Add nodes for each text segment
        for i, segment in enumerate(text_segments):
            graph.add_node(i, 
                          text=segment[:50] + "..." if len(segment) > 50 else segment,
                          length=len(segment),
                          segment_type='normal')
        
        # Add edges for sequential connections
        for i in range(len(text_segments) - 1):
            graph.add_edge(i, i + 1, 
                          edge_type='sequential',
                          weight=1.0)
        
        # Add edges for narrative loops
        if loops:
            for loop in loops:
                # Find the indices for this loop
                for i, segment in enumerate(text_segments):
                    if segment in loop.text_segments:
                        loop_start = i
                        break
                else:
                    continue
                
                loop_end = loop_start + len(loop.text_segments) - 1
                
                # Add loop closure edge
                graph.add_edge(loop_end, loop_start,
                              edge_type='loop_closure',
                              loop_type=loop.loop_type,
                              similarity=loop.semantic_similarity,
                              complexity=loop.complexity,
                              weight=loop.semantic_similarity)
                
                # Mark loop nodes
                for j in range(loop_start, loop_end + 1):
                    if j in graph.nodes:
                        graph.nodes[j]['segment_type'] = 'loop_member'
                        graph.nodes[j]['loop_id'] = loop.loop_id
        
        # Add edges for circular references
        if references:
            for ref in references:
                # Find source and target indices
                source_idx = None
                target_idx = None
                
                for i, segment in enumerate(text_segments):
                    if segment == ref.source_segment:
                        source_idx = i
                    if segment == ref.target_segment:
                        target_idx = i
                
                if source_idx is not None and target_idx is not None:
                    graph.add_edge(source_idx, target_idx,
                                  edge_type='circular_reference',
                                  reference_strength=ref.reference_strength,
                                  semantic_closure=ref.semantic_closure,
                                  weight=ref.reference_strength)
                    
                    # Mark reference nodes
                    graph.nodes[source_idx]['segment_type'] = 'reference_source'
                    graph.nodes[target_idx]['segment_type'] = 'reference_target'
        
        return graph
    
    def analyze_narrative_topology(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze the topological properties of the narrative graph.
        
        This provides insights into the overall structure and complexity
        of the narrative from a graph-theoretic perspective.
        
        Args:
            graph: Narrative graph to analyze
            
        Returns:
            Dictionary with topological analysis results
        """
        try:
            # Basic graph metrics
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            density = nx.density(graph)
            
            # Cycle analysis
            try:
                cycles = list(nx.simple_cycles(graph))
                num_cycles = len(cycles)
                cycle_lengths = [len(cycle) for cycle in cycles]
                avg_cycle_length = np.mean(cycle_lengths) if cycle_lengths else 0
            except:
                cycles = []
                num_cycles = 0
                avg_cycle_length = 0
            
            # Connectivity analysis
            is_connected = nx.is_weakly_connected(graph)
            num_components = nx.number_weakly_connected_components(graph)
            
            # Centrality measures
            try:
                in_degree_centrality = nx.in_degree_centrality(graph)
                out_degree_centrality = nx.out_degree_centrality(graph)
                betweenness_centrality = nx.betweenness_centrality(graph)
                
                # Find most central nodes
                max_in_centrality = max(in_degree_centrality.values()) if in_degree_centrality else 0
                max_out_centrality = max(out_degree_centrality.values()) if out_degree_centrality else 0
                max_betweenness = max(betweenness_centrality.values()) if betweenness_centrality else 0
            except:
                max_in_centrality = max_out_centrality = max_betweenness = 0
            
            # Edge type analysis
            edge_types = {}
            for _, _, data in graph.edges(data=True):
                edge_type = data.get('edge_type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            return {
                "basic_metrics": {
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "density": density,
                    "is_connected": is_connected,
                    "num_components": num_components
                },
                "cycle_analysis": {
                    "num_cycles": num_cycles,
                    "avg_cycle_length": avg_cycle_length,
                    "max_cycle_length": max(cycle_lengths) if cycle_lengths else 0,
                    "cycle_complexity": num_cycles / num_nodes if num_nodes > 0 else 0
                },
                "centrality_analysis": {
                    "max_in_centrality": max_in_centrality,
                    "max_out_centrality": max_out_centrality,
                    "max_betweenness": max_betweenness
                },
                "edge_type_distribution": edge_types,
                "structural_complexity": (num_cycles + len(edge_types)) / num_nodes if num_nodes > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Topology analysis failed: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(emb1, emb2) / (norm1 * norm2))
        except:
            return 0.0
    
    def _classify_loop_type(
        self, 
        embeddings: List[np.ndarray], 
        start_emb: np.ndarray, 
        end_emb: np.ndarray
    ) -> str:
        """Classify the type of narrative loop based on embedding trajectory."""
        if len(embeddings) < 3:
            return 'simple'
        
        # Calculate distances from start embedding
        distances = [np.linalg.norm(emb - start_emb) for emb in embeddings]
        
        # Analyze trajectory pattern
        max_distance = max(distances)
        final_distance = distances[-1]
        
        if final_distance < 0.1 * max_distance:
            return 'circular'  # Returns very close to start
        elif final_distance < 0.5 * max_distance:
            return 'convergent'  # Approaches start but doesn't quite close
        elif len(set(distances)) == len(distances):  # All distances different
            return 'spiral'  # Consistent directional movement
        else:
            return 'divergent'  # Moves away from start
    
    def _calculate_loop_complexity(self, embeddings: List[np.ndarray]) -> float:
        """Calculate complexity metric for a loop based on embedding variations."""
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(embeddings) - 1):
            dist = np.linalg.norm(embeddings[i+1] - embeddings[i])
            distances.append(dist)
        
        # Complexity is based on variance in step sizes
        return float(np.std(distances)) if distances else 0.0
    
    def _filter_overlapping_loops(self, loops: List[NarrativeLoop]) -> List[NarrativeLoop]:
        """Filter out overlapping loops, keeping the most significant ones."""
        if not loops:
            return []
        
        # Sort by significance (combination of similarity and complexity)
        sorted_loops = sorted(loops, 
                            key=lambda l: l.semantic_similarity * (1 + l.complexity), 
                            reverse=True)
        
        filtered = []
        for loop in sorted_loops:
            # Check if this loop significantly overlaps with any already selected
            overlaps = False
            for selected in filtered:
                # Simple overlap check - could be more sophisticated
                if len(set(loop.text_segments) & set(selected.text_segments)) > len(loop.text_segments) // 2:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(loop)
        
        return filtered
    
    def _compute_monodromy_matrix(self, channels: List[TextChannel]) -> np.ndarray:
        """Compute the monodromy matrix for a sequence of quantum channels."""
        try:
            # For now, return identity matrix as placeholder
            # In full implementation, this would compute the composition of channel matrices
            dim = 64
            return np.eye(dim, dtype=complex)
        except Exception as e:
            logger.warning(f"Monodromy computation failed: {e}")
            return np.eye(64, dtype=complex)
    
    def _calculate_loop_fidelity(self, initial_state: np.ndarray, final_state: np.ndarray) -> float:
        """Calculate how well a loop closes (fidelity between initial and final states)."""
        try:
            # Quantum fidelity calculation
            fidelity = np.trace(initial_state @ final_state).real
            return float(np.clip(fidelity, 0, 1))
        except Exception as e:
            logger.warning(f"Loop fidelity calculation failed: {e}")
            return 0.0
    
    def _compute_complex_residue(
        self, 
        initial_state: np.ndarray, 
        final_state: np.ndarray, 
        monodromy_matrix: np.ndarray
    ) -> Complex:
        """Compute the complex residue value for the loop."""
        try:
            # Simplified residue calculation based on trace difference
            trace_diff = np.trace(final_state - initial_state)
            # Add imaginary component based on monodromy matrix
            monodromy_trace = np.trace(monodromy_matrix)
            residue = complex(trace_diff.real, monodromy_trace.imag)
            return residue
        except Exception as e:
            logger.warning(f"Residue calculation failed: {e}")
            return complex(0, 0)
    
    def _calculate_phase_accumulation(self, states: List[np.ndarray]) -> float:
        """Calculate phase accumulation along the quantum trajectory."""
        try:
            total_phase = 0.0
            for i in range(len(states) - 1):
                # Calculate phase difference between consecutive states
                overlap = np.trace(states[i] @ states[i+1])
                phase = np.angle(overlap)
                total_phase += phase
            return float(total_phase)
        except Exception as e:
            logger.warning(f"Phase calculation failed: {e}")
            return 0.0
    
    def _analyze_semantic_coherence(self, segments: List[str]) -> float:
        """Analyze semantic coherence of text segments in a loop."""
        try:
            if len(segments) < 2:
                return 1.0
            
            # Calculate pairwise semantic similarities
            embeddings = [text_to_embedding_vector(seg) for seg in segments]
            similarities = []
            
            for i in range(len(embeddings) - 1):
                sim = self._calculate_embedding_similarity(embeddings[i], embeddings[i+1])
                similarities.append(sim)
            
            # Add similarity between last and first segment (loop closure)
            closure_sim = self._calculate_embedding_similarity(embeddings[-1], embeddings[0])
            similarities.append(closure_sim)
            
            return float(np.mean(similarities))
        except Exception as e:
            logger.warning(f"Semantic coherence analysis failed: {e}")
            return 0.0
    
    def _detect_singularities(
        self, 
        states: List[np.ndarray], 
        channels: List[TextChannel]
    ) -> List[Dict[str, Any]]:
        """Detect singularities in the quantum evolution."""
        singularities = []
        
        try:
            for i, state in enumerate(states):
                # Check for near-singular states (very low purity)
                purity = np.trace(state @ state).real
                
                if purity < 0.01:  # Very mixed state
                    singularities.append({
                        "type": "low_purity_singularity",
                        "position": i,
                        "purity": float(purity),
                        "description": f"Very mixed quantum state at position {i}"
                    })
                
                # Check for near-zero trace (should always be 1)
                trace = np.trace(state).real
                if abs(trace - 1.0) > 0.1:
                    singularities.append({
                        "type": "trace_anomaly",
                        "position": i,
                        "trace": float(trace),
                        "description": f"Trace anomaly at position {i}: {trace:.3f}"
                    })
        
        except Exception as e:
            logger.warning(f"Singularity detection failed: {e}")
        
        return singularities
    
    def _calculate_semantic_closure(self, path_segments: List[str]) -> float:
        """Calculate how well a path provides semantic closure."""
        try:
            if len(path_segments) < 2:
                return 0.0
            
            # Embedding similarity between start and end
            start_embedding = text_to_embedding_vector(path_segments[0])
            end_embedding = text_to_embedding_vector(path_segments[-1])
            
            return self._calculate_embedding_similarity(start_embedding, end_embedding)
        except Exception as e:
            logger.warning(f"Semantic closure calculation failed: {e}")
            return 0.0
    
    def _generate_residue_recommendations(
        self,
        residue_value: Complex,
        phase_accumulation: float,
        loop_fidelity: float,
        semantic_coherence: float,
        singularities: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on residue analysis."""
        recommendations = []
        
        # Analyze residue magnitude
        residue_magnitude = abs(residue_value)
        if residue_magnitude < self.residue_tolerance:
            recommendations.append("✓ Residue is negligible - loop is well-closed")
        elif residue_magnitude > 0.1:
            recommendations.append("⚠️ Large residue detected - significant loop discontinuity")
        
        # Analyze phase accumulation
        if abs(phase_accumulation) > 2 * np.pi:
            recommendations.append("🌀 Significant phase accumulation - complex narrative winding")
        elif abs(phase_accumulation) < 0.1:
            recommendations.append("📏 Minimal phase accumulation - coherent narrative loop")
        
        # Analyze loop fidelity
        if loop_fidelity > 0.9:
            recommendations.append("🔄 Excellent loop closure - narrative returns to initial state")
        elif loop_fidelity < 0.5:
            recommendations.append("⛔ Poor loop closure - narrative drift detected")
        
        # Analyze semantic coherence
        if semantic_coherence > 0.8:
            recommendations.append("📖 High semantic coherence - consistent narrative theme")
        elif semantic_coherence < 0.4:
            recommendations.append("📚 Low semantic coherence - fragmented narrative flow")
        
        # Analyze singularities
        if len(singularities) == 0:
            recommendations.append("✨ No singularities detected - smooth quantum evolution")
        else:
            recommendations.append(f"⚠️ {len(singularities)} singularities detected - check for problematic segments")
        
        return recommendations


# Convenience functions

def quick_residue_analysis(text_segments: List[str]) -> Dict[str, Any]:
    """
    Quick residue analysis of text segments.
    
    Args:
        text_segments: Text segments to analyze
        
    Returns:
        Dictionary with quick analysis results
    """
    engine = ResidueComputationEngine()
    
    # Detect loops
    loops = engine.detect_narrative_loops(text_segments)
    
    # Detect circular references
    references = engine.analyze_circular_references(text_segments)
    
    # Analyze the most significant loop if found
    residue_result = None
    if loops:
        most_significant_loop = max(loops, key=lambda l: l.semantic_similarity)
        residue_result = engine.compute_loop_residue(most_significant_loop)
    
    return {
        "text_segments_count": len(text_segments),
        "loops_detected": len(loops),
        "circular_references_detected": len(references),
        "most_significant_loop": {
            "loop_type": loops[0].loop_type if loops else None,
            "semantic_similarity": loops[0].semantic_similarity if loops else None,
            "complexity": loops[0].complexity if loops else None
        } if loops else None,
        "residue_analysis": {
            "residue_value": complex(residue_result.residue_value) if residue_result else None,
            "phase_accumulation": residue_result.phase_accumulation if residue_result else None,
            "loop_fidelity": residue_result.loop_fidelity if residue_result else None,
            "semantic_coherence": residue_result.semantic_coherence if residue_result else None
        } if residue_result else None,
        "recommendations": residue_result.recommendations if residue_result else [
            "No significant loops detected for residue analysis"
        ]
    }