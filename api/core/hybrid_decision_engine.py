"""
Quantum-LLM Hybrid Decision Engine

Determines optimal processing strategy by analyzing:
1. Quantum transformation quality indicators
2. Text complexity metrics
3. Task-specific requirements
4. Embedding space characteristics
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TransformationStrategy(Enum):
    PURE_QUANTUM = "pure_quantum"
    LLM_GUIDED = "llm_guided"
    HYBRID_ITERATIVE = "hybrid_iterative"
    QUANTUM_FIRST_LLM_REFINE = "quantum_first_llm_refine"
    LLM_FIRST_QUANTUM_EMBED = "llm_first_quantum_embed"
    HIERARCHICAL = "hierarchical"

@dataclass
class QualityMetrics:
    """Metrics for assessing transformation quality."""
    quantum_coherence: float  # How well quantum state represents semantic content
    semantic_preservation: float  # How much meaning is retained
    structural_integrity: float  # Punctuation, grammar, formatting
    information_density: float  # Concept richness per token
    embedding_alignment: float  # How well rho-embedding aligns with global embedding
    transformation_distance: float  # Quantum distance traveled
    
@dataclass
class TextComplexity:
    """Metrics for analyzing input text complexity."""
    token_count: int
    sentence_count: int
    avg_sentence_length: float
    paragraph_count: int
    punctuation_density: float
    concept_density: float  # Unique concepts per 100 words
    syntactic_complexity: float  # Parse tree depth, etc.
    domain_specificity: float  # Technical terminology ratio

@dataclass
class TaskRequirements:
    """Specific requirements for the transformation task."""
    preserve_structure: bool
    preserve_terminology: bool
    preserve_length: bool
    require_coherence: bool
    allow_creativity: bool
    target_readability: Optional[float] = None
    format_constraints: List[str] = None

class HybridDecisionEngine:
    """Decides optimal quantum-LLM processing strategy."""
    
    def __init__(self):
        self.decision_history = []
        self.performance_tracking = {}
        
    def analyze_text_complexity(self, text: str) -> TextComplexity:
        """Analyze complexity characteristics of input text."""
        import re
        
        tokens = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Count punctuation
        punctuation_chars = len(re.findall(r'[^\w\s]', text))
        punctuation_density = punctuation_chars / len(text) if text else 0
        
        # Estimate concept density (simplified)
        unique_words = len(set(word.lower() for word in tokens if len(word) > 3))
        concept_density = (unique_words / len(tokens)) * 100 if tokens else 0
        
        # Estimate syntactic complexity (simplified)
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
        syntactic_complexity = min(avg_sentence_length / 15, 1.0)  # Normalize to 0-1
        
        # Estimate domain specificity
        technical_indicators = len(re.findall(r'[A-Z]{2,}|ρ|σ|quantum|cognitive|phenomenological', text))
        domain_specificity = min(technical_indicators / len(tokens) * 100, 1.0) if tokens else 0
        
        return TextComplexity(
            token_count=len(tokens),
            sentence_count=len(sentences),
            avg_sentence_length=avg_sentence_length,
            paragraph_count=len(paragraphs),
            punctuation_density=punctuation_density,
            concept_density=concept_density,
            syntactic_complexity=syntactic_complexity,
            domain_specificity=domain_specificity
        )
    
    def assess_quantum_quality(self, 
                             original_text: str, 
                             transformed_text: str, 
                             quantum_state: np.ndarray,
                             transformation_distance: float) -> QualityMetrics:
        """Assess quality of quantum transformation."""
        
        # Quantum coherence - how well the quantum state represents content
        eigenvals = np.linalg.eigvals(quantum_state)
        eigenvals = eigenvals[eigenvals.real > 1e-10]
        purity = np.sum(eigenvals.real ** 2)
        entropy = -np.sum(eigenvals.real * np.log(eigenvals.real + 1e-10))
        quantum_coherence = purity * (1 - entropy / np.log(64))  # Normalized coherence
        
        # Semantic preservation (simplified)
        original_words = set(original_text.lower().split())
        transformed_words = set(transformed_text.lower().split())
        word_overlap = len(original_words & transformed_words) / len(original_words | transformed_words)
        semantic_preservation = word_overlap
        
        # Structural integrity
        import re
        original_punct = len(re.findall(r'[.!?;:,]', original_text))
        transformed_punct = len(re.findall(r'[.!?;:,]', transformed_text))
        original_sentences = len(re.split(r'[.!?]+', original_text))
        transformed_sentences = len(re.split(r'[.!?]+', transformed_text))
        
        punct_ratio = min(transformed_punct / original_punct, 1.0) if original_punct > 0 else 1.0
        sentence_ratio = min(transformed_sentences / original_sentences, 1.0) if original_sentences > 0 else 1.0
        structural_integrity = (punct_ratio + sentence_ratio) / 2
        
        # Information density
        original_unique = len(set(original_text.lower().split()))
        transformed_unique = len(set(transformed_text.lower().split()))
        density_ratio = transformed_unique / len(transformed_text.split()) if transformed_text.split() else 0
        baseline_density = original_unique / len(original_text.split()) if original_text.split() else 0
        information_density = min(density_ratio / baseline_density, 1.0) if baseline_density > 0 else density_ratio
        
        # Embedding alignment (would need actual embeddings to compute)
        embedding_alignment = 0.8  # Placeholder - would compute cosine similarity
        
        return QualityMetrics(
            quantum_coherence=float(quantum_coherence),
            semantic_preservation=semantic_preservation,
            structural_integrity=structural_integrity,
            information_density=information_density,
            embedding_alignment=embedding_alignment,
            transformation_distance=transformation_distance
        )
    
    def decide_strategy(self, 
                       text: str, 
                       transformation_request: Dict,
                       task_requirements: TaskRequirements) -> Tuple[TransformationStrategy, Dict]:
        """Decide optimal processing strategy based on analysis."""
        
        complexity = self.analyze_text_complexity(text)
        
        # Decision factors
        factors = {
            "text_length": complexity.token_count,
            "complexity_score": (complexity.syntactic_complexity + 
                               complexity.concept_density / 100 + 
                               complexity.domain_specificity) / 3,
            "structural_requirements": (task_requirements.preserve_structure + 
                                      task_requirements.preserve_terminology + 
                                      task_requirements.preserve_length) / 3,
            "creativity_allowance": task_requirements.allow_creativity,
            "format_complexity": len(task_requirements.format_constraints or [])
        }
        
        strategy_scores = {}
        
        # Pure Quantum: Best for structural changes, length preservation, high coherence
        pure_quantum_score = 0.0
        if complexity.token_count < 200:  # Short texts
            pure_quantum_score += 0.3
        if task_requirements.preserve_structure:
            pure_quantum_score += 0.25
        if task_requirements.preserve_length:
            pure_quantum_score += 0.25
        if factors["complexity_score"] < 0.5:  # Low complexity
            pure_quantum_score += 0.2
        
        strategy_scores[TransformationStrategy.PURE_QUANTUM] = pure_quantum_score
        
        # LLM Guided: Best for complex restructuring, creative tasks
        llm_guided_score = 0.0
        if complexity.token_count > 1000:  # Long texts
            llm_guided_score += 0.3
        if task_requirements.allow_creativity:
            llm_guided_score += 0.3
        if factors["format_complexity"] > 2:  # Complex formatting
            llm_guided_score += 0.25
        if factors["complexity_score"] > 0.7:  # High complexity
            llm_guided_score += 0.15
        
        strategy_scores[TransformationStrategy.LLM_GUIDED] = llm_guided_score
        
        # Quantum First, LLM Refine: Best balance for medium texts
        quantum_first_score = 0.0
        if 200 <= complexity.token_count <= 1000:  # Medium texts
            quantum_first_score += 0.4
        if task_requirements.preserve_structure and task_requirements.allow_creativity:
            quantum_first_score += 0.3
        if complexity.punctuation_density < 0.05:  # Low punctuation (may need LLM cleanup)
            quantum_first_score += 0.2
        if factors["structural_requirements"] > 0.5:
            quantum_first_score += 0.1
        
        strategy_scores[TransformationStrategy.QUANTUM_FIRST_LLM_REFINE] = quantum_first_score
        
        # LLM First, Quantum Embed: Best for semantic coherence with quantum benefits
        llm_first_score = 0.0
        if complexity.domain_specificity > 0.1:  # Technical content
            llm_first_score += 0.25
        if task_requirements.require_coherence:
            llm_first_score += 0.25
        if complexity.syntactic_complexity > 0.8:  # Complex syntax
            llm_first_score += 0.25
        if not task_requirements.preserve_length:  # Length flexibility
            llm_first_score += 0.25
        
        strategy_scores[TransformationStrategy.LLM_FIRST_QUANTUM_EMBED] = llm_first_score
        
        # Hybrid Iterative: Best for complex tasks requiring multiple passes
        hybrid_score = 0.0
        if complexity.token_count > 2000:  # Very long texts
            hybrid_score += 0.4
        if factors["format_complexity"] > 3:  # Very complex formatting
            hybrid_score += 0.3
        if (task_requirements.preserve_structure and 
            task_requirements.allow_creativity and 
            task_requirements.require_coherence):
            hybrid_score += 0.3
        
        strategy_scores[TransformationStrategy.HYBRID_ITERATIVE] = hybrid_score
        
        # Hierarchical: Best for medium-long texts needing detailed processing
        hierarchical_score = 0.0
        if 500 <= complexity.token_count <= 5000:  # Sweet spot for hierarchical processing
            hierarchical_score += 0.4
        if task_requirements.preserve_structure and task_requirements.preserve_terminology:
            hierarchical_score += 0.3
        if complexity.domain_specificity > 0.05:  # Some technical content
            hierarchical_score += 0.2
        if factors["complexity_score"] > 0.4 and factors["complexity_score"] < 0.8:  # Medium complexity
            hierarchical_score += 0.1
        
        strategy_scores[TransformationStrategy.HIERARCHICAL] = hierarchical_score
        
        # Select best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        
        decision_metadata = {
            "complexity_analysis": complexity,
            "decision_factors": factors,
            "strategy_scores": {s.value: score for s, score in strategy_scores.items()},
            "confidence": best_strategy[1],
            "reasoning": self._generate_reasoning(best_strategy[0], factors, complexity, task_requirements)
        }
        
        # Record decision for learning
        self.decision_history.append({
            "strategy": best_strategy[0],
            "factors": factors,
            "complexity": complexity,
            "confidence": best_strategy[1]
        })
        
        return best_strategy[0], decision_metadata
    
    def _generate_reasoning(self, 
                          strategy: TransformationStrategy, 
                          factors: Dict, 
                          complexity: TextComplexity, 
                          requirements: TaskRequirements) -> str:
        """Generate human-readable reasoning for the strategy choice."""
        
        reasons = []
        
        if strategy == TransformationStrategy.PURE_QUANTUM:
            if complexity.token_count < 200:
                reasons.append("short text suitable for direct quantum processing")
            if requirements.preserve_structure:
                reasons.append("structure preservation favors quantum coherence")
            if factors["complexity_score"] < 0.5:
                reasons.append("low complexity allows clean quantum transformation")
        
        elif strategy == TransformationStrategy.LLM_GUIDED:
            if complexity.token_count > 1000:
                reasons.append("long text requires LLM's contextual understanding")
            if requirements.allow_creativity:
                reasons.append("creative task benefits from LLM flexibility")
            if factors["complexity_score"] > 0.7:
                reasons.append("high complexity needs LLM's linguistic sophistication")
        
        elif strategy == TransformationStrategy.QUANTUM_FIRST_LLM_REFINE:
            if 200 <= complexity.token_count <= 1000:
                reasons.append("medium-length text ideal for quantum-then-LLM pipeline")
            if complexity.punctuation_density < 0.05:
                reasons.append("sparse punctuation suggests need for LLM cleanup")
            if requirements.preserve_structure and requirements.allow_creativity:
                reasons.append("balanced requirements favor hybrid approach")
        
        elif strategy == TransformationStrategy.LLM_FIRST_QUANTUM_EMBED:
            if complexity.domain_specificity > 0.1:
                reasons.append("technical content benefits from LLM understanding then quantum embedding")
            if requirements.require_coherence:
                reasons.append("coherence requirement favors LLM-first approach")
        
        elif strategy == TransformationStrategy.HYBRID_ITERATIVE:
            if complexity.token_count > 2000:
                reasons.append("very long text requires iterative processing")
            if factors["format_complexity"] > 3:
                reasons.append("complex formatting needs multiple refinement passes")
        
        elif strategy == TransformationStrategy.HIERARCHICAL:
            if 500 <= complexity.token_count <= 5000:
                reasons.append("medium-long text ideal for hierarchical chunking")
            if requirements.preserve_structure and requirements.preserve_terminology:
                reasons.append("structure/terminology preservation benefits from chunk-level processing")
            if complexity.domain_specificity > 0.05:
                reasons.append("technical content handled well by hierarchical approach")
        
        return "; ".join(reasons)
    
    def evaluate_outcome(self, 
                        strategy: TransformationStrategy,
                        quality_metrics: QualityMetrics,
                        user_satisfaction: Optional[float] = None) -> Dict:
        """Record outcome for strategy learning."""
        
        outcome_score = (
            quality_metrics.quantum_coherence * 0.2 +
            quality_metrics.semantic_preservation * 0.25 +
            quality_metrics.structural_integrity * 0.25 +
            quality_metrics.information_density * 0.15 +
            quality_metrics.embedding_alignment * 0.15
        )
        
        if user_satisfaction is not None:
            outcome_score = outcome_score * 0.7 + user_satisfaction * 0.3
        
        if strategy.value not in self.performance_tracking:
            self.performance_tracking[strategy.value] = []
        
        self.performance_tracking[strategy.value].append({
            "outcome_score": outcome_score,
            "quality_metrics": quality_metrics,
            "user_satisfaction": user_satisfaction
        })
        
        return {
            "outcome_score": outcome_score,
            "strategy_performance": self.get_strategy_performance(strategy)
        }
    
    def get_strategy_performance(self, strategy: TransformationStrategy) -> Dict:
        """Get performance statistics for a strategy."""
        
        if strategy.value not in self.performance_tracking:
            return {"sample_size": 0}
        
        outcomes = [result["outcome_score"] for result in self.performance_tracking[strategy.value]]
        
        return {
            "sample_size": len(outcomes),
            "avg_score": np.mean(outcomes),
            "score_std": np.std(outcomes),
            "success_rate": len([s for s in outcomes if s > 0.7]) / len(outcomes)
        }
    
    def recommend_adjustments(self, 
                            strategy: TransformationStrategy,
                            quality_metrics: QualityMetrics) -> List[str]:
        """Recommend adjustments based on quality assessment."""
        
        recommendations = []
        
        if quality_metrics.quantum_coherence < 0.5:
            recommendations.append("Consider increasing quantum state preparation steps")
        
        if quality_metrics.semantic_preservation < 0.6:
            recommendations.append("Add semantic alignment constraints to LLM prompts")
        
        if quality_metrics.structural_integrity < 0.7:
            recommendations.append("Implement post-processing validation for punctuation and formatting")
        
        if quality_metrics.information_density < 0.6:
            recommendations.append("Reduce compression tendency in transformation")
        
        if quality_metrics.embedding_alignment < 0.7:
            recommendations.append("Refine quantum-to-embedding projection matrix")
        
        if quality_metrics.transformation_distance < 1e-10:
            recommendations.append("Increase transformation strength or check POVM effectiveness")
        
        return recommendations


def create_task_requirements_from_request(transformation_request: Dict) -> TaskRequirements:
    """Convert transformation request to TaskRequirements object."""
    
    return TaskRequirements(
        preserve_structure=transformation_request.get("preserve_structure", True),
        preserve_terminology=transformation_request.get("preserve_terminology", True),
        preserve_length=transformation_request.get("preserve_length", False),
        require_coherence=transformation_request.get("require_coherence", True),
        allow_creativity=transformation_request.get("creativity", 0.5) > 0.6,
        target_readability=transformation_request.get("target_readability"),
        format_constraints=transformation_request.get("format_constraints", [])
    )