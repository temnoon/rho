#!/usr/bin/env python3
"""
Rho Space Designer: Create Custom Semantic Spaces for Specific Purposes

Instead of using arbitrary embedding dimensions, this system designs rho spaces 
with purposeful rhetorical axes that align with transformation goals.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class RhetoricalAxis:
    """Defines a purposeful dimension in rho space"""
    name: str
    description: str
    positive_examples: List[str]  # Text that scores high on this axis
    negative_examples: List[str]  # Text that scores low on this axis
    weight: float = 1.0
    
    def __post_init__(self):
        if len(self.positive_examples) == 0:
            raise ValueError(f"Axis '{self.name}' must have positive examples")

@dataclass
class RhoSpaceDesign:
    """Complete specification for a custom rho space"""
    name: str
    purpose: str  # What this space is designed for
    axes: List[RhetoricalAxis]
    target_dimensions: int = 64
    
    def get_axis_by_name(self, name: str) -> Optional[RhetoricalAxis]:
        return next((axis for axis in self.axes if axis.name == name), None)

class RhoSpaceDesigner:
    """Designs and builds custom rho spaces aligned with specific purposes"""
    
    def __init__(self, embed_function):
        self.embed = embed_function
        self.designed_spaces = {}
        
    def create_transformation_space(self, name: str, transformation_pairs: List[Tuple[str, str]]) -> RhoSpaceDesign:
        """
        Create a rho space optimized for a specific transformation
        
        Args:
            name: Name for this transformation space
            transformation_pairs: List of (source, target) text pairs
        
        Returns:
            RhoSpaceDesign optimized for this transformation
        """
        axes = []
        
        # Analyze transformation pairs to discover rhetorical dimensions
        for i, (source, target) in enumerate(transformation_pairs):
            axis_name = f"transform_axis_{i+1}"
            axis = RhetoricalAxis(
                name=axis_name,
                description=f"Transformation from '{source[:50]}...' to '{target[:50]}...'",
                positive_examples=[target],
                negative_examples=[source],
                weight=1.0
            )
            axes.append(axis)
        
        # Add meta-axes for transformation quality
        axes.extend([
            RhetoricalAxis(
                name="preservation",
                description="How much meaning is preserved during transformation",
                positive_examples=[pair[0] for pair in transformation_pairs],  # Sources preserve meaning
                negative_examples=["Random unrelated text", "Gibberish", "Complete topic change"],
                weight=2.0  # Very important for good transformations
            ),
            RhetoricalAxis(
                name="coherence", 
                description="Internal consistency of transformed text",
                positive_examples=["Well-structured narrative", "Logical progression", "Clear connections"],
                negative_examples=["Contradictory statements", "Random fragments", "Incoherent jumble"],
                weight=1.5
            )
        ])
        
        return RhoSpaceDesign(
            name=f"{name}_transformation_space",
            purpose=f"Optimized for {name} transformations",
            axes=axes,
            target_dimensions=64
        )
    
    def create_narrative_space(self) -> RhoSpaceDesign:
        """Create a rho space optimized for narrative transformations"""
        
        axes = [
            # Core narrative dimensions
            RhetoricalAxis(
                name="temporal_setting",
                description="When the story takes place",
                positive_examples=[
                    "In the distant future, among the stars",
                    "During the Renaissance in Florence", 
                    "In medieval times with knights and castles",
                    "In the roaring twenties jazz age"
                ],
                negative_examples=[
                    "The action was swift",  # Action, not time
                    "The character felt deeply",  # Emotion, not time
                    "The logical conclusion was clear"  # Logic, not time
                ]
            ),
            
            RhetoricalAxis(
                name="spatial_setting", 
                description="Where the story takes place",
                positive_examples=[
                    "On the surface of Mars in domed cities",
                    "In the streets of Victorian London",
                    "Deep in an enchanted forest",
                    "Aboard a sailing ship on stormy seas",
                    "In a bustling cyberpunk metropolis"
                ],
                negative_examples=[
                    "The hero felt brave",  # Emotion, not place
                    "Thinking quickly, she decided",  # Cognition, not place
                    "The rapid sequence of events"  # Pacing, not place
                ]
            ),
            
            RhetoricalAxis(
                name="genre_flavor",
                description="What kind of story this is",
                positive_examples=[
                    "Dark magic swirled around the ancient tome",
                    "The detective examined the crime scene carefully", 
                    "Laser blasts echoed through the space station",
                    "Their forbidden love bloomed despite the danger",
                    "The monster lurked in the shadows, waiting"
                ],
                negative_examples=[
                    "The simple sentence was clear",  # Style, not genre
                    "Yesterday, today, and tomorrow",  # Time, not genre
                    "He walked down the street"  # Generic, no genre markers
                ]
            ),
            
            RhetoricalAxis(
                name="emotional_tone",
                description="The emotional atmosphere of the narrative",
                positive_examples=[
                    "Joy filled her heart as she danced",
                    "Melancholy settled over the empty house",
                    "Terror gripped him as shadows moved",
                    "Peaceful contentment washed over the valley",
                    "Fierce determination drove her forward"
                ],
                negative_examples=[
                    "The building was made of brick",  # Physical description
                    "First, second, third in sequence",  # Structure
                    "The logical argument proceeded"  # Reasoning
                ]
            ),
            
            RhetoricalAxis(
                name="narrative_agency",
                description="How much control characters have over events", 
                positive_examples=[
                    "She boldly chose her own destiny",
                    "With decisive action, he changed everything",
                    "They seized control of their fate",
                    "Her willpower overcame all obstacles"
                ],
                negative_examples=[
                    "Events happened to him beyond his control",
                    "Fate swept her along helplessly", 
                    "Random chance determined the outcome",
                    "He was merely a passive observer"
                ]
            ),
            
            RhetoricalAxis(
                name="social_dynamics",
                description="How characters relate to each other and society",
                positive_examples=[
                    "The community rallied together in support",
                    "Isolated and alone, she faced the challenge",
                    "Their rivalry sparked creative tension",
                    "Formal protocols governed every interaction",
                    "Casual friendship blossomed into something more"
                ],
                negative_examples=[
                    "The mountain was very tall",  # Physical, not social
                    "Logical reasoning prevailed",  # Cognitive, not social
                    "Time moved slowly forward"  # Temporal, not social
                ]
            ),
            
            # Meta-narrative axes
            RhetoricalAxis(
                name="narrative_distance",
                description="How close or distant the storytelling feels",
                positive_examples=[
                    "I felt my heart pounding as I ran",  # Close, first person
                    "She experienced every sensation vividly",  # Close, intimate third
                    "The reader feels drawn into the moment"  # Direct engagement
                ],
                negative_examples=[
                    "The historical record shows that events occurred",  # Distant, academic
                    "One observes that patterns emerge over time",  # Abstract, removed
                    "Analysis reveals the underlying structure"  # Analytical distance
                ]
            ),
            
            RhetoricalAxis(
                name="semantic_density",
                description="How much meaning is packed into the language",
                positive_examples=[
                    "Symbolism layered upon metaphor revealed deeper truths",
                    "Every word carried weight beyond its surface meaning",
                    "Rich imagery painted landscapes of the soul"
                ],
                negative_examples=[
                    "He walked to the store",  # Simple, direct
                    "The facts were clearly stated",  # Straightforward
                    "Basic information was provided"  # Minimal meaning
                ]
            )
        ]
        
        return RhoSpaceDesign(
            name="narrative_transformation_space",
            purpose="Optimized for flexible narrative transformations",
            axes=axes,
            target_dimensions=64
        )
    
    def build_projection_matrix(self, design: RhoSpaceDesign) -> np.ndarray:
        """
        Build the W matrix that projects embeddings into the designed rho space
        
        Returns:
            W matrix (target_dims x embedding_dims) that maps to rhetorical axes
        """
        logger.info(f"Building projection matrix for '{design.name}' with {len(design.axes)} axes")
        
        # Collect all example texts and their axis labels
        all_texts = []
        axis_scores = []  # Will be (n_texts, n_axes)
        
        for axis in design.axes:
            # Add positive examples
            for text in axis.positive_examples:
                all_texts.append(text)
                scores = [0.0] * len(design.axes)
                axis_idx = design.axes.index(axis)
                scores[axis_idx] = 1.0 * axis.weight  # Positive score for this axis
                axis_scores.append(scores)
            
            # Add negative examples
            for text in axis.negative_examples:
                all_texts.append(text)
                scores = [0.0] * len(design.axes)
                axis_idx = design.axes.index(axis)
                scores[axis_idx] = -1.0 * axis.weight  # Negative score for this axis
                axis_scores.append(scores)
        
        # Embed all texts
        logger.info(f"Embedding {len(all_texts)} example texts...")
        embeddings = []
        for text in all_texts:
            emb = self.embed(text)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)  # (n_texts, embedding_dim)
        axis_scores = np.array(axis_scores)  # (n_texts, n_axes)
        
        logger.info(f"Embeddings shape: {embeddings.shape}, Axis scores shape: {axis_scores.shape}")
        
        # Learn mapping from embeddings to axis scores using linear regression
        from sklearn.linear_model import Ridge
        
        # For each target dimension, we want to learn a combination of axes
        W_components = []
        
        for i in range(design.target_dimensions):
            if i < len(design.axes):
                # Use the specific axis for this dimension
                target_scores = axis_scores[:, i]
                
                # Learn linear combination of embedding dimensions that predicts this axis
                ridge = Ridge(alpha=1.0)
                ridge.fit(embeddings, target_scores)
                W_components.append(ridge.coef_)
            else:
                # For extra dimensions, use PCA on remaining variance
                if i == len(design.axes):
                    # Compute residual after accounting for learned axes
                    predicted = np.zeros_like(embeddings[:, :1])
                    for j, w_comp in enumerate(W_components):
                        predicted += np.outer(axis_scores[:, j], w_comp)
                    
                    residual_embeddings = embeddings - predicted[:, :embeddings.shape[1]]
                    
                    # PCA on residual
                    pca = PCA(n_components=min(design.target_dimensions - len(design.axes), 
                                             residual_embeddings.shape[0] - 1,
                                             residual_embeddings.shape[1]))
                    pca.fit(residual_embeddings)
                    residual_components = pca.components_
                
                # Use residual PCA components for extra dimensions
                extra_idx = i - len(design.axes)
                if extra_idx < len(residual_components):
                    W_components.append(residual_components[extra_idx])
                else:
                    # Random component if we run out
                    random_comp = np.random.normal(0, 0.01, embeddings.shape[1])
                    random_comp /= np.linalg.norm(random_comp) + 1e-12
                    W_components.append(random_comp)
        
        W = np.array(W_components)  # (target_dims, embedding_dims)
        
        logger.info(f"Built projection matrix W with shape: {W.shape}")
        
        # Store the design for later use
        self.designed_spaces[design.name] = design
        
        return W
    
    def evaluate_space_quality(self, design: RhoSpaceDesign, W: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well the designed space captures the intended rhetorical axes
        """
        metrics = {}
        
        for axis in design.axes:
            # Test if positive examples score higher than negative on this axis
            pos_scores = []
            neg_scores = []
            
            for text in axis.positive_examples:
                emb = self.embed(text)
                rho_vec = W @ emb
                axis_idx = design.axes.index(axis)
                if axis_idx < len(rho_vec):
                    pos_scores.append(rho_vec[axis_idx])
            
            for text in axis.negative_examples:
                emb = self.embed(text)
                rho_vec = W @ emb
                axis_idx = design.axes.index(axis)
                if axis_idx < len(rho_vec):
                    neg_scores.append(rho_vec[axis_idx])
            
            if pos_scores and neg_scores:
                # Good separation means positive examples score higher
                separation = np.mean(pos_scores) - np.mean(neg_scores)
                metrics[f"{axis.name}_separation"] = separation
                metrics[f"{axis.name}_positive_mean"] = np.mean(pos_scores)
                metrics[f"{axis.name}_negative_mean"] = np.mean(neg_scores)
        
        metrics["overall_quality"] = np.mean([v for k, v in metrics.items() if "_separation" in k])
        
        return metrics

# Predefined space designs
def get_narrative_space_design() -> RhoSpaceDesign:
    """Get the standard narrative transformation space"""
    designer = RhoSpaceDesigner(None)  # embed function not needed for design
    return designer.create_narrative_space()

def get_genre_transformation_space() -> RhoSpaceDesign:
    """Space optimized for genre transformations"""
    return RhoSpaceDesign(
        name="genre_transformation_space",
        purpose="Transform between different story genres",
        axes=[
            RhetoricalAxis(
                name="fantasy_realism",
                description="Fantasy vs realistic elements",
                positive_examples=[
                    "Magic sparkled through the air as dragons soared overhead",
                    "The wizard cast ancient spells with his enchanted staff",
                    "Mystical portals opened to other dimensional realms"
                ],
                negative_examples=[
                    "The businessman walked to his office building",
                    "Medical research revealed important scientific findings", 
                    "Economic factors influenced the political decision"
                ]
            ),
            RhetoricalAxis(
                name="technology_level",
                description="Technological sophistication of the setting",
                positive_examples=[
                    "Neural interfaces connected minds across the galaxy",
                    "Quantum computers processed infinite parallel realities",
                    "Nanobots repaired tissue at the molecular level"
                ],
                negative_examples=[
                    "Horses pulled wooden carts along dirt roads",
                    "Candles provided the only light in the stone chamber",
                    "Hand-forged tools served the village blacksmith"
                ]
            ),
            RhetoricalAxis(
                name="mystery_clarity",
                description="How mysterious vs clear the narrative is",
                positive_examples=[
                    "Strange clues hinted at dark secrets hidden in shadows",
                    "The cryptic message revealed nothing while suggesting everything",
                    "Mysterious forces moved behind scenes unseen"
                ],
                negative_examples=[
                    "The facts were clearly presented in logical order",
                    "Direct communication eliminated all ambiguity",
                    "Transparent motives drove straightforward actions"
                ]
            )
        ]
    )

if __name__ == "__main__":
    # Example usage
    print("Testing Rho Space Designer...")
    
    # This would need the actual embed function
    # designer = RhoSpaceDesigner(embed_function)
    # narrative_design = designer.create_narrative_space()
    # print(f"Created space: {narrative_design.name} with {len(narrative_design.axes)} axes")