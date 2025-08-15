#!/usr/bin/env python3
"""
Rho-Native Transformation System

Instead of hardcoded string replacements, this system learns transformation
directions in the designed rho space and applies them through matrix evolution.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from scipy.linalg import logm, expm

logger = logging.getLogger(__name__)

@dataclass
class TransformationVector:
    """A learned direction in rho space that represents a semantic transformation"""
    name: str
    description: str
    direction: np.ndarray  # Direction vector in rho space
    strength: float = 1.0
    examples: List[Tuple[str, str]] = None  # (source, target) pairs used to learn this
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []

@dataclass
class TransformationLibrary:
    """Collection of learned transformations"""
    name: str
    transformations: Dict[str, TransformationVector]
    rho_space_name: str  # Which designed space these work in
    
    def add_transformation(self, transformation: TransformationVector):
        self.transformations[transformation.name] = transformation
    
    def get_transformation(self, name: str) -> Optional[TransformationVector]:
        return self.transformations.get(name)
    
    def compose_transformations(self, *names: str, weights: List[float] = None) -> np.ndarray:
        """Combine multiple transformations with optional weights"""
        if weights is None:
            weights = [1.0] * len(names)
        
        if len(names) != len(weights):
            raise ValueError("Number of transformation names must match number of weights")
        
        combined_direction = np.zeros_like(list(self.transformations.values())[0].direction)
        
        for name, weight in zip(names, weights):
            if name in self.transformations:
                combined_direction += weight * self.transformations[name].direction
        
        return combined_direction

class RhoTransformationEngine:
    """Learn and apply transformations through rho space evolution"""
    
    def __init__(self, embed_function, project_to_rho_function, rho_to_text_function=None):
        self.embed = embed_function
        self.project_to_rho = project_to_rho_function
        self.rho_to_text = rho_to_text_function or self._fallback_rho_to_text
        self.libraries = {}
    
    def learn_transformation_from_examples(self, 
                                         name: str,
                                         description: str,
                                         example_pairs: List[Tuple[str, str]],
                                         library_name: str = "default") -> TransformationVector:
        """
        Learn a transformation direction from example (source, target) pairs
        """
        logger.info(f"Learning transformation '{name}' from {len(example_pairs)} examples")
        
        source_rhos = []
        target_rhos = []
        
        for source_text, target_text in example_pairs:
            # Get rho matrices for source and target
            source_emb = self.embed(source_text)
            target_emb = self.embed(target_text)
            
            source_vector = self.project_to_rho(source_emb)
            target_vector = self.project_to_rho(target_emb)
            
            # Convert vectors to rho matrices
            source_rho = self._vector_to_rho_matrix(source_vector)
            target_rho = self._vector_to_rho_matrix(target_vector)
            
            source_rhos.append(source_rho)
            target_rhos.append(target_rho)
        
        # Learn the average transformation direction
        directions = []
        for source_rho, target_rho in zip(source_rhos, target_rhos):
            # Simple linear direction (could be more sophisticated)
            direction = target_rho - source_rho
            directions.append(direction)
        
        # Average the directions
        avg_direction = np.mean(directions, axis=0)
        
        # Normalize to unit length for consistent strength application
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-12:
            avg_direction = avg_direction / norm
        
        transformation = TransformationVector(
            name=name,
            description=description,
            direction=avg_direction,
            strength=1.0,
            examples=example_pairs
        )
        
        # Add to library
        if library_name not in self.libraries:
            self.libraries[library_name] = TransformationLibrary(
                name=library_name,
                transformations={},
                rho_space_name="unknown"
            )
        
        self.libraries[library_name].add_transformation(transformation)
        
        logger.info(f"Learned transformation '{name}' with direction norm: {norm:.4f}")
        return transformation
    
    def learn_transformation_from_description(self, 
                                            name: str,
                                            description: str,
                                            source_description: str,
                                            target_description: str,
                                            library_name: str = "default") -> TransformationVector:
        """
        Learn a transformation from textual descriptions of source and target domains
        """
        logger.info(f"Learning transformation '{name}' from descriptions")
        
        # Create synthetic examples based on descriptions
        source_examples = [
            source_description,
            f"This text exemplifies {source_description}",
            f"A clear example of {source_description}",
            f"Typical {source_description} content"
        ]
        
        target_examples = [
            target_description,
            f"This text exemplifies {target_description}",
            f"A clear example of {target_description}",
            f"Typical {target_description} content"
        ]
        
        # Create example pairs
        example_pairs = list(zip(source_examples, target_examples))
        
        return self.learn_transformation_from_examples(
            name, description, example_pairs, library_name
        )
    
    def apply_transformation(self, 
                           text: str, 
                           transformation_name: str,
                           strength: float = 1.0,
                           library_name: str = "default") -> str:
        """
        Apply a learned transformation to text
        """
        if library_name not in self.libraries:
            raise ValueError(f"Library '{library_name}' not found")
        
        library = self.libraries[library_name]
        transformation = library.get_transformation(transformation_name)
        
        if not transformation:
            raise ValueError(f"Transformation '{transformation_name}' not found in library '{library_name}'")
        
        return self._apply_transformation_vector(text, transformation.direction, strength)
    
    def apply_transformation_vector(self, 
                                  text: str, 
                                  direction: np.ndarray,
                                  strength: float = 1.0) -> str:
        """
        Apply a transformation direction vector to text
        """
        return self._apply_transformation_vector(text, direction, strength)
    
    def _apply_transformation_vector(self, text: str, direction: np.ndarray, strength: float) -> str:
        """Internal method to apply transformation via rho evolution"""
        
        # Get current rho state of the text
        emb = self.embed(text)
        rho_vector = self.project_to_rho(emb)
        
        # Convert vector to proper rho matrix if needed
        current_rho = self._vector_to_rho_matrix(rho_vector)
        
        # Ensure direction is compatible with rho matrix dimensions
        if direction.ndim == 1:
            direction = self._vector_to_rho_matrix(direction)
        
        # Apply transformation direction
        transformed_rho = current_rho + strength * direction
        
        # Ensure the result is a valid rho matrix (positive semidefinite, trace 1)
        transformed_rho = self._normalize_rho_matrix(transformed_rho)
        
        # Generate text from the transformed rho state
        return self.rho_to_text(transformed_rho, original_text=text)
    
    def _normalize_rho_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Ensure rho matrix is valid (positive semidefinite, trace 1)"""
        
        # Make Hermitian
        rho = (rho + rho.conj().T) / 2
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        # Clip negative eigenvalues to small positive values
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Normalize to trace 1
        eigenvals = eigenvals / np.sum(eigenvals)
        
        # Reconstruct
        rho_normalized = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        
        return rho_normalized.real  # Take real part for numerical stability
    
    def _vector_to_rho_matrix(self, vector: np.ndarray) -> np.ndarray:
        """Convert a 1D vector to a 2D rho matrix (density matrix)"""
        vector = np.asarray(vector)
        
        if vector.ndim == 2:
            return vector  # Already a matrix
        
        if vector.ndim != 1:
            raise ValueError(f"Expected 1D or 2D array, got {vector.ndim}D")
        
        # Method 1: Outer product (creates rank-1 density matrix)
        # Normalize vector first
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        
        # Create density matrix as outer product |ψ⟩⟨ψ|
        rho = np.outer(vector, vector.conj())
        
        # Ensure it's normalized (trace = 1)
        trace = np.trace(rho)
        if trace > 1e-12:
            rho = rho / trace
        
        return rho.real  # Take real part for numerical stability
    
    def _fallback_rho_to_text(self, rho: np.ndarray, original_text: str = "") -> str:
        """
        Fallback method for generating text from rho state
        
        This is a placeholder until we have a proper rho-conditioned text generator.
        Currently returns the original text with a note about the transformation.
        """
        
        # Analyze the rho matrix to extract semantic properties
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        purity = np.sum(eigenvals**2)
        entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-12))
        
        # Find dominant dimensions
        dominant_indices = np.argsort(eigenvals)[-3:][::-1]
        
        # Simple analysis-based modification of text
        # This is temporary until we have proper rho-to-text generation
        
        if purity > 0.8:
            prefix = "With crystalline clarity, "
        elif purity < 0.3:
            prefix = "Through layered complexity, "
        else:
            prefix = ""
        
        if entropy > 2.5:
            suffix = " - a narrative rich with interconnected meanings."
        elif entropy < 1.0:
            suffix = " - a story of focused purpose."
        else:
            suffix = ""
        
        # Return modified text as placeholder
        if original_text:
            return f"{prefix}{original_text}{suffix}"
        else:
            return f"{prefix}The transformed narrative emerges from the rho state{suffix}"
    
    def create_narrative_library(self) -> TransformationLibrary:
        """Create a library of common narrative transformations"""
        
        library_name = "narrative_transformations"
        
        # Setting transformations
        self.learn_transformation_from_description(
            name="earth_to_mars",
            description="Transform Earth settings to Mars equivalents",
            source_description="Earth locations, cities, natural environments, familiar places",
            target_description="Martian colonies, domes, alien landscapes, off-world settlements",
            library_name=library_name
        )
        
        self.learn_transformation_from_description(
            name="modern_to_historical",
            description="Transform modern settings to historical periods",
            source_description="Contemporary technology, modern cities, current day references",
            target_description="Historical periods, ancient settings, medieval times, past eras",
            library_name=library_name
        )
        
        # Genre transformations
        self.learn_transformation_from_description(
            name="realistic_to_fantasy",
            description="Transform realistic elements to fantasy equivalents",
            source_description="Realistic scenarios, everyday situations, ordinary events",
            target_description="Magical elements, fantasy creatures, mystical events, supernatural powers",
            library_name=library_name
        )
        
        # Tone transformations
        self.learn_transformation_from_description(
            name="serious_to_humorous",
            description="Transform serious tone to humorous",
            source_description="Serious, formal, grave, solemn, dignified tone",
            target_description="Humorous, playful, witty, comedic, lighthearted tone",
            library_name=library_name
        )
        
        # Perspective transformations
        self.learn_transformation_from_description(
            name="first_to_third_person",
            description="Transform first person to third person narrative",
            source_description="First person narrative, I statements, personal perspective",
            target_description="Third person narrative, he/she statements, external perspective",
            library_name=library_name
        )
        
        return self.libraries[library_name]
    
    def save_library(self, library_name: str, filepath: str):
        """Save a transformation library to disk"""
        if library_name not in self.libraries:
            raise ValueError(f"Library '{library_name}' not found")
        
        library = self.libraries[library_name]
        
        # Convert to serializable format
        data = {
            "name": library.name,
            "rho_space_name": library.rho_space_name,
            "transformations": {}
        }
        
        for name, transformation in library.transformations.items():
            data["transformations"][name] = {
                "name": transformation.name,
                "description": transformation.description,
                "direction": transformation.direction.tolist(),
                "strength": transformation.strength,
                "examples": transformation.examples
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved library '{library_name}' to {filepath}")
    
    def load_library(self, filepath: str) -> TransformationLibrary:
        """Load a transformation library from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        library = TransformationLibrary(
            name=data["name"],
            transformations={},
            rho_space_name=data["rho_space_name"]
        )
        
        for name, trans_data in data["transformations"].items():
            transformation = TransformationVector(
                name=trans_data["name"],
                description=trans_data["description"],
                direction=np.array(trans_data["direction"]),
                strength=trans_data["strength"],
                examples=trans_data["examples"]
            )
            library.add_transformation(transformation)
        
        self.libraries[library.name] = library
        logger.info(f"Loaded library '{library.name}' with {len(library.transformations)} transformations")
        
        return library

# Utility functions for common transformations
def create_setting_transformation(engine: RhoTransformationEngine, 
                                source_setting: str, 
                                target_setting: str) -> str:
    """Create a transformation between two settings"""
    
    transformation_name = f"{source_setting.lower().replace(' ', '_')}_to_{target_setting.lower().replace(' ', '_')}"
    
    return engine.learn_transformation_from_description(
        name=transformation_name,
        description=f"Transform {source_setting} to {target_setting}",
        source_description=f"Stories set in {source_setting}",
        target_description=f"Stories set in {target_setting}"
    )

if __name__ == "__main__":
    print("Testing Rho Transformation Engine...")
    # This would need the actual functions
    # engine = RhoTransformationEngine(embed_fn, project_fn, generate_fn)
    # library = engine.create_narrative_library()
    # print(f"Created library with {len(library.transformations)} transformations")