#!/usr/bin/env python3
"""
Rho-Conditioned Text Generation

This module implements sophisticated text generation from rho matrices,
using the semantic properties encoded in the quantum state to guide
narrative construction.
"""

import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RhoTextPattern:
    """A pattern for generating text based on rho properties"""
    name: str
    description: str
    rho_condition: Callable[[np.ndarray], bool]  # Function to test if rho matches this pattern
    text_templates: List[str]  # Templates with {placeholders}
    semantic_extractors: Dict[str, Callable[[np.ndarray], str]]  # Extract semantic content from rho

@dataclass
class SemanticContent:
    """Extracted semantic information from rho matrix"""
    dominant_themes: List[str]
    emotional_tone: str
    narrative_structure: str
    temporal_setting: str
    spatial_setting: str
    genre_markers: List[str]
    agency_level: str
    social_dynamics: str

class RhoTextGenerator:
    """Generate text from rho matrices using learned semantic patterns"""
    
    def __init__(self, embed_function, space_design=None):
        self.embed = embed_function
        self.space_design = space_design
        self.patterns = []
        self.semantic_vocabulary = self._load_semantic_vocabulary()
        self._initialize_patterns()
    
    def _load_semantic_vocabulary(self) -> Dict[str, List[str]]:
        """Load vocabulary for different semantic dimensions"""
        return {
            "temporal_settings": [
                "in distant futures", "during ancient times", "in the present moment",
                "across parallel timelines", "in forgotten eras", "during transformation periods",
                "in moments of change", "through eternal cycles", "in fleeting instances"
            ],
            "spatial_settings": [
                "on alien worlds", "in hidden realms", "within urban landscapes", 
                "across vast distances", "in intimate spaces", "through digital domains",
                "in natural environments", "within constructed realities", "between dimensions"
            ],
            "emotional_tones": [
                "with profound melancholy", "through joyous celebration", "amid tense anticipation",
                "in peaceful contemplation", "with fierce determination", "through gentle compassion",
                "amid chaotic energy", "in solemn reflection", "with playful whimsy"
            ],
            "genre_markers": [
                "where magic flows like water", "as technology shapes destiny",
                "while mysteries unfold slowly", "where love conquers all",
                "as darkness threatens light", "where science reveals truth",
                "while heroes rise to challenge", "as ordinary becomes extraordinary"
            ],
            "agency_descriptors": [
                "characters forge their own paths", "fate guides every step",
                "choices ripple through consequence", "external forces drive change",
                "willpower shapes reality", "circumstances determine outcomes",
                "individuals transcend limitations", "systems constrain possibilities"
            ],
            "social_dynamics": [
                "communities unite in purpose", "isolation defines the journey",
                "relationships form the foundation", "conflict drives the narrative",
                "cooperation builds bridges", "competition fuels progress",
                "traditions anchor identity", "change challenges conventions"
            ]
        }
    
    def _initialize_patterns(self):
        """Initialize text generation patterns based on rho properties"""
        
        # High purity patterns (focused, clear narratives)
        self.patterns.append(RhoTextPattern(
            name="crystalline_narrative",
            description="High purity rho states generate focused, clear narratives",
            rho_condition=lambda rho: self._get_purity(rho) > 0.7,
            text_templates=[
                "With singular focus, {narrative_core}. {temporal_setting}, {spatial_setting}, {emotional_tone}.",
                "{narrative_core} emerges with crystalline clarity. {genre_marker}, {agency_descriptor}.",
                "The essence distills to this: {narrative_core}. {social_dynamic}, {temporal_setting}."
            ],
            semantic_extractors={
                "narrative_core": self._extract_primary_theme,
                "temporal_setting": self._extract_temporal_setting,
                "spatial_setting": self._extract_spatial_setting,
                "emotional_tone": self._extract_emotional_tone,
                "genre_marker": self._extract_genre_marker,
                "agency_descriptor": self._extract_agency_level,
                "social_dynamic": self._extract_social_dynamic
            }
        ))
        
        # High entropy patterns (complex, layered narratives)
        self.patterns.append(RhoTextPattern(
            name="complex_narrative", 
            description="High entropy rho states generate complex, layered narratives",
            rho_condition=lambda rho: self._get_entropy(rho) > 2.0,
            text_templates=[
                "Multiple threads weave together: {theme_1}, yet also {theme_2}, while {theme_3} emerges beneath. {temporal_setting}, {emotional_tone}.",
                "Layers upon layers reveal themselves. {narrative_core} intertwines with {secondary_theme}, {spatial_setting}. {social_dynamic}, {genre_marker}.",
                "The story branches in many directions. {theme_1} leads to {theme_2}, which transforms into {theme_3}. {agency_descriptor}, {temporal_setting}."
            ],
            semantic_extractors={
                "narrative_core": self._extract_primary_theme,
                "secondary_theme": self._extract_secondary_theme,
                "theme_1": self._extract_theme_variation,
                "theme_2": self._extract_theme_variation,
                "theme_3": self._extract_theme_variation,
                "temporal_setting": self._extract_temporal_setting,
                "spatial_setting": self._extract_spatial_setting,
                "emotional_tone": self._extract_emotional_tone,
                "genre_marker": self._extract_genre_marker,
                "agency_descriptor": self._extract_agency_level,
                "social_dynamic": self._extract_social_dynamic
            }
        ))
        
        # Balanced patterns (moderate complexity)
        self.patterns.append(RhoTextPattern(
            name="balanced_narrative",
            description="Balanced rho states generate well-structured narratives",
            rho_condition=lambda rho: 0.3 <= self._get_purity(rho) <= 0.7 and 1.0 <= self._get_entropy(rho) <= 2.0,
            text_templates=[
                "{narrative_core} unfolds {temporal_setting}. {spatial_setting}, {emotional_tone}, {genre_marker}.",
                "The story develops as {narrative_core} meets {secondary_theme}. {agency_descriptor}, {social_dynamic}.",
                "{emotional_tone}, {narrative_core} emerges. {temporal_setting}, {spatial_setting}, {genre_marker}."
            ],
            semantic_extractors={
                "narrative_core": self._extract_primary_theme,
                "secondary_theme": self._extract_secondary_theme,
                "temporal_setting": self._extract_temporal_setting,
                "spatial_setting": self._extract_spatial_setting,
                "emotional_tone": self._extract_emotional_tone,
                "genre_marker": self._extract_genre_marker,
                "agency_descriptor": self._extract_agency_level,
                "social_dynamic": self._extract_social_dynamic
            }
        ))
        
        # Transformation patterns (for actively transformed rho)
        self.patterns.append(RhoTextPattern(
            name="transformation_narrative",
            description="Recently transformed rho states show transition markers",
            rho_condition=lambda rho: self._has_transformation_signature(rho),
            text_templates=[
                "As {original_aspect} transforms into {new_aspect}, {narrative_core} emerges. {temporal_setting}, {emotional_tone}.",
                "The shift from {original_aspect} to {new_aspect} reveals {narrative_core}. {spatial_setting}, {genre_marker}.",
                "Through transformation, {narrative_core} evolves. {original_aspect} becomes {new_aspect}, {agency_descriptor}."
            ],
            semantic_extractors={
                "narrative_core": self._extract_primary_theme,
                "original_aspect": self._extract_source_signature,
                "new_aspect": self._extract_target_signature,
                "temporal_setting": self._extract_temporal_setting,
                "spatial_setting": self._extract_spatial_setting,
                "emotional_tone": self._extract_emotional_tone,
                "genre_marker": self._extract_genre_marker,
                "agency_descriptor": self._extract_agency_level
            }
        ))
    
    def generate_text(self, rho: np.ndarray, original_text: str = "", context: Dict[str, Any] = None) -> str:
        """Generate text from rho matrix using semantic patterns"""
        
        context = context or {}
        
        # If we have original text, use content-preserving transformation
        if original_text and len(original_text.strip()) > 20:
            return self._transform_preserving_content(rho, original_text, context)
        
        # Otherwise use template-based generation
        return self._generate_from_templates(rho, original_text, context)
    
    def _transform_preserving_content(self, rho: np.ndarray, original_text: str, context: Dict[str, Any] = None) -> str:
        """Transform text while preserving its core content and structure"""
        
        # Analyze rho properties to determine transformation approach
        purity = self._get_purity(rho)
        entropy = self._get_entropy(rho)
        
        # Break text into sentences for transformation
        sentences = self._split_into_sentences(original_text)
        transformed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                transformed_sentences.append(sentence)
                continue
                
            # Apply semantic transformations based on rho properties
            transformed_sentence = self._transform_sentence(sentence, rho, i, len(sentences))
            transformed_sentences.append(transformed_sentence)
        
        # Reassemble with preserved structure
        result = ' '.join(transformed_sentences)
        
        # Apply global transformations based on rho state
        result = self._apply_global_transformations(result, rho, original_text)
        
        return result
    
    def _transform_sentence(self, sentence: str, rho: np.ndarray, sentence_idx: int, total_sentences: int) -> str:
        """Transform a single sentence based on rho properties"""
        
        # Get transformation strength based on rho properties
        eigenvals = np.linalg.eigvals(rho)
        dominant_dims = np.argsort(eigenvals)[-3:][::-1]
        
        # Apply dimensional transformations
        transformed = sentence
        
        # Temporal transformation (dimension 0)
        if len(eigenvals) > 0 and eigenvals[0] > 0.1:
            transformed = self._apply_temporal_transformation(transformed, eigenvals[0])
        
        # Spatial transformation (dimension 1) 
        if len(eigenvals) > 1 and eigenvals[1] > 0.1:
            transformed = self._apply_spatial_transformation(transformed, eigenvals[1])
        
        # Genre transformation (dimension 2)
        if len(eigenvals) > 2 and eigenvals[2] > 0.1:
            transformed = self._apply_genre_transformation(transformed, eigenvals[2])
        
        # Emotional transformation (dimension 3)
        if len(eigenvals) > 3 and eigenvals[3] > 0.1:
            transformed = self._apply_emotional_transformation(transformed, eigenvals[3])
        
        return transformed
    
    def _apply_temporal_transformation(self, text: str, strength: float) -> str:
        """Apply temporal setting transformations while preserving meaning"""
        
        # Light temporal hints without changing core content
        if strength > 0.2:
            # Add temporal context markers - be more careful with word boundaries
            if " is " in text and not text.startswith("In ages") and "has always been" not in text:
                text = text.replace(" is ", " has always been ", 1)
            elif " are " in text and not text.startswith("In times") and "have long been" not in text:
                text = text.replace(" are ", " have long been ", 1)
            elif text.endswith(" is") and "has always been" not in text:
                text = text[:-3] + " has always been"
            elif text.endswith(" are") and "have long been" not in text:
                text = text[:-4] + " have long been"
        
        return text
    
    def _apply_spatial_transformation(self, text: str, strength: float) -> str:
        """Apply spatial transformations while preserving content"""
        
        # Subtle spatial context without losing meaning
        if strength > 0.2:
            # Add spatial depth markers - be more careful with replacement
            if " the " in text and "far-reaching" not in text:
                text = text.replace(" the ", " the far-reaching ", 1)
        
        return text
    
    def _apply_genre_transformation(self, text: str, strength: float) -> str:
        """Apply genre transformations while preserving philosophical content"""
        
        if strength > 0.2:
            # Add mystical/fantastical elements to abstract concepts
            replacements = {
                "consciousness": "mystical awareness",
                "language": "ancient word-craft", 
                "mind": "inner realm",
                "thoughts": "mental emanations",
                "ideas": "conceptual energies",
                "understanding": "deeper knowing"
            }
            
            for original, replacement in replacements.items():
                if original in text.lower():
                    # Only replace first occurrence to avoid over-transformation
                    idx = text.lower().find(original)
                    if idx != -1:
                        text = text[:idx] + replacement + text[idx + len(original):]
                        break
        
        return text
    
    def _apply_emotional_transformation(self, text: str, strength: float) -> str:
        """Apply emotional tone transformations"""
        
        if strength > 0.2:
            # Add emotional depth markers
            if "problem" in text:
                text = text.replace("problem", "troubling dilemma")
            elif "truth" in text:
                text = text.replace("truth", "profound truth")
            elif "reality" in text:
                text = text.replace("reality", "deeper reality")
        
        return text
    
    def _apply_global_transformations(self, text: str, rho: np.ndarray, original_text: str) -> str:
        """Apply global transformations that affect the entire text"""
        
        purity = self._get_purity(rho)
        entropy = self._get_entropy(rho)
        
        # Add framing based on rho properties
        if purity > 0.7:
            # High purity: add clarity markers
            if not text.startswith("With crystalline clarity"):
                text = f"With crystalline clarity, {text[0].lower() + text[1:]}"
        elif entropy > 2.0:
            # High entropy: add complexity markers  
            if not text.startswith("Through layered complexity"):
                text = f"Through layered complexity, {text[0].lower() + text[1:]}"
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure"""
        
        # Simple sentence splitting that preserves meaning units
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_from_templates(self, rho: np.ndarray, original_text: str, context: Dict[str, Any] = None) -> str:
        """Generate text using templates (fallback for when no original text)"""
        
        context = context or {}
        
        # Find matching pattern
        matching_pattern = None
        for pattern in self.patterns:
            if pattern.rho_condition(rho):
                matching_pattern = pattern
                break
        
        # Fallback to balanced pattern if no match
        if not matching_pattern:
            matching_pattern = next(p for p in self.patterns if p.name == "balanced_narrative")
        
        logger.info(f"Using pattern: {matching_pattern.name}")
        
        # Extract semantic content
        extracted_content = {}
        for placeholder, extractor in matching_pattern.semantic_extractors.items():
            try:
                extracted_content[placeholder] = extractor(rho, original_text, context)
            except Exception as e:
                logger.warning(f"Failed to extract {placeholder}: {e}")
                extracted_content[placeholder] = f"[{placeholder}]"
        
        # Choose template based on rho properties
        template_index = self._select_template_index(rho, len(matching_pattern.text_templates))
        template = matching_pattern.text_templates[template_index]
        
        # Fill template
        try:
            generated_text = template.format(**extracted_content)
        except KeyError as e:
            logger.warning(f"Missing placeholder {e} in template")
            # Fallback with available content
            available_keys = set(extracted_content.keys())
            template_keys = set(re.findall(r'{(\w+)}', template))
            missing_keys = template_keys - available_keys
            
            for key in missing_keys:
                extracted_content[key] = f"[{key}]"
            
            generated_text = template.format(**extracted_content)
        
        # Post-process for coherence
        generated_text = self._post_process_text(generated_text, rho, original_text)
        
        return generated_text
    
    def _get_purity(self, rho: np.ndarray) -> float:
        """Calculate purity of rho matrix (Tr(ρ²))"""
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = np.real(eigenvals)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        return np.sum(eigenvals**2)
    
    def _get_entropy(self, rho: np.ndarray) -> float:
        """Calculate von Neumann entropy of rho matrix"""
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = np.real(eigenvals)
        eigenvals = np.maximum(eigenvals, 1e-12)  # Avoid log(0)
        return -np.sum(eigenvals * np.log(eigenvals))
    
    def _has_transformation_signature(self, rho: np.ndarray) -> bool:
        """Detect if rho shows signs of recent transformation"""
        # Look for non-diagonal dominant structure (indicates transformation)
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        # Check if dominant eigenvector has significant off-diagonal correlations
        dominant_idx = np.argmax(eigenvals)
        dominant_vec = eigenvecs[:, dominant_idx]
        
        # Simple measure: variance in eigenvector components
        vec_variance = np.var(np.real(dominant_vec))
        return vec_variance > 0.1  # Threshold for "structured" eigenvector
    
    def _extract_primary_theme(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract the primary thematic content from rho matrix"""
        
        # If we have space design, use axis projections
        if self.space_design and hasattr(self, '_axis_to_theme_mapping'):
            return self._extract_via_axes(rho, "primary_theme")
        
        # Fallback: analyze eigenstructure
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        dominant_idx = np.argmax(eigenvals)
        
        # Map dominant dimension to theme based on its properties
        if dominant_idx < 8:  # First 8 dimensions are narrative axes
            themes = [
                "temporal transitions shape the narrative",
                "spatial boundaries define the story",
                "genre elements color the experience", 
                "emotional undercurrents drive the plot",
                "character agency determines outcomes",
                "social forces influence direction",
                "narrative distance creates perspective",
                "semantic density enriches meaning"
            ]
            return themes[dominant_idx]
        else:
            return "emergent themes weave through the narrative"
    
    def _extract_secondary_theme(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract secondary thematic content"""
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        # Find second most dominant dimension
        second_idx = np.argsort(eigenvals)[-2]
        
        if second_idx < 8:
            themes = [
                "temporal echoes",
                "spatial resonances", 
                "genre transformations",
                "emotional depths",
                "character growth",
                "social dynamics",
                "narrative layers",
                "semantic richness"
            ]
            return themes[second_idx]
        else:
            return "subtle undercurrents"
    
    def _extract_theme_variation(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract thematic variations for complex narratives"""
        # Use different dimensions for variety
        eigenvals = np.linalg.eigvals(rho)
        
        # Select based on matrix properties
        purity = self._get_purity(rho)
        entropy = self._get_entropy(rho)
        
        variations = [
            "transformative journeys",
            "hidden connections",
            "emerging possibilities", 
            "deepening mysteries",
            "evolving relationships",
            "shifting perspectives",
            "unfolding revelations",
            "interweaving destinies"
        ]
        
        # Use matrix properties to select variation
        idx = int((purity + entropy) * len(variations) / 4) % len(variations)
        return variations[idx]
    
    def _extract_temporal_setting(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract temporal setting information"""
        # Use first dimension (temporal_setting axis)
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        if len(eigenvals) > 0:
            temporal_strength = eigenvals[0] if len(eigenvals) > 0 else 0.5
            vocab = self.semantic_vocabulary["temporal_settings"]
            idx = int(temporal_strength * len(vocab)) % len(vocab)
            return vocab[idx]
        
        return "in times both ancient and new"
    
    def _extract_spatial_setting(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract spatial setting information"""
        # Use second dimension (spatial_setting axis) 
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        if len(eigenvals) > 1:
            spatial_strength = eigenvals[1]
            vocab = self.semantic_vocabulary["spatial_settings"]
            idx = int(spatial_strength * len(vocab)) % len(vocab)
            return vocab[idx]
        
        return "across landscapes both real and imagined"
    
    def _extract_emotional_tone(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract emotional tone"""
        # Use fourth dimension (emotional_tone axis)
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        if len(eigenvals) > 3:
            emotion_strength = eigenvals[3]
            vocab = self.semantic_vocabulary["emotional_tones"]
            idx = int(emotion_strength * len(vocab)) % len(vocab)
            return vocab[idx]
        
        return "with emotions running deep"
    
    def _extract_genre_marker(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract genre markers"""
        # Use third dimension (genre_flavor axis)
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        if len(eigenvals) > 2:
            genre_strength = eigenvals[2]
            vocab = self.semantic_vocabulary["genre_markers"]
            idx = int(genre_strength * len(vocab)) % len(vocab)
            return vocab[idx]
        
        return "where stories take unexpected forms"
    
    def _extract_agency_level(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract character agency level"""
        # Use fifth dimension (narrative_agency axis)
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        if len(eigenvals) > 4:
            agency_strength = eigenvals[4]
            vocab = self.semantic_vocabulary["agency_descriptors"]
            idx = int(agency_strength * len(vocab)) % len(vocab)
            return vocab[idx]
        
        return "where choices echo through time"
    
    def _extract_social_dynamic(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract social dynamics"""
        # Use sixth dimension (social_dynamics axis)
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        if len(eigenvals) > 5:
            social_strength = eigenvals[5]
            vocab = self.semantic_vocabulary["social_dynamics"]
            idx = int(social_strength * len(vocab)) % len(vocab)
            return vocab[idx]
        
        return "where connections shape destiny"
    
    def _extract_source_signature(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract signature of transformation source"""
        # Analyze lower eigenvalues (residual from original state)
        eigenvals = np.sort(np.linalg.eigvals(rho))
        
        if len(eigenvals) > 2:
            source_signature = eigenvals[len(eigenvals)//4]  # Lower quartile
            if source_signature > 0.1:
                return "familiar foundations"
            else:
                return "distant origins"
        
        return "previous forms"
    
    def _extract_target_signature(self, rho: np.ndarray, original_text: str = "", context: Dict = None) -> str:
        """Extract signature of transformation target"""
        # Analyze higher eigenvalues (dominant in transformed state)
        eigenvals = np.sort(np.linalg.eigvals(rho))
        
        if len(eigenvals) > 2:
            target_signature = eigenvals[-len(eigenvals)//4]  # Upper quartile
            if target_signature > 0.3:
                return "emergent possibilities"
            else:
                return "subtle changes"
        
        return "new expressions"
    
    def _select_template_index(self, rho: np.ndarray, num_templates: int) -> int:
        """Select template based on rho properties"""
        # Use trace and determinant to select template
        trace = np.trace(rho)
        det = np.linalg.det(rho)
        
        # Combine properties to get template index
        selection_value = (trace + abs(det)) % 1.0
        return int(selection_value * num_templates) % num_templates
    
    def _post_process_text(self, text: str, rho: np.ndarray, original_text: str) -> str:
        """Post-process generated text for coherence and flow"""
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure proper sentence ending
        if text and text[-1] not in '.!?':
            text += "."
        
        # Remove redundant spaces
        text = re.sub(r'\s+', ' ', text)
        
        # If we have original text context, try to maintain some connection
        if original_text:
            # Check if transformation preserves key concepts
            original_words = set(original_text.lower().split())
            if len(original_words) > 0:
                # Add subtle connection if completely disconnected
                generated_words = set(text.lower().split())
                if len(original_words.intersection(generated_words)) == 0:
                    # Add a connecting phrase
                    text = f"Transformed from its origins, {text[0].lower() + text[1:]}"
        
        return text

# Integration functions

def create_rho_text_generator(embed_function, space_design=None) -> RhoTextGenerator:
    """Create a new rho text generator with the given embedding function"""
    return RhoTextGenerator(embed_function, space_design)

def enhance_transformation_engine_with_generation(transformation_engine, space_design=None):
    """Replace the fallback text generation in a transformation engine"""
    
    generator = create_rho_text_generator(transformation_engine.embed, space_design)
    
    # Replace the fallback method
    transformation_engine.rho_to_text = generator.generate_text
    
    logger.info("Enhanced transformation engine with rho-conditioned text generation")
    return transformation_engine

if __name__ == "__main__":
    print("Testing Rho Text Generator...")
    
    # This would need the actual embed function
    # generator = create_rho_text_generator(embed_function)
    # test_rho = np.random.rand(8, 8)
    # test_rho = (test_rho + test_rho.T) / 2  # Make Hermitian
    # test_rho = test_rho / np.trace(test_rho)  # Normalize
    # result = generator.generate_text(test_rho, "Original story about London")
    # print(f"Generated: {result}")