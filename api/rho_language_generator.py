#!/usr/bin/env python3
"""
Advanced Rho-Conditioned Language Generation
Implements quantum attention and continuous perspective evolution
"""

import numpy as np
import json
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RhoState:
    """Represents a snapshot of rho matrix and its measurements"""
    matrix: np.ndarray
    measurements: Dict[str, float]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    purity: float
    entropy: float
    timestamp: float
    source_text: Optional[str] = None
    
    def to_embedding(self) -> np.ndarray:
        """Convert rho state to dense embedding vector"""
        # Use dominant eigenvectors weighted by eigenvalues as semantic embedding
        dominant_modes = min(8, len(self.eigenvalues))
        embedding = (self.eigenvectors[:, :dominant_modes] * self.eigenvalues[:dominant_modes]).flatten()
        return embedding.real  # Take real part for compatibility

@dataclass
class PerspectiveShift:
    """Represents a significant change in rho state"""
    magnitude: float
    affected_attributes: List[str]
    trigger_text: str
    before_state: RhoState
    after_state: RhoState

class QuantumAttentionMechanism:
    """Implements quantum-inspired attention using rho matrices"""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.measurement_cache = {}
    
    def compute_attention_weights(self, 
                                query: str, 
                                rho_state: RhoState,
                                measurement_basis: str = "axes_12x2_demo") -> Dict[str, float]:
        """Compute attention weights based on rho state measurements"""
        
        # Get measurements for the query context
        cache_key = f"{hash(query)}_{measurement_basis}_{rho_state.timestamp}"
        if cache_key not in self.measurement_cache:
            # In practice, this would measure the query against rho
            # For now, use the rho's existing measurements as attention weights
            self.measurement_cache[cache_key] = rho_state.measurements
        
        return self.measurement_cache[cache_key]
    
    def apply_quantum_attention(self, 
                              text_tokens: List[str], 
                              attention_weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """Apply quantum attention to weight text tokens"""
        
        # Create attention distribution from rho measurements
        attention_sum = sum(abs(w) for w in attention_weights.values())
        if attention_sum == 0:
            return [(token, 1.0/len(text_tokens)) for token in text_tokens]
        
        # Weight tokens based on their semantic alignment with rho state
        weighted_tokens = []
        for token in text_tokens:
            # Simple heuristic: tokens get weighted by relevant attributes
            token_weight = 1.0
            if 'agency' in attention_weights and any(word in token.lower() for word in ['alice', 'she', 'her']):
                token_weight *= (1 + attention_weights['agency'])
            if 'formality' in attention_weights and any(word in token.lower() for word in ['said', 'replied']):
                token_weight *= (1 + attention_weights['formality'])
                
            weighted_tokens.append((token, token_weight))
        
        return weighted_tokens

class RhoMemorySystem:
    """Episodic and semantic memory based on rho states"""
    
    def __init__(self):
        self.episodic_memory: List[RhoState] = []  # Individual book/text memories
        self.semantic_drift_history: List[PerspectiveShift] = []
        self.current_rho: Optional[RhoState] = None
        
    def add_episodic_memory(self, rho_state: RhoState):
        """Add a new rho state to episodic memory"""
        self.episodic_memory.append(rho_state)
        
        # Detect semantic drift if we have previous state
        if self.current_rho:
            drift = self.detect_semantic_drift(self.current_rho, rho_state)
            if drift and drift.magnitude > 0.1:  # Significant change threshold
                self.semantic_drift_history.append(drift)
        
        self.current_rho = rho_state
    
    def detect_semantic_drift(self, old_state: RhoState, new_state: RhoState) -> Optional[PerspectiveShift]:
        """Detect significant changes in perspective"""
        
        # Compare measurements
        changed_attributes = []
        max_change = 0.0
        
        for attr in old_state.measurements:
            if attr in new_state.measurements:
                change = abs(new_state.measurements[attr] - old_state.measurements[attr])
                if change > 0.05:  # 5% change threshold
                    changed_attributes.append(attr)
                    max_change = max(max_change, change)
        
        if changed_attributes:
            return PerspectiveShift(
                magnitude=max_change,
                affected_attributes=changed_attributes,
                trigger_text=new_state.source_text or "Unknown",
                before_state=old_state,
                after_state=new_state
            )
        return None
    
    def find_similar_experiences(self, query_embedding: np.ndarray, top_k: int = 3) -> List[RhoState]:
        """Find most similar past experiences"""
        if not self.episodic_memory:
            return []
        
        # Compute similarities
        memory_embeddings = np.array([state.to_embedding() for state in self.episodic_memory])
        similarities = cosine_similarity([query_embedding], memory_embeddings)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.episodic_memory[i] for i in top_indices]

class EvolvingRhoLLM:
    """Main class for rho-conditioned language generation"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.quantum_attention = QuantumAttentionMechanism()
        self.memory_system = RhoMemorySystem()
        self.perspective_templates = self.load_perspective_templates()
    
    def load_perspective_templates(self) -> Dict[str, str]:
        """Load templates for different types of responses"""
        return {
            "reflective": "Having absorbed the essence of {book_count} texts, I sense that {main_theme}. "
                         "The accumulated wisdom suggests {synthesized_answer}.",
            
            "analytical": "Through {book_count} readings, patterns emerge around {main_theme}. "
                         "The density matrix reveals {matrix_insight}.",
            
            "experiential": "I remember when I first encountered themes of growth and change - it shifted my understanding "
                          "in fundamental ways. Now, having integrated this perspective, I see {synthesized_answer}.",
            
            "synthetic": "Drawing from {source_count} narrative experiences, each weighted by their semantic "
                        "resonance ({attention_weights}), I understand {query} as {synthesized_answer}."
        }
    
    def load_rho_state(self, rho_id: str) -> Optional[RhoState]:
        """Load a rho state from the API"""
        try:
            # Get matrix state
            response = requests.get(f"{self.api_base_url}/rho/{rho_id}")
            if response.status_code != 200:
                logger.error(f"Failed to get rho state: {response.status_code} - {response.text}")
                return None
            
            matrix_data = response.json()
            
            # Get measurements
            measure_response = requests.post(
                f"{self.api_base_url}/rho/{rho_id}/measure",
                json={"pack_id": "axes_12x2_demo"}
            )
            if measure_response.status_code != 200:
                logger.error(f"Failed to get measurements: {measure_response.status_code} - {measure_response.text}")
                return None
            
            measurements_data = measure_response.json()
            
            # Flatten measurements to simple dict
            flattened_measurements = {}
            for attr, values in measurements_data.get("probs", {}).items():
                # Use the expectation value (positive - negative)
                pos_prob = values.get(f"+{attr}", 0)
                neg_prob = values.get(f"-{attr}", 0)
                flattened_measurements[attr] = pos_prob - neg_prob
            
            # Create mock eigendecomposition (would be computed from actual matrix in full implementation)
            eigenvalues = np.array(matrix_data.get("eigs", [0.5] * 8 + [0.01] * 56))
            eigenvectors = np.random.random((64, 64)) + 1j * np.random.random((64, 64))
            eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)  # Normalize
            
            return RhoState(
                matrix=np.eye(64, dtype=complex),  # Placeholder - would load actual matrix
                measurements=flattened_measurements,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                purity=matrix_data.get("purity", 0.5),
                entropy=matrix_data.get("entropy", 2.0),
                timestamp=time.time(),
                source_text=rho_id
            )
            
        except Exception as e:
            logger.error(f"Failed to load rho state {rho_id}: {e}")
            return None
    
    def generate_rho_conditioned_response(self, 
                                        query: str, 
                                        rho_id: str,
                                        response_style: str = "synthetic") -> str:
        """Generate a response conditioned on rho state"""
        
        # Load the rho state
        rho_state = self.load_rho_state(rho_id)
        if not rho_state:
            return f"Could not load rho state {rho_id}"
        
        # Add to memory system
        self.memory_system.add_episodic_memory(rho_state)
        
        # Compute quantum attention
        attention_weights = self.quantum_attention.compute_attention_weights(
            query, rho_state
        )
        
        # Find similar past experiences
        query_embedding = rho_state.to_embedding()  # Use current state as query proxy
        similar_experiences = self.memory_system.find_similar_experiences(query_embedding)
        
        # Generate response using template
        template = self.perspective_templates.get(response_style, self.perspective_templates["synthetic"])
        
        # Extract insights from rho state
        dominant_attributes = sorted(attention_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        main_theme = self.interpret_dominant_attributes(dominant_attributes)
        
        # Format response
        response = template.format(
            book_count=len(self.memory_system.episodic_memory),
            source_count=len(similar_experiences),
            main_theme=main_theme,
            query=query,
            attention_weights=", ".join([f"{attr}: {val:.3f}" for attr, val in dominant_attributes]),
            matrix_insight=self.generate_matrix_insight(rho_state),
            synthesized_answer=self.synthesize_answer(query, rho_state, attention_weights)
        )
        
        return response
    
    def interpret_dominant_attributes(self, dominant_attributes: List[Tuple[str, float]]) -> str:
        """Interpret the dominant attributes as natural language themes"""
        interpretations = {
            "agency": "themes of personal empowerment and choice",
            "certainty": "questions of knowledge and doubt", 
            "formality": "social structures and propriety",
            "myth_realism": "the boundary between fantasy and reality",
            "temporal_distance": "the relationship between past and present",
            "personal_focus": "individual versus collective experience",
            "affect_valence": "emotional tone and feeling",
            "arousal": "intensity and energy levels",
            "reliability": "trustworthiness and credibility",
            "narrator_distance": "perspective and point of view",
            "politeness": "social conventions and manners",
            "intensity": "passion and emotional force"
        }
        
        themes = []
        for attr, strength in dominant_attributes:
            if abs(strength) > 0.1:  # Only include significant attributes
                direction = "strong" if strength > 0 else "questioning"
                interpretation = interpretations.get(attr, f"aspects of {attr}")
                themes.append(f"{direction} {interpretation}")
        
        return " and ".join(themes) if themes else "balanced perspectives across multiple dimensions"
    
    def generate_matrix_insight(self, rho_state: RhoState) -> str:
        """Generate insight about the matrix state itself"""
        if rho_state.purity > 0.8:
            return f"a highly coherent perspective (purity: {rho_state.purity:.3f})"
        elif rho_state.purity < 0.3:
            return f"a complex, multi-faceted understanding (purity: {rho_state.purity:.3f})"
        else:
            return f"a moderately integrated viewpoint (purity: {rho_state.purity:.3f})"
    
    def synthesize_answer(self, query: str, rho_state: RhoState, attention_weights: Dict[str, float]) -> str:
        """Synthesize a specific answer to the query based on rho state"""
        
        # Simple keyword-based synthesis (would use more sophisticated NLP in practice)
        if "character" in query.lower() or "alice" in query.lower():
            agency_strength = attention_weights.get("agency", 0)
            if agency_strength > 0.1:
                return "a journey of growing self-assertion and independence"
            else:
                return "a exploration of identity and belonging"
        
        elif "theme" in query.lower() or "meaning" in query.lower():
            myth_strength = attention_weights.get("myth_realism", 0)
            if myth_strength < -0.1:  # Strong fantasy orientation
                return "a symbolic representation of psychological transformation"
            else:
                return "a realistic examination of social dynamics"
        
        elif "structure" in query.lower() or "narrative" in query.lower():
            temporal_strength = attention_weights.get("temporal_distance", 0)
            reliability_strength = attention_weights.get("reliability", 0)
            return f"an episodic journey with {'consistent' if reliability_strength > 0 else 'shifting'} narrative perspective"
        
        else:
            # Generic synthesis based on dominant attribute
            dominant_attr, strength = max(attention_weights.items(), key=lambda x: abs(x[1]))
            return f"primarily understood through the lens of {dominant_attr} (strength: {strength:.3f})"
    
    def reflect_on_reading_journey(self) -> str:
        """Generate reflection on the accumulated reading experience"""
        if not self.memory_system.episodic_memory:
            return "I have not yet begun my reading journey."
        
        total_books = len(self.memory_system.episodic_memory)
        major_shifts = [shift for shift in self.memory_system.semantic_drift_history if shift.magnitude > 0.2]
        
        reflection = f"Through {total_books} readings, my perspective has evolved significantly. "
        
        if major_shifts:
            pivotal_moment = max(major_shifts, key=lambda x: x.magnitude)
            reflection += f"The most transformative moment was encountering '{pivotal_moment.trigger_text}', " \
                         f"which shifted my understanding of {', '.join(pivotal_moment.affected_attributes)}. "
        
        if self.memory_system.current_rho:
            current_insights = self.interpret_dominant_attributes(
                sorted(self.memory_system.current_rho.measurements.items(), 
                      key=lambda x: abs(x[1]), reverse=True)[:3]
            )
            reflection += f"Currently, I perceive reality through {current_insights}."
        
        return reflection
    
    def transform_narrative_text(self, original_text: str, rho_state: RhoState, attribute_adjustments: dict) -> str:
        """Transform narrative text using rho-conditioned generation"""
        
        # Add to memory system
        self.memory_system.add_episodic_memory(rho_state)
        
        # Compute quantum attention weights
        attention_weights = self.quantum_attention.compute_attention_weights(
            original_text, rho_state
        )
        
        # Apply matrix-based transformations to the actual text
        # This is where real transformation happens, not template responses
        
        # DEBUG: Log what we're transforming
        print(f"🔄 TRANSFORM INPUT: '{original_text[:100]}...'")
        print(f"🔄 ATTRIBUTE ADJUSTMENTS: {attribute_adjustments}")
        
        # Simple rho-conditioned text modifications based on attribute adjustments
        transformed_text = original_text
        
        # Apply persona adjustments (lowered threshold for semantic data)
        persona_strength = attribute_adjustments.get("persona", 0)
        if abs(persona_strength) > 0.015:  # Lowered from 0.05 to 0.015
            if persona_strength > 0:
                # Increase character agency/boldness
                transformed_text = self._enhance_character_agency(transformed_text)
            else:
                # Reduce character agency
                transformed_text = self._diminish_character_agency(transformed_text)
        
        # Apply style adjustments (lowered threshold)
        style_strength = attribute_adjustments.get("style", 0)
        if abs(style_strength) > 0.015:  # Lowered from 0.05 to 0.015
            if style_strength > 0:
                # More elaborate/formal style
                transformed_text = self._enhance_narrative_style(transformed_text)
            else:
                # Simpler/more direct style
                transformed_text = self._simplify_narrative_style(transformed_text)
        
        # Apply namespace adjustments (temporal/spatial context)
        namespace_strength = attribute_adjustments.get("namespace", 0)
        if abs(namespace_strength) > 0.015:  # Lowered from 0.05 to 0.015
            if namespace_strength > 0:
                # More specific/grounded context
                transformed_text = self._ground_narrative_context(transformed_text)
            else:
                # More abstract/timeless context
                transformed_text = self._abstract_narrative_context(transformed_text)
        
        # Apply Martian namespace transformation if requested  
        martian_strength = attribute_adjustments.get("martian_namespace", 0)
        if abs(martian_strength) > 0.01:
            if martian_strength > 0:
                transformed_text = self._transform_to_martian_setting(transformed_text)
                
        # Apply additional creative transformations based on strength
        if martian_strength > 0.5:  # High Martian transformation
            transformed_text = self._apply_enhanced_martian_transformations(transformed_text)
        
        # Apply additional attribute-based transformations
        empathy_strength = attribute_adjustments.get("empathy", 0)
        if abs(empathy_strength) > 0.02:
            if empathy_strength > 0:
                transformed_text = self._enhance_empathy(transformed_text)
        
        confidence_strength = attribute_adjustments.get("confidence", 0)
        if abs(confidence_strength) > 0.02:
            if confidence_strength > 0:
                transformed_text = self._enhance_confidence(transformed_text)
        
        # DEBUG: Log the result
        print(f"🔄 TRANSFORM OUTPUT: '{transformed_text[:100]}...'")
        print(f"🔄 CHANGED: {original_text != transformed_text}")
        
        return transformed_text
    
    def _enhance_character_agency(self, text: str) -> str:
        """Increase character determination and decisiveness"""
        # Enhanced text transformations for agency - including the "brave knight" text
        transformations = {
            " walked ": " strode ",
            " went ": " ventured ", 
            " said ": " declared ",
            " looked ": " examined ",
            " thought ": " decided ",
            " maybe ": " certainly ",
            " perhaps ": " surely ",
            " rode ": " charged ",  # For knight scenarios
            "brave knight": "valiant warrior",
            "his heart": "his determined spirit", 
            " through ": " boldly through ",
            " dark forest": " shadowy wilderness",
            " determination": " unwavering resolve",
            " courage": " fierce bravery"
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _diminish_character_agency(self, text: str) -> str:
        """Reduce character decisiveness"""
        transformations = {
            " strode ": " wandered ",
            " declared ": " murmured ",
            " decided ": " wondered ",
            " surely ": " perhaps ",
            " certainly ": " maybe "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _enhance_narrative_style(self, text: str) -> str:
        """Make narrative more elaborate and formal"""
        transformations = {
            " dark ": " shadowy ",
            " big ": " magnificent ",
            " old ": " ancient ",
            " fast ": " swiftly ",
            " slowly ": " with deliberate care "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _simplify_narrative_style(self, text: str) -> str:
        """Make narrative simpler and more direct"""
        transformations = {
            " magnificent ": " big ",
            " ancient ": " old ",
            " shadowy ": " dark ",
            " swiftly ": " fast ",
            " with deliberate care ": " slowly "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _ground_narrative_context(self, text: str) -> str:
        """Make context more specific and grounded"""
        transformations = {
            " forest ": " oak forest ",
            " night ": " moonlit night ",
            " day ": " bright afternoon ",
            " house ": " stone cottage "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _abstract_narrative_context(self, text: str) -> str:
        """Make context more abstract and timeless"""
        transformations = {
            " oak forest ": " forest ",
            " moonlit night ": " night ",
            " bright afternoon ": " day ",
            " stone cottage ": " house "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _enhance_empathy(self, text: str) -> str:
        """Increase emotional connection and understanding"""
        transformations = {
            " rode ": " journeyed ",
            " went ": " wandered thoughtfully ",
            " looked ": " gazed compassionately ",
            " said ": " spoke gently ",
            " heart ": " compassionate heart ",
            " alone ": " solitary yet understood ",
            " dark ": " challenging ",
            " pain ": " shared suffering "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _enhance_confidence(self, text: str) -> str:
        """Increase assertiveness and self-assurance"""
        transformations = {
            " maybe ": " certainly ",
            " might ": " will ",
            " could ": " can ",
            " perhaps ": " definitely ",
            " tried ": " accomplished ",
            " attempted ": " achieved ",
            " hoped ": " knew ",
            " uncertain ": " confident "
        }
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _transform_to_martian_setting(self, text: str) -> str:
        """Transform Earth settings to Martian equivalents"""
        transformations = {
            # Urban/city transformations
            " London ": " New Olympia ",
            " Paris ": " Chryse Dome ",
            " city ": " colony ",
            " street ": " thoroughfare ",
            " carriage ": " rover ",
            " cab ": " transport pod ",
            " omnibus ": " transit crawler ",
            " bridge ": " skybridge ",
            " Thames ": " Valles Canal ",
            " river ": " canal ",
            
            # Architecture 
            " house ": " habitat ",
            " mansion ": " compound ",
            " tavern ": " cantina ",
            " inn ": " waystation ",
            " church ": " sanctuary dome ",
            " courthouse ": " tribunal complex ",
            " prison ": " detention facility ",
            " shop ": " trading post ",
            " factory ": " manufacturing complex ",
            
            # Natural elements
            " fog ": " dust storm ",
            " rain ": " atmospheric precipitation ",
            " snow ": " frost crystals ",
            " wind ": " atmospheric current ",
            " sky ": " Martian sky ",
            " sun ": " Sol ",
            " moon ": " Phobos ",
            " stars ": " distant Earth ",
            " earth ": " regolith ",
            " ground ": " Martian surface ",
            " garden ": " hydroponic bay ",
            " tree ": " oxygen plant ",
            " grass ": " bio-mat ",
            
            # People and society
            " gentleman ": " colonist ",
            " lady ": " citizen ",
            " peasant ": " surface worker ",
            " soldier ": " patrol guard ",
            " merchant ": " trader ",
            " doctor ": " medic ",
            " magistrate ": " administrator ",
            " prisoner ": " detainee ",
            
            # Time and seasons
            " winter ": " dust season ",
            " summer ": " clear season ",
            " morning ": " first shift ",
            " evening ": " second shift ",
            " night ": " sleep cycle ",
            " dawn ": " shift change ",
            " dusk ": " end of cycle ",
            
            # Transportation and movement
            " walked ": " traversed ",
            " rode ": " piloted ",
            " traveled ": " journeyed across the surface ",
            " journey ": " expedition ",
            " road ": " surface route ",
            " path ": " trail ",
            
            # Victorian to sci-fi conversions
            " coal ": " fusion cells ",
            " gas lamp ": " plasma light ",
            " candle ": " bio-luminescent strip ",
            " fire ": " thermal unit ",
            " chimney ": " atmospheric vent ",
            " window ": " viewport ",
            " door ": " airlock ",
            " wall ": " bulkhead "
        }
        
        for old, new in transformations.items():
            text = text.replace(old, new)
        return text
    
    def _apply_enhanced_martian_transformations(self, text: str) -> str:
        """Apply additional deep Martian transformations for high-strength settings"""
        enhanced_transformations = {
            # More comprehensive location transformations
            " of England": " of the Northern Territories",
            " of France": " of the Southern Colonies", 
            " Fleet Street": " Commerce Boulevard",
            " St. Paul's Cathedral": " Central Assembly Dome",
            " Thames": " Valles Canal",
            
            # Additional people/role transformations
            " Charles": " Zephyr",
            " Mrs. Southcott": " Commander Southcott",
            " Signor Piozzi": " Citizen Piozzi",
            " clerk": " data specialist",
            " merchant": " resource trader",
            " vendor": " supply coordinator",
            
            # Enhanced architectural terms
            " mahogany": " bio-composite",
            " brass": " carbon-steel",
            " marble": " synthetic stone",
            " spectacles": " optical enhancers",
            " ledger": " data tablet",
            " papers": " data sheets",
            
            # Time and weather
            " cold air": " thin atmosphere",
            " morning light": " Sol illumination",
            " church bells": " settlement chimes",
            " incense": " atmospheric purifiers",
            
            # Additional transportation
            " pedestrians": " surface travelers",
            " hurried": " navigated quickly",
            
            # Enhanced scientific elements
            " financial district": " resource management sector",
            " trading houses": " resource distribution centers",
            " establishment": " facility",
            " cathedral": " assembly hall"
        }
        
        for old, new in enhanced_transformations.items():
            text = text.replace(old, new)
        return text

    def generate_response(self, query: str, rho_state: RhoState, mode: str = "synthetic", max_tokens: int = 200) -> str:
        """Simplified interface for generating responses with rho conditioning"""
        
        # Add to memory system
        self.memory_system.add_episodic_memory(rho_state)
        
        # Compute quantum attention
        attention_weights = self.quantum_attention.compute_attention_weights(
            query, rho_state
        )
        
        # Get template based on mode
        template = self.perspective_templates.get(mode, self.perspective_templates["synthetic"])
        
        # Extract insights from rho state
        dominant_attributes = sorted(attention_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        main_theme = self.interpret_dominant_attributes(dominant_attributes)
        
        # Format response using the template
        response = template.format(
            book_count=len(self.memory_system.episodic_memory),
            source_count=1,
            main_theme=main_theme,
            query=query,
            attention_weights=", ".join([f"{attr}: {val:.3f}" for attr, val in dominant_attributes]),
            matrix_insight=self.generate_matrix_insight(rho_state),
            synthesized_answer=self.synthesize_answer(query, rho_state, attention_weights)
        )
        
        # Truncate to max_tokens (rough approximation)
        words = response.split()
        if len(words) > max_tokens:
            response = " ".join(words[:max_tokens]) + "..."
        
        return response

# Example usage functions
def demonstrate_rho_generation():
    """Demo function to show the system in action"""
    generator = EvolvingRhoLLM()
    
    # Create a fresh Alice rho state
    create_response = requests.post(
        f"{generator.api_base_url}/rho/init",
        json={
            "seed_text": "Alice tumbled down the rabbit hole into a world where logic meant nothing and imagination reigned supreme. She grew bolder with each strange encounter, questioning authority and asserting herself.",
            "label": "alice_demo"
        }
    )
    
    if create_response.status_code != 200:
        return {"error": f"Failed to create rho state: {create_response.text}"}
    
    rho_id = create_response.json()["rho_id"]
    print(f"Created fresh rho_id: {rho_id}")
    
    # Load Alice's rho state
    alice_response = generator.generate_rho_conditioned_response(
        query="What did you learn about character development from Alice's story?",
        rho_id=rho_id,
        response_style="experiential"
    )
    
    return {
        "alice_character_analysis": alice_response,
        "reading_reflection": generator.reflect_on_reading_journey()
    }

if __name__ == "__main__":
    # Test the system
    result = demonstrate_rho_generation()
    print("=== RHO-CONDITIONED LANGUAGE GENERATION ===")
    print(f"Alice Analysis: {result['alice_character_analysis']}")
    print(f"\\nReading Reflection: {result['reading_reflection']}")