"""
Narrative Attribute Algebra System

A comprehensive framework for extracting, manipulating, and applying narrative attributes
through quantum density matrix operations. Organizes transformations into three fundamental
dimensions: Namespace, Style, and Persona.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AttributeVector:
    """A vector in ρ-space representing a specific narrative attribute"""
    name: str
    category: str  # "namespace", "style", or "persona"
    vector: np.ndarray
    strength: float = 1.0
    description: str = ""
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []

@dataclass
class NarrativeSignature:
    """Complete narrative signature with three attribute dimensions"""
    namespace: Dict[str, float]  # {attribute_name: weight}
    style: Dict[str, float]
    persona: Dict[str, float]
    pure_essence: Optional[np.ndarray] = None  # Content vector independent of attributes
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class NarrativeAttributeTaxonomy:
    """Defines the complete taxonomy of narrative attributes"""
    
    NAMESPACE_ATTRIBUTES = {
        # Temporal/Historical
        "contemporary": "Modern, current-day settings and references",
        "historical": "Past eras, historical periods, vintage contexts",
        "futuristic": "Advanced technology, space-age, cyberpunk elements",
        "timeless": "No specific temporal markers, universal themes",
        
        # Geographic/Cultural
        "urban": "City environments, metropolitan settings",
        "rural": "Countryside, natural environments, pastoral",
        "cosmic": "Space, alien worlds, galactic contexts",
        "domestic": "Home, family, intimate personal spaces",
        
        # Reality Level
        "realistic": "Grounded in everyday reality",
        "fantasy": "Magical, supernatural, mystical elements",
        "surreal": "Dream-like, absurdist, reality-bending",
        "mythological": "Archetypal, legendary, epic frameworks",
        
        # Social Context
        "professional": "Workplace, business, institutional",
        "academic": "Educational, scholarly, intellectual",
        "criminal": "Underworld, noir, detective contexts",
        "military": "Armed forces, conflict, strategic"
    }
    
    STYLE_ATTRIBUTES = {
        # Register & Formality
        "formal": "Sophisticated language, ceremonial tone",
        "informal": "Casual, conversational, relaxed",
        "academic": "Scholarly, precise, technical",
        "colloquial": "Everyday speech, idiomatic expressions",
        
        # Elaboration & Density
        "elaborate": "Rich detail, extensive description",
        "sparse": "Minimal, essential, stripped-down",
        "dense": "Information-packed, complex structure",
        "flowing": "Smooth, easy, natural progression",
        
        # Emotional Register
        "affective": "Emotionally charged, evaluative",
        "neutral": "Objective, detached, factual",
        "passionate": "Intense, dramatic, heightened",
        "subtle": "Understated, nuanced, restrained",
        
        # Narrative Technique
        "immediate": "Present-tense, urgent, in-the-moment",
        "reflective": "Contemplative, retrospective, thoughtful",
        "dramatic": "High tension, conflict-driven",
        "meditative": "Peaceful, introspective, philosophical"
    }
    
    PERSONA_ATTRIBUTES = {
        # Narrator Reliability
        "reliable": "Trustworthy, authoritative narrator",
        "unreliable": "Questionable, biased, subjective narrator",
        "omniscient": "All-knowing, god-like perspective",
        "limited": "Restricted knowledge, human perspective",
        
        # Emotional Stance
        "optimistic": "Hopeful, positive outlook",
        "pessimistic": "Dark, negative worldview",
        "cynical": "Skeptical, distrustful, world-weary",
        "naive": "Innocent, trusting, inexperienced",
        
        # Involvement Level
        "engaged": "Personally invested, involved",
        "detached": "Objective, distant, clinical",
        "intimate": "Close, personal, confessional",
        "authoritative": "Expert, commanding, definitive",
        
        # Perspective Type
        "first_person": "I/we narrator, personal experience",
        "third_person": "He/she/they narrator, external view",
        "second_person": "You narrator, direct address",
        "collective": "We/us narrator, group perspective"
    }
    
    @classmethod
    def get_all_attributes(cls) -> Dict[str, Dict[str, str]]:
        """Get complete taxonomy organized by category"""
        return {
            "namespace": cls.NAMESPACE_ATTRIBUTES,
            "style": cls.STYLE_ATTRIBUTES,
            "persona": cls.PERSONA_ATTRIBUTES
        }
    
    @classmethod
    def get_attribute_category(cls, attribute_name: str) -> Optional[str]:
        """Find which category an attribute belongs to"""
        for category, attributes in cls.get_all_attributes().items():
            if attribute_name in attributes:
                return category
        return None

class NarrativeAttributeExtractor:
    """Extracts narrative attributes from text using POVM measurements and analysis"""
    
    def __init__(self, groq_client=None):
        self.groq_client = groq_client
        self.taxonomy = NarrativeAttributeTaxonomy()
    
    async def extract_signature(self, text: str, use_llm: bool = True) -> NarrativeSignature:
        """Extract complete narrative signature from text"""
        
        # Method 1: POVM-based extraction (quantum measurements)
        povm_signature = await self._extract_via_povms(text)
        
        # Method 2: LLM-based analysis (semantic understanding)
        if use_llm and self.groq_client:
            llm_signature = await self._extract_via_llm(text)
            # Combine POVM and LLM results
            signature = self._combine_signatures(povm_signature, llm_signature)
        else:
            signature = povm_signature
        
        # Extract pure essence (content independent of style/persona/namespace)
        signature.pure_essence = await self._extract_pure_essence(text)
        
        return signature
    
    async def _extract_via_povms(self, text: str) -> NarrativeSignature:
        """Extract signature using existing POVM measurements"""
        # This would interface with the existing POVM system
        # For now, return a basic structure
        return NarrativeSignature(
            namespace={},
            style={},
            persona={},
            metadata={"extraction_method": "povm"}
        )
    
    async def _extract_via_llm(self, text: str) -> NarrativeSignature:
        """Extract signature using LLM analysis"""
        if not self.groq_client:
            return NarrativeSignature(namespace={}, style={}, persona={})
        
        prompt = self._build_extraction_prompt(text)
        
        try:
            response = await self.groq_client.generate_text(prompt, max_tokens=800)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return NarrativeSignature(namespace={}, style={}, persona={})
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for LLM-based attribute extraction"""
        
        # Create organized attribute lists
        namespace_attrs = list(self.taxonomy.NAMESPACE_ATTRIBUTES.keys())
        style_attrs = list(self.taxonomy.STYLE_ATTRIBUTES.keys())
        persona_attrs = list(self.taxonomy.PERSONA_ATTRIBUTES.keys())
        
        return f"""Analyze the following text and extract its narrative attributes. Rate each attribute from 0.0 (not present) to 1.0 (strongly present).

TEXT:
{text[:1000]}{"..." if len(text) > 1000 else ""}

Provide ratings for these attributes in JSON format:

NAMESPACE (what world/domain):
{', '.join(namespace_attrs)}

STYLE (how it's told):
{', '.join(style_attrs)}

PERSONA (who's telling it):
{', '.join(persona_attrs)}

Return as JSON:
{{
  "namespace": {{"attribute": 0.5, ...}},
  "style": {{"attribute": 0.5, ...}},
  "persona": {{"attribute": 0.5, ...}}
}}

Only include attributes with scores > 0.1. Be precise and objective."""
    
    def _parse_llm_response(self, response: str) -> NarrativeSignature:
        """Parse LLM response into NarrativeSignature"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                return NarrativeSignature(
                    namespace=data.get("namespace", {}),
                    style=data.get("style", {}),
                    persona=data.get("persona", {}),
                    metadata={"extraction_method": "llm"}
                )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
        
        return NarrativeSignature(namespace={}, style={}, persona={})
    
    def _combine_signatures(self, povm_sig: NarrativeSignature, llm_sig: NarrativeSignature) -> NarrativeSignature:
        """Combine POVM and LLM signatures intelligently"""
        # Average the weights, prioritizing non-zero values
        combined_namespace = self._merge_dicts(povm_sig.namespace, llm_sig.namespace)
        combined_style = self._merge_dicts(povm_sig.style, llm_sig.style)
        combined_persona = self._merge_dicts(povm_sig.persona, llm_sig.persona)
        
        return NarrativeSignature(
            namespace=combined_namespace,
            style=combined_style,
            persona=combined_persona,
            metadata={"extraction_method": "combined"}
        )
    
    def _merge_dicts(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> Dict[str, float]:
        """Merge two attribute dictionaries"""
        result = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key, 0.0)
            val2 = dict2.get(key, 0.0)
            # Average if both present, otherwise use the non-zero value
            if val1 > 0 and val2 > 0:
                result[key] = (val1 + val2) / 2
            else:
                result[key] = max(val1, val2)
        
        # Filter out very low values
        return {k: v for k, v in result.items() if v > 0.1}
    
    async def _extract_pure_essence(self, text: str) -> np.ndarray:
        """Extract content essence independent of style/persona/namespace"""
        # This would use the embedding system to extract semantic content
        # while filtering out stylistic markers
        # For now, return a placeholder
        return np.random.randn(64)  # Placeholder 64D vector

class NarrativeAttributeOperator:
    """Performs mathematical operations on narrative attributes in ρ-space"""
    
    def __init__(self):
        self.taxonomy = NarrativeAttributeTaxonomy()
    
    def add_signatures(self, base: NarrativeSignature, modifier: NarrativeSignature, 
                      strength: float = 1.0) -> NarrativeSignature:
        """Add modifier signature to base signature"""
        return NarrativeSignature(
            namespace=self._add_dicts(base.namespace, modifier.namespace, strength),
            style=self._add_dicts(base.style, modifier.style, strength),
            persona=self._add_dicts(base.persona, modifier.persona, strength),
            pure_essence=base.pure_essence,  # Preserve original content
            metadata={"operation": "add", "strength": strength}
        )
    
    def subtract_signatures(self, base: NarrativeSignature, modifier: NarrativeSignature,
                           strength: float = 1.0) -> NarrativeSignature:
        """Subtract modifier signature from base signature"""
        return NarrativeSignature(
            namespace=self._subtract_dicts(base.namespace, modifier.namespace, strength),
            style=self._subtract_dicts(base.style, modifier.style, strength),
            persona=self._subtract_dicts(base.persona, modifier.persona, strength),
            pure_essence=base.pure_essence,
            metadata={"operation": "subtract", "strength": strength}
        )
    
    def blend_signatures(self, sig1: NarrativeSignature, sig2: NarrativeSignature,
                        blend_ratio: float = 0.5) -> NarrativeSignature:
        """Blend two signatures with specified ratio"""
        return NarrativeSignature(
            namespace=self._blend_dicts(sig1.namespace, sig2.namespace, blend_ratio),
            style=self._blend_dicts(sig1.style, sig2.style, blend_ratio),
            persona=self._blend_dicts(sig1.persona, sig2.persona, blend_ratio),
            pure_essence=self._blend_vectors(sig1.pure_essence, sig2.pure_essence, blend_ratio),
            metadata={"operation": "blend", "ratio": blend_ratio}
        )
    
    def _add_dicts(self, base: Dict[str, float], modifier: Dict[str, float], 
                   strength: float) -> Dict[str, float]:
        """Add modifier to base with strength scaling"""
        result = base.copy()
        for key, value in modifier.items():
            result[key] = result.get(key, 0.0) + (value * strength)
            # Clamp to [0, 1] range
            result[key] = max(0.0, min(1.0, result[key]))
        return result
    
    def _subtract_dicts(self, base: Dict[str, float], modifier: Dict[str, float],
                       strength: float) -> Dict[str, float]:
        """Subtract modifier from base with strength scaling"""
        result = base.copy()
        for key, value in modifier.items():
            if key in result:
                result[key] = result[key] - (value * strength)
                result[key] = max(0.0, result[key])
                # Remove if very small
                if result[key] < 0.05:
                    del result[key]
        return result
    
    def _blend_dicts(self, dict1: Dict[str, float], dict2: Dict[str, float],
                     ratio: float) -> Dict[str, float]:
        """Blend two dictionaries with specified ratio"""
        result = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key, 0.0)
            val2 = dict2.get(key, 0.0)
            result[key] = val1 * (1 - ratio) + val2 * ratio
            
        # Filter out very small values
        return {k: v for k, v in result.items() if v > 0.05}
    
    def _blend_vectors(self, vec1: Optional[np.ndarray], vec2: Optional[np.ndarray],
                      ratio: float) -> Optional[np.ndarray]:
        """Blend two vectors with specified ratio"""
        if vec1 is None:
            return vec2
        if vec2 is None:
            return vec1
        return vec1 * (1 - ratio) + vec2 * ratio

class NarrativeAttributeBalancer:
    """AI agent to help balance attribute combinations for natural results"""
    
    def __init__(self, groq_client=None):
        self.groq_client = groq_client
        self.taxonomy = NarrativeAttributeTaxonomy()
    
    async def suggest_balancing(self, signature: NarrativeSignature) -> Dict[str, Any]:
        """Analyze signature and suggest improvements for natural language"""
        
        # Check for potential conflicts or excessive weightings
        issues = self._detect_issues(signature)
        
        if self.groq_client and issues:
            suggestions = await self._get_llm_suggestions(signature, issues)
        else:
            suggestions = self._get_rule_based_suggestions(issues)
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "balanced_signature": self._apply_balancing(signature, suggestions)
        }
    
    def _detect_issues(self, signature: NarrativeSignature) -> List[Dict[str, Any]]:
        """Detect potential issues with the signature"""
        issues = []
        
        # Check for excessive attribute weights
        for category, attributes in [
            ("namespace", signature.namespace),
            ("style", signature.style), 
            ("persona", signature.persona)
        ]:
            for attr, weight in attributes.items():
                if weight > 0.9:
                    issues.append({
                        "type": "excessive_weight",
                        "category": category,
                        "attribute": attr,
                        "weight": weight,
                        "description": f"{attr} weight ({weight:.2f}) may cause unnatural language"
                    })
        
        # Check for conflicting attributes
        conflicts = self._find_conflicts(signature)
        issues.extend(conflicts)
        
        return issues
    
    def _find_conflicts(self, signature: NarrativeSignature) -> List[Dict[str, Any]]:
        """Find conflicting attribute combinations"""
        conflicts = []
        
        # Define known conflicts
        conflict_pairs = [
            ("formal", "informal"),
            ("elaborate", "sparse"),
            ("optimistic", "pessimistic"),
            ("reliable", "unreliable"),
            ("realistic", "fantasy"),
        ]
        
        for attr1, attr2 in conflict_pairs:
            weight1 = self._get_attribute_weight(signature, attr1)
            weight2 = self._get_attribute_weight(signature, attr2)
            
            if weight1 > 0.5 and weight2 > 0.5:
                conflicts.append({
                    "type": "conflict",
                    "attributes": [attr1, attr2],
                    "weights": [weight1, weight2],
                    "description": f"Conflicting attributes: {attr1} ({weight1:.2f}) vs {attr2} ({weight2:.2f})"
                })
        
        return conflicts
    
    def _get_attribute_weight(self, signature: NarrativeSignature, attr_name: str) -> float:
        """Get weight of an attribute from signature"""
        for category_dict in [signature.namespace, signature.style, signature.persona]:
            if attr_name in category_dict:
                return category_dict[attr_name]
        return 0.0
    
    async def _get_llm_suggestions(self, signature: NarrativeSignature, 
                                  issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get AI suggestions for balancing"""
        if not self.groq_client:
            return []
        
        prompt = f"""As a narrative language expert, analyze this attribute signature and suggest improvements:

CURRENT SIGNATURE:
Namespace: {signature.namespace}
Style: {signature.style}  
Persona: {signature.persona}

DETECTED ISSUES:
{json.dumps(issues, indent=2)}

Suggest specific adjustments to create more natural, balanced language. Consider:
1. Reducing excessive weights (>0.9)
2. Resolving conflicts between opposing attributes
3. Enhancing coherence across categories

Return suggestions as JSON array:
[{{"attribute": "name", "current_weight": 0.9, "suggested_weight": 0.7, "reason": "explanation"}}]"""
        
        try:
            response = await self.groq_client.generate_text(prompt, max_tokens=600)
            return self._parse_suggestions(response)
        except Exception as e:
            logger.error(f"Failed to get LLM suggestions: {e}")
            return []
    
    def _parse_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM suggestions response"""
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse suggestions: {e}")
        return []
    
    def _get_rule_based_suggestions(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get rule-based suggestions for balancing"""
        suggestions = []
        
        for issue in issues:
            if issue["type"] == "excessive_weight":
                suggestions.append({
                    "attribute": issue["attribute"],
                    "current_weight": issue["weight"],
                    "suggested_weight": min(0.8, issue["weight"]),
                    "reason": "Reduced to prevent unnatural language pressure"
                })
            elif issue["type"] == "conflict":
                # Suggest reducing both conflicting attributes
                for i, attr in enumerate(issue["attributes"]):
                    suggestions.append({
                        "attribute": attr,
                        "current_weight": issue["weights"][i],
                        "suggested_weight": issue["weights"][i] * 0.7,
                        "reason": f"Reduced to resolve conflict with {issue['attributes'][1-i]}"
                    })
        
        return suggestions
    
    def _apply_balancing(self, signature: NarrativeSignature, 
                        suggestions: List[Dict[str, Any]]) -> NarrativeSignature:
        """Apply balancing suggestions to create new signature"""
        balanced = NarrativeSignature(
            namespace=signature.namespace.copy(),
            style=signature.style.copy(),
            persona=signature.persona.copy(),
            pure_essence=signature.pure_essence,
            metadata={"balanced": True}
        )
        
        for suggestion in suggestions:
            attr_name = suggestion["attribute"]
            new_weight = suggestion["suggested_weight"]
            
            # Find and update the attribute
            for category_dict in [balanced.namespace, balanced.style, balanced.persona]:
                if attr_name in category_dict:
                    category_dict[attr_name] = new_weight
                    break
        
        return balanced

class NarrativeAttributeManager:
    """High-level manager for narrative attribute operations"""
    
    def __init__(self, groq_client=None):
        self.extractor = NarrativeAttributeExtractor(groq_client)
        self.operator = NarrativeAttributeOperator()
        self.balancer = NarrativeAttributeBalancer(groq_client)
        self.saved_signatures = {}  # In-memory storage
    
    async def extract_and_save(self, text: str, signature_name: str) -> NarrativeSignature:
        """Extract signature from text and save it"""
        signature = await self.extractor.extract_signature(text)
        self.saved_signatures[signature_name] = signature
        logger.info(f"Saved signature '{signature_name}'")
        return signature
    
    def get_signature(self, name: str) -> Optional[NarrativeSignature]:
        """Get saved signature by name"""
        return self.saved_signatures.get(name)
    
    def list_signatures(self) -> List[str]:
        """List all saved signature names"""
        return list(self.saved_signatures.keys())
    
    async def apply_signature(self, text: str, signature_name: str, 
                             strength: float = 1.0) -> Dict[str, Any]:
        """Apply saved signature to new text"""
        signature = self.get_signature(signature_name)
        if not signature:
            raise ValueError(f"Signature '{signature_name}' not found")
        
        # Extract current signature of the text
        current_sig = await self.extractor.extract_signature(text)
        
        # Apply the signature
        modified_sig = self.operator.add_signatures(current_sig, signature, strength)
        
        # Balance the result
        balancing_result = await self.balancer.suggest_balancing(modified_sig)
        
        return {
            "original_signature": current_sig,
            "applied_signature": signature,
            "modified_signature": modified_sig,
            "balanced_signature": balancing_result["balanced_signature"],
            "balancing_suggestions": balancing_result["suggestions"]
        }
    
    def save_signatures_to_file(self, filepath: str):
        """Save all signatures to JSON file"""
        data = {}
        for name, signature in self.saved_signatures.items():
            data[name] = {
                "namespace": signature.namespace,
                "style": signature.style,
                "persona": signature.persona,
                "metadata": signature.metadata
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} signatures to {filepath}")
    
    def load_signatures_from_file(self, filepath: str):
        """Load signatures from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, sig_data in data.items():
            signature = NarrativeSignature(
                namespace=sig_data["namespace"],
                style=sig_data["style"],
                persona=sig_data["persona"],
                metadata=sig_data.get("metadata", {})
            )
            self.saved_signatures[name] = signature
        
        logger.info(f"Loaded {len(data)} signatures from {filepath}")

# Example usage and demo functions
async def demo_narrative_attributes():
    """Demonstration of the narrative attribute system"""
    
    # Initialize with Groq client (would need actual client)
    manager = NarrativeAttributeManager()  # groq_client would go here
    
    # Example texts with different signatures
    texts = {
        "academic": "The research indicates that quantum mechanical principles may apply to narrative analysis through density matrix representations.",
        "casual": "So basically, this whole quantum story thing is pretty cool - it's like using physics to understand how we tell stories.",
        "fantasy": "In the mystical realm of narrative consciousness, the ancient art of story-weaving draws upon the fundamental forces of semantic reality."
    }
    
    # Extract and save signatures
    for name, text in texts.items():
        signature = await manager.extract_and_save(text, name)
        print(f"\nExtracted '{name}' signature:")
        print(f"  Namespace: {signature.namespace}")
        print(f"  Style: {signature.style}")
        print(f"  Persona: {signature.persona}")
    
    # Demonstrate signature application
    base_text = "The cat sat on the mat."
    
    for sig_name in ["academic", "casual", "fantasy"]:
        result = await manager.apply_signature(base_text, sig_name)
        print(f"\nApplying '{sig_name}' signature to: '{base_text}'")
        print(f"  Balanced signature: {result['balanced_signature'].style}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_narrative_attributes())