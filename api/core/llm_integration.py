"""
LLM Integration for Quantum Narrative Transformation.

This module handles integration with various LLM providers (Ollama, OpenAI, etc.)
to generate narrative transformations based on quantum state measurements.
"""

import json
import os
import requests
import logging
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from groq import Groq

logger = logging.getLogger(__name__)

@dataclass
class QuantumNarrativeContext:
    """Context for quantum-guided narrative generation."""
    original_text: str
    target_attributes: Dict[str, float]
    current_measurements: Dict[str, float]
    quantum_diagnostics: Dict[str, Any]

class OllamaClient:
    """Client for local Ollama LLM integration."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gpt-oss:20b"):
        self.base_url = base_url
        self.model = model
        
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
            
    def generate_narrative_transformation(self, context: QuantumNarrativeContext, pack_id: str = None) -> str:
        """Generate a narrative transformation using quantum measurements."""
        
        # Build a sophisticated prompt based on quantum measurements
        prompt = self._build_quantum_prompt(context, pack_id)
        
        try:
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2000,
                }
            }
            
            logger.info(f"Sending request to Ollama: {self.base_url}/api/generate")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=300  # 5 minutes
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                logger.info(f"Ollama response length: {len(generated_text)}")
                logger.info(f"Ollama full response keys: {list(result.keys())}")
                
                # Extract just the transformed narrative (remove any meta-commentary)
                return self._extract_narrative(generated_text)
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"LLM generation failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to generate narrative: {e}")
            raise
    
    def _build_quantum_prompt(self, context: QuantumNarrativeContext, pack_id: str = None) -> str:
        """Build a sophisticated prompt using quantum measurements."""
        
        # Extract key measurements for linguistic guidance
        measurements = context.current_measurements
        
        # Get pack metadata for intelligent interpretation
        pack_info = self._get_pack_info(pack_id) if pack_id else None
        
        # Create attribute descriptions based on measurements
        attribute_guidance = self._generate_attribute_guidance(measurements, pack_info)
        guidance_text = "\n".join(f"• {guidance}" for guidance in attribute_guidance)
        
        return f"""You are a quantum-guided narrative transformer. Your task is to rewrite the given text according to specific linguistic attributes derived from quantum density matrix measurements.

ORIGINAL TEXT:
{context.original_text}

QUANTUM MEASUREMENTS (0.0 = low, 1.0 = high):
{self._format_measurements(measurements)}

TRANSFORMATION GUIDELINES:
{guidance_text}

INSTRUCTIONS:
1. Rewrite the original text to match the specified linguistic attributes
2. Preserve the core meaning and content while adjusting style and register
3. Make the transformation substantial enough to reflect the quantum measurements
4. Output ONLY the transformed text, no meta-commentary, explanations, or questions
5. Do NOT ask for more information - use the provided measurements to guide your transformation
6. End your response with ---END---

TRANSFORMED TEXT:
"""

    def _get_pack_info(self, pack_id: str) -> Dict[str, Any]:
        """Get pack metadata for interpretation."""
        try:
            # Import here to avoid circular imports
            from routes.povm_routes import PACKS
            return PACKS.get(pack_id, {})
        except:
            return {}
    
    def _generate_attribute_guidance(self, measurements: Dict[str, float], pack_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent guidance based on measurements and pack metadata."""
        guidance = []
        
        # Group measurements by axis (remove label suffix)
        axes_measurements = {}
        for key, value in measurements.items():
            # Split on underscores and try to find axis vs label
            parts = key.split("_")
            if len(parts) >= 2:
                # Try different combinations to find the axis vs label split
                for i in range(1, len(parts)):
                    axis_id = "_".join(parts[:i])
                    label = "_".join(parts[i:])
                    
                    # Check if this axis exists in pack info
                    if pack_info and "axes" in pack_info:
                        axis_found = any(axis.get("id") == axis_id for axis in pack_info["axes"])
                        if axis_found:
                            if axis_id not in axes_measurements:
                                axes_measurements[axis_id] = {}
                            axes_measurements[axis_id][label] = value
                            break
                    else:
                        # Fallback: assume last part is label
                        if i == len(parts) - 1:
                            if axis_id not in axes_measurements:
                                axes_measurements[axis_id] = {}
                            axes_measurements[axis_id][label] = value
        
        # Generate guidance for each axis
        for axis_id, axis_measurements in axes_measurements.items():
            guidance.extend(self._generate_axis_guidance(axis_id, axis_measurements, pack_info))
        
        return guidance
    
    def _generate_axis_guidance(self, axis_id: str, axis_measurements: Dict[str, float], pack_info: Dict[str, Any]) -> List[str]:
        """Generate guidance for a specific axis."""
        guidance = []
        
        # Get axis metadata from pack info
        axis_info = None
        if pack_info and "axes" in pack_info:
            for axis in pack_info["axes"]:
                if axis.get("id") == axis_id:
                    axis_info = axis
                    break
        
        # Find the measurement with highest value (the dominant pole)
        if not axis_measurements:
            return guidance
            
        dominant_label = max(axis_measurements.keys(), key=lambda k: axis_measurements[k])
        dominant_value = axis_measurements[dominant_label]
        
        # Generate guidance based on axis semantics and dominant pole
        if dominant_value > 0.6:  # Significant lean toward one pole
            guidance_text = self._get_semantic_guidance(axis_id, dominant_label, dominant_value, axis_info)
            if guidance_text:
                guidance.append(guidance_text)
        
        return guidance
    
    def _get_semantic_guidance(self, axis_id: str, dominant_label: str, value: float, axis_info: Dict[str, Any]) -> str:
        """Get semantic guidance for an axis-label combination."""
        
        # Use axis metadata if available
        if axis_info:
            description = axis_info.get("description", "")
            labels = axis_info.get("labels", [])
            
            # Generate contextual guidance
            if "formal" in dominant_label.lower() or "formal" in description.lower():
                return f"Use formal register and sophisticated language (strength: {value:.2f})"
            elif "informal" in dominant_label.lower() or "casual" in description.lower():
                return f"Use casual, conversational language (strength: {value:.2f})"
            elif "narrative" in dominant_label.lower() or "story" in description.lower():
                return f"Emphasize narrative structure and storytelling elements (strength: {value:.2f})"
            elif "involved" in dominant_label.lower() or "personal" in description.lower():
                return f"Use personal, engaged language with first-person perspective (strength: {value:.2f})"
            elif "affect" in dominant_label.lower() or "emotion" in description.lower():
                return f"Express emotional content and evaluative stance (strength: {value:.2f})"
            elif "elaborat" in dominant_label.lower() or "detail" in description.lower():
                return f"Use detailed descriptions and elaborated content (strength: {value:.2f})"
        
        # Fallback: generic guidance based on label names
        readable_axis = axis_id.replace("_", " ").title()
        readable_label = dominant_label.replace("_", " ").title()
        return f"Adjust {readable_axis} toward {readable_label} style (strength: {value:.2f})"
    
    def _format_measurements(self, measurements: Dict[str, float]) -> str:
        """Format measurements for the prompt."""
        formatted = []
        for key, value in measurements.items():
            # Convert key to readable format
            readable_key = key.replace("_", " ").title()
            formatted.append(f"• {readable_key}: {value:.3f}")
        return "\n".join(formatted)
        
    def _extract_narrative(self, generated_text: str) -> str:
        """Extract just the narrative content from the generated response."""
        logger.info(f"Raw LLM response: {repr(generated_text)}")
        
        # If the text is short and appears to be meta-commentary (asking for info), return it as-is
        # This handles cases where the LLM asks questions instead of transforming
        if len(generated_text) < 500:
            if any(phrase in generated_text.lower() for phrase in [
                "i'm ready to", "could you please", "i need", "provide those details", "need more information"
            ]):
                logger.info(f"LLM provided meta-response: {repr(generated_text.strip())}")
                return generated_text.strip()
            elif not any(phrase in generated_text.lower() for phrase in [
                "transformed text:", "guideline", "instruction"
            ]):
                logger.info(f"Returning short response directly: {repr(generated_text.strip())}")
                # Remove the end marker if present
                clean_text = generated_text.strip()
                if clean_text.endswith("---END---"):
                    clean_text = clean_text[:-9].strip()
                return clean_text
        
        # Remove any meta-commentary before the actual narrative
        lines = generated_text.split('\n')
        narrative_lines = []
        found_start = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip obvious meta-commentary
            if any(skip_phrase in line.lower() for skip_phrase in [
                "transformed text:", "rewritten version:", "here is", "the text becomes",
                "following the guidelines", "according to", "based on"
            ]):
                found_start = True
                continue
                
            # Stop at end marker
            if "---END---" in line:
                break
                
            # Include content lines
            if found_start or not any(meta_phrase in line.lower() for meta_phrase in [
                "quantum", "measurement", "guideline", "instruction", "attribute"
            ]):
                narrative_lines.append(line)
                found_start = True
        
        result = '\n'.join(narrative_lines).strip()
        logger.info(f"Extracted narrative: {repr(result)}")
        
        # Clean up the result
        final_result = result if result else generated_text.strip()
        if final_result.endswith("---END---"):
            final_result = final_result[:-9].strip()
        
        return final_result

    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text response for a given prompt"""
        try:
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temp for more consistent JSON responses
                    "top_p": 0.9,
                    "num_predict": max_tokens,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=60  # 1 minute for attribute translation
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "{}"  # Return empty JSON as fallback
                
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return "{}"  # Return empty JSON as fallback

class GroqClient:
    """Client for Groq LLM integration using macOS keychain for API key."""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
        
    def _get_api_key(self) -> str:
        """Retrieve Groq API key from macOS keychain or environment."""
        # First try environment variable
        env_key = os.getenv("GROQ_API_KEY")
        if env_key:
            logger.info("Using Groq API key from environment variable")
            return env_key
        
        # Try multiple keychain entries
        keychain_attempts = [
            # New format (password entry)
            ['security', 'find-generic-password', '-l', 'groq api key', '-a', 'dreegle@gmail.com', '-w'],
            ['security', 'find-generic-password', '-l', 'groq api key', '-w'],
            # Old format (service entry)
            ['security', 'find-generic-password', '-s', 'com.humanizer.humanizer-lighthouse.groq', '-w'],
        ]
        
        for attempt in keychain_attempts:
            try:
                result = subprocess.run(attempt, capture_output=True, text=True, check=True)
                key = result.stdout.strip()
                if key:
                    logger.info(f"Found Groq API key using: {' '.join(attempt[1:4])}")
                    return key
            except subprocess.CalledProcessError:
                continue
        
        logger.error("Failed to retrieve Groq API key from any source")
        raise Exception("Groq API key not found in keychain or environment")
    
    def _initialize_client(self):
        """Initialize the Groq client with API key."""
        try:
            api_key = self._get_api_key()
            self.client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Groq client is available."""
        return self.client is not None
    
    def generate_narrative_transformation(self, context: QuantumNarrativeContext, pack_id: str = None) -> str:
        """Generate a narrative transformation using quantum measurements via Groq."""
        if not self.is_available():
            raise Exception("Groq client not available")
        
        # Build prompt using the same logic as Ollama
        prompt = self._build_quantum_prompt(context, pack_id)
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant",  # Fast, current model
                temperature=0.7,
                max_tokens=2000,
            )
            
            generated_text = chat_completion.choices[0].message.content
            logger.info(f"Groq response length: {len(generated_text)}")
            
            # Extract just the transformed narrative (remove any meta-commentary)
            return self._extract_narrative(generated_text)
            
        except Exception as e:
            logger.error(f"Failed to generate narrative with Groq: {e}")
            raise
    
    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text response for a given prompt using Groq."""
        if not self.is_available():
            return "{}"
        
        try:
            # For thinking models, use hidden reasoning format to get clean output
            extra_params = {}
            if "gpt-oss" in "openai/gpt-oss-120b":  # Check if it's a thinking model
                extra_params["reasoning_format"] = "hidden"
                extra_params["reasoning_effort"] = "medium"
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="openai/gpt-oss-120b",  # Better model for quality responses
                temperature=0.3,
                max_tokens=max_tokens,
                **extra_params
            )
            
            content = chat_completion.choices[0].message.content
            if content:
                # Clean output for both thinking and non-thinking models
                return self._extract_final_answer(content.strip())
            return ""
            
        except Exception as e:
            logger.error(f"Failed to generate text with Groq: {e}")
            return "{}"
    
    # Reuse the same prompt building and text extraction methods as OllamaClient
    def _build_quantum_prompt(self, context: QuantumNarrativeContext, pack_id: str = None) -> str:
        """Build a sophisticated prompt using quantum measurements."""
        measurements = context.current_measurements
        pack_info = self._get_pack_info(pack_id) if pack_id else None
        attribute_guidance = self._generate_attribute_guidance(measurements, pack_info)
        guidance_text = "\n".join(f"• {guidance}" for guidance in attribute_guidance)
        
        return f"""You are a quantum-guided narrative transformer. Your task is to rewrite the given text according to specific linguistic attributes derived from quantum density matrix measurements.

ORIGINAL TEXT:
{context.original_text}

QUANTUM MEASUREMENTS (0.0 = low, 1.0 = high):
{self._format_measurements(measurements)}

TRANSFORMATION GUIDELINES:
{guidance_text}

INSTRUCTIONS:
1. Rewrite the original text to match the specified linguistic attributes
2. Preserve the core meaning and content while adjusting style and register
3. Make the transformation substantial enough to reflect the quantum measurements
4. Output ONLY the transformed text, no meta-commentary, explanations, or questions
5. Do NOT ask for more information - use the provided measurements to guide your transformation
6. End your response with ---END---

TRANSFORMED TEXT:
"""

    # Copy the helper methods from OllamaClient
    def _get_pack_info(self, pack_id: str) -> Dict[str, Any]:
        """Get pack metadata for interpretation."""
        try:
            from routes.povm_routes import PACKS
            return PACKS.get(pack_id, {})
        except:
            return {}
    
    def _generate_attribute_guidance(self, measurements: Dict[str, float], pack_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent guidance based on measurements and pack metadata."""
        guidance = []
        axes_measurements = {}
        
        for key, value in measurements.items():
            parts = key.split("_")
            if len(parts) >= 2:
                for i in range(1, len(parts)):
                    axis_id = "_".join(parts[:i])
                    label = "_".join(parts[i:])
                    
                    if pack_info and "axes" in pack_info:
                        axis_found = any(axis.get("id") == axis_id for axis in pack_info["axes"])
                        if axis_found:
                            if axis_id not in axes_measurements:
                                axes_measurements[axis_id] = {}
                            axes_measurements[axis_id][label] = value
                            break
                    else:
                        if i == len(parts) - 1:
                            if axis_id not in axes_measurements:
                                axes_measurements[axis_id] = {}
                            axes_measurements[axis_id][label] = value
        
        for axis_id, axis_measurements in axes_measurements.items():
            guidance.extend(self._generate_axis_guidance(axis_id, axis_measurements, pack_info))
        
        return guidance
    
    def _generate_axis_guidance(self, axis_id: str, axis_measurements: Dict[str, float], pack_info: Dict[str, Any]) -> List[str]:
        """Generate guidance for a specific axis."""
        guidance = []
        axis_info = None
        
        if pack_info and "axes" in pack_info:
            for axis in pack_info["axes"]:
                if axis.get("id") == axis_id:
                    axis_info = axis
                    break
        
        if not axis_measurements:
            return guidance
            
        dominant_label = max(axis_measurements.keys(), key=lambda k: axis_measurements[k])
        dominant_value = axis_measurements[dominant_label]
        
        if dominant_value > 0.6:
            guidance_text = self._get_semantic_guidance(axis_id, dominant_label, dominant_value, axis_info)
            if guidance_text:
                guidance.append(guidance_text)
        
        return guidance
    
    def _get_semantic_guidance(self, axis_id: str, dominant_label: str, value: float, axis_info: Dict[str, Any]) -> str:
        """Get semantic guidance for an axis-label combination."""
        if axis_info:
            description = axis_info.get("description", "")
            
            if "formal" in dominant_label.lower() or "formal" in description.lower():
                return f"Use formal register and sophisticated language (strength: {value:.2f})"
            elif "informal" in dominant_label.lower() or "casual" in description.lower():
                return f"Use casual, conversational language (strength: {value:.2f})"
            elif "narrative" in dominant_label.lower() or "story" in description.lower():
                return f"Emphasize narrative structure and storytelling elements (strength: {value:.2f})"
            elif "involved" in dominant_label.lower() or "personal" in description.lower():
                return f"Use personal, engaged language with first-person perspective (strength: {value:.2f})"
            elif "affect" in dominant_label.lower() or "emotion" in description.lower():
                return f"Express emotional content and evaluative stance (strength: {value:.2f})"
            elif "elaborat" in dominant_label.lower() or "detail" in description.lower():
                return f"Use detailed descriptions and elaborated content (strength: {value:.2f})"
        
        readable_axis = axis_id.replace("_", " ").title()
        readable_label = dominant_label.replace("_", " ").title()
        return f"Adjust {readable_axis} toward {readable_label} style (strength: {value:.2f})"
    
    def _format_measurements(self, measurements: Dict[str, float]) -> str:
        """Format measurements for the prompt."""
        formatted = []
        for key, value in measurements.items():
            readable_key = key.replace("_", " ").title()
            formatted.append(f"• {readable_key}: {value:.3f}")
        return "\n".join(formatted)
        
    def _extract_narrative(self, generated_text: str) -> str:
        """Extract just the narrative content from the generated response."""
        logger.info(f"Raw Groq response: {repr(generated_text)}")
        
        if len(generated_text) < 500:
            if any(phrase in generated_text.lower() for phrase in [
                "i'm ready to", "could you please", "i need", "provide those details", "need more information"
            ]):
                return generated_text.strip()
            elif not any(phrase in generated_text.lower() for phrase in [
                "transformed text:", "guideline", "instruction"
            ]):
                clean_text = generated_text.strip()
                if clean_text.endswith("---END---"):
                    clean_text = clean_text[:-9].strip()
                return clean_text
        
        lines = generated_text.split('\n')
        narrative_lines = []
        found_start = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(skip_phrase in line.lower() for skip_phrase in [
                "transformed text:", "rewritten version:", "here is", "the text becomes",
                "following the guidelines", "according to", "based on"
            ]):
                found_start = True
                continue
                
            if "---END---" in line:
                break
                
            if found_start or not any(meta_phrase in line.lower() for meta_phrase in [
                "quantum", "measurement", "guideline", "instruction", "attribute"
            ]):
                narrative_lines.append(line)
                found_start = True
        
        result = '\n'.join(narrative_lines).strip()
        final_result = result if result else generated_text.strip()
        if final_result.endswith("---END---"):
            final_result = final_result[:-9].strip()
        
        return final_result

    def _extract_final_answer(self, content: str) -> str:
        """Extract final answer from both thinking and non-thinking model outputs."""
        logger.info(f"Raw model response: {repr(content)}")
        
        if not content:
            return ""
        
        # Handle thinking model outputs with reasoning tags
        if "<think>" in content and "</think>" in content:
            # Extract content outside of thinking tags
            import re
            # Remove all content between <think> and </think> tags (including nested)
            think_pattern = r'<think>.*?</think>'
            clean_content = re.sub(think_pattern, '', content, flags=re.DOTALL)
            
            # Also handle other reasoning tag variations
            reasoning_patterns = [
                r'<reasoning>.*?</reasoning>',
                r'<thought>.*?</thought>',
                r'<analysis>.*?</analysis>'
            ]
            
            for pattern in reasoning_patterns:
                clean_content = re.sub(pattern, '', clean_content, flags=re.DOTALL)
            
            # Clean up whitespace
            clean_content = re.sub(r'\n\s*\n', '\n\n', clean_content.strip())
            
            if clean_content:
                logger.info(f"Extracted content after removing thinking tags: {repr(clean_content)}")
                return clean_content
        
        # For non-thinking models or when reasoning_format="hidden", use existing extraction
        return self._extract_narrative(content)


# Global client instances
ollama_client = OllamaClient()
groq_client = GroqClient()

# Convenience function for the new POVM-attribute system
async def generate_text(prompt: str, max_tokens: int = 500) -> str:
    """Generate text using Groq client first, then fallback to Ollama"""
    if groq_client.is_available():
        try:
            return await groq_client.generate_text(prompt, max_tokens)
        except Exception as e:
            logger.warning(f"Groq generation failed, falling back to Ollama: {e}")
    
    return await ollama_client.generate_text(prompt, max_tokens)

# Function to get the best available narrative transformer
def get_narrative_transformer() -> any:
    """Get the best available LLM client for narrative transformation"""
    if groq_client.is_available():
        logger.info("Using Groq client for narrative transformation")
        return groq_client
    elif ollama_client.is_available():
        logger.info("Using Ollama client for narrative transformation")
        return ollama_client
    else:
        raise Exception("No LLM clients available")