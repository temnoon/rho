"""
Centralized Configuration Management System for Rho Project

This module provides unified configuration management for all components:
- API and web service ports
- LLM and embedding model settings
- Prompt templates with versioning
- Database and file paths
- Quantum system parameters

All hardcoded values should be eliminated in favor of this system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass 
class ServiceConfig:
    """Configuration for individual services."""
    name: str
    internal_port: int
    external_port: int
    host: str = "0.0.0.0"
    protocol: str = "http"
    
    @property
    def internal_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.internal_port}"
    
    @property
    def external_url(self) -> str:
        return f"{self.protocol}://localhost:{self.external_port}"

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str
    model: str
    api_key_env: str
    api_url: str
    default_temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    thinking_tags: List[str] = field(default_factory=lambda: ["<think>", "</think>"])

@dataclass
class PromptTemplate:
    """Template for LLM prompts with metadata."""
    name: str
    version: str
    template: str
    description: str
    required_variables: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    last_used: Optional[str] = None

class RhoCentralConfig:
    """
    Central configuration manager for the entire Rho ecosystem.
    
    Eliminates all hardcoded values and provides unified access to:
    - Service endpoints and ports
    - LLM provider configurations
    - Prompt templates with versioning
    - File paths and database settings
    - Quantum system parameters
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.config_file = self.config_dir / "rho_config.json"
        self.prompts_file = self.config_dir / "prompt_templates.json"
        
        # Load configuration
        self._load_config()
        self._load_prompts()
        self._apply_env_overrides()
        
    def _load_config(self):
        """Load main configuration from file or use defaults."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = self._get_default_config()
            self._save_config(config_data)
        
        self._parse_config(config_data)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration values."""
        return {
            "version": "1.0.0",
            "services": {
                "api": {
                    "name": "rho-api",
                    "internal_port": 8000,
                    "external_port": 8192,
                    "host": "0.0.0.0"
                },
                "web": {
                    "name": "rho-web", 
                    "internal_port": 80,
                    "external_port": 5173,
                    "host": "0.0.0.0"
                },
                "analytic_lexicology": {
                    "name": "analytic-lexicology-interface",
                    "internal_port": 5174,
                    "external_port": 5174,
                    "host": "0.0.0.0"
                }
            },
            "quantum": {
                "rho_dimension": 64,
                "embedding_dimension": 1536,
                "default_alpha": 0.3,
                "max_text_length": 10000,
                "min_text_length": 10
            },
            "llm_providers": {
                "groq": {
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant",
                    "api_key_env": "GROQ_API_KEY",
                    "api_url": "https://api.groq.com/openai/v1/chat/completions",
                    "default_temperature": 0.7,
                    "max_tokens": 2000,
                    "timeout": 60
                },
                "ollama": {
                    "provider": "ollama",
                    "model": "gpt-oss:20b",
                    "api_key_env": null,
                    "api_url": "http://localhost:11434",
                    "default_temperature": 0.7,
                    "max_tokens": 2000,
                    "timeout": 30
                }
            },
            "embedding": {
                "default_provider": "openai",
                "model": "text-embedding-3-large",
                "api_key_env": "OPENAI_API_KEY",
                "dimension": 1536,
                "batch_size": 100
            },
            "data_paths": {
                "base_dir": "data",
                "state_file": "state.json",
                "packs_file": "packs.json",
                "matrices_dir": "matrices",
                "alien_mappings": "alien_mappings.json"
            },
            "povm": {
                "default_pack": "advanced_narrative_pack",
                "max_measurement_attributes": 8
            },
            "api_limits": {
                "max_similar_passages": 5,
                "max_gutenberg_results": 20,
                "max_context_examples": 5,
                "frontend_refresh_interval": 1000
            }
        }
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """Parse loaded configuration into typed objects."""
        # Services
        self.services = {}
        for service_name, service_data in config_data.get("services", {}).items():
            self.services[service_name] = ServiceConfig(**service_data)
        
        # LLM Providers
        self.llm_providers = {}
        for provider_name, provider_data in config_data.get("llm_providers", {}).items():
            self.llm_providers[provider_name] = LLMConfig(**provider_data)
        
        # Simple config sections
        self.quantum = config_data.get("quantum", {})
        self.embedding = config_data.get("embedding", {})
        self.data_paths = config_data.get("data_paths", {})
        self.povm = config_data.get("povm", {})
        self.api_limits = config_data.get("api_limits", {})
        
        # Version tracking
        self.version = config_data.get("version", "1.0.0")
    
    def _load_prompts(self):
        """Load prompt templates from file."""
        if self.prompts_file.exists():
            with open(self.prompts_file, 'r') as f:
                prompts_data = json.load(f)
        else:
            prompts_data = self._get_default_prompts()
            self._save_prompts(prompts_data)
        
        self.prompts = {}
        for prompt_name, prompt_data in prompts_data.items():
            self.prompts[prompt_name] = PromptTemplate(**prompt_data)
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Default prompt templates."""
        return {
            "quantum_narrative_transformer": {
                "name": "quantum_narrative_transformer",
                "version": "1.0.0",
                "description": "Main prompt for quantum-guided narrative transformation",
                "required_variables": ["original_text", "guidance_text", "target_attributes"],
                "template": """You are a quantum-guided narrative transformer. Your task is to rewrite the given text according to specific linguistic attributes derived from quantum density matrix measurements.

ORIGINAL TEXT:
{original_text}

QUANTUM GUIDANCE:
{guidance_text}

TARGET ATTRIBUTES:
{target_attributes}

Instructions:
1. Preserve the core narrative meaning and structure
2. Apply the quantum-derived linguistic transformations
3. Maintain readability and coherence
4. Focus on subtle stylistic adjustments guided by the measurements

TRANSFORMED TEXT:"""
            },
            "attribute_explanation": {
                "name": "attribute_explanation",
                "version": "1.0.0", 
                "description": "Explains quantum measurement results in natural language",
                "required_variables": ["attribute_name", "measurement_value", "interpretation"],
                "template": """Explain the quantum measurement '{attribute_name}' with value {measurement_value}.

Context: {interpretation}

Provide a clear, accessible explanation of what this measurement tells us about the text's linguistic properties."""
            },
            "field_analysis": {
                "name": "field_analysis",
                "version": "1.0.0",
                "description": "Analyzes lexical field relationships",
                "required_variables": ["selected_words", "field_measurements", "commutator_analysis"],
                "template": """Analyze the lexical field relationships for these words: {selected_words}

Field Measurements:
{field_measurements}

Commutator Analysis:
{commutator_analysis}

Explain how these words interact within the quantum semantic space and what their relationships reveal about the text's structure."""
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Port overrides
        if api_port := os.getenv("RHO_API_PORT"):
            self.services["api"].external_port = int(api_port)
        
        if web_port := os.getenv("RHO_WEB_PORT"):
            self.services["web"].external_port = int(web_port)
        
        # LLM overrides
        if groq_model := os.getenv("GROQ_MODEL"):
            self.llm_providers["groq"].model = groq_model
        
        # Quantum parameter overrides
        if rho_dim := os.getenv("RHO_DIMENSION"):
            self.quantum["rho_dimension"] = int(rho_dim)
    
    def _save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file."""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _save_prompts(self, prompts_data: Dict[str, Any]):
        """Save prompts to file."""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.prompts_file, 'w') as f:
            json.dump(prompts_data, f, indent=2)
    
    # === Service Access Methods ===
    
    def get_api_url(self, external: bool = True) -> str:
        """Get API service URL."""
        service = self.services["api"]
        return service.external_url if external else service.internal_url
    
    def get_web_url(self, external: bool = True) -> str:
        """Get web service URL.""" 
        service = self.services["web"]
        return service.external_url if external else service.internal_url
    
    def get_analytic_lexicology_url(self) -> str:
        """Get analytic lexicology interface URL."""
        return self.services["analytic_lexicology"].external_url
    
    # === LLM Provider Access ===
    
    def get_llm_config(self, provider: str = "groq") -> LLMConfig:
        """Get LLM provider configuration."""
        if provider not in self.llm_providers:
            raise ValueError(f"Unknown LLM provider: {provider}")
        return self.llm_providers[provider]
    
    def get_api_key(self, provider: str = "groq") -> Optional[str]:
        """Get API key for LLM provider."""
        config = self.get_llm_config(provider)
        if config.api_key_env:
            return os.getenv(config.api_key_env)
        return None
    
    # === Prompt Management ===
    
    def get_prompt(self, name: str) -> PromptTemplate:
        """Get prompt template by name."""
        if name not in self.prompts:
            raise ValueError(f"Unknown prompt template: {name}")
        return self.prompts[name]
    
    def render_prompt(self, name: str, **variables) -> str:
        """Render prompt template with variables."""
        template = self.get_prompt(name)
        
        # Check required variables
        missing = set(template.required_variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables for {name}: {missing}")
        
        # Update usage tracking
        template.usage_count += 1
        template.last_used = datetime.now().isoformat()
        
        # Render template
        return template.template.format(**variables)
    
    def log_prompt_usage(self, name: str, variables: Dict[str, Any], response: str):
        """Log prompt usage for analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_name": name,
            "prompt_version": self.prompts[name].version,
            "variables": variables,
            "response_length": len(response),
            "success": True
        }
        
        # This could be extended to write to a dedicated prompt log file
        logger.info(f"Prompt usage: {json.dumps(log_entry)}")
    
    # === Path Management ===
    
    def get_data_path(self, filename: Optional[str] = None) -> Path:
        """Get data directory path or specific file."""
        base = Path(self.data_paths["base_dir"])
        return base / filename if filename else base
    
    def get_state_path(self) -> Path:
        """Get state file path."""
        return self.get_data_path(self.data_paths["state_file"])
    
    def get_packs_path(self) -> Path:
        """Get POVM packs file path."""
        return self.get_data_path(self.data_paths["packs_file"])
    
    def get_matrices_dir(self) -> Path:
        """Get matrices directory path."""
        return self.get_data_path(self.data_paths["matrices_dir"])
    
    # === Configuration Export ===
    
    def to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dictionary."""
        return {
            "version": self.version,
            "services": {name: {
                "name": svc.name,
                "internal_port": svc.internal_port,
                "external_port": svc.external_port,
                "host": svc.host
            } for name, svc in self.services.items()},
            "quantum": self.quantum,
            "embedding": self.embedding,
            "data_paths": self.data_paths,
            "povm": self.povm,
            "api_limits": self.api_limits,
            "llm_providers": {name: {
                "provider": llm.provider,
                "model": llm.model,
                "api_key_env": llm.api_key_env,
                "api_url": llm.api_url,
                "default_temperature": llm.default_temperature,
                "max_tokens": llm.max_tokens,
                "timeout": llm.timeout
            } for name, llm in self.llm_providers.items()}
        }
    
    def get_prompt_usage_stats(self) -> Dict[str, Any]:
        """Get prompt usage statistics."""
        return {
            name: {
                "version": prompt.version,
                "usage_count": prompt.usage_count,
                "last_used": prompt.last_used,
                "description": prompt.description
            }
            for name, prompt in self.prompts.items()
        }

# Global configuration instance
config = RhoCentralConfig()

# Convenience exports for backward compatibility
API_URL = config.get_api_url()
WEB_URL = config.get_web_url()
DIM = config.quantum["rho_dimension"]
DATA_DIR = str(config.get_data_path())