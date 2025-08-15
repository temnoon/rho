"""
Centralized configuration for the Rho Quantum Narrative System.

All magic numbers, model settings, and system parameters in one place.
"""

import os
from pathlib import Path
from typing import Dict, Any

class RhoConfig:
    """Central configuration manager for the Rho system"""
    
    # Core quantum system parameters
    RHO_DIMENSION = 64
    EMBEDDING_DIMENSION = 1536  # OpenAI text-embedding-3-large
    DEFAULT_ALPHA = 0.3  # Exponential moving blend parameter
    
    # Server configuration
    API_PORT = 8192
    WEB_PORT = 8080
    HOST = "0.0.0.0"
    
    # LLM and embedding models
    GROQ_MODEL = "openai/gpt-oss-120b"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    
    # Un-Earthify system parameters
    ALIEN_WORD_MAX_LENGTH = 6
    CONTEXT_WINDOW_SIZE = 100
    ALIEN_COMPLEXITY_THRESHOLD = 8  # Earth term length threshold for complex alien words
    
    # File paths and directories
    DATA_DIR = "data"
    ALIEN_MAPPINGS_FILE = "alien_mappings.json"
    POVM_PACKS_FILE = "packs.json"
    STATE_FILE = "state.json"
    
    # POVM measurement parameters
    DEFAULT_POVM_PACK = "advanced_narrative_pack"
    MAX_MEASUREMENT_ATTRIBUTES = 8
    
    # Text processing parameters
    MAX_TEXT_LENGTH = 10000  # Characters
    MIN_TEXT_LENGTH = 10
    DEFAULT_REGENERATION_LENGTH_TARGET = None  # Use original length
    
    # API response limits
    MAX_SIMILAR_PASSAGES = 5
    MAX_GUTENBERG_RESULTS = 20
    MAX_CONTEXT_EXAMPLES = 5
    
    # Frontend configuration
    FRONTEND_REFRESH_INTERVAL = 1000  # ms
    MAX_EXPLORATION_RESULTS = 10
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        'GROQ_API_KEY': 'groq_api_key',
        'OPENAI_API_KEY': 'openai_api_key', 
        'RHO_DIMENSION': ('rho_dimension', int),
        'API_PORT': ('api_port', int),
        'EMBED_URL': 'embed_url',
        'DATA_DIR': 'data_dir'
    }
    
    def __init__(self):
        """Initialize configuration with environment variable overrides"""
        self._load_from_env()
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        for env_var, config_attr in self.ENV_MAPPINGS.items():
            env_value = os.getenv(env_var)
            if env_value:
                if isinstance(config_attr, tuple):
                    attr_name, converter = config_attr
                    setattr(self, attr_name.upper(), converter(env_value))
                else:
                    setattr(self, config_attr.upper(), env_value)
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.RHO_DIMENSION > 0, "RHO_DIMENSION must be positive"
        assert self.EMBEDDING_DIMENSION > 0, "EMBEDDING_DIMENSION must be positive"
        assert 0 < self.DEFAULT_ALPHA < 1, "DEFAULT_ALPHA must be between 0 and 1"
        assert self.ALIEN_WORD_MAX_LENGTH > 2, "ALIEN_WORD_MAX_LENGTH must be at least 3"
    
    def get_data_path(self, filename: str = None) -> Path:
        """Get path to data directory or specific file"""
        data_path = Path(self.DATA_DIR)
        if filename:
            return data_path / filename
        return data_path
    
    def get_alien_mappings_path(self) -> Path:
        """Get path to alien mappings file"""
        return self.get_data_path(self.ALIEN_MAPPINGS_FILE)
    
    def get_povm_packs_path(self) -> Path:
        """Get path to POVM packs file"""
        return self.get_data_path(self.POVM_PACKS_FILE)
    
    def get_state_path(self) -> Path:
        """Get path to state file"""
        return self.get_data_path(self.STATE_FILE)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            attr: getattr(self, attr) 
            for attr in dir(self) 
            if attr.isupper() and not attr.startswith('_')
        }
    
    def __repr__(self):
        """String representation of configuration"""
        config_items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"RhoConfig({', '.join(config_items)})"

# Global configuration instance
config = RhoConfig()

# Convenience exports for backward compatibility
DIM = config.RHO_DIMENSION
DATA_DIR = config.DATA_DIR
EMBED_DIM = config.EMBEDDING_DIMENSION
API_PORT = config.API_PORT