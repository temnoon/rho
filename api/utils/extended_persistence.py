"""
Extended persistence utilities for comprehensive data storage.

This module extends the basic persistence to include:
- Distillation studio sessions and results
- Channel composition libraries
- Integrability test history
- Residue computation results
- All user-generated content for full session recovery
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from .persistence import (
    DATA_DIR, save_json_atomic, numpy_to_json_serializable, 
    json_to_numpy, ensure_data_dir
)

logger = logging.getLogger(__name__)

# Extended data files
DISTILLATION_SESSIONS_FILE = "distillation_sessions.json"
CHANNEL_COMPOSITIONS_FILE = "channel_compositions.json"
INTEGRABILITY_HISTORY_FILE = "integrability_history.json"
RESIDUE_RESULTS_FILE = "residue_results.json"
USER_PREFERENCES_FILE = "user_preferences.json"
EXPORTED_EMBEDDINGS_FILE = "exported_embeddings.json"


def save_distillation_sessions(sessions: Dict[str, Any]) -> bool:
    """
    Save distillation studio sessions for recovery.
    
    Args:
        sessions: Dictionary of distillation sessions
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        filepath = os.path.join(DATA_DIR, DISTILLATION_SESSIONS_FILE)
        
        # Include timestamp for each session
        timestamped_sessions = {}
        for session_id, session_data in sessions.items():
            timestamped_sessions[session_id] = {
                **session_data,
                'last_updated': session_data.get('last_updated', 'unknown'),
                'created_at': session_data.get('created_at', 'unknown')
            }
        
        serializable_data = numpy_to_json_serializable(timestamped_sessions)
        save_json_atomic(filepath, serializable_data)
        
        logger.info(f"Saved {len(sessions)} distillation sessions to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save distillation sessions: {e}")
        return False


def load_distillation_sessions() -> Dict[str, Any]:
    """
    Load distillation studio sessions.
    
    Returns:
        Dictionary of distillation sessions
    """
    try:
        filepath = os.path.join(DATA_DIR, DISTILLATION_SESSIONS_FILE)
        
        if not os.path.exists(filepath):
            logger.info(f"No distillation sessions file found at {filepath}")
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sessions = json_to_numpy(data)
        logger.info(f"Loaded {len(sessions)} distillation sessions from {filepath}")
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to load distillation sessions: {e}")
        return {}


def save_channel_compositions(compositions: Dict[str, Any]) -> bool:
    """
    Save channel composition library.
    
    Args:
        compositions: Dictionary of composed channels
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        filepath = os.path.join(DATA_DIR, CHANNEL_COMPOSITIONS_FILE)
        
        # Include metadata for each composition
        enhanced_compositions = {}
        for comp_id, comp_data in compositions.items():
            enhanced_compositions[comp_id] = {
                **comp_data,
                'saved_at': comp_data.get('saved_at', 'unknown'),
                'composition_type': comp_data.get('composition_type', 'unknown')
            }
        
        # Note: Channel objects need special serialization
        serializable_data = _serialize_channel_compositions(enhanced_compositions)
        save_json_atomic(filepath, serializable_data)
        
        logger.info(f"Saved {len(compositions)} channel compositions to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save channel compositions: {e}")
        return False


def load_channel_compositions() -> Dict[str, Any]:
    """
    Load channel composition library.
    
    Returns:
        Dictionary of composed channels
    """
    try:
        filepath = os.path.join(DATA_DIR, CHANNEL_COMPOSITIONS_FILE)
        
        if not os.path.exists(filepath):
            logger.info(f"No channel compositions file found at {filepath}")
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        compositions = _deserialize_channel_compositions(data)
        logger.info(f"Loaded {len(compositions)} channel compositions from {filepath}")
        return compositions
        
    except Exception as e:
        logger.error(f"Failed to load channel compositions: {e}")
        return {}


def save_integrability_history(history: List[Dict[str, Any]]) -> bool:
    """
    Save integrability test history.
    
    Args:
        history: List of integrability test results
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        filepath = os.path.join(DATA_DIR, INTEGRABILITY_HISTORY_FILE)
        
        # Add timestamps and enhance metadata
        enhanced_history = []
        for test_result in history:
            enhanced_result = {
                **test_result,
                'test_timestamp': test_result.get('test_timestamp', 'unknown'),
                'test_type': test_result.get('test_type', 'unknown')
            }
            enhanced_history.append(enhanced_result)
        
        serializable_data = numpy_to_json_serializable(enhanced_history)
        save_json_atomic(filepath, serializable_data)
        
        logger.info(f"Saved {len(history)} integrability tests to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save integrability history: {e}")
        return False


def load_integrability_history() -> List[Dict[str, Any]]:
    """
    Load integrability test history.
    
    Returns:
        List of integrability test results
    """
    try:
        filepath = os.path.join(DATA_DIR, INTEGRABILITY_HISTORY_FILE)
        
        if not os.path.exists(filepath):
            logger.info(f"No integrability history file found at {filepath}")
            return []
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        history = json_to_numpy(data)
        logger.info(f"Loaded {len(history)} integrability tests from {filepath}")
        return history
        
    except Exception as e:
        logger.error(f"Failed to load integrability history: {e}")
        return []


def save_exported_embeddings(embeddings: Dict[str, Any]) -> bool:
    """
    Save exported rho-embeddings from distillation studio.
    
    Args:
        embeddings: Dictionary of exported embeddings
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        filepath = os.path.join(DATA_DIR, EXPORTED_EMBEDDINGS_FILE)
        
        # Load existing embeddings
        existing_embeddings = load_exported_embeddings()
        
        # Merge with new embeddings
        merged_embeddings = {**existing_embeddings, **embeddings}
        
        # Add export metadata
        for embedding_id, embedding_data in merged_embeddings.items():
            if 'export_metadata' not in embedding_data:
                embedding_data['export_metadata'] = {
                    'exported_at': embedding_data.get('created_at', 'unknown'),
                    'file_format': 'json',
                    'schema_version': '1.0'
                }
        
        serializable_data = numpy_to_json_serializable(merged_embeddings)
        save_json_atomic(filepath, serializable_data)
        
        logger.info(f"Saved {len(merged_embeddings)} exported embeddings to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save exported embeddings: {e}")
        return False


def load_exported_embeddings() -> Dict[str, Any]:
    """
    Load exported rho-embeddings.
    
    Returns:
        Dictionary of exported embeddings
    """
    try:
        filepath = os.path.join(DATA_DIR, EXPORTED_EMBEDDINGS_FILE)
        
        if not os.path.exists(filepath):
            logger.info(f"No exported embeddings file found at {filepath}")
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        embeddings = json_to_numpy(data)
        logger.info(f"Loaded {len(embeddings)} exported embeddings from {filepath}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to load exported embeddings: {e}")
        return {}


def save_user_preferences(preferences: Dict[str, Any]) -> bool:
    """
    Save user interface preferences and settings.
    
    Args:
        preferences: Dictionary of user preferences
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        filepath = os.path.join(DATA_DIR, USER_PREFERENCES_FILE)
        
        # Add timestamp
        timestamped_prefs = {
            **preferences,
            'last_updated': preferences.get('last_updated', 'unknown'),
            'preferences_version': '1.0'
        }
        
        save_json_atomic(filepath, timestamped_prefs)
        
        logger.info(f"Saved user preferences to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save user preferences: {e}")
        return False


def load_user_preferences() -> Dict[str, Any]:
    """
    Load user interface preferences and settings.
    
    Returns:
        Dictionary of user preferences
    """
    try:
        filepath = os.path.join(DATA_DIR, USER_PREFERENCES_FILE)
        
        if not os.path.exists(filepath):
            logger.info(f"No user preferences file found at {filepath}")
            return _get_default_preferences()
        
        with open(filepath, "r", encoding="utf-8") as f:
            preferences = json.load(f)
        
        logger.info(f"Loaded user preferences from {filepath}")
        return preferences
        
    except Exception as e:
        logger.error(f"Failed to load user preferences: {e}")
        return _get_default_preferences()


def export_complete_session(session_name: Optional[str] = None) -> str:
    """
    Export complete session data to a single file.
    
    Args:
        session_name: Optional name for the session export
        
    Returns:
        Path to the exported session file
    """
    try:
        from datetime import datetime
        
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        ensure_data_dir()
        export_filepath = os.path.join(DATA_DIR, f"{session_name}_complete.json")
        
        # Gather all data
        from routes.matrix_routes import STATE
        from routes.povm_routes import PACKS
        from core.channel_composition import CHANNEL_COMPOSER
        from core.integrability_testing import TESTER
        
        complete_session = {
            'session_metadata': {
                'session_name': session_name,
                'exported_at': datetime.now().isoformat(),
                'schema_version': '1.0',
                'components': [
                    'quantum_states', 'povm_packs', 'channel_compositions',
                    'distillation_sessions', 'integrability_history',
                    'exported_embeddings', 'user_preferences'
                ]
            },
            'quantum_states': STATE,
            'povm_packs': PACKS,
            'channel_compositions': {
                'composed_channels': CHANNEL_COMPOSER.composed_channels,
                'channel_library': CHANNEL_COMPOSER.channel_library,
                'composition_history': CHANNEL_COMPOSER.composition_history
            },
            'distillation_sessions': load_distillation_sessions(),
            'integrability_history': TESTER.test_history if hasattr(TESTER, 'test_history') else [],
            'exported_embeddings': load_exported_embeddings(),
            'user_preferences': load_user_preferences()
        }
        
        # Convert to serializable format
        serializable_session = numpy_to_json_serializable(complete_session)
        
        # Save complete session
        save_json_atomic(export_filepath, serializable_session)
        
        logger.info(f"Exported complete session to {export_filepath}")
        return export_filepath
        
    except Exception as e:
        logger.error(f"Failed to export complete session: {e}")
        return ""


def import_complete_session(session_filepath: str) -> bool:
    """
    Import complete session data from a file.
    
    Args:
        session_filepath: Path to the session export file
        
    Returns:
        True if successful
    """
    try:
        if not os.path.exists(session_filepath):
            logger.error(f"Session file not found: {session_filepath}")
            return False
        
        with open(session_filepath, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        # Convert back from JSON
        session_data = json_to_numpy(session_data)
        
        # Import each component
        from routes.matrix_routes import STATE
        from routes.povm_routes import PACKS
        
        # Clear and restore quantum states
        STATE.clear()
        STATE.update(session_data.get('quantum_states', {}))
        
        # Clear and restore POVM packs
        PACKS.clear()
        PACKS.update(session_data.get('povm_packs', {}))
        
        # Restore channel compositions
        channel_data = session_data.get('channel_compositions', {})
        if channel_data:
            from core.channel_composition import CHANNEL_COMPOSER
            CHANNEL_COMPOSER.composed_channels.update(channel_data.get('composed_channels', {}))
            CHANNEL_COMPOSER.channel_library.update(channel_data.get('channel_library', {}))
            CHANNEL_COMPOSER.composition_history.extend(channel_data.get('composition_history', []))
        
        # Restore other components
        save_distillation_sessions(session_data.get('distillation_sessions', {}))
        save_exported_embeddings(session_data.get('exported_embeddings', {}))
        save_user_preferences(session_data.get('user_preferences', {}))
        
        logger.info(f"Successfully imported complete session from {session_filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to import complete session: {e}")
        return False


def get_data_directory_info() -> Dict[str, Any]:
    """
    Get information about the data directory and stored files.
    
    Returns:
        Dictionary with data directory information
    """
    try:
        ensure_data_dir()
        
        info = {
            'data_directory': DATA_DIR,
            'directory_exists': os.path.exists(DATA_DIR),
            'files': {},
            'total_size_bytes': 0
        }
        
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                filepath = os.path.join(DATA_DIR, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    info['files'][filename] = {
                        'size_bytes': stat.st_size,
                        'modified_at': stat.st_mtime,
                        'readable': os.access(filepath, os.R_OK),
                        'writable': os.access(filepath, os.W_OK)
                    }
                    info['total_size_bytes'] += stat.st_size
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get data directory info: {e}")
        return {'error': str(e)}


# Helper functions

def _serialize_channel_compositions(compositions: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize channel compositions with special handling for channel objects."""
    serialized = {}
    
    for comp_id, comp_data in compositions.items():
        serialized_comp = {}
        
        for key, value in comp_data.items():
            if key == 'final_channel':
                # Store channel metadata instead of the object
                serialized_comp[key] = {
                    'channel_type': 'composed_channel',
                    'serialized': True,
                    'note': 'Channel object not directly serializable'
                }
            elif key == 'channel_nodes':
                # Serialize channel nodes
                serialized_nodes = {}
                for node_id, node in value.items():
                    serialized_nodes[node_id] = {
                        'node_id': node.node_id,
                        'channel_type': node.channel_type,
                        'parameters': node.parameters,
                        'metadata': node.metadata,
                        'created_at': node.created_at
                    }
                serialized_comp[key] = serialized_nodes
            else:
                serialized_comp[key] = numpy_to_json_serializable(value)
        
        serialized[comp_id] = serialized_comp
    
    return serialized


def _deserialize_channel_compositions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize channel compositions with special handling for channel objects."""
    # Note: Full deserialization would require reconstructing channel objects
    # For now, we store the metadata and indicate that channels need reconstruction
    return json_to_numpy(data)


def _get_default_preferences() -> Dict[str, Any]:
    """Get default user preferences."""
    return {
        'theme': 'light',
        'default_channel_type': 'rank_one_update',
        'default_alpha': 0.3,
        'auto_save_enabled': True,
        'auto_save_interval': 30,
        'default_distillation_strategy': 'comprehensive',
        'show_advanced_options': False,
        'animation_enabled': True,
        'visualization_quality': 'high'
    }


# Enhanced auto-saver that includes extended data
class ExtendedAutoSaver:
    """Extended auto-saver that includes all user-generated content."""
    
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.enabled = True
        self._thread = None
        self._stop_event = None
    
    def start(self, state_getter, packs_getter, additional_data_getters=None):
        """
        Start extended automatic saving.
        
        Args:
            state_getter: Function that returns current state dict
            packs_getter: Function that returns current packs dict
            additional_data_getters: Optional dict of additional data getter functions
        """
        if not self.enabled or self._thread is not None:
            return
        
        import threading
        
        self._stop_event = threading.Event()
        
        def extended_save_loop():
            while not self._stop_event.wait(self.interval):
                try:
                    # Save basic data
                    from .persistence import save_state, save_packs
                    
                    state = state_getter()
                    if state:
                        save_state(state)
                    
                    packs = packs_getter()
                    if packs:
                        save_packs(packs)
                    
                    # Save extended data
                    if additional_data_getters:
                        for data_type, getter_func in additional_data_getters.items():
                            try:
                                data = getter_func()
                                if data_type == 'distillation_sessions':
                                    save_distillation_sessions(data)
                                elif data_type == 'channel_compositions':
                                    save_channel_compositions(data)
                                elif data_type == 'exported_embeddings':
                                    save_exported_embeddings(data)
                                # Add more data types as needed
                            except Exception as e:
                                logger.error(f"Failed to save {data_type}: {e}")
                
                except Exception as e:
                    logger.error(f"Extended auto-save failed: {e}")
        
        self._thread = threading.Thread(target=extended_save_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started extended auto-save with interval {self.interval}s")
    
    def stop(self):
        """Stop extended automatic saving."""
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join()
        self._thread = None
        self._stop_event = None
        logger.info("Stopped extended auto-save")


# Global extended auto-saver instance
extended_auto_saver = ExtendedAutoSaver()