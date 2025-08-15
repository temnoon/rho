"""
Data persistence utilities for quantum state and POVM pack storage.

This module handles saving and loading quantum states and POVM packs
to/from disk for session persistence.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PERSISTENCE_FILE = "state.json"
PACKS_FILE = "packs.json"

# Auto-save configuration
AUTO_SAVE_ENABLED = True
AUTO_SAVE_INTERVAL = 30  # seconds


def ensure_data_dir():
    """Ensure data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def save_json_atomic(path: str, obj: Any) -> None:
    """
    Atomic JSON save operation to prevent corruption.
    
    Args:
        path: File path to save to
        obj: Object to serialize
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.rename(tmp, path)


def numpy_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays to JSON-serializable format.
    
    Args:
        obj: Object that may contain numpy arrays
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            # Convert complex arrays to [real, imag] format
            return {"__complex_array__": True, "real": obj.real.tolist(), "imag": obj.imag.tolist()}
        else:
            return obj.tolist()
    elif isinstance(obj, (np.complex64, np.complex128, complex)):
        # Convert complex numbers to [real, imag] format
        return {"__complex__": True, "real": float(obj.real), "imag": float(obj.imag)}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json_serializable(item) for item in obj]
    else:
        return obj


def json_to_numpy(obj: Any) -> Any:
    """
    Convert JSON-loaded data back to numpy arrays where appropriate.
    
    Args:
        obj: Object loaded from JSON
        
    Returns:
        Object with numpy arrays restored
    """
    if isinstance(obj, dict):
        # Handle complex number reconstruction
        if obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])
        elif obj.get("__complex_array__"):
            real_part = np.array(obj["real"])
            imag_part = np.array(obj["imag"])
            return real_part + 1j * imag_part
        
        result = {}
        for k, v in obj.items():
            if k == "rho" and isinstance(v, (list, dict)):
                # Convert matrix back to numpy (handle both complex and real)
                if isinstance(v, dict) and v.get("__complex_array__"):
                    real_part = np.array(v["real"])
                    imag_part = np.array(v["imag"])
                    result[k] = real_part + 1j * imag_part
                else:
                    result[k] = np.array(v)
            elif k in ["effects", "basis_vector"] and isinstance(v, list):
                # Convert POVM effects or vectors
                if isinstance(v[0], list):
                    # List of matrices
                    result[k] = [np.array(matrix) for matrix in v]
                else:
                    # Single vector
                    result[k] = np.array(v)
            else:
                result[k] = json_to_numpy(v)
        return result
    elif isinstance(obj, list):
        return [json_to_numpy(item) for item in obj]
    else:
        return obj


def save_state(state_dict: Dict[str, Dict[str, Any]], 
              filename: Optional[str] = None) -> bool:
    """
    Save quantum state dictionary to disk.
    
    Args:
        state_dict: Dictionary of quantum states
        filename: Optional custom filename
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        
        if filename is None:
            filename = PERSISTENCE_FILE
        
        filepath = os.path.join(DATA_DIR, filename)
        
        # Convert numpy arrays to JSON-serializable format
        serializable_state = numpy_to_json_serializable(state_dict)
        
        # Save atomically
        save_json_atomic(filepath, serializable_state)
        
        logger.info(f"Saved {len(state_dict)} quantum states to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        return False


def load_state(filename: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load quantum state dictionary from disk.
    
    Args:
        filename: Optional custom filename
        
    Returns:
        Dictionary of quantum states (empty if load fails)
    """
    try:
        if filename is None:
            filename = PERSISTENCE_FILE
        
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.info(f"No state file found at {filepath}")
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        state_dict = json_to_numpy(data)
        
        logger.info(f"Loaded {len(state_dict)} quantum states from {filepath}")
        return state_dict
        
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return {}


def save_packs(packs_dict: Dict[str, Dict[str, Any]], 
              filename: Optional[str] = None) -> bool:
    """
    Save POVM packs dictionary to disk.
    
    Args:
        packs_dict: Dictionary of POVM packs
        filename: Optional custom filename
        
    Returns:
        True if successful
    """
    try:
        ensure_data_dir()
        
        if filename is None:
            filename = PACKS_FILE
        
        filepath = os.path.join(DATA_DIR, filename)
        
        # Convert numpy arrays to JSON-serializable format
        serializable_packs = numpy_to_json_serializable(packs_dict)
        
        # Save atomically
        save_json_atomic(filepath, serializable_packs)
        
        logger.info(f"Saved {len(packs_dict)} POVM packs to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save packs: {e}")
        return False


def load_packs(filename: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load POVM packs dictionary from disk.
    
    Args:
        filename: Optional custom filename
        
    Returns:
        Dictionary of POVM packs (empty if load fails)
    """
    try:
        if filename is None:
            filename = PACKS_FILE
        
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.info(f"No packs file found at {filepath}")
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        packs_dict = json_to_numpy(data)
        
        logger.info(f"Loaded {len(packs_dict)} POVM packs from {filepath}")
        return packs_dict
        
    except Exception as e:
        logger.error(f"Failed to load packs: {e}")
        return {}


def backup_data(backup_suffix: Optional[str] = None) -> bool:
    """
    Create backup copies of current data files.
    
    Args:
        backup_suffix: Optional suffix for backup files
        
    Returns:
        True if successful
    """
    try:
        import shutil
        from datetime import datetime
        
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ensure_data_dir()
        
        # Backup state file
        state_path = os.path.join(DATA_DIR, PERSISTENCE_FILE)
        if os.path.exists(state_path):
            backup_path = os.path.join(DATA_DIR, f"state_backup_{backup_suffix}.json")
            shutil.copy2(state_path, backup_path)
            logger.info(f"Backed up state to {backup_path}")
        
        # Backup packs file
        packs_path = os.path.join(DATA_DIR, PACKS_FILE)
        if os.path.exists(packs_path):
            backup_path = os.path.join(DATA_DIR, f"packs_backup_{backup_suffix}.json")
            shutil.copy2(packs_path, backup_path)
            logger.info(f"Backed up packs to {backup_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False


def cleanup_old_backups(keep_count: int = 5) -> None:
    """
    Remove old backup files, keeping only the most recent ones.
    
    Args:
        keep_count: Number of backup files to keep
    """
    try:
        ensure_data_dir()
        
        # Find backup files
        backup_files = []
        for filename in os.listdir(DATA_DIR):
            if filename.startswith(("state_backup_", "packs_backup_")):
                filepath = os.path.join(DATA_DIR, filename)
                mtime = os.path.getmtime(filepath)
                backup_files.append((mtime, filepath))
        
        # Sort by modification time (newest first)
        backup_files.sort(reverse=True)
        
        # Remove old files
        for _, filepath in backup_files[keep_count:]:
            os.remove(filepath)
            logger.info(f"Removed old backup: {os.path.basename(filepath)}")
        
    except Exception as e:
        logger.error(f"Failed to cleanup backups: {e}")


class AutoSaver:
    """
    Automatic background saving for quantum states and POVM packs.
    """
    
    def __init__(self, interval: int = AUTO_SAVE_INTERVAL):
        self.interval = interval
        self.enabled = AUTO_SAVE_ENABLED
        self._thread = None
        self._stop_event = None
    
    def start(self, state_getter, packs_getter):
        """
        Start automatic saving.
        
        Args:
            state_getter: Function that returns current state dict
            packs_getter: Function that returns current packs dict
        """
        if not self.enabled or self._thread is not None:
            return
        
        import threading
        import time
        
        self._stop_event = threading.Event()
        
        def save_loop():
            while not self._stop_event.wait(self.interval):
                try:
                    # Save state
                    state = state_getter()
                    if state:
                        save_state(state)
                    
                    # Save packs
                    packs = packs_getter()
                    if packs:
                        save_packs(packs)
                        
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
        
        self._thread = threading.Thread(target=save_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started auto-save with interval {self.interval}s")
    
    def stop(self):
        """Stop automatic saving."""
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join()
        self._thread = None
        self._stop_event = None
        logger.info("Stopped auto-save")


# Global auto-saver instance
auto_saver = AutoSaver()