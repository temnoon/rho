"""
Persistence Management API Routes

Provides endpoints for saving, loading, and managing all user data
including distillation sessions, channel compositions, and complete session exports.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime

from utils.extended_persistence import (
    save_distillation_sessions, load_distillation_sessions,
    save_channel_compositions, load_channel_compositions,
    save_exported_embeddings, load_exported_embeddings,
    save_user_preferences, load_user_preferences,
    export_complete_session, import_complete_session,
    get_data_directory_info, extended_auto_saver
)

router = APIRouter(prefix="/persistence", tags=["persistence"])
logger = logging.getLogger(__name__)


class SaveSessionRequest(BaseModel):
    """Request to save a distillation session."""
    session_id: str
    session_data: Dict[str, Any]


class ExportRequest(BaseModel):
    """Request to export session data."""
    session_name: Optional[str] = None
    include_components: List[str] = ["all"]


class UserPreferencesRequest(BaseModel):
    """Request to save user preferences."""
    preferences: Dict[str, Any]


@router.post("/distillation/save")
async def save_distillation_session(request: SaveSessionRequest):
    """
    Save a distillation studio session.
    """
    try:
        # Load existing sessions
        existing_sessions = load_distillation_sessions()
        
        # Add timestamp metadata
        session_data = {
            **request.session_data,
            'last_updated': datetime.now().isoformat(),
            'session_id': request.session_id
        }
        
        # Update sessions
        existing_sessions[request.session_id] = session_data
        
        # Save back to disk
        success = save_distillation_sessions(existing_sessions)
        
        if success:
            return {
                "success": True,
                "session_id": request.session_id,
                "message": "Distillation session saved successfully",
                "saved_at": session_data['last_updated']
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save session")
            
    except Exception as e:
        logger.error(f"Failed to save distillation session: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.get("/distillation/load/{session_id}")
async def load_distillation_session(session_id: str):
    """
    Load a specific distillation studio session.
    """
    try:
        sessions = load_distillation_sessions()
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {
            "success": True,
            "session_id": session_id,
            "session_data": sessions[session_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load distillation session: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


@router.get("/distillation/list")
async def list_distillation_sessions():
    """
    List all saved distillation studio sessions.
    """
    try:
        sessions = load_distillation_sessions()
        
        # Create summary for each session
        session_summaries = {}
        for session_id, session_data in sessions.items():
            session_summaries[session_id] = {
                "session_id": session_id,
                "last_updated": session_data.get('last_updated', 'unknown'),
                "created_at": session_data.get('created_at', 'unknown'),
                "narrative_length": session_data.get('narrative_text_length', 0),
                "distillation_stage": session_data.get('distillation_stage', 'unknown'),
                "has_final_embedding": 'final_rho_embedding' in session_data
            }
        
        return {
            "success": True,
            "sessions": session_summaries,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to list distillation sessions: {e}")
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.delete("/distillation/{session_id}")
async def delete_distillation_session(session_id: str):
    """
    Delete a distillation studio session.
    """
    try:
        sessions = load_distillation_sessions()
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Remove session
        deleted_session = sessions.pop(session_id)
        
        # Save updated sessions
        success = save_distillation_sessions(sessions)
        
        if success:
            return {
                "success": True,
                "deleted_session_id": session_id,
                "message": "Session deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save after deletion")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete distillation session: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.post("/embeddings/save")
async def save_rho_embeddings(embeddings: Dict[str, Any]):
    """
    Save exported rho-embeddings.
    """
    try:
        # Add metadata
        timestamped_embeddings = {}
        for embedding_id, embedding_data in embeddings.items():
            timestamped_embeddings[embedding_id] = {
                **embedding_data,
                'saved_at': datetime.now().isoformat()
            }
        
        success = save_exported_embeddings(timestamped_embeddings)
        
        if success:
            return {
                "success": True,
                "saved_embeddings": len(embeddings),
                "message": "Rho-embeddings saved successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save embeddings")
            
    except Exception as e:
        logger.error(f"Failed to save rho-embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.get("/embeddings/list")
async def list_saved_embeddings():
    """
    List all saved rho-embeddings.
    """
    try:
        embeddings = load_exported_embeddings()
        
        # Create summary for each embedding
        embedding_summaries = {}
        for embedding_id, embedding_data in embeddings.items():
            metadata = embedding_data.get('distillation_metadata', {})
            embedding_summaries[embedding_id] = {
                "embedding_id": embedding_id,
                "saved_at": embedding_data.get('saved_at', 'unknown'),
                "strategy": metadata.get('strategy', 'unknown'),
                "channel_type": metadata.get('channel_type', 'unknown'),
                "original_text_length": metadata.get('original_text_length', 0),
                "has_essence_components": 'essence_components' in embedding_data
            }
        
        return {
            "success": True,
            "embeddings": embedding_summaries,
            "total_embeddings": len(embeddings)
        }
        
    except Exception as e:
        logger.error(f"Failed to list embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.get("/embeddings/load/{embedding_id}")
async def load_rho_embedding(embedding_id: str):
    """
    Load a specific rho-embedding.
    """
    try:
        embeddings = load_exported_embeddings()
        
        if embedding_id not in embeddings:
            raise HTTPException(status_code=404, detail=f"Embedding {embedding_id} not found")
        
        return {
            "success": True,
            "embedding_id": embedding_id,
            "embedding_data": embeddings[embedding_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


@router.post("/preferences/save")
async def save_preferences(request: UserPreferencesRequest):
    """
    Save user interface preferences.
    """
    try:
        preferences = {
            **request.preferences,
            'last_updated': datetime.now().isoformat()
        }
        
        success = save_user_preferences(preferences)
        
        if success:
            return {
                "success": True,
                "message": "User preferences saved successfully",
                "saved_at": preferences['last_updated']
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save preferences")
            
    except Exception as e:
        logger.error(f"Failed to save preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.get("/preferences/load")
async def load_preferences():
    """
    Load user interface preferences.
    """
    try:
        preferences = load_user_preferences()
        
        return {
            "success": True,
            "preferences": preferences
        }
        
    except Exception as e:
        logger.error(f"Failed to load preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


@router.post("/export/complete")
async def export_complete_session_endpoint(request: ExportRequest):
    """
    Export complete session data to a downloadable file.
    """
    try:
        export_path = export_complete_session(request.session_name)
        
        if not export_path:
            raise HTTPException(status_code=500, detail="Export failed")
        
        # Get file info
        file_size = os.path.getsize(export_path)
        filename = os.path.basename(export_path)
        
        return {
            "success": True,
            "export_path": export_path,
            "filename": filename,
            "file_size_bytes": file_size,
            "message": "Complete session exported successfully",
            "download_url": f"/persistence/download/{filename}"
        }
        
    except Exception as e:
        logger.error(f"Failed to export complete session: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/import/complete")
async def import_complete_session_endpoint(file: UploadFile = File(...)):
    """
    Import complete session data from an uploaded file.
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_filepath = temp_file.name
        
        try:
            # Import the session
            success = import_complete_session(temp_filepath)
            
            if success:
                return {
                    "success": True,
                    "filename": file.filename,
                    "file_size_bytes": len(content),
                    "message": "Complete session imported successfully"
                }
            else:
                raise HTTPException(status_code=500, detail="Import failed")
                
        finally:
            # Clean up temp file
            os.unlink(temp_filepath)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import complete session: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/status")
async def get_persistence_status():
    """
    Get status of the persistence system and data directory.
    """
    try:
        data_info = get_data_directory_info()
        
        # Check if auto-saver is running
        auto_saver_status = {
            "enabled": extended_auto_saver.enabled,
            "running": extended_auto_saver._thread is not None,
            "interval_seconds": extended_auto_saver.interval
        }
        
        return {
            "success": True,
            "data_directory_info": data_info,
            "auto_saver_status": auto_saver_status,
            "persistence_files": {
                "quantum_states": "state.json",
                "povm_packs": "packs.json",
                "distillation_sessions": "distillation_sessions.json",
                "channel_compositions": "channel_compositions.json",
                "exported_embeddings": "exported_embeddings.json",
                "user_preferences": "user_preferences.json"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get persistence status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/backup/create")
async def create_backup():
    """
    Create a backup of all persistent data.
    """
    try:
        from utils.persistence import backup_data
        
        backup_success = backup_data()
        
        if backup_success:
            return {
                "success": True,
                "message": "Backup created successfully",
                "backup_timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Backup creation failed")
            
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/cleanup/old_backups")
async def cleanup_old_backups(keep_count: int = 5):
    """
    Clean up old backup files.
    """
    try:
        from utils.persistence import cleanup_old_backups
        
        cleanup_old_backups(keep_count)
        
        return {
            "success": True,
            "message": f"Cleaned up old backups, keeping {keep_count} most recent",
            "keep_count": keep_count
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup backups: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download exported files.
    """
    try:
        from fastapi.responses import FileResponse
        from utils.extended_persistence import DATA_DIR
        
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='application/json'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# Auto-save management endpoints

@router.post("/autosave/start")
async def start_autosave():
    """
    Start the extended auto-save system.
    """
    try:
        from routes.matrix_routes import STATE
        from routes.povm_routes import PACKS
        
        # Start extended auto-saver with additional data getters
        additional_getters = {
            'distillation_sessions': load_distillation_sessions,
            'exported_embeddings': load_exported_embeddings
        }
        
        extended_auto_saver.start(
            state_getter=lambda: STATE,
            packs_getter=lambda: PACKS,
            additional_data_getters=additional_getters
        )
        
        return {
            "success": True,
            "message": "Extended auto-save started successfully",
            "interval_seconds": extended_auto_saver.interval
        }
        
    except Exception as e:
        logger.error(f"Failed to start auto-save: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-save start failed: {str(e)}")


@router.post("/autosave/stop")
async def stop_autosave():
    """
    Stop the extended auto-save system.
    """
    try:
        extended_auto_saver.stop()
        
        return {
            "success": True,
            "message": "Extended auto-save stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop auto-save: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-save stop failed: {str(e)}")


@router.get("/autosave/status")
async def get_autosave_status():
    """
    Get status of the auto-save system.
    """
    try:
        status = {
            "enabled": extended_auto_saver.enabled,
            "running": extended_auto_saver._thread is not None,
            "interval_seconds": extended_auto_saver.interval
        }
        
        return {
            "success": True,
            "autosave_status": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get auto-save status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")