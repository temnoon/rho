"""
Queue management routes for the Rho Quantum Narrative System.

This module provides endpoints for job queue management and batch processing.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queue", tags=["queue"])

# Persistent queue directory
QUEUE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "batch_queue")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "batch_results")

# Ensure directories exist
os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: float
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None


class BatchJobRequest(BaseModel):
    operation: str  # 'quantum_transformation_batch', 'gutenberg_analysis_batch'
    parameters: Dict
    items: List[Dict]
    scheduled_for: Optional[str] = None  # ISO datetime for scheduled execution

class QuantumTransformationBatch(BaseModel):
    texts: List[str]
    transformation_type: str = "enhance_narrative"
    strength: float = 0.7
    creativity: float = 0.8
    preservation: float = 0.8
    complexity: float = 0.5
    scheduled_for: Optional[str] = None

class GutenbergAnalysisBatch(BaseModel):
    book_ids: List[int]
    transformation_type: str = "enhance_narrative"
    strength: float = 0.7
    max_chars_per_book: int = 10000
    scheduled_for: Optional[str] = None


def _load_all_jobs() -> Dict:
    """Load all jobs from persistent storage."""
    jobs = {}
    try:
        for filename in os.listdir(QUEUE_DIR):
            if filename.startswith('job_') and filename.endswith('.json'):
                job_file = os.path.join(QUEUE_DIR, filename)
                with open(job_file, 'r') as f:
                    job = json.load(f)
                    jobs[job['job_id']] = job
    except Exception as e:
        logger.error(f"Error loading jobs: {e}")
    return jobs

def _save_job(job: Dict):
    """Save job to persistent storage."""
    try:
        job_file = os.path.join(QUEUE_DIR, f"job_{job['job_id']}.json")
        with open(job_file, 'w') as f:
            json.dump(job, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving job {job['job_id']}: {e}")
        raise

@router.get("/status")
def get_queue_status():
    """Get overall queue status"""
    jobs = _load_all_jobs()
    
    total_jobs = len(jobs)
    pending_jobs = len([job for job in jobs.values() if job["status"] == "pending"])
    running_jobs = len([job for job in jobs.values() if job["status"] == "running"])
    completed_jobs = len([job for job in jobs.values() if job["status"] == "completed"])
    failed_jobs = len([job for job in jobs.values() if job["status"] == "failed"])
    
    # Check worker status
    worker_state_file = os.path.join(QUEUE_DIR, "worker_state.json")
    worker_status = {"running": False, "last_updated": None}
    
    if os.path.exists(worker_state_file):
        try:
            with open(worker_state_file, 'r') as f:
                worker_status = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading worker state: {e}")
    
    return {
        "total_jobs": total_jobs,
        "pending": pending_jobs,
        "running": running_jobs,
        "completed": completed_jobs,
        "failed": failed_jobs,
        "queue_active": running_jobs > 0 or pending_jobs > 0,
        "worker_status": worker_status
    }


@router.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Get status of a specific job"""
    jobs = _load_all_jobs()
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(**job)


@router.post("/batch_submit")
def submit_batch_job(request: BatchJobRequest):
    """Submit a batch job to the queue"""
    job_id = str(uuid.uuid4())
    
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "completed_at": None,
        "progress": 0.0,
        "result": None,
        "error": None,
        "operation": request.operation,
        "parameters": request.parameters,
        "items": request.items,
        "scheduled_for": request.scheduled_for
    }
    
    # Save to persistent storage
    _save_job(job)
    
    logger.info(f"Submitted batch job {job_id} with {len(request.items)} items")
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "estimated_duration": len(request.items) * 30,  # More realistic estimate (30s per item)
        "items_count": len(request.items),
        "scheduled_for": request.scheduled_for
    }


@router.post("/cancel/{job_id}")
def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    jobs = _load_all_jobs()
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    job["status"] = "cancelled"
    job["completed_at"] = time.time()
    
    # Save updated job
    _save_job(job)
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "cancelled_at": job["completed_at"]
    }

# New convenient endpoints for quantum transformations

@router.post("/quantum_batch")
def submit_quantum_transformation_batch(request: QuantumTransformationBatch):
    """Submit a batch of quantum transformations"""
    items = []
    for i, text in enumerate(request.texts):
        items.append({
            "text": text,
            "transformation_type": request.transformation_type,
            "strength": request.strength
        })
    
    batch_request = BatchJobRequest(
        operation="quantum_transformation_batch",
        parameters={
            "default_transformation": request.transformation_type,
            "default_strength": request.strength,
            "creativity": request.creativity,
            "preservation": request.preservation,
            "complexity": request.complexity
        },
        items=items,
        scheduled_for=request.scheduled_for
    )
    
    return submit_batch_job(batch_request)

@router.post("/gutenberg_batch")  
def submit_gutenberg_analysis_batch(request: GutenbergAnalysisBatch):
    """Submit a batch of Gutenberg book analyses"""
    items = []
    for book_id in request.book_ids:
        items.append({
            "book_id": book_id,
            "analysis_type": "full_transformation"
        })
    
    batch_request = BatchJobRequest(
        operation="gutenberg_analysis_batch",
        parameters={
            "transformation_type": request.transformation_type,
            "strength": request.strength,
            "max_chars_per_book": request.max_chars_per_book
        },
        items=items,
        scheduled_for=request.scheduled_for
    )
    
    return submit_batch_job(batch_request)

@router.get("/results/{job_id}")
def get_job_results(job_id: str):
    """Get detailed results for a completed job"""
    result_file = os.path.join(RESULTS_DIR, f"result_{job_id}.json")
    
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        logger.error(f"Error loading results for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Error loading results")

@router.get("/results")
def list_all_results():
    """List all available job results"""
    try:
        results = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.startswith('result_') and filename.endswith('.json'):
                result_file = os.path.join(RESULTS_DIR, filename)
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    job_id = filename.replace('result_', '').replace('.json', '')
                    results.append({
                        "job_id": job_id,
                        "operation": result_data.get('operation'),
                        "completed_at": result_data.get('completed_at'),
                        "total_items": result_data.get('total_items'),
                        "successful_items": result_data.get('successful_items'),
                        "failed_items": result_data.get('failed_items')
                    })
                except Exception as e:
                    logger.warning(f"Error reading result file {filename}: {e}")
        
        # Sort by completion time (newest first)
        results.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        raise HTTPException(status_code=500, detail="Error listing results")