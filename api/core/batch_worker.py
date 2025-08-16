"""
Batch Worker for Quantum Transformation Queue System.

Processes queued jobs asynchronously with persistence and cron integration.
"""

import os
import json
import time
import logging
import threading
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Configuration
QUEUE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "batch_queue")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "batch_results")
WORKER_STATE_FILE = os.path.join(QUEUE_DIR, "worker_state.json")
MAX_CONCURRENT_JOBS = 3
WORKER_SLEEP_INTERVAL = 5  # seconds

# Ensure directories exist
os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class BatchWorker:
    """Worker process for executing queued quantum transformation jobs."""
    
    def __init__(self, api_base_url: str = "http://localhost:8192"):
        self.api_base_url = api_base_url
        self.running = False
        self.worker_thread = None
        self.current_jobs = {}  # job_id -> thread
        self.stats = {
            "started_at": None,
            "jobs_processed": 0,
            "jobs_succeeded": 0,
            "jobs_failed": 0,
            "last_activity": None
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down batch worker...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the batch worker."""
        if self.running:
            logger.warning("Batch worker is already running")
            return
        
        self.running = True
        self.stats["started_at"] = datetime.now().isoformat()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Batch worker started")
        self._save_worker_state()
    
    def stop(self):
        """Stop the batch worker gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping batch worker...")
        self.running = False
        
        # Wait for current jobs to complete (with timeout)
        for job_id, thread in self.current_jobs.items():
            logger.info(f"Waiting for job {job_id} to complete...")
            thread.join(timeout=30)
            if thread.is_alive():
                logger.warning(f"Job {job_id} did not complete within timeout")
        
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
        
        self._save_worker_state()
        logger.info("Batch worker stopped")
    
    def _worker_loop(self):
        """Main worker loop."""
        logger.info("Batch worker loop started")
        
        while self.running:
            try:
                # Clean up completed job threads
                self._cleanup_completed_jobs()
                
                # Check for new jobs if we have capacity
                if len(self.current_jobs) < MAX_CONCURRENT_JOBS:
                    pending_jobs = self._get_pending_jobs()
                    
                    for job_file in pending_jobs[:MAX_CONCURRENT_JOBS - len(self.current_jobs)]:
                        job = self._load_job(job_file)
                        if job:
                            self._start_job_thread(job)
                
                # Update worker state
                self._save_worker_state()
                
                # Sleep before next iteration
                time.sleep(WORKER_SLEEP_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(WORKER_SLEEP_INTERVAL)
    
    def _cleanup_completed_jobs(self):
        """Remove completed job threads."""
        completed_jobs = []
        for job_id, thread in self.current_jobs.items():
            if not thread.is_alive():
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            del self.current_jobs[job_id]
    
    def _get_pending_jobs(self) -> List[str]:
        """Get list of pending job files."""
        try:
            job_files = []
            for filename in os.listdir(QUEUE_DIR):
                if filename.startswith('job_') and filename.endswith('.json'):
                    filepath = os.path.join(QUEUE_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            job = json.load(f)
                        if job.get('status') == 'pending':
                            job_files.append(filename)
                    except Exception as e:
                        logger.warning(f"Could not read job file {filename}: {e}")
            
            # Sort by creation time (oldest first)
            job_files.sort(key=lambda f: os.path.getctime(os.path.join(QUEUE_DIR, f)))
            return job_files
            
        except Exception as e:
            logger.error(f"Error getting pending jobs: {e}")
            return []
    
    def _load_job(self, job_file: str) -> Optional[Dict]:
        """Load job from file."""
        try:
            filepath = os.path.join(QUEUE_DIR, job_file)
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading job {job_file}: {e}")
            return None
    
    def _start_job_thread(self, job: Dict):
        """Start a thread to process a job."""
        job_id = job['job_id']
        thread = threading.Thread(
            target=self._process_job,
            args=(job,),
            name=f"job_{job_id[:8]}"
        )
        thread.start()
        self.current_jobs[job_id] = thread
        logger.info(f"Started processing job {job_id}")
    
    def _process_job(self, job: Dict):
        """Process a single batch job."""
        job_id = job['job_id']
        operation = job['operation']
        
        try:
            # Update job status to running
            self._update_job_status(job_id, 'running', progress=0.0)
            
            if operation == 'quantum_transformation_batch':
                self._process_transformation_batch(job)
            elif operation == 'gutenberg_analysis_batch':
                self._process_gutenberg_batch(job)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Mark job as completed
            self._update_job_status(job_id, 'completed', progress=1.0)
            self.stats['jobs_succeeded'] += 1
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self._update_job_status(job_id, 'failed', error=str(e))
            self.stats['jobs_failed'] += 1
        
        finally:
            self.stats['jobs_processed'] += 1
            self.stats['last_activity'] = datetime.now().isoformat()
    
    def _process_transformation_batch(self, job: Dict):
        """Process a batch of quantum transformations."""
        job_id = job['job_id']
        items = job['items']
        parameters = job['parameters']
        
        results = []
        total_items = len(items)
        
        for i, item in enumerate(items):
            try:
                # Extract transformation parameters
                text = item.get('text', '')
                transformation_type = item.get('transformation_type', parameters.get('default_transformation', 'enhance_narrative'))
                strength = item.get('strength', parameters.get('default_strength', 0.7))
                
                # Call hybrid transformation API for intelligent strategy selection
                response = requests.post(
                    f"{self.api_base_url}/transformations/hybrid-apply",
                    json={
                        "text": text,
                        "transformation_name": transformation_type,
                        "strength": strength,
                        "creativity_level": parameters.get('creativity', 0.8),
                        "preservation_level": parameters.get('preservation', 0.8),
                        "complexity_target": parameters.get('complexity', 0.5),
                        "auto_strategy": True
                    },
                    timeout=300  # 5 minute timeout per transformation
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    results.append({
                        "item_index": i,
                        "success": True,
                        "original_text": text[:200] + "..." if len(text) > 200 else text,
                        "transformed_text": result_data.get('transformed_text', ''),
                        "quantum_distance": result_data.get('quantum_distance', 0.0),
                        "audit_trail": result_data.get('audit_trail', {}),
                        "processing_time": result_data.get('processing_time', 0.0),
                        "strategy_used": result_data.get('strategy_used', 'unknown'),
                        "decision_reasoning": result_data.get('decision_reasoning', ''),
                        "quality_assessment": result_data.get('quality_assessment', {}),
                        "recommendations": result_data.get('recommendations', [])
                    })
                else:
                    results.append({
                        "item_index": i,
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "original_text": text[:200] + "..." if len(text) > 200 else text
                    })
                
                # Update progress
                progress = (i + 1) / total_items
                self._update_job_status(job_id, 'running', progress=progress)
                
                # Brief pause between transformations
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing item {i} in job {job_id}: {e}")
                results.append({
                    "item_index": i,
                    "success": False,
                    "error": str(e),
                    "original_text": text[:200] + "..." if len(text) > 200 else text
                })
        
        # Save results
        self._save_job_results(job_id, {
            "operation": "quantum_transformation_batch",
            "total_items": total_items,
            "successful_items": len([r for r in results if r.get('success')]),
            "failed_items": len([r for r in results if not r.get('success')]),
            "results": results,
            "parameters": parameters,
            "completed_at": datetime.now().isoformat()
        })
    
    def _process_gutenberg_batch(self, job: Dict):
        """Process a batch of Gutenberg book analyses."""
        job_id = job['job_id']
        items = job['items']
        parameters = job['parameters']
        
        results = []
        total_items = len(items)
        
        for i, item in enumerate(items):
            try:
                book_id = item.get('book_id')
                analysis_type = item.get('analysis_type', 'full_transformation')
                
                if analysis_type == 'full_transformation':
                    # Get book text
                    book_response = requests.get(
                        f"{self.api_base_url}/gutenberg/books/{book_id}",
                        timeout=60
                    )
                    
                    if book_response.status_code == 200:
                        book_data = book_response.json()
                        text = book_data['text']
                        
                        # Limit text length for batch processing
                        max_chars = parameters.get('max_chars_per_book', 10000)
                        if len(text) > max_chars:
                            text = text[:max_chars]
                        
                        # Transform the text
                        transform_response = requests.post(
                            f"{self.api_base_url}/transformations/demo-apply",
                            json={
                                "text": text,
                                "transformation_name": parameters.get('transformation_type', 'enhance_narrative'),
                                "strength": parameters.get('strength', 0.7)
                            },
                            timeout=300
                        )
                        
                        if transform_response.status_code == 200:
                            transform_data = transform_response.json()
                            results.append({
                                "item_index": i,
                                "book_id": book_id,
                                "book_title": book_data.get('title', 'Unknown'),
                                "success": True,
                                "quantum_distance": transform_data.get('quantum_distance', 0.0),
                                "text_length": len(text),
                                "transformed_length": len(transform_data.get('transformed_text', '')),
                                "processing_time": transform_data.get('processing_time', 0.0)
                            })
                        else:
                            results.append({
                                "item_index": i,
                                "book_id": book_id,
                                "success": False,
                                "error": f"Transform API error: {transform_response.status_code}"
                            })
                    else:
                        results.append({
                            "item_index": i,
                            "book_id": book_id,
                            "success": False,
                            "error": f"Book API error: {book_response.status_code}"
                        })
                
                # Update progress
                progress = (i + 1) / total_items
                self._update_job_status(job_id, 'running', progress=progress)
                
                # Pause between books
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing book {book_id} in job {job_id}: {e}")
                results.append({
                    "item_index": i,
                    "book_id": book_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Save results
        self._save_job_results(job_id, {
            "operation": "gutenberg_analysis_batch",
            "total_items": total_items,
            "successful_items": len([r for r in results if r.get('success')]),
            "failed_items": len([r for r in results if not r.get('success')]),
            "results": results,
            "parameters": parameters,
            "completed_at": datetime.now().isoformat()
        })
    
    def _update_job_status(self, job_id: str, status: str, progress: float = None, error: str = None):
        """Update job status in persistent storage."""
        try:
            job_file = os.path.join(QUEUE_DIR, f"job_{job_id}.json")
            
            if os.path.exists(job_file):
                with open(job_file, 'r') as f:
                    job = json.load(f)
                
                job['status'] = status
                if progress is not None:
                    job['progress'] = progress
                if error:
                    job['error'] = error
                if status in ['completed', 'failed']:
                    job['completed_at'] = time.time()
                
                with open(job_file, 'w') as f:
                    json.dump(job, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error updating job status for {job_id}: {e}")
    
    def _save_job_results(self, job_id: str, results: Dict):
        """Save job results to disk."""
        try:
            result_file = os.path.join(RESULTS_DIR, f"result_{job_id}.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved results for job {job_id}")
        except Exception as e:
            logger.error(f"Error saving results for job {job_id}: {e}")
    
    def _save_worker_state(self):
        """Save worker state for monitoring."""
        try:
            state = {
                "running": self.running,
                "current_jobs": list(self.current_jobs.keys()),
                "stats": self.stats,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(WORKER_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving worker state: {e}")


def main():
    """Main entry point for running the batch worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Batch Worker')
    parser.add_argument('--check-once', action='store_true', 
                       help='Check for jobs once and exit (for cron)')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as continuous daemon')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.check_once:
        # Single run mode for cron
        logger.info("Quantum Batch Worker - Single Check Mode")
        worker = BatchWorker()
        
        try:
            # Check for jobs and process any available
            pending_jobs = worker._get_pending_jobs()
            if pending_jobs:
                logger.info(f"Found {len(pending_jobs)} pending jobs")
                worker.start()
                
                # Wait for jobs to complete or timeout after 10 minutes
                timeout = 600  # 10 minutes
                start_time = time.time()
                
                while worker.running and (time.time() - start_time) < timeout:
                    time.sleep(5)
                    if len(worker.current_jobs) == 0:
                        # No jobs running, we can exit
                        break
                
                worker.stop()
            else:
                logger.info("No pending jobs found")
                
        except Exception as e:
            logger.error(f"Error in single check mode: {e}")
            
    else:
        # Continuous daemon mode
        logger.info("Starting Quantum Batch Worker - Daemon Mode")
        
        worker = BatchWorker()
        
        try:
            worker.start()
            
            # Keep main thread alive
            while worker.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            worker.stop()


if __name__ == "__main__":
    main()