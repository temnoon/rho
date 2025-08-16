"""
FastAPI application for the Rho Quantum Narrative System.

This is the main application file that sets up the FastAPI app and includes
all the route modules. It should remain small and focused on app configuration.
"""

import os
import logging
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Union

# Import configuration
from config import config, DIM, DATA_DIR

# Import route modules
from routes.matrix_routes import router as matrix_router
from routes.povm_routes import router as povm_router
from routes.advanced_routes import router as advanced_router
from routes.database_routes import router as database_router
from routes.attribute_routes import router as attribute_router
from routes.queue_routes import router as queue_router
from routes.povm_attribute_routes import router as povm_attribute_router
from routes.narrative_attribute_routes import router as narrative_attribute_router
from routes.un_earthify_routes import router as un_earthify_router
from routes.channel_audit_routes import router as channel_audit_router
from routes.integrability_routes import router as integrability_router
from routes.channel_composition_routes import router as channel_composition_router
from routes.persistence_routes import router as persistence_router
from routes.matrix_library_routes import router as matrix_library_router
from routes.aplg_routes import router as aplg_router
from routes.residue_routes import router as residue_router
from routes.consent_routes import router as consent_router
from routes.aplg_test_routes import router as aplg_test_router
from routes.invariant_editor_routes import router as invariant_editor_router
from routes.curriculum_routes import router as curriculum_router
from routes.visualization_routes import router as visualization_router
from routes.transformation_routes import router as transformation_router
from routes.gutenberg_routes import router as gutenberg_router
from utils.persistence import auto_saver, load_state, load_packs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request models
class BookIngestionReq(BaseModel):
    gutenberg_id: Union[str, int]
    chunk_size: int = 300  # characters per chunk (smaller for more chunks)
    reading_alpha: float = 0.3
    
    @field_validator('gutenberg_id')
    @classmethod
    def validate_gutenberg_id(cls, v):
        return str(v)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Rho Quantum Narrative System")
    
    # Load persisted data
    from routes.matrix_routes import STATE
    from routes.povm_routes import PACKS
    
    loaded_state = load_state()
    if loaded_state:
        STATE.update(loaded_state)
        logger.info(f"Loaded {len(loaded_state)} quantum states")
    
    loaded_packs = load_packs()
    if loaded_packs:
        PACKS.update(loaded_packs)
        logger.info(f"Loaded {len(loaded_packs)} POVM packs")
    
    # Start extended auto-saver
    from utils.extended_persistence import extended_auto_saver, load_distillation_sessions, load_exported_embeddings
    
    additional_getters = {
        'distillation_sessions': load_distillation_sessions,
        'exported_embeddings': load_exported_embeddings
    }
    
    extended_auto_saver.start(
        state_getter=lambda: STATE,
        packs_getter=lambda: PACKS,
        additional_data_getters=additional_getters
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Rho Quantum Narrative System")
    extended_auto_saver.stop()


# Create FastAPI app
app = FastAPI(
    title="Rho Quantum Narrative System",
    description="A quantum-inspired system for narrative analysis and generation using density matrices and POVM measurements",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(matrix_router)
app.include_router(povm_router)
app.include_router(advanced_router)
app.include_router(database_router)
app.include_router(attribute_router)
app.include_router(queue_router)
app.include_router(povm_attribute_router)
app.include_router(narrative_attribute_router)
app.include_router(un_earthify_router)
app.include_router(channel_audit_router)
app.include_router(integrability_router)
app.include_router(channel_composition_router)
app.include_router(persistence_router)
app.include_router(matrix_library_router)
app.include_router(aplg_router)
app.include_router(residue_router)
app.include_router(consent_router)
app.include_router(aplg_test_router)
app.include_router(invariant_editor_router)
app.include_router(curriculum_router)
app.include_router(visualization_router)
app.include_router(transformation_router)
app.include_router(gutenberg_router)

# Basic routes
@app.get("/")
def root():
    """Root endpoint with system information."""
    from routes.matrix_routes import get_matrix_count
    from routes.povm_routes import get_pack_count
    
    return {
        "message": "Rho Quantum Narrative System",
        "version": "1.0.0",
        "dimension": DIM,
        "matrices": get_matrix_count(),
        "povm_packs": get_pack_count(),
        "status": "operational"
    }


@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    from routes.matrix_routes import get_matrix_count
    from routes.povm_routes import get_pack_count
    
    return {
        "ok": True,
        "dim": DIM,
        "packs": get_pack_count(),
        "rhos": get_matrix_count()
    }


@app.post("/embed")
def embed_endpoint(text: str):
    """
    Expose the embedding bridge for debugging. Returns raw global embedding.
    """
    from core.embedding import embed
    
    if not text.strip():
        return {"error": "Empty text provided"}
    
    embedding = embed(text.strip())
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "embedding": embedding.tolist(),
        "dimension": len(embedding)
    }


@app.post("/project")
def project_endpoint(request: dict):
    """
    Project a given embedding (or text to be embedded) into local 64-D vector v (unit-norm).
    """
    from core.embedding import embed, project_to_local
    import numpy as np
    
    text = request.get("text")
    embedding = request.get("embedding")
    
    if text:
        x = embed(text.strip())
    elif embedding:
        x = np.array(embedding)
    else:
        return {"error": "Must provide either text or embedding"}
    
    v = project_to_local(x)
    
    return {
        "local_vector": v.tolist(),
        "norm": float(np.linalg.norm(v)),
        "dimension": DIM
    }


# Admin routes for testing/debugging
@app.post("/admin/save_all")
def admin_save_all():
    """Save state and packs to disk."""
    from routes.matrix_routes import STATE
    from routes.povm_routes import PACKS
    from utils.persistence import save_state, save_packs
    
    state_saved = save_state(STATE)
    packs_saved = save_packs(PACKS)
    
    return {
        "state_saved": state_saved,
        "packs_saved": packs_saved,
        "matrices": len(STATE),
        "packs": len(PACKS)
    }


@app.post("/admin/load_all")
def admin_load_all():
    """Load state and packs from disk."""
    from routes.matrix_routes import STATE
    from routes.povm_routes import PACKS
    from utils.persistence import load_state, load_packs
    
    # Clear current data
    STATE.clear()
    PACKS.clear()
    
    # Load from disk
    loaded_state = load_state()
    loaded_packs = load_packs()
    
    STATE.update(loaded_state)
    PACKS.update(loaded_packs)
    
    return {
        "loaded": True,
        "matrices": len(STATE),
        "packs": len(PACKS)
    }


@app.post("/admin/clear_all")
def admin_clear_all():
    """Clear all data (for testing)."""
    from routes.matrix_routes import STATE, clear_all_state
    from routes.povm_routes import PACKS, clear_all_packs
    
    clear_all_state()
    clear_all_packs()
    
    return {
        "cleared": True,
        "matrices": len(STATE),
        "packs": len(PACKS)
    }


@app.get("/gutenberg/search/{query}")
def gutenberg_search(query: str, limit: int = 20, offset: int = 0, author_filter: str = ""):
    """Search Gutenberg books using real API"""
    try:
        from core.gutenberg_integration import gutenberg_client
        
        books = gutenberg_client.search_books(
            query=query, 
            author=author_filter, 
            limit=limit
        )
        
        # Convert to expected format
        results = []
        for book in books[offset:offset + limit]:
            results.append({
                "id": book.id,
                "title": book.title,
                "author": book.author,
                "url": book.url,
                "subjects": book.subjects,
                "download_count": book.download_count
            })
        
        return {
            "books": results,
            "total": len(books),
            "query": query,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Gutenberg search failed: {e}")
        # Fallback to mock data
        mock_books = [
            {"id": 35, "title": "The Time Machine", "author": "H.G. Wells", "url": "https://gutenberg.org/ebooks/35", "subjects": ["Science Fiction"], "download_count": 1000},
            {"id": 36, "title": "The War of the Worlds", "author": "H.G. Wells", "url": "https://gutenberg.org/ebooks/36", "subjects": ["Science Fiction"], "download_count": 800},
            {"id": 5230, "title": "The Invisible Man", "author": "H.G. Wells", "url": "https://gutenberg.org/ebooks/5230", "subjects": ["Science Fiction"], "download_count": 600},
            {"id": 159, "title": "The Island of Doctor Moreau", "author": "H.G. Wells", "url": "https://gutenberg.org/ebooks/159", "subjects": ["Science Fiction"], "download_count": 400},
        ]
        
        results = [book for book in mock_books if query.lower() in book["title"].lower() or query.lower() in book["author"].lower()]
        
        if author_filter:
            results = [book for book in results if author_filter.lower() in book["author"].lower()]
        
        paginated_results = results[offset:offset + limit]
        
        return {
            "books": paginated_results,
            "total": len(results),
            "query": query,
            "limit": limit,
            "offset": offset
        }

@app.post("/gutenberg/ingest")
def ingest_gutenberg_book(req: BookIngestionReq):
    """Download and process a Project Gutenberg book"""
    import time
    import numpy as np
    
    logger.info(f"Ingesting Gutenberg book {req.gutenberg_id}")
    
    try:
        from core.gutenberg_integration import gutenberg_client
        from routes.matrix_routes import STATE, rho_init
        
        # Get book text
        book_text = gutenberg_client.get_book_text(int(req.gutenberg_id))
        if not book_text:
            raise HTTPException(status_code=404, detail=f"Could not retrieve book {req.gutenberg_id}")
        
        # Extract passages with smaller chunks to get hundreds instead of 20
        passages = gutenberg_client.extract_passages(book_text, min_length=req.chunk_size//2, max_length=req.chunk_size*2)
        logger.info(f"Extracted {len(passages)} passages from book {req.gutenberg_id} (text length: {len(book_text)} chars)")
        
        # Create a matrix for this book
        rho_response = rho_init(label=f"Book {req.gutenberg_id}")
        book_rho_id = rho_response["rho_id"]
        
        # Extract title/author from Gutendex API or text parsing
        title, author = gutenberg_client.get_book_metadata(int(req.gutenberg_id))
        
        # Store book metadata
        if book_rho_id in STATE:
            STATE[book_rho_id]["meta"] = {
                "type": "book",
                "gutenberg_id": req.gutenberg_id,
                "title": title,
                "author": author,
                "total_chunks": len(passages),
                "chunks_processed": 0,
                "chunks": passages
            }
        
        return {
            "book_rho_id": book_rho_id,
            "title": title,
            "author": author,
            "total_chunks": len(passages),
            "text_length": len(book_text),
            "ready_for_reading": True
        }
        
    except Exception as e:
        logger.error(f"Failed to ingest book {req.gutenberg_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest book: {str(e)}")

@app.post("/gutenberg/{book_rho_id}/read_chunk/{chunk_index}")
def read_book_chunk(book_rho_id: str, chunk_index: int):
    """Read a specific chunk of a book into the matrix"""
    from routes.matrix_routes import STATE, rho_read
    from models.requests import ReadReq
    
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book not found")
    
    try:
        book_data = STATE[book_rho_id]
        chunks = book_data.get("meta", {}).get("chunks", [])
        
        if chunk_index >= len(chunks):
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        chunk_text = chunks[chunk_index]
        
        # Read the chunk into the matrix
        read_request = ReadReq(
            raw_text=chunk_text,
            alpha=book_data.get("meta", {}).get("reading_alpha", 0.3)
        )
        read_response = rho_read(book_rho_id, read_request)
        
        # Update progress
        book_data["meta"]["chunks_processed"] = max(
            book_data["meta"]["chunks_processed"], 
            chunk_index + 1
        )
        
        return {
            "success": True,
            "chunk_index": chunk_index,
            "passage_preview": chunk_text,  # Show full text instead of truncated
            "rho_id": book_rho_id,
            "chunks_processed": book_data["meta"]["chunks_processed"],
            "total_chunks": book_data["meta"]["total_chunks"]
        }
        
    except Exception as e:
        logger.error(f"Failed to read chunk {chunk_index} for book {book_rho_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read chunk: {str(e)}")

@app.get("/gutenberg/{book_rho_id}/progress")
def get_book_progress(book_rho_id: str):
    """Get reading progress for a book"""
    from routes.matrix_routes import STATE
    import numpy as np
    
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book_data = STATE[book_rho_id]
    meta = book_data.get("meta", {})
    rho = book_data.get("rho")
    
    total_chunks = meta.get("total_chunks", 0)
    chunks_processed = meta.get("chunks_processed", 0)
    completion_percentage = (chunks_processed / total_chunks * 100) if total_chunks > 0 else 0
    
    # Calculate matrix state properties
    matrix_state = {
        "eigs": [0.0] * 64,  # Default eigenvalues
        "purity": 0.0,
        "entropy": 0.0
    }
    
    if rho is not None:
        try:
            # Get eigenvalues
            eigenvals = np.linalg.eigvals(rho)
            eigenvals = np.real(eigenvals[eigenvals.real > 1e-10])  # Filter out near-zero values
            eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
            
            # Pad or truncate to 64 values
            eigs_list = eigenvals.tolist()
            if len(eigs_list) < 64:
                eigs_list.extend([0.0] * (64 - len(eigs_list)))
            elif len(eigs_list) > 64:
                eigs_list = eigs_list[:64]
            
            matrix_state["eigs"] = eigs_list
            
            # Calculate purity (tr(ρ²))
            rho_squared = np.dot(rho, rho)
            purity = float(np.real(np.trace(rho_squared)))
            matrix_state["purity"] = max(0.0, min(1.0, purity))
            
            # Calculate von Neumann entropy
            entropy = 0.0
            for ev in eigenvals:
                if ev > 1e-10:
                    entropy -= ev * np.log2(ev)
            matrix_state["entropy"] = float(entropy)
            
        except Exception as e:
            logger.warning(f"Failed to calculate matrix state for {book_rho_id}: {e}")
    
    return {
        "book_rho_id": book_rho_id,
        "total_chunks": total_chunks,
        "chunks_processed": chunks_processed,
        "completion_percentage": completion_percentage,
        "title": meta.get("title", "Unknown"),
        "author": meta.get("author", "Unknown"),
        "matrix_state": matrix_state
    }


@app.get("/matrices/available")
def available_matrices():
    """Get list of available matrices"""
    from routes.matrix_routes import STATE
    
    matrices = []
    for rho_id, state_data in STATE.items():
        matrices.append({
            "id": rho_id,
            "label": state_data.get("label", f"rho_{rho_id[:8]}"),
            "created": state_data.get("created_at", "unknown"),
            "narratives_count": len(state_data.get("narratives", [])),
            "operations_count": len(state_data.get("ops", [])),
            "dual_pair": state_data.get("dual_pair", None)
        })
    
    return {"matrices": matrices, "count": len(matrices)}


@app.get("/attributes/list")
def list_attributes():
    """List all available attributes from POVM packs."""
    from routes.povm_routes import PACKS
    
    attributes = {}
    
    for pack_id, pack_data in PACKS.items():
        pack_attributes = {}
        for axis in pack_data.get("axes", []):
            axis_id = axis.get("id")
            if axis_id:
                pack_attributes[axis_id] = {
                    "labels": axis.get("labels", ["low", "high"]),
                    "description": axis.get("description", ""),
                    "category": axis.get("category", "uncategorized")
                }
        
        if pack_attributes:
            attributes[pack_id] = {
                "description": pack_data.get("description", ""),
                "attributes": pack_attributes
            }
    
    return attributes




@app.post("/admin/load_advanced_pack")
def load_advanced_pack():
    """Load the advanced narrative POVM pack from JSON file."""
    import json
    import numpy as np
    from fastapi import HTTPException
    from routes.povm_routes import PACKS
    
    pack_file = os.path.join(DATA_DIR, "advanced_narrative_pack.json")
    
    try:
        with open(pack_file, 'r', encoding='utf-8') as f:
            pack_data = json.load(f)
        
        # Generate actual POVM effects for each axis
        for axis in pack_data["axes"]:
            if not axis.get("effects"):
                # Create random POVM effects for demonstration
                # In a real implementation, these would be learned from data
                dim = 64
                effect_pos = np.random.rand(dim, dim)
                effect_pos = effect_pos @ effect_pos.T  # Make positive semidefinite
                effect_pos = effect_pos / np.trace(effect_pos)  # Normalize
                
                effect_neg = np.eye(dim) - effect_pos  # Complementary effect
                effect_neg = np.maximum(effect_neg, 0)  # Ensure non-negative
                effect_neg = effect_neg / np.trace(effect_neg) if np.trace(effect_neg) > 0 else effect_neg
                
                axis["effects"] = [effect_neg.tolist(), effect_pos.tolist()]
                axis["num_effects"] = 2
        
        # Store the pack
        PACKS[pack_data["pack_id"]] = pack_data
        
        logger.info(f"Loaded advanced narrative pack with {len(pack_data['axes'])} axes")
        
        return {
            "success": True,
            "pack_id": pack_data["pack_id"],
            "axes_loaded": len(pack_data["axes"]),
            "categories": list(set(axis.get("category", "uncategorized") for axis in pack_data["axes"]))
        }
        
    except Exception as e:
        logger.error(f"Failed to load advanced pack: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load pack: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.API_PORT)