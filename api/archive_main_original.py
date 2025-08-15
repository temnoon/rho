#!/usr/bin/env python3
"""
FastAPI backend for the Rho Small‑Embedding Demo — Humanizer Node

Provides:
- /rho/init            : create a new rho state
- /rho/{rho_id}        : fetch current rho + diagnostics
- /rho/{rho_id}/read   : read text (alpha) -> update rho
- /rho/{rho_id}/measure: apply POVM pack(s)
- /rho/{rho_id}/reset  : reset rho (identity or seed text)
- /packs               : list/add packs
- /math/eig            : eigendecomposition diagnostics for a given rho
- /embed               : optional embedding bridge (pluggable)
- /project             : project global embedding x -> local 64-D v via W
- /explain             : structured explanation of last operation (simple)

Notes:
- This is intentionally minimal and transparent for demo / teaching.
- Replace the `embed()` function or provide EMBED_URL environment variable to
  integrate with real embedding services (bge-small, MiniLM, E5, etc).
- If a W matrix file is present at `data/w.npy` and its second dim != 64, it's used
  to project global embeddings to 64-D before normalization.
"""

from __future__ import annotations
import os
import sys
import time
import json
import uuid
import math
import logging
import atexit
import threading
from typing import Any, Dict, List, Optional

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests

# --- Configuration ---
DIM = 64
DATA_DIR = os.environ.get("RHO_DATA_DIR", "rho/api/data")
W_FILE = os.path.join(DATA_DIR, "w.npy")
PORT = int(os.environ.get("PORT", "8000"))
EMBED_URL = os.environ.get("EMBED_URL")  # optional embedding service (POST {"text": "..."} -> {"embedding": [...]})
EMBED_DIM_DEFAULT = 64  # used when EMBED_URL is not provided

os.makedirs(DATA_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rho_demo")

# Import new rho space design and transformation modules
try:
    from rho_space_designer import RhoSpaceDesigner, RhoSpaceDesign, get_narrative_space_design
    from rho_transformations import RhoTransformationEngine, TransformationLibrary
    from rho_text_generator import enhance_transformation_engine_with_generation, create_rho_text_generator
    from attribute_library import get_attribute_library, AttributeDefinition
    RHO_DESIGN_AVAILABLE = True
    logger.info("Rho space design, transformation, text generation, and attribute library modules loaded successfully")
except ImportError as e:
    logger.warning(f"Rho design modules not available: {e}")
    RHO_DESIGN_AVAILABLE = False

# --- App init ---
app = FastAPI(title="Humanizer Rho Demo", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory / on-disk storage (simple) ---
STATE: Dict[str, Dict[str, Any]] = {}  # rho_id -> {"rho": np.ndarray, "log": [ops], "meta": {...}}

# Auto-persistence settings
AUTO_SAVE_ENABLED = True
AUTO_SAVE_INTERVAL = 30  # seconds
PERSISTENCE_FILE = "rho_state_auto.json"
PACKS: Dict[str, Dict[str, Any]] = {}  # pack_id -> pack dict

# --- Utilities / Helpers ---


def save_json_atomic(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def learn_projection_matrix_from_samples() -> np.ndarray:
    """
    Learn a projection matrix W from a sample of texts using PCA.
    This creates a mapping from embedding space to 64D that preserves
    the most important semantic dimensions.
    """
    logger.info("Learning projection matrix W from sample texts...")
    
    # Sample texts covering different domains and styles
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
        "The meeting will discuss quarterly revenue projections and market analysis.",
        "I can't believe it's already Monday morning again.",
        "The quantum superposition of particles enables remarkable computational possibilities.",
        "She gazed wistfully at the sunset, remembering their last conversation.",
        "The algorithm efficiently processes large datasets using parallel computation.",
        "Love is a temporary madness erupting like volcanoes and then subsiding.",
        "Scientific evidence suggests that climate change accelerates ecosystem disruption.",
        "The old oak tree stood majestically in the center of the meadow.",
        "Breaking news: The stock market responds to unexpected economic indicators.",
        "Children laughed as they played in the fountain's cooling spray.",
        "The philosophical implications of consciousness remain deeply mysterious.",
        "Recipe: Combine flour, eggs, and milk to create a smooth batter.",
        "The detective carefully examined the evidence left at the crime scene.",
        "Meditation brings peace to the restless mind and troubled heart.",
        "Advanced robotics revolutionizes manufacturing processes worldwide.",
        "Her smile illuminated the room like sunshine after rain."
    ]
    
    try:
        # Get embeddings for all sample texts  
        embeddings = []
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for text in sample_texts:
            # Get raw embedding without W projection
            emb = model.encode(text).astype(float)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        logger.info(f"Collected {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # Use PCA to find the most important dimensions (limited by sample size)
        from sklearn.decomposition import PCA
        n_components = min(DIM, len(embeddings), embeddings.shape[1])
        logger.info(f"Using PCA with {n_components} components (limited by {len(embeddings)} samples)")
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        
        # W is the PCA components matrix (DIM x embedding_dim)
        W_components = pca.components_
        
        # If we have fewer components than DIM, pad with zeros or random vectors
        if W_components.shape[0] < DIM:
            logger.info(f"Padding W matrix from {W_components.shape[0]} to {DIM} components")
            embedding_dim = W_components.shape[1]
            W = np.zeros((DIM, embedding_dim))
            W[:W_components.shape[0], :] = W_components
            # Fill remaining rows with small random values
            for i in range(W_components.shape[0], DIM):
                W[i, :] = np.random.normal(0, 0.01, embedding_dim)
                W[i, :] /= np.linalg.norm(W[i, :]) + 1e-12  # Normalize
        else:
            W = W_components
        
        # Save the learned matrix
        np.save(W_FILE, W)
        logger.info(f"Learned and saved W matrix with shape {W.shape} to {W_FILE}")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5
        
        return W
        
    except Exception as e:
        logger.exception("Failed to learn projection matrix, falling back to identity")
        # Return an identity matrix that matches expected embedding dimensions
        # Default to 384 for sentence-transformers all-MiniLM-L6-v2
        return np.eye(DIM, 384)

def load_w_matrix() -> np.ndarray:
    """
    Load projection W if present. Expected shape: (64, m) or (64,64).
    If absent, learn it from sample data.
    """
    if os.path.exists(W_FILE):
        try:
            W = np.load(W_FILE)
            if W.shape[0] != DIM:
                logger.warning("W matrix first dim != DIM, forcing to DIM by trunc/pad")
            # If W has wrong first dim, try to fix/truncate/pad
            if W.shape[0] < DIM:
                W = np.pad(W, ((0, DIM - W.shape[0]), (0, 0)))
            elif W.shape[0] > DIM:
                W = W[:DIM, :]
            logger.info(f"Loaded W matrix from {W_FILE} with shape {W.shape}")
            return W
        except Exception as e:
            logger.exception("Failed to load W matrix; will learn new one.")
    
    # Learn a new projection matrix
    logger.info("No W matrix found, learning from sample data...")
    return learn_projection_matrix_from_samples()


W = load_w_matrix()  # shape (DIM, m) or (DIM, DIM)


def psd_project(A: np.ndarray) -> np.ndarray:
    """
    Make symmetric, clip negative eigenvalues, reconstruct, normalize trace to 1.
    """
    A = 0.5 * (A + A.T)
    try:
        w, V = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        # fallback: small identity
        logger.exception("eigh failed in psd_project; returning maximally mixed state")
        return np.eye(DIM) / DIM
    w = np.clip(w, 0.0, None)
    A = (V * w) @ V.T
    tr = float(np.trace(A))
    if tr <= 0 or not np.isfinite(tr):
        return np.eye(DIM) / DIM
    return A / tr


def diagnostics(rho: np.ndarray, top_k: int = 8) -> Dict[str, Any]:
    """Return trace, purity, entropy, eigvals (descending)"""
    rho_sym = 0.5 * (rho + rho.T)
    # ensure numeric stability
    try:
        w = np.linalg.eigvalsh(rho_sym)
    except np.linalg.LinAlgError:
        logger.exception("eigvalsh failed in diagnostics; using fallback")
        w = np.ones(DIM) / DIM
    w = np.clip(w, 1e-12, None)
    purity = float(np.sum(w * w))
    entropy = float(-np.sum(w * np.log(w)))
    eigs_sorted = sorted([float(x) for x in w], reverse=True)
    return {
        "dim": DIM,
        "trace": float(np.trace(rho_sym)),
        "purity": purity,
        "entropy": entropy,
        "eigs": eigs_sorted[: max(top_k, 1)],
    }


def deterministic_embed_stub(text: str, out_dim: int) -> np.ndarray:
    """
    Deterministic pseudo-embedding for demo/testing: uses a hash to seed RNG.
    Returns unit-normalized vector of length out_dim.
    """
    h = abs(hash(text)) & 0xFFFFFFFF
    rng = np.random.default_rng(h)
    x = rng.standard_normal(out_dim)
    norm = np.linalg.norm(x) + 1e-12
    x = x / norm
    return x


def embed(text: str) -> np.ndarray:
    """
    Real semantic embedding function with multiple backends:
    1. Ollama with nomic-embed-text (preferred)
    2. EMBED_URL service (if set)
    3. Local sentence-transformers model
    4. Deterministic stub (fallback only)
    """
    # Try Ollama with nomic-embed-text first
    try:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10.0
        )
        if r.status_code == 200:
            payload = r.json()
            if "embedding" in payload:
                vec = np.array(payload["embedding"], dtype=float)
                logger.debug(f"Using Ollama nomic-embed-text, dim={len(vec)}")
                return vec
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")
    
    # Try external embedding URL if set
    if EMBED_URL:
        try:
            r = requests.post(EMBED_URL, json={"text": text}, timeout=5.0)
            r.raise_for_status()
            payload = r.json()
            for key in ("embedding", "emb", "vector", "vec"):
                if key in payload:
                    vec = np.array(payload[key], dtype=float)
                    logger.debug(f"Using external service, dim={len(vec)}")
                    return vec
            raise ValueError("No embedding in response")
        except Exception as e:
            logger.debug(f"External embedding service failed: {e}")
    
    # Use local sentence-transformers model
    try:
        if not hasattr(embed, '_model'):
            # Load model on first use
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
            embed._model = SentenceTransformer('all-MiniLM-L6-v2')  # 384D, fast and good
        
        vec = embed._model.encode(text)
        logger.debug(f"Using sentence-transformers, dim={len(vec)}")
        return vec.astype(float)
    except Exception as e:
        logger.warning(f"Sentence-transformers failed: {e}")
    
    # Final fallback to deterministic stub (should rarely happen now)
    logger.warning("All embedding methods failed, using deterministic stub")
    return deterministic_embed_stub(text, EMBED_DIM_DEFAULT)


def project_to_local(x: np.ndarray) -> np.ndarray:
    """
    Project global embedding x (shape m,) to local DIM via W (DIM x m).
    If W is identity and x is DIM, it's a nop. Then unit-normalize.
    """
    x = np.asarray(x, dtype=float)
    if W.shape[1] != x.shape[0]:
        # If W second dim doesn't match, attempt to adapt:
        if W.shape[1] < x.shape[0]:
            x2 = x[: W.shape[1]]
        else:
            # pad x
            x2 = np.pad(x, (0, W.shape[1] - x.shape[0]))
    else:
        x2 = x
    v = W @ x2
    norm = np.linalg.norm(v) + 1e-12
    return v / norm


def rho_matrix_to_list(rho: np.ndarray) -> List[List[float]]:
    return rho.tolist()


# --- Pydantic models ---


class ReadReq(BaseModel):
    raw_text: Optional[str] = None
    text_id: Optional[str] = None
    alpha: float = 0.2


class MeasureReq(BaseModel):
    pack_id: str


class PackAxis(BaseModel):
    id: str
    labels: List[str]
    u: List[float]


class PackModel(BaseModel):
    pack_id: str
    axes: List[PackAxis]


class ExplainReq(BaseModel):
    rho_id: str
    last_n: int = 1


# --- Initialize demo packs (hand-crafted simple orthonormal-ish axes) ---


def init_demo_packs() -> None:
    if PACKS:
        return
    # Simple canonical axes along basis vectors 0..11
    axes = []
    for i, axis_name in enumerate(
        [
            "narrator_distance",
            "reliability",
            "affect_valence",
            "arousal",
            "agency",
            "myth_realism",
            "formality",
            "personal_focus",
            "temporal_distance",
            "certainty",
            "politeness",
            "intensity",
        ]
    ):
        u = [0.0] * DIM
        idx = i % DIM
        u[idx] = 1.0
        labels = ["+" + axis_name, "-" + axis_name]
        axes.append({"id": axis_name, "labels": labels, "u": u})
    demo_pack = {"pack_id": "axes_12x2_demo", "axes": axes}
    PACKS[demo_pack["pack_id"]] = demo_pack
    logger.info("Initialized demo POVM pack 'axes_12x2_demo'")


init_demo_packs()

# --- API routes ---


@app.post("/rho/init")
def rho_init(seed_text: Optional[str] = None, label: Optional[str] = None):
    """
    Create a new rho (maximally mixed or seeded from seed_text).
    Returns diagnostics.
    """
    rho_id = str(uuid.uuid4())
    if seed_text:
        x = embed(seed_text)
        v = project_to_local(x)
        rho = np.outer(v, v)
        rho = psd_project(rho)
    else:
        rho = np.eye(DIM) / DIM
    STATE[rho_id] = {"rho": rho, "log": [], "meta": {"label": label or "visitor"}}
    logger.info("Created rho %s (seeded=%s)", rho_id, bool(seed_text))
    return {"rho_id": rho_id, **diagnostics(rho)}


@app.get("/rho/{rho_id}")
def rho_get(rho_id: str):
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    rho = item["rho"]
    return {"rho_id": rho_id, **diagnostics(rho)}


@app.post("/rho/{rho_id}/read")
def rho_read(rho_id: str, req: ReadReq):
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    if not (req.raw_text or req.text_id):
        raise HTTPException(status_code=400, detail="raw_text or text_id required")
    text = req.raw_text or f"[text_id:{req.text_id}]"
    # embed -> project -> pure state -> blend -> psd_project
    x = embed(text)
    # If embedding dimension != DIM, project via W
    v = project_to_local(x)
    pure = np.outer(v, v)
    rho_old = item["rho"]
    alpha = float(req.alpha)
    alpha = max(0.0, min(1.0, alpha))
    rho_new = (1.0 - alpha) * rho_old + alpha * pure
    rho_new = psd_project(rho_new)
    item["rho"] = rho_new
    log_entry = {"ts": time.time(), "op": "read", "alpha": alpha, "text": text}
    item["log"].append(log_entry)
    logger.info("rho %s read op alpha=%.3f text_len=%d", rho_id, alpha, len(text))
    return {"rho_id": rho_id, **diagnostics(rho_new)}


@app.post("/rho/{rho_id}/measure")
def rho_measure(rho_id: str, req: MeasureReq):
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    pack = PACKS.get(req.pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="pack_id not found")
    rho = item["rho"]
    I = np.eye(DIM)
    probs: Dict[str, Dict[str, float]] = {}
    for axis in pack["axes"]:
        u = np.array(axis["u"], dtype=float)
        nu = np.linalg.norm(u) + 1e-12
        if nu == 0:
            # treat as uniformly neutral axis
            p_plus = 0.0
            p_minus = 1.0
        else:
            u_unit = u / nu
            E_plus = np.outer(u_unit, u_unit)
            E_minus = I - E_plus
            p_plus = float(np.trace(E_plus @ rho))
            p_minus = float(np.trace(E_minus @ rho))
        probs[axis["id"]] = {axis["labels"][0]: p_plus, axis["labels"][1]: p_minus}
    # log measurement
    item["log"].append({"ts": time.time(), "op": "measure", "pack_id": req.pack_id, "probs": probs})
    return {"rho_id": rho_id, "pack_id": req.pack_id, "probs": probs}


@app.post("/rho/{rho_id}/reset")
def rho_reset(rho_id: str, seed_text: Optional[str] = None):
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    if seed_text:
        x = embed(seed_text)
        v = project_to_local(x)
        rho = np.outer(v, v)
        rho = psd_project(rho)
    else:
        rho = np.eye(DIM) / DIM
    item["rho"] = rho
    item["log"].append({"ts": time.time(), "op": "reset", "seed": bool(seed_text)})
    return {"rho_id": rho_id, **diagnostics(rho)}


@app.get("/packs")
def packs_list():
    return {"packs": list(PACKS.keys())}


@app.post("/packs")
def packs_add(pack: PackModel):
    if pack.pack_id in PACKS:
        raise HTTPException(status_code=400, detail="pack_id already exists")
    # simple validation of axis lengths
    for axis in pack.axes:
        if len(axis.u) != DIM:
            # accept but normalize/pad/truncate
            u = list(axis.u)
            if len(u) < DIM:
                u = u + [0.0] * (DIM - len(u))
            else:
                u = u[:DIM]
            axis.u = u  # type: ignore
    PACKS[pack.pack_id] = pack.dict()
    logger.info("Added pack %s with %d axes", pack.pack_id, len(pack.axes))
    return {"ok": True, "pack_id": pack.pack_id}


@app.post("/math/eig")
def math_eig(rho_id: str):
    item = STATE.get(rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    rho = 0.5 * (item["rho"] + item["rho"].T)
    try:
        w, V = np.linalg.eigh(rho)
    except np.linalg.LinAlgError:
        raise HTTPException(status_code=500, detail="eigendecomposition failed")
    w = np.clip(w, 0.0, None)
    # return top eigenvalues and corresponding eigenvectors
    idx = np.argsort(w)[::-1]
    top_vals = [float(w[i]) for i in idx[:8]]
    top_vecs = [V[:, i].tolist() for i in idx[:8]]
    return {"rho_id": rho_id, "eigs": top_vals, "vecs": top_vecs}


@app.post("/embed")
def embed_endpoint(text: str):
    """
    Expose the embedding bridge for debugging. Returns raw global embedding.
    Note: embedding might have dim != DIM; call /project to convert to 64-D local v.
    """
    x = embed(text)
    return {"dim": int(x.shape[0]), "embedding": x.tolist()}


@app.post("/project")
def project_endpoint(text: Optional[str] = None, embedding: Optional[List[float]] = None):
    """
    Project a given embedding (or text to be embedded) into local 64-D vector v (unit-norm).
    """
    if text is None and embedding is None:
        raise HTTPException(status_code=400, detail="text or embedding required")
    if embedding is not None:
        x = np.asarray(embedding, dtype=float)
    else:
        x = embed(text)  # type: ignore
    v = project_to_local(x)
    return {"v_dim": int(v.shape[0]), "v": v.tolist()}


@app.post("/explain")
def explain(req: ExplainReq):
    """
    Lightweight explain: return a JSON explanation of the last n operations for a rho_id.
    The explanation is deterministic and focuses on transparency (input -> math -> output).
    """
    item = STATE.get(req.rho_id)
    if item is None:
        raise HTTPException(status_code=404, detail="rho_id not found")
    logs = item.get("log", [])[-req.last_n :]
    explanations = []
    for entry in logs:
        if entry["op"] == "read":
            text = entry.get("text", "[text]")
            alpha = entry.get("alpha", 0.0)
            explanations.append(
                {
                    "op": "read",
                    "summary": f"Read text (len={len(text)}), blended with alpha={alpha:.3f}",
                    "math": "v = project(embed(text));  rho <- (1-alpha)*rho + alpha*|v><v| ; normalize && PSD-project",
                }
            )
        elif entry["op"] == "measure":
            pack_id = entry.get("pack_id")
            explanations.append(
                {
                    "op": "measure",
                    "summary": f"Measured POVM pack {pack_id}",
                    "math": "for each axis u: p_plus = Tr(|u><u| rho); p_minus = 1 - p_plus",
                }
            )
        elif entry["op"] == "reset":
            explanations.append(
                {"op": "reset", "summary": "Reset rho to seed or maximally mixed", "math": "rho = I/D or |v><v|"}
            )
        else:
            explanations.append({"op": entry.get("op"), "summary": "logged op", "math": ""})
    return {"rho_id": req.rho_id, "explanations": explanations}


# --- Auto-persistence functions ---
def auto_save_state():
    """Automatically save state to prevent data loss"""
    if not AUTO_SAVE_ENABLED or not STATE:
        return
    
    try:
        # Use the same logic as admin_save_all but with different filename
        s_path = os.path.join(DATA_DIR, PERSISTENCE_FILE)
        
        # Convert STATE to JSON-serializable format
        serial_state = {}
        for k, v in STATE.items():
            # Handle complex matrices by converting to real/imag parts
            rho_matrix = v["rho"]
            if np.iscomplexobj(rho_matrix):
                rho_serializable = {
                    "real": rho_matrix.real.tolist(),
                    "imag": rho_matrix.imag.tolist()
                }
            else:
                rho_serializable = {"real": rho_matrix.tolist(), "imag": None}
            
            serial_state[k] = {
                "rho": rho_serializable,
                "log": v.get("log", []),
                "meta": v.get("meta", {}),
            }
        
        # Save atomically
        save_json_atomic(s_path, serial_state)
        logger.info(f"Auto-saved {len(STATE)} matrices to {s_path}")
        
    except Exception as e:
        logger.error(f"Auto-save failed: {e}")

def auto_load_state():
    """Load state on startup if available"""
    s_path = os.path.join(DATA_DIR, PERSISTENCE_FILE)
    
    if not os.path.exists(s_path):
        logger.info("No auto-save file found, starting fresh")
        return
    
    try:
        with open(s_path, "r") as f:
            serial_state = json.load(f)
        
        # Clear and reload STATE
        STATE.clear()
        for k, v in serial_state.items():
            # Reconstruct complex matrix from real/imag parts
            rho_data = v["rho"]
            if isinstance(rho_data, dict) and "real" in rho_data:
                real_part = np.array(rho_data["real"], dtype=float)
                imag_part = np.array(rho_data["imag"], dtype=float) if rho_data["imag"] is not None else np.zeros_like(real_part)
                rho = real_part + 1j * imag_part
            else:
                # Legacy format - assume real matrix
                rho = np.array(rho_data, dtype=complex)
            
            STATE[k] = {
                "rho": rho, 
                "log": v.get("log", []), 
                "meta": v.get("meta", {})
            }
        
        logger.info(f"Auto-loaded {len(STATE)} matrices from {s_path}")
        
    except Exception as e:
        logger.error(f"Auto-load failed: {e}")

def periodic_auto_save():
    """Background thread for periodic saving"""
    def save_loop():
        while True:
            time.sleep(AUTO_SAVE_INTERVAL)
            auto_save_state()
    
    if AUTO_SAVE_ENABLED:
        thread = threading.Thread(target=save_loop, daemon=True)
        thread.start()
        logger.info(f"Started auto-save thread (interval: {AUTO_SAVE_INTERVAL}s)")

# --- Simple persistence endpoints (save/load state to disk) ---


@app.post("/admin/save_all")
def admin_save_all():
    """
    Save state and packs to disk under DATA_DIR. Intended for demo / local use.
    """
    try:
        s_path = os.path.join(DATA_DIR, "state.json")
        p_path = os.path.join(DATA_DIR, "packs.json")
        # Convert numpy arrays
        serial_state = {}
        for k, v in STATE.items():
            serial_state[k] = {
                "rho": v["rho"].tolist(),
                "log": v.get("log", []),
                "meta": v.get("meta", {}),
            }
        save_json_atomic(s_path, serial_state)
        save_json_atomic(p_path, PACKS)
        return {"ok": True, "saved_state": s_path, "saved_packs": p_path}
    except Exception as e:
        logger.exception("Failed to save all")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/load_all")
def admin_load_all():
    """
    Load state and packs from disk (if present). Overwrites in-memory STATE and PACKS.
    """
    try:
        s_path = os.path.join(DATA_DIR, "state.json")
        p_path = os.path.join(DATA_DIR, "packs.json")
        if os.path.exists(p_path):
            with open(p_path, "r", encoding="utf-8") as f:
                packs = json.load(f)
            PACKS.clear()
            PACKS.update(packs)
        if os.path.exists(s_path):
            with open(s_path, "r", encoding="utf-8") as f:
                serial_state = json.load(f)
            STATE.clear()
            for k, v in serial_state.items():
                rho = np.array(v["rho"], dtype=float)
                STATE[k] = {"rho": rho, "log": v.get("log", []), "meta": v.get("meta", {})}
        return {"ok": True, "loaded_state": s_path if os.path.exists(s_path) else None}
    except Exception as e:
        logger.exception("Failed to load all")
        raise HTTPException(status_code=500, detail=str(e))


# --- Project Gutenberg Integration ---

import requests
import re
import asyncio
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    URGENT = "urgent"

# Global processing queue
PROCESSING_QUEUE = []
COMPLETED_JOBS = []
QUEUE_LOCK = threading.Lock()
BACKGROUND_PROCESSOR = None

class BookIngestionReq(BaseModel):
    gutenberg_id: str
    chunk_size: int = 500  # characters per chunk
    reading_alpha: float = 0.3

class BatchJobRequest(BaseModel):
    search_queries: List[str]
    instructions: str = "Process books with standard settings for comprehensive analysis"
    chunk_size: int = 400
    reading_alpha: float = 0.25
    priority: JobPriority = JobPriority.MEDIUM
    max_books_per_query: int = 3
    auto_finalize: bool = True

class QueuedJob(BaseModel):
    job_id: str
    book_title: str
    book_author: str
    gutenberg_id: str
    search_query: str
    instructions: str
    chunk_size: int
    reading_alpha: float
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    book_rho_id: Optional[str] = None
    progress: float = 0.0
    chunks_processed: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    agent_notes: Optional[str] = None

def generate_job_id() -> str:
    """Generate unique job ID"""
    return f"job_{int(time.time())}_{hash(str(time.time())) % 10000:04d}"

def intelligent_scheduler_agent(job_request: BatchJobRequest) -> List[QueuedJob]:
    """Intelligent agent that evaluates search queries and creates optimized job queue"""
    
    # Agent analysis of the request
    agent_analysis = f"""
BATCH PROCESSING ANALYSIS:
Query Count: {len(job_request.search_queries)}
Instructions: {job_request.instructions}
Priority Level: {job_request.priority}
Max Books/Query: {job_request.max_books_per_query}

AGENT RECOMMENDATIONS:
"""
    
    jobs_to_queue = []
    
    for query in job_request.search_queries:
        # Search for books matching this query
        try:
            # Use existing search logic
            catalog = [
                {"id": "11", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll"},
                {"id": "1342", "title": "Pride and Prejudice", "author": "Jane Austen"}, 
                {"id": "84", "title": "Frankenstein", "author": "Mary Shelley"},
                {"id": "174", "title": "The Picture of Dorian Gray", "author": "Oscar Wilde"},
                {"id": "74", "title": "The Adventures of Tom Sawyer", "author": "Mark Twain"},
                {"id": "2701", "title": "Moby Dick", "author": "Herman Melville"},
                {"id": "345", "title": "Dracula", "author": "Bram Stoker"},
                {"id": "1661", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle"},
                {"id": "76", "title": "Adventures of Huckleberry Finn", "author": "Mark Twain"},
                {"id": "98", "title": "A Tale of Two Cities", "author": "Charles Dickens"},
                {"id": "1400", "title": "Great Expectations", "author": "Charles Dickens"},
                {"id": "844", "title": "The Importance of Being Earnest", "author": "Oscar Wilde"},
                {"id": "46", "title": "A Christmas Carol", "author": "Charles Dickens"},
                {"id": "1260", "title": "Jane Eyre", "author": "Charlotte Brontë"},
                {"id": "161", "title": "The Secret Garden", "author": "Frances Hodgson Burnett"}
            ]
            
            # Filter books for this query
            query_lower = query.lower()
            matching_books = []
            
            for book in catalog:
                if (query_lower in book["title"].lower() or 
                    query_lower in book["author"].lower() or 
                    any(word in book["title"].lower() for word in query_lower.split()) or
                    any(word in book["author"].lower() for word in query_lower.split())):
                    matching_books.append(book)
            
            # Limit results per query
            matching_books = matching_books[:job_request.max_books_per_query]
            
            # Agent prioritization logic
            for book in matching_books:
                # Calculate intelligent priority based on book characteristics
                book_priority = job_request.priority
                
                # Boost priority for classic literature
                if any(word in book["title"].lower() for word in ["alice", "moby", "frankenstein", "dracula"]):
                    if book_priority == JobPriority.MEDIUM:
                        book_priority = JobPriority.HIGH
                
                # Agent notes about why this book was selected
                agent_notes = f"Selected for query '{query}' - "
                if len(book["title"]) > 30:
                    agent_notes += "Complex narrative expected, will benefit from hierarchical analysis. "
                if any(author in book["author"] for author in ["Dickens", "Shelley", "Carroll"]):
                    agent_notes += "Notable author with rich thematic content. "
                if "adventure" in book["title"].lower():
                    agent_notes += "Adventure narrative - good for testing character development tracking. "
                
                job = QueuedJob(
                    job_id=generate_job_id(),
                    book_title=book["title"],
                    book_author=book["author"],
                    gutenberg_id=book["id"],
                    search_query=query,
                    instructions=job_request.instructions,
                    chunk_size=job_request.chunk_size,
                    reading_alpha=job_request.reading_alpha,
                    priority=book_priority,
                    status=JobStatus.QUEUED,
                    created_at=datetime.now(),
                    agent_notes=agent_notes
                )
                
                jobs_to_queue.append(job)
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
    
    # Sort by priority (urgent first, then high, medium, low)
    priority_order = {JobPriority.URGENT: 4, JobPriority.HIGH: 3, JobPriority.MEDIUM: 2, JobPriority.LOW: 1}
    jobs_to_queue.sort(key=lambda x: priority_order[x.priority], reverse=True)
    
    return jobs_to_queue

@app.get("/gutenberg/search/{query}")
def search_gutenberg(query: str, limit: int = 100, offset: int = 0, author_filter: str = ""):
    """Search Project Gutenberg catalog using their autocomplete API"""
    books = []
    
    try:
        # Use Project Gutenberg's autocomplete API which is more reliable
        search_url = "https://www.gutenberg.org/ebooks/search/"
        params = {
            "query": query if not author_filter else f"{query} {author_filter}",
            "format": "json",
            "max_records": limit
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Gutenberg returns [query, titles[], authors[], urls[]]
                if isinstance(data, list) and len(data) >= 4:
                    titles = data[1] if len(data) > 1 else []
                    authors = data[2] if len(data) > 2 else []
                    urls = data[3] if len(data) > 3 else []
                    
                    # Skip first element which is display info
                    for i in range(1, min(len(titles), len(authors), len(urls)) + 1):
                        if i < len(titles) and i < len(authors) and i < len(urls):
                            title = titles[i] if i < len(titles) else ""
                            author = authors[i] if i < len(authors) else "Unknown Author"
                            url = urls[i] if i < len(urls) else ""
                            
                            # Extract ID from URL like "/ebooks/123.json"
                            book_id = ""
                            if url and "/ebooks/" in url:
                                import re
                                id_match = re.search(r'/ebooks/(\d+)', url)
                                if id_match:
                                    book_id = id_match.group(1)
                            
                            if title and book_id:
                                books.append({
                                    "id": book_id,
                                    "title": title.strip(),
                                    "author": author or "Unknown Author",
                                    "year": "Unknown",
                                    "genre": "Literature"
                                })
                
                if books:
                    # Apply pagination
                    total_results = len(books)
                    paginated_books = books[offset:offset + limit]
                    
                    return {
                        "query": query,
                        "results": paginated_books,
                        "total": total_results,
                        "offset": offset,
                        "limit": limit,
                        "has_more": offset + limit < total_results,
                        "source": "gutenberg_api"
                    }
                    
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse Gutenberg API response: {e}, falling back to catalog")
        
    except requests.RequestException as e:
        logger.warning(f"Gutenberg API request failed: {e}, falling back to catalog")
    
    # Fallback to curated catalog if API fails
    catalog = [
        {"id": "11", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "year": "1865", "genre": "Fantasy"},
        {"id": "1342", "title": "Pride and Prejudice", "author": "Jane Austen", "year": "1813", "genre": "Romance"}, 
        {"id": "84", "title": "Frankenstein", "author": "Mary Shelley", "year": "1818", "genre": "Horror"},
        {"id": "174", "title": "The Picture of Dorian Gray", "author": "Oscar Wilde", "year": "1890", "genre": "Gothic"},
        {"id": "74", "title": "The Adventures of Tom Sawyer", "author": "Mark Twain", "year": "1876", "genre": "Adventure"},
        {"id": "2701", "title": "Moby Dick", "author": "Herman Melville", "year": "1851", "genre": "Adventure"},
        {"id": "345", "title": "Dracula", "author": "Bram Stoker", "year": "1897", "genre": "Horror"},
        {"id": "1661", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "year": "1892", "genre": "Mystery"},
        {"id": "76", "title": "Adventures of Huckleberry Finn", "author": "Mark Twain", "year": "1884", "genre": "Adventure"},
        {"id": "98", "title": "A Tale of Two Cities", "author": "Charles Dickens", "year": "1859", "genre": "Historical"},
        {"id": "1400", "title": "Great Expectations", "author": "Charles Dickens", "year": "1861", "genre": "Literary"},
        {"id": "844", "title": "The Importance of Being Earnest", "author": "Oscar Wilde", "year": "1895", "genre": "Comedy"},
        {"id": "46", "title": "A Christmas Carol", "author": "Charles Dickens", "year": "1843", "genre": "Christmas"},
        {"id": "1260", "title": "Jane Eyre", "author": "Charlotte Brontë", "year": "1847", "genre": "Romance"},
        {"id": "161", "title": "The Secret Garden", "author": "Frances Hodgson Burnett", "year": "1911", "genre": "Children"},
        # H.G. Wells collection
        {"id": "35", "title": "The Time Machine", "author": "H. G. Wells", "year": "1895", "genre": "Science Fiction"},
        {"id": "36", "title": "The War of the Worlds", "author": "H. G. Wells", "year": "1898", "genre": "Science Fiction"},
        {"id": "5230", "title": "The Invisible Man", "author": "H. G. Wells", "year": "1897", "genre": "Science Fiction"},
        {"id": "159", "title": "The Island of Dr. Moreau", "author": "H. G. Wells", "year": "1896", "genre": "Science Fiction"},
        {"id": "26140", "title": "The First Men in the Moon", "author": "H. G. Wells", "year": "1901", "genre": "Science Fiction"},
        {"id": "1743", "title": "When the Sleeper Wakes", "author": "H. G. Wells", "year": "1910", "genre": "Science Fiction"},
        {"id": "30155", "title": "The Food of the Gods and How It Came to Earth", "author": "H. G. Wells", "year": "1904", "genre": "Science Fiction"},
        # More classics
        {"id": "1952", "title": "The Yellow Wallpaper", "author": "Charlotte Perkins Gilman", "year": "1892", "genre": "Horror"},
        {"id": "43", "title": "The Strange Case of Dr. Jekyll and Mr. Hyde", "author": "Robert Louis Stevenson", "year": "1886", "genre": "Horror"},
        {"id": "120", "title": "Treasure Island", "author": "Robert Louis Stevenson", "year": "1883", "genre": "Adventure"},
        {"id": "219", "title": "Heart of Darkness", "author": "Joseph Conrad", "year": "1899", "genre": "Literary"},
        {"id": "1155", "title": "The Moonstone", "author": "Wilkie Collins", "year": "1868", "genre": "Mystery"},
        {"id": "41", "title": "The Legend of Sleepy Hollow", "author": "Washington Irving", "year": "1820", "genre": "Horror"},
        {"id": "140", "title": "The Jungle Book", "author": "Rudyard Kipling", "year": "1894", "genre": "Children"},
        {"id": "32", "title": "Flatland", "author": "Edwin A. Abbott", "year": "1884", "genre": "Science Fiction"},
        {"id": "205", "title": "Walden", "author": "Henry David Thoreau", "year": "1854", "genre": "Philosophy"},
        {"id": "1232", "title": "The Prince", "author": "Niccolò Machiavelli", "year": "1532", "genre": "Politics"},
        {"id": "1934", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "year": "1892", "genre": "Mystery"},
        {"id": "244", "title": "A Study in Scarlet", "author": "Arthur Conan Doyle", "year": "1887", "genre": "Mystery"},
        {"id": "2097", "title": "The Sign of the Four", "author": "Arthur Conan Doyle", "year": "1890", "genre": "Mystery"},
        {"id": "834", "title": "The Hound of the Baskervilles", "author": "Arthur Conan Doyle", "year": "1902", "genre": "Mystery"},
        # Shakespeare
        {"id": "1513", "title": "Romeo and Juliet", "author": "William Shakespeare", "year": "1597", "genre": "Tragedy"},
        {"id": "1524", "title": "Hamlet", "author": "William Shakespeare", "year": "1601", "genre": "Tragedy"},
        {"id": "1533", "title": "Macbeth", "author": "William Shakespeare", "year": "1606", "genre": "Tragedy"},
        {"id": "1540", "title": "A Midsummer Night's Dream", "author": "William Shakespeare", "year": "1595", "genre": "Comedy"},
        {"id": "1041", "title": "The Tempest", "author": "William Shakespeare", "year": "1611", "genre": "Comedy"},
        # More Dickens
        {"id": "580", "title": "Oliver Twist", "author": "Charles Dickens", "year": "1838", "genre": "Social"},
        {"id": "766", "title": "David Copperfield", "author": "Charles Dickens", "year": "1850", "genre": "Literary"},
        {"id": "786", "title": "The Old Curiosity Shop", "author": "Charles Dickens", "year": "1841", "genre": "Social"},
        # American Literature
        {"id": "25344", "title": "The Scarlet Letter", "author": "Nathaniel Hawthorne", "year": "1850", "genre": "Romance"},
        {"id": "16328", "title": "Beowulf", "author": "Anonymous", "year": "1000", "genre": "Epic"},
        {"id": "1184", "title": "The Count of Monte Cristo", "author": "Alexandre Dumas", "year": "1844", "genre": "Adventure"},
        {"id": "1257", "title": "The Three Musketeers", "author": "Alexandre Dumas", "year": "1844", "genre": "Adventure"},
        {"id": "829", "title": "Gulliver's Travels", "author": "Jonathan Swift", "year": "1726", "genre": "Satire"},
        {"id": "28054", "title": "The Brothers Karamazov", "author": "Fyodor Dostoyevsky", "year": "1880", "genre": "Philosophy"},
        {"id": "2554", "title": "Crime and Punishment", "author": "Fyodor Dostoyevsky", "year": "1866", "genre": "Psychology"},
        {"id": "2600", "title": "War and Peace", "author": "Leo Tolstoy", "year": "1869", "genre": "Historical"},
        {"id": "1399", "title": "Anna Karenina", "author": "Leo Tolstoy", "year": "1877", "genre": "Romance"},
        # Gothic and Horror
        {"id": "209", "title": "The Turn of the Screw", "author": "Henry James", "year": "1898", "genre": "Horror"},
        {"id": "31284", "title": "The Castle of Otranto", "author": "Horace Walpole", "year": "1764", "genre": "Gothic"},
        {"id": "145", "title": "Wuthering Heights", "author": "Emily Brontë", "year": "1847", "genre": "Gothic"},
        # Poetry
        {"id": "1065", "title": "The Raven", "author": "Edgar Allan Poe", "year": "1845", "genre": "Poetry"},
        {"id": "2148", "title": "Leaves of Grass", "author": "Walt Whitman", "year": "1855", "genre": "Poetry"},
        # Adventure
        {"id": "1268", "title": "Around the World in Eighty Days", "author": "Jules Verne", "year": "1873", "genre": "Adventure"},
        {"id": "164", "title": "Twenty Thousand Leagues Under the Sea", "author": "Jules Verne", "year": "1870", "genre": "Adventure"},
        {"id": "103", "title": "Around the World in Eighty Days", "author": "Jules Verne", "year": "1873", "genre": "Adventure"},
        # More Popular Books - Expanded catalog
        {"id": "1080", "title": "A Modest Proposal", "author": "Jonathan Swift", "year": "1729", "genre": "Satire"},
        {"id": "5200", "title": "Metamorphosis", "author": "Franz Kafka", "year": "1915", "genre": "Fiction"},
        {"id": "1998", "title": "Thus Spoke Zarathustra", "author": "Friedrich Nietzsche", "year": "1883", "genre": "Philosophy"},
        {"id": "135", "title": "Les Misérables", "author": "Victor Hugo", "year": "1862", "genre": "Historical"},
        {"id": "25344", "title": "The Scarlet Letter", "author": "Nathaniel Hawthorne", "year": "1850", "genre": "Romance"},
        {"id": "205", "title": "Walden", "author": "Henry David Thoreau", "year": "1854", "genre": "Philosophy"},
        {"id": "1184", "title": "The Count of Monte Cristo", "author": "Alexandre Dumas", "year": "1844", "genre": "Adventure"},
        {"id": "120", "title": "Treasure Island", "author": "Robert Louis Stevenson", "year": "1883", "genre": "Adventure"},
        {"id": "2000", "title": "Don Quixote", "author": "Miguel de Cervantes", "year": "1605", "genre": "Adventure"},
        {"id": "4300", "title": "Ulysses", "author": "James Joyce", "year": "1922", "genre": "Modernist"},
        {"id": "28054", "title": "The Brothers Karamazov", "author": "Fyodor Dostoyevsky", "year": "1880", "genre": "Philosophy"},
        {"id": "1952", "title": "The Yellow Wallpaper", "author": "Charlotte Perkins Gilman", "year": "1892", "genre": "Gothic"},
        {"id": "32", "title": "Narrative of the Life of Frederick Douglass, an American Slave", "author": "Frederick Douglass", "year": "1845", "genre": "Autobiography"},
        {"id": "158", "title": "Emma", "author": "Jane Austen", "year": "1815", "genre": "Romance"},
        {"id": "141", "title": "Mansfield Park", "author": "Jane Austen", "year": "1814", "genre": "Romance"},
        {"id": "105", "title": "Persuasion", "author": "Jane Austen", "year": "1817", "genre": "Romance"},
        {"id": "1237", "title": "The Republic", "author": "Plato", "year": "-380", "genre": "Philosophy"},
        {"id": "2542", "title": "A Doll's House", "author": "Henrik Ibsen", "year": "1879", "genre": "Drama"},
        {"id": "37106", "title": "Little Women", "author": "Louisa May Alcott", "year": "1868", "genre": "Children"},
        {"id": "74", "title": "Adventures of Tom Sawyer", "author": "Mark Twain", "year": "1876", "genre": "Adventure"},
        {"id": "245", "title": "Anne of Green Gables", "author": "L. M. Montgomery", "year": "1908", "genre": "Children"},
        {"id": "28885", "title": "A Study in Scarlet", "author": "Arthur Conan Doyle", "year": "1887", "genre": "Mystery"},
        {"id": "2097", "title": "The Sign of the Four", "author": "Arthur Conan Doyle", "year": "1890", "genre": "Mystery"},
        {"id": "1244", "title": "The Memoirs of Sherlock Holmes", "author": "Arthur Conan Doyle", "year": "1894", "genre": "Mystery"}
    ]
    
    # Apply author filter if specified
    if author_filter:
        catalog = [book for book in catalog if author_filter.lower() in book["author"].lower()]
    
    # Filter results based on query
    query_lower = query.lower()
    results = []
    
    for book in catalog:
        if (query_lower in book["title"].lower() or 
            query_lower in book["author"].lower() or 
            any(word in book["title"].lower() for word in query_lower.split()) or
            any(word in book["author"].lower() for word in query_lower.split()) or
            query_lower in book.get("genre", "").lower()):
            results.append(book)
    
    # If no matches, return popular suggestions
    if not results:
        results = catalog[:20]
    
    # Apply pagination
    total_results = len(results)
    paginated_results = results[offset:offset + limit]
    
    return {
        "query": query,
        "results": paginated_results,
        "total": total_results,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total_results
    }

@app.post("/gutenberg/ingest")
def ingest_gutenberg_book(req: BookIngestionReq):
    """Download and chunk-process a Project Gutenberg book"""
    logger.info(f"Ingesting Gutenberg book {req.gutenberg_id} with chunk_size={req.chunk_size}, reading_alpha={req.reading_alpha}")
    try:
        # Download the book
        url = f"https://www.gutenberg.org/files/{req.gutenberg_id}/{req.gutenberg_id}-0.txt"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Book {req.gutenberg_id} not found")
        
        text = response.text
        
        # Store original text for title/author extraction before cleaning
        original_text = text
        
        # Clean the text (remove Project Gutenberg headers/footers)
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK", 
            "*END*THE SMALL PRINT!"
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "End of the Project Gutenberg"
        ]
        
        for marker in start_markers:
            if marker in text:
                text = text.split(marker, 1)[1]
                break
        
        for marker in end_markers:
            if marker in text:
                text = text.split(marker)[0]
                break
        
        # Extract title and author from the cleaned text - look for patterns in early lines
        lines = text.strip().split('\n')[:20]  # First 20 lines after header
        
        title = f"Book {req.gutenberg_id}"  # fallback
        author = "Unknown"  # fallback
        
        # Look for title patterns in the early lines
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('[') or len(line) < 5:
                continue
                
            # Common title patterns
            if i < 10 and len(line) > 10 and len(line) < 80:
                # Check if it looks like a title (proper case, not all caps, reasonable length)
                if (line[0].isupper() and 
                    not line.isupper() and 
                    not any(word in line.lower() for word in ['chapter', 'contents', 'edition', 'illustration']) and
                    any(c.isalpha() for c in line)):
                    title = line
                    break
        
        # Look for "by [Author]" pattern
        for line in lines:
            line = line.strip()
            if line.lower().startswith('by ') and len(line) > 3 and len(line) < 50:
                author = line[3:].strip()
                break
        
        # Fallback patterns for common Gutenberg formats
        if title == f"Book {req.gutenberg_id}":
            # Try alternative extraction from original text before cleaning
            original_lines = original_text.split('\n')[:30]
            for line in original_lines:
                line = line.strip()
                if ('adventures' in line.lower() or 'tale' in line.lower() or 
                    'story' in line.lower() or 'life' in line.lower()) and len(line) > 8:
                    title = line
                    break
        
        # Create chunks
        chunks = []
        for i in range(0, len(text), req.chunk_size):
            chunk = text[i:i + req.chunk_size].strip()
            if chunk:  # Skip empty chunks
                chunks.append({
                    "index": len(chunks),
                    "text": chunk,
                    "start_pos": i,
                    "end_pos": min(i + req.chunk_size, len(text))
                })
        
        # Create a book rho matrix
        book_rho_id = f"book_{req.gutenberg_id}_{int(time.time())}"
        rho = np.eye(DIM, dtype=complex) / DIM  # Start with maximum entropy
        
        # Initialize hierarchical matrices
        detail_matrix = np.eye(DIM, dtype=complex) / DIM
        narrative_matrix = np.eye(DIM, dtype=complex) / DIM  
        thematic_matrix = np.eye(DIM, dtype=complex) / DIM
        
        STATE[book_rho_id] = {
            "rho": rho,  # Main composite matrix
            "detail_matrix": detail_matrix,
            "narrative_matrix": narrative_matrix,
            "thematic_matrix": thematic_matrix,
            "log": [],
            "meta": {
                "type": "book",
                "gutenberg_id": req.gutenberg_id,
                "title": title,
                "author": author,
                "total_chunks": len(chunks),
                "chunk_size": req.chunk_size,
                "reading_alpha": req.reading_alpha,
                "chunks_processed": 0,
                "created": time.time(),
                "chunks": chunks,  # Store actual chunks
                "reading_reflections": [],  # LLM commentary on each chunk
                "narrative_summaries": [],  # Section summaries
                "thematic_insights": [],   # Chapter-level insights
                "final_reflections": []    # Post-reading top-k insights
            }
        }
        
        return {
            "book_rho_id": book_rho_id,
            "title": title,
            "author": author,
            "total_chunks": len(chunks),
            "text_length": len(text),
            "chunks": chunks[:5],  # Return first 5 chunks as preview
            "ready_for_reading": True
        }
        
    except requests.RequestException as e:
        logger.error(f"Failed to download Gutenberg book {req.gutenberg_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download book: {str(e)}")
    except Exception as e:
        logger.error(f"Book processing failed for Gutenberg book {req.gutenberg_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Book processing failed: {str(e)}")

def create_text_projection_matrix(text: str) -> np.ndarray:
    """Create a projection matrix from text using existing logic"""
    # Use the same projection logic as the original rho reading
    text_bytes = text.encode('utf-8')
    hash_val = hash(text) % 1000000
    np.random.seed(abs(hash_val))
    
    # Create a projection matrix based on text characteristics
    proj = np.random.random((DIM, DIM)) + 1j * np.random.random((DIM, DIM))
    proj = (proj + proj.conj().T) / 2  # Make Hermitian
    proj = psd_project(proj)  # Ensure positive semidefinite
    
    return proj

def create_narrative_summary(chunks: list, book_meta: dict) -> str:
    """Create a narrative summary from a group of chunks"""
    combined_text = " ".join([chunk["text"] for chunk in chunks])
    
    # Simple extractive summarization based on key narrative elements
    sentences = [s.strip() for s in combined_text.split('.') if len(s.strip()) > 20]
    
    # Look for key narrative indicators
    key_sentences = []
    for sentence in sentences:
        score = 0
        # Dialog indicators
        if any(word in sentence.lower() for word in ['said', 'asked', 'replied', '"']):
            score += 2
        # Action indicators  
        if any(word in sentence.lower() for word in ['went', 'came', 'ran', 'walked', 'opened']):
            score += 2
        # Emotional indicators
        if any(word in sentence.lower() for word in ['felt', 'thought', 'wondered', 'surprised']):
            score += 1
        # Character names (simple detection)
        if any(word[0].isupper() for word in sentence.split() if len(word) > 2):
            score += 1
            
        if score >= 2:
            key_sentences.append(sentence)
    
    # Take top sentences or fallback to beginning
    summary_sentences = key_sentences[:3] if key_sentences else sentences[:2]
    summary = ". ".join(summary_sentences) + "."
    
    # Add context
    title = book_meta.get("title", "this work")
    return f"In {title}: {summary}"

def extract_thematic_insight(narrative_summaries: list, book_meta: dict) -> str:
    """Extract thematic insights from narrative summaries"""
    if not narrative_summaries:
        return f"Thematic development begins in {book_meta.get('title', 'this work')}."
    
    # Analyze themes across summaries
    all_summaries = " ".join([ns["summary"] for ns in narrative_summaries])
    
    # Theme detection patterns
    themes = []
    if any(word in all_summaries.lower() for word in ['journey', 'path', 'way', 'travel']):
        themes.append("journey/transformation")
    if any(word in all_summaries.lower() for word in ['love', 'heart', 'emotion', 'feeling']):
        themes.append("emotional development") 
    if any(word in all_summaries.lower() for word in ['power', 'control', 'authority', 'rule']):
        themes.append("power dynamics")
    if any(word in all_summaries.lower() for word in ['truth', 'real', 'honest', 'lie']):
        themes.append("truth vs deception")
    if any(word in all_summaries.lower() for word in ['grow', 'change', 'become', 'transform']):
        themes.append("personal growth")
    
    # Character development
    if any(word in all_summaries.lower() for word in ['character', 'person', 'man', 'woman', 'child']):
        themes.append("character development")
    
    primary_theme = themes[0] if themes else "narrative progression"
    title = book_meta.get("title", "this work")
    author = book_meta.get("author", "the author")
    
    return f"{author}'s exploration of {primary_theme} deepens in {title}, as characters navigate complex situations that reveal underlying patterns of meaning."

def generate_llm_commentary(chunk_text: str, chunk_index: int, book_context: dict) -> str:
    """Generate LLM commentary/reflection on a chunk"""
    # Simplified commentary generation - in practice, use a real LLM
    title = book_context.get("title", "")
    author = book_context.get("author", "")
    
    # Analyze key elements in the chunk
    commentary_elements = []
    
    if any(word in chunk_text.lower() for word in ['said', 'asked', 'replied', '"', "'"]):
        commentary_elements.append("dialogue_present")
    if any(word in chunk_text.lower() for word in ['suddenly', 'then', 'next', 'after']):
        commentary_elements.append("narrative_progression")
    if any(word in chunk_text.lower() for word in ['felt', 'thought', 'wondered', 'seemed']):
        commentary_elements.append("introspection")
    if any(word in chunk_text.lower() for word in ['beautiful', 'dark', 'bright', 'strange']):
        commentary_elements.append("atmospheric")
    
    # Generate contextual commentary
    commentary_templates = {
        "dialogue_present": f"Character voices emerge distinctly here, revealing personality through speech patterns.",
        "narrative_progression": f"The story momentum shifts - this feels like a pivotal moment in {title}.",
        "introspection": f"Deep psychological insight - {author}'s exploration of inner experience.",
        "atmospheric": f"Rich sensory detail that establishes mood and setting effectively."
    }
    
    if commentary_elements:
        primary_element = commentary_elements[0]
        commentary = commentary_templates.get(primary_element, "Significant narrative development.")
    else:
        commentary = f"Foundational text that builds the world of {title}."
    
    # Add reading position context
    if chunk_index < 10:
        commentary += " [Early establishment phase]"
    elif chunk_index % 20 == 0:  # Every 20th chunk
        commentary += " [Potential turning point]"
    
    return commentary

@app.post("/gutenberg/{book_rho_id}/read_chunk/{chunk_index}")
def read_chunk(book_rho_id: str, chunk_index: int):
    """Read a specific chunk with hierarchical processing and LLM commentary"""
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book matrix not found")
    
    book_data = STATE[book_rho_id]
    book_meta = book_data["meta"]
    
    if book_meta.get("type") != "book":
        raise HTTPException(status_code=400, detail="Not a book matrix")
    
    if chunk_index >= len(book_meta["chunks"]):
        raise HTTPException(status_code=400, detail="Chunk index out of range")
    
    # Get the actual chunk text
    chunk = book_meta["chunks"][chunk_index]
    chunk_text = chunk["text"]
    reading_alpha = book_meta["reading_alpha"]
    
    # Generate LLM commentary for this chunk
    commentary = generate_llm_commentary(chunk_text, chunk_index, book_meta)
    
    # Store the reflection
    reflection = {
        "chunk_index": chunk_index,
        "commentary": commentary,
        "chunk_preview": chunk_text[:100],
        "timestamp": time.time(),
        "significance_score": len([w for w in ["pivotal", "turning", "significant", "distinct"] if w in commentary.lower()])
    }
    book_data["meta"]["reading_reflections"].append(reflection)
    
    # Process chunk into hierarchical matrices
    
    # 1. Detail Matrix - Raw chunk processing
    detail_proj = create_text_projection_matrix(chunk_text)
    current_detail = book_data["detail_matrix"]
    new_detail = (1 - reading_alpha) * current_detail + reading_alpha * detail_proj
    book_data["detail_matrix"] = psd_project(new_detail)
    
    # 2. Narrative Matrix - Every 5 chunks, create narrative summary
    if (chunk_index + 1) % 5 == 0:
        recent_chunks = book_meta["chunks"][max(0, chunk_index-4):chunk_index+1]
        narrative_summary = create_narrative_summary(recent_chunks, book_meta)
        book_data["meta"]["narrative_summaries"].append({
            "chunks_range": [max(0, chunk_index-4), chunk_index+1],
            "summary": narrative_summary,
            "timestamp": time.time()
        })
        
        # Update narrative matrix
        narrative_proj = create_text_projection_matrix(narrative_summary)
        current_narrative = book_data["narrative_matrix"] 
        new_narrative = (1 - reading_alpha * 0.7) * current_narrative + (reading_alpha * 0.7) * narrative_proj
        book_data["narrative_matrix"] = psd_project(new_narrative)
    
    # 3. Thematic Matrix - Every 20 chunks, extract thematic insights
    if (chunk_index + 1) % 20 == 0:
        recent_summaries = book_data["meta"]["narrative_summaries"][-4:] if len(book_data["meta"]["narrative_summaries"]) >= 4 else book_data["meta"]["narrative_summaries"]
        thematic_insight = extract_thematic_insight(recent_summaries, book_meta)
        book_data["meta"]["thematic_insights"].append({
            "chunk_milestone": chunk_index + 1,
            "insight": thematic_insight,
            "timestamp": time.time()
        })
        
        # Update thematic matrix
        theme_proj = create_text_projection_matrix(thematic_insight)
        current_thematic = book_data["thematic_matrix"]
        new_thematic = (1 - reading_alpha * 0.5) * current_thematic + (reading_alpha * 0.5) * theme_proj
        book_data["thematic_matrix"] = psd_project(new_thematic)
    
    # 4. Update composite matrix (blend of all three)
    composite_alpha = reading_alpha * 0.6
    detail_weight = 0.5
    narrative_weight = 0.3  
    thematic_weight = 0.2
    
    weighted_matrix = (detail_weight * book_data["detail_matrix"] + 
                      narrative_weight * book_data["narrative_matrix"] +
                      thematic_weight * book_data["thematic_matrix"])
    
    current_composite = book_data["rho"]
    new_composite = (1 - composite_alpha) * current_composite + composite_alpha * weighted_matrix
    book_data["rho"] = psd_project(new_composite)
    
    # Update processing count
    book_data["meta"]["chunks_processed"] += 1
    
    # Log the reading action
    book_data["log"].append({
        "action": "read_chunk_hierarchical",
        "chunk_index": chunk_index,
        "text_preview": chunk_text[:100],
        "commentary": commentary,
        "matrices_updated": ["detail"] + 
                          (["narrative"] if (chunk_index + 1) % 5 == 0 else []) +
                          (["thematic"] if (chunk_index + 1) % 20 == 0 else []),
        "timestamp": time.time()
    })
    
    # Check if book reading is complete and auto-merge into global consciousness
    if book_data["meta"]["chunks_processed"] >= book_data["meta"]["total_chunks"]:
        logger.info(f"📚 Book '{book_meta.get('title', book_rho_id)}' reading completed! Auto-merging into global consciousness...")
        try:
            merge_book_into_global_rho(book_rho_id)
            logger.info(f"✅ Successfully merged '{book_meta.get('title', book_rho_id)}' into global consciousness")
        except Exception as e:
            logger.error(f"❌ Failed to auto-merge '{book_meta.get('title', book_rho_id)}': {e}")
    
    return {
        "book_rho_id": book_rho_id,
        "chunk_index": chunk_index,
        "chunk_preview": chunk_text[:200],
        "llm_commentary": commentary,
        "chunks_processed": book_data["meta"]["chunks_processed"],
        "total_chunks": book_data["meta"]["total_chunks"],
        "progress": book_data["meta"]["chunks_processed"] / book_data["meta"]["total_chunks"],
        "matrix_state": diagnostics(new_composite),
        "hierarchical_status": {
            "detail_updates": chunk_index + 1,
            "narrative_updates": len(book_data["meta"]["narrative_summaries"]),
            "thematic_updates": len(book_data["meta"]["thematic_insights"]),
            "total_reflections": len(book_data["meta"]["reading_reflections"])
        }
    }

@app.get("/gutenberg/{book_rho_id}/progress")
def get_book_progress(book_rho_id: str):
    """Get reading progress for a book"""
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book not found")
    
    meta = STATE[book_rho_id]["meta"]
    rho = STATE[book_rho_id]["rho"]
    
    return {
        "book_rho_id": book_rho_id,
        "title": meta.get("title", "Unknown"),
        "author": meta.get("author", "Unknown"),
        "chunks_processed": meta.get("chunks_processed", 0),
        "total_chunks": meta.get("total_chunks", 0),
        "progress": meta.get("chunks_processed", 0) / max(meta.get("total_chunks", 1), 1),
        "matrix_state": diagnostics(rho),
        "reading_log": STATE[book_rho_id]["log"][-10:]  # Last 10 entries
    }

@app.post("/gutenberg/{book_rho_id}/finalize_reading")
def finalize_reading(book_rho_id: str, top_k: int = 5):
    """Complete book reading with post-reading reflection and top-k insight selection"""
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book_data = STATE[book_rho_id]
    meta = book_data["meta"]
    
    if meta.get("type") != "book":
        raise HTTPException(status_code=400, detail="Not a book matrix")
    
    # Check if book is fully read
    if meta["chunks_processed"] < meta["total_chunks"]:
        return {
            "status": "incomplete",
            "message": f"Book not finished: {meta['chunks_processed']}/{meta['total_chunks']} chunks read"
        }
    
    # Generate post-reading reflection on all collected insights
    all_reflections = meta["reading_reflections"]
    narrative_summaries = meta["narrative_summaries"] 
    thematic_insights = meta["thematic_insights"]
    
    # Score all reflections for significance
    scored_reflections = []
    for reflection in all_reflections:
        score = reflection["significance_score"]
        
        # Boost score based on position (beginning/end more important)
        chunk_idx = reflection["chunk_index"]
        total_chunks = meta["total_chunks"]
        if chunk_idx < total_chunks * 0.1:  # First 10%
            score += 2
        elif chunk_idx > total_chunks * 0.9:  # Last 10%
            score += 2
        elif chunk_idx > total_chunks * 0.4 and chunk_idx < total_chunks * 0.6:  # Middle
            score += 1
            
        # Boost for thematic keywords
        commentary = reflection["commentary"].lower()
        if any(word in commentary for word in ["pivotal", "turning", "significant", "transformation", "revelation"]):
            score += 3
        if any(word in commentary for word in ["character", "development", "change", "growth"]):
            score += 2
        if any(word in commentary for word in ["theme", "symbolic", "meaning", "deeper"]):
            score += 2
            
        scored_reflections.append({
            **reflection,
            "final_significance_score": score
        })
    
    # Select top-k most significant insights
    top_insights = sorted(scored_reflections, key=lambda x: x["final_significance_score"], reverse=True)[:top_k]
    
    # Generate final post-reading reflection
    title = meta.get("title", "this work")
    author = meta.get("author", "the author")
    
    final_reflection = f"""Upon completing {title} by {author}, the reading consciousness identifies {len(top_insights)} pivotal moments that shaped understanding:

{chr(10).join([f"• Chunk {insight['chunk_index']}: {insight['commentary']}" for insight in top_insights])}

These insights reveal the wordless architecture of meaning that emerged through {meta['chunks_processed']} textual encounters, crystallizing into lasting comprehension that transcends the original words."""
    
    # Store final reflections
    meta["final_reflections"] = top_insights
    meta["post_reading_reflection"] = final_reflection
    meta["reading_completed"] = time.time()
    
    # Create final integration matrix incorporating top insights
    insights_text = " ".join([insight["commentary"] for insight in top_insights])
    final_insights_proj = create_text_projection_matrix(final_reflection + " " + insights_text)
    
    # Blend final insights into the composite matrix
    integration_alpha = 0.3  # Strong influence from final reflection
    current_composite = book_data["rho"]
    final_composite = (1 - integration_alpha) * current_composite + integration_alpha * final_insights_proj
    book_data["rho"] = psd_project(final_composite)
    
    # Log completion
    book_data["log"].append({
        "action": "finalize_reading",
        "top_k_insights": len(top_insights),
        "final_reflection_length": len(final_reflection),
        "reading_duration": time.time() - meta["created"],
        "timestamp": time.time()
    })
    
    return {
        "book_rho_id": book_rho_id,
        "status": "completed",
        "title": title,
        "author": author,
        "reading_stats": {
            "total_chunks": meta["total_chunks"],
            "total_reflections": len(all_reflections),
            "narrative_summaries": len(narrative_summaries),
            "thematic_insights": len(thematic_insights),
            "top_insights_selected": len(top_insights),
            "reading_duration": time.time() - meta["created"]
        },
        "top_insights": top_insights,
        "final_reflection": final_reflection,
        "final_matrix_state": diagnostics(book_data["rho"]),
        "hierarchical_matrices": {
            "detail_matrix": diagnostics(book_data["detail_matrix"]),
            "narrative_matrix": diagnostics(book_data["narrative_matrix"]),
            "thematic_matrix": diagnostics(book_data["thematic_matrix"])
        }
    }

@app.get("/gutenberg/{book_rho_id}/insights")
def get_reading_insights(book_rho_id: str):
    """Get all reading insights and reflections for a book"""
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book_data = STATE[book_rho_id]
    meta = book_data["meta"]
    
    return {
        "book_rho_id": book_rho_id,
        "title": meta.get("title", "Unknown"),
        "author": meta.get("author", "Unknown"),
        "reading_progress": meta["chunks_processed"] / meta["total_chunks"],
        "is_completed": "final_reflections" in meta,
        "insights": {
            "chunk_reflections": meta["reading_reflections"],
            "narrative_summaries": meta["narrative_summaries"],
            "thematic_insights": meta["thematic_insights"],
            "final_reflections": meta.get("final_reflections", []),
            "post_reading_reflection": meta.get("post_reading_reflection", "")
        },
        "matrix_evolution": {
            "current_composite": diagnostics(book_data["rho"]),
            "detail_focus": diagnostics(book_data["detail_matrix"]),
            "narrative_flow": diagnostics(book_data["narrative_matrix"]),
            "thematic_depth": diagnostics(book_data["thematic_matrix"])
        }
    }

# --- Batch Processing Queue System ---

def background_book_processor():
    """Background thread that processes queued jobs"""
    global PROCESSING_QUEUE, COMPLETED_JOBS
    
    while True:
        try:
            job_to_process = None
            
            # Find next job to process
            with QUEUE_LOCK:
                for job in PROCESSING_QUEUE:
                    if job.status == JobStatus.QUEUED:
                        job.status = JobStatus.PROCESSING
                        job.started_at = datetime.now()
                        job_to_process = job
                        break
            
            if job_to_process is None:
                time.sleep(5)  # No jobs, wait 5 seconds
                continue
            
            logger.info(f"Processing job {job_to_process.job_id}: {job_to_process.book_title}")
            
            # Process the book
            try:
                # 1. Ingest the book
                ingest_req = BookIngestionReq(
                    gutenberg_id=job_to_process.gutenberg_id,
                    chunk_size=job_to_process.chunk_size,
                    reading_alpha=job_to_process.reading_alpha
                )
                
                # Simulate the ingestion call
                url = f"https://www.gutenberg.org/files/{job_to_process.gutenberg_id}/{job_to_process.gutenberg_id}-0.txt"
                response = requests.get(url, timeout=30)
                
                if response.status_code != 200:
                    raise Exception(f"Failed to download book {job_to_process.gutenberg_id}")
                
                # Process text and create chunks (simplified version of ingest logic)
                text = response.text
                
                # Clean text
                for marker in ["*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
                    if marker in text:
                        text = text.split(marker, 1)[1]
                        break
                        
                for marker in ["*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
                    if marker in text:
                        text = text.split(marker)[0]
                        break
                
                # Create chunks
                chunks = []
                for i in range(0, len(text), job_to_process.chunk_size):
                    chunk = text[i:i + job_to_process.chunk_size].strip()
                    if chunk:
                        chunks.append({
                            "index": len(chunks),
                            "text": chunk,
                            "start_pos": i,
                            "end_pos": min(i + job_to_process.chunk_size, len(text))
                        })
                
                # Create matrices and state
                book_rho_id = f"batch_book_{job_to_process.gutenberg_id}_{int(time.time())}"
                rho = np.eye(DIM, dtype=complex) / DIM
                detail_matrix = np.eye(DIM, dtype=complex) / DIM
                narrative_matrix = np.eye(DIM, dtype=complex) / DIM  
                thematic_matrix = np.eye(DIM, dtype=complex) / DIM
                
                STATE[book_rho_id] = {
                    "rho": rho,
                    "detail_matrix": detail_matrix,
                    "narrative_matrix": narrative_matrix,
                    "thematic_matrix": thematic_matrix,
                    "log": [],
                    "meta": {
                        "type": "batch_book",
                        "job_id": job_to_process.job_id,
                        "gutenberg_id": job_to_process.gutenberg_id,
                        "title": job_to_process.book_title,
                        "author": job_to_process.book_author,
                        "total_chunks": len(chunks),
                        "chunk_size": job_to_process.chunk_size,
                        "reading_alpha": job_to_process.reading_alpha,
                        "chunks_processed": 0,
                        "created": time.time(),
                        "chunks": chunks,
                        "reading_reflections": [],
                        "narrative_summaries": [],
                        "thematic_insights": [],
                        "final_reflections": []
                    }
                }
                
                job_to_process.book_rho_id = book_rho_id
                job_to_process.total_chunks = len(chunks)
                
                # 2. Process all chunks
                for chunk_idx in range(len(chunks)):
                    if job_to_process.status == JobStatus.CANCELLED:
                        break
                        
                    # Process this chunk (simplified version of read_chunk logic)
                    chunk = chunks[chunk_idx]
                    chunk_text = chunk["text"]
                    
                    # Generate commentary
                    commentary = generate_llm_commentary(chunk_text, chunk_idx, STATE[book_rho_id]["meta"])
                    
                    # Update matrices (simplified)
                    detail_proj = create_text_projection_matrix(chunk_text)
                    current_detail = STATE[book_rho_id]["detail_matrix"]
                    new_detail = (1 - job_to_process.reading_alpha) * current_detail + job_to_process.reading_alpha * detail_proj
                    STATE[book_rho_id]["detail_matrix"] = psd_project(new_detail)
                    
                    # Update composite
                    STATE[book_rho_id]["rho"] = STATE[book_rho_id]["detail_matrix"]
                    
                    # Update progress
                    job_to_process.chunks_processed = chunk_idx + 1
                    job_to_process.progress = (chunk_idx + 1) / len(chunks)
                    STATE[book_rho_id]["meta"]["chunks_processed"] = chunk_idx + 1
                    
                    # Add reflection
                    STATE[book_rho_id]["meta"]["reading_reflections"].append({
                        "chunk_index": chunk_idx,
                        "commentary": commentary,
                        "chunk_preview": chunk_text[:100],
                        "timestamp": time.time(),
                        "significance_score": len([w for w in ["pivotal", "turning", "significant", "distinct"] if w in commentary.lower()])
                    })
                    
                    # Brief pause to prevent overwhelming
                    time.sleep(0.1)
                
                # 3. Finalize reading if requested
                if job_to_process.status != JobStatus.CANCELLED:
                    # Simplified finalization
                    all_reflections = STATE[book_rho_id]["meta"]["reading_reflections"]
                    top_insights = sorted(all_reflections, 
                                        key=lambda x: x["significance_score"], 
                                        reverse=True)[:5]
                    
                    STATE[book_rho_id]["meta"]["final_reflections"] = top_insights
                    STATE[book_rho_id]["meta"]["reading_completed"] = time.time()
                    
                    # Mark job as completed
                    job_to_process.status = JobStatus.COMPLETED
                    job_to_process.completed_at = datetime.now()
                    
                    # Auto-merge into global consciousness
                    try:
                        merge_book_into_global_rho(book_rho_id)
                    except Exception as merge_error:
                        logger.error(f"Failed to merge {book_rho_id} into global consciousness: {merge_error}")
                    
                    logger.info(f"Completed job {job_to_process.job_id}: {job_to_process.book_title}")
                
            except Exception as e:
                logger.error(f"Job {job_to_process.job_id} failed: {str(e)}")
                job_to_process.status = JobStatus.FAILED
                job_to_process.error_message = str(e)
                job_to_process.completed_at = datetime.now()
                
        except Exception as e:
            logger.error(f"Background processor error: {str(e)}")
            time.sleep(10)

def start_background_processor():
    """Start the background processing thread"""
    global BACKGROUND_PROCESSOR
    if BACKGROUND_PROCESSOR is None or not BACKGROUND_PROCESSOR.is_alive():
        BACKGROUND_PROCESSOR = threading.Thread(target=background_book_processor, daemon=True)
        BACKGROUND_PROCESSOR.start()
        logger.info("Background book processor started")

@app.post("/queue/batch_submit")
def submit_batch_job(request: BatchJobRequest):
    """Submit a batch job with search queries and instructions"""
    
    # Check queue capacity (limit to 50 total jobs)
    current_total = len(PROCESSING_QUEUE) + len(COMPLETED_JOBS)
    if current_total >= 50:
        return {
            "error": "Queue at capacity",
            "message": f"Currently processing {current_total} jobs. Please wait for some to complete before adding more.",
            "queue_status": {
                "queued": len([j for j in PROCESSING_QUEUE if j.status == JobStatus.QUEUED]),
                "processing": len([j for j in PROCESSING_QUEUE if j.status == JobStatus.PROCESSING]),
                "completed": len(COMPLETED_JOBS)
            }
        }
    
    # Use intelligent agent to analyze and schedule jobs
    jobs = intelligent_scheduler_agent(request)
    
    # Add jobs to global queue
    with QUEUE_LOCK:
        PROCESSING_QUEUE.extend(jobs)
    
    # Start background processor if not running
    start_background_processor()
    
    return {
        "batch_id": f"batch_{int(time.time())}",
        "jobs_queued": len(jobs),
        "jobs": [
            {
                "job_id": job.job_id,
                "book_title": job.book_title,
                "book_author": job.book_author,
                "priority": job.priority,
                "agent_notes": job.agent_notes
            } for job in jobs
        ],
        "estimated_processing_time_minutes": len(jobs) * 3,  # Rough estimate
        "agent_analysis": f"Intelligent agent processed {len(request.search_queries)} queries and scheduled {len(jobs)} books for reading with priority-based ordering."
    }

@app.get("/queue/status")
def get_queue_status():
    """Get current queue status"""
    with QUEUE_LOCK:
        queued_jobs = [job for job in PROCESSING_QUEUE if job.status == JobStatus.QUEUED]
        processing_jobs = [job for job in PROCESSING_QUEUE if job.status == JobStatus.PROCESSING]
        completed_jobs = [job for job in PROCESSING_QUEUE if job.status == JobStatus.COMPLETED]
        failed_jobs = [job for job in PROCESSING_QUEUE if job.status == JobStatus.FAILED]
    
    return {
        "queue_summary": {
            "queued": len(queued_jobs),
            "processing": len(processing_jobs),
            "completed": len(completed_jobs),
            "failed": len(failed_jobs),
            "total": len(PROCESSING_QUEUE)
        },
        "current_jobs": {
            "queued": [
                {
                    "job_id": job.job_id,
                    "book_title": job.book_title,
                    "priority": job.priority,
                    "created_at": job.created_at.isoformat(),
                    "agent_notes": job.agent_notes
                } for job in queued_jobs[:10]  # Show first 10
            ],
            "processing": [
                {
                    "job_id": job.job_id,
                    "book_title": job.book_title,
                    "progress": job.progress,
                    "chunks_processed": job.chunks_processed,
                    "total_chunks": job.total_chunks,
                    "started_at": job.started_at.isoformat() if job.started_at else None
                } for job in processing_jobs
            ]
        },
        "processor_status": {
            "background_processor_running": BACKGROUND_PROCESSOR is not None and BACKGROUND_PROCESSOR.is_alive()
        }
    }

@app.get("/queue/job/{job_id}")
def get_job_status(job_id: str):
    """Get detailed status of a specific job"""
    with QUEUE_LOCK:
        job = next((j for j in PROCESSING_QUEUE if j.job_id == job_id), None)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = {
        "job_id": job.job_id,
        "book_title": job.book_title,
        "book_author": job.book_author,
        "status": job.status,
        "priority": job.priority,
        "progress": job.progress,
        "chunks_processed": job.chunks_processed,
        "total_chunks": job.total_chunks,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "agent_notes": job.agent_notes,
        "error_message": job.error_message
    }
    
    # Add book data if available
    if job.book_rho_id and job.book_rho_id in STATE:
        book_data = STATE[job.book_rho_id]
        result["matrix_state"] = diagnostics(book_data["rho"])
        result["reading_insights"] = {
            "total_reflections": len(book_data["meta"]["reading_reflections"]),
            "latest_reflections": book_data["meta"]["reading_reflections"][-3:] if book_data["meta"]["reading_reflections"] else [],
            "final_reflections": book_data["meta"].get("final_reflections", [])
        }
    
    return result

@app.post("/queue/cancel/{job_id}")
def cancel_job(job_id: str):
    """Cancel a queued or processing job"""
    with QUEUE_LOCK:
        job = next((j for j in PROCESSING_QUEUE if j.job_id == job_id), None)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now()
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": f"Job {job_id} ({job.book_title}) has been cancelled"
    }

# Initialize background processor on startup
start_background_processor()

# Initialize auto-persistence
auto_load_state()  # Load existing data on startup
periodic_auto_save()  # Start periodic saving
atexit.register(auto_save_state)  # Save on shutdown

# --- Database Management Endpoints ---

@app.get("/database/state")
def database_state():
    """Get comprehensive database state information"""
    rho_info = []
    total_narratives = 0
    
    for rho_id, item in STATE.items():
        log_entries = item.get("log", [])
        narrative_count = len([entry for entry in log_entries if entry.get("op") == "read" or entry.get("action") == "read_chunk_hierarchical"])
        
        # For composite matrices (like global_consciousness) with no direct narratives,
        # aggregate count from constituent book matrices
        if narrative_count == 0 and (rho_id == COMPOSITE_MATRIX_ID or rho_id == "global_consciousness"):
            meta = item.get("meta", {})
            book_titles = meta.get("book_titles", [])
            for book_title in book_titles:
                book_rho_id = book_title.replace("Book_", "")
                if book_rho_id in STATE:
                    book_item = STATE[book_rho_id]
                    book_log_entries = book_item.get("log", [])
                    narrative_count += len([entry for entry in book_log_entries if entry.get("op") == "read" or entry.get("action") == "read_chunk_hierarchical"])
        total_narratives += narrative_count
        
        rho_info.append({
            "rho_id": rho_id,
            "narrative_count": narrative_count,
            "operations": len(log_entries),
            "last_operation": log_entries[-1] if log_entries else None,
            "diagnostics": diagnostics(item["rho"]),
            "meta": item.get("meta", {})
        })
    
    return {
        "total_matrices": len(STATE),
        "total_narratives": total_narratives,
        "total_packs": len(PACKS),
        "matrices": rho_info,
        "data_dir": DATA_DIR
    }

@app.get("/database/narratives/{rho_id}")
def get_narratives(rho_id: str):
    """Get all narratives that have been read into a specific rho"""
    item = STATE.get(rho_id)
    if not item:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    narratives = []
    for entry in item.get("log", []):
        if entry.get("op") == "read":
            narratives.append({
                "timestamp": entry.get("ts"),
                "text": entry.get("text", ""),
                "alpha": entry.get("alpha", 0.0)
            })
        elif entry.get("action") == "read_chunk_hierarchical":
            narratives.append({
                "timestamp": entry.get("timestamp", 0),
                "text": entry.get("text_preview", ""),
                "alpha": 0.25,  # Default reading_alpha for books
                "chunk_index": entry.get("chunk_index", 0),
                "commentary": entry.get("commentary", "")
            })
    
    return {"rho_id": rho_id, "narratives": narratives}

@app.delete("/database/matrix/{rho_id}")
def delete_matrix(rho_id: str):
    """Delete a specific matrix by ID"""
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Matrix not found")
    
    # Prevent deletion of important system matrices
    protected_matrices = ["global_consciousness", COMPOSITE_MATRIX_ID, PREVIEW_MATRIX_ID]
    if rho_id in protected_matrices:
        raise HTTPException(status_code=400, detail=f"Cannot delete protected matrix: {rho_id}")
    
    matrix_info = {
        "rho_id": rho_id,
        "narratives": len([entry for entry in STATE[rho_id].get("log", []) if entry.get("op") == "read" or entry.get("action") == "read_chunk_hierarchical"]),
        "operations": len(STATE[rho_id].get("log", [])),
        "meta": STATE[rho_id].get("meta", {})
    }
    
    del STATE[rho_id]
    return {"deleted": matrix_info}

class CleanupCriteria(BaseModel):
    max_narratives: Optional[int] = None
    min_narratives: Optional[int] = None
    label_pattern: Optional[str] = None
    exclude_labels: Optional[List[str]] = None
    older_than_hours: Optional[float] = None
    
@app.post("/database/cleanup")
def cleanup_matrices(criteria: CleanupCriteria):
    """Delete matrices based on criteria"""
    import time
    
    deleted_matrices = []
    protected_matrices = {"global_consciousness", COMPOSITE_MATRIX_ID, PREVIEW_MATRIX_ID}
    
    # Collect matrices that match criteria
    to_delete = []
    current_time = time.time()
    
    for rho_id, item in STATE.items():
        if rho_id in protected_matrices:
            continue
            
        # Check narrative count criteria
        log_entries = item.get("log", [])
        narrative_count = len([entry for entry in log_entries if entry.get("op") == "read" or entry.get("action") == "read_chunk_hierarchical"])
        
        if criteria.max_narratives is not None and narrative_count > criteria.max_narratives:
            continue
        if criteria.min_narratives is not None and narrative_count < criteria.min_narratives:
            continue
            
        # Check label pattern
        label = item.get("meta", {}).get("label", "")
        if criteria.label_pattern and criteria.label_pattern not in label:
            continue
        if criteria.exclude_labels and any(excl in label for excl in criteria.exclude_labels):
            continue
            
        # Check age criteria
        if criteria.older_than_hours:
            created_time = item.get("meta", {}).get("created", current_time)
            age_hours = (current_time - created_time) / 3600
            if age_hours < criteria.older_than_hours:
                continue
        
        # Matrix matches criteria - mark for deletion
        matrix_info = {
            "rho_id": rho_id,
            "narratives": narrative_count,
            "operations": len(log_entries),
            "label": label,
            "meta": item.get("meta", {})
        }
        to_delete.append((rho_id, matrix_info))
    
    # Delete matrices
    for rho_id, matrix_info in to_delete:
        del STATE[rho_id]
        deleted_matrices.append(matrix_info)
    
    return {
        "deleted_count": len(deleted_matrices),
        "deleted_matrices": deleted_matrices,
        "criteria_used": criteria.dict()
    }

@app.post("/database/cleanup/duplicates")  
def cleanup_duplicates():
    """Remove duplicate matrices with identical properties"""
    import time
    
    # Group matrices by their key properties
    groups = {}
    protected_matrices = {"global_consciousness", COMPOSITE_MATRIX_ID, PREVIEW_MATRIX_ID}
    
    for rho_id, item in STATE.items():
        if rho_id in protected_matrices:
            continue
            
        # Create signature based on narrative count, purity, entropy
        log_entries = item.get("log", [])
        narrative_count = len([entry for entry in log_entries if entry.get("op") == "read" or entry.get("action") == "read_chunk_hierarchical"])
        diagnostics = globals()['diagnostics'](item["rho"])  # Call diagnostics function
        
        signature = (
            narrative_count,
            round(diagnostics["purity"], 6),
            round(diagnostics["entropy"], 3),
            item.get("meta", {}).get("label", "")
        )
        
        if signature not in groups:
            groups[signature] = []
        groups[signature].append((rho_id, item))
    
    # Keep only the oldest matrix in each duplicate group
    deleted_matrices = []
    for signature, matrices in groups.items():
        if len(matrices) <= 1:
            continue
            
        # Sort by creation time (oldest first)
        matrices.sort(key=lambda x: x[1].get("meta", {}).get("created", time.time()))
        
        # Keep first (oldest), delete the rest
        for rho_id, item in matrices[1:]:
            matrix_info = {
                "rho_id": rho_id,
                "narratives": signature[0],
                "purity": signature[1], 
                "entropy": signature[2],
                "label": signature[3],
                "duplicate_of": matrices[0][0]
            }
            del STATE[rho_id]
            deleted_matrices.append(matrix_info)
    
    return {
        "deleted_count": len(deleted_matrices),
        "deleted_matrices": deleted_matrices,
        "duplicate_groups_found": len([g for g in groups.values() if len(g) > 1])
    }

class BatchImportReq(BaseModel):
    texts: List[str]
    alpha: float = 0.2
    rho_id: Optional[str] = None

@app.post("/database/import_batch")
def import_batch(req: BatchImportReq):
    """Import multiple texts into a density matrix"""
    rho_id = req.rho_id
    if rho_id and rho_id not in STATE:
        raise HTTPException(status_code=404, detail="rho_id not found")
    
    if not rho_id:
        # Create new matrix
        rho_id = str(uuid.uuid4())
        STATE[rho_id] = {"rho": np.eye(DIM) / DIM, "log": [], "meta": {"label": "batch_import"}}
    
    results = []
    for text in req.texts:
        if not text or not text.strip():
            continue
            
        x = embed(text)
        v = project_to_local(x)
        pure = np.outer(v, v)
        rho_old = STATE[rho_id]["rho"]
        rho_new = (1.0 - req.alpha) * rho_old + req.alpha * pure
        rho_new = psd_project(rho_new)
        STATE[rho_id]["rho"] = rho_new
        
        log_entry = {"ts": time.time(), "op": "read", "alpha": req.alpha, "text": text}
        STATE[rho_id]["log"].append(log_entry)
        
        results.append({"text_preview": text[:100], "success": True})
    
    return {
        "rho_id": rho_id,
        "imported_count": len(results),
        "results": results,
        **diagnostics(STATE[rho_id]["rho"])
    }

# --- Dual Matrix Management ---

COMPOSITE_MATRIX_ID = "composite_matrix"
PREVIEW_MATRIX_ID = "preview_matrix"

@app.post("/matrix/dual/init")
def init_dual_matrices():
    """Initialize composite and preview matrices"""
    # Composite starts as maximally mixed
    STATE[COMPOSITE_MATRIX_ID] = {
        "rho": np.eye(DIM) / DIM, 
        "log": [], 
        "meta": {"label": "composite", "type": "composite"}
    }
    
    # Preview starts as copy of composite
    STATE[PREVIEW_MATRIX_ID] = {
        "rho": STATE[COMPOSITE_MATRIX_ID]["rho"].copy(), 
        "log": [], 
        "meta": {"label": "preview", "type": "preview"}
    }
    
    return {
        "composite": {"rho_id": COMPOSITE_MATRIX_ID, **diagnostics(STATE[COMPOSITE_MATRIX_ID]["rho"])},
        "preview": {"rho_id": PREVIEW_MATRIX_ID, **diagnostics(STATE[PREVIEW_MATRIX_ID]["rho"])}
    }

class PreviewReq(BaseModel):
    text: str
    alpha: float = 0.2

@app.post("/matrix/dual/preview_narrative")
def preview_narrative(req: PreviewReq):
    """Preview effect of adding narrative without applying to composite"""
    if PREVIEW_MATRIX_ID not in STATE or COMPOSITE_MATRIX_ID not in STATE:
        raise HTTPException(status_code=400, detail="Dual matrices not initialized")
    
    # Reset preview to composite state
    STATE[PREVIEW_MATRIX_ID]["rho"] = STATE[COMPOSITE_MATRIX_ID]["rho"].copy()
    
    # Apply text to preview
    x = embed(req.text)
    v = project_to_local(x)
    pure = np.outer(v, v)
    rho_preview = (1.0 - req.alpha) * STATE[PREVIEW_MATRIX_ID]["rho"] + req.alpha * pure
    rho_preview = psd_project(rho_preview)
    STATE[PREVIEW_MATRIX_ID]["rho"] = rho_preview
    
    # Calculate difference
    diff = STATE[PREVIEW_MATRIX_ID]["rho"] - STATE[COMPOSITE_MATRIX_ID]["rho"]
    diff_magnitude = float(np.linalg.norm(diff))
    
    return {
        "text_preview": req.text[:200],
        "alpha": req.alpha,
        "composite": diagnostics(STATE[COMPOSITE_MATRIX_ID]["rho"]),
        "preview": diagnostics(STATE[PREVIEW_MATRIX_ID]["rho"]),
        "difference_magnitude": diff_magnitude,
        "eigenvalue_changes": {
            "before": diagnostics(STATE[COMPOSITE_MATRIX_ID]["rho"])["eigs"][:5],
            "after": diagnostics(STATE[PREVIEW_MATRIX_ID]["rho"])["eigs"][:5]
        }
    }

@app.post("/matrix/dual/apply_preview")
def apply_preview():
    """Apply the preview changes to the composite matrix"""
    if PREVIEW_MATRIX_ID not in STATE or COMPOSITE_MATRIX_ID not in STATE:
        raise HTTPException(status_code=400, detail="Dual matrices not initialized")
    
    # Copy preview to composite
    STATE[COMPOSITE_MATRIX_ID]["rho"] = STATE[PREVIEW_MATRIX_ID]["rho"].copy()
    STATE[COMPOSITE_MATRIX_ID]["log"].append({
        "ts": time.time(), 
        "op": "apply_preview", 
        "note": "Applied preview changes to composite"
    })
    
    return {
        "composite": {"rho_id": COMPOSITE_MATRIX_ID, **diagnostics(STATE[COMPOSITE_MATRIX_ID]["rho"])},
        "message": "Preview changes applied to composite matrix"
    }

# --- LLM-like Query Interface ---

class QueryReq(BaseModel):
    query: str
    rho_id: Optional[str] = None

@app.post("/matrix/query")
def query_matrix(req: QueryReq):
    """Query the density matrix as if it were an LLM, using loaded narratives"""
    rho_id = req.rho_id or COMPOSITE_MATRIX_ID
        
    if rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Matrix not found")
    
    item = STATE[rho_id]
    
    # Get all narratives that have been loaded
    narratives = []
    for entry in item.get("log", []):
        if entry.get("op") == "read":
            narratives.append(entry.get("text", ""))
        elif entry.get("action") == "read_chunk_hierarchical":
            # Include book chunks as narratives
            text_preview = entry.get("text_preview", "")
            if text_preview:
                narratives.append(text_preview)
    
    # If this is a composite matrix (like global_consciousness) with no direct narratives,
    # aggregate narratives from constituent book matrices
    if not narratives and (rho_id == COMPOSITE_MATRIX_ID or rho_id == "global_consciousness"):
        meta = item.get("meta", {})
        book_titles = meta.get("book_titles", [])
        for book_title in book_titles:
            # Convert book title to matrix ID format
            book_rho_id = book_title.replace("Book_", "")
            if book_rho_id in STATE:
                book_item = STATE[book_rho_id]
                for entry in book_item.get("log", []):
                    if entry.get("op") == "read":
                        narratives.append(entry.get("text", ""))
                    elif entry.get("action") == "read_chunk_hierarchical":
                        text_preview = entry.get("text_preview", "")
                        if text_preview:
                            narratives.append(text_preview)
    
    if not narratives:
        return {
            "query": req.query,
            "response": "No narratives have been loaded into this matrix yet.",
            "narrative_count": 0,
            "matrix_state": diagnostics(item["rho"])
        }
    
    # Simple response generation based on matrix state and narratives
    # This is a simplified implementation - in practice you'd want more sophisticated text generation
    rho_diag = diagnostics(item["rho"])
    
    response_parts = [
        f"Based on {len(narratives)} narratives in this matrix (purity: {rho_diag['purity']:.3f}):",
        f"The loaded narratives include: {' ... '.join([n[:50] for n in narratives[:3]])}{'...' if len(narratives) > 3 else ''}",
        f"Matrix entropy: {rho_diag['entropy']:.3f}, dominant eigenvalue: {rho_diag['eigs'][0]:.3f}"
    ]
    
    return {
        "query": req.query,
        "response": " ".join(response_parts),
        "narrative_count": len(narratives),
        "matrix_state": rho_diag,
        "loaded_narratives": [n[:100] for n in narratives[:5]]  # First 5 for reference
    }

# --- Attribute Extraction and Manipulation ---

class AttributeExtractionReq(BaseModel):
    text: str
    rho_id: Optional[str] = None

class AttributeAdjustmentReq(BaseModel):
    persona_strength: float = 0.0  # -1 to +1
    namespace_strength: float = 0.0  # -1 to +1  
    style_strength: float = 0.0  # -1 to +1
    base_rho_id: str

class NarrativeRegenerationReq(BaseModel):
    original_text: str
    adjusted_rho_id: str

# Comprehensive mapping of narrative attributes to POVM basis vectors
ATTRIBUTE_MAPPING = {
    # Core Persona Attributes
    "persona": {
        "basis_vectors": [0, 1, 7, 9],  # personal_focus, reliability, certainty, politeness
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Voice personality and perspective",
        "category": "persona"
    },
    "authority": {
        "basis_vectors": [1, 7, 6, 4],  # reliability, certainty, formality, agency
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Authoritative and commanding presence",
        "category": "persona"
    },
    "empathy": {
        "basis_vectors": [9, 2, 0, 5],  # politeness, affect_valence, personal_focus, myth_realism
        "positive_direction": [1.0, 1.0, 1.0, -1.0],
        "description": "Emotional understanding and connection",
        "category": "persona"
    },
    "confidence": {
        "basis_vectors": [7, 4, 11, 1],  # certainty, agency, intensity, reliability
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Self-assurance and conviction",
        "category": "persona"
    },
    "humility": {
        "basis_vectors": [7, 4, 9, 6],  # certainty, agency, politeness, formality
        "positive_direction": [-1.0, -1.0, 1.0, -1.0],
        "description": "Modest and unassuming tone",
        "category": "persona"
    },
    "curiosity": {
        "basis_vectors": [0, 5, 11, 8],  # personal_focus, myth_realism, intensity, temporal_distance
        "positive_direction": [1.0, -1.0, 1.0, -1.0],
        "description": "Inquisitive and exploratory nature",
        "category": "persona"
    },
    
    # Namespace/Context Attributes
    "namespace": {
        "basis_vectors": [5, 8, 2],  # myth_realism, temporal_distance, affect_valence  
        "positive_direction": [-1.0, -1.0, 1.0],
        "description": "Contextual domain and reality frame",
        "category": "context"
    },
    "scientific": {
        "basis_vectors": [5, 7, 6, 1],  # myth_realism, certainty, formality, reliability
        "positive_direction": [-1.0, 1.0, 1.0, 1.0],
        "description": "Scientific and analytical framework",
        "category": "context"
    },
    "mystical": {
        "basis_vectors": [5, 8, 2, 11],  # myth_realism, temporal_distance, affect_valence, intensity
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Spiritual and transcendent context",
        "category": "context"
    },
    "historical": {
        "basis_vectors": [8, 6, 5, 1],  # temporal_distance, formality, myth_realism, reliability
        "positive_direction": [1.0, 1.0, 0.5, 1.0],
        "description": "Historical and traditional perspective",
        "category": "context"
    },
    "futuristic": {
        "basis_vectors": [8, 11, 4, 5],  # temporal_distance, intensity, agency, myth_realism
        "positive_direction": [-1.0, 1.0, 1.0, -1.0],
        "description": "Forward-looking and speculative",
        "category": "context"
    },
    "psychological": {
        "basis_vectors": [0, 2, 3, 9],  # personal_focus, affect_valence, arousal, politeness
        "positive_direction": [1.0, 0.5, 1.0, 1.0],
        "description": "Mental and emotional analysis",
        "category": "context"
    },
    "philosophical": {
        "basis_vectors": [5, 8, 7, 0],  # myth_realism, temporal_distance, certainty, personal_focus
        "positive_direction": [0.5, 0.5, -0.5, 1.0],
        "description": "Abstract and contemplative reasoning",
        "category": "context"
    },
    
    # Style Attributes
    "style": {
        "basis_vectors": [6, 11, 4, 3],  # formality, intensity, agency, arousal
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Expressive manner and energy",
        "category": "style"
    },
    "elegance": {
        "basis_vectors": [6, 9, 2, 11],  # formality, politeness, affect_valence, intensity
        "positive_direction": [1.0, 1.0, 1.0, -0.5],
        "description": "Refined and graceful expression",
        "category": "style"
    },
    "urgency": {
        "basis_vectors": [11, 3, 4, 8],  # intensity, arousal, agency, temporal_distance
        "positive_direction": [1.0, 1.0, 1.0, -1.0],
        "description": "Pressing and immediate tone",
        "category": "style"
    },
    "playfulness": {
        "basis_vectors": [6, 2, 3, 9],  # formality, affect_valence, arousal, politeness
        "positive_direction": [-1.0, 1.0, 1.0, 1.0],
        "description": "Light-hearted and fun approach",
        "category": "style"
    },
    "darkness": {
        "basis_vectors": [2, 11, 3, 5],  # affect_valence, intensity, arousal, myth_realism
        "positive_direction": [-1.0, 1.0, 1.0, 1.0],
        "description": "Dark and somber atmosphere",
        "category": "style"
    },
    "clarity": {
        "basis_vectors": [6, 7, 1, 4],  # formality, certainty, reliability, agency
        "positive_direction": [0.5, 1.0, 1.0, 1.0],
        "description": "Clear and direct communication",
        "category": "style"
    },
    "complexity": {
        "basis_vectors": [6, 8, 5, 11],  # formality, temporal_distance, myth_realism, intensity
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Sophisticated and layered expression",
        "category": "style"
    },
    "simplicity": {
        "basis_vectors": [6, 7, 9, 2],  # formality, certainty, politeness, affect_valence
        "positive_direction": [-1.0, 1.0, 1.0, 1.0],
        "description": "Simple and straightforward approach",
        "category": "style"
    },
    
    # Emotional Attributes
    "passion": {
        "basis_vectors": [11, 3, 2, 4],  # intensity, arousal, affect_valence, agency
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Passionate and fervent expression",
        "category": "emotion"
    },
    "serenity": {
        "basis_vectors": [11, 3, 2, 9],  # intensity, arousal, affect_valence, politeness
        "positive_direction": [-1.0, -1.0, 1.0, 1.0],
        "description": "Calm and peaceful tone",
        "category": "emotion"
    },
    "melancholy": {
        "basis_vectors": [2, 3, 8, 11],  # affect_valence, arousal, temporal_distance, intensity
        "positive_direction": [-1.0, -0.5, 1.0, 0.5],
        "description": "Wistful and contemplative sadness",
        "category": "emotion"
    },
    "joy": {
        "basis_vectors": [2, 3, 11, 9],  # affect_valence, arousal, intensity, politeness
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Joyful and celebratory energy",
        "category": "emotion"
    },
    "tension": {
        "basis_vectors": [3, 11, 7, 4],  # arousal, intensity, certainty, agency
        "positive_direction": [1.0, 1.0, -0.5, 1.0],
        "description": "Suspenseful and tense atmosphere",
        "category": "emotion"
    },
    "wonder": {
        "basis_vectors": [2, 5, 0, 11],  # affect_valence, myth_realism, personal_focus, intensity
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Awe and amazement",
        "category": "emotion"
    },
    
    # Rhetorical Attributes
    "persuasion": {
        "basis_vectors": [4, 7, 1, 11],  # agency, certainty, reliability, intensity
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Convincing and influential approach",
        "category": "rhetoric"
    },
    "objectivity": {
        "basis_vectors": [0, 2, 6, 1],  # personal_focus, affect_valence, formality, reliability
        "positive_direction": [-1.0, 0.0, 1.0, 1.0],
        "description": "Neutral and unbiased perspective",
        "category": "rhetoric"
    },
    "subjectivity": {
        "basis_vectors": [0, 2, 11, 9],  # personal_focus, affect_valence, intensity, politeness
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Personal and experiential viewpoint",
        "category": "rhetoric"
    },
    "narrative": {
        "basis_vectors": [8, 0, 4, 2],  # temporal_distance, personal_focus, agency, affect_valence
        "positive_direction": [0.5, 1.0, 1.0, 1.0],
        "description": "Story-driven and sequential",
        "category": "rhetoric"
    },
    "analytical": {
        "basis_vectors": [7, 1, 6, 4],  # certainty, reliability, formality, agency
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Logical and systematic breakdown",
        "category": "rhetoric"
    },
    "poetic": {
        "basis_vectors": [5, 2, 11, 8],  # myth_realism, affect_valence, intensity, temporal_distance
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Artistic and metaphorical language",
        "category": "rhetoric"
    },
    
    # Temporal Attributes
    "immediacy": {
        "basis_vectors": [8, 11, 3, 4],  # temporal_distance, intensity, arousal, agency
        "positive_direction": [-1.0, 1.0, 1.0, 1.0],
        "description": "Present-moment focus and relevance",
        "category": "temporal"
    },
    "timelessness": {
        "basis_vectors": [8, 5, 6, 1],  # temporal_distance, myth_realism, formality, reliability
        "positive_direction": [0.0, 1.0, 1.0, 1.0],
        "description": "Universal and enduring quality",
        "category": "temporal"
    },
    "nostalgia": {
        "basis_vectors": [8, 2, 0, 5],  # temporal_distance, affect_valence, personal_focus, myth_realism
        "positive_direction": [1.0, 0.5, 1.0, 1.0],
        "description": "Wistful longing for the past",
        "category": "temporal"
    },
    "anticipation": {
        "basis_vectors": [8, 3, 4, 11],  # temporal_distance, arousal, agency, intensity
        "positive_direction": [-1.0, 1.0, 1.0, 1.0],
        "description": "Forward-looking expectation",
        "category": "temporal"
    },
    
    # Social Attributes
    "intimacy": {
        "basis_vectors": [0, 9, 6, 2],  # personal_focus, politeness, formality, affect_valence
        "positive_direction": [1.0, 1.0, -1.0, 1.0],
        "description": "Close and personal connection",
        "category": "social"
    },
    "distance": {
        "basis_vectors": [0, 6, 9, 1],  # personal_focus, formality, politeness, reliability
        "positive_direction": [-1.0, 1.0, 0.0, 1.0],
        "description": "Professional and detached manner",
        "category": "social"
    },
    "collaboration": {
        "basis_vectors": [9, 4, 0, 2],  # politeness, agency, personal_focus, affect_valence
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Cooperative and inclusive approach",
        "category": "social"
    },
    "rebellion": {
        "basis_vectors": [4, 11, 6, 7],  # agency, intensity, formality, certainty
        "positive_direction": [1.0, 1.0, -1.0, 1.0],
        "description": "Defiant and unconventional stance",
        "category": "social"
    },
    "conformity": {
        "basis_vectors": [6, 9, 1, 4],  # formality, politeness, reliability, agency
        "positive_direction": [1.0, 1.0, 1.0, -0.5],
        "description": "Adherence to norms and expectations",
        "category": "social"
    },
    
    # Sensory/Aesthetic Attributes
    "visual": {
        "basis_vectors": [11, 2, 5, 0],  # intensity, affect_valence, myth_realism, personal_focus
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Rich visual imagery and description",
        "category": "sensory"
    },
    "tactile": {
        "basis_vectors": [3, 11, 0, 2],  # arousal, intensity, personal_focus, affect_valence
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Physical sensations and texture",
        "category": "sensory"
    },
    "auditory": {
        "basis_vectors": [11, 8, 3, 5],  # intensity, temporal_distance, arousal, myth_realism
        "positive_direction": [1.0, 0.5, 1.0, 1.0],
        "description": "Sound, rhythm, and musical quality",
        "category": "sensory"
    },
    "minimalism": {
        "basis_vectors": [6, 11, 2, 9],  # formality, intensity, affect_valence, politeness
        "positive_direction": [-0.5, -1.0, 0.5, 1.0],
        "description": "Stripped-down and essential elements",
        "category": "aesthetic"
    },
    "maximalism": {
        "basis_vectors": [11, 5, 6, 2],  # intensity, myth_realism, formality, affect_valence
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Rich, elaborate, and abundant expression",
        "category": "aesthetic"
    },
    
    # Cognitive Attributes
    "logic": {
        "basis_vectors": [7, 1, 4, 6],  # certainty, reliability, agency, formality
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Rational and systematic thinking",
        "category": "cognitive"
    },
    "intuition": {
        "basis_vectors": [7, 5, 0, 2],  # certainty, myth_realism, personal_focus, affect_valence
        "positive_direction": [-0.5, 1.0, 1.0, 1.0],
        "description": "Instinctive and felt understanding",
        "category": "cognitive"
    },
    "memory": {
        "basis_vectors": [8, 0, 2, 5],  # temporal_distance, personal_focus, affect_valence, myth_realism
        "positive_direction": [1.0, 1.0, 0.5, 1.0],
        "description": "Recollection and remembrance",
        "category": "cognitive"
    },
    "imagination": {
        "basis_vectors": [5, 11, 2, 4],  # myth_realism, intensity, affect_valence, agency
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Creative and inventive thinking",
        "category": "cognitive"
    },
    "focus": {
        "basis_vectors": [4, 7, 11, 1],  # agency, certainty, intensity, reliability
        "positive_direction": [1.0, 1.0, 1.0, 1.0],
        "description": "Concentrated and directed attention",
        "category": "cognitive"
    },
    "wandering": {
        "basis_vectors": [4, 0, 5, 8],  # agency, personal_focus, myth_realism, temporal_distance
        "positive_direction": [-0.5, 1.0, 1.0, 1.0],
        "description": "Free-flowing and meandering thought",
        "category": "cognitive"
    }
}

@app.post("/attributes/extract")
def extract_attributes(req: AttributeExtractionReq):
    """Extract Persona, Namespace, and Style attributes from text using POVM measurements"""
    
    # Create temporary rho from the text
    x = embed(req.text)
    v = project_to_local(x)
    text_rho = np.outer(v, v)
    text_rho = psd_project(text_rho)
    
    # Apply POVM measurements for each attribute
    attributes = {}
    I = np.eye(DIM)
    
    for attr_name, attr_config in ATTRIBUTE_MAPPING.items():
        # Create composite measurement operator for this attribute
        total_score = 0.0
        weight_sum = 0.0
        
        for i, basis_idx in enumerate(attr_config["basis_vectors"]):
            # Create unit vector for this basis
            u = np.zeros(DIM)
            u[basis_idx] = 1.0
            
            # Measurement operator
            E_plus = np.outer(u, u)
            prob = float(np.trace(E_plus @ text_rho))
            
            # Apply directional weighting
            direction = attr_config["positive_direction"][i]
            weighted_prob = prob * direction
            
            total_score += weighted_prob
            weight_sum += abs(direction)
        
        # Normalize to [-1, +1] range
        normalized_score = total_score / weight_sum if weight_sum > 0 else 0.0
        normalized_score = max(-1.0, min(1.0, normalized_score * 4 - 2.0))  # Scale for visibility
        
        attributes[attr_name] = {
            "strength": normalized_score,
            "description": attr_config["description"],
            "basis_contribution": total_score,
            "components": []
        }
        
        # Add component breakdown
        for i, basis_idx in enumerate(attr_config["basis_vectors"]):
            u = np.zeros(DIM)
            u[basis_idx] = 1.0
            E_plus = np.outer(u, u)
            prob = float(np.trace(E_plus @ text_rho))
            
            # Get the axis name from the demo pack
            axis_name = f"axis_{basis_idx}"
            if "axes_12x2_demo" in PACKS:
                demo_pack = PACKS["axes_12x2_demo"]
                if basis_idx < len(demo_pack["axes"]):
                    axis_name = demo_pack["axes"][basis_idx]["id"]
            
            attributes[attr_name]["components"].append({
                "axis": axis_name,
                "basis_index": basis_idx,
                "probability": prob,
                "direction": attr_config["positive_direction"][i],
                "weighted_contribution": prob * attr_config["positive_direction"][i]
            })
    
    return {
        "text_preview": req.text[:200],
        "attributes": attributes,
        "matrix_diagnostics": diagnostics(text_rho)
    }

@app.post("/attributes/adjust_matrix")
def adjust_matrix(req: AttributeAdjustmentReq):
    """Create adjusted density matrix based on attribute strength adjustments"""
    
    if req.base_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Base matrix not found")
    
    base_rho = STATE[req.base_rho_id]["rho"].copy()
    adjusted_rho = base_rho.copy()
    
    # Apply adjustments for each attribute
    adjustments = {
        "persona": req.persona_strength,
        "namespace": req.namespace_strength, 
        "style": req.style_strength
    }
    
    adjustment_log = []
    
    for attr_name, strength_delta in adjustments.items():
        if abs(strength_delta) < 0.01:  # Skip tiny adjustments
            continue
            
        attr_config = ATTRIBUTE_MAPPING[attr_name]
        
        # Create adjustment operator
        adjustment_operator = np.zeros((DIM, DIM))
        
        for i, basis_idx in enumerate(attr_config["basis_vectors"]):
            # Unit vector for this basis dimension
            u = np.zeros(DIM)
            u[basis_idx] = 1.0
            
            # Direction and magnitude of adjustment
            direction = attr_config["positive_direction"][i]
            adjustment_magnitude = strength_delta * direction * 0.1  # Scale factor
            
            # Add rank-1 adjustment
            adjustment_operator += adjustment_magnitude * np.outer(u, u)
        
        # Apply adjustment to matrix
        adjusted_rho = adjusted_rho + adjustment_operator
        
        adjustment_log.append({
            "attribute": attr_name,
            "strength_delta": strength_delta,
            "operator_norm": float(np.linalg.norm(adjustment_operator))
        })
    
    # Ensure result is valid density matrix
    adjusted_rho = psd_project(adjusted_rho)
    
    # Store adjusted matrix
    adjusted_id = f"adjusted_{req.base_rho_id}_{int(time.time())}"
    STATE[adjusted_id] = {
        "rho": adjusted_rho,
        "log": [{
            "ts": time.time(),
            "op": "attribute_adjustment",
            "base_id": req.base_rho_id,
            "adjustments": adjustments,
            "adjustment_log": adjustment_log
        }],
        "meta": {"label": "adjusted", "type": "adjusted", "base_id": req.base_rho_id}
    }
    
    return {
        "adjusted_rho_id": adjusted_id,
        "base_diagnostics": diagnostics(base_rho),
        "adjusted_diagnostics": diagnostics(adjusted_rho),
        "adjustments_applied": adjustments,
        "adjustment_log": adjustment_log,
        "difference_magnitude": float(np.linalg.norm(adjusted_rho - base_rho))
    }

def apply_narrative_transformations(text: str, attribute_scores: dict, rho_diag: dict, modifications: list) -> str:
    """Apply rho-conditioned language generation for true post-lexical transformations"""
    logger.info(f"🔥 TRANSFORMATION CALLED: {text[:50]}... with scores: {attribute_scores}")
    try:
        # Import the quantum language generator
        from rho_language_generator import EvolvingRhoLLM, RhoState
        
        # Create a RhoState from the current matrix diagnostics
        rho_state = RhoState(
            matrix=np.eye(64),  # Placeholder - would use actual adjusted matrix
            measurements=attribute_scores,
            eigenvalues=np.array(rho_diag.get("eigs", [0.1] * 8)),
            eigenvectors=np.eye(64),  # Placeholder
            purity=rho_diag["purity"],
            entropy=rho_diag["entropy"],
            timestamp=time.time(),
            source_text=text
        )
        
        # Initialize the rho-conditioned LLM with correct port
        rho_llm = EvolvingRhoLLM(api_base_url="http://localhost:8192")
        
        # Determine the transformation mode based on attribute adjustments
        mode = "analytical"  # Default
        if abs(attribute_scores.get("persona", 0)) > 0.3:
            mode = "experiential"
        elif abs(attribute_scores.get("style", 0)) > 0.3:
            mode = "synthetic"
        
        # Generate transformation using quantum attention
        transformation_prompt = f"""Transform this narrative using the following matrix-guided adjustments:
Original: {text}

Attribute adjustments applied:
- Persona: {attribute_scores.get("persona", 0):.2f}
- Namespace: {attribute_scores.get("namespace", 0):.2f}  
- Style: {attribute_scores.get("style", 0):.2f}

Matrix properties:
- Purity: {rho_diag["purity"]:.3f} ({"focused" if rho_diag["purity"] > 0.3 else "diverse"})
- Entropy: {rho_diag["entropy"]:.3f} ({"complex" if rho_diag["entropy"] > 2.5 else "simple"})

Generate a semantically transformed version that reflects these quantum state changes."""

        # Use the NEW transform_narrative_text method instead of generate_response
        logger.info(f"🎯 CALLING transform_narrative_text with text: {text[:50]}...")
        response = rho_llm.transform_narrative_text(
            original_text=text,
            rho_state=rho_state,
            attribute_adjustments=attribute_scores
        )
        logger.info(f"🎯 TRANSFORMATION RESPONSE: {response[:50]}...")
        
        # The new method returns transformed text directly
        transformed = response
        
        # Return the real transformation result without fallback
        logger.info(f"🎯 REAL TRANSFORMATION RESULT: '{transformed}'")
        return transformed
        
    except Exception as e:
        logger.error(f"🚨 Rho-conditioned generation failed with exception: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        # Fallback to matrix-based transformation
        logger.warning(f"🔄 FALLING BACK to simple transformations due to: {e}")
        fallback_result = apply_matrix_based_transformation(text, attribute_scores, rho_diag)
        logger.info(f"🔄 Fallback transformation result: {fallback_result[:50]}...")
        return fallback_result

def apply_matrix_based_transformation(text: str, attribute_scores: dict, rho_diag: dict) -> str:
    """Fallback: Apply matrix-guided transformations without LLM"""
    import re
    
    transformed = text
    
    # Apply transformations based on attribute strengths
    persona_score = attribute_scores.get("persona", 0)
    style_score = attribute_scores.get("style", 0)
    namespace_score = attribute_scores.get("namespace", 0)
    
    # Persona adjustments
    if persona_score > 0.3:
        transformed = re.sub(r'\bhis\b', "the protagonist's", transformed, flags=re.IGNORECASE)
        transformed = re.sub(r'\bhe\b', "our central figure", transformed, flags=re.IGNORECASE)
    elif persona_score < -0.3:
        transformed = re.sub(r'knight', 'figure', transformed)
        
    # Style adjustments  
    if style_score > 0.3:
        transformed = re.sub(r'\bdark\b', 'shadow-veiled', transformed)
        transformed = re.sub(r'\bforest\b', 'sylvan depths', transformed)
    elif style_score < -0.3:
        transformed = re.sub(r'Ancient', 'Old', transformed)
        
    # Namespace adjustments
    if namespace_score > 0.3:
        transformed = re.sub(r'forest', 'forest realm of the ancients', transformed)
    elif namespace_score < -0.3:
        transformed = re.sub(r'the dark forest', 'shadows', transformed)
    
    # Matrix property adjustments
    if rho_diag["purity"] > 0.3:
        transformed = f"With crystalline clarity, {transformed.lower()}"
    elif rho_diag["entropy"] > 3.0:
        transformed += " Yet beneath lay infinite complexities, weaving through dimensions of meaning."
    
    return transformed

@app.post("/attributes/regenerate_narrative")  
def regenerate_narrative(req: NarrativeRegenerationReq):
    """Generate narrative variation based on adjusted density matrix"""
    logger.info(f"🎯 REGENERATE_NARRATIVE called with rho_id: {req.adjusted_rho_id}")
    
    if req.adjusted_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Adjusted matrix not found")
    
    adjusted_rho = STATE[req.adjusted_rho_id]["rho"]
    
    # This is a simplified narrative generation based on matrix state
    # In a full implementation, you'd use the adjusted matrix to guide
    # a more sophisticated text generation process
    
    # Extract key eigenvalues and eigenvectors for narrative guidance
    w, V = np.linalg.eigh(adjusted_rho)
    dominant_indices = np.argsort(w)[-3:][::-1]  # Top 3 eigenvalues
    
    # Analyze matrix properties
    rho_diag = diagnostics(adjusted_rho)
    
    # Simple narrative modification based on matrix properties
    # This is a placeholder - in practice you'd integrate with an LLM
    modifications = []
    
    if rho_diag["purity"] > 0.2:
        modifications.append("focused and coherent")
    if rho_diag["entropy"] < 2.0:
        modifications.append("structured")
    if w[dominant_indices[0]] > 0.3:
        modifications.append("strongly characterized")
    
    # Extract attribute measurements from adjusted matrix
    I = np.eye(DIM)
    attribute_scores = {}
    
    for attr_name, attr_config in ATTRIBUTE_MAPPING.items():
        total_score = 0.0
        for i, basis_idx in enumerate(attr_config["basis_vectors"]):
            u = np.zeros(DIM)
            u[basis_idx] = 1.0
            E_plus = np.outer(u, u)
            prob = float(np.trace(E_plus @ adjusted_rho))
            direction = attr_config["positive_direction"][i]
            total_score += prob * direction
        
        attribute_scores[attr_name] = total_score / len(attr_config["basis_vectors"])
    
    # Generate modified narrative with actual text transformations
    logger.info(f"🎯 ABOUT TO CALL apply_narrative_transformations with text: {req.original_text[:50]}...")
    modified_narrative = apply_narrative_transformations(
        req.original_text, 
        attribute_scores, 
        rho_diag,
        modifications
    )
    logger.info(f"🎯 TRANSFORMATION RESULT: {modified_narrative[:50]}...")
    
    return {
        "original_text": req.original_text,
        "modified_narrative": modified_narrative,
        "matrix_state": rho_diag,
        "attribute_scores": attribute_scores,
        "modification_notes": modifications,
        "dominant_eigenvalues": [float(w[i]) for i in dominant_indices],
        "regeneration_method": "matrix_guided_simple"
    }

# Custom attribute creation request model
class CustomAttributeReq(BaseModel):
    name: str
    strength: float
    text: str

def generate_descriptive_attribute_name(text: str, embedding: np.ndarray, rho_matrix: np.ndarray) -> str:
    """Generate a descriptive name for an attribute based on its embedding characteristics"""
    
    # Analyze text content for semantic themes
    text_lower = text.lower()
    themes = []
    
    # Emotional/tonal themes
    if any(word in text_lower for word in ['happy', 'joy', 'delight', 'cheerful', 'bright']):
        themes.append("joy")
    elif any(word in text_lower for word in ['sad', 'melancholy', 'sorrow', 'grief', 'dark']):
        themes.append("melancholy") 
    elif any(word in text_lower for word in ['anger', 'rage', 'fury', 'hostile', 'aggressive']):
        themes.append("intensity")
    elif any(word in text_lower for word in ['calm', 'peaceful', 'serene', 'tranquil', 'quiet']):
        themes.append("serenity")
    
    # Cognitive/perspective themes
    if any(word in text_lower for word in ['think', 'reason', 'logic', 'rational', 'analyze']):
        themes.append("analytical")
    elif any(word in text_lower for word in ['feel', 'intuition', 'sense', 'instinct', 'gut']):
        themes.append("intuitive")
    elif any(word in text_lower for word in ['creative', 'imaginative', 'artistic', 'innovative']):
        themes.append("creative")
    
    # Social/interpersonal themes
    if any(word in text_lower for word in ['together', 'community', 'cooperation', 'team', 'collective']):
        themes.append("collaborative")
    elif any(word in text_lower for word in ['individual', 'personal', 'self', 'private', 'alone']):
        themes.append("personal")
    elif any(word in text_lower for word in ['formal', 'official', 'professional', 'proper']):
        themes.append("formal")
    elif any(word in text_lower for word in ['casual', 'informal', 'relaxed', 'friendly']):
        themes.append("informal")
    
    # Action/agency themes
    if any(word in text_lower for word in ['action', 'decisive', 'bold', 'assertive', 'confident']):
        themes.append("assertive")
    elif any(word in text_lower for word in ['passive', 'gentle', 'soft', 'subtle', 'yielding']):
        themes.append("gentle")
    
    # Temporal themes
    if any(word in text_lower for word in ['past', 'history', 'memory', 'nostalgia', 'tradition']):
        themes.append("nostalgic")
    elif any(word in text_lower for word in ['future', 'vision', 'hope', 'tomorrow', 'next']):
        themes.append("anticipatory")
    elif any(word in text_lower for word in ['now', 'present', 'immediate', 'current', 'today']):
        themes.append("immediate")
    
    # Analyze embedding characteristics
    rho_diag = diagnostics(rho_matrix)
    
    # Use rho diagnostics to refine naming
    complexity_suffix = ""
    if rho_diag["purity"] > 0.6:
        complexity_suffix = "_focused"
    elif rho_diag["entropy"] > 3.0:
        complexity_suffix = "_complex"
    elif rho_diag["purity"] < 0.3:
        complexity_suffix = "_diverse"
    
    # Generate name based on themes and characteristics
    if themes:
        primary_theme = themes[0]
        if len(themes) > 1:
            return f"{primary_theme}_{themes[1]}{complexity_suffix}"
        else:
            return f"{primary_theme}{complexity_suffix}"
    else:
        # Fallback based on embedding characteristics
        if rho_diag["purity"] > 0.5:
            return f"coherent_attribute_{hash(text[:50]) % 1000}"
        else:
            return f"mixed_attribute_{hash(text[:50]) % 1000}"

@app.post("/attributes/create_custom")
def create_custom_attribute(req: CustomAttributeReq):
    """Create a new custom attribute based on text analysis"""
    
    # Embed the text to understand its semantic space
    x = embed(req.text)
    v = project_to_local(x)
    text_rho = np.outer(v, v)
    text_rho = psd_project(text_rho)
    
    # Generate descriptive name if none provided or if requested
    descriptive_name = req.name
    if not req.name or req.name.lower() in ['auto', 'generate', 'automatic']:
        descriptive_name = generate_descriptive_attribute_name(req.text, x, text_rho)
    
    # Find dominant eigenvalues/eigenvectors
    w, V = np.linalg.eigh(text_rho)
    dominant_indices = np.argsort(w)[-5:][::-1]  # Top 5 eigenvalues
    
    # Create POVM components from dominant eigenvector projections
    povm_components = []
    for i, idx in enumerate(dominant_indices[:3]):  # Use top 3 for custom attribute
        eigenvector = V[:, idx]
        # Project to specific basis vectors for interpretability
        basis_idx = int(np.argmax(np.abs(eigenvector)))
        probability = float(w[idx])
        
        povm_components.append({
            "basis_index": basis_idx,
            "probability": probability,
            "eigenvector_contribution": float(np.abs(eigenvector[basis_idx])),
            "component_weight": req.strength
        })
    
    # Generate enhanced semantic description
    rho_diag = diagnostics(text_rho)
    description = f"Attribute '{descriptive_name}' with strength {req.strength:.2f}. "
    
    # Add semantic analysis to description
    text_lower = req.text.lower()
    semantic_hints = []
    if any(word in text_lower for word in ['emotion', 'feel', 'heart']):
        semantic_hints.append("emotional resonance")
    if any(word in text_lower for word in ['think', 'mind', 'reason']):
        semantic_hints.append("cognitive focus")
    if any(word in text_lower for word in ['social', 'people', 'relationship']):
        semantic_hints.append("interpersonal dynamics")
    if any(word in text_lower for word in ['time', 'moment', 'duration']):
        semantic_hints.append("temporal awareness")
    
    if semantic_hints:
        description += f"Captures {', '.join(semantic_hints)}. "
    
    if rho_diag["purity"] > 0.3:
        description += "High coherence indicates focused semantic space. "
    if rho_diag["entropy"] > 2.5:
        description += "Complex multi-dimensional semantic profile."
    else:
        description += "Concentrated semantic profile."
    
    return {
        "attribute_name": descriptive_name,
        "original_name": req.name,
        "strength": req.strength,
        "povm_components": povm_components,
        "semantic_description": description,
        "source_text_diagnostics": rho_diag,
        "basis_coverage": len(povm_components),
        "auto_generated_name": descriptive_name != req.name
    }

@app.get("/attributes/list")
def list_attributes():
    """List all available attributes grouped by category"""
    categories = {}
    for attr_name, attr_config in ATTRIBUTE_MAPPING.items():
        category = attr_config.get("category", "other")
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "name": attr_name,
            "description": attr_config["description"],
            "basis_vectors": attr_config["basis_vectors"],
            "dimension_count": len(attr_config["basis_vectors"])
        })
    
    return {
        "categories": categories,
        "total_attributes": len(ATTRIBUTE_MAPPING),
        "category_counts": {cat: len(attrs) for cat, attrs in categories.items()}
    }

class SentencePreviewReq(BaseModel):
    text: str
    attribute_adjustments: Dict[str, float]  # attribute_name -> strength (-1.0 to 1.0)
    preview_sentences: int = 3  # Number of sentences to preview

@app.post("/attributes/preview_sentences")
def preview_sentence_modifications(req: SentencePreviewReq):
    """Preview how attribute adjustments would modify individual sentences"""
    
    # Split text into sentences (simple approach)
    import re
    sentences = re.split(r'[.!?]+', req.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {"error": "No sentences found in text"}
    
    # Limit to requested number of sentences
    preview_sentences = sentences[:req.preview_sentences]
    
    results = []
    
    for i, sentence in enumerate(preview_sentences):
        if not sentence:
            continue
            
        # Extract current attributes from sentence
        x = embed(sentence)
        v = project_to_local(x)
        sentence_rho = np.outer(v, v)
        sentence_rho = psd_project(sentence_rho)
        
        # Get original attribute measurements
        original_attrs = {}
        I = np.eye(DIM)
        
        for attr_name, adjustment in req.attribute_adjustments.items():
            if attr_name in ATTRIBUTE_MAPPING:
                attr_config = ATTRIBUTE_MAPPING[attr_name]
                total_score = 0.0
                weight_sum = 0.0
                
                for j, basis_idx in enumerate(attr_config["basis_vectors"]):
                    u = np.zeros(DIM)
                    u[basis_idx] = 1.0
                    E_plus = np.outer(u, u)
                    prob = float(np.trace(E_plus @ sentence_rho))
                    direction = attr_config["positive_direction"][j]
                    weighted_prob = prob * direction
                    total_score += weighted_prob
                    weight_sum += abs(direction)
                
                original_attrs[attr_name] = total_score / weight_sum if weight_sum > 0 else 0.0
        
        # Apply attribute adjustments to create modified rho
        modified_rho = sentence_rho.copy()
        
        for attr_name, adjustment in req.attribute_adjustments.items():
            if attr_name in ATTRIBUTE_MAPPING and abs(adjustment) > 0.01:
                attr_config = ATTRIBUTE_MAPPING[attr_name]
                
                # Create combined POVM operator for this attribute
                combined_E = np.zeros((DIM, DIM))
                for j, basis_idx in enumerate(attr_config["basis_vectors"]):
                    u = np.zeros(DIM)
                    u[basis_idx] = 1.0
                    E_component = np.outer(u, u)
                    direction = attr_config["positive_direction"][j]
                    combined_E += direction * E_component
                
                # Apply adjustment: rho_new = rho + adjustment * (E * rho * E† - rho)
                adjustment_operator = combined_E @ modified_rho @ combined_E.T
                modified_rho = (1.0 - abs(adjustment)) * modified_rho + abs(adjustment) * adjustment_operator
                modified_rho = psd_project(modified_rho)
        
        # Generate modified text using transformation
        try:
            modified_sentence = apply_narrative_transformations(
                sentence, req.attribute_adjustments, diagnostics(modified_rho), []
            )
        except Exception as e:
            logger.warning(f"Transformation failed for sentence {i}: {e}")
            modified_sentence = sentence  # Fallback to original
        
        # Calculate new attribute scores
        modified_attrs = {}
        for attr_name, adjustment in req.attribute_adjustments.items():
            if attr_name in ATTRIBUTE_MAPPING:
                attr_config = ATTRIBUTE_MAPPING[attr_name]
                total_score = 0.0
                weight_sum = 0.0
                
                for j, basis_idx in enumerate(attr_config["basis_vectors"]):
                    u = np.zeros(DIM)
                    u[basis_idx] = 1.0
                    E_plus = np.outer(u, u)
                    prob = float(np.trace(E_plus @ modified_rho))
                    direction = attr_config["positive_direction"][j]
                    weighted_prob = prob * direction
                    total_score += weighted_prob
                    weight_sum += abs(direction)
                
                modified_attrs[attr_name] = total_score / weight_sum if weight_sum > 0 else 0.0
        
        results.append({
            "sentence_index": i,
            "original_sentence": sentence,
            "modified_sentence": modified_sentence,
            "original_attributes": original_attrs,
            "modified_attributes": modified_attrs,
            "attribute_changes": {
                attr: modified_attrs.get(attr, 0) - original_attrs.get(attr, 0)
                for attr in req.attribute_adjustments.keys()
                if attr in ATTRIBUTE_MAPPING
            },
            "transformation_success": modified_sentence != sentence
        })
    
    return {
        "preview_results": results,
        "total_sentences": len(sentences),
        "previewed_sentences": len(results),
        "adjustments_applied": req.attribute_adjustments
    }

class BatchAttributeExtractionReq(BaseModel):
    texts: List[str]
    attributes_filter: Optional[List[str]] = None  # Only extract specific attributes

@app.post("/attributes/extract_batch")
def extract_attributes_batch(req: BatchAttributeExtractionReq):
    """Extract attributes from multiple texts efficiently"""
    
    if not req.texts:
        return {"error": "No texts provided"}
    
    # Determine which attributes to extract
    target_attributes = req.attributes_filter if req.attributes_filter else list(ATTRIBUTE_MAPPING.keys())
    target_attributes = [attr for attr in target_attributes if attr in ATTRIBUTE_MAPPING]
    
    results = []
    
    for text_idx, text in enumerate(req.texts):
        if not text.strip():
            results.append({
                "text_index": text_idx,
                "text_preview": "",
                "attributes": {},
                "error": "Empty text"
            })
            continue
            
        try:
            # Create rho from text
            x = embed(text)
            v = project_to_local(x)
            text_rho = np.outer(v, v)
            text_rho = psd_project(text_rho)
            
            # Extract attributes
            attributes = {}
            I = np.eye(DIM)
            
            for attr_name in target_attributes:
                attr_config = ATTRIBUTE_MAPPING[attr_name]
                total_score = 0.0
                weight_sum = 0.0
                
                for i, basis_idx in enumerate(attr_config["basis_vectors"]):
                    u = np.zeros(DIM)
                    u[basis_idx] = 1.0
                    E_plus = np.outer(u, u)
                    prob = float(np.trace(E_plus @ text_rho))
                    direction = attr_config["positive_direction"][i]
                    weighted_prob = prob * direction
                    total_score += weighted_prob
                    weight_sum += abs(direction)
                
                normalized_score = total_score / weight_sum if weight_sum > 0 else 0.0
                normalized_score = max(-1.0, min(1.0, normalized_score * 4 - 2.0))
                
                attributes[attr_name] = {
                    "strength": normalized_score,
                    "description": attr_config["description"],
                    "category": attr_config["category"]
                }
            
            results.append({
                "text_index": text_idx,
                "text_preview": text[:100],
                "attributes": attributes,
                "matrix_entropy": float(-np.trace(text_rho @ np.log(text_rho + 1e-10))),
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Attribute extraction failed for text {text_idx}: {e}")
            results.append({
                "text_index": text_idx,
                "text_preview": text[:100],
                "attributes": {},
                "error": str(e),
                "success": False
            })
    
    return {
        "results": results,
        "total_texts": len(req.texts),
        "successful_extractions": sum(1 for r in results if r.get("success", False)),
        "attributes_extracted": len(target_attributes),
        "attribute_names": target_attributes
    }

# --- Global Shared Rho System ---
GLOBAL_RHO_ID = "global_consciousness"

def ensure_global_rho():
    """Ensure global shared rho exists"""
    if GLOBAL_RHO_ID not in STATE:
        STATE[GLOBAL_RHO_ID] = {
            "rho": np.eye(DIM, dtype=complex) / DIM,
            "detail_matrix": np.eye(DIM, dtype=complex) / DIM,
            "narrative_matrix": np.eye(DIM, dtype=complex) / DIM,
            "thematic_matrix": np.eye(DIM, dtype=complex) / DIM,
            "log": [],
            "meta": {
                "label": "Global Literary Consciousness",
                "books_processed": 0,
                "total_chunks": 0,
                "total_tokens": 0,
                "book_titles": [],
                "created": time.time(),
                "last_updated": time.time()
            }
        }
        logger.info("Created global shared rho consciousness")

def merge_book_into_global_rho(book_rho_id: str):
    """Merge a completed book rho into the global consciousness"""
    ensure_global_rho()
    
    if book_rho_id not in STATE:
        logger.warning(f"Book rho {book_rho_id} not found for merging")
        return
    
    book_state = STATE[book_rho_id]
    global_state = STATE[GLOBAL_RHO_ID]
    
    # Merge the density matrices (weighted average)
    alpha = 0.3  # Integration strength
    global_state["rho"] = (1 - alpha) * global_state["rho"] + alpha * book_state["rho"]
    global_state["rho"] = psd_project(global_state["rho"])
    
    # Update metadata
    meta = global_state["meta"]
    meta["books_processed"] += 1
    meta["total_chunks"] += book_state["meta"].get("chunks_processed", 0)
    meta["total_tokens"] += book_state["meta"].get("chunks_processed", 0) * 400  # Approx tokens
    meta["last_updated"] = time.time()
    
    # Add book title from job info or rho_id
    book_meta = book_state.get("meta", {})
    if "book_title" in book_meta:
        book_title = book_meta["book_title"]
    else:
        # Extract from completed jobs
        matching_job = None
        for job in COMPLETED_JOBS:
            if hasattr(job, 'gutenberg_id') and f"batch_book_{job.gutenberg_id}" in book_rho_id:
                book_title = job.book_title
                break
        else:
            book_title = f"Book_{book_rho_id}"
    
    if book_title not in meta["book_titles"]:
        meta["book_titles"].append(book_title)
    
    logger.info(f"Merged {book_title} into global consciousness - now contains {meta['books_processed']} books")

@app.get("/rho/global/status")
def get_global_rho_status():
    """Get comprehensive status of the global shared rho"""
    ensure_global_rho()
    global_state = STATE[GLOBAL_RHO_ID]
    rho = global_state["rho"]
    meta = global_state["meta"]
    
    # Compute current diagnostics
    diag = diagnostics(rho)
    
    return {
        "rho_id": GLOBAL_RHO_ID,
        "meta": meta,
        "matrix_state": diag,
        "processing_queue": {
            "queued": len([j for j in PROCESSING_QUEUE if j.status == JobStatus.QUEUED]),
            "processing": len([j for j in PROCESSING_QUEUE if j.status == JobStatus.PROCESSING]),
            "completed": len(COMPLETED_JOBS)
        },
        "available_books": len([k for k in STATE.keys() if k.startswith("batch_book_")])
    }

@app.post("/rho/global/merge_book")
def merge_single_book_into_global(req: dict):
    """Merge a specific book into global consciousness"""
    book_rho_id = req.get("book_rho_id")
    if not book_rho_id:
        raise HTTPException(status_code=400, detail="book_rho_id required")
    
    if book_rho_id not in STATE:
        raise HTTPException(status_code=404, detail="Book matrix not found")
    
    book_meta = STATE[book_rho_id].get("meta", {})
    if book_meta.get("type") != "book":
        raise HTTPException(status_code=400, detail="Not a book matrix")
    
    ensure_global_rho()
    
    try:
        merge_book_into_global_rho(book_rho_id)
        book_title = book_meta.get("title", book_rho_id)
        return {
            "message": f"Successfully merged '{book_title}' into global consciousness",
            "book_rho_id": book_rho_id,
            "global_status": get_global_rho_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge book: {str(e)}")

@app.post("/rho/global/merge_all")
def merge_all_books_into_global():
    """Merge all existing book rhos into global consciousness"""
    ensure_global_rho()
    # Include both batch_book_ and book_ patterns
    book_rhos = [k for k in STATE.keys() 
                 if k.startswith("batch_book_") or (k.startswith("book_") and STATE[k].get("meta", {}).get("type") == "book")]
    
    for book_rho_id in book_rhos:
        merge_book_into_global_rho(book_rho_id)
    
    return {
        "message": f"Merged {len(book_rhos)} books into global consciousness",
        "global_status": get_global_rho_status()
    }

# --- Matrix listing endpoint ---
@app.get("/matrices/available")
def list_available_matrices():
    """List all available matrices with metadata for tab selection"""
    matrices = []
    
    for rho_id, state in STATE.items():
        meta = state.get("meta", {})
        rho = state["rho"]
        
        # Determine matrix type and display info
        if rho_id == GLOBAL_RHO_ID:
            display_name = "🧠 Global Literary Consciousness"
            description = f"Merged consciousness from {meta.get('books_processed', 0)} books"
            matrix_type = "global"
        elif meta.get("type") == "book":
            title = meta.get("title", "Unknown Book")
            author = meta.get("author", "Unknown Author") 
            chunks_processed = meta.get("chunks_processed", 0)
            total_chunks = meta.get("total_chunks", 0)
            completion = f"({chunks_processed}/{total_chunks} chunks)"
            display_name = f"📖 {title}"
            description = f"by {author} {completion}"
            matrix_type = "book"
        else:
            # Legacy/other matrices
            display_name = f"🔬 {meta.get('label', rho_id)}"
            description = "Legacy matrix"
            matrix_type = "legacy"
        
        # Get basic matrix stats
        try:
            purity = float(np.trace(rho @ rho).real)
            entropy = -float(np.trace(rho @ np.log(rho + 1e-12)).real)
            
            # Handle NaN/inf values
            if not np.isfinite(purity):
                purity = 0.0
            if not np.isfinite(entropy):
                entropy = 0.0
        except Exception as e:
            purity = 0.0
            entropy = 0.0
            
        matrices.append({
            "rho_id": rho_id,
            "display_name": display_name,
            "description": description,
            "type": matrix_type,
            "purity": purity,
            "entropy": entropy,
            "created": meta.get("created", time.time()),
            "meta": meta
        })
    
    # Sort: global first, then books by most recent, then legacy
    def sort_key(matrix):
        if matrix["type"] == "global":
            return (0, 0)
        elif matrix["type"] == "book":
            return (1, -matrix["created"])  # Most recent books first
        else:
            return (2, -matrix["created"])  # Legacy last
    
    matrices.sort(key=sort_key)
    
    return {
        "matrices": matrices,
        "total_count": len(matrices)
    }

# --- Debug endpoint ---
@app.get("/debug/rho_list")
def debug_rho_list():
    """Debug endpoint to list all available rho_ids"""
    return {"rho_ids": list(STATE.keys())}

# --- Health endpoint ---
@app.get("/healthz")
def healthz():
    return {"ok": True, "dim": DIM, "packs": len(PACKS), "rhos": len(STATE)}


# --- Rho Space Design and Transformation APIs ---

# Global transformation engine (initialized lazily)
_transformation_engine = None

def get_transformation_engine():
    """Get or create the global transformation engine"""
    global _transformation_engine
    if _transformation_engine is None and RHO_DESIGN_AVAILABLE:
        _transformation_engine = RhoTransformationEngine(
            embed_function=embed,
            project_to_rho_function=project_to_local,
            rho_to_text_function=None  # Will be enhanced below
        )
        
        # Enhance with sophisticated text generation
        try:
            narrative_design = get_narrative_space_design()
            enhance_transformation_engine_with_generation(_transformation_engine, narrative_design)
            logger.info("Enhanced transformation engine with rho-conditioned text generation")
        except Exception as e:
            logger.warning(f"Failed to enhance text generation, using fallback: {e}")
        
        # Create default narrative library
        try:
            _transformation_engine.create_narrative_library()
            logger.info("Created default narrative transformation library")
        except Exception as e:
            logger.error(f"Failed to create narrative library: {e}")
    return _transformation_engine

class TransformationRequest(BaseModel):
    text: str
    transformation_name: str
    strength: float = 1.0
    library_name: str = "narrative_transformations"

class LearnTransformationRequest(BaseModel):
    name: str
    description: str
    source_description: str
    target_description: str
    library_name: str = "custom_transformations"

@app.post("/transformations/apply")
def apply_transformation(req: TransformationRequest):
    """Apply a learned transformation to text"""
    if not RHO_DESIGN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Rho design modules not available")
    
    try:
        engine = get_transformation_engine()
        if not engine:
            raise HTTPException(status_code=500, detail="Transformation engine not available")
        
        result = engine.apply_transformation(
            text=req.text,
            transformation_name=req.transformation_name,
            strength=req.strength,
            library_name=req.library_name
        )
        
        return {
            "original_text": req.text,
            "transformed_text": result,
            "transformation": req.transformation_name,
            "strength": req.strength,
            "library": req.library_name
        }
        
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

@app.post("/transformations/learn")
def learn_transformation_from_description(req: LearnTransformationRequest):
    """Learn a new transformation from textual descriptions"""
    if not RHO_DESIGN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Rho design modules not available")
    
    try:
        engine = get_transformation_engine()
        if not engine:
            raise HTTPException(status_code=500, detail="Transformation engine not available")
        
        transformation = engine.learn_transformation_from_description(
            name=req.name,
            description=req.description,
            source_description=req.source_description,
            target_description=req.target_description,
            library_name=req.library_name
        )
        
        return {
            "name": transformation.name,
            "description": transformation.description,
            "library": req.library_name,
            "direction_norm": float(np.linalg.norm(transformation.direction)),
            "status": "learned"
        }
        
    except Exception as e:
        logger.error(f"Learning transformation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")

@app.get("/transformations/list")
def list_transformations(library_name: str = "narrative_transformations"):
    """List available transformations in a library"""
    if not RHO_DESIGN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Rho design modules not available")
    
    try:
        engine = get_transformation_engine()
        if not engine:
            raise HTTPException(status_code=500, detail="Transformation engine not available")
        
        if library_name not in engine.libraries:
            return {"library": library_name, "transformations": [], "message": "Library not found"}
        
        library = engine.libraries[library_name]
        transformations = []
        
        for name, transformation in library.transformations.items():
            transformations.append({
                "name": transformation.name,
                "description": transformation.description,
                "strength": transformation.strength,
                "examples_count": len(transformation.examples),
                "direction_norm": float(np.linalg.norm(transformation.direction))
            })
        
        return {
            "library": library_name,
            "transformations": transformations,
            "total_count": len(transformations)
        }
        
    except Exception as e:
        logger.error(f"Listing transformations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")

@app.get("/transformations/libraries")
def list_transformation_libraries():
    """List all available transformation libraries"""
    if not RHO_DESIGN_AVAILABLE:
        return {"libraries": [], "message": "Rho design modules not available"}
    
    try:
        engine = get_transformation_engine()
        if not engine:
            return {"libraries": [], "message": "Transformation engine not available"}
        
        libraries = []
        for name, library in engine.libraries.items():
            libraries.append({
                "name": library.name,
                "rho_space": library.rho_space_name,
                "transformation_count": len(library.transformations)
            })
        
        return {"libraries": libraries}
        
    except Exception as e:
        logger.error(f"Listing libraries failed: {e}")
        return {"libraries": [], "error": str(e)}

# === COMPREHENSIVE ATTRIBUTE MANAGEMENT ENDPOINTS ===

class AttributeSearchRequest(BaseModel):
    query: str = ""
    category: Optional[str] = None  # namespace, persona, style
    tags: Optional[List[str]] = None
    limit: int = 50

class AttributeFavoriteRequest(BaseModel):
    attribute_name: str

class CustomAttributeRequest(BaseModel):
    name: str
    category: str
    subcategory: str
    description: str
    positive_examples: List[str]
    negative_examples: List[str]
    tags: Optional[List[str]] = None
    strength_range: Optional[Tuple[float, float]] = (-2.0, 2.0)

@app.get("/attributes/library/full")
def get_full_attribute_library():
    """Get the complete attribute library with all categories and collections"""
    if not RHO_DESIGN_AVAILABLE:
        return {"error": "Attribute library not available"}
    
    try:
        library = get_attribute_library()
        return library.export_library()
    except Exception as e:
        logger.error(f"Failed to get attribute library: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attributes/collections")
def get_attribute_collections():
    """Get all attribute collections for browsing"""
    if not RHO_DESIGN_AVAILABLE:
        return {"collections": []}
    
    try:
        library = get_attribute_library()
        collections = {}
        
        for name, collection in library.collections.items():
            collections[name] = {
                "name": collection.name,
                "description": collection.description,
                "tags": collection.tags,
                "attribute_count": len(collection.attributes),
                "attributes": [
                    {
                        "name": attr.name,
                        "description": attr.description,
                        "category": attr.category,
                        "subcategory": attr.subcategory,
                        "tags": attr.tags,
                        "is_favorite": attr.name in library.favorites
                    } for attr in collection.attributes
                ]
            }
        
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attributes/category/{category}")
def get_attributes_by_category(category: str):
    """Get all attributes in a specific category"""
    if not RHO_DESIGN_AVAILABLE:
        return {"attributes": []}
    
    if category not in ["namespace", "persona", "style"]:
        raise HTTPException(status_code=400, detail="Category must be namespace, persona, or style")
    
    try:
        library = get_attribute_library()
        attributes = library.get_attributes_by_category(category)
        
        result = []
        for attr in attributes:
            result.append({
                "name": attr.name,
                "description": attr.description,
                "category": attr.category,
                "subcategory": attr.subcategory,
                "tags": attr.tags,
                "positive_examples": attr.positive_examples[:3],  # Limit examples
                "is_favorite": attr.name in library.favorites
            })
        
        return {
            "category": category,
            "attributes": result,
            "total_count": len(result)
        }
    except Exception as e:
        logger.error(f"Failed to get category {category}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attributes/search")
def search_attributes(request: AttributeSearchRequest):
    """Search attributes by query, category, or tags"""
    if not RHO_DESIGN_AVAILABLE:
        return {"attributes": []}
    
    try:
        library = get_attribute_library()
        attributes = library.search_attributes(
            query=request.query,
            category=request.category,
            tags=request.tags
        )
        
        # Limit results
        attributes = attributes[:request.limit]
        
        result = []
        for attr in attributes:
            result.append({
                "name": attr.name,
                "description": attr.description,
                "category": attr.category,
                "subcategory": attr.subcategory,
                "tags": attr.tags,
                "positive_examples": attr.positive_examples[:2],
                "is_favorite": attr.name in library.favorites
            })
        
        return {
            "query": request.query,
            "category": request.category,
            "tags": request.tags,
            "attributes": result,
            "total_count": len(result)
        }
    except Exception as e:
        logger.error(f"Attribute search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attributes/favorites")
def get_favorite_attributes():
    """Get user's favorite attributes"""
    if not RHO_DESIGN_AVAILABLE:
        return {"favorites": []}
    
    try:
        library = get_attribute_library()
        favorites = library.get_favorites()
        
        result = []
        for attr in favorites:
            result.append({
                "name": attr.name,
                "description": attr.description,
                "category": attr.category,
                "subcategory": attr.subcategory,
                "tags": attr.tags,
                "positive_examples": attr.positive_examples[:2]
            })
        
        return {
            "favorites": result,
            "total_count": len(result)
        }
    except Exception as e:
        logger.error(f"Failed to get favorites: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attributes/favorites/add")
def add_to_favorites(request: AttributeFavoriteRequest):
    """Add an attribute to favorites"""
    if not RHO_DESIGN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Attribute library not available")
    
    try:
        library = get_attribute_library()
        success = library.add_to_favorites(request.attribute_name)
        
        if success:
            return {"message": f"Added {request.attribute_name} to favorites"}
        else:
            raise HTTPException(status_code=404, detail="Attribute not found")
    except Exception as e:
        logger.error(f"Failed to add favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attributes/favorites/remove")
def remove_from_favorites(request: AttributeFavoriteRequest):
    """Remove an attribute from favorites"""
    if not RHO_DESIGN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Attribute library not available")
    
    try:
        library = get_attribute_library()
        success = library.remove_from_favorites(request.attribute_name)
        
        if success:
            return {"message": f"Removed {request.attribute_name} from favorites"}
        else:
            return {"message": f"{request.attribute_name} was not in favorites"}
    except Exception as e:
        logger.error(f"Failed to remove favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attributes/create")
def create_custom_attribute(request: CustomAttributeRequest):
    """Create a custom user attribute"""
    if not RHO_DESIGN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Attribute library not available")
    
    try:
        library = get_attribute_library()
        
        attr_def = AttributeDefinition(
            name=request.name,
            category=request.category,
            subcategory=request.subcategory,
            description=request.description,
            positive_examples=request.positive_examples,
            negative_examples=request.negative_examples,
            tags=request.tags or [],
            strength_range=request.strength_range
        )
        
        success = library.create_custom_attribute(attr_def)
        
        if success:
            return {"message": f"Created custom attribute: {request.name}"}
        else:
            raise HTTPException(status_code=409, detail="Attribute name already exists")
    except Exception as e:
        logger.error(f"Failed to create custom attribute: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attributes/suggestions/{category}")
def get_attribute_suggestions(category: str, subcategory: Optional[str] = None):
    """Get curated attribute suggestions for a category"""
    if not RHO_DESIGN_AVAILABLE:
        return {"suggestions": []}
    
    if category not in ["namespace", "persona", "style"]:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    # Curated suggestions for each category
    suggestions = {
        "namespace": {
            "popular": ["high_fantasy", "cyberpunk_dystopian", "ancient_classical", "space_exploration"],
            "beginner_friendly": ["medieval_feudal", "urban_fantasy", "renaissance_humanist"],
            "advanced": ["post_scarcity", "enlightenment_rational", "industrial_mechanical"]
        },
        "persona": {
            "popular": ["wise_sage", "curious_explorer", "passionate_advocate", "playful_trickster"],
            "beginner_friendly": ["compassionate_healer", "pragmatic_realist", "serene_contemplative"],
            "advanced": ["rebellious_maverick", "analytical_critic", "intuitive_synthesizer"]
        },
        "style": {
            "popular": ["conversational_casual", "poetic_lyrical", "urgent_imperative"],
            "beginner_friendly": ["eloquent_formal", "minimalist_sparse", "rhythmic_flowing"],
            "advanced": ["ornate_elaborate", "systematic_methodical", "intuitive_associative"]
        }
    }
    
    try:
        library = get_attribute_library()
        category_suggestions = suggestions.get(category, {})
        
        result = {}
        for suggestion_type, attr_names in category_suggestions.items():
            attrs = []
            for name in attr_names:
                attr = library.get_attribute(name)
                if attr:
                    attrs.append({
                        "name": attr.name,
                        "description": attr.description,
                        "subcategory": attr.subcategory,
                        "is_favorite": name in library.favorites
                    })
            result[suggestion_type] = attrs
        
        return {"category": category, "suggestions": result}
    except Exception as e:
        logger.error(f"Failed to get suggestions for {category}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# If running directly via `python main.py`, start uvicorn (useful for local dev).
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Rho demo API on port %d", PORT)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
