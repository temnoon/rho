"""
Embedding and projection utilities for quantum state construction.

This module handles the bridge between high-dimensional semantic embeddings
and the local 64-dimensional quantum state space.
"""

import numpy as np
import hashlib
import os
import logging
from typing import Optional, Union
import requests

logger = logging.getLogger(__name__)

# Configuration
DIM = 64
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
W_MATRIX_PATH = os.path.join(DATA_DIR, "w.npy")
EMBED_URL = os.getenv("EMBED_URL", "")


def deterministic_embed_stub(text: str, out_dim: int) -> np.ndarray:
    """
    Deterministic pseudo-embedding for demo/testing: uses a hash to seed RNG.
    
    Args:
        text: Input text to embed
        out_dim: Output embedding dimension
        
    Returns:
        Deterministic embedding vector
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(text_hash[:4], "big")
    
    rng = np.random.RandomState(seed)
    raw = rng.randn(out_dim)
    
    # Add some structure based on text properties
    if len(text) > 50:
        raw[0] += 0.3  # Length signal
    if "?" in text:
        raw[1] += 0.4  # Question signal
    if "!" in text:
        raw[2] += 0.4  # Exclamation signal
    if any(word in text.lower() for word in ["love", "happy", "joy"]):
        raw[3] += 0.5  # Positive sentiment
    if any(word in text.lower() for word in ["sad", "angry", "dark"]):
        raw[4] += 0.5  # Negative sentiment
    
    return raw / (np.linalg.norm(raw) + 1e-10)


def embed(text: str) -> np.ndarray:
    """
    Real semantic embedding function with multiple backends.
    
    Priority order:
    1. External embedding service (if EMBED_URL is set)
    2. HuggingFace sentence-transformers (if available)
    3. Deterministic stub (fallback)
    
    Args:
        text: Text to embed
        
    Returns:
        Semantic embedding vector
    """
    if not text or not text.strip():
        return np.zeros(768)  # Default to 768D for better models
    
    text = text.strip()
    
    # Try Ollama nomic-embed-text first (local, fast, reliable)
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=15  # Local should be fast, but allow time for model loading
        )
        if response.status_code == 200:
            data = response.json()
            embedding = data["embedding"]
            logger.info(f"Successfully embedded with Ollama nomic-embed-text ({len(embedding)}D)")
            return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Ollama nomic-embed-text failed: {e}")
    
    # Try external embedding service
    if EMBED_URL:
        try:
            response = requests.post(
                EMBED_URL,
                json={"text": text},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    return np.array(data["embedding"], dtype=np.float32)
        except Exception as e:
            logger.warning(f"External embedding service failed: {e}")
    
    # Try OpenAI embeddings as fallback (if API key is available)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": "text-embedding-3-large",
                    "dimensions": 1536  # Optimal balance of quality and efficiency
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                logger.info(f"Successfully embedded with OpenAI text-embedding-3-large (1536D)")
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
    
    # Try HuggingFace sentence-transformers with better models
    try:
        from sentence_transformers import SentenceTransformer
        
        # Progressive model selection for better quality
        model_options = [
            ('all-mpnet-base-v2', 768),      # Best quality, 768D
            ('all-MiniLM-L12-v2', 384),     # Good balance
            ('all-MiniLM-L6-v2', 384),      # Lightweight fallback
        ]
        
        for model_name, dim in model_options:
            try:
                logger.info(f"Loading embedding model: {model_name} ({dim}D)")
                model = SentenceTransformer(model_name)
                embedding = model.encode([text])[0]
                logger.info(f"Successfully embedded text with {model_name}, output dim: {len(embedding)}")
                return embedding.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
                
    except ImportError:
        logger.warning("sentence-transformers not available, using deterministic stub")
    except Exception as e:
        logger.warning(f"All HuggingFace embedding models failed: {e}")
    
    # Fallback to deterministic stub with warning
    logger.warning("🚨 USING MOCK EMBEDDINGS - Install sentence-transformers for real semantic embeddings")
    return deterministic_embed_stub(text, 768).astype(np.float32)  # Use 768D for consistency


def learn_projection_matrix_from_samples() -> np.ndarray:
    """
    Learn a projection matrix W from a sample of texts using PCA.
    
    This creates a mapping from global embedding space to local 64D space
    that preserves the most important semantic variations.
    
    Returns:
        Projection matrix W of shape (64, embedding_dim)
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "To be or not to be, that is the question.",
        "I have a dream that one day this nation will rise up.",
        "Four score and seven years ago our fathers brought forth.",
        "We hold these truths to be self-evident.",
        "Space: the final frontier. These are the voyages.",
        "Elementary, my dear Watson.",
        "May the Force be with you.",
        "Houston, we have a problem.",
        "I'll be back.",
        "Show me the money!",
        "Life is like a box of chocolates.",
        "The truth is out there.",
        "Winter is coming.",
        "Knowledge is power.",
        "Time heals all wounds.",
        "Actions speak louder than words.",
        "The pen is mightier than the sword."
    ]
    
    # Generate embeddings for sample texts
    embeddings = []
    for text in sample_texts:
        emb = embed(text)
        embeddings.append(emb)
    
    X = np.array(embeddings)  # Shape: (n_samples, embedding_dim)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute PCA
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Take available components and pad to DIM if needed
    n_components = min(DIM, Vt.shape[0])
    W = np.zeros((DIM, Vt.shape[1]))
    W[:n_components, :] = Vt[:n_components, :]
    
    # If we have fewer components than DIM, fill remaining rows with random orthogonal vectors
    if n_components < DIM:
        # Generate random vectors for remaining dimensions
        np.random.seed(42)  # For reproducibility
        for i in range(n_components, DIM):
            # Create random vector
            random_vec = np.random.randn(Vt.shape[1])
            # Orthogonalize against existing rows
            for j in range(i):
                random_vec -= np.dot(random_vec, W[j, :]) * W[j, :]
            # Normalize
            norm = np.linalg.norm(random_vec)
            if norm > 1e-10:
                W[i, :] = random_vec / norm
            else:
                # Fallback to standard basis vector if normalization fails
                if i < Vt.shape[1]:
                    W[i, i] = 1.0
    
    logger.info(f"Learned projection matrix W: {W.shape}")
    return W


def load_w_matrix() -> np.ndarray:
    """
    Load projection W if present. Expected shape: (64, m) or (64,64).
    If not found, learn a new one and save it.
    
    Returns:
        Projection matrix W
    """
    if os.path.exists(W_MATRIX_PATH):
        try:
            W = np.load(W_MATRIX_PATH)
            if W.shape[0] == DIM:
                logger.info(f"Loaded projection matrix W: {W.shape}")
                return W
            else:
                logger.warning(f"Invalid W matrix shape: {W.shape}, expected ({DIM}, ?)")
        except Exception as e:
            logger.warning(f"Failed to load W matrix: {e}")
    
    # Learn new projection matrix
    W = learn_projection_matrix_from_samples()
    
    # Save it
    try:
        os.makedirs(os.path.dirname(W_MATRIX_PATH), exist_ok=True)
        np.save(W_MATRIX_PATH, W)
        logger.info(f"Saved new projection matrix to {W_MATRIX_PATH}")
    except Exception as e:
        logger.warning(f"Failed to save W matrix: {e}")
    
    return W


def project_to_local(x: np.ndarray) -> np.ndarray:
    """
    Project global embedding x (shape m,) to local DIM via W (DIM x m).
    
    Args:
        x: Global embedding vector
        
    Returns:
        Local 64-dimensional vector (unit normalized)
    """
    W = load_w_matrix()
    
    # Handle dimension mismatch
    if W.shape[1] != len(x):
        # Truncate or pad x to match W
        if len(x) > W.shape[1]:
            x = x[:W.shape[1]]
        else:
            x_padded = np.zeros(W.shape[1])
            x_padded[:len(x)] = x
            x = x_padded
    
    # Project to local space
    v = W @ x
    
    # Normalize to unit vector
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        v = v / norm
    else:
        # Fallback to random unit vector
        v = np.random.randn(DIM)
        v = v / np.linalg.norm(v)
    
    return v


def create_text_projection_matrix(text: str) -> np.ndarray:
    """
    Create a projection matrix from text using existing logic.
    
    Args:
        text: Source text
        
    Returns:
        Projection matrix for the text
    """
    # Embed the text
    x = embed(text)
    
    # Project to local space
    v = project_to_local(x)
    
    # Create rank-1 projection matrix
    return np.outer(v, v)


def text_to_embedding_vector(text: str) -> np.ndarray:
    """
    Convert text to a normalized embedding vector for channel operations.
    
    This function is specifically for quantum channel applications where
    we need the embedding vector itself, not a density matrix.
    
    Args:
        text: Input text
        
    Returns:
        Normalized embedding vector in local 64D space
    """
    # Embed and project to local space
    x = embed(text)
    v = project_to_local(x)
    
    # Normalize to unit vector for channel operations
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # Fallback: random unit vector if embedding is zero
        v = np.random.randn(DIM)
        norm = np.linalg.norm(v)
    
    return v / norm


def hierarchical_embed(text: str, max_chunks: int = 8) -> dict:
    """
    Create hierarchical embeddings for both detail and summary matching.
    
    This function:
    1. Splits long text into semantic chunks
    2. Creates embeddings for each chunk (detail level)
    3. Creates an embedding for the full text (summary level)
    4. Combines them into a hierarchical structure for better cosine search
    
    Args:
        text: Input text to embed hierarchically
        max_chunks: Maximum number of chunks for detail embedding
        
    Returns:
        Dictionary with hierarchical embedding data
    """
    if not text or not text.strip():
        return {
            "summary_embedding": np.zeros(768),
            "detail_embeddings": [],
            "chunk_texts": [],
            "combined_embedding": np.zeros(768),
            "text_length": 0,
            "num_chunks": 0
        }
    
    text = text.strip()
    text_length = len(text)
    
    # Split text into semantic chunks for detail embedding
    chunks = split_text_semantically(text, max_chunks)
    
    # Create detail embeddings for each chunk
    detail_embeddings = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embedding = embed(chunk)
            detail_embeddings.append(chunk_embedding)
    
    # Create summary embedding for the full text
    summary_embedding = embed(text)
    
    # Combine embeddings: weighted average of summary + details
    if detail_embeddings:
        detail_stack = np.array(detail_embeddings)
        detail_mean = np.mean(detail_stack, axis=0)
        
        # Weight summary more heavily for shorter texts, details more for longer texts
        summary_weight = max(0.3, 1.0 - (text_length / 5000))  # 0.3 to 1.0
        detail_weight = 1.0 - summary_weight
        
        combined_embedding = (summary_weight * summary_embedding + 
                            detail_weight * detail_mean)
        
        # Normalize the combined embedding
        combined_embedding = combined_embedding / (np.linalg.norm(combined_embedding) + 1e-10)
    else:
        combined_embedding = summary_embedding
    
    return {
        "summary_embedding": summary_embedding,
        "detail_embeddings": detail_embeddings,
        "chunk_texts": chunks,
        "combined_embedding": combined_embedding,
        "text_length": text_length,
        "num_chunks": len(chunks)
    }


def split_text_semantically(text: str, max_chunks: int = 8) -> list:
    """
    Split text into semantic chunks for hierarchical embedding.
    
    Uses multiple strategies:
    1. Paragraph boundaries
    2. Sentence boundaries  
    3. Length-based splitting as fallback
    
    Args:
        text: Text to split
        max_chunks: Maximum number of chunks to create
        
    Returns:
        List of text chunks
    """
    if len(text) < 200:  # Short text, don't split
        return [text]
    
    # Strategy 1: Split by paragraphs (double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= max_chunks and all(len(p) > 50 for p in paragraphs):
        return paragraphs[:max_chunks]
    
    # Strategy 2: Split by sentences
    import re
    sentence_ends = re.compile(r'[.!?]+\s+')
    sentences = [s.strip() for s in sentence_ends.split(text) if s.strip()]
    
    if len(sentences) <= max_chunks:
        return sentences[:max_chunks]
    
    # Strategy 3: Combine sentences into chunks of reasonable size
    chunks = []
    current_chunk = ""
    target_chunk_size = len(text) // max_chunks
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < target_chunk_size or not current_chunk:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            
            if len(chunks) >= max_chunks - 1:  # Save space for last chunk
                break
    
    # Add remaining text as last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks[:max_chunks]


def text_to_rho(text: str) -> np.ndarray:
    """
    Convert text directly to a density matrix using hierarchical embedding.
    
    Args:
        text: Input text
        
    Returns:
        Density matrix representing the text
    """
    from .quantum_state import create_pure_state
    
    # Use hierarchical embedding for better semantic representation
    hierarchical_data = hierarchical_embed(text)
    
    # Use combined embedding that includes both summary and detail information
    combined_embedding = hierarchical_data["combined_embedding"]
    v = project_to_local(combined_embedding)
    
    # Create pure state density matrix
    return create_pure_state(v)