"""
Hierarchical Chunking System for Groq Optimization

Creates a tree structure of overlapping chunks:
1. Base chunks: 300 tokens with 60-100 token overlap
2. Summary levels: Progressively combine chunks until single summary
3. Maintains context and coherence across transformations
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    level: int  # 0 = base chunks, 1+ = summary levels
    parent_chunks: List[str] = None  # IDs of chunks this summarizes
    overlap_before: int = 0
    overlap_after: int = 0

@dataclass
class ChunkTree:
    """Tree structure representing hierarchical chunks."""
    chunks: Dict[str, TextChunk]
    levels: Dict[int, List[str]]  # level -> chunk_ids
    root_chunk_id: Optional[str] = None

class HierarchicalChunker:
    """Creates hierarchical chunk structures for efficient Groq processing."""
    
    def __init__(self, 
                 base_chunk_size: int = 300,
                 min_overlap: int = 60,
                 max_overlap: int = 100,
                 summary_ratio: float = 0.4):
        """
        Initialize chunker.
        
        Args:
            base_chunk_size: Target tokens per base chunk
            min_overlap: Minimum token overlap between chunks
            max_overlap: Maximum token overlap between chunks  
            summary_ratio: Ratio for summary compression (0.4 = 40% of original)
        """
        self.base_chunk_size = base_chunk_size
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.summary_ratio = summary_ratio
    
    def estimate_token_count(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 0.75 words)."""
        words = len(text.split())
        return int(words * 1.33)  # Conservative estimate
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """Find sentence boundary positions for natural chunk breaks."""
        boundaries = []
        
        # Find sentence endings
        sentence_endings = re.finditer(r'[.!?]+\s+', text)
        for match in sentence_endings:
            boundaries.append(match.end())
        
        # Add paragraph breaks
        paragraph_breaks = re.finditer(r'\n\s*\n', text)
        for match in paragraph_breaks:
            boundaries.append(match.end())
        
        # Sort and deduplicate
        boundaries = sorted(set(boundaries))
        return boundaries
    
    def create_base_chunks(self, text: str) -> List[TextChunk]:
        """Create base-level chunks with overlap."""
        chunks = []
        total_tokens = self.estimate_token_count(text)
        
        if total_tokens <= self.base_chunk_size:
            # Single chunk for short text
            chunks.append(TextChunk(
                id="chunk_0_0",
                text=text,
                start_pos=0,
                end_pos=len(text),
                token_count=total_tokens,
                level=0
            ))
            return chunks
        
        sentence_boundaries = self.find_sentence_boundaries(text)
        words = text.split()
        
        # Calculate chunk positions
        chunk_starts = []
        pos = 0
        chunk_id = 0
        
        while pos < len(words):
            chunk_starts.append(pos)
            
            # Find end position for this chunk
            target_end = pos + self.base_chunk_size
            
            if target_end >= len(words):
                # Last chunk
                break
            
            # Try to end at sentence boundary
            word_pos_in_text = len(' '.join(words[:target_end]))
            best_boundary = None
            
            for boundary in sentence_boundaries:
                if word_pos_in_text - 50 <= boundary <= word_pos_in_text + 50:
                    best_boundary = boundary
                    break
            
            if best_boundary:
                # Convert boundary back to word position
                text_before_boundary = text[:best_boundary]
                words_before_boundary = len(text_before_boundary.split())
                pos = max(pos + self.base_chunk_size - self.max_overlap, words_before_boundary)
            else:
                # No good boundary, use token-based split
                pos = target_end - self.min_overlap
            
            chunk_id += 1
        
        # Create chunks with calculated positions
        for i, start_word_pos in enumerate(chunk_starts):
            if i + 1 < len(chunk_starts):
                end_word_pos = chunk_starts[i + 1] + self.max_overlap
                end_word_pos = min(end_word_pos, len(words))
            else:
                end_word_pos = len(words)
            
            chunk_text = ' '.join(words[start_word_pos:end_word_pos])
            
            # Calculate overlaps
            overlap_before = 0
            overlap_after = 0
            
            if i > 0:
                prev_end = chunk_starts[i] + self.max_overlap if i < len(chunk_starts) - 1 else len(words)
                overlap_before = max(0, min(self.max_overlap, start_word_pos - chunk_starts[i-1]))
            
            if i + 1 < len(chunk_starts):
                overlap_after = max(0, min(self.max_overlap, end_word_pos - chunk_starts[i+1]))
            
            chunks.append(TextChunk(
                id=f"chunk_0_{i}",
                text=chunk_text,
                start_pos=start_word_pos,
                end_pos=end_word_pos,
                token_count=self.estimate_token_count(chunk_text),
                level=0,
                overlap_before=overlap_before,
                overlap_after=overlap_after
            ))
        
        logger.info(f"Created {len(chunks)} base chunks from {total_tokens} tokens")
        return chunks
    
    def create_summary_level(self, chunks: List[TextChunk], level: int) -> List[TextChunk]:
        """Create next level of summary chunks."""
        if len(chunks) <= 1:
            return chunks
        
        summary_chunks = []
        group_size = max(2, min(4, len(chunks) // 2))  # Group 2-4 chunks per summary
        
        for i in range(0, len(chunks), group_size):
            group = chunks[i:i + group_size]
            
            # Combine text from group
            combined_text = "\n\n".join(chunk.text for chunk in group)
            
            # Create summary chunk
            summary_chunk = TextChunk(
                id=f"chunk_{level}_{i // group_size}",
                text=combined_text,
                start_pos=group[0].start_pos,
                end_pos=group[-1].end_pos,
                token_count=self.estimate_token_count(combined_text),
                level=level,
                parent_chunks=[chunk.id for chunk in group]
            )
            
            summary_chunks.append(summary_chunk)
        
        logger.info(f"Created {len(summary_chunks)} level-{level} summary chunks from {len(chunks)} level-{level-1} chunks")
        return summary_chunks
    
    def build_chunk_tree(self, text: str) -> ChunkTree:
        """Build complete hierarchical chunk tree."""
        start_time = time.time()
        
        # Create base chunks
        base_chunks = self.create_base_chunks(text)
        
        # Build tree structure
        all_chunks = {chunk.id: chunk for chunk in base_chunks}
        levels = {0: [chunk.id for chunk in base_chunks]}
        
        # Create summary levels
        current_level_chunks = base_chunks
        level = 1
        
        while len(current_level_chunks) > 1:
            summary_chunks = self.create_summary_level(current_level_chunks, level)
            
            # Add to tree
            for chunk in summary_chunks:
                all_chunks[chunk.id] = chunk
            
            levels[level] = [chunk.id for chunk in summary_chunks]
            current_level_chunks = summary_chunks
            level += 1
        
        # Determine root chunk
        root_chunk_id = None
        if current_level_chunks:
            root_chunk_id = current_level_chunks[0].id
        
        tree = ChunkTree(
            chunks=all_chunks,
            levels=levels,
            root_chunk_id=root_chunk_id
        )
        
        duration = time.time() - start_time
        logger.info(f"Built chunk tree with {len(all_chunks)} total chunks across {len(levels)} levels in {duration:.2f}s")
        
        return tree
    
    def get_transformation_order(self, tree: ChunkTree) -> List[List[str]]:
        """Get optimal order for processing chunks (bottom-up)."""
        order = []
        
        # Process from base level up
        for level in sorted(tree.levels.keys()):
            order.append(tree.levels[level])
        
        return order
    
    def merge_transformed_chunks(self, 
                                tree: ChunkTree, 
                                transformed_chunks: Dict[str, str]) -> str:
        """Merge transformed chunks back into coherent text."""
        if not tree.root_chunk_id or tree.root_chunk_id not in transformed_chunks:
            # Fallback: concatenate all base level chunks
            base_level = min(tree.levels.keys())
            base_chunk_ids = tree.levels[base_level]
            
            result_parts = []
            for chunk_id in base_chunk_ids:
                if chunk_id in transformed_chunks:
                    result_parts.append(transformed_chunks[chunk_id])
            
            return "\n\n".join(result_parts)
        
        # Return the root (fully summarized) transformation
        return transformed_chunks[tree.root_chunk_id]
    
    def get_chunk_context(self, chunk_id: str, tree: ChunkTree) -> Dict:
        """Get context information for a chunk to inform transformation."""
        chunk = tree.chunks[chunk_id]
        
        context = {
            "chunk_id": chunk_id,
            "level": chunk.level,
            "position_info": {
                "start": chunk.start_pos,
                "end": chunk.end_pos,
                "token_count": chunk.token_count
            },
            "structure_info": {
                "total_chunks_at_level": len(tree.levels[chunk.level]),
                "chunk_index": tree.levels[chunk.level].index(chunk_id),
                "has_overlap_before": chunk.overlap_before > 0,
                "has_overlap_after": chunk.overlap_after > 0
            }
        }
        
        # Add parent information for summary chunks
        if chunk.parent_chunks:
            context["parent_chunks"] = chunk.parent_chunks
            context["summarizes_chunks"] = len(chunk.parent_chunks)
        
        # Add neighbor information
        level_chunks = tree.levels[chunk.level]
        chunk_index = level_chunks.index(chunk_id)
        
        if chunk_index > 0:
            context["previous_chunk"] = level_chunks[chunk_index - 1]
        if chunk_index < len(level_chunks) - 1:
            context["next_chunk"] = level_chunks[chunk_index + 1]
        
        return context

def create_hierarchical_transformation_plan(text: str, 
                                           transformation_request: Dict) -> Dict:
    """Create a transformation plan using hierarchical chunking."""
    chunker = HierarchicalChunker()
    tree = chunker.build_chunk_tree(text)
    
    processing_order = chunker.get_transformation_order(tree)
    
    plan = {
        "tree": tree,
        "processing_order": processing_order,
        "total_chunks": len(tree.chunks),
        "levels": len(tree.levels),
        "estimated_groq_calls": len(tree.chunks),
        "transformation_strategy": "hierarchical_bottom_up"
    }
    
    return plan