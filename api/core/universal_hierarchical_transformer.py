"""
Universal Hierarchical Transformer

LLM-agnostic hierarchical chunking system that works with any LLM provider.
Uses configuration-driven approach for maximum flexibility.
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Tuple, Optional
import requests
import numpy as np

from .hierarchical_chunking import (
    HierarchicalChunker, 
    ChunkTree, 
    TextChunk,
    create_hierarchical_transformation_plan
)

logger = logging.getLogger(__name__)

class LLMConfig:
    """Configuration for different LLM providers."""
    
    def __init__(self, config_dict: Dict):
        self.provider = config_dict.get('provider', 'groq')
        self.model = config_dict.get('model', 'openai/gpt-oss-20b')
        self.api_key = config_dict.get('api_key')
        self.api_url = config_dict.get('api_url', 'https://api.groq.com/openai/v1/chat/completions')
        self.headers = config_dict.get('headers', {})
        self.default_params = config_dict.get('default_params', {})
        self.thinking_tags = config_dict.get('thinking_tags', ['<think>', '</think>'])
        self.max_tokens = config_dict.get('max_tokens', 2000)
        self.timeout = config_dict.get('timeout', 60)
        self.stop_sequences = config_dict.get('stop_sequences', [])
        
    @classmethod
    def from_env(cls, provider: str = None) -> 'LLMConfig':
        """Create config from environment variables."""
        provider = provider or os.getenv('DEFAULT_LLM_PROVIDER', 'groq')
        
        if provider == 'groq':
            return cls({
                'provider': 'groq',
                'model': os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'),
                'api_key': os.getenv('GROQ_API_KEY'),
                'api_url': 'https://api.groq.com/openai/v1/chat/completions',
                'thinking_tags': ['<think>', '</think>'],
                'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', '2000')),
                'timeout': int(os.getenv('GROQ_TIMEOUT', '60')),
                'stop_sequences': ["\\n\\n---", "\\n\\n*", "EXPLANATION:", "ANALYSIS:"]
            })
        elif provider == 'openai':
            return cls({
                'provider': 'openai',
                'model': os.getenv('OPENAI_MODEL', 'gpt-4'),
                'api_key': os.getenv('OPENAI_API_KEY'),
                'api_url': 'https://api.openai.com/v1/chat/completions',
                'thinking_tags': [],  # OpenAI models don't use thinking tags
                'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
                'timeout': int(os.getenv('OPENAI_TIMEOUT', '60')),
                'stop_sequences': []
            })
        elif provider == 'anthropic':
            return cls({
                'provider': 'anthropic',
                'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'api_url': 'https://api.anthropic.com/v1/messages',
                'thinking_tags': ['<thinking>', '</thinking>'],
                'max_tokens': int(os.getenv('ANTHROPIC_MAX_TOKENS', '2000')),
                'timeout': int(os.getenv('ANTHROPIC_TIMEOUT', '60')),
                'stop_sequences': []
            })
        elif provider == 'ollama':
            return cls({
                'provider': 'ollama',
                'model': os.getenv('OLLAMA_MODEL', 'gpt-oss:20b'),
                'api_key': None,  # Ollama doesn't require API key
                'api_url': os.getenv('OLLAMA_API_URL', 'http://localhost:11434/v1/chat/completions'),
                'thinking_tags': ['<think>', '</think>'],  # gpt-oss models use thinking tags
                'max_tokens': int(os.getenv('OLLAMA_MAX_TOKENS', '2000')),
                'timeout': int(os.getenv('OLLAMA_TIMEOUT', '120')),  # Ollama can be slower
                'stop_sequences': ["\\n\\n---", "\\n\\n*", "EXPLANATION:", "ANALYSIS:"]
            })
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class UniversalHierarchicalTransformer:
    """LLM-agnostic hierarchical transformer."""
    
    def __init__(self, 
                 llm_config: LLMConfig = None,
                 max_concurrent_requests: int = 3,
                 chunker_config: Dict = None):
        self.llm_config = llm_config or LLMConfig.from_env()
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize chunker with config
        chunker_params = chunker_config or {}
        self.chunker = HierarchicalChunker(
            base_chunk_size=chunker_params.get('base_chunk_size', 300),
            min_overlap=chunker_params.get('min_overlap', 60),
            max_overlap=chunker_params.get('max_overlap', 100),
            summary_ratio=chunker_params.get('summary_ratio', 0.4)
        )
    
    def create_prompt(self, 
                     chunk: TextChunk, 
                     transformation_request: Dict,
                     context: Dict) -> str:
        """Create LLM prompt based on chunk level and context."""
        
        if chunk.level == 0:
            return self._create_base_chunk_prompt(chunk, transformation_request, context)
        else:
            return self._create_summary_chunk_prompt(chunk, transformation_request, context)
    
    def _create_base_chunk_prompt(self, 
                                 chunk: TextChunk, 
                                 transformation_request: Dict,
                                 context: Dict) -> str:
        """Create prompt for base-level chunks."""
        transformation_type = transformation_request.get('transformation_name', 'enhance_narrative')
        strength = transformation_request.get('strength', 0.7)
        
        # Context awareness
        position_context = ""
        if context.get('structure_info', {}).get('chunk_index', 0) == 0:
            position_context = "This is the opening section. "
        elif context.get('next_chunk') is None:
            position_context = "This is the concluding section. "
        else:
            position_context = "This is a middle section. "
        
        overlap_context = ""
        if context.get('structure_info', {}).get('has_overlap_before'):
            overlap_context += "The beginning may overlap with previous content. "
        if context.get('structure_info', {}).get('has_overlap_after'):
            overlap_context += "The ending may overlap with following content. "
        
        prompt = f"""Transform this text segment using the "{transformation_type}" style.

CONTEXT: {position_context}{overlap_context}This is part {context.get('structure_info', {}).get('chunk_index', 0) + 1} of {context.get('structure_info', {}).get('total_chunks_at_level', 1)} segments.

REQUIREMENTS:
- Apply {transformation_type} transformation with {strength:.1f} strength
- Maintain semantic coherence and flow
- Preserve key information and concepts
- Keep natural transitions for segment boundaries
- Output ONLY the transformed text, no explanations

TEXT SEGMENT:
{chunk.text}

TRANSFORMED VERSION:"""
        
        return prompt
    
    def _create_summary_chunk_prompt(self, 
                                   chunk: TextChunk, 
                                   transformation_request: Dict,
                                   context: Dict) -> str:
        """Create prompt for summary-level chunks."""
        transformation_type = transformation_request.get('transformation_name', 'enhance_narrative')
        
        parent_count = len(chunk.parent_chunks) if chunk.parent_chunks else 0
        
        prompt = f"""Synthesize and refine this multi-segment text using the "{transformation_type}" style.

CONTEXT: This combines {parent_count} previously transformed segments into a coherent whole.

REQUIREMENTS:
- Unify the segments into flowing, coherent text
- Smooth transitions between sections
- Maintain the {transformation_type} style throughout
- Preserve all key information and concepts
- Remove redundancy from overlapping segments
- Output ONLY the unified text, no explanations

SEGMENTS TO UNIFY:
{chunk.text}

UNIFIED VERSION:"""
        
        return prompt
    
    def _prepare_llm_request(self, prompt: str, chunk: TextChunk) -> Dict:
        """Prepare LLM request based on provider configuration."""
        base_request = {
            "model": self.llm_config.model,
            "temperature": 0.3,
            "max_tokens": min(self.llm_config.max_tokens, chunk.token_count * 2),
            "top_p": 0.9
        }
        
        if self.llm_config.stop_sequences:
            base_request["stop"] = self.llm_config.stop_sequences
        
        # Provider-specific formatting
        if self.llm_config.provider in ['groq', 'openai', 'ollama']:
            base_request["messages"] = [
                {
                    "role": "system",
                    "content": "You are a precise text transformation engine. Output only the transformed text with no additional commentary, explanations, or formatting."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        elif self.llm_config.provider == 'anthropic':
            base_request = {
                "model": self.llm_config.model,
                "max_tokens": min(self.llm_config.max_tokens, chunk.token_count * 2),
                "messages": [
                    {
                        "role": "user",
                        "content": f"Human: {prompt}\n\nAssistant: I'll transform this text as requested:"
                    }
                ]
            }
        
        # Merge with default params
        base_request.update(self.llm_config.default_params)
        return base_request
    
    def _prepare_headers(self) -> Dict:
        """Prepare headers for LLM request."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.llm_config.provider in ['groq', 'openai']:
            headers["Authorization"] = f"Bearer {self.llm_config.api_key}"
        elif self.llm_config.provider == 'anthropic':
            headers["x-api-key"] = self.llm_config.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.llm_config.provider == 'ollama':
            # Ollama doesn't require authentication headers
            pass
        
        # Merge with config headers
        headers.update(self.llm_config.headers)
        return headers
    
    def _extract_response_content(self, response_data: Dict) -> str:
        """Extract content from LLM response based on provider."""
        if self.llm_config.provider in ['groq', 'openai', 'ollama']:
            return response_data["choices"][0]["message"]["content"].strip()
        elif self.llm_config.provider == 'anthropic':
            return response_data["content"][0]["text"].strip()
        else:
            raise ValueError(f"Unknown provider: {self.llm_config.provider}")
    
    def _extract_thinking_content(self, raw_content: str) -> Tuple[str, str]:
        """Extract thinking content if provider uses thinking tags."""
        if not self.llm_config.thinking_tags:
            return raw_content, ""
        
        import re
        start_tag, end_tag = self.llm_config.thinking_tags
        pattern = f'{re.escape(start_tag)}(.*?){re.escape(end_tag)}'
        
        think_matches = re.findall(pattern, raw_content, re.DOTALL)
        thinking_content = "\n".join(think_matches) if think_matches else ""
        
        # Remove thinking tags from output
        clean_content = re.sub(pattern, '', raw_content, flags=re.DOTALL).strip()
        
        # If clean_content is empty but we have thinking_content, 
        # the model may have put everything inside thinking tags
        if not clean_content and thinking_content:
            logger.warning(f"Model put all content inside thinking tags, using thinking content as output")
            # Return the thinking content as the actual content
            return thinking_content, ""
        
        return clean_content, thinking_content
    
    async def transform_chunk_with_llm(self, 
                                      chunk: TextChunk, 
                                      transformation_request: Dict,
                                      context: Dict) -> Tuple[str, Dict]:
        """Transform a single chunk using configured LLM."""
        start_time = time.time()
        
        if not self.llm_config.api_key and self.llm_config.provider != 'ollama':
            raise ValueError(f"{self.llm_config.provider.title()} API key required for hierarchical transformation")
        
        prompt = self.create_prompt(chunk, transformation_request, context)
        llm_request = self._prepare_llm_request(prompt, chunk)
        headers = self._prepare_headers()
        
        try:
            response = requests.post(
                self.llm_config.api_url,
                headers=headers,
                json=llm_request,
                timeout=self.llm_config.timeout
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                raw_content = self._extract_response_content(result)
                
                # Extract thinking content if applicable
                transformed_text, thinking_content = self._extract_thinking_content(raw_content)
                
                # For Ollama, also check for reasoning field
                if self.llm_config.provider == 'ollama' and 'reasoning' in result["choices"][0]["message"]:
                    ollama_reasoning = result["choices"][0]["message"]["reasoning"]
                    if ollama_reasoning:
                        thinking_content = ollama_reasoning if not thinking_content else f"{thinking_content}\n\nOllama Reasoning: {ollama_reasoning}"
                
                # Clean up the response
                if "TRANSFORMED VERSION:" in transformed_text:
                    transformed_text = transformed_text.split("TRANSFORMED VERSION:")[-1].strip()
                
                if "UNIFIED VERSION:" in transformed_text:
                    transformed_text = transformed_text.split("UNIFIED VERSION:")[-1].strip()
                
                # Remove common wrapper patterns
                transformed_text = transformed_text.strip('"\' \n\r\t')
                
                # Log thinking content for debugging
                if thinking_content:
                    logger.debug(f"Chunk {chunk.id} thinking: {thinking_content[:200]}...")
                
                # Debug logging for content extraction issues
                if len(transformed_text) == 0:
                    logger.warning(f"Chunk {chunk.id} resulted in empty content after extraction. Raw response: {raw_content[:200]}...")
                elif len(transformed_text) < 10:
                    logger.warning(f"Chunk {chunk.id} resulted in very short content ({len(transformed_text)} chars): '{transformed_text}'")
                
                audit_info = {
                    "chunk_id": chunk.id,
                    "chunk_level": chunk.level,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "duration": duration,
                    "success": True,
                    "original_tokens": chunk.token_count,
                    "transformed_length": len(transformed_text),
                    "prompt_type": "base_chunk" if chunk.level == 0 else "summary_chunk",
                    "thinking_content": thinking_content,
                    "has_thinking": bool(thinking_content),
                    "raw_response": raw_content[:500] + "..." if len(raw_content) > 500 else raw_content,
                    "content_extraction_issue": len(transformed_text) < 10
                }
                
                logger.info(f"✅ {self.llm_config.provider.title()} transformed chunk {chunk.id}: {chunk.token_count} tokens → {len(transformed_text)} chars ({duration:.2f}s)")
                
                return transformed_text, audit_info
            
            else:
                error_msg = f"{self.llm_config.provider.title()} API error: {response.status_code}"
                logger.error(f"❌ {error_msg} for chunk {chunk.id}")
                
                return None, {
                    "chunk_id": chunk.id,
                    "provider": self.llm_config.provider,
                    "duration": duration,
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{self.llm_config.provider.title()} request failed: {str(e)}"
            logger.error(f"❌ {error_msg} for chunk {chunk.id}")
            
            return None, {
                "chunk_id": chunk.id,
                "provider": self.llm_config.provider,
                "duration": duration,
                "success": False,
                "error": error_msg
            }
    
    async def process_chunk_level(self, 
                                 chunk_ids: List[str], 
                                 tree: ChunkTree,
                                 transformation_request: Dict,
                                 transformed_chunks: Dict[str, str]) -> Tuple[Dict[str, str], List[Dict]]:
        """Process all chunks at a given level concurrently."""
        level = tree.chunks[chunk_ids[0]].level
        logger.info(f"🔄 Processing level {level} with {len(chunk_ids)} chunks using {self.llm_config.provider}")
        
        results = {}
        audit_info = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_single_chunk(chunk_id: str):
            async with semaphore:
                chunk = tree.chunks[chunk_id]
                
                # For summary chunks, update text with transformed parent content
                if chunk.level > 0 and chunk.parent_chunks:
                    parent_texts = []
                    for parent_id in chunk.parent_chunks:
                        if parent_id in transformed_chunks:
                            parent_texts.append(transformed_chunks[parent_id])
                        else:
                            # Fallback to original text
                            parent_texts.append(tree.chunks[parent_id].text)
                    
                    # Update chunk text with transformed parent content
                    chunk.text = "\n\n".join(parent_texts)
                
                context = self.chunker.get_chunk_context(chunk_id, tree)
                transformed_text, chunk_audit = await self.transform_chunk_with_llm(
                    chunk, transformation_request, context
                )
                
                return chunk_id, transformed_text, chunk_audit
        
        # Process chunks concurrently
        tasks = [process_single_chunk(chunk_id) for chunk_id in chunk_ids]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error(f"Chunk processing failed: {result}")
                continue
            
            chunk_id, transformed_text, chunk_audit = result
            
            if transformed_text:
                results[chunk_id] = transformed_text
                transformed_chunks[chunk_id] = transformed_text
            
            audit_info.append(chunk_audit)
        
        logger.info(f"✅ Completed level {level}: {len(results)}/{len(chunk_ids)} chunks successful")
        return results, audit_info
    
    async def transform_hierarchically(self, 
                                     text: str, 
                                     transformation_request: Dict) -> Tuple[str, Dict]:
        """Perform hierarchical transformation using configured LLM."""
        start_time = time.time()
        
        # Create transformation plan
        plan = create_hierarchical_transformation_plan(text, transformation_request)
        tree = plan["tree"]
        processing_order = plan["processing_order"]
        
        logger.info(f"🚀 Starting hierarchical transformation with {self.llm_config.provider}: {plan['total_chunks']} chunks across {plan['levels']} levels")
        
        # Track all transformations and audit info
        transformed_chunks = {}
        all_audit_info = []
        
        # Process each level bottom-up
        for level_chunks in processing_order:
            level_results, level_audit = await self.process_chunk_level(
                level_chunks, tree, transformation_request, transformed_chunks
            )
            all_audit_info.extend(level_audit)
        
        # Merge final result
        final_text = self.chunker.merge_transformed_chunks(tree, transformed_chunks)
        
        total_duration = time.time() - start_time
        
        # Create comprehensive audit trail
        audit_trail = {
            "transformation_strategy": "universal_hierarchical",
            "llm_provider": self.llm_config.provider,
            "llm_model": self.llm_config.model,
            "chunk_tree_info": {
                "total_chunks": plan['total_chunks'],
                "levels": plan['levels'],
                "base_chunks": len(tree.levels[0]),
                "estimated_llm_calls": plan['estimated_groq_calls']  # rename this
            },
            "processing_summary": {
                "total_duration": total_duration,
                "successful_chunks": len([a for a in all_audit_info if a.get('success')]),
                "failed_chunks": len([a for a in all_audit_info if not a.get('success')]),
                "average_chunk_duration": sum(a.get('duration', 0) for a in all_audit_info) / len(all_audit_info) if all_audit_info else 0
            },
            "chunk_details": all_audit_info,
            "final_result": {
                "original_length": len(text),
                "transformed_length": len(final_text),
                "compression_ratio": len(final_text) / len(text) if text else 1.0
            }
        }
        
        logger.info(f"🎉 Hierarchical transformation complete with {self.llm_config.provider}: {len(text)} → {len(final_text)} chars in {total_duration:.2f}s")
        
        return final_text, audit_trail

# Async wrapper for integration with existing sync code
def transform_text_hierarchically(text: str, 
                                transformation_request: Dict, 
                                llm_config: LLMConfig = None) -> Tuple[str, Dict]:
    """Synchronous wrapper for hierarchical transformation."""
    transformer = UniversalHierarchicalTransformer(llm_config=llm_config)
    
    # Run in async context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            transformer.transform_hierarchically(text, transformation_request)
        )
        return result
    finally:
        loop.close()