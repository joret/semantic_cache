#!/usr/bin/env python3
"""
Semantic Cache with LLM Fallback

This module combines the semantic cache with Ollama LLM integration.
When there's a cache miss, it queries the local LLM and stores the response.
"""

import sys
import os
import time
from typing import Optional, Dict, Any
import logging

# Add parent directory to path to import semantic_cache
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_cache import SemanticCache
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class CachedLLM:
    """
    A semantic cache that falls back to local LLM when there's a cache miss.
    """
    
    def __init__(self,
                 model_name: str = "llama3:latest",
                 cache_model: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = './llm_cache',
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 1000,
                 ollama_url: str = "http://localhost:11434",
                 system_prompt: Optional[str] = None,
                 llm_options: Optional[Dict[str, Any]] = None):
        """
        Initialize the cached LLM system.
        
        Args:
            model_name: Ollama model name to use
            cache_model: Sentence transformer model for embeddings
            cache_dir: Directory for cache storage
            similarity_threshold: Minimum similarity for cache hits
            max_cache_size: Maximum cache entries
            ollama_url: Ollama server URL
            system_prompt: Optional system prompt for the LLM
            llm_options: Additional options for LLM generation
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm_options = llm_options or {}
        
        # Initialize semantic cache
        self.cache = SemanticCache(
            model_name=cache_model,
            cache_dir=cache_dir,
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size
        )
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(base_url=ollama_url)
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'total_queries': 0,
            'total_llm_time': 0.0,
            'total_cache_time': 0.0
        }
        
        logger.info(f"Initialized CachedLLM with model: {model_name}")
    
    def is_llm_available(self) -> bool:
        """Check if the LLM is available."""
        return self.ollama_client.is_available()
    
    def query(self, prompt: str, use_cache: bool = True) -> str:
        """
        Query the system with semantic caching and LLM fallback.
        
        Args:
            prompt: The input prompt
            use_cache: Whether to use cache (default: True)
            
        Returns:
            Response from cache or LLM
        """
        self.stats['total_queries'] += 1
        start_time = time.time()
        
        # Try cache first if enabled
        if use_cache:
            cached_response = self.cache.get(prompt)
            if cached_response:
                self.stats['cache_hits'] += 1
                self.stats['total_cache_time'] += time.time() - start_time
                logger.info("Cache hit - returning cached response")
                return cached_response
        
        # Cache miss - query LLM
        self.stats['cache_misses'] += 1
        # logger.info("Cache miss - querying LLM")
        
        # if not self.is_llm_available():
        #     raise Exception("LLM is not available. Make sure Ollama is running.")
        
        llm_start_time = time.time()
        try:
            response = self.ollama_client.generate(
                prompt=prompt,
                model=self.model_name,
                system=self.system_prompt,
                options=self.llm_options
            )
            
            llm_time = time.time() - llm_start_time
            self.stats['llm_calls'] += 1
            self.stats['total_llm_time'] += llm_time
            
            # Store in cache
            if use_cache:
                self.cache.put(
                    prompt, 
                    response, 
                    metadata={
                        'model': self.model_name,
                        'timestamp': time.time(),
                        'llm_time': llm_time,
                        'cache_miss': True
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise Exception(f"Failed to generate response: {e}")
    
    def query_stream(self, prompt: str, use_cache: bool = True):
        """
        Query the system with streaming response.
        
        Args:
            prompt: The input prompt
            use_cache: Whether to use cache (default: True)
            
        Yields:
            Chunks of response text
        """
        self.stats['total_queries'] += 1
        
        # Try cache first if enabled
        if use_cache:
            cached_response = self.cache.get(prompt)
            if cached_response:
                self.stats['cache_hits'] += 1
                logger.info("Cache hit - returning cached response")
                # Yield cached response as single chunk
                yield cached_response
                return
        
        # Cache miss - stream from LLM
        self.stats['cache_misses'] += 1
        logger.info("Cache miss - streaming from LLM")
        
        if not self.is_llm_available():
            raise Exception("LLM is not available. Make sure Ollama is running.")
        
        llm_start_time = time.time()
        full_response = ""
        
        try:
            for chunk in self.ollama_client.generate_stream(
                prompt=prompt,
                model=self.model_name,
                system=self.system_prompt,
                options=self.llm_options
            ):
                full_response += chunk
                yield chunk
            
            llm_time = time.time() - llm_start_time
            self.stats['llm_calls'] += 1
            self.stats['total_llm_time'] += llm_time
            
            # Store full response in cache
            if use_cache and full_response:
                self.cache.put(
                    prompt, 
                    full_response, 
                    metadata={
                        'model': self.model_name,
                        'timestamp': time.time(),
                        'llm_time': llm_time,
                        'cache_miss': True,
                        'streamed': True
                    }
                )
            
            logger.info(f"Streaming LLM response completed in {llm_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error streaming from LLM: {e}")
            raise Exception(f"Failed to stream response: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        cache_stats = self.cache.stats()
        
        # Calculate hit rate
        hit_rate = 0.0
        if self.stats['total_queries'] > 0:
            hit_rate = self.stats['cache_hits'] / self.stats['total_queries']
        
        # Calculate average times
        avg_llm_time = 0.0
        if self.stats['llm_calls'] > 0:
            avg_llm_time = self.stats['total_llm_time'] / self.stats['llm_calls']
        
        avg_cache_time = 0.0
        if self.stats['cache_hits'] > 0:
            avg_cache_time = self.stats['total_cache_time'] / self.stats['cache_hits']
        
        return {
            'llm_model': self.model_name,
            'llm_available': self.is_llm_available(),
            'total_queries': self.stats['total_queries'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'llm_calls': self.stats['llm_calls'],
            'hit_rate': hit_rate,
            'avg_llm_time': avg_llm_time,
            'avg_cache_time': avg_cache_time,
            'total_llm_time': self.stats['total_llm_time'],
            'total_cache_time': self.stats['total_cache_time'],
            'cache_stats': cache_stats
        }
    
    def clear_cache(self):
        """Clear the semantic cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def search_cache(self, query: str, top_k: int = 5):
        """Search the cache for similar prompts."""
        return self.cache.search(query, top_k)
    
    def list_available_models(self) -> list:
        """List available Ollama models."""
        return self.ollama_client.list_models()
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        return self.ollama_client.pull_model(model_name)

def main():
    """Example usage of CachedLLM."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cached LLM with Semantic Cache')
    parser.add_argument('--model', default='llama3:latest', help='Ollama model name')
    parser.add_argument('--prompt', required=True, help='Input prompt')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--stream', action='store_true', help='Use streaming')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize cached LLM
    cached_llm = CachedLLM(model_name=args.model)
    
    if not cached_llm.is_llm_available():
        print("❌ LLM is not available. Make sure Ollama is running.")
        return
    
    print(f"🤖 Querying model: {args.model}")
    print(f"📝 Prompt: {args.prompt}")
    print("-" * 50)
    
    try:
        if args.stream:
            print("Response (streaming):")
            for chunk in cached_llm.query_stream(args.prompt, use_cache=not args.no_cache):
                print(chunk, end='', flush=True)
            print("\n")
        else:
            response = cached_llm.query(args.prompt, use_cache=not args.no_cache)
            print(f"Response: {response}")
        
        if args.stats:
            stats = cached_llm.get_stats()
            print("\nStatistics:")
            for key, value in stats.items():
                if key != 'cache_stats':
                    print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
