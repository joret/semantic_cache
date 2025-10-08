#!/usr/bin/env python3
"""
Enhanced Semantic Cache with LLM Fallback for Web Integration

This module extends the base cached LLM with real-time partial search capabilities
for web applications, including as-you-type suggestions and WebSocket support.
"""

import sys
import os
import time
import asyncio
import json
from typing import Optional, Dict, Any, List, Tuple
import logging
from dataclasses import dataclass

# Add parent directory to path to import semantic_cache
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_cache import SemanticCache
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)

@dataclass
class SearchSuggestion:
    """Represents a search suggestion for partial input."""
    prompt: str
    response: str
    similarity: float
    action: str = "use_cached"
    metadata: Optional[Dict[str, Any]] = None

class SearchCache:
    """Cache for search results to improve performance."""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, query: str) -> Optional[List[SearchSuggestion]]:
        """Get cached search results if still valid."""
        if query in self.cache:
            results, timestamp = self.cache[query]
            if time.time() - timestamp < self.ttl:
                return results
            else:
                del self.cache[query]
        return None
    
    def set(self, query: str, results: List[SearchSuggestion]):
        """Cache search results with timestamp."""
        self.cache[query] = (results, time.time())
    
    def clear(self):
        """Clear all cached search results."""
        self.cache.clear()

class CachedLLMWeb:
    """
    Enhanced semantic cache with real-time partial search for web applications.
    """
    
    def __init__(self,
                 model_name: str = "llama3:latest",
                 cache_model: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = './llm_cache_web',
                 similarity_threshold: float = 0.85,
                 partial_search_threshold: float = 0.6,
                 max_cache_size: int = 1000,
                 ollama_url: str = "http://localhost:11434",
                 system_prompt: Optional[str] = None,
                 llm_options: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced cached LLM for web integration.
        
        Args:
            model_name: Ollama model name to use
            cache_model: Sentence transformer model for embeddings
            cache_dir: Directory for cache storage
            similarity_threshold: Minimum similarity for cache hits
            partial_search_threshold: Minimum similarity for partial search suggestions
            max_cache_size: Maximum cache entries
            ollama_url: Ollama server URL
            system_prompt: Optional system prompt for the LLM
            llm_options: Additional options for LLM generation
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm_options = llm_options or {}
        self.partial_search_threshold = partial_search_threshold
        
        # Initialize semantic cache
        self.cache = SemanticCache(
            model_name=cache_model,
            cache_dir=cache_dir,
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size
        )
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(base_url=ollama_url)
        
        # Initialize search cache
        self.search_cache = SearchCache()
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'partial_searches': 0,
            'total_queries': 0,
            'total_llm_time': 0.0,
            'total_cache_time': 0.0,
            'total_search_time': 0.0
        }
        
        logger.info(f"Initialized CachedLLMWeb with model: {model_name}")
    
    def is_llm_available(self) -> bool:
        """Check if the LLM is available."""
        return self.ollama_client.is_available()
    
    def query(self, prompt: str, use_cache: bool = True) -> str:
        """
        Query the system with semantic caching and LLM fallback.
        Same as base implementation but with enhanced statistics.
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
        logger.info("Cache miss - querying LLM")
        
        if not self.is_llm_available():
            raise Exception("LLM is not available. Make sure Ollama is running.")
        
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
            
            logger.info(f"LLM response generated in {llm_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise Exception(f"Failed to generate response: {e}")
    
    def search_partial(self, partial_prompt: str, top_k: int = 5) -> List[SearchSuggestion]:
        """
        Search for similar prompts using partial input.
        
        Args:
            partial_prompt: Partial input text to search for
            top_k: Maximum number of suggestions to return
            
        Returns:
            List of search suggestions
        """
        start_time = time.time()
        
        # Check search cache first
        cached_results = self.search_cache.get(partial_prompt)
        if cached_results:
            logger.info("Partial search cache hit")
            return cached_results
        
        # Check minimum word count
        words = partial_prompt.strip().split()
        if len(words) < 3:
            return []
        
        self.stats['partial_searches'] += 1
        
        # Generate embedding for partial prompt
        partial_embedding = self.cache._generate_embedding(partial_prompt)
        
        # Find similar prompts
        suggestions = []
        for i, cached_embedding in enumerate(self.cache.embeddings):
            similarity = self.cache._calculate_similarity(partial_embedding, cached_embedding)
            if similarity > self.partial_search_threshold:
                prompt_hash = self.cache.prompt_hashes[i]
                entry = self.cache.cache[prompt_hash]
                
                suggestion = SearchSuggestion(
                    prompt=entry.prompt,
                    response=entry.response,
                    similarity=float(similarity),  # Convert numpy float32 to Python float
                    action="use_cached",
                    metadata=entry.metadata
                )
                suggestions.append(suggestion)
        
        # Sort by similarity and limit results
        suggestions.sort(key=lambda x: x.similarity, reverse=True)
        suggestions = suggestions[:top_k]
        
        # Cache the results
        self.search_cache.set(partial_prompt, suggestions)
        
        search_time = time.time() - start_time
        self.stats['total_search_time'] += search_time
        
        logger.info(f"Partial search completed in {search_time:.3f}s, found {len(suggestions)} suggestions")
        return suggestions
    
    def should_search_partial(self, text: str) -> bool:
        """
        Determine if partial search should be triggered based on input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if partial search should be performed
        """
        words = text.strip().split()
        
        # Minimum word count
        if len(words) < 3:
            return False
        
        # Check for natural breaks (sentence endings)
        if text.strip().endswith(('.', '?', '!', ':')):
            return True
        
        # Check for common question patterns
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should']
        if any(word.lower() in question_words for word in words[:3]):
            return True
        
        # Check for minimum length (5+ words or 30+ characters)
        return len(words) >= 3 or len(text.strip()) >= 30
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics including partial search metrics."""
        cache_stats = self.cache.stats()
        
        # Calculate hit rates
        total_queries = self.stats['total_queries']
        cache_hit_rate = 0.0
        if total_queries > 0:
            cache_hit_rate = self.stats['cache_hits'] / total_queries
        
        # Calculate average times
        avg_llm_time = 0.0
        if self.stats['llm_calls'] > 0:
            avg_llm_time = self.stats['total_llm_time'] / self.stats['llm_calls']
        
        avg_cache_time = 0.0
        if self.stats['cache_hits'] > 0:
            avg_cache_time = self.stats['total_cache_time'] / self.stats['cache_hits']
        
        avg_search_time = 0.0
        if self.stats['partial_searches'] > 0:
            avg_search_time = self.stats['total_search_time'] / self.stats['partial_searches']
        
        return {
            'llm_model': self.model_name,
            'llm_available': self.is_llm_available(),
            'total_queries': total_queries,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'llm_calls': self.stats['llm_calls'],
            'partial_searches': self.stats['partial_searches'],
            'cache_hit_rate': cache_hit_rate,
            'avg_llm_time': avg_llm_time,
            'avg_cache_time': avg_cache_time,
            'avg_search_time': avg_search_time,
            'total_llm_time': self.stats['total_llm_time'],
            'total_cache_time': self.stats['total_cache_time'],
            'total_search_time': self.stats['total_search_time'],
            'search_cache_size': len(self.search_cache.cache),
            'cache_stats': cache_stats
        }
    
    def clear_cache(self):
        """Clear both the semantic cache and search cache."""
        self.cache.clear()
        self.search_cache.clear()
        logger.info("All caches cleared")
    
    def search_cache_full(self, query: str, top_k: int = 5):
        """Search the full cache for similar prompts (same as base implementation)."""
        return self.cache.search(query, top_k)
    
    def list_available_models(self) -> List[str]:
        """List available Ollama models."""
        return self.ollama_client.list_models()
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        return self.ollama_client.pull_model(model_name)

def main():
    """Example usage of CachedLLMWeb."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Cached LLM for Web Integration')
    parser.add_argument('--model', default='llama3:latest', help='Ollama model name')
    parser.add_argument('--prompt', help='Input prompt')
    parser.add_argument('--partial', help='Partial prompt for search suggestions')
    parser.add_argument('--top-k', type=int, default=5, help='Number of suggestions')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize cached LLM
    cached_llm = CachedLLMWeb(model_name=args.model)
    
    if not cached_llm.is_llm_available():
        print("❌ LLM is not available. Make sure Ollama is running.")
        return
    
    print("✅ LLM is available")
    
    if args.partial:
        print(f"🔍 Searching for partial prompt: '{args.partial}'")
        suggestions = cached_llm.search_partial(args.partial, args.top_k)
        
        if suggestions:
            print(f"Found {len(suggestions)} suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"\n{i}. Similarity: {suggestion.similarity:.3f}")
                print(f"   Prompt: {suggestion.prompt}")
                print(f"   Response: {suggestion.response[:100]}...")
        else:
            print("No suggestions found")
    
    elif args.prompt:
        print(f"🤖 Querying: '{args.prompt}'")
        response = cached_llm.query(args.prompt)
        print(f"Response: {response}")
    
    if args.stats:
        stats = cached_llm.get_stats()
        print("\n📊 Statistics:")
        for key, value in stats.items():
            if key != 'cache_stats':
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
