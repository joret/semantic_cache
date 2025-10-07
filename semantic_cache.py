#!/usr/bin/env python3
"""
Semantic Cache for AI Prompts

A semantic cache that stores AI prompts and their responses, allowing for
intelligent retrieval based on semantic similarity rather than exact matches.
"""

import json
import hashlib
import pickle
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a single cache entry with prompt, response, and metadata."""
    prompt: str
    response: str
    timestamp: float
    prompt_hash: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class SemanticCache:
    """
    A semantic cache for AI prompts that uses embeddings to find similar prompts.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = './cache',
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 1000):
        """
        Initialize the semantic cache.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory to store cache files
            similarity_threshold: Minimum similarity score for cache hits
            max_cache_size: Maximum number of entries to keep in cache
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        self.embeddings: List[np.ndarray] = []
        self.prompt_hashes: List[str] = []
        
        # Load existing cache
        self._load_cache()
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for the given text."""
        return self.embedding_model.encode(text)
    
    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate a hash for the prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def _find_most_similar(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the most similar cached prompt to the query.
        
        Returns:
            Tuple of (prompt_hash, similarity_score) or (None, 0.0) if no match found
        """
        if not self.embeddings:
            return None, 0.0
        
        similarities = []
        for cached_embedding in self.embeddings:
            similarity = self._calculate_similarity(query_embedding, cached_embedding)
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        if max_similarity >= self.similarity_threshold:
            max_index = similarities.index(max_similarity)
            return self.prompt_hashes[max_index], max_similarity
        
        return None, max_similarity
    def get(self, prompt: str) -> Optional[str]:
        """
        Retrieve a cached response for a semantically similar prompt.
        
        Args:
            prompt: The input prompt to search for
            
        Returns:
            Cached response if similar prompt found, None otherwise
        """
        query_embedding = self._generate_embedding(prompt)
        prompt_hash, similarity = self._find_most_similar(query_embedding)
        
        if prompt_hash and prompt_hash in self.cache:
            logger.info(f"Cache hit with similarity: {similarity:.3f}")
            return self.cache[prompt_hash].response
        
        logger.info(f"Cache miss. Best similarity: {similarity:.3f}")
        return None
    
    def put(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a prompt-response pair in the cache.
        
        Args:
            prompt: The input prompt
            response: The AI response
            metadata: Optional metadata to store with the entry
        """
        prompt_hash = self._generate_prompt_hash(prompt)
        embedding = self._generate_embedding(prompt)
        
        # Create cache entry
        entry = CacheEntry(
            prompt=prompt,
            response=response,
            timestamp=time.time(),
            prompt_hash=prompt_hash,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Check if we need to remove old entries
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        # Store in cache
        self.cache[prompt_hash] = entry
        self.embeddings.append(embedding)
        self.prompt_hashes.append(prompt_hash)
        
        logger.info(f"Cached new entry: {prompt_hash}")
        
        # Save to disk
        self._save_cache()
    
    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_hash = min(self.cache.keys(), key=lambda h: self.cache[h].timestamp)
        
        # Remove from all data structures
        del self.cache[oldest_hash]
        
        # Find and remove from embeddings and hashes
        try:
            index = self.prompt_hashes.index(oldest_hash)
            del self.embeddings[index]
            del self.prompt_hashes[index]
        except ValueError:
            pass
        
        logger.info(f"Evicted oldest entry: {oldest_hash}")
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_dir / 'semantic_cache.pkl'
        
        # Convert numpy arrays to lists for JSON serialization
        cache_data = {}
        for hash_key, entry in self.cache.items():
            entry_dict = asdict(entry)
            if entry_dict['embedding'] is not None:
                entry_dict['embedding'] = entry_dict['embedding'].tolist()
            cache_data[hash_key] = entry_dict
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cache saved to {cache_file}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / 'semantic_cache.pkl'
        
        if not cache_file.exists():
            logger.info("No existing cache found")
            return
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Reconstruct cache entries
            for hash_key, entry_dict in cache_data.items():
                if entry_dict['embedding'] is not None:
                    entry_dict['embedding'] = np.array(entry_dict['embedding'])
                
                entry = CacheEntry(**entry_dict)
                self.cache[hash_key] = entry
                self.embeddings.append(entry.embedding)
                self.prompt_hashes.append(hash_key)
            
            logger.info(f"Loaded {len(self.cache)} entries from cache")
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache.clear()
            self.embeddings.clear()
            self.prompt_hashes.clear()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.embeddings.clear()
        self.prompt_hashes.clear()
        
        # Remove cache file
        cache_file = self.cache_dir / 'semantic_cache.pkl'
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_cache_size,
            'similarity_threshold': self.similarity_threshold,
            'model_name': self.model_name,
            'cache_dir': str(self.cache_dir)
        }
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search for similar prompts in the cache.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (prompt, response, similarity_score)
        """
        query_embedding = self._generate_embedding(query)
        similarities = []
        
        for i, cached_embedding in enumerate(self.embeddings):
            similarity = self._calculate_similarity(query_embedding, cached_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for i, (idx, similarity) in enumerate(similarities[:top_k]):
            prompt_hash = self.prompt_hashes[idx]
            entry = self.cache[prompt_hash]
            results.append((entry.prompt, entry.response, similarity))
        
        return results

def main():
    """Command line interface for the semantic cache."""
    parser = argparse.ArgumentParser(description='Semantic Cache for AI Prompts')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    parser.add_argument('--cache-dir', default='./cache', help='Cache directory')
    parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold')
    parser.add_argument('--max-size', type=int, default=1000, help='Maximum cache size')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get cached response for a prompt')
    get_parser.add_argument('prompt', help='The prompt to search for')
    
    # Put command
    put_parser = subparsers.add_parser('put', help='Store a prompt-response pair')
    put_parser.add_argument('prompt', help='The input prompt')
    put_parser.add_argument('response', help='The AI response')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar prompts')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the cache')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache
    cache = SemanticCache(
        model_name=args.model,
        cache_dir=args.cache_dir,
        similarity_threshold=args.threshold,
        max_cache_size=args.max_size
    )
    
    if args.command == 'get':
        response = cache.get(args.prompt)
        if response:
            print("Cached response found:")
            print(response)
        else:
            print("No cached response found")
    
    elif args.command == 'put':
        cache.put(args.prompt, args.response)
        print("Entry cached successfully")
    
    elif args.command == 'search':
        results = cache.search(args.query, args.top_k)
        if results:
            print(f"Found {len(results)} similar prompts:")
            for i, (prompt, response, similarity) in enumerate(results, 1):
                print(f"\n{i}. Similarity: {similarity:.3f}")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Response: {response[:100]}...")
        else:
            print("No similar prompts found")
    
    elif args.command == 'stats':
        stats = cache.stats()
        print("Cache Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == 'clear':
        cache.clear()
        print("Cache cleared")

if __name__ == '__main__':
    main()
