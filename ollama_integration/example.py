#!/usr/bin/env python3
"""
Simple example of using Cached LLM
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cached_llm import CachedLLM

def main():
    """Run a simple example."""
    print("🚀 Cached LLM Example")
    print("=" * 40)
    
    # Initialize cached LLM
    cached_llm = CachedLLM(
        model_name="llama3:latest",
        similarity_threshold=0.8,
        system_prompt="You are a helpful AI assistant. Provide concise, accurate answers."
    )
    
    # Check if LLM is available
    if not cached_llm.is_llm_available():
        print("❌ Ollama is not available!")
        print("Please make sure Ollama is running:")
        print("  ollama serve")
        print("  ollama pull llama2")
        return
    
    print("✅ Ollama is available")
    
    # Example prompts
    prompts = [
        "What is artificial intelligence?",
        "Tell me about AI",
        "Explain machine learning",
        "How does ML work?",
        "What is Python programming?",
        "Tell me about the Python language"
    ]
    
    print("\nTesting semantic caching with similar prompts...")
    print("-" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Query: {prompt}")
        
        start_time = time.time()
        try:
            response = cached_llm.query(prompt)
            query_time = time.time() - start_time
            
            print(f"   Time: {query_time:.2f}s")
            print(f"   Response: {response[:100]}...")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # Show final statistics
    print("\n" + "=" * 50)
    print("Final Statistics:")
    stats = cached_llm.get_stats()
    
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Misses: {stats['cache_misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"LLM Calls: {stats['llm_calls']}")
    print(f"Avg LLM Time: {stats['avg_llm_time']:.2f}s")
    print(f"Avg Cache Time: {stats['avg_cache_time']:.3f}s")
    
    # Show cache contents
    print(f"\nCache Contents ({stats['cache_stats']['total_entries']} entries):")
    search_results = cached_llm.search_cache("artificial intelligence", top_k=3)
    for i, (prompt, response, similarity) in enumerate(search_results, 1):
        print(f"  {i}. {similarity:.3f} - {prompt}")

if __name__ == "__main__":
    main()
