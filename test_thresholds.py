#!/usr/bin/env python3
"""
Test different similarity thresholds
"""

from semantic_cache import SemanticCache

def test_thresholds():
    """Test how different thresholds affect cache hits."""
    
    # Test data
    cache = SemanticCache()
    cache.put("What is artificial intelligence?", "AI is the simulation of human intelligence in machines.")
    
    # Test queries with different similarity levels
    test_queries = [
        "What is artificial intelligence?",  # Exact match
        "Tell me about AI",  # Very similar
        "Explain artificial intelligence",  # Similar
        "What is machine learning?",  # Related but different
        "How does weather work?",  # Completely different
    ]
    
    thresholds = [0.95, 0.85, 0.75, 0.65]
    
    print("Testing Different Similarity Thresholds")
    print("=" * 50)
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        print("-" * 20)
        
        # Create new cache with this threshold
        test_cache = SemanticCache(similarity_threshold=threshold)
        test_cache.put("What is artificial intelligence?", "AI is the simulation of human intelligence in machines.")
        
        for query in test_queries:
            result = test_cache.get(query)
            if result:
                print(f"✓ HIT:  '{query}'")
            else:
                print(f"✗ MISS: '{query}'")

if __name__ == "__main__":
    test_thresholds()
