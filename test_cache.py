#!/usr/bin/env python3
"""
Simple test script for the semantic cache
"""

from semantic_cache import SemanticCache
import time

def test_basic_functionality():
    """Test basic cache functionality."""
    print("Testing basic functionality...")
    
    # Create cache
    cache = SemanticCache(similarity_threshold=0.7)
    
    # Test data
    test_data = [
        ("What is Python?", "Python is a programming language."),
        ("How do I learn programming?", "Start with basics, practice regularly, build projects."),
        ("What is machine learning?", "ML is a subset of AI that learns from data."),
    ]
    
    # Store data
    for prompt, response in test_data:
        cache.put(prompt, response)
    
    # Test retrieval
    test_queries = [
        "Tell me about Python programming",
        "How can I become a programmer?",
        "Explain machine learning",
        "What is the weather like?",  # Should not match
    ]
    
    print("\nTesting cache retrieval:")
    for query in test_queries:
        result = cache.get(query)
        if result:
            print(f"✓ HIT: {query} -> {result}")
        else:
            print(f"✗ MISS: {query}")
    
    # Test statistics
    stats = cache.stats()
    print(f"\nCache stats: {stats}")
    
    print("Basic functionality test completed!")

def test_performance():
    """Test cache performance."""
    print("\nTesting performance...")
    
    cache = SemanticCache(similarity_threshold=0.8)
    
    # Generate test data
    num_entries = 100
    prompts = [f"Explain concept {i} in detail" for i in range(num_entries)]
    responses = [f"Detailed explanation of concept {i} with examples and use cases." for i in range(num_entries)]
    
    # Measure storage time
    start_time = time.time()
    for prompt, response in zip(prompts, responses):
        cache.put(prompt, response)
    storage_time = time.time() - start_time
    
    print(f"Stored {num_entries} entries in {storage_time:.3f} seconds")
    print(f"Average storage time: {storage_time/num_entries*1000:.2f} ms per entry")
    
    # Measure retrieval time
    start_time = time.time()
    hits = 0
    for i in range(20):
        query = f"Tell me about concept {i}"
        result = cache.get(query)
        if result:
            hits += 1
    retrieval_time = time.time() - start_time
    
    print(f"Retrieved 20 queries in {retrieval_time:.3f} seconds")
    print(f"Average retrieval time: {retrieval_time/20*1000:.2f} ms per query")
    print(f"Cache hits: {hits}/20 ({hits/20*100:.1f}%)")
    
    print("Performance test completed!")

def main():
    """Run all tests."""
    print("Semantic Cache Test Suite")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_performance()
        print("\n" + "=" * 40)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
