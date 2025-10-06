#!/usr/bin/env python3
"""
Example usage of the semantic cache for AI prompts
"""

from semantic_cache import SemanticCache
from config import load_config, update_config
import time

def basic_usage_example():
    """Basic usage example of the semantic cache."""
    print("=== Basic Usage Example ===")
    
    # Initialize cache with default settings
    cache = SemanticCache()
    
    # Store some prompt-response pairs
    prompts_and_responses = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Tell me about artificial intelligence", "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans."),
        ("How does machine learning work?", "Machine learning is a method of data analysis that automates analytical model building. It uses algorithms that iteratively learn from data to find hidden insights."),
        ("What is Python programming?", "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, and AI."),
    ]
    
    print("Storing prompt-response pairs...")
    for prompt, response in prompts_and_responses:
        cache.put(prompt, response)
        print(f"✓ Cached: {prompt[:50]}...")
    
    print("\nTesting semantic similarity...")
    
    # Test similar queries
    test_queries = [
        "What's the main city in France?",  # Similar to capital question
        "Explain AI to me",  # Similar to AI question
        "How do ML algorithms function?",  # Similar to ML question
        "Tell me about the Python language",  # Similar to Python question
        "What is the weather like?",  # Not similar to any cached prompts
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = cache.get(query)
        if response:
            print(f"Cache HIT: {response}")
        else:
            print("Cache MISS: No similar prompt found")
    
    # Show cache statistics
    print(f"\nCache Statistics: {cache.stats()}")

def advanced_usage_example():
    """Advanced usage with custom configuration and metadata."""
    print("\n=== Advanced Usage Example ===")
    
    # Load configuration
    config = load_config()
    print(f"Loaded config: {config}")
    
    # Create cache with custom settings
    cache = SemanticCache(
        model_name='all-MiniLM-L6-v2',
        cache_dir='./advanced_cache',
        similarity_threshold=0.8,  # Lower threshold for more matches
        max_cache_size=100
    )
    
    # Store prompts with metadata
    prompts_with_metadata = [
        {
            "prompt": "How do I create a REST API in Python?",
            "response": "You can create a REST API in Python using frameworks like Flask or FastAPI. Here's a basic example with Flask...",
            "metadata": {
                "topic": "programming",
                "language": "python",
                "difficulty": "intermediate",
                "tags": ["api", "rest", "web", "python"]
            }
        },
        {
            "prompt": "What are the best practices for database design?",
            "response": "Database design best practices include: 1) Normalize your data, 2) Use appropriate data types, 3) Create proper indexes, 4) Design for scalability...",
            "metadata": {
                "topic": "database",
                "difficulty": "advanced",
                "tags": ["database", "design", "sql", "performance"]
            }
        }
    ]
    
    print("Storing prompts with metadata...")
    for item in prompts_with_metadata:
        cache.put(
            item["prompt"], 
            item["response"], 
            metadata=item["metadata"]
        )
        print(f"✓ Cached with metadata: {item['prompt'][:50]}...")
    
    # Search for similar prompts
    print("\nSearching for similar prompts...")
    search_queries = [
        "Python API development",
        "Database optimization techniques",
        "Web development with Python"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: {query}")
        results = cache.search(query, top_k=3)
        
        if results:
            for i, (prompt, response, similarity) in enumerate(results, 1):
                print(f"  {i}. Similarity: {similarity:.3f}")
                print(f"     Prompt: {prompt[:60]}...")
                print(f"     Response: {response[:60]}...")
        else:
            print("  No similar prompts found")

def configuration_example():
    """Example of configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # Load current configuration
    config = load_config()
    print(f"Current configuration: {config}")
    
    # Update configuration
    print("\nUpdating configuration...")
    update_config(
        similarity_threshold=0.9,
        max_cache_size=500,
        enable_logging=True
    )
    
    # Load updated configuration
    updated_config = load_config()
    print(f"Updated configuration: {updated_config}")

def performance_example():
    """Example showing performance characteristics."""
    print("\n=== Performance Example ===")
    
    cache = SemanticCache(similarity_threshold=0.7)
    
    # Measure cache performance
    import random
    
    # Generate some test data
    test_prompts = [
        f"Explain concept {i} in detail" for i in range(50)
    ]
    
    test_responses = [
        f"This is a detailed explanation of concept {i} with various aspects and examples." 
        for i in range(50)
    ]
    
    print("Measuring cache performance...")
    
    # Time the caching process
    start_time = time.time()
    for prompt, response in zip(test_prompts, test_responses):
        cache.put(prompt, response)
    cache_time = time.time() - start_time
    
    print(f"Time to cache {len(test_prompts)} entries: {cache_time:.3f} seconds")
    
    # Time the retrieval process
    start_time = time.time()
    hits = 0
    for i in range(20):
        # Try to retrieve with slight variations
        query = f"Tell me about concept {i}"
        result = cache.get(query)
        if result:
            hits += 1
    retrieval_time = time.time() - start_time
    
    print(f"Time to retrieve 20 queries: {retrieval_time:.3f} seconds")
    print(f"Cache hits: {hits}/20")
    print(f"Hit rate: {hits/20*100:.1f}%")
    
    # Show final statistics
    stats = cache.stats()
    print(f"Final cache statistics: {stats}")

def main():
    """Run all examples."""
    print("Semantic Cache Examples")
    print("=" * 50)
    
    try:
        basic_usage_example()
        advanced_usage_example()
        configuration_example()
        performance_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
