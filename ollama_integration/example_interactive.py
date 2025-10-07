#!/usr/bin/env python3
"""
Interactive example of using Cached LLM with step-by-step execution
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cached_llm import CachedLLM

def wait_for_input(prompt_num, total_prompts):
    """Wait for user to press Enter to continue."""
    print(f"\n⏸️  Press ENTER to continue to prompt {prompt_num}/{total_prompts} (or 'q' + ENTER to quit)")
    user_input = input().strip().lower()
    if user_input == 'q':
        print("👋 Quitting...")
        sys.exit(0)
    print("▶️  Continuing...")

def main():
    """Run a simple example."""
    print("🚀 Cached LLM Interactive Example")
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
        print("  ollama pull llama3:latest")
        return
    
    print("✅ Ollama is available")
    
    # Ask if user wants to clear cache
    print("\n🧹 Cache Management:")
    print("   The cache may contain previous responses that could affect this demo.")
    clear_cache = input("   Do you want to clear the cache before starting? (y/n): ").strip().lower()
    
    if clear_cache in ['y', 'yes']:
        cached_llm.clear_cache()
        print("   ✅ Cache cleared!")
    else:
        print("   ⏭️  Keeping existing cache entries")
    
    # Show current cache status
    current_stats = cached_llm.get_stats()
    print(f"   📊 Current cache entries: {current_stats['cache_stats']['total_entries']}")
    
    # Example prompts - designed to demonstrate cache hits and misses
    prompts = [
        # Group 1: AI/ML related (should hit cache)
        "What is artificial intelligence?",
        "Tell me about AI",
        "Explain artificial intelligence",
        "What is AI?",
        "How does artificial intelligence work?",
        "Can you explain AI to me?",
        "What is machine learning?",
        "Tell me about ML",
        "Explain machine learning",
        "How does ML work?",
        
        # Group 2: Python programming (should hit cache)
        "What is Python programming?",
        "Tell me about Python",
        "Explain Python programming",
        "What is the Python language?",
        "How do I learn Python?",
        "What is Python used for?",
        
        # Group 3: Different topics (should miss cache)
        "What is the capital of France?",
        "How do I cook pasta?",
        "What is quantum physics?",
        "Tell me about the weather"
    ]
    
    print("\nTesting semantic caching with similar prompts...")
    print("-" * 50)
    print("📋 Instructions:")
    print("   - Press ENTER to continue to the next prompt")
    print("   - Type 'q' + ENTER to quit at any time")
    print("   - Watch for cache hits vs misses!")
    print("   - Notice how similar prompts get cached responses!")
    print("-" * 50)
    
    total_prompts = len(prompts)
    
    for i, prompt in enumerate(prompts, 1):
        # Wait for user input before each prompt (except the first one)
        if i > 1:
            wait_for_input(i, total_prompts)
        
        # Determine expected cache behavior
        if i <= 10:  # AI/ML group
            expected = "HIT"
            group = "AI/ML"
        elif i <= 16:  # Python group
            expected = "HIT"
            group = "Python"
        else:  # Different topics
            expected = "MISS"
            group = "Other"
        
        print(f"\n{'='*60}")
        print(f"PROMPT {i}/{total_prompts} - {group} Group")
        print(f"{'='*60}")
        print(f"Query: {prompt}")
        print(f"Expected: {expected}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            response = cached_llm.query(prompt)
            query_time = time.time() - start_time
            
            print(f"⏱️  Time: {query_time:.2f}s")
            print(f"📝 Response: {response[:200]}...")
            
            # Show current stats after each query
            current_stats = cached_llm.get_stats()
            print(f"\n📊 Current Statistics:")
            print(f"   Hit Rate: {current_stats['hit_rate']:.1%}")
            print(f"   Total Queries: {current_stats['total_queries']}")
            print(f"   Cache Hits: {current_stats['cache_hits']}")
            print(f"   LLM Calls: {current_stats['llm_calls']}")
            print(f"   Avg LLM Time: {current_stats['avg_llm_time']:.2f}s")
            print(f"   Avg Cache Time: {current_stats['avg_cache_time']:.3f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print(f"{'='*60}")
    
    # Show final statistics
    print("\n" + "=" * 60)
    print("🎉 FINAL RESULTS")
    print("=" * 60)
    stats = cached_llm.get_stats()
    
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Misses: {stats['cache_misses']}")
    print(f"LLM Calls: {stats['llm_calls']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Avg LLM Time: {stats['avg_llm_time']:.2f}s")
    print(f"Avg Cache Time: {stats['avg_cache_time']:.3f}s")
    print(f"Total LLM Time: {stats['total_llm_time']:.2f}s")
    print(f"Total Cache Time: {stats['total_cache_time']:.3f}s")
    
    # Show cache contents
    print(f"\nCache Contents ({stats['cache_stats']['total_entries']} entries):")
    search_results = cached_llm.search_cache("artificial intelligence", top_k=3)
    for i, (prompt, response, similarity) in enumerate(search_results, 1):
        print(f"  {i}. {similarity:.3f} - {prompt}")
    
    # Analysis
    print(f"\n📊 Cache Analysis:")
    print(f"   Expected hits: 16 (AI/ML + Python groups)")
    print(f"   Expected misses: 4 (different topics)")
    print(f"   Actual hit rate: {stats['hit_rate']:.1%}")
    
    if stats['hit_rate'] > 0.7:
        print("   ✅ Excellent cache performance!")
    elif stats['hit_rate'] > 0.5:
        print("   ✅ Good cache performance!")
    else:
        print("   ⚠️  Lower than expected cache performance")
        print("   💡 Try lowering similarity_threshold for more hits")
    
    print(f"\n💡 Key Observations:")
    print(f"   - Similar prompts (AI/ML, Python) should hit the cache")
    print(f"   - Different topics should miss the cache")
    print(f"   - Cache responses are much faster than LLM calls")
    print(f"   - Semantic similarity finds related prompts automatically")

if __name__ == "__main__":
    main()
