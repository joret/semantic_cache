#!/usr/bin/env python3
"""
Simple example of using Cached LLM
"""

import sys
import os
import time
import keyboard

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cached_llm import CachedLLM

def wait_for_spacebar(prompt_num, total_prompts):
    """Wait for user to press spacebar to continue."""
    print(f"\n⏸️  Press SPACEBAR to continue to prompt {prompt_num}/{total_prompts} (or 'q' to quit)")
    while True:
        try:
            key = keyboard.read_key()
            if key == 'space':
                print("▶️  Continuing...")
                break
            elif key == 'q':
                print("👋 Quitting...")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n👋 Quitting...")
            sys.exit(0)

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
        "Tell me about the weather",
        "What is the meaning of life?",
        "How do I change a tire?",
        "What is photosynthesis?",
        "Tell me about dinosaurs",
        "How do I play guitar?",
        "What is the stock market?"
    ]
    
    print("\nTesting semantic caching with similar prompts...")
    print("-" * 50)
    print("📋 Instructions:")
    print("   - Press SPACEBAR to continue to the next prompt")
    print("   - Press 'q' to quit at any time")
    print("   - Watch for cache hits vs misses!")
    print("-" * 50)
    
    total_prompts = len(prompts)
    
    for i, prompt in enumerate(prompts, 1):
        # Wait for user input before each prompt (except the first one)
        if i > 1:
            wait_for_spacebar(i, total_prompts)
        
        # Determine expected cache behavior
        if i <= 10:  # AI/ML group
            expected = "HIT"
        elif i <= 16:  # Python group
            expected = "HIT"
        else:  # Different topics
            expected = "MISS"
        
        print(f"\n{'='*60}")
        print(f"PROMPT {i}/{total_prompts}")
        print(f"{'='*60}")
        print(f"Query: {prompt}")
        print(f"Expected: {expected}")
        print(f"{'='*60}")
        
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
