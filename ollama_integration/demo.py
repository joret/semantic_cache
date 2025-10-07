#!/usr/bin/env python3
"""
Interactive demo for Cached LLM with Semantic Cache
"""

import sys
import os
import time
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cached_llm import CachedLLM

def interactive_demo():
    """Run interactive demo of cached LLM."""
    print("🤖 Cached LLM Demo")
    print("=" * 50)
    print("This demo shows semantic caching with local LLM fallback.")
    print("Type 'quit' to exit, 'help' for commands.")
    print()
    
    # Initialize cached LLM
    try:
        cached_llm = CachedLLM(
            model_name="llama3:latest",
            similarity_threshold=0.85,
            system_prompt="You are a helpful AI assistant. Provide concise, accurate answers."
        )
    except Exception as e:
        print(f"Error initializing cached LLM: {e}")
        return
    
    # Check if LLM is available
    if not cached_llm.is_llm_available():
        print("❌ Ollama is not available!")
        print("Please make sure Ollama is running on http://localhost:11434")
        print("You can start it with: ollama serve")
        return
    
    print("✅ Ollama is available")
    
    # Show available models
    models = cached_llm.list_available_models()
    if models:
        print(f"📋 Available models: {', '.join(models)}")
    else:
        print("⚠️  No models found. You may need to pull a model:")
        print("   ollama pull llama2")
    
    print()
    
    while True:
        try:
            user_input = input("Enter a prompt (or 'help'/'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'stats':
                show_stats(cached_llm)
                continue
            elif user_input.lower() == 'search':
                search_cache(cached_llm)
                continue
            elif user_input.lower() == 'clear':
                cached_llm.clear_cache()
                print("✅ Cache cleared")
                continue
            elif user_input.lower() == 'models':
                show_models(cached_llm)
                continue
            elif not user_input:
                continue
            
            # Query the cached LLM
            print(f"🔍 Querying: '{user_input}'")
            start_time = time.time()
            
            try:
                response = cached_llm.query(user_input)
                query_time = time.time() - start_time
                
                print(f"⏱️  Query time: {query_time:.2f} seconds")
                print(f"📝 Response: {response}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

def print_help():
    """Print help information."""
    print("""
Available commands:
  help     - Show this help message
  quit     - Exit the demo
  stats    - Show cache and LLM statistics
  search   - Search the cache for similar prompts
  clear    - Clear the cache
  models   - Show available Ollama models
  
You can also enter any prompt to test the cached LLM!
""")

def show_stats(cached_llm: CachedLLM):
    """Show system statistics."""
    stats = cached_llm.get_stats()
    
    print("📊 System Statistics")
    print("-" * 30)
    print(f"LLM Model: {stats['llm_model']}")
    print(f"LLM Available: {stats['llm_available']}")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Misses: {stats['cache_misses']}")
    print(f"LLM Calls: {stats['llm_calls']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Avg LLM Time: {stats['avg_llm_time']:.2f}s")
    print(f"Avg Cache Time: {stats['avg_cache_time']:.3f}s")
    print(f"Total LLM Time: {stats['total_llm_time']:.2f}s")
    print(f"Total Cache Time: {stats['total_cache_time']:.3f}s")
    
    cache_stats = stats['cache_stats']
    print(f"\nCache Details:")
    print(f"  Total Entries: {cache_stats['total_entries']}")
    print(f"  Max Size: {cache_stats['max_size']}")
    print(f"  Similarity Threshold: {cache_stats['similarity_threshold']}")
    print()

def search_cache(cached_llm: CachedLLM):
    """Search the cache for similar prompts."""
    query = input("Enter search query: ").strip()
    if not query:
        return
    
    print(f"🔍 Searching cache for: '{query}'")
    results = cached_llm.search_cache(query, top_k=5)
    
    if results:
        print(f"Found {len(results)} similar prompts:")
        for i, (prompt, response, similarity) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {similarity:.3f}")
            print(f"   Prompt: {prompt}")
            print(f"   Response: {response[:100]}...")
    else:
        print("No similar prompts found in cache.")
    print()

def show_models(cached_llm: CachedLLM):
    """Show available Ollama models."""
    models = cached_llm.list_available_models()
    
    if models:
        print("📋 Available Ollama Models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    else:
        print("No models found. You may need to pull a model:")
        print("  ollama pull llama2")
    print()

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_help()
        return
    
    interactive_demo()

if __name__ == "__main__":
    main()
