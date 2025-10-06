#!/usr/bin/env python3
"""
Interactive demo of the semantic cache
"""

from semantic_cache import SemanticCache
import sys

def interactive_demo():
    """Run an interactive demo of the semantic cache."""
    print("🤖 Semantic Cache Demo")
    print("=" * 50)
    print("This demo shows how the semantic cache works with AI prompts.")
    print("Type 'quit' to exit, 'help' for commands.")
    print()
    
    # Initialize cache
    cache = SemanticCache(similarity_threshold=0.75)
    
    # Pre-populate with some examples
    examples = [
        ("What is artificial intelligence?", "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans."),
        ("How do I learn Python programming?", "Start with Python basics, practice with small projects, use online resources like tutorials and documentation."),
        ("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."),
        ("How to create a web API?", "You can create web APIs using frameworks like Flask (Python), Express (Node.js), or Spring Boot (Java)."),
        ("What is data science?", "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data."),
    ]
    
    print("Pre-loading cache with example prompts...")
    for prompt, response in examples:
        cache.put(prompt, response)
    print(f"✓ Loaded {len(examples)} example prompts")
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
                stats = cache.stats()
                print(f"Cache Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue
            elif user_input.lower() == 'search':
                query = input("Enter search query: ").strip()
                results = cache.search(query, top_k=3)
                if results:
                    print(f"Found {len(results)} similar prompts:")
                    for i, (prompt, response, similarity) in enumerate(results, 1):
                        print(f"  {i}. Similarity: {similarity:.3f}")
                        print(f"     Prompt: {prompt}")
                        print(f"     Response: {response[:100]}...")
                else:
                    print("No similar prompts found.")
                print()
                continue
            elif not user_input:
                continue
            
            # Try to get cached response
            print(f"🔍 Searching for: '{user_input}'")
            response = cache.get(user_input)
            
            if response:
                print(f"✅ Cache HIT! Found similar prompt.")
                print(f"📝 Response: {response}")
            else:
                print("❌ Cache MISS. No similar prompt found.")
                print("💡 You could add this prompt to the cache.")
                
                # Ask if user wants to add it
                add_response = input("Would you like to add a response for this prompt? (y/n): ").strip().lower()
                if add_response == 'y':
                    new_response = input("Enter the response: ").strip()
                    if new_response:
                        cache.put(user_input, new_response)
                        print("✅ Prompt and response added to cache!")
            
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
  stats    - Show cache statistics
  search   - Search for similar prompts
  
You can also enter any prompt to test semantic similarity!
""")

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_help()
        return
    
    interactive_demo()

if __name__ == "__main__":
    main()
