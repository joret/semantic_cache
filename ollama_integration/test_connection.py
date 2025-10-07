#!/usr/bin/env python3
"""
Test Ollama connection and basic functionality
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_client import OllamaClient
from cached_llm import CachedLLM

def test_ollama_connection():
    """Test basic Ollama connection."""
    print("🔍 Testing Ollama Connection")
    print("=" * 30)
    
    client = OllamaClient()
    
    # Test connection
    if not client.is_available():
        print("❌ Ollama server is not available")
        print("Make sure Ollama is running on http://localhost:11434")
        return False
    
    print("✅ Ollama server is available")
    
    # List models
    models = client.list_models()
    if models:
        print(f"📋 Available models: {', '.join(models)}")
    else:
        print("⚠️  No models found")
        return False
    
    return True

def test_cached_llm():
    """Test cached LLM functionality."""
    print("\n🤖 Testing Cached LLM")
    print("=" * 30)
    
    try:
        # Initialize cached LLM
        cached_llm = CachedLLM(model_name="llama3:latest")
        
        if not cached_llm.is_llm_available():
            print("❌ LLM is not available")
            return False
        
        print("✅ Cached LLM initialized successfully")
        
        # Test a simple query
        print("🔍 Testing simple query...")
        response = cached_llm.query("What is 2+2?")
        print(f"✅ Response: {response[:100]}...")
        
        # Test cache hit
        print("🔍 Testing cache hit...")
        response2 = cached_llm.query("What is two plus two?")
        print(f"✅ Cached response: {response2[:100]}...")
        
        # Show statistics
        stats = cached_llm.get_stats()
        print(f"📊 Hit rate: {stats['hit_rate']:.1%}")
        print(f"📊 Total queries: {stats['total_queries']}")
        print(f"📊 Cache hits: {stats['cache_hits']}")
        print(f"📊 LLM calls: {stats['llm_calls']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cached LLM: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Ollama Integration Test Suite")
    print("=" * 40)
    
    # Test Ollama connection
    if not test_ollama_connection():
        print("\n❌ Ollama connection test failed")
        return
    
    # Test cached LLM
    if not test_cached_llm():
        print("\n❌ Cached LLM test failed")
        return
    
    print("\n✅ All tests passed!")
    print("🎉 Ollama integration is working correctly!")

if __name__ == "__main__":
    main()
