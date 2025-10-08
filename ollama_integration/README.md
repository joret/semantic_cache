# Semantic Cache with Ollama LLM Integration

This module extends the semantic cache to work with local LLMs via Ollama. When there's a cache miss, it automatically queries the local LLM and stores the response for future use.

## Features

- **Semantic Caching**: Intelligent cache based on semantic similarity
- **Local LLM Integration**: Uses Ollama for local AI responses
- **Automatic Fallback**: Queries LLM on cache misses
- **Streaming Support**: Real-time response streaming
- **Performance Metrics**: Detailed statistics and timing
- **Model Management**: Easy model switching and management

## Prerequisites

1. **Install Ollama**: [https://ollama.ai](https://ollama.ai)
2. **Start Ollama**: `ollama serve`
3. **Pull a model**: `ollama pull llama2`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure you have the main semantic_cache module
# (it should be in the parent directory)
```

## TODO
- Upgrade Python 
- Remove call to tags when cache hit
- test other approaches 1. Vectorized Operations (Current + Optimized), 2. Approximate Nearest Neighbors (FAISS), 3. Locality Sensitive Hashing (LSH):
- prepare some theory about vectors and cosine similarity between two embeddings.
- Point out similarities of techniques between rag and semantic cache




## Quick Start

### Basic Usage

```python
from cached_llm import CachedLLM

# Initialize cached LLM
cached_llm = CachedLLM(
    model_name="llama2",
    similarity_threshold=0.8
)

# Query with automatic caching
response = cached_llm.query("What is artificial intelligence?")
print(response)
```

### Command Line Usage

```bash
# Basic query
python cached_llm.py --model llama2 --prompt "What is AI?"

# Streaming response
python cached_llm.py --model llama2 --prompt "Explain machine learning" --stream

# Disable cache
python cached_llm.py --model llama2 --prompt "What is Python?" --no-cache

# Show statistics
python cached_llm.py --model llama2 --prompt "Hello" --stats
```

### Interactive Demo

```bash
python demo.py
```

## Configuration Options

### CachedLLM Parameters

- `model_name`: Ollama model to use (default: "llama2")
- `cache_model`: Sentence transformer for embeddings (default: 'all-MiniLM-L6-v2')
- `cache_dir`: Cache storage directory (default: './llm_cache')
- `similarity_threshold`: Minimum similarity for cache hits (default: 0.85)
- `max_cache_size`: Maximum cache entries (default: 1000)
- `ollama_url`: Ollama server URL (default: "http://localhost:11434")
- `system_prompt`: Optional system prompt for LLM
- `llm_options`: Additional LLM generation options

### Example Configuration

```python
cached_llm = CachedLLM(
    model_name="codellama",
    similarity_threshold=0.9,
    system_prompt="You are a coding assistant. Provide clear, concise code examples.",
    llm_options={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1000
    }
)
```

## API Reference

### CachedLLM Class

#### Methods

- `query(prompt, use_cache=True)`: Query with caching
- `query_stream(prompt, use_cache=True)`: Stream response with caching
- `get_stats()`: Get performance statistics
- `clear_cache()`: Clear the semantic cache
- `search_cache(query, top_k=5)`: Search cache for similar prompts
- `list_available_models()`: List available Ollama models
- `pull_model(model_name)`: Pull model from Ollama registry
- `is_llm_available()`: Check if LLM is available

#### Statistics

The `get_stats()` method returns:

```python
{
    'llm_model': 'llama2',
    'llm_available': True,
    'total_queries': 10,
    'cache_hits': 7,
    'cache_misses': 3,
    'llm_calls': 3,
    'hit_rate': 0.7,
    'avg_llm_time': 2.5,
    'avg_cache_time': 0.001,
    'total_llm_time': 7.5,
    'total_cache_time': 0.007,
    'cache_stats': {...}
}
```

## Examples

### Basic Example

```python
from cached_llm import CachedLLM

# Initialize
cached_llm = CachedLLM(model_name="llama2")

# First query - will call LLM
response1 = cached_llm.query("What is Python?")
print(f"Response 1: {response1}")

# Similar query - will use cache
response2 = cached_llm.query("Tell me about Python programming")
print(f"Response 2: {response2}")  # Same as response1!

# Show statistics
stats = cached_llm.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### Streaming Example

```python
from cached_llm import CachedLLM

cached_llm = CachedLLM(model_name="llama2")

print("Streaming response:")
for chunk in cached_llm.query_stream("Explain quantum computing"):
    print(chunk, end='', flush=True)
print()
```

### Custom System Prompt

```python
from cached_llm import CachedLLM

cached_llm = CachedLLM(
    model_name="llama2",
    system_prompt="You are a helpful coding assistant. Always provide code examples when relevant."
)

response = cached_llm.query("How do I create a REST API in Python?")
print(response)
```

## Performance Tips

1. **Adjust Similarity Threshold**: Higher values (0.9+) for strict matching, lower (0.7-0.8) for more lenient
2. **Use Appropriate Models**: Smaller models for faster responses, larger for better quality
3. **Monitor Hit Rates**: Aim for 60-80% hit rate for optimal performance
4. **Cache Size**: Larger caches improve hit rates but use more memory

## Troubleshooting

### Common Issues

1. **Ollama not available**
   ```bash
   # Start Ollama
   ollama serve
   
   # Check if running
   curl http://localhost:11434/api/tags
   ```

2. **No models available**
   ```bash
   # Pull a model
   ollama pull llama2
   
   # List available models
   ollama list
   ```

3. **Slow responses**
   - Use smaller models (llama2:7b instead of llama2:13b)
   - Adjust similarity threshold
   - Check system resources

4. **Low cache hit rate**
   - Lower similarity threshold
   - Increase cache size
   - Check prompt variations

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.INFO)

cached_llm = CachedLLM(model_name="llama2")
# Now you'll see detailed logs
```

## Supported Models

Any Ollama model works, including:

- **llama2**: General purpose (default)
- **codellama**: Code generation
- **mistral**: Fast and efficient
- **neural-chat**: Conversational AI
- **vicuna**: Open-source chat model

## Use Cases

- **AI Chatbots**: Cache common questions and responses
- **Code Generation**: Cache similar code requests
- **Documentation**: Cache frequently asked questions
- **Content Creation**: Cache similar content requests
- **API Responses**: Reduce LLM API calls

## License

Same as the main semantic cache project.
