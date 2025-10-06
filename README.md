# Semantic Cache for AI Prompts

A powerful semantic cache system that stores AI prompts and their responses, enabling intelligent retrieval based on semantic similarity rather than exact string matches. This is particularly useful for AI applications where similar prompts should return cached responses to improve performance and reduce API costs.

## Features

- **Semantic Similarity**: Uses sentence transformers to find semantically similar prompts
- **Configurable Thresholds**: Adjustable similarity thresholds for cache hits
- **Multiple Storage Options**: In-memory caching with optional disk persistence
- **Metadata Support**: Store additional metadata with each cache entry
- **CLI Interface**: Command-line tools for cache management
- **Configuration Management**: JSON-based configuration system
- **Performance Optimized**: Efficient embedding storage and similarity calculations
- **Automatic Eviction**: LRU-style eviction when cache reaches maximum size

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic_cache
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install as a package:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from semantic_cache import SemanticCache

# Initialize cache
cache = SemanticCache()

# Store a prompt-response pair
cache.put(
    "What is the capital of France?", 
    "The capital of France is Paris."
)

# Retrieve similar prompt
response = cache.get("What's the main city in France?")
print(response)  # "The capital of France is Paris."
```

### Command Line Usage

```bash
# Store a prompt-response pair
python semantic_cache.py put "What is AI?" "Artificial Intelligence is..."

# Retrieve cached response
python semantic_cache.py get "Tell me about artificial intelligence"

# Search for similar prompts
python semantic_cache.py search "machine learning" --top-k 3

# View cache statistics
python semantic_cache.py stats

# Clear the cache
python semantic_cache.py clear
```

## Configuration

The cache can be configured through a JSON configuration file or programmatically:

```python
from config import load_config, update_config

# Load configuration
config = load_config()

# Update configuration
update_config(
    similarity_threshold=0.9,
    max_cache_size=1000,
    model_name='all-MiniLM-L6-v2'
)
```

### Configuration Options

- `model_name`: Sentence transformer model to use (default: 'all-MiniLM-L6-v2')
- `cache_dir`: Directory for cache storage (default: './cache')
- `similarity_threshold`: Minimum similarity for cache hits (default: 0.85)
- `max_cache_size`: Maximum number of cache entries (default: 1000)
- `auto_save`: Automatically save cache to disk (default: True)
- `save_interval`: Save after every N operations (default: 10)
- `enable_logging`: Enable logging output (default: True)
- `log_level`: Logging level (default: 'INFO')

## Advanced Usage

### Custom Configuration

```python
cache = SemanticCache(
    model_name='all-MiniLM-L6-v2',
    cache_dir='./my_cache',
    similarity_threshold=0.8,
    max_cache_size=500
)
```

### Storing Metadata

```python
cache.put(
    "How to create a REST API?",
    "Use Flask or FastAPI to create REST APIs...",
    metadata={
        "topic": "programming",
        "difficulty": "intermediate",
        "tags": ["api", "python", "web"]
    }
)
```

### Search Functionality

```python
# Search for similar prompts
results = cache.search("Python web development", top_k=5)

for prompt, response, similarity in results:
    print(f"Similarity: {similarity:.3f}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
```

### Cache Statistics

```python
stats = cache.stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Max size: {stats['max_size']}")
print(f"Similarity threshold: {stats['similarity_threshold']}")
```

## Examples

Run the included examples to see the cache in action:

```bash
python examples.py
```

The examples demonstrate:
- Basic usage patterns
- Advanced configuration
- Metadata storage
- Performance characteristics
- Configuration management

## API Reference

### SemanticCache Class

#### Constructor
```python
SemanticCache(
    model_name: str = 'all-MiniLM-L6-v2',
    cache_dir: str = './cache',
    similarity_threshold: float = 0.85,
    max_cache_size: int = 1000
)
```

#### Methods

- `get(prompt: str) -> Optional[str]`: Retrieve cached response for similar prompt
- `put(prompt: str, response: str, metadata: Optional[Dict] = None)`: Store prompt-response pair
- `search(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]`: Search for similar prompts
- `clear() -> None`: Clear all cache entries
- `stats() -> Dict[str, Any]`: Get cache statistics

### Configuration Management

- `load_config() -> CacheConfig`: Load configuration from file
- `save_config() -> None`: Save current configuration
- `update_config(**kwargs) -> None`: Update configuration values

## Performance Considerations

- **Embedding Model**: The choice of sentence transformer model affects both accuracy and speed
- **Similarity Threshold**: Higher thresholds reduce false positives but may miss valid matches
- **Cache Size**: Larger caches provide better hit rates but use more memory
- **Batch Operations**: Consider batching multiple operations for better performance

## Supported Models

The cache works with any sentence transformer model from the [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) library. Popular choices include:

- `all-MiniLM-L6-v2`: Fast, good quality (default)
- `all-mpnet-base-v2`: Higher quality, slower
- `all-MiniLM-L12-v2`: Balanced quality and speed
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

## Use Cases

- **AI Chatbots**: Cache common questions and responses
- **Code Generation**: Cache similar code generation requests
- **Documentation Systems**: Cache frequently asked questions
- **API Response Caching**: Reduce API calls for similar requests
- **Content Generation**: Cache similar content generation prompts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model Download**: First run may take time to download the embedding model
3. **Memory Usage**: Large caches consume significant memory; adjust `max_cache_size` accordingly
4. **Similarity Threshold**: If getting too many/few cache hits, adjust the similarity threshold

### Performance Tips

- Use smaller embedding models for faster inference
- Adjust similarity threshold based on your use case
- Consider batch processing for multiple queries
- Monitor cache hit rates to optimize configuration

## Changelog

### v1.0.0
- Initial release
- Basic semantic caching functionality
- CLI interface
- Configuration management
- Metadata support
- Performance optimizations
