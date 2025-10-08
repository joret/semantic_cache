# Semantic Cache Web Integration

A web-based interface for the semantic cache with real-time suggestions as users type. This extends the terminal-based ollama_integration with a modern web interface that provides instant feedback and suggestions.

## Features

- **Real-time Search Suggestions**: Get suggestions as you type (after 10+ words)
- **WebSocket Communication**: Instant updates and real-time search
- **Modern Web Interface**: Beautiful, responsive design
- **Cache Management**: Clear cache, view statistics
- **Dual Communication**: WebSocket + HTTP fallback
- **Live Statistics**: Real-time performance metrics
- **Mobile Responsive**: Works on all devices

## Prerequisites

1. **Install Ollama**: [https://ollama.ai](https://ollama.ai)
2. **Start Ollama**: `ollama serve`
3. **Pull a model**: `ollama pull llama3:latest`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure you have the main semantic_cache module
# (it should be in the parent directory)
```

## Quick Start

### Start the Web Application

```bash
python app.py
```

The web application will start on `http://localhost:5051`

### Access the Interface

1. Open your browser and go to `http://localhost:5051`
2. Start typing a question in the input field
3. Watch for real-time suggestions after 10+ words
4. Click on suggestions to use cached responses
5. Submit queries to get AI responses

## How It Works

### Real-time Suggestions

1. **Type Detection**: System detects when you've typed enough (10+ words or question patterns)
2. **Semantic Search**: Searches cached responses for similar prompts
3. **Live Suggestions**: Shows matching prompts with similarity scores
4. **Instant Use**: Click any suggestion to use the cached response

### WebSocket Integration

- **Real-time Updates**: Instant search results via WebSocket
- **HTTP Fallback**: Falls back to HTTP requests if WebSocket fails
- **Live Statistics**: Real-time performance metrics
- **Connection Status**: Shows server connection status

### Cache Management

- **Automatic Caching**: New responses are automatically cached
- **Cache Statistics**: View hit rates, response times, etc.
- **Cache Clearing**: Clear all cached responses
- **Search Cache**: Separate cache for search results (5-minute TTL)

## API Endpoints

### HTTP Endpoints

- `GET /` - Main web interface
- `GET /api/status` - System status
- `POST /api/query` - Submit a query
- `POST /api/search` - Search for similar prompts
- `GET /api/stats` - Get statistics
- `POST /api/clear-cache` - Clear the cache

### WebSocket Events

- `search_request` - Request search suggestions
- `search_response` - Receive search results
- `query_request` - Submit a query
- `query_response` - Receive query response
- `connect` - Client connected
- `disconnect` - Client disconnected

## Configuration

### Environment Variables

```bash
export FLASK_ENV=development  # For development mode
export FLASK_DEBUG=1          # Enable debug mode
```

### Application Settings

Modify `app.py` to change:

- **Model**: Change `model_name` parameter
- **Similarity Threshold**: Adjust `similarity_threshold`
- **Partial Search Threshold**: Adjust `partial_search_threshold`
- **Cache Directory**: Change `cache_dir`
- **Server Port**: Modify `socketio.run(port=5000)`

## Usage Examples

### Basic Usage

1. **Start typing**: "What is artificial intelligence and how does it work in"
2. **See suggestions**: Similar cached questions appear automatically
3. **Click suggestion**: Use a cached response instantly
4. **Submit query**: Get new AI response if no good match

### Advanced Features

- **Keyboard shortcuts**: Ctrl+Enter to submit
- **Real-time stats**: Watch performance metrics update
- **Cache management**: Clear cache when needed
- **Mobile support**: Responsive design for all devices

## Performance Features

### Search Optimization

- **Debounced Search**: 500ms delay to avoid excessive requests
- **Search Caching**: Results cached for 5 minutes
- **Minimum Word Count**: Only searches after 3+ words
- **Smart Triggering**: Searches on question patterns

### Real-time Updates

- **WebSocket Priority**: Uses WebSocket when available
- **HTTP Fallback**: Falls back to HTTP if WebSocket fails
- **Connection Monitoring**: Shows connection status
- **Auto-reconnect**: Attempts to reconnect if disconnected

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   ```bash
   # Check if the server is running
   curl http://localhost:5000/api/status
   ```

2. **No Suggestions Appearing**
   - Make sure you've typed 10+ words
   - Check if there are cached responses
   - Verify Ollama is running

3. **Slow Response Times**
   - Check Ollama server status
   - Verify model is loaded
   - Check system resources

4. **Cache Not Working**
   - Clear browser cache
   - Restart the web application
   - Check cache directory permissions

### Debug Mode

```bash
# Run with debug logging
FLASK_DEBUG=1 python app.py
```

### Check Logs

The application logs to console. Look for:
- Connection status messages
- Search request logs
- Error messages
- Performance metrics

## Development

### Project Structure

```
ollama_web_integration/
├── app.py                 # Flask application
├── cached_llm_web.py      # Enhanced cached LLM
├── ollama_client.py       # Ollama client
├── templates/
│   └── index.html         # Main web interface
├── static/
│   └── js/
│       └── app.js         # Frontend JavaScript
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Adding Features

1. **New API Endpoints**: Add routes to `app.py`
2. **Frontend Changes**: Modify `templates/index.html` and `static/js/app.js`
3. **Cache Logic**: Extend `cached_llm_web.py`
4. **WebSocket Events**: Add handlers to `app.py`

### Testing

```bash
# Test the API
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?"}'

# Test search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"text": "What is artificial intelligence"}'
```

## Comparison with Terminal Version

| Feature | Terminal Version | Web Version |
|---------|------------------|-------------|
| **Interface** | Command line | Web browser |
| **Suggestions** | Manual search | Real-time auto |
| **Communication** | HTTP only | WebSocket + HTTP |
| **Statistics** | On-demand | Live updates |
| **Cache Management** | CLI commands | Web interface |
| **Mobile Support** | No | Yes |
| **Multi-user** | Single user | Multiple users |

## Security Considerations

- **No Authentication**: Currently no user authentication
- **Local Only**: Designed for local use (localhost:5000)
- **Input Validation**: Basic validation on inputs
- **CORS Enabled**: Allows cross-origin requests

For production use, consider adding:
- User authentication
- Rate limiting
- Input sanitization
- HTTPS support
- Access controls

## License

Same as the main semantic cache project.
