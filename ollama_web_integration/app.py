#!/usr/bin/env python3
"""
Flask Web Application for Semantic Cache with Real-time Suggestions

This web application provides a user-friendly interface for the semantic cache
with real-time partial search suggestions as users type.
"""

import sys
import os
import json
import time
import asyncio
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cached_llm_web import CachedLLMWeb, SearchSuggestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'semantic_cache_web_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global cached LLM instance
cached_llm: CachedLLMWeb = None

def initialize_cached_llm():
    """Initialize the cached LLM system."""
    global cached_llm
    try:
        cached_llm = CachedLLMWeb(
            model_name="llama3:latest",
            similarity_threshold=0.85,
            partial_search_threshold=0.6,
            system_prompt="You are a helpful AI assistant. Provide concise, accurate answers.",
            cache_dir="./llm_cache_web"
        )
        logger.info("CachedLLMWeb initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CachedLLMWeb: {e}")
        return False

@app.route('/')
def index():
    """Main page with the chat interface."""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get system status."""
    if not cached_llm:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'llm_available': cached_llm.is_llm_available(),
        'model': cached_llm.model_name,
        'cache_entries': cached_llm.cache.stats()['total_entries']
    })

@app.route('/api/query', methods=['POST'])
def api_query():
    """Handle full query requests."""
    if not cached_llm:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Query the cached LLM
        response = cached_llm.query(prompt)
        
        return jsonify({
            'response': response,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    """Handle partial search requests."""
    if not cached_llm:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        partial_prompt = data.get('text', '').strip()
        top_k = data.get('top_k', 5)
        
        if not partial_prompt:
            return jsonify({'suggestions': []})
        
        # Check if we should search
        if not cached_llm.should_search_partial(partial_prompt):
            return jsonify({'suggestions': []})
        
        # Perform partial search
        suggestions = cached_llm.search_partial(partial_prompt, top_k)
        
        # Convert to JSON-serializable format
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append({
                'prompt': suggestion.prompt,
                'response': suggestion.response[:200] + '...' if len(suggestion.response) > 200 else suggestion.response,
                'similarity': round(float(suggestion.similarity), 3),  # Ensure it's a Python float
                'action': suggestion.action,
                'metadata': suggestion.metadata
            })
        
        return jsonify({
            'suggestions': suggestions_data,
            'query': partial_prompt,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """Get system statistics."""
    if not cached_llm:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        stats = cached_llm.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
def api_clear_cache():
    """Clear the cache."""
    if not cached_llm:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        cached_llm.clear_cache()
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to semantic cache server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('search_request')
def handle_search_request(data):
    """Handle real-time search requests via WebSocket."""
    if not cached_llm:
        emit('search_response', {'error': 'System not initialized'})
        return
    
    try:
        text = data.get('text', '').strip()
        top_k = data.get('top_k', 5)
        
        if not text or not cached_llm.should_search_partial(text):
            emit('search_response', {'suggestions': [], 'query': text})
            return
        
        # Perform partial search
        suggestions = cached_llm.search_partial(text, top_k)
        
        # Convert to JSON-serializable format
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append({
                'prompt': suggestion.prompt,
                'response': suggestion.response[:150] + '...' if len(suggestion.response) > 150 else suggestion.response,
                'similarity': round(float(suggestion.similarity), 3),  # Ensure it's a Python float
                'action': suggestion.action
            })
        
        emit('search_response', {
            'suggestions': suggestions_data,
            'query': text,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in WebSocket search: {e}")
        emit('search_response', {'error': str(e)})

@socketio.on('query_request')
def handle_query_request(data):
    """Handle query requests via WebSocket."""
    if not cached_llm:
        emit('query_response', {'error': 'System not initialized'})
        return
    
    try:
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            emit('query_response', {'error': 'No prompt provided'})
            return
        
        # Query the cached LLM
        response = cached_llm.query(prompt)
        
        emit('query_response', {
            'response': response,
            'prompt': prompt,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in WebSocket query: {e}")
        emit('query_response', {'error': str(e)})

@socketio.on('join_room')
def handle_join_room(data):
    """Handle joining a room for targeted updates."""
    room = data.get('room', 'default')
    join_room(room)
    emit('status', {'message': f'Joined room: {room}'})

@socketio.on('leave_room')
def handle_leave_room(data):
    """Handle leaving a room."""
    room = data.get('room', 'default')
    leave_room(room)
    emit('status', {'message': f'Left room: {room}'})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the web application."""
    print("🚀 Starting Semantic Cache Web Application")
    print("=" * 50)
    
    # Initialize the cached LLM
    if not initialize_cached_llm():
        print("❌ Failed to initialize cached LLM")
        return
    
    # Check if LLM is available
    if not cached_llm.is_llm_available():
        print("⚠️  Warning: LLM is not available")
        print("   Make sure Ollama is running on http://localhost:11434")
        print("   The web app will start but queries will fail")
    else:
        print("✅ LLM is available")
    
    print(f"📊 Cache entries: {cached_llm.cache.stats()['total_entries']}")
    print("🌐 Starting web server...")
    
    # Run the Flask-SocketIO app
    socketio.run(
        app,
        host='0.0.0.0',
        port=5051,
        debug=True,
        allow_unsafe_werkzeug=True
    )

if __name__ == '__main__':
    main()
