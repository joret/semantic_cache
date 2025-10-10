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
from sklearn.decomposition import PCA
import numpy as np

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

def reduce_embedding_dimensions(embedding: np.ndarray, all_embeddings: list = None, n_components: int = 2):
    """
    Reduce embedding dimensions for visualization using PCA.
    
    Args:
        embedding: Original embedding array to reduce
        all_embeddings: All cached embeddings for PCA fitting (required)
        n_components: Number of dimensions to reduce to (default: 3)
        
    Returns:
        Reduced embedding array or None if not enough samples
    """
    # Only perform PCA if we have enough cached embeddings
    if not all_embeddings or len(all_embeddings) < n_components:
        return None
    
    try:
        # Stack all embeddings into a matrix
        embeddings_matrix = np.stack(all_embeddings)
        
        # Fit PCA on all embeddings
        pca = PCA(n_components=n_components)
        pca.fit(embeddings_matrix)
        
        # Transform the query embedding
        embedding_2d = embedding.reshape(1, -1)
        reduced = pca.transform(embedding_2d)
        
        return reduced.flatten()
    except Exception as e:
        logger.error(f"PCA reduction failed: {e}")
        return None

def get_closest_embeddings_3d(query_embedding: np.ndarray, cache_instance, top_k: int = 10):
    """
    Get the closest cached embeddings in 2D space for visualization.
    
    Args:
        query_embedding: The query embedding
        cache_instance: The semantic cache instance
        top_k: Maximum number of closest embeddings to return
        
    Returns:
        Dict with 2D coordinates for query and closest embeddings, or None
    """
    if not cache_instance.embeddings or len(cache_instance.embeddings) < 2:
        return None
    
    try:
        # Get all embeddings
        all_embeddings = cache_instance.embeddings
        
        # Stack into matrix for PCA
        embeddings_matrix = np.stack(all_embeddings)
        
        # Add query embedding to the matrix for consistent PCA transformation
        combined_matrix = np.vstack([embeddings_matrix, query_embedding.reshape(1, -1)])
        
        # Apply PCA for 2D
        pca = PCA(n_components=2)
        reduced_all = pca.fit_transform(combined_matrix)
        
        # Separate query (last one) from cached embeddings
        reduced_cached = reduced_all[:-1]
        reduced_query = reduced_all[-1]
        
        # Calculate similarities and angles to find closest
        similarities = []
        for i, cached_emb in enumerate(all_embeddings):
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            )
            # Calculate angle in degrees
            angle_rad = np.arccos(np.clip(similarity, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            similarities.append((i, float(similarity), float(angle_deg)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k closest and top_k farthest
        closest = similarities[:top_k]
        farthest = similarities[-top_k:] if len(similarities) > top_k else []
        
        # Build visualization data for 2D
        plot_data = {
            'query': {
                'x': float(reduced_query[0]),
                'y': float(reduced_query[1]),
                'label': 'Current Query'
            },
            'closest': [],
            'farthest': []
        }
        
        # Add closest embeddings
        for i, sim_tuple in enumerate(closest):
            idx, similarity, angle = sim_tuple
            prompt_hash = cache_instance.prompt_hashes[idx]
            entry = cache_instance.cache[prompt_hash]
            
            plot_data['closest'].append({
                'x': float(reduced_cached[idx][0]),
                'y': float(reduced_cached[idx][1]),
                'label': entry.prompt[:50] + ('...' if len(entry.prompt) > 50 else ''),
                'similarity': float(similarity),
                'angle': float(angle)
            })
        
        # Add farthest embeddings
        for i, sim_tuple in enumerate(farthest):
            idx, similarity, angle = sim_tuple
            prompt_hash = cache_instance.prompt_hashes[idx]
            entry = cache_instance.cache[prompt_hash]
            
            plot_data['farthest'].append({
                'x': float(reduced_cached[idx][0]),
                'y': float(reduced_cached[idx][1]),
                'label': entry.prompt[:50] + ('...' if len(entry.prompt) > 50 else ''),
                'similarity': float(similarity),
                'angle': float(angle)
            })
        
        return plot_data
        
    except Exception as e:
        logger.error(f"Failed to create 3D plot data: {e}")
        return None

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
        
        # Generate embedding for the prompt
        embedding = cached_llm.cache._generate_embedding(prompt)
        
        # Reduce dimensions for visualization
        all_embeddings = cached_llm.cache.embeddings if cached_llm.cache.embeddings else None
        reduced_embedding = reduce_embedding_dimensions(embedding, all_embeddings=all_embeddings, n_components=2)
        
        embedding_data = {
            'values': embedding.tolist(),
            'shape': list(embedding.shape),
            'dtype': str(embedding.dtype),
            'min': float(embedding.min()),
            'max': float(embedding.max()),
            'mean': float(embedding.mean())
        }
        
        # Only include reduced values if PCA was successful
        if reduced_embedding is not None:
            embedding_data['reduced_values'] = reduced_embedding.tolist()
            embedding_data['reduced_shape'] = list(reduced_embedding.shape)
        
        # Get 3D plot data with closest embeddings
        plot_data = get_closest_embeddings_3d(embedding, cached_llm.cache, top_k=10)
        if plot_data:
            embedding_data['plot_data'] = plot_data
        
        return jsonify({
            'response': response,
            'timestamp': time.time(),
            'embedding': embedding_data
        })
        
    except Exception as e:
        logger.error(f"Error in query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embedding-preview', methods=['POST'])
def api_embedding_preview():
    """Get embedding visualization for partial input (real-time)."""
    if not cached_llm:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        partial_text = data.get('text', '').strip()
        
        if not partial_text:
            return jsonify({'plot_data': None})
        
        # Check if we should generate embedding (same threshold as search)
        if not cached_llm.should_search_partial(partial_text):
            return jsonify({'plot_data': None})
        
        # Generate embedding for partial text
        embedding = cached_llm.cache._generate_embedding(partial_text)
        
        # Get 2D plot data
        plot_data = get_closest_embeddings_3d(embedding, cached_llm.cache, top_k=10)
        
        return jsonify({
            'plot_data': plot_data,
            'text': partial_text
        })
        
    except Exception as e:
        logger.error(f"Error in embedding preview: {e}")
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
        
        # Generate embedding for the prompt
        embedding = cached_llm.cache._generate_embedding(prompt)
        
        # Reduce dimensions for visualization
        all_embeddings = cached_llm.cache.embeddings if cached_llm.cache.embeddings else None
        reduced_embedding = reduce_embedding_dimensions(embedding, all_embeddings=all_embeddings, n_components=2)
        
        embedding_data = {
            'values': embedding.tolist(),
            'shape': list(embedding.shape),
            'dtype': str(embedding.dtype),
            'min': float(embedding.min()),
            'max': float(embedding.max()),
            'mean': float(embedding.mean())
        }
        
        # Only include reduced values if PCA was successful
        if reduced_embedding is not None:
            embedding_data['reduced_values'] = reduced_embedding.tolist()
            embedding_data['reduced_shape'] = list(reduced_embedding.shape)
        
        # Get 3D plot data with closest embeddings
        plot_data = get_closest_embeddings_3d(embedding, cached_llm.cache, top_k=10)
        if plot_data:
            embedding_data['plot_data'] = plot_data
        
        emit('query_response', {
            'response': response,
            'prompt': prompt,
            'timestamp': time.time(),
            'embedding': embedding_data
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
