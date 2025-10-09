/**
 * Semantic Cache Web Application JavaScript
 * Handles real-time search suggestions and WebSocket communication
 */

class SemanticCacheApp {
    constructor() {
        this.socket = null;
        this.searchTimeout = null;
        this.isConnected = false;
        this.currentQuery = '';
        
        this.initializeElements();
        this.initializeSocket();
        this.bindEvents();
        this.loadStatus();
        this.loadStats();
        
        // Refresh stats every 10 seconds
        setInterval(() => this.loadStats(), 60000);
    }
    
    initializeElements() {
        this.elements = {
            promptInput: document.getElementById('prompt-input'),
            submitBtn: document.getElementById('submit-btn'),
            clearBtn: document.getElementById('clear-btn'),
            clearCacheBtn: document.getElementById('clear-cache-btn'),
            suggestionsContainer: document.getElementById('suggestions-container'),
            suggestionsList: document.getElementById('suggestions-list'),
            responseSection: document.getElementById('response-section'),
            responseContainer: document.getElementById('response-container'),
            llmStatus: document.getElementById('llm-status'),
            llmStatusText: document.getElementById('llm-status-text'),
            cacheEntries: document.getElementById('cache-entries'),
            modelName: document.getElementById('model-name'),
            hitRate: document.getElementById('hit-rate'),
            totalQueries: document.getElementById('total-queries'),
            partialSearches: document.getElementById('partial-searches'),
            avgCacheTime: document.getElementById('avg-cache-time'),
            avgLlmTime: document.getElementById('avg-llm-time')
        };
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
        });
        
        this.socket.on('search_response', (data) => {
            this.handleSearchResponse(data);
        });
        
        this.socket.on('query_response', (data) => {
            this.handleQueryResponse(data);
        });
        
        this.socket.on('status', (data) => {
            console.log('Status:', data.message);
        });
    }
    
    bindEvents() {
        // Input events
        this.elements.promptInput.addEventListener('input', (e) => {
            this.handleInputChange(e.target.value);
        });
        
        this.elements.promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.submitQuery();
            }
        });
        
        // Button events
        this.elements.submitBtn.addEventListener('click', () => {
            this.submitQuery();
        });
        
        this.elements.clearBtn.addEventListener('click', () => {
            this.clearInput();
        });
        
        this.elements.clearCacheBtn.addEventListener('click', () => {
            this.clearCache();
        });
    }
    
    handleInputChange(text) {
        this.currentQuery = text;
        this.updateSubmitButton();
        
        // Clear previous timeout
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }
        
        // Debounce search requests
        this.searchTimeout = setTimeout(() => {
            this.performSearch(text);
        }, 500); // 500ms delay
    }
    
    updateSubmitButton() {
        const hasText = this.currentQuery.trim().length > 0;
        this.elements.submitBtn.disabled = !hasText;
    }
    
    performSearch(text) {
        if (!text.trim() || text.trim().length < 5) {
            this.hideSuggestions();
            return;
        }
        
        if (this.isConnected) {
            this.socket.emit('search_request', {
                text: text,
                top_k: 5
            });
        } else {
            // Fallback to HTTP request
            this.performHttpSearch(text);
        }
    }
    
    performHttpSearch(text) {
        fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                top_k: 5
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Search error:', data.error);
                return;
            }
            this.handleSearchResponse(data);
        })
        .catch(error => {
            console.error('Search request failed:', error);
        });
    }
    
    handleSearchResponse(data) {
        if (data.error) {
            console.error('Search error:', data.error);
            this.hideSuggestions();
            return;
        }
        
        const suggestions = data.suggestions || [];
        
        if (suggestions.length > 0) {
            this.showSuggestions(suggestions);
        } else {
            this.hideSuggestions();
        }
    }
    
    showSuggestions(suggestions) {
        this.elements.suggestionsList.innerHTML = '';
        
        // Find the highest similarity for determining which gets "will be returned from cache"
        const maxSimilarity = Math.max(...suggestions.map(s => s.similarity));
        
        suggestions.forEach((suggestion, index) => {
            const isTopMatch = suggestion.similarity === maxSimilarity;
            const suggestionElement = this.createSuggestionElement(suggestion, index, isTopMatch);
            this.elements.suggestionsList.appendChild(suggestionElement);
        });
        
        this.elements.suggestionsContainer.classList.remove('hidden');
        this.elements.suggestionsContainer.classList.add('fade-in');
    }
    
    createSuggestionElement(suggestion, index, isTopMatch) {
        const div = document.createElement('div');
        
        // Check if similarity is above threshold (0.85 default)
        const similarityThreshold = 0.85;
        const isCacheHit = suggestion.similarity >= similarityThreshold;
        
        div.className = isCacheHit ? 'suggestion suggestion-cache-hit' : 'suggestion';
        
        // Determine the indicator message
        let indicatorHTML = '';
        if (isCacheHit) {
            if (isTopMatch) {
                indicatorHTML = '<div class="cache-hit-indicator"><i class="fas fa-check-circle"></i> Highest similarity. Will be returned from cache</div>';
            } else {
                indicatorHTML = '<div class="cache-hit-indicator cache-hit-threshold"><i class="fas fa-check"></i> Crossed similarity threshold</div>';
            }
        }
        
        div.innerHTML = `
            <div class="suggestion-header">
                <div class="suggestion-prompt">
                    ${isCacheHit ? '<i class="fas fa-bolt" style="color: #dc3545; margin-right: 5px;"></i>' : ''}
                    ${this.escapeHtml(suggestion.prompt)}
                </div>
                <div class="similarity-badge ${isCacheHit ? 'badge-cache-hit' : ''}">${(suggestion.similarity * 100).toFixed(1)}% match</div>
            </div>
            <div class="suggestion-response">${this.escapeHtml(suggestion.response)}</div>
            ${indicatorHTML}
        `;
        
        div.addEventListener('click', () => {
            this.useSuggestion(suggestion);
        });
        
        return div;
    }
    
    useSuggestion(suggestion) {
        this.elements.promptInput.value = suggestion.prompt;
        this.currentQuery = suggestion.prompt;
        this.updateSubmitButton();
        this.hideSuggestions();
        this.submitQuery();
    }
    
    hideSuggestions() {
        this.elements.suggestionsContainer.classList.add('hidden');
        this.elements.suggestionsContainer.classList.remove('fade-in');
    }
    
    submitQuery() {
        const prompt = this.currentQuery.trim();
        if (!prompt) return;
        
        this.showLoading();
        this.hideSuggestions();
        
        if (this.isConnected) {
            this.socket.emit('query_request', { prompt: prompt });
        } else {
            this.performHttpQuery(prompt);
        }
    }
    
    performHttpQuery(prompt) {
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                this.showError(data.error);
                return;
            }
            this.showResponse(data.response);
        })
        .catch(error => {
            this.showError('Request failed: ' + error.message);
        });
    }
    
    handleQueryResponse(data) {
        if (data.error) {
            this.showError(data.error);
            return;
        }
        this.showResponse(data.response);
    }
    
    showLoading() {
        this.queryStartTime = Date.now();
        this.elements.responseContainer.innerHTML = `
            <div class="loading-spinner"></div>
            Generating response...
        `;
        this.elements.responseContainer.className = 'response-container loading';
        this.elements.responseSection.classList.remove('hidden');
    }
    
    showResponse(response, wasCached = null) {
        const queryTime = this.queryStartTime ? ((Date.now() - this.queryStartTime) / 1000).toFixed(2) : null;
        
        // Split response into lines
        const lines = response.split('\n');
        const shouldCollapse = lines.length > 5;
        
        let responseHTML = '';
        
        if (shouldCollapse) {
            // Show first 5 lines with expand button
            const previewLines = lines.slice(0, 5).join('\n');
            const fullResponse = response;
            
            responseHTML = `
                <div class="response-preview">
                    <div class="response-text" id="response-preview-text">${this.escapeHtml(previewLines)}</div>
                    <div class="response-text hidden" id="response-full-text">${this.escapeHtml(fullResponse)}</div>
                    <button class="expand-btn" id="expand-btn" onclick="window.semanticCacheApp.toggleResponseExpansion()">
                        <i class="fas fa-chevron-down"></i>
                        Show More (${lines.length - 5} more lines)
                    </button>
                </div>
            `;
        } else {
            // Show full response if 5 lines or less
            responseHTML = `<div class="response-text">${this.escapeHtml(response)}</div>`;
        }
        
        // Add response time indicator if available
        if (queryTime !== null) {
            const timeClass = queryTime < 1 ? 'fast' : queryTime < 5 ? 'medium' : 'slow';
            const timeColor = queryTime < 1 ? '#28a745' : queryTime < 5 ? '#ffc107' : '#dc3545';
            responseHTML += `
                <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px; font-size: 0.9rem; color: #666;">
                    <i class="fas fa-stopwatch" style="color: ${timeColor};"></i>
                    <strong>Response Time:</strong> ${queryTime}s
                    ${wasCached !== null ? (wasCached ? '<span style="color: #28a745;"> (✓ From Cache)</span>' : '<span style="color: #007bff;"> (⚡ From LLM)</span>') : ''}
                </div>
            `;
        }
        
        this.elements.responseContainer.innerHTML = responseHTML;
        this.elements.responseContainer.className = 'response-container';
        this.elements.responseSection.classList.remove('hidden');
        this.elements.responseSection.classList.add('fade-in');
        
        // Refresh stats after getting a response
        this.loadStats();
    }
    
    toggleResponseExpansion() {
        const previewText = document.getElementById('response-preview-text');
        const fullText = document.getElementById('response-full-text');
        const expandBtn = document.getElementById('expand-btn');
        
        if (!previewText || !fullText || !expandBtn) return;
        
        const isExpanded = !fullText.classList.contains('hidden');
        
        if (isExpanded) {
            // Collapse
            previewText.classList.remove('hidden');
            fullText.classList.add('hidden');
            expandBtn.innerHTML = '<i class="fas fa-chevron-down"></i> Show More';
        } else {
            // Expand
            previewText.classList.add('hidden');
            fullText.classList.remove('hidden');
            expandBtn.innerHTML = '<i class="fas fa-chevron-up"></i> Show Less';
        }
    }
    
    showError(error) {
        this.elements.responseContainer.innerHTML = `
            <div style="color: #dc3545;">
                <i class="fas fa-exclamation-triangle"></i>
                Error: ${this.escapeHtml(error)}
            </div>
        `;
        this.elements.responseContainer.className = 'response-container';
        this.elements.responseSection.classList.remove('hidden');
    }
    
    clearInput() {
        this.elements.promptInput.value = '';
        this.currentQuery = '';
        this.updateSubmitButton();
        this.hideSuggestions();
        this.hideResponse();
    }
    
    hideResponse() {
        this.elements.responseSection.classList.add('hidden');
    }
    
    clearCache() {
        if (!confirm('Are you sure you want to clear the cache? This will remove all cached responses.')) {
            return;
        }
        
        fetch('/api/clear-cache', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error clearing cache: ' + data.error);
                return;
            }
            alert('Cache cleared successfully!');
            this.loadStatus();
            this.loadStats();
        })
        .catch(error => {
            alert('Failed to clear cache: ' + error.message);
        });
    }
    
    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'ok') {
                this.elements.llmStatus.className = data.llm_available ? 'status-dot' : 'status-dot error';
                this.elements.llmStatusText.textContent = data.llm_available ? 'Available' : 'Unavailable';
                this.elements.cacheEntries.textContent = data.cache_entries;
                this.elements.modelName.textContent = data.model;
            } else {
                this.elements.llmStatus.className = 'status-dot error';
                this.elements.llmStatusText.textContent = 'Error';
            }
        } catch (error) {
            console.error('Failed to load status:', error);
            this.elements.llmStatus.className = 'status-dot error';
            this.elements.llmStatusText.textContent = 'Error';
        }
    }
    
    updateStatusBar(stats) {
        // Update status bar with key metrics
        this.elements.hitRate.textContent = (stats.cache_hit_rate * 100).toFixed(1) + '%';
        this.elements.totalQueries.textContent = stats.total_queries;
        this.elements.partialSearches.textContent = stats.partial_searches;
        this.elements.cacheEntries.textContent = stats.cache_stats.total_entries;
        
        // Update response times
        const avgCacheTimeMs = (stats.avg_cache_time * 1000).toFixed(1);
        this.elements.avgCacheTime.textContent = avgCacheTimeMs + 'ms';
        
        const avgLlmTime = stats.avg_llm_time.toFixed(2);
        this.elements.avgLlmTime.textContent = avgLlmTime + 's';
    }
    
    async loadStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            if (data.error) {
                console.error('Stats error:', data.error);
                return;
            }
            
            this.updateStatusBar(data);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.semanticCacheApp = new SemanticCacheApp();
});

// Handle page visibility changes to refresh status when tab becomes active
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.semanticCacheApp) {
        window.semanticCacheApp.loadStatus();
        window.semanticCacheApp.loadStats();
    }
});
