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
            avgLlmTime: document.getElementById('avg-llm-time'),
            embeddingSection: document.getElementById('embedding-section'),
            embeddingContent: document.getElementById('embedding-content'),
            embeddingContainer: document.getElementById('embedding-container'),
            embeddingPlotLive: document.getElementById('embedding-plot-container-live'),
            embeddingHeader: document.getElementById('embedding-header'),
            embeddingChevron: document.getElementById('embedding-chevron'),
            toggleEmbeddingBtn: document.getElementById('toggle-embedding-btn'),
            plotKValue: document.getElementById('plot-k-value')
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
        
        this.elements.toggleEmbeddingBtn.addEventListener('click', () => {
            this.toggleEmbedding();
        });
        
        this.elements.plotKValue.addEventListener('change', () => {
            this.updatePlot();
        });
        
        this.elements.embeddingHeader.addEventListener('click', () => {
            this.toggleEmbeddingSection();
        });
    }
    
    handleInputChange(text) {
        this.currentQuery = text;
        this.updateSubmitButton();
        
        // Clear previous timeout
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }
        
        // Debounce search and embedding preview requests
        this.searchTimeout = setTimeout(() => {
            this.performSearch(text);
            this.performEmbeddingPreview(text);
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
    
    performEmbeddingPreview(text) {
        if (!text.trim()) {
            return;
        }
        
        // Request embedding preview
        fetch('/api/embedding-preview', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Embedding preview error:', data.error);
                return;
            }
            if (data.plot_data) {
                this.showLiveEmbeddingPlot(data.plot_data);
            }
        })
        .catch(error => {
            console.error('Embedding preview request failed:', error);
        });
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
            if (data.embedding) {
                this.showEmbedding(data.embedding);
            }
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
        if (data.embedding) {
            this.showEmbedding(data.embedding);
        }
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
    
    toggleEmbeddingSection() {
        const isHidden = this.elements.embeddingContent.classList.contains('hidden');
        if (isHidden) {
            // Expand section
            this.elements.embeddingContent.classList.remove('hidden');
            this.elements.embeddingChevron.style.transform = 'rotate(90deg)';
        } else {
            // Collapse section
            this.elements.embeddingContent.classList.add('hidden');
            this.elements.embeddingChevron.style.transform = 'rotate(0deg)';
        }
    }
    
    toggleEmbedding() {
        const isHidden = this.elements.embeddingContainer.classList.contains('hidden');
        if (isHidden) {
            // Show details
            this.elements.embeddingContainer.classList.remove('hidden');
            this.elements.toggleEmbeddingBtn.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Details';
        } else {
            // Hide details
            this.elements.embeddingContainer.classList.add('hidden');
            this.elements.toggleEmbeddingBtn.innerHTML = '<i class="fas fa-eye"></i> Show Details';
        }
    }
    
    showEmbedding(embeddingData) {
        if (!embeddingData) return;
        
        // Store plot data for later use (when k changes)
        this.currentPlotData = embeddingData.plot_data;
        
        const values = embeddingData.values;
        const reducedValues = embeddingData.reduced_values;
        const truncatedArray = [
            ...values.slice(0, 10),
            '...',
            ...values.slice(-10)
        ];
        
        let reducedHTML = '';
        if (reducedValues && reducedValues.length === 2) {
            reducedHTML = `
                <div class="embedding-array" style="background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin-bottom: 10px;">
                    <strong>📊 2D Reduced Embedding (PCA Visualization):</strong><br>
                    <div style="font-size: 1.1rem; margin: 10px 0; color: #2e7d32;">
                        [${reducedValues.map(v => v.toFixed(6)).join(', ')}]
                    </div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 8px;">
                        <i class="fas fa-info-circle"></i> Original ${values.length}D embedding reduced to 2D using PCA
                    </div>
                </div>
            `;
        } else {
            reducedHTML = `
                <div class="embedding-array" style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 10px;">
                    <strong>⚠️ 2D Reduction Not Available:</strong><br>
                    <div style="font-size: 0.9rem; margin: 10px 0; color: #856404;">
                        Need at least 2 cached embeddings to perform PCA reduction.
                    </div>
                    <div style="font-size: 0.8rem; color: #666;">
                        <i class="fas fa-info-circle"></i> Current cache size: ${values.length < 2 ? '0-1' : 'insufficient'} embeddings
                    </div>
                </div>
            `;
        }
        
        this.elements.embeddingContainer.innerHTML = `
            <div class="embedding-stats">
                <strong>Embedding Statistics:</strong><br>
                Original Shape: ${embeddingData.shape.join(' × ')} (${values.length} dimensions)<br>
                Data Type: ${embeddingData.dtype}<br>
                Min: ${embeddingData.min.toFixed(4)}<br>
                Max: ${embeddingData.max.toFixed(4)}<br>
                Mean: ${embeddingData.mean.toFixed(4)}
            </div>
            ${reducedHTML}
            <div class="embedding-array">
                <strong>Full Array Sample (showing first 10 and last 10 of ${values.length}):</strong><br>
                <div style="font-size: 0.8rem; color: #666; margin-top: 5px;">
                    [${truncatedArray.map(v => typeof v === 'number' ? v.toFixed(4) : v).join(', ')}]
                </div>
            </div>
            <div id="embedding-plot-container" style="margin-top: 15px;"></div>
        `;
        
        this.elements.embeddingSection.classList.remove('hidden');
        
        // Render 3D plot if plot data is available
        if (this.currentPlotData) {
            this.updatePlot();
        }
    }
    
    updatePlot() {
        // Get the selected k value
        const kValue = parseInt(this.elements.plotKValue.value) || 4;
        
        // Update both live and detail plots
        if (this.currentLivePlotData) {
            this.renderLivePlot(this.currentLivePlotData, kValue);
        }
        if (this.currentPlotData) {
            this.render2DPlot(this.currentPlotData, kValue);
        }
    }
    
    showLiveEmbeddingPlot(plotData) {
        if (!plotData) return;
        
        // Store for later updates
        this.currentLivePlotData = plotData;
        
        // Show the embedding section header (content stays collapsed by default)
        this.elements.embeddingSection.classList.remove('hidden');
        
        // Render the live plot (even if collapsed, it will be ready when expanded)
        const kValue = parseInt(this.elements.plotKValue.value) || 4;
        this.renderLivePlot(plotData, kValue);
    }
    
    renderLivePlot(plotData, kValue = 4) {
        if (!plotData || typeof Plotly === 'undefined') return;
        
        // Use same rendering as detail plot but in different container
        const closestSubset = (plotData.closest || []).slice(0, kValue);
        const farthestSubset = (plotData.farthest || []).slice(0, kValue);
        
        const traces = [];
        
        // Query vector
        traces.push({
            x: [0, plotData.query.x],
            y: [0, plotData.query.y],
            mode: 'lines',
            type: 'scatter',
            line: { color: 'red', width: 3 },
            hoverinfo: 'skip',
            showlegend: false
        });
        
        // Query point
        traces.push({
            x: [plotData.query.x],
            y: [plotData.query.y],
            mode: 'markers',
            type: 'scatter',
            name: 'Typing...',
            marker: {
                size: 12,
                color: 'red',
                symbol: 'diamond',
                line: { color: 'darkred', width: 2 }
            },
            hovertemplate: 'Current Input<br>Position: (%{x:.3f}, %{y:.3f})<extra></extra>'
        });
        
        // Origin
        traces.push({
            x: [0],
            y: [0],
            mode: 'markers',
            type: 'scatter',
            name: 'Origin',
            marker: { size: 8, color: 'black' },
            hovertemplate: 'Origin (0, 0)<extra></extra>'
        });
        
        // Closest vectors and points
        closestSubset.forEach((point, idx) => {
            traces.push({
                x: [0, point.x],
                y: [0, point.y],
                mode: 'lines',
                type: 'scatter',
                line: { color: this.getColorBySimil(point.similarity), width: 2, dash: 'dot' },
                hoverinfo: 'skip',
                showlegend: false
            });
        });
        
        if (closestSubset.length > 0) {
            traces.push({
                x: closestSubset.map(p => p.x),
                y: closestSubset.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: `${kValue} Closest`,
                marker: {
                    size: 10,
                    color: closestSubset.map(p => p.similarity),
                    colorscale: 'Viridis',
                    showscale: false,
                    line: { color: 'black', width: 1 }
                },
                text: closestSubset.map(p => p.label),
                hovertemplate: '<b>%{text}</b><br>Similarity: %{marker.color:.3f}<br>Angle: %{customdata:.1f}°<extra></extra>',
                customdata: closestSubset.map(p => p.angle)
            });
        }
        
        // Farthest vectors and points
        farthestSubset.forEach((point, idx) => {
            traces.push({
                x: [0, point.x],
                y: [0, point.y],
                mode: 'lines',
                type: 'scatter',
                line: { color: 'rgba(150, 150, 150, 0.3)', width: 1.5, dash: 'dash' },
                hoverinfo: 'skip',
                showlegend: false
            });
        });
        
        if (farthestSubset.length > 0) {
            traces.push({
                x: farthestSubset.map(p => p.x),
                y: farthestSubset.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: `${kValue} Farthest`,
                marker: {
                    size: 8,
                    color: 'gray',
                    symbol: 'x',
                    line: { color: 'gray', width: 1 }
                },
                text: farthestSubset.map(p => p.label),
                hovertemplate: '<b>%{text}</b><br>Similarity: %{customdata[0]:.3f}<br>Angle: %{customdata[1]:.1f}°<extra></extra>',
                customdata: farthestSubset.map(p => [p.similarity, p.angle])
            });
        }
        
        const layout = {
            title: {
                text: `Live Embedding Position - ${kValue} Closest + ${kValue} Farthest`,
                font: { size: 14 }
            },
            xaxis: { 
                title: 'PC1',
                zeroline: true,
                zerolinewidth: 2,
                zerolinecolor: 'lightgray'
            },
            yaxis: { 
                title: 'PC2',
                zeroline: true,
                zerolinewidth: 2,
                zerolinecolor: 'lightgray'
            },
            showlegend: true,
            height: 400,
            hovermode: 'closest',
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: false,
            displaylogo: false
        };
        
        Plotly.newPlot('embedding-plot-container-live', traces, layout, config);
    }
    
    render2DPlot(plotData, kValue = 4) {
        if (!plotData || typeof Plotly === 'undefined') return;
        
        // Limit closest and farthest points to kValue
        const closestSubset = (plotData.closest || plotData.cached || []).slice(0, kValue);
        const farthestSubset = (plotData.farthest || []).slice(0, kValue);
        
        const traces = [];
        
        // Add vector from origin to query point
        const queryVector = {
            x: [0, plotData.query.x],
            y: [0, plotData.query.y],
            mode: 'lines',
            type: 'scatter',
            name: 'Query Vector',
            line: {
                color: 'red',
                width: 4
            },
            hoverinfo: 'skip',
            showlegend: false
        };
        traces.push(queryVector);
        
        // Add query point
        const queryTrace = {
            x: [plotData.query.x],
            y: [plotData.query.y],
            mode: 'markers+text',
            type: 'scatter',
            name: 'Current Query',
            text: [plotData.query.label],
            textposition: 'top center',
            marker: {
                size: 15,
                color: 'red',
                symbol: 'diamond',
                line: {
                    color: 'darkred',
                    width: 2
                }
            },
            hovertemplate: '<b>%{text}</b><br>Position: (%{x:.3f}, %{y:.3f})<extra></extra>'
        };
        traces.push(queryTrace);
        
        // Add origin point
        const originTrace = {
            x: [0],
            y: [0],
            mode: 'markers',
            type: 'scatter',
            name: 'Origin',
            marker: {
                size: 10,
                color: 'black',
                symbol: 'circle'
            },
            hovertemplate: 'Origin (0, 0)<extra></extra>'
        };
        traces.push(originTrace);
        
        // Add vectors for closest embeddings
        closestSubset.forEach((point, idx) => {
            const cachedVector = {
                x: [0, point.x],
                y: [0, point.y],
                mode: 'lines',
                type: 'scatter',
                name: `Closest Vector ${idx + 1}`,
                line: {
                    color: this.getColorBySimil(point.similarity),
                    width: 2,
                    dash: 'dot'
                },
                hoverinfo: 'skip',
                showlegend: false
            };
            traces.push(cachedVector);
        });
        
        // Add vectors for farthest embeddings
        farthestSubset.forEach((point, idx) => {
            const cachedVector = {
                x: [0, point.x],
                y: [0, point.y],
                mode: 'lines',
                type: 'scatter',
                name: `Farthest Vector ${idx + 1}`,
                line: {
                    color: 'rgba(150, 150, 150, 0.4)',
                    width: 1.5,
                    dash: 'dash'
                },
                hoverinfo: 'skip',
                showlegend: false
            };
            traces.push(cachedVector);
        });
        
        // Closest points trace
        if (closestSubset.length > 0) {
            const closestX = closestSubset.map(p => p.x);
            const closestY = closestSubset.map(p => p.y);
            const closestLabels = closestSubset.map(p => p.label);
            const closestSimilarities = closestSubset.map(p => p.similarity);
            const closestAngles = closestSubset.map(p => p.angle);
            
            const closestTrace = {
                x: closestX,
                y: closestY,
                mode: 'markers+text',
                type: 'scatter',
                name: `${kValue} Closest`,
                text: closestLabels,
                textposition: 'top center',
                marker: {
                    size: 12,
                    color: closestSimilarities,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {
                        title: 'Similarity',
                        thickness: 15,
                        len: 0.5
                    },
                    line: {
                        color: 'black',
                        width: 1
                    }
                },
                hovertemplate: '<b>%{text}</b><br>' +
                              'Similarity: %{marker.color:.3f}<br>' +
                              'Angle: %{customdata:.1f}°' +
                              '<extra></extra>',
                customdata: closestAngles
            };
            traces.push(closestTrace);
        }
        
        // Farthest points trace
        if (farthestSubset.length > 0) {
            const farthestX = farthestSubset.map(p => p.x);
            const farthestY = farthestSubset.map(p => p.y);
            const farthestLabels = farthestSubset.map(p => p.label);
            const farthestSimilarities = farthestSubset.map(p => p.similarity);
            const farthestAngles = farthestSubset.map(p => p.angle);
            
            const farthestTrace = {
                x: farthestX,
                y: farthestY,
                mode: 'markers+text',
                type: 'scatter',
                name: `${kValue} Farthest`,
                text: farthestLabels,
                textposition: 'bottom center',
                marker: {
                    size: 10,
                    color: farthestSimilarities,
                    colorscale: [[0, 'rgb(150, 150, 150)'], [1, 'rgb(200, 200, 200)']],
                    line: {
                        color: 'gray',
                        width: 1
                    },
                    symbol: 'x'
                },
                hovertemplate: '<b>%{text}</b><br>' +
                              'Similarity: %{marker.color:.3f}<br>' +
                              'Angle: %{customdata:.1f}°' +
                              '<extra></extra>',
                customdata: farthestAngles
            };
            traces.push(farthestTrace);
        }
        
        const layout = {
            title: {
                text: `2D Embedding Visualization - ${kValue} Closest + ${kValue} Farthest (PCA)`,
                font: { size: 16 }
            },
            xaxis: { 
                title: 'PC1 (Principal Component 1)',
                zeroline: true,
                zerolinewidth: 2,
                zerolinecolor: 'lightgray'
            },
            yaxis: { 
                title: 'PC2 (Principal Component 2)',
                zeroline: true,
                zerolinewidth: 2,
                zerolinecolor: 'lightgray'
            },
            showlegend: true,
            height: 500,
            hovermode: 'closest',
            margin: { l: 60, r: 30, t: 60, b: 60 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        };
        
        Plotly.newPlot('embedding-plot-container', traces, layout, config);
    }
    
    getColorBySimil(similarity) {
        // Return color based on similarity value
        if (similarity >= 0.85) return 'rgb(255, 0, 0)';      // Red - cache hit
        if (similarity >= 0.70) return 'rgb(255, 165, 0)';    // Orange - high
        if (similarity >= 0.50) return 'rgb(255, 255, 0)';    // Yellow - medium
        return 'rgb(0, 100, 255)';                             // Blue - low
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
