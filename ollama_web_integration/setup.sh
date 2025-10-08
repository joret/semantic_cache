#!/bin/bash

# Setup script for Ollama Web Integration

echo "Setting up Semantic Cache Web Integration"
echo "========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "Please install Ollama from: https://ollama.ai"
    echo "Or run: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

echo "✅ Ollama is installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 3
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "❌ Failed to start Ollama. Please start it manually:"
        echo "   ollama serve"
        exit 1
    fi
fi

echo "✅ Ollama is running"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if llama3 model is available
if ! ollama list | grep -q "llama3"; then
    echo "⚠️  llama3 model not found. Pulling model..."
    echo "This may take a while depending on your internet connection..."
    ollama pull llama3:latest
    
    if [ $? -eq 0 ]; then
        echo "✅ llama3:latest model installed successfully"
    else
        echo "❌ Failed to install llama3:latest model"
        echo "You can install it manually with: ollama pull llama3:latest"
    fi
else
    echo "✅ llama3 model is available"
fi

# Make scripts executable
chmod +x app.py
chmod +x cached_llm_web.py

echo "✅ Scripts made executable"

# Create cache directory
mkdir -p llm_cache_web
echo "✅ Cache directory created"

# Test the installation
echo "Testing installation..."
python3 -c "
import sys
import os
sys.path.append('..')
from cached_llm_web import CachedLLMWeb

try:
    cached_llm = CachedLLMWeb()
    if cached_llm.is_llm_available():
        print('✅ Installation test passed!')
    else:
        print('❌ LLM not available')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
"

echo ""
echo "Setup completed successfully!"
echo ""
echo "Usage:"
echo "  python3 app.py                    # Start web application"
echo "  python3 cached_llm_web.py --help  # CLI usage"
echo ""
echo "Web Application:"
echo "  1. Run: python3 app.py"
echo "  2. Open: http://localhost:5000"
echo "  3. Start typing to see real-time suggestions!"
echo ""
echo "Features:"
echo "  - Real-time search suggestions as you type"
echo "  - WebSocket communication for instant updates"
echo "  - Beautiful responsive web interface"
echo "  - Live statistics and cache management"
echo "  - Mobile-friendly design"
echo ""
echo "For more information, see README.md"
