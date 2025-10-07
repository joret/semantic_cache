#!/bin/bash

# Setup script for Ollama Integration

echo "Setting up Semantic Cache with Ollama Integration"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

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

# Check if llama2 model is available
if ! ollama list | grep -q "llama2"; then
    echo "⚠️  llama2 model not found. Pulling model..."
    echo "This may take a while depending on your internet connection..."
    ollama pull llama2
    
    if [ $? -eq 0 ]; then
        echo "✅ llama2 model installed successfully"
    else
        echo "❌ Failed to install llama2 model"
        echo "You can install it manually with: ollama pull llama2"
    fi
else
    echo "✅ llama2 model is available"
fi

# Make scripts executable
chmod +x demo.py
chmod +x example.py
chmod +x cached_llm.py

echo "✅ Scripts made executable"

# Create cache directory
mkdir -p llm_cache
echo "✅ Cache directory created"

# Test the installation
echo "Testing installation..."
python3 -c "
import sys
import os
sys.path.append('..')
from cached_llm import CachedLLM

try:
    cached_llm = CachedLLM()
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
echo "Usage examples:"
echo "  python3 demo.py                    # Interactive demo"
echo "  python3 example.py                 # Simple example"
echo "  python3 cached_llm.py --help       # CLI usage"
echo ""
echo "For more information, see README.md"
