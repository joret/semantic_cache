#!/bin/bash

# Installation script for semantic cache

echo "Installing Semantic Cache for AI Prompts..."
echo "=========================================="

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

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Make scripts executable
chmod +x semantic_cache.py
chmod +x examples.py
chmod +x test_cache.py

echo "✓ Scripts made executable"

# Create cache directory
mkdir -p cache
echo "✓ Cache directory created"

# Run basic test
echo "Running basic test..."
python3 test_cache.py

if [ $? -eq 0 ]; then
    echo "✓ Basic test passed"
else
    echo "✗ Basic test failed"
    exit 1
fi

echo ""
echo "Installation completed successfully!"
echo ""
echo "Usage examples:"
echo "  python3 semantic_cache.py put 'What is AI?' 'Artificial Intelligence is...'"
echo "  python3 semantic_cache.py get 'Tell me about artificial intelligence'"
echo "  python3 semantic_cache.py search 'machine learning'"
echo "  python3 examples.py"
echo ""
echo "For more information, see README.md"
