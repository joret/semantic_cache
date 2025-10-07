#!/usr/bin/env python3
"""
Ollama client for local LLM integration
"""

import requests
import json
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama local LLM server.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL of Ollama server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def generate(self, 
                 prompt: str, 
                 model: str = "llama2",
                 system: Optional[str] = None,
                 options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: The input prompt
            model: Model name to use
            system: Optional system prompt
            options: Additional model options
            
        Returns:
            Generated response text
        """
        try:
            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            if system:
                payload["system"] = system
            
            if options:
                payload["options"] = options
            
            logger.info(f"Generating response with model: {model}")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get("response", "")
            
            end_time = time.time()
            logger.info(f"Generated response in {end_time - start_time:.2f} seconds")
            
            return generated_text.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise Exception(f"Failed to generate response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Failed to generate response: {e}")
    
    def generate_stream(self, 
                       prompt: str, 
                       model: str = "llama2",
                       system: Optional[str] = None,
                       options: Optional[Dict[str, Any]] = None):
        """
        Generate streaming response from Ollama model.
        
        Args:
            prompt: The input prompt
            model: Model name to use
            system: Optional system prompt
            options: Additional model options
            
        Yields:
            Chunks of generated text
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True
            }
            
            if system:
                payload["system"] = system
            
            if options:
                payload["options"] = options
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise Exception(f"Failed to stream response: {e}")
    
    def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model: Model name to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model}")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=300  # 5 minutes for model download
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/show",
                params={"name": model},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info for {model}: {e}")
            return None

def test_ollama_connection():
    """Test function to check Ollama connection."""
    client = OllamaClient()
    
    print("Testing Ollama Connection")
    print("=" * 30)
    
    if not client.is_available():
        print("❌ Ollama server is not available")
        print("Make sure Ollama is running on http://localhost:11434")
        return False
    
    print("✅ Ollama server is available")
    
    models = client.list_models()
    if models:
        print(f"📋 Available models: {', '.join(models)}")
    else:
        print("⚠️  No models found")
    
    return True

if __name__ == "__main__":
    test_ollama_connection()
