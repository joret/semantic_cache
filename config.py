#!/usr/bin/env python3
"""
Configuration management for semantic cache
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration class for semantic cache settings."""
    model_name: str = 'all-MiniLM-L6-v2'
    cache_dir: str = './cache'
    similarity_threshold: float = 0.85
    max_cache_size: int = 1000
    auto_save: bool = True
    save_interval: int = 10  # Save after every N operations
    embedding_dimension: int = 384  # Default for all-MiniLM-L6-v2
    enable_logging: bool = True
    log_level: str = 'INFO'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheConfig':
        """Create config from dictionary."""
        return cls(**data)

class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_file: str = 'semantic_cache_config.json'):
        self.config_file = Path(config_file)
        self.config = CacheConfig()
    
    def load_config(self) -> CacheConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                self.config = CacheConfig.from_dict(data)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
                self.config = CacheConfig()
        else:
            logger.info("No config file found. Using defaults.")
            self.save_config()  # Create default config file
        
        return self.config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
        
        self.save_config()
    
    def get_config(self) -> CacheConfig:
        """Get current configuration."""
        return self.config
    
    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self.config = CacheConfig()
        self.save_config()
        logger.info("Configuration reset to defaults")

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> CacheConfig:
    """Get the global configuration."""
    return config_manager.get_config()

def load_config() -> CacheConfig:
    """Load configuration from file."""
    return config_manager.load_config()

def save_config() -> None:
    """Save current configuration."""
    config_manager.save_config()

def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    config_manager.update_config(**kwargs)
