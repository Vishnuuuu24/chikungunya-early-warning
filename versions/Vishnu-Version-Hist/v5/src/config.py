"""
Configuration loader for Chikungunya EWS.
Loads YAML config and provides typed access to settings.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/config_default.yaml
        
    Returns:
        Dictionary containing all configuration settings
    """
    if config_path is None:
        # Default to project root config
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config_default.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(relative_path: str) -> Path:
    """
    Get absolute path for a data file.
    
    Args:
        relative_path: Path relative to project root (e.g., "data/raw/file.csv")
        
    Returns:
        Absolute Path object
    """
    return get_project_root() / relative_path


# Convenience: load default config on module import
try:
    CONFIG = load_config()
except FileNotFoundError:
    CONFIG = {}
