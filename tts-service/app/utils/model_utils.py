import os
import json
import logging
from typing import Dict, List, Optional, Any
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

logger = logging.getLogger(__name__)

def get_model_list() -> List[Dict[str, Any]]:
    """
    Get list of available models from TTS

    Returns:
        List[Dict]: List of available models with their details
    """
    try:
        model_manager = ModelManager()
        models = model_manager.list_models()
        # Convert string models to dictionary format
        formatted_models = []
        for model in models:
            if isinstance(model, str):
                # Parse model string format: type/language/dataset/model
                parts = model.split('/')
                if len(parts) >= 4:
                    formatted_models.append({
                        "model_name": model,
                        "type": parts[0],
                        "language": parts[1],
                        "dataset": parts[2],
                        "model": parts[3]
                    })
            elif isinstance(model, dict):
                formatted_models.append(model)
        return formatted_models
    except Exception as e:
        logger.error(f"Error getting model list: {e}")
        return []

def download_model(model_name: str, output_path: str) -> bool:
    """
    Download a model

    Args:
        model_name (str): Name of the model to download
        output_path (str): Directory to save the model

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        model_manager = ModelManager()
        model_manager.download_model(model_name)
        return True
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        return False

def get_model_info(model_name: str, models_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific model

    Args:
        model_name (str): Name of the model
        models_dir (str): Directory where models are stored

    Returns:
        Optional[Dict]: Model information if found, None otherwise
    """
    try:
        model_manager = ModelManager()
        model_info = model_manager.list_models()
        print(f"Getting from Model info: {model_info}, model_name: {model_name}")
        model_info = next(
            (m for m in model_info if (isinstance(m, dict) and m.get("model_name") == model_name) or (isinstance(m, str) and m == model_name)),
            None
        )

        if model_info:
            # Convert string model info to dictionary if needed
            if isinstance(model_info, str):
                parts = model_info.split('/')
                model_info = {
                    "model_name": model_info,
                    "type": parts[0],
                    "language": parts[1],
                    "dataset": parts[2],
                    "model": parts[3]
                }

            model_path = os.path.join(models_dir, model_name.replace('/', '--'))
            model_info["is_downloaded"] = os.path.exists(model_path)

        return model_info
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        return None

def save_model_config(model_name: str, config: Dict[str, Any], models_dir: str) -> bool:
    """
    Save model configuration

    Args:
        model_name (str): Name of the model
        config (Dict): Model configuration
        models_dir (str): Directory where models are stored

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        config_path = os.path.join(models_dir, model_name, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving model config for {model_name}: {e}")
        return False

def load_model_config(model_name: str, models_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load model configuration

    Args:
        model_name (str): Name of the model
        models_dir (str): Directory where models are stored

    Returns:
        Optional[Dict]: Model configuration if found, None otherwise
    """
    try:
        config_path = os.path.join(models_dir, model_name, "config.json")
        if not os.path.exists(config_path):
            return None

        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading model config for {model_name}: {e}")
        return None
