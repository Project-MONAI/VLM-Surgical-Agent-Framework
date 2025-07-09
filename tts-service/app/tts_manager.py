# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
import torch
import asyncio
from typing import Dict, List, Optional, Any
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS
from .utils import (
    get_model_list,
    download_model,
    get_model_info,
    audio_to_bytes,
    bytes_to_audio,
    normalize_audio,
    resample_audio
)
from .schemas import ModelInfo
import traceback
import numpy as np
import time

DEFAULT_MODEL = "tts_models/en/ljspeech/vits"  # Changed to VITS for better quality
MAX_LOADED_MODELS = 3  # Maximum number of models to keep in memory

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, models_dir: str = "/root/.local/share/tts", cache_dir: str = "/app/cache", use_cuda: bool = True):
        self.models_dir = os.path.abspath(models_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize loaded models dictionary with LRU tracking
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_load_lock = asyncio.Lock()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model manager
        self.model_manager = ModelManager()

        # Load model list
        self.available_models = self.model_manager.list_models()

        # Initialize default model
        self._initialize_default_model()

        self.synthesizer = None
        self.current_model = None
        logger.info("TTS Manager initialized")

    def _initialize_default_model(self):
        """Initialize the default model if not already downloaded"""
        try:
            model_path = os.path.join(self.models_dir, DEFAULT_MODEL.replace('/', '--'))
            if not os.path.exists(model_path):
                self.logger.info(f"Downloading default model: {DEFAULT_MODEL}")
                self.model_manager.download_model(DEFAULT_MODEL)
                self.logger.info("Default model downloaded successfully")

            # Initialize synthesizer with the default model
            self.synthesizer = Synthesizer(
                tts_checkpoint=os.path.join(model_path, "model_file.pth"),
                tts_config_path=os.path.join(model_path, "config.json"),
                use_cuda=self.use_cuda
            )
            self.current_model = DEFAULT_MODEL
            self.logger.info("Default model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing default model: {e}")
            raise

    def list_models(self) -> List[ModelInfo]:
        """List all available models with their status"""
        models = []
        for model in self.available_models:
            # Handle both string and dictionary model formats
            if isinstance(model, str):
                model_name = model
                model_info = {}
            else:
                model_name = model.get("model_name", "")
                model_info = model

            # Use the same path as the download/load functions for consistency
            model_path = os.path.join('/root/.local/share/tts', model_name.replace('/', '--'))
            is_downloaded = os.path.exists(model_path)
            is_loaded = model_name in self.loaded_models

            models.append(ModelInfo(
                name=model_name,
                description=model_info.get("description", ""),
                language=model_info.get("language", ""),
                speakers=model_info.get("speakers", []),
                is_downloaded=is_downloaded,
                is_loaded=is_loaded,
                download_status="not_started"  # Add default download status
            ))
        return models

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model"""
        try:
            logger.debug(f"Getting info for model: {model_name}")
            # Find the model in the available models list
            model_info = None
            for m in self.available_models:
                if isinstance(m, dict) and m.get("model_name") == model_name:
                    model_info = m
                    break
                elif isinstance(m, str) and m == model_name:
                    model_info = {}
                    break

            if model_info is None:
                raise ValueError(f"Model {model_name} not found")

            # Use the same path as the download/load functions for consistency
            model_path = os.path.join('/root/.local/share/tts', model_name.replace('/', '--'))

            # Check if model is actually downloaded by verifying required files
            is_downloaded = False
            if os.path.exists(model_path):
                files = os.listdir(model_path)
                # Check for model file (.pth or .pt) and config file (.json)
                has_model_file = any(f.endswith('.pth') or f.endswith('.pt') for f in files)
                has_config_file = any(f.endswith('.json') for f in files)
                is_downloaded = has_model_file and has_config_file
                logger.debug(f"Model {model_name} files: {files}, is_downloaded: {is_downloaded}")

            is_loaded = model_name in self.loaded_models

            return ModelInfo(
                name=model_name,
                description=model_info.get("description", "") if isinstance(model_info, dict) else "",
                language=model_info.get("language", "") if isinstance(model_info, dict) else "",
                speakers=model_info.get("speakers", []) if isinstance(model_info, dict) else [],
                is_downloaded=is_downloaded,
                is_loaded=is_loaded,
                download_status="not_started"
            )
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}", exc_info=True)
            raise

    async def download_model(self, model_name: str) -> bool:
        """Download a model"""
        try:
            logger.info(f"Starting download of model: {model_name}")
            start_time = time.time()

            # Run download in a thread pool to avoid blocking
            def download():
                try:
                    # Get model info
                    model_info = None
                    for m in self.available_models:
                        if isinstance(m, dict) and m.get("model_name") == model_name:
                            model_info = m
                            break
                        elif isinstance(m, str) and m == model_name:
                            model_info = {"model_name": model_name}
                            break

                    if not model_info:
                        raise ValueError(f"Model {model_name} not found in available models")

                    logger.info(f"Downloading model {model_name}")

                    # Initialize TTS with the model name
                    # This will trigger the download if needed
                    tts = TTS(model_name=model_name, progress_bar=False)

                    # The TTS class downloads to /root/.local/share/tts/
                    # We'll use that path directly
                    model_path = os.path.join('/root/.local/share/tts', model_name.replace('/', '--'))

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model directory not found at {model_path}")

                    # List files in the model path
                    files = os.listdir(model_path)
                    logger.info(f"Files in model path: {files}")

                    if not files:
                        raise FileNotFoundError(f"No files found in {model_path}")

                except Exception as e:
                    logger.error(f"Error in download thread: {str(e)}")
                    raise

            await asyncio.get_event_loop().run_in_executor(None, download)

            # Verify the model files exist
            model_path = os.path.join('/root/.local/share/tts', model_name.replace('/', '--'))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found at {model_path}")

            # List all files in the model directory
            files = os.listdir(model_path)
            logger.info(f"Downloaded files for model {model_name}: {files}")

            if not files:
                raise FileNotFoundError(f"No files found in model directory {model_path}")

            download_time = time.time() - start_time
            logger.info(f"Model {model_name} downloaded successfully in {download_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}", exc_info=True)
            raise

    async def _load_model(self, model_name: str) -> Optional[Synthesizer]:
        """Load a model into memory with locking to prevent race conditions"""
        async with self.model_load_lock:
            if model_name in self.loaded_models:
                # Update LRU tracking
                self.loaded_models[model_name]['last_used'] = asyncio.get_event_loop().time()
                return self.loaded_models[model_name]['synthesizer']

            try:
                logger.info(f"Loading model: {model_name}")

                # Check if we need to unload some models
                if len(self.loaded_models) >= MAX_LOADED_MODELS:
                    self._unload_least_recently_used()

                # Check if model exists and download if needed
                model_path = os.path.join('/root/.local/share/tts', model_name.replace('/', '--'))
                if not os.path.exists(model_path):
                    logger.info(f"Downloading model: {model_name}")
                    await self.download_model(model_name)

                # List all files in the model directory
                files = os.listdir(model_path)
                logger.info(f"Files available for model {model_name}: {files}")

                if not files:
                    raise FileNotFoundError(f"No files found in model directory {model_path}")

                # Find model and config files
                model_file = None
                config_file = None

                for file in files:
                    if file.endswith('.pth') or file.endswith('.pt'):
                        model_file = file
                    elif file.endswith('.json'):
                        config_file = file

                if not model_file:
                    # Try to find in subdirectories
                    for root, _, filenames in os.walk(model_path):
                        for filename in filenames:
                            if filename.endswith('.pth') or filename.endswith('.pt'):
                                model_file = os.path.join(root, filename)
                                break
                        if model_file:
                            break

                if not model_file:
                    raise FileNotFoundError(f"No model file found in {model_path}")

                # Initialize synthesizer with explicit paths
                if config_file:
                    synthesizer = Synthesizer(
                        tts_checkpoint=os.path.join(model_path, model_file),
                        tts_config_path=os.path.join(model_path, config_file),
                        use_cuda=self.use_cuda
                    )
                else:
                    synthesizer = Synthesizer(
                        tts_checkpoint=os.path.join(model_path, model_file),
                        use_cuda=self.use_cuda
                    )

                # Store model with metadata
                self.loaded_models[model_name] = {
                    'synthesizer': synthesizer,
                    'last_used': asyncio.get_event_loop().time()
                }

                self.current_model = model_name
                logger.info(f"Model {model_name} loaded successfully")
                return synthesizer

            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
                raise

    def _unload_least_recently_used(self):
        """Unload the least recently used model"""
        if not self.loaded_models:
            return

        lru_model = min(
            self.loaded_models.items(),
            key=lambda x: x[1]['last_used']
        )[0]

        self.unload_model(lru_model)

    async def generate_speech(
        self,
        text: str,
        model_name: Optional[str] = None,
        speaker_name: Optional[str] = None,
        language: Optional[str] = None
    ) -> Optional[bytes]:
        """Generate speech from text"""
        try:
            logger.info(f"Generating speech for text: {text[:50]}...")
            start_time = time.time()

            # Use default model if none specified
            model_name = model_name or DEFAULT_MODEL
            logger.info(f"Using model: {model_name}")

            # Load model if needed
            try:
                tts = await self._load_model(model_name)
                if not tts:
                    logger.error(f"Failed to load model: {model_name}")
                    return None
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to load model {model_name}: {str(e)}")

            # Check if model is multi-speaker by looking for speakers.json
            model_path = os.path.join('/root/.local/share/tts', model_name.replace('/', '--'))
            speakers_file = os.path.join(model_path, 'speakers.json')

            if os.path.exists(speakers_file):
                # This is a multi-speaker model
                if not speaker_name:
                    # Try to get available speakers from the file
                    try:
                        import json
                        with open(speakers_file, 'r') as f:
                            speakers = json.load(f)
                            if isinstance(speakers, list) and speakers:
                                # For YourTTS, use a numeric speaker ID
                                speaker_name = "0"
                            elif isinstance(speakers, dict) and speakers:
                                # For other models, use the first speaker key
                                speaker_name = list(speakers.keys())[0]
                    except Exception as e:
                        logger.warning(f"Could not read speakers file: {e}")
                        # Use a generic default speaker ID
                        speaker_name = "0"
                logger.info(f"Using speaker: {speaker_name} for multi-speaker model")

            # Generate speech
            try:
                logger.info("Starting speech generation...")
                wav = tts.tts(text, speaker_name, language)
                logger.info("Speech generation completed")
            except Exception as e:
                logger.error(f"Error during TTS generation: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to generate speech: {str(e)}")

            # Convert to bytes with explicit numpy handling
            try:
                # Ensure wav is a numpy array
                if not isinstance(wav, np.ndarray):
                    wav = np.array(wav)

                # Scale and convert to int16
                wav_scaled = (wav * 32767).astype(np.int16)

                # Create proper WAV file format with headers
                import io
                import wave

                # Create a BytesIO buffer to write WAV data
                wav_buffer = io.BytesIO()

                # Write WAV file with proper headers
                with wave.open(wav_buffer, 'wb') as wav_file:
                    # Set WAV parameters
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(22050)  # 22.05 kHz sample rate
                    wav_file.writeframes(wav_scaled.tobytes())

                # Get the complete WAV file as bytes
                wav_bytes = wav_buffer.getvalue()
                wav_buffer.close()

                generation_time = time.time() - start_time
                logger.info(f"Speech generated successfully in {generation_time:.2f} seconds")

                return wav_bytes
            except Exception as e:
                logger.error(f"Error converting wav to bytes: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to convert audio to bytes: {str(e)}")

        except ValueError as e:
            logger.error(f"Value error in speech generation: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in speech generation: {str(e)}", exc_info=True)
            raise ValueError(f"Unexpected error during speech generation: {str(e)}")

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if self.use_cuda:
                torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            self.logger.info(f"Model {model_name} unloaded")

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models

        Returns:
            Dict[str, Any]: Dictionary containing available models and their details
        """
        try:
            models = self.model_manager.list_models()
            logger.debug(f"Found {len(models)} available models")
            return {
                "models": models,
                "current_model": self.current_model
            }
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}", exc_info=True)
            return {"models": [], "current_model": None}
