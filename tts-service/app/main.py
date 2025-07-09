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
import logging
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import io
import json
import os
from .tts_manager import TTSManager
from .schemas import TTSRequest, TTSResponse, ModelInfo, ModelDownloadRequest
from .websocket_schemas import WebSocketTTSRequest, WebSocketTTSResponse
import traceback
import sys
import numpy as np
import time
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech service using Coqui TTS",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TTS manager
tts_manager = TTSManager()

# Track model download status
model_download_status = {}

@app.get("/")
async def root():
    return {"message": "Welcome to Coqui TTS Service"}

@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models with their status"""
    try:
        models = tts_manager.list_models()
        # Add download status to each model
        for model in models:
            model.download_status = model_download_status.get(model.name, "not_started")
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return {"error": str(e)}

@app.get("/api/models/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str):
    """Get information about a specific model"""
    try:
        model_info = tts_manager.get_model_info(model_name)
        # Add download status
        model_info.download_status = model_download_status.get(model_name, "not_started")
        return model_info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/download")
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Download a model in the background"""
    try:
        # Check if model is already being downloaded
        if model_download_status.get(request.model_name) == "downloading":
            return {"status": "already_downloading", "message": f"Model {request.model_name} is already being downloaded"}

        # Check if model is already downloaded
        model_info = tts_manager.get_model_info(request.model_name)
        if model_info.is_downloaded:
            return {"status": "already_downloaded", "message": f"Model {request.model_name} is already downloaded"}

        # Start download in background
        model_download_status[request.model_name] = "downloading"
        background_tasks.add_task(download_model_task, request.model_name)

        return {
            "status": "started",
            "message": f"Download of model {request.model_name} has started"
        }
    except Exception as e:
        logger.error(f"Error starting model download: {e}")
        model_download_status[request.model_name] = "failed"
        raise HTTPException(status_code=500, detail=str(e))

async def download_model_task(model_name: str):
    """Background task to download a model"""
    try:
        await tts_manager.download_model(model_name)
        model_download_status[model_name] = "completed"
        logger.info(f"Model {model_name} downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        model_download_status[model_name] = "failed"

@app.get("/api/models/{model_name}/status")
async def get_model_download_status(model_name: str):
    """Get the download status of a model"""
    status = model_download_status.get(model_name, "not_started")
    return {"model_name": model_name, "status": status}

@app.post("/api/tts")
async def generate_speech(request: TTSRequest, format: str = Query("wav", enum=["wav", "json"])):
    """
    Generate speech from text

    Args:
        request: TTS request containing text and optional parameters
        format: Response format - "wav" for direct audio streaming (default), "json" for base64 encoded audio
    """
    try:
        # Validate text length
        if len(request.text) > 1000:  # Example limit
            raise HTTPException(
                status_code=400,
                detail="Text length exceeds maximum limit of 1000 characters"
            )

        # Validate model exists and is downloaded
        try:
            model_info = tts_manager.get_model_info(request.model_name)
            if not model_info.is_downloaded:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model_name} is not downloaded. Please download it first."
                )
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found"
            )

        # Generate speech with timeout
        try:
            audio_data = await asyncio.wait_for(
                tts_manager.generate_speech(
                    text=request.text,
                    model_name=request.model_name,
                    speaker_name=request.speaker_name,
                    language=request.language
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Speech generation timed out after 30 seconds"
            )

        if audio_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate speech. Please check the model configuration and try again."
            )

        if format == "wav":
            # Return WAV file directly with proper headers for web audio
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav",
                headers={
                    "Content-Length": str(len(audio_data)),
                    "Cache-Control": "no-cache",
                    "X-Content-Type-Options": "nosniff"
                }
            )
        else:
            # Return JSON with base64 encoded audio
            return TTSResponse(
                audio=audio_data,
                sample_rate=22050,  # Default sample rate for most TTS models
                model_name=request.model_name,
                speaker_name=request.speaker_name,
                language=request.language
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating speech: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection accepted")

    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {message}")

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                continue

            if "text" not in data:
                await websocket.send_json({
                    "type": "error",
                    "message": "Missing 'text' field in request"
                })
                continue

            text = data["text"]
            model = data.get("model", "tts_models/en/ljspeech/vits")

            logger.info(f"Starting speech generation for text: {text}")
            logger.debug(f"Using model: {model}")

            try:
                # Send progress update
                await websocket.send_json({
                    "type": "progress",
                    "message": "Generating speech...",
                    "progress": 0
                })

                # Generate speech with timeout
                try:
                    audio_data = await asyncio.wait_for(
                        tts_manager.generate_speech(text, model),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Speech generation timed out")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Speech generation timed out"
                    })
                    continue

                if audio_data:
                    # Send progress update
                    await websocket.send_json({
                        "type": "progress",
                        "message": "Speech generated, sending audio...",
                        "progress": 90
                    })

                    # Send audio data
                    await websocket.send_bytes(audio_data)

                    # Send completion message
                    await websocket.send_json({
                        "type": "complete",
                        "message": "Speech generation completed"
                    })
                    logger.info("Speech generation completed successfully")
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to generate speech"
                    })
                    logger.error("Failed to generate speech")

            except Exception as e:
                logger.error(f"Error during speech generation: {str(e)}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error generating speech: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8082"))
    uvicorn.run(app, host="0.0.0.0", port=port)

