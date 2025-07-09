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
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator

class WebSocketTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    model_name: Optional[str] = "tts_models/en/ljspeech/vits"
    speaker_name: Optional[str] = None
    language: Optional[str] = None

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v

    @validator('model_name')
    def validate_model_name(cls, v):
        if v and not v.startswith('tts_models/'):
            raise ValueError('Model name must start with tts_models/')
        return v

class WebSocketTTSResponse(BaseModel):
    status: Literal["started", "processing", "completed", "error", "disconnected"] = "started"
    message: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[float] = None  # Add progress tracking (0-100)
