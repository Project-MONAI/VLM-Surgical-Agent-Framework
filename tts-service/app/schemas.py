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
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import base64

class ModelInfo(BaseModel):
    name: str
    description: Optional[str]
    language: Optional[str]
    speakers: Optional[List[str]]
    is_downloaded: bool
    is_loaded: bool
    download_status: Optional[str] = "not_started"

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    model_name: Optional[str] = "tts_models/en/ljspeech/vits"
    speaker_name: Optional[str] = None
    language: Optional[str] = None

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v

class TTSResponse(BaseModel):
    audio: str  # Base64 encoded audio data
    sample_rate: int
    model_name: str
    speaker_name: Optional[str] = None
    language: Optional[str] = None

    @validator('audio', pre=True)
    def encode_audio(cls, v):
        if isinstance(v, bytes):
            return base64.b64encode(v).decode('utf-8')
        return v

class ModelDownloadRequest(BaseModel):
    model_name: str = Field(..., min_length=1)

    @validator('model_name')
    def validate_model_name(cls, v):
        if not v.startswith('tts_models/'):
            raise ValueError('Model name must start with tts_models/')
        return v
