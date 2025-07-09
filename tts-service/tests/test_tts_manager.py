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
import pytest
from app.tts_manager import TTSManager

@pytest.fixture
def tts_manager():
    return TTSManager(models_dir="test_models", cache_dir="test_cache", use_cuda=False)

def test_initialization(tts_manager):
    assert tts_manager.models_dir == "test_models"
    assert tts_manager.cache_dir == "test_cache"
    assert tts_manager.use_cuda is False

def test_list_models(tts_manager):
    models = tts_manager.list_models()
    assert isinstance(models, list)
    assert len(models) > 0

def test_get_model_info(tts_manager):
    model_name = "tts_models/en/ljspeech/vits"
    model_info = tts_manager.get_model_info(model_name)
    assert model_info is not None
    assert model_info.name == model_name

def test_generate_speech(tts_manager):
    text = "Hello, this is a test."
    audio_data = tts_manager.generate_speech(text)
    assert audio_data is not None
    assert isinstance(audio_data, bytes)
