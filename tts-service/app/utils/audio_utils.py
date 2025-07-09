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
import io
import soundfile as sf
import numpy as np
from typing import Union, Tuple

def audio_to_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert numpy array audio to bytes

    Args:
        audio (np.ndarray): Audio data as numpy array
        sample_rate (int): Sample rate of the audio

    Returns:
        bytes: Audio data as bytes
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    return buffer.getvalue()

def bytes_to_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Convert bytes to numpy array audio

    Args:
        audio_bytes (bytes): Audio data as bytes

    Returns:
        Tuple[np.ndarray, int]: Audio data as numpy array and sample rate
    """
    buffer = io.BytesIO(audio_bytes)
    audio, sample_rate = sf.read(buffer)
    return audio, sample_rate

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to range [-1, 1]

    Args:
        audio (np.ndarray): Audio data as numpy array

    Returns:
        np.ndarray: Normalized audio data
    """
    return audio / np.max(np.abs(audio))

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate

    Args:
        audio (np.ndarray): Audio data as numpy array
        orig_sr (int): Original sample rate
        target_sr (int): Target sample rate

    Returns:
        np.ndarray: Resampled audio data
    """
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
