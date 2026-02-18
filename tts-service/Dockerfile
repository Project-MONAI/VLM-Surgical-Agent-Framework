# Use NVIDIA CUDA 12.8 base for Blackwell (sm_120) and newer GPUs; cu118 only supports up to sm_90
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
# Reduce GPU memory fragmentation and improve inference latency
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    espeak-ng \
    python3 \
    python3-pip \
    libsndfile1 \
    libportaudio2 \
    libasound-dev \
    portaudio19-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# PyTorch 2.7 with CUDA 12.8 for Blackwell (sm_120) and newer GPUs
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir espeak-phonemizer

# Copy application code
COPY . .

# Create necessary directories and symlink TTS cache to volume mount
RUN mkdir -p /root/.local/share /app/models /app/cache && \
    ln -sf /app/models /root/.local/share/tts

# Expose port
EXPOSE 8082

# Run the application with WebSocket support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082", "--ws", "websockets"]
