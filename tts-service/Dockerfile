# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

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

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --ignore-installed --no-cache-dir -r requirements.txt && \
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
