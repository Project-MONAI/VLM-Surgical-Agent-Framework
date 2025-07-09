# Host the Surgical Agent Framework in Docker Containers

This repository provides Docker containers for running the Surgical Agent Framework. The framework is split into four main components:

1. **vLLM Server**: Hosts the large language model for agent interactions
2. **Whisper Server**: Provides real-time speech-to-text capabilities
3. **UI Server**: Serves the web interface and coordinates communication
4. **TTS Server**: Provides text-to-speech voice synthesis capabilities

Each component runs in its own container for better isolation and scalability. The containers communicate over the host network for simplicity in development.

**NOTE**: The setup only works on opening browser on the host machine, not through the ssh-tunnel and broadcasting over a local network.

## Quick Start (Recommended)

Use the automated script to easily build and run all components:

```bash
cd docker
./run-surgical-agents.sh
```

This will automatically:
- Check Docker availability
- Download the surgical LLM model if needed
- Build all Docker images
- Start all containers
- Show status and available endpoints

### Script Usage

```bash
./run-surgical-agents.sh [ACTION] [COMPONENT]
```

**Available Actions:**
- `build` - Build Docker images only
- `run` - Run containers (assumes images exist)
- `build_and_run` - Build images and run containers (default)
- `download` - Download the surgical LLM model
- `stop` - Stop running containers
- `logs` - Show container logs
- `status` - Show container status
- `help` - Show help message

**Available Components (optional):**
- `vllm` - vLLM server only
- `whisper` - Whisper server only
- `ui` - UI server only
- `tts` - TTS server only
- (no component) - All components (default)

### Examples

```bash
# Build and run everything (default)
./run-surgical-agents.sh

# Build all components
./run-surgical-agents.sh build

# Run only the UI server
./run-surgical-agents.sh run ui

# Stop all containers
./run-surgical-agents.sh stop

# View logs for vLLM server
./run-surgical-agents.sh logs vllm

# Check status of all containers
./run-surgical-agents.sh status

# Download the surgical model only
./run-surgical-agents.sh download
```

### Configuration

The following environment variables can be used to customize the deployment:

**GPU_MEMORY_UTILIZATION** - Controls how much GPU memory the vLLM server uses (default: 0.25)
```bash
# Use 50% of GPU memory instead of default 25%
GPU_MEMORY_UTILIZATION=0.5 ./run-surgical-agents.sh

# Or for specific operations
GPU_MEMORY_UTILIZATION=0.8 ./run-surgical-agents.sh run vllm
```

**VLLM_ENFORCE_EAGER** - Enable enforce eager mode for vLLM execution (default: false)
```bash
# Enable enforce eager mode for debugging or compatibility
VLLM_ENFORCE_EAGER=true ./run-surgical-agents.sh run vllm

# Combine multiple environment variables
GPU_MEMORY_UTILIZATION=0.5 VLLM_ENFORCE_EAGER=true ./run-surgical-agents.sh
```

### Service Endpoints

Once running, the following services will be available:
- **vLLM Server**: http://localhost:8000 (OpenAI API compatible)
- **Whisper Server**: http://localhost:8765 (Speech-to-Text)
- **UI Server**: http://localhost:8050 (Web Interface)
- **TTS Server**: http://localhost:8082 (Text-to-Speech)

## Manual Docker Commands (Advanced)

The following manual commands are available for advanced users who prefer direct Docker control:

### vllm

- Build

```bash
git clone -b v0.8.4-dgpu git@github.com:mingxin-zheng/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t vlm-surgical-agents:vllm-openai-v0.8.3-dgpu \
  --build-arg RUN_WHEEL_CHECK=false
rm -rf vllm
```

- Download the model to `<path-to-repo>/models/llm` as the [README](../README.md) describes

- Run
```bash
docker run -it --rm --net host --gpus all \
  -v <path-to-repo>/models:/vllm-workspace/models \
  vlm-surgical-agents:vllm-openai-v0.8.3-dgpu \
  --model models/llm/Llama-3.2-11B-lora-surgical-4bit \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --disable-mm-preprocessor-cache \
  --load-format bitsandbytes \
  --quantization bitsandbytes
```

### whisper

- Build
```bash
docker build \
  -t vlm-surgical-agents:whisper-dgpu \
  -f docker/Dockerfile.whisper .
```

- Run (model will be automatically downloaded)
```bash
docker run -it --rm --gpus all --net host \
  -v <path-to-repo>/models/whisper:/root/whisper \
  vlm-surgical-agents:whisper-dgpu \
  --model_cache_dir /root/whisper
```

### UI (hosting the web services and the agent framework)

- Build
```bash
docker build -t vlm-surgical-agents:ui -f docker/Dockerfile.ui .
```

- Run
```bash
docker run -it --rm --net host vlm-surgical-agents:ui
```

You can now access the UI at http://localhost:8050

### TTS (Text-to-Speech)

The Surgical Agent Framework supports **two Text-to-Speech (TTS) options**:

1. **Local TTS Service** (Default) - runs on your hardware
2. **ElevenLabs TTS** - Cloud-based, requires API key

#### Quick TTS Setup

The TTS service is included when you run all services:

```bash
# Build and run all services (including local TTS)
./run-surgical-agents.sh

# Or run local TTS service only
./run-surgical-agents.sh run tts
```

#### Test the local TTS Integration

```bash
# Run the test script to verify everything is working
python3 ../test-tts.py
```

#### Using TTS in the Web Interface

1. Open http://localhost:8050 in your browser
2. In the "Text-to-Speech" panel:
   - ✅ Enable voice responses
   - 🎯 Select "Local TTS" (default)
3. Start a conversation and enjoy voice responses!

#### TTS Service Management

```bash
# Start TTS service
./run-surgical-agents.sh run tts

# Stop TTS service
./run-surgical-agents.sh stop tts

# View TTS logs
./run-surgical-agents.sh logs tts

# Check service status
./run-surgical-agents.sh status
```

#### Model Storage

The TTS model is stored persistently in:
- **Host Directory**: `./tts-service/models/`
- **Container Path**: `/root/.local/share/tts` (symlinked to volume)
- **Auto-download**: The model (`tts_models/en/ljspeech/vits`) downloads automatically on first use

#### TTS Service Endpoints

When running, the TTS service is available at:
- **Health Check:** http://localhost:8082/api/health
- **API Documentation:** http://localhost:8082/docs
- **Models List:** http://localhost:8082/api/models

#### Manual Docker Commands

- Build
```bash
docker build -t vlm-surgical-agents:tts -f tts-service/Dockerfile tts-service
```

- Run (models will be automatically downloaded on first use)
```bash
docker run -it --rm --gpus all --net host \
  -v <path-to-repo>/tts-service/models:/app/models \
  -v <path-to-repo>/tts-service/cache:/app/cache \
  -e TTS_MODELS_DIR=/app/models \
  -e TTS_CACHE_DIR=/app/cache \
  -e TTS_USE_CUDA=true \
  -e PORT=8082 \
  vlm-surgical-agents:tts
```

#### TTS Troubleshooting

**TTS Service Won't Start:**
```bash
# Check if port 8082 is in use
sudo netstat -tlnp | grep 8082

# Check Docker logs
./run-surgical-agents.sh logs tts
```

**No Audio Output:**
```bash
# Test the integration
python3 ../test-tts.py

# Check browser audio permissions
```

**GPU Not Detected:**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## Troubleshooting

### Check Container Status
```bash
./run-surgical-agents.sh status
```

### View Container Logs  
```bash
./run-surgical-agents.sh logs [component]
```

### Stop All Containers
```bash  
./run-surgical-agents.sh stop
```

### Rebuild Everything
```bash
./run-surgical-agents.sh stop
./run-surgical-agents.sh build_and_run
```
