# TTS Service

A Text-to-Speech service built with FastAPI and Coqui TTS, containerized with Docker and NVIDIA GPU support.

## Features

- RESTful API for text-to-speech conversion
- Support for multiple TTS models
- GPU acceleration with NVIDIA CUDA
- Model management (list, download, info)
- Health check endpoint

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (image uses CUDA 12.8—adds Blackwell support; also supports Ampere, Hopper, and earlier architectures)
- NVIDIA Container Toolkit installed
- At least 4GB of GPU memory recommended

## System Requirements

- Ubuntu 22.04 or later
- NVIDIA GPU with CUDA 12.8 support (adds Blackwell; also supports Ampere, Hopper, and earlier)
- Docker 20.10 or later
- Docker Compose 2.0 or later
- NVIDIA Container Toolkit 1.13 or later

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tts-service
```

2. Build and run the service:
```bash
docker compose up --build
```

The service will be available at `http://localhost:8082`.

## API Endpoints

### Health Check
```bash
GET /api/health
```
Returns the health status of the service.

### List Models
```bash
GET /api/models
```
Returns a list of all available TTS models.

### Get Model Info
```bash
GET /api/models/{model_name}
```
Returns information about a specific model.

### Download Model
```bash
POST /api/models/download
{
    "model_name": "tts_models/en/ljspeech/tacotron2-DDC"
}
```
Downloads a specific model.

### Generate Speech
```bash
POST /api/tts
{
    "text": "Hello, world!",
    "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
    "speaker_name": null,
    "language": null
}
```
Generates speech from text using the specified model.

## Environment Variables

- `TTS_MODELS_DIR`: Directory to store downloaded models (default: `/app/models`)
- `TTS_CACHE_DIR`: Directory for caching (default: `/app/cache`)
- `TTS_USE_CUDA`: Whether to use CUDA for GPU acceleration (default: `true`)

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
uvicorn app.main:app --reload
```

## Performance

The service is tuned for lower latency and better throughput:

- **Non-blocking synthesis**: TTS inference runs in a dedicated thread pool so the API stays responsive and can serve health/other requests while generating speech.
- **Inference mode**: Synthesis uses `torch.inference_mode()` to disable autograd and reduce overhead.
- **TF32**: On Ampere+ GPUs (e.g. A100, RTX 30xx), TF32 is enabled for faster matmuls with no quality change.
- **GPU build**: The Docker image installs PyTorch 2.7 with CUDA 12.8 (adds Blackwell support; also supports Ampere, Hopper, and earlier architectures); ensure `nvidia-docker` or Docker with GPU support is used when running.
- **Memory**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to reduce GPU allocator fragmentation.

For even faster responses you can:

- Use a smaller/faster model (e.g. some single-speaker VITS or Tacotron2 variants).
- Keep text segments short (e.g. sentence-level) to reduce per-request synthesis time.
- Ensure the container is bound to a GPU (`--gpus all` or similar) and that no CPU-only PyTorch is installed.

## Troubleshooting

### Common Issues

1. **Model Download Issues**
   - Check internet connectivity
   - Verify sufficient disk space in models directory
   - Check model name format: `type/language/dataset/model`

2. **GPU Memory Issues**
   - Monitor GPU memory usage: `nvidia-smi`
   - Consider using a smaller model if memory is constrained
   - Adjust batch size if needed

### Logs

- Container logs: `docker logs <container name>`
- Application logs: Check the logs directory in the container

## License

Apache 2.0
