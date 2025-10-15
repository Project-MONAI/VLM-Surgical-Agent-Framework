#!/bin/bash
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

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the absolute path of the repository (parent directory since we're in docker/)
REPO_PATH=$(dirname $(pwd))

# Ensure ~/.local/bin is in PATH (for huggingface-cli and other user-installed tools)
export PATH="$HOME/.local/bin:$PATH"

# Detect architecture
ARCH=$(uname -m)
echo -e "${BLUE}🔍 Detected architecture: $ARCH${NC}"

POLICY_IMAGE="vlm-surgical-agents:policy-runner"
POLICY_CONTAINER="vlm-surgical-policy"
POLICY_CONFIG_PATH="${REPO_PATH}/configs/policy_runner.yaml"
POLICY_RUNNER_ROBOT_PORT=""
POLICY_RUNNER_CKPT_PATH=""
POLICY_RUNNER_CKPT_DIR=""
POLICY_RUNNER_CAMERA_DEVICES=()
POLICY_RUNNER_CALIBRATION_DIR=""

# Set vLLM image based on architecture
if [[ "$ARCH" == "x86_64" ]]; then
    VLLM_IMAGE="vllm/vllm-openai:latest"
    echo -e "${BLUE}💡 Using official vLLM image for x86_64: $VLLM_IMAGE${NC}"
elif [[ "$ARCH" == "aarch64" ]]; then
    VLLM_IMAGE="vlm-surgical-agents:vllm-openai-v0.8.3-dgpu"
    echo -e "${BLUE}💡 Will build custom vLLM image for aarch64: $VLLM_IMAGE${NC}"
else
    echo -e "${YELLOW}⚠️  Unknown architecture $ARCH, defaulting to build from source${NC}"
    VLLM_IMAGE="vlm-surgical-agents:vllm-openai-v0.8.3-dgpu"
fi

echo -e "${BLUE}🏥 VLM Surgical Agent Framework Setup${NC}"
echo -e "${BLUE}======================================${NC}"

# Function to get model name following precedence: ENV > global.yaml > default
get_model_name() {
    # First check environment variable
    if [ -n "$VLLM_MODEL_NAME" ]; then
        echo "$VLLM_MODEL_NAME"
        return
    fi

    # Then check global.yaml
    local global_config="${REPO_PATH}/configs/global.yaml"
    if [ -f "$global_config" ]; then
        # Extract quoted model_name from valid YAML (expects quoted values)
        local model_name=$(grep "^model_name:" "$global_config" | \
                          sed 's/^model_name:[[:space:]]*//' | \
                          sed 's/^"//' | sed 's/"$//')
        if [ -n "$model_name" ]; then
            echo "$model_name"
            return
        fi
    fi

    # Finally use hardcoded default
    echo "models/llm/Llama-3.2-11B-Vision-Surgical-CholecT50"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker is running${NC}"
}

# Function to download the NVIDIA Llama-3.2-11B-Vision-Surgical-CholecT50 model
download_nvidia_llama_model() {
    local model_dir="${REPO_PATH}/models/llm/Llama-3.2-11B-Vision-Surgical-CholecT50"

    echo -e "\n${BLUE}📥 Downloading NVIDIA Llama-3.2-11B-Vision-Surgical-CholecT50 model...${NC}"

    # Install Hugging Face CLI if not present
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}📦 Installing Hugging Face CLI...${NC}"
        pip install --upgrade huggingface-hub --user
        echo -e "${BLUE}💡 Installed to ~/.local/bin (already in PATH)${NC}"
    fi

    # Create models/llm directory with proper permissions
    if [ ! -d "${REPO_PATH}/models/llm" ]; then
        echo -e "${YELLOW}📁 Creating models/llm directory...${NC}"
        sudo mkdir -p "${REPO_PATH}/models/llm"
        sudo chown -R $USER:$USER "${REPO_PATH}/models/llm"
    fi

    # Download the model using Hugging Face CLI
    echo -e "${YELLOW}🔄 Downloading model using Hugging Face CLI (this may take a while - ~20GB)...${NC}"
    echo -e "${BLUE}💡 Download can be resumed if interrupted${NC}"

    huggingface-cli download nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50 \
        --local-dir "$model_dir" \
        --resume-download \
        --local-dir-use-symlinks False

    if [ -f "$model_dir/config.json" ]; then
        echo -e "${GREEN}✅ Model downloaded successfully to $model_dir${NC}"
    else
        echo -e "${RED}❌ Failed to download model${NC}"
        return 1
    fi

}

# Function to check if NVIDIA Llama model exists and download if needed
ensure_nvidia_llama_model() {
    local model_name=$(get_model_name)
    local model_dir="${REPO_PATH}/${model_name}"
    local model_config="${model_dir}/config.json"

    # Currently this function only handles downloading
    # NVIDIA/Llama-3.2-11B-Vision-Surgical-CholecT50 model from Hugging Face.
    if [[ "$model_name" != *"Llama-3.2-11B-Vision-Surgical-CholecT50"* ]]; then
        return 0
    fi

    if [ -f "$model_config" ]; then
        echo -e "${GREEN}✅ NVIDIA Llama surgical model found at $model_dir${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠️  NVIDIA Llama surgical model not found at $model_dir${NC}"
    echo -e "${BLUE}📥 Will download the model now...${NC}"
    download_nvidia_llama_model
    return $?
}

# Function to build vLLM
build_vllm() {
    echo -e "\n${BLUE}🔨 Setting up vLLM Server...${NC}"

    if [[ "$ARCH" == "x86_64" ]]; then
        echo -e "${YELLOW}📥 Pulling official vLLM image for x86_64...${NC}"
        docker pull $VLLM_IMAGE
        echo -e "${GREEN}✅ vLLM image ready${NC}"
    else
        echo -e "${YELLOW}🔨 Building vLLM from source for $ARCH...${NC}"
        cd "$REPO_PATH"

        if [ ! -d "vllm" ]; then
            echo -e "${YELLOW}📥 Cloning vLLM repository...${NC}"
            git clone -b v0.8.4-dgpu https://github.com/mingxin-zheng/vllm.git
        else
            echo -e "${YELLOW}📦 vLLM repository exists, pulling latest changes...${NC}"
            cd vllm && git pull && cd ..
        fi

        cd vllm
        echo -e "${YELLOW}🔨 Building vLLM Docker image...${NC}"
        DOCKER_BUILDKIT=1 docker build . \
            --file docker/Dockerfile \
            --target vllm-openai \
            -t $VLLM_IMAGE \
            --build-arg RUN_WHEEL_CHECK=false

        echo -e "${GREEN}✅ vLLM build completed${NC}"
    fi
}

# Function to build Whisper
build_whisper() {
    echo -e "\n${BLUE}🔨 Building Whisper Server...${NC}"
    docker build \
        -t vlm-surgical-agents:whisper-dgpu \
        -f "$REPO_PATH/docker/Dockerfile.whisper" "$REPO_PATH"
    echo -e "${GREEN}✅ Whisper build completed${NC}"
}

# Function to build UI
build_ui() {
    echo -e "\n${BLUE}🔨 Building UI Server...${NC}"
    docker build -t vlm-surgical-agents:ui -f "$REPO_PATH/docker/Dockerfile.ui" "$REPO_PATH"
    echo -e "${GREEN}✅ UI build completed${NC}"
}

# Function to ensure TTS directories exist
ensure_tts_directories() {
    local tts_models_dir="${REPO_PATH}/tts-service/models"
    local tts_cache_dir="${REPO_PATH}/tts-service/cache"

    if [ ! -d "$tts_models_dir" ]; then
        echo -e "${YELLOW}📁 Creating TTS models directory...${NC}"
        mkdir -p "$tts_models_dir"
    fi

    if [ ! -d "$tts_cache_dir" ]; then
        echo -e "${YELLOW}📁 Creating TTS cache directory...${NC}"
        mkdir -p "$tts_cache_dir"
    fi

    echo -e "${GREEN}✅ TTS directories ready${NC}"
}

# Function to build TTS service
build_tts() {
    echo -e "\n${BLUE}🔨 Building TTS Server...${NC}"
    ensure_tts_directories
    docker build -t vlm-surgical-agents:tts -f "$REPO_PATH/tts-service/Dockerfile" "$REPO_PATH/tts-service"
    echo -e "${GREEN}✅ TTS build completed${NC}"
}

build_policy_runner() {
    echo -e "\n${BLUE}🔨 Building Policy Runner...${NC}"
    docker build -t ${POLICY_IMAGE} -f "$REPO_PATH/docker/Dockerfile.policy_runner" "$REPO_PATH"
    echo -e "${GREEN}✅ Policy Runner build completed${NC}"
}

validate_policy_config() {
    local config_path="${1:-$POLICY_CONFIG_PATH}"

    if [ ! -f "$config_path" ]; then
        echo -e "${RED}❌ Policy runner config not found at $config_path${NC}"
        return 1
    fi

    local python_output
    if ! python_output=$(
        python3 - "$config_path" "$REPO_PATH" 2>&1 <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyYAML is required to validate the policy runner config. Install it with 'pip install pyyaml'."
    ) from exc


config_path = Path(sys.argv[1]).expanduser()
repo_path = Path(sys.argv[2]).expanduser()

if not config_path.is_file():
    raise FileNotFoundError(f"Config file '{config_path}' does not exist.")

with config_path.open("r", encoding="utf-8") as fh:
    config = yaml.safe_load(fh)

if config is None:
    raise ValueError(f"Config file '{config_path}' is empty.")

def check_for_placeholders(node, path="root"):
    if isinstance(node, dict):
        for key, value in node.items():
            check_for_placeholders(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            check_for_placeholders(value, f"{path}[{idx}]")
    elif isinstance(node, str):
        if "__REQUIRED__" in node:
            raise ValueError(f"Config value at {path} contains placeholder '__REQUIRED__'.")

check_for_placeholders(config)

ckpt_path = Path(config.get("ckpt_path", "")).expanduser()
if not ckpt_path.is_absolute():
    ckpt_path = (repo_path / ckpt_path).resolve()

robot_cfg = config.get("robot") or {}
robot_port = robot_cfg.get("port")
if not robot_port:
    raise ValueError("robot.port must be set in the policy config.")

if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint '{ckpt_path}' does not exist.")

# Extract camera device paths
camera_devices = []
cameras_cfg = robot_cfg.get("cameras", {})
for cam_name, cam_cfg in cameras_cfg.items():
    if isinstance(cam_cfg, dict):
        index_or_path = cam_cfg.get("index_or_path")
        if index_or_path and isinstance(index_or_path, str) and index_or_path.startswith("/dev/"):
            camera_devices.append(index_or_path)

# Extract calibration directory
calibration_dir = robot_cfg.get("calibration_dir")
if calibration_dir:
    calibration_dir = Path(calibration_dir).expanduser()

print(f"robot_port={robot_port}")
print(f"ckpt_path={ckpt_path}")
print(f"ckpt_dir={ckpt_path.parent}")
for cam_dev in camera_devices:
    print(f"camera_device={cam_dev}")
if calibration_dir:
    print(f"calibration_dir={calibration_dir}")
PY
    ); then
        echo -e "${RED}❌ Policy runner config validation failed:${NC}"
        echo "$python_output"
        return 1
    fi

    POLICY_RUNNER_ROBOT_PORT=""
    POLICY_RUNNER_CKPT_PATH=""
    POLICY_RUNNER_CKPT_DIR=""
    POLICY_RUNNER_CAMERA_DEVICES=()
    POLICY_RUNNER_CALIBRATION_DIR=""

    while IFS= read -r line; do
        case "$line" in
            robot_port=*)
                POLICY_RUNNER_ROBOT_PORT="${line#robot_port=}"
                ;;
            ckpt_path=*)
                POLICY_RUNNER_CKPT_PATH="${line#ckpt_path=}"
                ;;
            ckpt_dir=*)
                POLICY_RUNNER_CKPT_DIR="${line#ckpt_dir=}"
                ;;
            camera_device=*)
                POLICY_RUNNER_CAMERA_DEVICES+=("${line#camera_device=}")
                ;;
            calibration_dir=*)
                POLICY_RUNNER_CALIBRATION_DIR="${line#calibration_dir=}"
                ;;
        esac
    done <<< "$python_output"

    if [ -z "$POLICY_RUNNER_ROBOT_PORT" ]; then
        echo -e "${RED}❌ Unable to determine robot port from policy config${NC}"
        return 1
    fi

    if [ -z "$POLICY_RUNNER_CKPT_PATH" ]; then
        echo -e "${RED}❌ Unable to determine checkpoint path from policy config${NC}"
        return 1
    fi

    if [ -z "$POLICY_RUNNER_CKPT_DIR" ]; then
        POLICY_RUNNER_CKPT_DIR=$(dirname "$POLICY_RUNNER_CKPT_PATH")
    fi

    echo -e "${GREEN}✅ Policy runner config validated${NC}"
    return 0
}

run_policy_runner() {
    echo -e "\n${BLUE}🚀 Starting Policy Runner...${NC}"
    if ! validate_policy_config "$POLICY_CONFIG_PATH"; then
        return 1
    fi

    local docker_cmd=(
        docker run -d
        --name "${POLICY_CONTAINER}"
        --net host
        --gpus all
        -v "${REPO_PATH}:${REPO_PATH}"
        -v "${POLICY_CONFIG_PATH}:/workspace/configs/policy_runner.yaml:ro"
        -e POLICY_RUNNER_CONFIG=/workspace/configs/policy_runner.yaml
    )



    if [ -n "$POLICY_RUNNER_ROBOT_PORT" ]; then
        docker_cmd+=(--device "${POLICY_RUNNER_ROBOT_PORT}:${POLICY_RUNNER_ROBOT_PORT}")
    fi

    # Add camera devices
    for cam_device in "${POLICY_RUNNER_CAMERA_DEVICES[@]}"; do
        if [ -e "$cam_device" ]; then
            docker_cmd+=(--device "${cam_device}:${cam_device}")
        else
            echo -e "${YELLOW}⚠️  Warning: Camera device ${cam_device} not found${NC}"
        fi
    done

    if [ -n "$POLICY_RUNNER_CKPT_DIR" ]; then
        docker_cmd+=(-v "${POLICY_RUNNER_CKPT_DIR}:${POLICY_RUNNER_CKPT_DIR}:ro")
    fi

    # Mount calibration directory if specified in config
    if [ -n "$POLICY_RUNNER_CALIBRATION_DIR" ]; then
        local host_calibration_dir="$POLICY_RUNNER_CALIBRATION_DIR"
        # Transform /home/<username>/ to /root/ for container path
        local container_calibration_dir=$(echo "$host_calibration_dir" | sed -E 's|^/home/[^/]+/|/root/|')
        
        if [ -d "$host_calibration_dir" ]; then
            docker_cmd+=(-v "${host_calibration_dir}:${container_calibration_dir}:ro")
            echo -e "${GREEN}✅ Mounting calibration directory: ${host_calibration_dir} -> ${container_calibration_dir}${NC}"
        else
            echo -e "${YELLOW}⚠️  Warning: Calibration directory not found at ${host_calibration_dir}${NC}"
        fi
    fi

    docker_cmd+=("${POLICY_IMAGE}")

    "${docker_cmd[@]}"
    echo -e "${GREEN}✅ Policy Runner started${NC}"
}

# Function to stop containers
stop_containers() {
    local component="$1"
    local containers

    case "$component" in
        vllm)
            containers="vlm-surgical-vllm"
            ;;
        whisper)
            containers="vlm-surgical-whisper"
            ;;
        ui)
            containers="vlm-surgical-ui"
            ;;
        tts)
            containers="vlm-surgical-tts"
            ;;
        policy)
            containers="${POLICY_CONTAINER}"
            ;;
        *)
            containers="vlm-surgical-vllm vlm-surgical-whisper vlm-surgical-ui vlm-surgical-tts ${POLICY_CONTAINER}"
            ;;
    esac

    echo -e "\n${YELLOW}🛑 Stopping containers: $containers${NC}"
    for container in $containers; do
        docker stop $container 2>/dev/null && echo -e "${GREEN}✅ Stopped $container${NC}" || echo -e "${YELLOW}⚠️  $container not running${NC}"
        docker rm $container 2>/dev/null || true
    done
}

# Function to run vLLM server
run_vllm() {
    echo -e "\n${BLUE}🚀 Starting vLLM Server...${NC}"

    # Ensure the NVIDIA Llama surgical model is available (if needed)
    if ! ensure_nvidia_llama_model; then
        echo -e "${RED}❌ Failed to ensure NVIDIA Llama surgical model is available. Cannot start vLLM server.${NC}"
        return 1
    fi

    # Set default GPU memory utilization if not provided
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.25}
    echo -e "${BLUE}💡 Using GPU memory utilization: ${GPU_MEMORY_UTILIZATION}${NC}"

    # Set enforce eager mode if requested
    VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}
    ENFORCE_EAGER_FLAG=""
    if [[ "${VLLM_ENFORCE_EAGER,,}" == "true" ]]; then
        ENFORCE_EAGER_FLAG="--enforce-eager"
        echo -e "${BLUE}💡 Using enforce eager mode${NC}"
    fi

    # Get model name following precedence: ENV > global.yaml > default
    local model_name=$(get_model_name)
    echo -e "${BLUE}💡 Using model: $model_name${NC}"

    docker run -d \
        --name vlm-surgical-vllm \
        --net host \
        --gpus all \
        -v ${REPO_PATH}/models:/vllm-workspace/models \
        -e VLLM_MODEL_NAME \
        -e VLLM_URL \
        --restart unless-stopped \
        $VLLM_IMAGE \
        --model $model_name \
        --served-model-name surgical-vlm \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        ${ENFORCE_EAGER_FLAG} \
        --max-model-len 4096 \
        --max-num-seqs 8 \
        --disable-mm-preprocessor-cache \
        --load-format bitsandbytes \
        --quantization bitsandbytes
    echo -e "${GREEN}✅ vLLM Server started${NC}"
}

# Function to run Whisper server
run_whisper() {
    echo -e "\n${BLUE}🚀 Starting Whisper Server...${NC}"
    
    # Set default Whisper port if not provided
    WHISPER_PORT=${WHISPER_PORT:-43001}
    echo -e "${BLUE}💡 Using Whisper port: ${WHISPER_PORT}${NC}"
    
    docker run -d \
        --name vlm-surgical-whisper \
        --gpus all \
        --net host \
        -v ${REPO_PATH}/models/whisper:/root/whisper \
        -e WHISPER_PORT=${WHISPER_PORT} \
        --restart unless-stopped \
        vlm-surgical-agents:whisper-dgpu \
        --model_cache_dir /root/whisper
    echo -e "${GREEN}✅ Whisper Server started${NC}"
}

# Function to run UI server
run_ui() {
    echo -e "\n${BLUE}🚀 Starting UI Server...${NC}"
    docker run -d \
        --name vlm-surgical-ui \
        --net host \
        -e VLLM_MODEL_NAME \
        -e VLLM_URL \
        --restart unless-stopped \
        vlm-surgical-agents:ui
    echo -e "${GREEN}✅ UI Server started${NC}"
}

# Function to run TTS server
run_tts() {
    echo -e "\n${BLUE}🚀 Starting TTS Server...${NC}"
    ensure_tts_directories
    docker run -d \
        --name vlm-surgical-tts \
        --net host \
        --gpus all \
        -v ${REPO_PATH}/tts-service/models:/app/models \
        -v ${REPO_PATH}/tts-service/cache:/app/cache \
        -e TTS_MODELS_DIR=/app/models \
        -e TTS_CACHE_DIR=/app/cache \
        -e TTS_USE_CUDA=true \
        -e PORT=8082 \
        --restart unless-stopped \
        vlm-surgical-agents:tts
    echo -e "${GREEN}✅ TTS Server started${NC}"
}

# Function to show status
show_status() {
    echo -e "\n${BLUE}📊 Container Status:${NC}"
    echo -e "${BLUE}==================${NC}"

    # Show container status with more useful info
    local containers=$(docker ps --filter "name=vlm-surgical" --format "{{.Names}}" 2>/dev/null)

    if [ -z "$containers" ]; then
        echo "No containers found"
    else
        docker ps --filter "name=vlm-surgical" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null

        echo -e "\n${BLUE}📡 Service Endpoints:${NC}"
        echo -e "${BLUE}====================${NC}"

        # Check vLLM status
        local vllm_status=$(docker ps --filter "name=vlm-surgical-vllm" --format "{{.Status}}" 2>/dev/null)
        if [[ "$vllm_status" =~ ^Up ]]; then
            echo -e "${GREEN}✅ vLLM Server:${NC} http://localhost:8000 (OpenAI API) - $vllm_status"
        elif [ -n "$vllm_status" ]; then
            echo -e "${YELLOW}⚠️  vLLM Server:${NC} $vllm_status"
        else
            echo -e "${RED}❌ vLLM Server:${NC} Not found"
        fi

        # Check Whisper status
        local whisper_status=$(docker ps --filter "name=vlm-surgical-whisper" --format "{{.Status}}" 2>/dev/null)
        if [[ "$whisper_status" =~ ^Up ]]; then
            echo -e "${GREEN}✅ Whisper Server:${NC} http://localhost:8765 (Speech-to-Text) - $whisper_status"
        elif [ -n "$whisper_status" ]; then
            echo -e "${YELLOW}⚠️  Whisper Server:${NC} $whisper_status"
        else
            echo -e "${RED}❌ Whisper Server:${NC} Not found"
        fi

        # Check UI status
        local ui_status=$(docker ps --filter "name=vlm-surgical-ui" --format "{{.Status}}" 2>/dev/null)
        if [[ "$ui_status" =~ ^Up ]]; then
            echo -e "${GREEN}✅ UI Server:${NC} http://localhost:8050 (Web Interface) - $ui_status"
        elif [ -n "$ui_status" ]; then
            echo -e "${YELLOW}⚠️  UI Server:${NC} $ui_status"
        else
            echo -e "${RED}❌ UI Server:${NC} Not found"
        fi

        # Check TTS status
        local tts_status=$(docker ps --filter "name=vlm-surgical-tts" --format "{{.Status}}" 2>/dev/null)
        if [[ "$tts_status" =~ ^Up ]]; then
            echo -e "${GREEN}✅ TTS Server:${NC} http://localhost:8082 (Text-to-Speech) - $tts_status"
        elif [ -n "$tts_status" ]; then
            echo -e "${YELLOW}⚠️  TTS Server:${NC} $tts_status"
        else
            echo -e "${RED}❌ TTS Server:${NC} Not found"
        fi

        # Check Policy runner status
        local policy_status=$(docker ps --filter "name=${POLICY_CONTAINER}" --format "{{.Status}}" 2>/dev/null)
        if [[ "$policy_status" =~ ^Up ]]; then
            echo -e "${GREEN}✅ Policy Runner:${NC} Uses config ${POLICY_CONFIG_PATH} - $policy_status"
        elif [ -n "$policy_status" ]; then
            echo -e "${YELLOW}⚠️  Policy Runner:${NC} $policy_status"
        else
            echo -e "${RED}❌ Policy Runner:${NC} Not found"
        fi
    fi

    echo -e "\n${YELLOW}📝 Useful commands:${NC}"
    echo -e "  View logs: ./run-surgical-agents.sh logs [component]"
    echo -e "  Stop all:  ./run-surgical-agents.sh stop"
    echo -e "  Start all: ./run-surgical-agents.sh build_and_run"
}

# Function to show logs
show_logs() {
    local component="$1"
    case "$component" in
        vllm)
            echo -e "${BLUE}📋 vLLM Server Logs:${NC}"
            if docker ps -a --filter "name=vlm-surgical-vllm" --format "{{.Names}}" | grep -q "vlm-surgical-vllm"; then
                docker logs vlm-surgical-vllm --tail 50
            else
                echo "vLLM container not found"
            fi
            ;;
        whisper)
            echo -e "${BLUE}📋 Whisper Server Logs:${NC}"
            if docker ps -a --filter "name=vlm-surgical-whisper" --format "{{.Names}}" | grep -q "vlm-surgical-whisper"; then
                docker logs vlm-surgical-whisper --tail 50
            else
                echo "Whisper container not found"
            fi
            ;;
        ui)
            echo -e "${BLUE}📋 UI Server Logs:${NC}"
            if docker ps -a --filter "name=vlm-surgical-ui" --format "{{.Names}}" | grep -q "vlm-surgical-ui"; then
                docker logs vlm-surgical-ui --tail 50
            else
                echo "UI container not found"
            fi
            ;;
        tts)
            echo -e "${BLUE}📋 TTS Server Logs:${NC}"
            if docker ps -a --filter "name=vlm-surgical-tts" --format "{{.Names}}" | grep -q "vlm-surgical-tts"; then
                docker logs vlm-surgical-tts --tail 50
            else
                echo "TTS container not found"
            fi
            ;;
        policy)
            echo -e "${BLUE}📋 Policy Runner Logs:${NC}"
            if docker ps -a --filter "name=${POLICY_CONTAINER}" --format "{{.Names}}" | grep -q "${POLICY_CONTAINER}"; then
                docker logs ${POLICY_CONTAINER} --tail 50
            else
                echo "Policy runner container not found"
            fi
            ;;
        *)
            echo -e "${BLUE}📋 All Container Logs:${NC}"
            echo -e "${BLUE}--- vLLM Logs ---${NC}"
            if docker ps -a --filter "name=vlm-surgical-vllm" --format "{{.Names}}" | grep -q "vlm-surgical-vllm"; then
                docker logs vlm-surgical-vllm --tail 30 | head -20
            else
                echo "vLLM container not found"
            fi
            echo -e "\n${BLUE}--- Whisper Logs ---${NC}"
            if docker ps -a --filter "name=vlm-surgical-whisper" --format "{{.Names}}" | grep -q "vlm-surgical-whisper"; then
                docker logs vlm-surgical-whisper --tail 30 | head -20
            else
                echo "Whisper container not found"
            fi
            echo -e "\n${BLUE}--- UI Logs ---${NC}"
            if docker ps -a --filter "name=vlm-surgical-ui" --format "{{.Names}}" | grep -q "vlm-surgical-ui"; then
                docker logs vlm-surgical-ui --tail 30 | head -20
            else
                echo "UI container not found"
            fi
            echo -e "\n${BLUE}--- TTS Logs ---${NC}"
            if docker ps -a --filter "name=vlm-surgical-tts" --format "{{.Names}}" | grep -q "vlm-surgical-tts"; then
                docker logs vlm-surgical-tts --tail 30 | head -20
            else
                echo "TTS container not found"
            fi
            echo -e "\n${BLUE}--- Policy Runner Logs ---${NC}"
            if docker ps -a --filter "name=${POLICY_CONTAINER}" --format "{{.Names}}" | grep -q "${POLICY_CONTAINER}"; then
                docker logs ${POLICY_CONTAINER} --tail 30 | head -20
            else
                echo "Policy runner container not found"
            fi
            ;;
    esac
}

# Function to handle build command
handle_build() {
    local component="$1"
    check_docker

    case "$component" in
        vllm)
            build_vllm
            ;;
        whisper)
            build_whisper
            ;;
        ui)
            build_ui
            ;;
        tts)
            build_tts
            ;;
        policy)
            build_policy_runner
            ;;
        *)
            build_vllm
            build_whisper
            build_ui
            build_tts
            build_policy_runner
            echo -e "\n${GREEN}✅ All images built successfully!${NC}"
            ;;
    esac
}

# Function to handle run command
handle_run() {
    local component="$1"
    check_docker

    case "$component" in
        vllm)
            stop_containers "vllm"
            run_vllm
            ;;
        whisper)
            stop_containers "whisper"
            run_whisper
            ;;
        ui)
            stop_containers "ui"
            run_ui
            ;;
        tts)
            stop_containers "tts"
            run_tts
            ;;
        policy)
            stop_containers "policy"
            run_policy_runner
            ;;
        *)
            stop_containers
            run_vllm
            sleep 5
            run_whisper
            sleep 3
            run_tts
            sleep 2
            run_ui
            sleep 2
            run_policy_runner
            show_status
            ;;
    esac
}

# Function to handle build_and_run command
handle_build_and_run() {
    local component="$1"
    check_docker

    case "$component" in
        vllm)
            build_vllm
            stop_containers "vllm"
            run_vllm
            echo -e "${GREEN}✅ vLLM built and started${NC}"
            ;;
        whisper)
            build_whisper
            stop_containers "whisper"
            run_whisper
            echo -e "${GREEN}✅ Whisper built and started${NC}"
            ;;
        ui)
            build_ui
            stop_containers "ui"
            run_ui
            echo -e "${GREEN}✅ UI built and started${NC}"
            ;;
        tts)
            build_tts
            stop_containers "tts"
            run_tts
            echo -e "${GREEN}✅ TTS built and started${NC}"
            ;;
        policy)
            build_policy_runner
            stop_containers "policy"
            run_policy_runner
            echo -e "${GREEN}✅ Policy runner built and started${NC}"
            ;;
        *)
            build_vllm
            build_whisper
            build_ui
            build_tts
            build_policy_runner
            stop_containers
            run_vllm
            sleep 5
            run_whisper
            sleep 3
            run_tts
            sleep 2
            run_ui
            sleep 2
            run_policy_runner
            show_status
            ;;
    esac
}

# Function to show help
show_help() {
    echo -e "${BLUE}Usage: $0 [ACTION] [COMPONENT]${NC}"
    echo -e ""
    echo -e "${YELLOW}ACTIONS:${NC}"
    echo -e "  build          Build images"
    echo -e "  run            Run containers (assumes images exist)"
    echo -e "  build_and_run  Build images and run containers"
    echo -e "  download       Download the surgical LLM model"
    echo -e "  stop           Stop running containers"
    echo -e "  logs           Show container logs"
    echo -e "  status         Show container status"
    echo -e "  help           Show this help message"
    echo -e ""
    echo -e "${YELLOW}COMPONENTS (optional):${NC}"
    echo -e "  vllm           vLLM server only"
    echo -e "  whisper        Whisper server only"
    echo -e "  ui             UI server only"
    echo -e "  tts            TTS server only"
    echo -e "  policy         Policy runner only"
    echo -e "  (no component) All components (default)"
    echo -e ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "${BLUE}  Default (no arguments):${NC}"
    echo -e "  $0                      # Build and run all components"
    echo -e ""
    echo -e "${BLUE}  Build Commands:${NC}"
    echo -e "  $0 build                # Build all components"
    echo -e "  $0 build vllm           # Build only vLLM server"
    echo -e "  $0 build whisper        # Build only Whisper server"
    echo -e "  $0 build ui             # Build only UI server"
    echo -e "  $0 build tts            # Build only TTS server"
    echo -e "  $0 build policy         # Build only Policy runner"
    echo -e ""
    echo -e "${BLUE}  Run Commands:${NC}"
    echo -e "  $0 run                  # Run all components"
    echo -e "  $0 run vllm             # Run only vLLM server"
    echo -e "  $0 run whisper          # Run only Whisper server"
    echo -e "  $0 run ui               # Run only UI server"
    echo -e "  $0 run tts              # Run only TTS server"
    echo -e "  $0 run policy           # Run only Policy runner"
    echo -e ""
    echo -e "${BLUE}  Build and Run Commands:${NC}"
    echo -e "  $0 build_and_run        # Build and run all components"
    echo -e "  $0 build_and_run vllm   # Build and run only vLLM server"
    echo -e "  $0 build_and_run whisper # Build and run only Whisper server"
    echo -e "  $0 build_and_run ui     # Build and run only UI server"
    echo -e "  $0 build_and_run tts    # Build and run only TTS server"
    echo -e "  $0 build_and_run policy # Build and run only Policy runner"
    echo -e ""
    echo -e "${BLUE}  Stop Commands:${NC}"
    echo -e "  $0 stop                 # Stop all containers"
    echo -e "  $0 stop vllm            # Stop only vLLM server"
    echo -e "  $0 stop whisper         # Stop only Whisper server"
    echo -e "  $0 stop ui              # Stop only UI server"
    echo -e "  $0 stop tts             # Stop only TTS server"
    echo -e "  $0 stop policy          # Stop only Policy runner"
    echo -e ""
    echo -e "${BLUE}  Logs Commands:${NC}"
    echo -e "  $0 logs                 # Show logs for all containers"
    echo -e "  $0 logs vllm            # Show vLLM server logs"
    echo -e "  $0 logs whisper         # Show Whisper server logs"
    echo -e "  $0 logs ui              # Show UI server logs"
    echo -e "  $0 logs tts             # Show TTS server logs"
    echo -e "  $0 logs policy          # Show Policy runner logs"
    echo -e ""
    echo -e "${BLUE}  Download Command:${NC}"
    echo -e "  $0 download             # Download surgical LLM model"
    echo -e ""
    echo -e "${BLUE}  Status Command:${NC}"
    echo -e "  $0 status               # Show all container status"
    echo -e ""
    echo -e "${YELLOW}ENVIRONMENT VARIABLES:${NC}"
    echo -e "  GPU_MEMORY_UTILIZATION  Set GPU memory utilization for vLLM (default: 0.25)"
    echo -e "                          Example: GPU_MEMORY_UTILIZATION=0.5 $0 run vllm"
    echo -e "  VLLM_ENFORCE_EAGER      Enable enforce eager mode for vLLM (default: false)"
    echo -e "                          Example: VLLM_ENFORCE_EAGER=true $0 run vllm"
}

# Parse command line arguments
ACTION="${1:-build_and_run}"
COMPONENT="${2:-}"

case "$ACTION" in
    build)
        handle_build "$COMPONENT"
        ;;
    run)
        handle_run "$COMPONENT"
        ;;
    build_and_run)
        handle_build_and_run "$COMPONENT"
        ;;
    download)
        download_surgical_model
        ;;
    stop)
        stop_containers "$COMPONENT"
        echo -e "${GREEN}✅ Containers stopped${NC}"
        ;;
    logs)
        show_logs "$COMPONENT"
        ;;
    status)
        show_status
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown action: $ACTION${NC}"
        echo -e "${YELLOW}💡 Run '$0 help' to see available actions${NC}"
        exit 1
        ;;
esac
