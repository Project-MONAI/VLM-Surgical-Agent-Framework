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

# Ensure ~/.local/bin is in PATH (for hf and other user-installed tools)
export PATH="$HOME/.local/bin:$PATH"

# Detect architecture
ARCH=$(uname -m)
echo -e "${BLUE}🔍 Detected architecture: $ARCH${NC}"

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
    echo "models/llm/Qwen2.5-VL-7B-Surg-CholecT50"
}

# Function to get served model name following precedence: ENV > global.yaml > default
get_served_model_name() {
    # First check environment variable
    if [ -n "$SERVED_MODEL_NAME" ]; then
        echo "$SERVED_MODEL_NAME"
        return
    fi

    # Then check global.yaml
    local global_config="${REPO_PATH}/configs/global.yaml"
    if [ -f "$global_config" ]; then
        # Extract quoted served_model_name from valid YAML (expects quoted values)
        local served_model_name=$(grep "^served_model_name:" "$global_config" | \
                          sed 's/^served_model_name:[[:space:]]*//' | \
                          sed 's/^"//' | sed 's/"$//')
        if [ -n "$served_model_name" ]; then
            echo "$served_model_name"
            return
        fi
    fi

    # Finally use hardcoded default
    echo "surgical-vlm"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker is running${NC}"
}

# Function to download the NVIDIA Qwen2.5-VL-7B-Surg-CholecT50 model
download_nvidia_qwen_model() {
    local model_dir="${REPO_PATH}/models/llm/Qwen2.5-VL-7B-Surg-CholecT50"

    echo -e "\n${BLUE}📥 Downloading NVIDIA Qwen2.5-VL-7B-Surg-CholecT50 model...${NC}"

    # Install Hugging Face CLI if not present
    if ! command -v hf &> /dev/null; then
        echo -e "${YELLOW}📦 Installing Hugging Face CLI...${NC}"
        pip install --upgrade huggingface-hub --user
        echo -e "${BLUE}💡 Installed to ~/.local/bin (already in PATH)${NC}"
    fi

    # Create models/llm directory with proper permissions
    if [ ! -d "${REPO_PATH}/models/llm" ]; then
        echo -e "${YELLOW}📁 Creating models/llm directory...${NC}"
        mkdir -p "${REPO_PATH}/models/llm"
    fi

    # Download the model using Hugging Face CLI
    echo -e "${YELLOW}🔄 Downloading model using Hugging Face CLI (this may take a while - ~14GB)...${NC}"
    echo -e "${BLUE}💡 Download can be resumed if interrupted${NC}"

    hf download nvidia/Qwen2.5-VL-7B-Surg-CholecT50 \
        --local-dir "$model_dir" \

    if [ -f "$model_dir/config.json" ]; then
        echo -e "${GREEN}✅ Model downloaded successfully to $model_dir${NC}"
    else
        echo -e "${RED}❌ Failed to download model${NC}"
        return 1
    fi

}

# Function to check if NVIDIA Qwen model exists and download if needed
ensure_nvidia_qwen_model() {
    local model_name=$(get_model_name)
    local model_dir="${REPO_PATH}/${model_name}"
    local model_config="${model_dir}/config.json"

    # Currently this function only handles downloading
    # NVIDIA/Qwen2.5-VL-7B-Surg-CholecT50 model from Hugging Face.
    if [[ "$model_name" != *"Qwen2.5-VL-7B-Surg-CholecT50"* ]]; then
        return 0
    fi

    if [ -f "$model_config" ]; then
        echo -e "${GREEN}✅ NVIDIA Qwen surgical model found at $model_dir${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠️  NVIDIA Qwen surgical model not found at $model_dir${NC}"
    echo -e "${BLUE}📥 Will download the model now...${NC}"
    download_nvidia_qwen_model
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
        *)
            containers="vlm-surgical-vllm vlm-surgical-whisper vlm-surgical-ui vlm-surgical-tts"
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

    # Ensure the NVIDIA Qwen surgical model is available (if needed)
    if ! ensure_nvidia_qwen_model; then
        echo -e "${RED}❌ Failed to ensure NVIDIA Qwen surgical model is available. Cannot start vLLM server.${NC}"
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
    local served_model_name=$(get_served_model_name)
    echo -e "${BLUE}💡 Using served model: $served_model_name${NC}"

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
        --quantization bitsandbytes \
        $( [[ -n "${served_model_name}" ]] && echo --served-model-name "${served_model_name}" )
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
        *)
            build_vllm
            build_whisper
            build_ui
            build_tts
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
        *)
            stop_containers
            run_vllm
            sleep 5
            run_whisper
            sleep 3
            run_tts
            sleep 2
            run_ui
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
        *)
            build_vllm
            build_whisper
            build_ui
            build_tts
            stop_containers
            run_vllm
            sleep 5
            run_whisper
            sleep 3
            run_tts
            sleep 2
            run_ui
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
    echo -e ""
    echo -e "${BLUE}  Run Commands:${NC}"
    echo -e "  $0 run                  # Run all components"
    echo -e "  $0 run vllm             # Run only vLLM server"
    echo -e "  $0 run whisper          # Run only Whisper server"
    echo -e "  $0 run ui               # Run only UI server"
    echo -e "  $0 run tts              # Run only TTS server"
    echo -e ""
    echo -e "${BLUE}  Build and Run Commands:${NC}"
    echo -e "  $0 build_and_run        # Build and run all components"
    echo -e "  $0 build_and_run vllm   # Build and run only vLLM server"
    echo -e "  $0 build_and_run whisper # Build and run only Whisper server"
    echo -e "  $0 build_and_run ui     # Build and run only UI server"
    echo -e "  $0 build_and_run tts    # Build and run only TTS server"
    echo -e ""
    echo -e "${BLUE}  Stop Commands:${NC}"
    echo -e "  $0 stop                 # Stop all containers"
    echo -e "  $0 stop vllm            # Stop only vLLM server"
    echo -e "  $0 stop whisper         # Stop only Whisper server"
    echo -e "  $0 stop ui              # Stop only UI server"
    echo -e "  $0 stop tts             # Stop only TTS server"
    echo -e ""
    echo -e "${BLUE}  Logs Commands:${NC}"
    echo -e "  $0 logs                 # Show logs for all containers"
    echo -e "  $0 logs vllm            # Show vLLM server logs"
    echo -e "  $0 logs whisper         # Show Whisper server logs"
    echo -e "  $0 logs ui              # Show UI server logs"
    echo -e "  $0 logs tts             # Show TTS server logs"
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
