# Surgical Agentic Framework Demo

The Surgical Agentic Framework Demo is a multimodal agentic AI framework tailored for surgical procedures. It supports:

* **Speech-to-Text**: Real-time audio is captured, transcribed by Whisper.
* **VLM/LLM-based Conversational Agents**: A *selector agent* decides which specialized agent to invoke:
    *   ChatAgent for general Q&A,
    *   NotetakerAgent to record specific notes,
    *   AnnotationAgent to automatically annotate progress in the background,
    *   PostOpNoteAgent to summarize all data into a final post-operative note.
* **Robot Control Agent**: Real-world robot control via the GR00T N1.5 policy runner for executing scrub nurse tasks.
* **Text-to-Speech**: The system can speak back the AI's response if you enable TTS. There are options for local TTS models (Coqui), as well as an ElevenLabs API.
* **Computer Vision** or multimodal features are supported via a finetuned VLM (Vision Language Model), launched by vLLM.
* **Video Upload and Processing**: Support for uploading and analyzing surgical videos.
* **Live Streaming (WebRTC)**: Real-time analysis of live surgical streams via WebRTC with seamless mode switching between uploaded videos and live streams.
* **Post-Operation Note Generation**: Automatic generation of structured post-operative notes based on the procedure data.


## System Flow and Agent Overview

1. Microphone: The user clicks "Start Mic" in the web UI, or types a question.
2. Whisper ASR: Transcribes speech into text (via servers/whisper_online_server.py).
3. SelectorAgent: Receives text from the UI, corrects it (if needed), decides whether to direct it to:
    * ChatAgent (general Q&A about the procedure)
    * NotetakerAgent (records a note with timestamp + optional image frame)
    * RobotControlAgent (executes real-world robot manipulation tasks via the GR00T N1.5 policy runner)
    * In the background, AnnotationAgent is also generating structured "annotations" every 10 seconds.
4. NotetakerAgent: If chosen, logs the note in a JSON file.
5. AnnotationAgent: Runs automatically, storing procedure annotations in ```procedure_..._annotations.json```.
6. RobotControlAgent: Runs policy inference on hardware, controlling physical robots for scrub nurse tasks.
7. PostOpNoteAgent (optional final step): Summarizes the entire procedure, reading from both the annotation JSON and the notetaker JSON, producing a final structured post-op note.

## System Requirements

* Python 3.12 or higher
* Node.js 14.x or higher
* CUDA-compatible GPU (recommended) for model inference
* Microphone for voice input (optional)
* 16GB+ VRAM recommended

## Installation 

1. Clone or Download this repository:

```
git clone https://github.com/project-monai/vlm-surgical-agent-framework.git
cd VLM-Surgical-Agent-Framework
```

2. Setup vLLM (Optional)

vLLM is already configured in the project scripts. If you need to set up a custom vLLM server, see https://docs.vllm.ai/en/latest/getting_started/installation.html

3. Install Dependencies:

```
conda create -n surgical_agent_framework python=3.12
conda activate surgical_agent_framework
pip install -r requirements.txt
```

Note for Linux (PyAudio build): If `pip install pyaudio` fails with a missing header error like `portaudio.h: No such file or directory`, install the PortAudio development package first, then rerun pip install:

```
sudo apt-get update && sudo apt-get install -y portaudio19-dev
pip install -r requirements.txt
```

4. Install Node.js dependencies (for UI development):

Before installing, verify your Node/npm versions (Node ≥14; 18 LTS recommended):

```
node -v && npm -v
```

```
npm install
```

5. Models Folder:

* Where to put things

    * LLM checkpoints live in models/llm/
    * Whisper (speech‑to‑text) checkpoints live in models/whisper/ (they will be downloaded automatically at runtime the first time you invoke Whisper).

* Default LLM
    * This repository is pre‑configured for [NVIDIA Qwen2.5-VL-7B-Surg-CholecT50](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Surg-CholecT50), a surgical‑domain fine‑tuned variant of Qwen2.5-VL-7B. You may choose to replace it with a finetuned VLM of your choosing.

Download the default model from Hugging Face with Git LFS:

```
# Download the checkpoint into the expected folder
hf download nvidia/Qwen2.5-VL-7B-Surg-CholecT50 \
  --local-dir models/llm/Qwen2.5-VL-7B-Surg-CholecT50 \
  --local-dir-use-symlinks False     
```

* Serving engine
    * All LLMs are served through vLLM for streaming. Change the model path once in `configs/global.yaml` under `model_name` — both the agents and `scripts/run_vllm_server.sh` read this. You can override at runtime with `VLLM_MODEL_NAME`. To enable auto‑download when the folder is missing, set `model_repo` in `configs/global.yaml` (or export `MODEL_REPO`).

* Resulting folder layout

```
models/
  ├── llm/
  │   └── Qwen2.5-VL-7B-Surg-CholecT50/   ← LLM model files
  └── whisper/                            ← Whisper models (auto‑downloaded)
```

### Fine‑Tuning Your Own Surgical Model

If you want to adapt the framework to a different procedure (e.g., appendectomy, colectomy), you can fine‑tune a VLM and plug it into this stack with only config file changes. See:

- [FINETUNE.md](FINETUNE.md) — end‑to‑end guide covering:
  - Data curation and scene metadata
  - Visual‑instruction data generation (teacher–student)
  - Packing data in LLaVA‑style format
  - Training (LoRA/QLoRA) and validation
  - Exporting and serving with vLLM, and updating configs

6. Setup: 

* Edit ```scripts/start_app.sh``` if you need to change ports.
* Edit ```scripts/run_vllm_server.sh``` if you need to change quantization or VRAM utilization (4bit requires ~10GB VRAM). Model selection is controlled via `configs/global.yaml`.

7. Create necessary directories:

```bash
mkdir -p annotations uploaded_videos
```

## Alternative: Docker Deployment

For easier deployment and isolation, you can use Docker containers instead of the traditional installation:

```bash
cd docker
./run-surgical-agents.sh
```

This will automatically download models, build all necessary containers, and start the services. See [docker/README.md](docker/README.md) for detailed Docker deployment instructions.

## Robot Control and Policy Runner

The framework includes support for real-world robot control via the GR00T N1.5 policy runner. This enables physical surgical robots to execute manipulation tasks guided by vision-language models.

For setup instructions, configuration details, and hardware requirements, see [policy_runner/README.md](policy_runner/README.md).

## Running the Surgical Agentic Framework Demo

### Production Mode

1. Run the full stack with all services:

```
npm start
```

Or using the script directly:

```
./scripts/start_app.sh
```

What it does:

* Builds the CSS with Tailwind
* Starts vLLM server with the model on port 8000
* Waits 45 seconds for the model to load
* Starts Whisper (servers/whisper_online_server.py) on port 43001 (for ASR)
* Waits 5 seconds
* Launches ```python servers/app.py``` (the main Flask + WebSockets application)
* Waits for all processes to complete

### Development Mode

For UI development with hot-reloading CSS changes:

```
npm run dev:web
```

This starts:
* The CSS watch process for automatic Tailwind compilation
* The web server only (no LLM or Whisper)

For full stack development:

```
npm run dev:full
```

This is the same as production mode but also watches for CSS changes.

You can also use the development script for faster startup during development:

```
./scripts/dev.sh
```

2. **Open** your browser at ```http://127.0.0.1:8050```. You should see the Surgical Agentic Framework Demo interface:
    * A video sample (```sample_video.mp4```)
    * Chat console
    * A "Start Mic" button to begin ASR.

3. Try speaking or Typing:
    * If you say "Take a note: The gallbladder is severely inflamed," the system routes you to NotetakerAgent.
    * If you say "What are the next steps after dissecting the cystic duct?" it routes you to ChatAgent.
    * If you ask record-specific questions like "What meds is the patient on?" or "Any abnormal labs?", it routes you to EHRAgent (after you build the EHR index; see below).

4. Background Annotations:
    * Meanwhile, ```AnnotationAgent``` writes a file like: ```procedure_2025_01_18__10_25_03_annotations.json``` in the annotations folder very 10 seconds with structured timeline data.

## Uploading and Processing Videos

The framework supports two video source modes:

### Uploaded Videos
1. Click on the "Upload Video" button to add your own surgical videos
2. Browse the video library by clicking "Video Library" 
3. Select a video to analyze
4. Use the chat interface to ask questions about the video or create annotations

### Live Streaming (WebRTC)

The framework now supports real-time analysis of live surgical streams via WebRTC:

1. **Toggle to Live Stream Mode**: Select the "Live Stream" radio button in the video controls
2. **Configure Server URL**: Enter your WebRTC server URL (default: `http://localhost:8080`)
3. **Connect**: Click the "Connect" button to establish the WebRTC connection
4. **Monitor Status**: The connection status indicator will show:
   * Yellow: Connecting...
   * Green: Connected
   * Red: Error
   * Gray: Disconnected
5. **Auto Frame Capture**: The system automatically captures frames from the live stream for analysis
6. **Disconnect**: Click "Disconnect" when finished to cleanly close the connection

**WebRTC Server Requirements:**
* The WebRTC server must provide the following API endpoints:
  - `/iceServers` - Returns ICE server configuration
  - `/offer` - Accepts WebRTC offer and returns answer
* Compatible with the [Holohub live video server application](https://github.com/nvidia-holoscan/holohub) or any server implementing the same API

**Features:**
* Seamless switching between uploaded videos and live streams
* Automatic ICE server configuration with fallback STUN server
* Proper connection state management and cleanup
* Support for fullscreen and frame capture in both modes
* Real-time video analysis capabilities

## Generating Post-Operation Notes

After accumulating annotations and notes during a procedure:

1. Click the "Generate Post-Op Note" button
2. The system will analyze all annotations and notes
3. A structured post-operation note will be generated with:
   * Procedure information
   * Key findings
   * Procedure timeline
   * Complications

## EHR Q&A (Vector DB)

This repository includes a lightweight EHR retrieval pipeline:

- Build an EHR vector index from text/JSON files
- Query the index via an EHRAgent with the same vLLM backend
- A sample synthetic patient record is included at `ehr/patient_history.txt` to get you started

Steps:

1) Build the index from a directory of `.txt`, `.md`, or `.json` files

```
python scripts/ehr_build_index.py /path/to/ehr_docs ehr_index \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk_tokens 256 --overlap_tokens 32
```

2) Point the agent at the index by editing `configs/ehr_agent.yaml`:

- `ehr_index_dir`: set to `ehr_index` (or your output path)
- Optionally adjust `retrieval_top_k`, `context_max_chars`

3) You can test by querying via CLI (uses the same vLLM server):

```
python scripts/ehr_query.py --question "What medications is the patient on?"
```

4) Integration in app selection:

- `If the user asks about EHR/records (e.g., "labs", "medications", "allergies"), the request is routed to EHRAgent automatically.
- Make sure vLLM is running (`./scripts/run_vllm_server.sh`) and the EHR index exists.

## Troubleshooting

Common issues and solutions:

1. **WebSocket Connection Errors**:
   * Check firewall settings to ensure ports 49000 and 49001 are open
   * Ensure no other applications are using these ports
   * If you experience frequent timeouts, adjust the WebSocket configuration in `servers/web_server.py`

2. **Model Loading Errors**:
   * Verify model paths are correct in configuration files
   * Ensure you have sufficient GPU memory for the models
   * Check the log files for specific error messages

3. **Audio Transcription Issues**:
   * Verify your microphone is working correctly
   * Check that the Whisper server is running
   * Adjust microphone settings in your browser

4. **WebRTC Connection Issues**:
   * Ensure the WebRTC server is running and accessible at the configured URL
   * Check that the server implements the required `/iceServers` and `/offer` endpoints
   * Verify network connectivity and firewall settings for WebRTC ports
   * Check browser console for detailed WebRTC connection errors
   * Ensure the video element has `autoplay` and `playsinline` attributes for proper stream playback

## Text-to-Speech (TTS)

The framework supports both local and cloud-based TTS options:

### Local TTS Service (Recommended)
**Benefits**: Private, GPU-accelerated, Offline-capable

The TTS service uses a high-quality English VITS model ([Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/pdf/2106.06103)) (`tts_models/en/ljspeech/vits`) that automatically downloads on first use. The model is stored persistently in `./tts-service/models/` and will be available across container restarts.

### ElevenLabs TTS (Alternative)
For cloud-based premium quality TTS:
- Configure your ElevenLabs API key in the web interface
- No local storage or GPU resources required

## File Structure

A brief overview:

```
surgical_agentic_framework/
├── agents/                 <-- Agent implementations
│   ├── annotation_agent.py
│   ├── base_agent.py
│   ├── chat_agent.py
│   ├── ehr_agent.py
│   ├── notetaker_agent.py
│   ├── post_op_note_agent.py
│   └── selector_agent.py
├── ehr/                    <-- Retrieval components for EHR
│   ├── builder.py          <-- Builds FAISS index from text/JSON
│   └── store.py            <-- Loads/queries the index
├── configs/                <-- Configuration files
│   ├── annotation_agent.yaml
│   ├── chat_agent.yaml
│   ├── notetaker_agent.yaml
│   ├── post_op_note_agent.yaml
│   └── selector.yaml
├── models/                 <-- Model files
│   ├── llm/                <-- LLM model files
│   │   └── Llama-3.2-11B-lora-surgical-4bit/
│   └── whisper/            <-- Whisper models (downloaded at runtime)
├── scripts/                <-- Shell scripts for starting services
│   ├── dev.sh              <-- Development script for quick startup
│   ├── run_vllm_server.sh
│   ├── start_app.sh        <-- Main script to launch everything
│   └── start_web_dev.sh    <-- Web UI development script
│   ├── ehr_build_index.py  <-- Build EHR vector index
│   └── ehr_query.py        <-- Query EHRAgent via CLI
├── servers/                <-- Server implementations
│   ├── app.py              <-- Main application server
│   ├── uploaded_videos/    <-- Storage for uploaded videos
│   ├── web_server.py       <-- Web interface server
│   └── whisper_online_server.py <-- Whisper ASR server
├── utils/                  <-- Utility classes and functions
│   ├── chat_history.py
│   ├── logging_utils.py
│   └── response_handler.py
├── web/                    <-- Web interface assets
│   ├── static/             <-- CSS, JS, and other static assets
│   │   ├── audio.js
│   │   ├── bootstrap.bundle.min.js
│   │   ├── bootstrap.css
│   │   ├── chat.css
│   │   ├── jquery-3.6.3.min.js
│   │   ├── main.js
│   │   ├── nvidia-logo.png
│   │   ├── styles.css
│   │   ├── tailwind-custom.css
│   │   └── websocket.js
│   └── templates/
│       └── index.html
├── annotations/            <-- Stored procedure annotations
├── uploaded_videos/        <-- Uploaded video storage
├── README.md               <-- This file
├── package.json            <-- Node.js dependencies and scripts
├── postcss.config.js       <-- PostCSS configuration for Tailwind
├── tailwind.config.js      <-- Tailwind CSS configuration
├── vite.config.js          <-- Vite build configuration
└── requirements.txt        <-- Python dependencies
```
