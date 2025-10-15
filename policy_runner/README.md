# SO-ARM Starter Policy Runner

This script allows running policy models using DDS for communication, suitable for simulation robot control. The runner now supports a YAML configuration file and containerized execution, making it easier to reproduce the required environment.

## Supported Policies

*   **GR00T N1.5**: NVIDIA's foundation model for humanoid robots. Refer to [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) for more information.


## Configuration

A template configuration lives at `configs/policy_runner.yaml`. Copy or update this file and replace every value marked with `__REQUIRED__` before launching the runner. The container and helper scripts validate that:

- All placeholders have been removed.
- `robot.port` is provided so the container can expose the serial device.
- `ckpt_path` exists on the host and will be mounted read-only into the container.

You can pass the YAML file directly to the runner with the new `--config` flag:

```bash
python -m policy_runner.run_policy --config configs/policy_runner.yaml
```

Any CLI flags provided alongside `--config` override values from the file.

## Containerized Deployment

The policy runner container reuses the same NVIDIA PyTorch base image as the Whisper service and bootstraps dependencies via the existing setup scripts.

1. Edit `configs/policy_runner.yaml` as described above.
2. Build the image (one-time unless dependencies change):
   ```bash
   ./docker/run-surgical-agents.sh build policy
   ```
3. Start the service:
   ```bash
   ./docker/run-surgical-agents.sh run policy
   ```

The helper script validates the YAML before launch, maps the configured serial device into the container, mounts the checkpoint directory read-only, and keeps the container running under the name `vlm-surgical-policy`. Use `./docker/run-surgical-agents.sh logs policy` and `./docker/run-surgical-agents.sh status` to inspect the service, or `./docker/run-surgical-agents.sh stop policy` to shut it down.

To run the container manually:

```bash
docker run --rm -it \
  --gpus all \
  --net host \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -v $(pwd)/configs/policy_runner.yaml:/workspace/configs/policy_runner.yaml:ro \
  -v /absolute/path/to/checkpoints:/absolute/path/to/checkpoints:ro \
  vlm-surgical-agents:policy-runner \
  --verbose
```

Replace the device and checkpoint paths with values that match your configuration. Additional arguments appended to `docker run` are forwarded to `policy_runner.run_policy`.

## Manual Environment Setup (Optional)

> Note: This needs to be set up in a different environment than the rest of the agent framework.

Create a conda environment with python 3.11
```bash
conda create -n agent_so_arm python=3.11 -y
conda activate agent_so_arm
```

Run the script from the repository root:
```bash
bash policy_runner/tools/env_setup_so_arm_starter.sh
```
**⚠️ Expected Build Time**: The environment setup process takes approximately 10-20 minutes. You may encounter intermediary warnings about macaroon bakery library dependencies - these are non-critical and can be ignored.


Once setup is complete, you can start the robot with:


If you don't already know which port your follower arm is on, first run:
```bash
python policy_runner/third_party/lerobot/lerobot/find_port.py
```
Then, run the policy runner script. Using the YAML config is recommended:

```bash
python -m policy_runner.run_policy --config configs/policy_runner.yaml
```

Override any field by appending CLI flags. For example, to temporarily change the wrist camera:

```bash
python -m policy_runner.run_policy \
  --config configs/policy_runner.yaml \
  --robot.cameras.wrist='{"type":"opencv","index_or_path":6,"width":640,"height":480,"fps":30}'
```

## Remote Control

- The robot is homed to the configured start pose on launch, then waits in an idle state until it receives a `start` command over ZeroMQ.
- While the policy loop is running you can send `pause` to hold the current pose, or `reset` to pause and return to the nominal start pose.
- The control server binds to `tcp://*:5556` by default; change it with `--control.endpoint`.
- Supported commands (case-insensitive): `start`, `pause`, `reset`.

Example Python client:

```python
import zmq

ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect("tcp://localhost:5556")  # match the policy runner endpoint

socket.send_string("start")
print(socket.recv_string())  # -> "policy started"

socket.send_string("pause")
print(socket.recv_string())  # -> "policy paused"

socket.send_string("reset")
print(socket.recv_string())  # -> "policy reset to start pose"
```

#### TensorRT Inference

**Requirements:** TensorRT inference requires pre-built engine

- Convert PyTorch model to TensorRT engines

Export model to ONNX format
```sh
python -m policy_runner.gr00tn1_5.trt.export_onnx --ckpt_path <path_to_your_checkpoint>
```

Build TensorRT engines
```sh
bash policy_runner/gr00tn1_5/trt/build_engine.sh
```

You can choose to use Docker to build TensorRT engines.
```sh
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:25.06-py3 \
  bash policy_runner/gr00tn1_5/trt/build_engine.sh
```

- Run inference with TensorRT engines
```sh
python -m policy_runner.run_policy \
  --ckpt_path=<path_to_your_checkpoint> \
  --robot.port=<follower port> \
  --robot.id=so101_follower_arm \
  --robot.cameras.room='{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}' \
  --robot.cameras.wrist='{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30}'
  --trt \
  --trt_engine_path <path_to_tensorrt_engine_directory>
```

## Command Line Arguments

Here's a markdown table describing the command-line arguments:

| Argument                     | Description                                                              | Default Value                              |
|------------------------------|--------------------------------------------------------------------------|-------------------------------------------|
| `--config`                   | Path to a YAML file with runner settings.                                | None                                      |
| `--ckpt_path`                | Checkpoint path for the policy model. (Required unless provided via YAML)| N/A                                       |
| `--task_description`         | Task description text prompt for the policy.                             | `Grip the scissors and put it into the tray` |
| `--data_config`              | Data config name (used for GR00T N1.5).                                  | `so100_dualcam`                           |
| `--embodiment_tag`           | The embodiment tag for the model (used for GR00T N1.5).                  | `new_embodiment`                          |
| `--chunk_length`             | Length of the action chunk inferred by the policy per inference step.    | 16                                        |
| `--trt`                      | Enable TensorRT engine for accelerated inference.                        | False                                     |
| `--trt_engine_path`          | Path to the TensorRT engine files directory.                             | `gr00t_engine`                            |
| `--policy_device`            | Force policy device (e.g., cuda:0).                                      | `cuda` (if available)                     |
| `--control.endpoint`         | ZeroMQ bind endpoint for remote control commands.                        | `tcp://*:5556`                            |
| `--robot.type`               | Type of robot to use.                                                    | `so101_follower`                          |
| `--robot.port`               | Serial port for robot communication. (Required)                          | N/A                                       |
| `--robot.id`                 | Robot identifier.                                                        | `so101`                                   |
| `--robot.calibration_dir`    | Directory containing robot calibration files.                            | None                                      |
| `--robot.max_relative_target`| Scalar or comma-separated values used for motion safety.                 | None                                      |
| `--robot.use_degrees`        | Use degrees for robot positions instead of normalized values.            | False                                     |
| `--keep_torque_enabled`      | Do not disable torque on disconnect.                                     | False                                     |
| `--robot.fps`                | Target control loop frequency in Hz.                                     | 30.0                                      |
| `--robot.cameras.room`       | Room camera configuration (index/path or JSON).                          | None (Required)                           |
| `--robot.cameras.wrist`      | Wrist camera configuration (index/path or JSON).                         | None (Required)                           |
| `--room_key`                 | Observation key for the room camera.                                     | `video.room`                              |
| `--wrist_key`                | Observation key for the wrist camera.                                    | `video.wrist`                             |
| `--camera.width`             | Camera image width.                                                      | 640                                       |
| `--camera.height`            | Camera image height.                                                     | 480                                       |
| `--prealign_steps`           | Interpolation steps when homing the arm.                                 | 20                                        |
| `--prealign_pause`           | Pause between pre-alignment steps in seconds.                            | 0.1                                       |
| `--skip_prealign`            | Skip moving robot to start pose before inference.                        | False                                     |
| `--actions_to_apply`         | Number of actions from each chunk to execute before next inference.      | 12                                        |
| `--skip_calibration`         | Skip automatic calibration on connect.                                   | False                                     |
| `--verbose`                  | Enable verbose output and status logging.                                | False                                     |

## Performance Metrics

### Benchmark
- Runtime
```sh
python -m policy_runner.gr00tn1_5.trt.benchmark \
   --ckpt_path=<path_to_checkpoint>
   --inference_mode=<tensorrt_or_pytorch>
```

- Accuracy
```sh
python -m policy_runner.gr00tn1_5.trt.benchmark \
   --ckpt_path=<path_to_checkpoint>
   --inference_mode=compare
```

### Performance Results

| Hardware            | Inference Mode | Average Latency | Actions Predicted |
|---------------------|----------------|-----------------|-------------------|
| NVIDIA RTX 6000 Ada | PyTorch        | 42.16 ± 0.81 ms |        16         |
| NVIDIA RTX 6000 Ada | TensorRT       | 26.96 ± 1.86 ms |        16         |
