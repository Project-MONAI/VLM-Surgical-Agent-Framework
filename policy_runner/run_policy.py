#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-world bridge that streams LeRobot observations through the GR00T N1.5 policy."""

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, TypeVar

import numpy as np
from PIL import Image
import zmq
try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency checked during setup
    raise RuntimeError("PyYAML is required to load policy runner configs. Install it with 'pip install pyyaml'.") from exc

from policy_runner.gr00tn1_5.runners import GR00TN1_5_PolicyRunner

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is required at runtime
    raise RuntimeError("PyTorch must be installed to run the real-world policy bridge.") from exc

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config


DEFAULT_START_POSE_SERVO_UNITS = {
    "shoulder_pan": 2037.0,
    "shoulder_lift": 2120.0,
    "elbow_flex": 2057.0,
    "wrist_flex": 2047.0,
    "wrist_roll": 2047.0,
    "gripper": 2046.0,
}


T = TypeVar("T")


def retry_on_connection_error(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple = (ConnectionError, OSError),
    verbose: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on connection errors."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_attempts:
                        if verbose:
                            print(
                                f"[retry] {func.__name__} failed "
                                f"(attempt {attempt}/{max_attempts}): "
                                f"{exc}. Retrying in {current_delay:.2f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if verbose:
                            msg = (
                                f"[retry] {func.__name__} failed "
                                f"after {max_attempts} attempts."
                            )
                            print(msg)
            if last_exception:
                raise last_exception
            msg = f"{func.__name__} failed without raising exception"
            raise RuntimeError(msg)

        return wrapper

    return decorator


def _parse_structured(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(text)
        except (ValueError, SyntaxError) as err:
            raise ValueError(f"Unable to parse structured value: {text}") from err
        if not isinstance(result, dict):
            raise ValueError(f"Structured value must be a mapping, received: {type(result)}")
        return result


def _coerce_index_or_path(value: str | int) -> int | str | Path:
    if isinstance(value, int):
        return value
    value_str = str(value).strip()
    # Keep device paths as strings for OpenCV compatibility
    if value_str.startswith("/dev/"):
        return value_str
    # Convert video files to Path objects
    if value_str.startswith("/") or value_str.endswith(".mp4") or value_str.endswith(".mkv"):
        return Path(value_str)
    if value_str.isdigit():
        return int(value_str)
    try:
        return int(value_str)
    except ValueError:
        return Path(value_str)


def _parse_camera_block(
    spec: str | dict[str, Any],
    default_width: int,
    default_height: int,
    default_fps: float,
) -> CameraConfig:
    if isinstance(spec, str):
        if spec.strip().startswith("{"):
            data = _parse_structured(spec)
        else:
            data = {
                "type": "opencv",
                "index_or_path": spec,
            }
    else:
        data = spec

    cam_type = data.get("type", "opencv")
    if cam_type != "opencv":
        raise ValueError(f"Unsupported camera type '{cam_type}'. Only 'opencv' is currently supported.")

    index_or_path = data.get("index_or_path")
    if index_or_path is None:
        raise ValueError("OpenCV camera config requires 'index_or_path'.")

    width = int(data.get("width", default_width))
    height = int(data.get("height", default_height))
    fps = float(data.get("fps", default_fps))

    extra = {key: value for key, value in data.items() if key not in {"type", "index_or_path", "width", "height", "fps"}}

    return OpenCVCameraConfig(
        index_or_path=_coerce_index_or_path(index_or_path),
        width=width,
        height=height,
        fps=int(fps),
        **extra,
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{path}' does not exist.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file '{path}' is empty.")
    if not isinstance(data, dict):
        raise TypeError(f"Config file '{path}' must contain a mapping at the top level.")
    return data


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []

    def add_option(flag: str, value: Any) -> None:
        if value is None:
            return
        cli_args.extend([flag, str(value)])

    def add_bool(flag: str, value: Any) -> None:
        if bool(value):
            cli_args.append(flag)

    add_option("--ckpt_path", config.get("ckpt_path"))
    add_option("--task_description", config.get("task_description"))
    add_option("--data_config", config.get("data_config"))
    add_option("--embodiment_tag", config.get("embodiment_tag"))
    add_option("--chunk_length", config.get("chunk_length"))
    add_bool("--trt", config.get("trt"))
    add_option("--trt_engine_path", config.get("trt_engine_path"))
    add_option("--policy_device", config.get("policy_device"))

    control_cfg = config.get("control", {})
    if isinstance(control_cfg, dict):
        add_option("--control.endpoint", control_cfg.get("endpoint"))

    robot_cfg = config.get("robot", {})
    if isinstance(robot_cfg, dict):
        add_option("--robot.type", robot_cfg.get("type"))
        add_option("--robot.port", robot_cfg.get("port"))
        add_option("--robot.id", robot_cfg.get("id"))
        add_option("--robot.calibration_dir", robot_cfg.get("calibration_dir"))
        max_rel = robot_cfg.get("max_relative_target")
        if isinstance(max_rel, (list, tuple)):
            max_rel = ",".join(str(item) for item in max_rel)
        add_option("--robot.max_relative_target", max_rel)
        add_bool("--robot.use_degrees", robot_cfg.get("use_degrees"))
        add_option("--robot.fps", robot_cfg.get("fps"))

        cameras_cfg = robot_cfg.get("cameras", {})
        if isinstance(cameras_cfg, dict):
            room_cfg = cameras_cfg.get("room")
            if isinstance(room_cfg, dict):
                add_option("--robot.cameras.room", json.dumps(room_cfg))
            wrist_cfg = cameras_cfg.get("wrist")
            if isinstance(wrist_cfg, dict):
                add_option("--robot.cameras.wrist", json.dumps(wrist_cfg))

    add_option("--room_key", config.get("room_key"))
    add_option("--wrist_key", config.get("wrist_key"))

    camera_cfg = config.get("camera", {})
    if isinstance(camera_cfg, dict):
        add_option("--camera.width", camera_cfg.get("width"))
        add_option("--camera.height", camera_cfg.get("height"))

    prealign_cfg = config.get("prealign", {})
    if isinstance(prealign_cfg, dict):
        add_option("--prealign_steps", prealign_cfg.get("steps"))
        add_option("--prealign_pause", prealign_cfg.get("pause"))

    add_bool("--skip_prealign", config.get("skip_prealign"))
    add_option("--actions_to_apply", config.get("actions_to_apply"))
    add_bool("--skip_calibration", config.get("skip_calibration"))
    add_bool("--keep_torque_enabled", config.get("keep_torque_enabled"))
    add_bool("--verbose", config.get("verbose"))

    return cli_args


def _build_camera_configs(args: argparse.Namespace) -> dict[str, CameraConfig]:
    width = args.camera_width
    height = args.camera_height
    fps = args.robot_fps
    camera_configs: dict[str, CameraConfig] = {}

    if args.robot_cameras_room:
        camera_configs[args.room_key] = _parse_camera_block(args.robot_cameras_room, width, height, fps)
    if args.robot_cameras_wrist:
        camera_configs[args.wrist_key] = _parse_camera_block(args.robot_cameras_wrist, width, height, fps)

    if args.room_key not in camera_configs or args.wrist_key not in camera_configs:
        raise ValueError(
            "Both room and wrist camera configs are required. Provide them via --robot.cameras "
            "or the shorthand --robot.cameras.room / --robot.cameras.wrist flags."
        )
    return camera_configs


def _build_robot_config(args: argparse.Namespace, camera_configs: dict[str, CameraConfig]) -> RobotConfig:
    calibration_dir = Path(args.robot_calibration_dir).expanduser() if args.robot_calibration_dir else None
    
    # Transform path if it doesn't exist: /home/<username>/ -> /root/ for container compatibility
    if calibration_dir and not calibration_dir.exists():
        calibration_str = str(calibration_dir)
        if calibration_str.startswith("/home/"):
            # Replace /home/<username>/ with /root/
            parts = calibration_str.split("/")
            if len(parts) > 2:  # /home/<username>/...
                container_path = "/root/" + "/".join(parts[3:])
                container_calibration = Path(container_path)
                if container_calibration.exists():
                    print(f"[0MQ Msg] Transformed calibration path: {calibration_dir} -> {container_calibration}")
                    calibration_dir = container_calibration
    
    if args.robot_type != "so101_follower":
        raise ValueError(f"Unsupported robot type '{args.robot_type}'. Only 'so101_follower' is supported right now.")

    max_relative_target = None
    if args.robot_max_relative_target is not None:
        if "," in args.robot_max_relative_target:
            max_relative_target = [float(x) for x in args.robot_max_relative_target.split(",")]
        else:
            max_relative_target = float(args.robot_max_relative_target)

    return SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        cameras=camera_configs,
        calibration_dir=calibration_dir,
        disable_torque_on_disconnect=not args.keep_torque_enabled,
        max_relative_target=max_relative_target,
        use_degrees=args.robot_use_degrees,
    )


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _flatten_chunk(chunk: np.ndarray, chunk_length: int, dof: int) -> np.ndarray:
    chunk = np.asarray(chunk)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    if chunk.shape[0] != chunk_length:
        chunk = chunk.reshape(chunk_length, dof)
    if chunk.shape[1] != dof:
        raise ValueError(f"Unexpected action shape {chunk.shape}, expected (*, {dof})")
    return chunk


def _maybe_log_frame(image: np.ndarray, output_dir: Path, prefix: str, index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / f"{prefix}_{index:06d}.png")


def _check_port_permissions(port: str, verbose: bool = False) -> None:
    """
    Check if we have read/write permissions on the specified port.

    Args:
        port: The port path (e.g., /dev/ttyACM0)
        verbose: Whether to print detailed information

    Raises:
        PermissionError: If the port exists but we don't have sufficient permissions
        FileNotFoundError: If the port doesn't exist
    """
    # Only check device files
    if not port.startswith("/dev/"):
        if verbose:
            msg = (
                f"[0MQ Msg] Skipping permission check "
                f"for non-device port: {port}"
            )
            print(msg)
        return

    # Check if port exists
    if not os.path.exists(port):
        raise FileNotFoundError(
            f"Serial port '{port}' does not exist. Please check that:\n"
            f"  1. The device is connected\n"
            f"  2. The port path is correct\n"
            f"  3. The device drivers are loaded"
        )

    # Check read/write permissions
    can_read = os.access(port, os.R_OK)
    can_write = os.access(port, os.W_OK)

    if not (can_read and can_write):
        error_msg = (
            f"\n{'='*80}\n"
            f"PERMISSION ERROR: Insufficient permissions to "
            f"access '{port}'\n"
            f"{'='*80}\n"
            f"Current user does not have read/write access to "
            f"the serial port.\n\n"
            f"To fix this, run the following command:\n\n"
            f"    sudo chmod 666 {port}\n\n"
            f"{'='*80}\n"
        )
        raise PermissionError(error_msg)

    if verbose:
        msg = (
            f"[0MQ Msg] ✓ Port '{port}' is accessible "
            f"with read/write permissions"
        )
        print(msg)


def _setup_interrupt_handler(flag: Dict[str, bool]) -> Callable[[int, Any], None]:
    def handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal handler
        flag["stop"] = True
        print(f"Received signal {signum}. Shutting down the policy bridge...")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    return handler


def _convert_servo_to_normalized(
    servo_value: float, range_min: int, range_max: int
) -> float:
    """
    Convert servo units to normalized [-100, 100] range.

    Args:
        servo_value: Position in servo units (e.g., 0-4095)
        range_min: Minimum calibrated position for this joint
        range_max: Maximum calibrated position for this joint

    Returns:
        Normalized position in [-100, 100] range
    """
    if range_max <= range_min:
        raise ValueError(
            f"Invalid calibration: range_max ({range_max}) must be > "
            f"range_min ({range_min})"
        )
    # Formula: ((value - min) / (max - min)) * 200 - 100
    normalized = (
        ((servo_value - range_min) / (range_max - range_min)) * 200.0 - 100.0
    )
    return normalized


def _resolve_start_pose(
    robot, use_degrees: bool, verbose: bool = False
) -> dict[str, float]:
    """
    Resolve start pose, converting from servo units to appropriate format.

    Args:
        robot: Robot instance (needed for calibration data)
        use_degrees: Whether robot is using degree mode
        verbose: Whether to print conversion details

    Returns:
        Start pose dict with joint names and target positions
    """
    # Use default servo units pose
    data = DEFAULT_START_POSE_SERVO_UNITS.copy()

    start_pose: dict[str, float] = {}
    conversion_log = []

    # If not using degrees, convert servo units to normalized [-100, 100]
    if not use_degrees and robot.is_calibrated:
        for key, servo_value in data.items():
            try:
                servo_val = float(servo_value)
                # Get calibration for this joint
                if key in robot.calibration:
                    cal = robot.calibration[key]
                    normalized_val = _convert_servo_to_normalized(
                        servo_val, cal.range_min, cal.range_max
                    )
                    start_pose[str(key)] = normalized_val
                    conversion_log.append(
                        f"  {key}: {servo_val:.1f} servo units "
                        f"-> {normalized_val:.2f} normalized "
                        f"(range: {cal.range_min}-{cal.range_max})"
                    )
                else:
                    # Joint not in calibration, use as-is
                    start_pose[str(key)] = servo_val
                    conversion_log.append(
                        f"  {key}: {servo_val:.1f} (no calibration)"
                    )
            except (TypeError, ValueError) as exc:
                msg = (
                    f"Invalid start pose value for joint "
                    f"'{key}': {servo_value}"
                )
                raise ValueError(msg) from exc

        if verbose and conversion_log:
            print("\n[0MQ Msg] Start Pose Conversion:")
            print("  Servo units -> Normalized [-100, 100]")
            for line in conversion_log:
                print(line)
            print()
    else:
        # Use degrees mode or not calibrated: use values as-is
        for key, value in data.items():
            try:
                start_pose[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                msg = (
                    f"Invalid start pose value for joint "
                    f"'{key}': {value}"
                )
                raise ValueError(msg) from exc

    return start_pose


def _interpolate_to_pose(
    robot,
    joint_keys: Iterable[str],
    target_pose: dict[str, float],
    min_steps: int,
    pause_s: float,
    verbose: bool,
) -> None:
    @retry_on_connection_error(
        max_attempts=5, delay=0.1, backoff=1.5, verbose=verbose
    )
    def get_observation_with_retry():
        return robot.get_observation()

    observation = get_observation_with_retry()
    current = np.array(
        [observation[key] for key in joint_keys], dtype=np.float32
    )

    targets = []
    joint_names = []
    for idx, key in enumerate(joint_keys):
        joint_name = key.removesuffix(".pos")
        joint_names.append(joint_name)
        default = target_pose.get(key, current[idx])
        targets.append(float(target_pose.get(joint_name, default)))
    target_array = np.array(targets, dtype=np.float32)

    # Print detailed position information
    if verbose:
        print("\n" + "=" * 80)
        print("[0MQ Msg] Starting Position Alignment")
        print("=" * 80)
        print(f"{'Joint':<20} {'Current':<15} {'Target':<15} {'Delta':<15}")
        print("-" * 80)
        for idx, (jname, curr, tgt) in enumerate(
            zip(joint_names, current, target_array)
        ):
            delta = tgt - curr
            print(
                f"{jname:<20} {curr:<15.4f} {tgt:<15.4f} {delta:<15.4f}"
            )
        print("-" * 80)
        max_delta = float(np.max(np.abs(target_array - current)))
        print(f"Max absolute delta: {max_delta:.4f}")
        print("=" * 80 + "\n")

    if np.allclose(current, target_array, atol=1e-4):
        if verbose:
            msg = (
                "[0MQ Msg] Robot already at start pose; "
                "skipping pre-alignment."
            )
            print(msg)
        return

    steps = max(min_steps, 10)
    if verbose:
        max_d = float(np.max(np.abs(target_array - current)))
        print(
            f"[0MQ Msg] Aligning robot to start pose over "
            f"{steps} steps (max delta {max_d:.4f})."
        )

    for step in range(1, steps + 1):
        alpha = step / steps
        intermediate = current + (target_array - current) * alpha
        action = {
            joint_key: float(intermediate[idx])
            for idx, joint_key in enumerate(joint_keys)
        }

        @retry_on_connection_error(
            max_attempts=3, delay=0.05, backoff=1.5, verbose=verbose
        )
        def send_action_with_retry():
            robot.send_action(action)

        send_action_with_retry()

        if verbose and (step % max(1, steps // 5) == 0 or step == steps):
            progress_pct = alpha * 100
            msg = (
                f"[0MQ Msg] Alignment progress: "
                f"{step}/{steps} ({progress_pct:.1f}%)"
            )
            print(msg)

        if pause_s > 0:
            time.sleep(pause_s)

    if verbose:
        print("[0MQ Msg] Pre-alignment complete.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GR00T N1.5 policy on real hardware via DDS.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file containing runner arguments.",
    )

    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path for the policy model.")
    parser.add_argument("--task_description", type=str, default="Grip the scissors and put it into the tray")
    parser.add_argument("--data_config", type=str, default="so100_dualcam")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--chunk_length", type=int, default=16)
    parser.add_argument("--trt", action="store_true", help="Enable TensorRT inference.")
    parser.add_argument("--trt_engine_path", type=str, default="gr00t_engine")
    parser.add_argument("--policy_device", type=str, default=None, help="Force policy device (e.g., cuda:0).")
    parser.add_argument(
        "--control.endpoint",
        dest="control_endpoint",
        type=str,
        default="tcp://*:5556",
        help="ZeroMQ endpoint to bind for remote control commands.",
    )

    parser.add_argument("--robot.type", dest="robot_type", type=str, default="so101_follower")
    parser.add_argument("--robot.port", dest="robot_port", type=str, required=True)
    parser.add_argument("--robot.id", dest="robot_id", type=str, default="so101")
    parser.add_argument("--robot.calibration_dir", dest="robot_calibration_dir", type=str, default=None)
    parser.add_argument(
        "--robot.max_relative_target",
        dest="robot_max_relative_target",
        type=str,
        default=None,
        help="Scalar or comma separated values used for motion safety.",
    )
    parser.add_argument(
        "--robot.use_degrees",
        dest="robot_use_degrees",
        action="store_true",
        help=(
            "Use degrees for robot positions instead of raw servo units. "
            "Set this if your robot was calibrated with use_degrees=True. "
            "This affects how positions are read/written and the default "
            "start pose."
        ),
    )
    parser.add_argument(
        "--keep_torque_enabled",
        action="store_true",
        help="Do not disable torque on disconnect.",
    )
    parser.add_argument("--robot.fps", dest="robot_fps", type=float, default=30.0)

    parser.add_argument("--robot.cameras.room", dest="robot_cameras_room", type=str, default=None)
    parser.add_argument("--robot.cameras.wrist", dest="robot_cameras_wrist", type=str, default=None)
    parser.add_argument("--room_key", type=str, default="video.room", help="Observation key for the room camera.")
    parser.add_argument("--wrist_key", type=str, default="video.wrist", help="Observation key for the wrist camera.")

    parser.add_argument("--camera.width", dest="camera_width", type=int, default=640)
    parser.add_argument("--camera.height", dest="camera_height", type=int, default=480)

    parser.add_argument("--prealign_steps", type=int, default=20, help="Interpolation steps when homing the arm.")
    parser.add_argument(
        "--prealign_pause",
        type=float,
        default=0.1,
        help="Pause between pre-alignment steps in seconds.",
    )
    parser.add_argument(
        "--skip_prealign",
        action="store_true",
        help="Skip moving the robot to the simulation-aligned start pose before inference.",
    )
    parser.add_argument(
        "--actions_to_apply",
        type=int,
        default=12,
        help="Number of actions from each inferred chunk to execute on hardware before the next inference step.",
    )

    parser.add_argument("--skip_calibration", action="store_true", help="Skip automatic calibration on connect.")
    parser.add_argument("--verbose", action="store_true")

    argv = sys.argv[1:]
    config_path: str | None = None
    filtered_argv: list[str] = []

    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--config":
            if idx + 1 >= len(argv):
                raise ValueError("--config requires a path argument.")
            config_path = argv[idx + 1]
            skip_next = True
        elif token.startswith("--config="):
            config_path = token.split("=", 1)[1]
        else:
            filtered_argv.append(token)

    config_args: list[str] = []
    if config_path:
        config_data = _load_yaml_config(Path(config_path).expanduser())
        config_args = _config_to_cli_args(config_data)

    return parser.parse_args(config_args + filtered_argv)


def main() -> None:
    args = parse_args()
    # Check port permissions before attempting connection
    _check_port_permissions(args.robot_port, verbose=args.verbose)
    camera_configs = _build_camera_configs(args)
    robot_config = _build_robot_config(args, camera_configs)
    robot = make_robot_from_config(robot_config)

    if args.policy_device:
        device = args.policy_device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required when --policy_device is not provided.")
        device = "cuda"
    policy = GR00TN1_5_PolicyRunner(
        ckpt_path=args.ckpt_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        task_description=args.task_description,
        device=device,
        trt_engine_path=args.trt_engine_path,
        trt=args.trt,
    )

    dof = len(robot.action_features)
    joint_keys: Iterable[str] = list(robot.action_features.keys())
    room_key = args.room_key
    wrist_key = args.wrist_key

    should_stop = {"stop": False}
    _setup_interrupt_handler(should_stop)

    control_context = zmq.Context()
    control_socket = control_context.socket(zmq.REP)
    control_socket.bind(args.control_endpoint)
    control_poller = zmq.Poller()
    control_poller.register(control_socket, zmq.POLLIN)
    if args.verbose:
        print(f"[0MQ Msg] Listening for control commands on {args.control_endpoint}")

    start_command = "start"
    pause_command = "pause"
    reset_command = "reset"

    idle_sleep = 0.05
    policy_active = False
    reset_requested = False

    connected = False
    try:
        robot.connect(calibrate=not args.skip_calibration)
        connected = True

        # Resolve start pose after connection (needs calibration data)
        start_pose = _resolve_start_pose(
            robot, robot_config.use_degrees, args.verbose
        )
        if not args.skip_prealign:
            _interpolate_to_pose(
                robot=robot,
                joint_keys=joint_keys,
                target_pose=start_pose,
                min_steps=args.prealign_steps,
                pause_s=args.prealign_pause,
                verbose=args.verbose,
            )
    except Exception:
        # Explicitly disconnect in case partially connected hardware needs cleanup.
        if connected:
            try:
                robot.disconnect()
            except Exception:
                pass
        raise

    try:
        target_period = (
            1.0 / args.robot_fps if args.robot_fps > 0 else 0.0
        )
        status_interval = max(int(args.robot_fps), 1)
        loop_index = 0

        @retry_on_connection_error(
            max_attempts=5, delay=0.05, backoff=1.5, verbose=args.verbose
        )
        def get_observation_with_retry():
            return robot.get_observation()

        @retry_on_connection_error(
            max_attempts=3, delay=0.02, backoff=1.3, verbose=args.verbose
        )
        def send_action_with_retry(action_dict):
            robot.send_action(action_dict)

        def process_control_command() -> None:
            nonlocal policy_active, reset_requested
            try:
                raw_command = control_socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                control_socket.send_string("no command received")
                return

            command_text = raw_command.strip()
            command = command_text.lower()
            response = ""
            try:
                if command == start_command:
                    if not policy_active:
                        policy_active = True
                        response = "policy started"
                        print("[0MQ Msg] Received start command; enabling policy loop.")
                    else:
                        response = "policy already running"
                elif command == pause_command:
                    print("[0MQ Msg] Received pause command; holding current pose.")
                    policy_active = False
                    response = "policy paused"
                elif command == reset_command:
                    print("[0MQ Msg] Received reset command; will return to start pose.")
                    policy_active = False
                    reset_requested = True
                    response = "policy reset initiated"
                else:
                    response = f"unknown command: {command_text}"
            except Exception as exc:
                response = f"error: {exc}"
                if args.verbose:
                    print(f"[0MQ Msg] Control command failed: {exc}")
            finally:
                if not response:
                    response = "ok"
                control_socket.send_string(response)

        print(f"Policy runner started and listening on {args.control_endpoint}...")
        while not should_stop["stop"]:
            events = dict(control_poller.poll(timeout=0))
            if control_socket in events:
                process_control_command()

            # Handle reset request
            if reset_requested:
                print("[0MQ Msg] Executing reset to start pose...")
                _interpolate_to_pose(
                    robot=robot,
                    joint_keys=joint_keys,
                    target_pose=start_pose,
                    min_steps=args.prealign_steps,
                    pause_s=args.prealign_pause,
                    verbose=args.verbose,
                )
                reset_requested = False
                print("[0MQ Msg] Reset complete.")
                continue

            if not policy_active:
                time.sleep(idle_sleep)
                continue

            loop_start = time.perf_counter()

            observation = get_observation_with_retry()

            if room_key not in observation or wrist_key not in observation:
                raise KeyError(
                    f"Observation is missing camera data. Available keys: {list(observation.keys())}"
                )

            room_frame = np.asarray(observation[room_key])
            wrist_frame = np.asarray(observation[wrist_key])

            joint_vector = np.array([observation[key] for key in joint_keys], dtype=np.float32)

            policy_start = time.perf_counter()
            action_chunk = _tensor_to_numpy(
                policy.infer(room_img=room_frame, wrist_img=wrist_frame, current_state=joint_vector)
            )
            action_chunk = _flatten_chunk(action_chunk, args.chunk_length, dof).astype(np.float32)
            policy_elapsed = time.perf_counter() - policy_start

            actions_to_apply = max(1, min(args.actions_to_apply, action_chunk.shape[0]))
            interrupted = False
            for action_idx in range(actions_to_apply):
                current_action = {key: float(action_chunk[action_idx, idx]) for idx, key in enumerate(joint_keys)}
                if args.verbose:
                    print(f"Sending action: {current_action}")
                send_action_with_retry(current_action)

                events = dict(control_poller.poll(timeout=0))
                if control_socket in events:
                    process_control_command()
                    if not policy_active:
                        interrupted = True
                        break

            if interrupted:
                continue

            loop_index += 1
            loop_duration = time.perf_counter() - loop_start
            if args.verbose and loop_index % status_interval == 0:
                reported_fps = 1.0 / loop_duration if loop_duration > 0 else float("inf")
                print(
                    f"[0MQ Msg] iteration={loop_index} "
                    f"rate≈{reported_fps:.1f}Hz policy={policy_elapsed * 1e3:.1f}ms "
                    f"actions_applied={actions_to_apply} "
                    f"chunk_norm={float(np.linalg.norm(action_chunk[0])):.3f}"
                )

            if target_period > 0:
                remainder = target_period - loop_duration
                if remainder > 0:
                    time.sleep(remainder)
    finally:
        if connected:
            try:
                robot.disconnect()
            except Exception as exc:  # pragma: no cover - best-effort disconnect
                print(f"Failed to disconnect robot cleanly: {exc}")
        control_socket.close(linger=0)
        control_context.term()


if __name__ == "__main__":
    main()
