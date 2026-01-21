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

import threading
import time
import logging
import os
import json
import queue
from typing import List, Optional
from pydantic import BaseModel
from .base_agent import Agent

class SurgeryAnnotation(BaseModel):
    tools: List[str]
    anatomy: List[str]
    surgical_phase: str
    description: str
    # These are populated post-parse by the agent:
    timestamp: Optional[str] = None
    elapsed_time_seconds: Optional[float] = None

class AnnotationAgent(Agent):
    def __init__(self, settings_path, response_handler, frame_queue, agent_key=None, message_bus=None, procedure_start_str=None, on_annotation_callback=None):
        super().__init__(settings_path, response_handler, agent_key=agent_key, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)
        self.frame_queue = frame_queue
        self.time_step = self.agent_settings.get("time_step_seconds", 10)

        # Initialize callback (can be set later if not provided at construction)
        self.on_annotation_callback = on_annotation_callback

        if procedure_start_str is None:
            procedure_start_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
        self.procedure_start_str = procedure_start_str
        self.procedure_start = time.time()


        base_output_dir = self.agent_settings.get("annotation_output_dir", "procedure_outputs")
        subfolder = os.path.join(base_output_dir, f"procedure_{self.procedure_start_str}")
        os.makedirs(subfolder, exist_ok=True)

        self.annotation_filepath = os.path.join(subfolder, "annotation.json")
        self._logger.info(f"AnnotationAgent writing annotations to: {self.annotation_filepath}")

        self.annotations = []
        self.stop_event = threading.Event()

        # Start the background loop in a separate thread.
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()
        self._logger.info(f"AnnotationAgent background thread started (interval={self.time_step}s).")

    def _background_loop(self):
        # Flag to track if a valid video is loaded
        video_loaded = False
        consecutive_errors = 0
        max_consecutive_errors = 5

        while not self.stop_event.is_set():
            try:
                # Attempt to get image data from the frame queue.
                try:
                    frame_data = self.frame_queue.get_nowait()

                    # If we get here, we have a frame, so video is loaded
                    video_loaded = True
                    consecutive_errors = 0  # Reset error counter on successful frame fetch
                except queue.Empty:
                    self._logger.debug("No image data available; skipping annotation generation.")
                    time.sleep(self.time_step)
                    continue
                except Exception as e:
                    self._logger.error(f"Error accessing frame queue: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self._logger.critical(f"Too many consecutive errors ({consecutive_errors}). Pausing annotation processing for 30 seconds.")
                        time.sleep(30)  # Longer pause after too many errors
                        consecutive_errors = 0  # Reset after pause
                    time.sleep(self.time_step)
                    continue

                # Check frame data validity
                if not frame_data or not isinstance(frame_data, str) or len(frame_data) < 1000:
                    self._logger.warning("Invalid frame data received")
                    time.sleep(self.time_step)
                    continue

                # Only proceed with annotation if we've confirmed video is loaded
                if video_loaded:
                    annotation = self._generate_annotation(frame_data)
                    if annotation:
                        self.annotations.append(annotation)
                        try:
                            self.append_json_to_file(annotation, self.annotation_filepath)
                            self._logger.info(f"✓ Generated annotation: phase={annotation.get('surgical_phase', 'unknown')}, tools={annotation.get('tools', [])}")
                        except Exception as e:
                            self._logger.error(f"Failed to write annotation to file: {e}")

                        # Notify that a new annotation was generated

                        # Publish annotation event via message bus
                        if self.message_bus:
                            try:
                                self.message_bus.publish_event(
                                    sender_agent=self.agent_name,
                                    event_type="annotation_created",
                                    payload=annotation
                                )
                                self._logger.debug("Published annotation_created event via message bus")
                            except Exception as e:
                                self._logger.error(f"Error publishing annotation event: {e}", exc_info=True)

                        # Legacy callback support (backward compatibility)
                        if hasattr(self, 'on_annotation_callback') and self.on_annotation_callback:
                            try:
                                self.on_annotation_callback(annotation)
                                self._logger.debug(f"Annotation callback executed successfully")
                            except Exception as callback_error:
                                self._logger.error(f"Error in annotation callback: {callback_error}")
                        else:
                            self._logger.warning(f"No annotation callback set - annotations won't be sent to UI")
                    else:
                        self._logger.warning(f"Annotation generation returned None/empty result")
            except Exception as e:
                self._logger.error(f"Error in annotation background loop: {e}", exc_info=True)
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    self._logger.critical(f"Too many consecutive errors in background loop ({consecutive_errors}). Pausing for 30 seconds.")
                    time.sleep(30)
                    consecutive_errors = 0

            # Sleep between annotation attempts
            time.sleep(self.time_step)

    def _generate_annotation(self, frame_data):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})
        # Strong, explicit instruction for JSON shape to battle model drift
        user_content = (
            "Analyze the attached surgical image and return ONLY a JSON object with EXACTLY these keys: "
            "tools (array), anatomy (array), surgical_phase (string), description (string). "
            "Use only the allowed values: tools in [scissors, hook, clipper, grasper, bipolar, irrigator, none]; "
            "anatomy in [gallbladder, cystic_duct, cystic_artery, omentum, liver, blood_vessel, abdominal_wall, peritoneum, gut, specimen_bag, none]; "
            "surgical_phase in [preparation, calots_triangle_dissection, clipping_and_cutting, gallbladder_dissection, gallbladder_packaging, cleaning_and_coagulation, gallbladder_extraction]. "
            "Use underscores (e.g., clipping_and_cutting), never hyphens. "
            "If nothing is visible for tools or anatomy, use ['none'] for that field."
        )
        messages.append({"role": "user", "content": user_content})

        # Create a fallback annotation in case of errors
        fallback_annotation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "elapsed_time_seconds": time.time() - self.procedure_start,
            "tools": ["none"],
            "anatomy": ["none"],
            "surgical_phase": "preparation",  # Default to preparation phase
            "description": "Unable to analyze the current frame due to a processing error."
        }

        # First, check if the frame data is valid
        if not frame_data or len(frame_data) < 1000:  # Arbitrary minimum length for valid image data
            self._logger.warning("Invalid or empty frame data received")
            return None

        try:
            # Try to get a response from the model with retries, using JSON Schema via response_format
            max_retries = 2
            retry_count = 0
            raw_json_str = None

            while retry_count <= max_retries and raw_json_str is None:
                try:
                    raw_json_str = self.stream_image_response(
                        prompt=self.generate_prompt(user_content, []),
                        image_b64=frame_data,
                        temperature=0.3,
                        display_output=False,  # Don't show output to user
                        grammar=self.grammar,
                    )
                except Exception as e:
                    retry_count += 1
                    self._logger.warning(f"Annotation model error (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count > max_retries:
                        self._logger.error(f"All annotation attempts failed: {e}")
                        return fallback_annotation
                    time.sleep(1)  # Wait before retry

            if not raw_json_str:
                self._logger.warning("Empty response from model")
                return fallback_annotation

            self._logger.debug(f"Raw annotation response: {raw_json_str}")

            # Robust parsing and normalization to handle model drift
            try:
                obj = json.loads(raw_json_str)
            except Exception:
                # Try to extract valid JSON if the response contains malformed output
                try:
                    import re
                    json_match = re.search(r'\{.*\}', raw_json_str, re.DOTALL)
                    if json_match:
                        obj = json.loads(json_match.group(0))
                    else:
                        return fallback_annotation
                except Exception:
                    self._logger.warning("Failed to extract valid JSON from response")
                    return fallback_annotation

            try:
                normalized = self._normalize_annotation_json(obj)
                parsed = SurgeryAnnotation(**normalized)
            except Exception as e:
                self._logger.warning(f"Annotation parse error after normalization: {e}")
                return fallback_annotation

            # Create the annotation dict with timestamp
            annotation_dict = parsed.dict()
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            annotation_dict["timestamp"] = timestamp_str
            annotation_dict["elapsed_time_seconds"] = time.time() - self.procedure_start

            return annotation_dict

        except Exception as e:
            self._logger.warning(f"Annotation generation error: {e}")
            return fallback_annotation

    def process_request(self, input_data, chat_history):
        return {
            "name": "AnnotationAgent",
            "response": "AnnotationAgent runs in the background and generates annotations only when image data is available."
        }

    def stop(self):
        self.stop_event.set()
        self._logger.info("Stopping AnnotationAgent background thread.")
        self.thread.join()

    # --- helpers ---
    def _normalize_annotation_json(self, data: dict) -> dict:
        """
        Map various likely model outputs into the expected schema fields and values.
        Accepts keys like 'Tools', 'Anatomies', 'Phase', etc., and normalizes
        values (lower‑case, underscores, enums). Ensures required fields exist.
        """
        # Allowed enums per config
        tools_enum = {"scissors", "hook", "clipper", "grasper", "bipolar", "irrigator", "none"}
        anatomy_enum = {
            "gallbladder", "cystic_duct", "cystic_artery", "omentum", "liver",
            "blood_vessel", "abdominal_wall", "peritoneum", "gut", "specimen_bag", "none"
        }
        phase_enum = {
            "preparation",
            "calots_triangle_dissection",
            "clipping_and_cutting",
            "gallbladder_dissection",
            "gallbladder_packaging",
            "cleaning_and_coagulation",
            "gallbladder_extraction",
        }

        # Key normalization: accept alternatives
        def get_any(d, keys, default=None):
            for k in keys:
                if k in d:
                    return d[k]
            return default

        raw_tools = get_any(data, ["tools", "Tools", "tool", "Tool", "instruments", "Instruments"], [])
        raw_anatomy = get_any(data, ["anatomy", "Anatomy", "anatomies", "Anatomies", "structures", "Structures"], [])
        raw_phase = get_any(data, ["surgical_phase", "SurgicalPhase", "Surgical_Phase", "Phase", "phase"], "preparation")
        raw_desc = get_any(data, ["description", "Description", "desc", "Desc"], None)

        # Normalize lists
        def to_list(v):
            if v is None:
                return []
            if isinstance(v, list):
                return v
            return [v]

        tools_list = [str(x).strip().lower() for x in to_list(raw_tools)]
        anatomy_list = [str(x).strip().lower() for x in to_list(raw_anatomy)]

        # Map common synonyms
        synonym_map_tools = {"forceps": "grasper", "grasper": "grasper", "clip-applier": "clipper"}
        tools_list = [synonym_map_tools.get(x, x) for x in tools_list]

        # Enforce enums; if empty, use ['none']
        tools_list = [x for x in tools_list if x in tools_enum]
        if not tools_list:
            tools_list = ["none"]

        anatomy_list = [x.replace(" ", "_") for x in anatomy_list]
        anatomy_list = [x for x in anatomy_list if x in anatomy_enum]
        if not anatomy_list:
            anatomy_list = ["none"]

        # Normalize phase: lower, replace hyphens/spaces with underscores
        phase = str(raw_phase).strip().lower().replace("-", "_").replace(" ", "_")
        if phase not in phase_enum:
            # Try some heuristic corrections
            if "clip" in phase and "cut" in phase:
                phase = "clipping_and_cutting"
            elif "calot" in phase or "triangle" in phase:
                phase = "calots_triangle_dissection"
            elif "pack" in phase:
                phase = "gallbladder_packaging"
            elif "dissect" in phase and "gallbladder" in phase:
                phase = "gallbladder_dissection"
            elif "clean" in phase or "coag" in phase:
                phase = "cleaning_and_coagulation"
            elif "extract" in phase:
                phase = "gallbladder_extraction"
            else:
                phase = "preparation"

        # Description: if missing, synthesize a concise one
        description = raw_desc
        if not isinstance(description, str) or not description.strip():
            # Attempt simple synthesis
            if tools_list and anatomy_list and tools_list != ["none"] and anatomy_list != ["none"]:
                description = f"{tools_list[0]} interacting with {anatomy_list[0]}"
            else:
                description = "Scene reviewed; limited identifiable details"

        return {
            "tools": sorted(list(dict.fromkeys(tools_list))),
            "anatomy": sorted(list(dict.fromkeys(anatomy_list))),
            "surgical_phase": phase,
            "description": description.strip(),
        }
