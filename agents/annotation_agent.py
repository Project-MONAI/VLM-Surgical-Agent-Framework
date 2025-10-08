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
    def __init__(self, settings_path, response_handler, frame_queue, agent_key=None, procedure_start_str=None):
        super().__init__(settings_path, response_handler, agent_key=agent_key)
        self._logger = logging.getLogger(__name__)
        self.frame_queue = frame_queue  
        self.time_step = self.agent_settings.get("time_step_seconds", 10)

        # Allow per-agent configuration of normalization behaviour
        allowed_tools = self.agent_settings.get("allowed_tools")
        if isinstance(allowed_tools, list):
            self.tools_enum = {str(item).strip().lower() for item in allowed_tools if str(item).strip()}
        elif isinstance(allowed_tools, str) and allowed_tools.strip().lower() == "any":
            self.tools_enum = None
        else:
            self.tools_enum = {
                "scissors", "hook", "clipper", "grasper", "bipolar", "irrigator", "none"
            }

        allowed_anatomy = self.agent_settings.get("allowed_anatomy")
        if isinstance(allowed_anatomy, list):
            self.anatomy_enum = {str(item).strip().lower().replace(" ", "_") for item in allowed_anatomy if str(item).strip()}
        elif isinstance(allowed_anatomy, str) and allowed_anatomy.strip().lower() == "any":
            self.anatomy_enum = None
        else:
            self.anatomy_enum = {
                "gallbladder", "cystic_duct", "cystic_artery", "omentum", "liver",
                "blood_vessel", "abdominal_wall", "peritoneum", "gut", "specimen_bag", "none"
            }

        allowed_phases = self.agent_settings.get("allowed_phases")
        if isinstance(allowed_phases, list):
            self.phase_enum = {str(item).strip().lower().replace(" ", "_") for item in allowed_phases if str(item).strip()}
        elif isinstance(allowed_phases, str) and allowed_phases.strip().lower() == "any":
            self.phase_enum = None
        else:
            self.phase_enum = {
                "preparation",
                "calots_triangle_dissection",
                "clipping_and_cutting",
                "gallbladder_dissection",
                "gallbladder_packaging",
                "cleaning_and_coagulation",
                "gallbladder_extraction",
            }

        def _coerce_list(value, fallback):
            if value is None:
                return list(fallback)
            if isinstance(value, list):
                return value
            return [value]

        self.default_tools_value = _coerce_list(
            self.agent_settings.get("default_tools_value", ["none"] if self.tools_enum is not None else []),
            ["none"] if self.tools_enum is not None else []
        )
        self.default_anatomy_value = _coerce_list(
            self.agent_settings.get("default_anatomy_value", ["none"] if self.anatomy_enum is not None else []),
            ["none"] if self.anatomy_enum is not None else []
        )

        self.default_phase = str(self.agent_settings.get("default_phase", "preparation")).strip()
        if not self.default_phase:
            self.default_phase = "preparation"

        self.scene_fallback_description = self.agent_settings.get(
            "default_description",
            "Scene reviewed; limited identifiable details"
        )

        self.annotation_source = self.agent_settings.get("annotation_source")

        self._tool_synonym_map = {
            "forceps": "grasper",
            "grasper": "grasper",
            "clip-applier": "clipper",
        }

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
                            self._logger.debug(f"New annotation appended to file {self.annotation_filepath}")
                        except Exception as e:
                            self._logger.error(f"Failed to write annotation to file: {e}")
                            
                        # Notify that a new annotation was generated
                        if hasattr(self, 'on_annotation_callback') and self.on_annotation_callback:
                            try:
                                self.on_annotation_callback(annotation)
                            except Exception as callback_error:
                                self._logger.error(f"Error in annotation callback: {callback_error}")
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
            if self.annotation_source:
                annotation_dict["source"] = self.annotation_source

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

        raw_tools_list = to_list(raw_tools)
        raw_anatomy_list = to_list(raw_anatomy)

        if self.tools_enum is None:
            tools_list = [str(x).strip() for x in raw_tools_list if str(x).strip()]
            if not tools_list:
                tools_list = list(self.default_tools_value)
            tools_list = list(dict.fromkeys(tools_list))
        else:
            tools_list = []
            for item in raw_tools_list:
                token = str(item).strip().lower()
                token = self._tool_synonym_map.get(token, token)
                if token in self.tools_enum:
                    tools_list.append(token)
            if not tools_list:
                tools_list = [t for t in self.default_tools_value if t in self.tools_enum]
            if not tools_list:
                tools_list = [next(iter(self.tools_enum))]
            tools_list = sorted(list(dict.fromkeys(tools_list)))

        if self.anatomy_enum is None:
            anatomy_list = [str(x).strip() for x in raw_anatomy_list if str(x).strip()]
            if not anatomy_list:
                anatomy_list = list(self.default_anatomy_value)
            anatomy_list = list(dict.fromkeys(anatomy_list))
        else:
            anatomy_list = []
            for item in raw_anatomy_list:
                token = str(item).strip().lower().replace(" ", "_")
                if token in self.anatomy_enum:
                    anatomy_list.append(token)
            if not anatomy_list:
                anatomy_list = [a for a in self.default_anatomy_value if a in self.anatomy_enum]
            if not anatomy_list:
                anatomy_list = [next(iter(self.anatomy_enum))]
            anatomy_list = sorted(list(dict.fromkeys(anatomy_list)))

        phase = str(raw_phase).strip()
        if not phase:
            phase = self.default_phase

        if self.phase_enum is not None:
            phase_normalized = phase.lower().replace("-", "_").replace(" ", "_")
            if phase_normalized not in self.phase_enum:
                if "clip" in phase_normalized and "cut" in phase_normalized:
                    phase_normalized = "clipping_and_cutting"
                elif "calot" in phase_normalized or "triangle" in phase_normalized:
                    phase_normalized = "calots_triangle_dissection"
                elif "pack" in phase_normalized:
                    phase_normalized = "gallbladder_packaging"
                elif "dissect" in phase_normalized and "gallbladder" in phase_normalized:
                    phase_normalized = "gallbladder_dissection"
                elif "clean" in phase_normalized or "coag" in phase_normalized:
                    phase_normalized = "cleaning_and_coagulation"
                elif "extract" in phase_normalized:
                    phase_normalized = "gallbladder_extraction"
            if phase_normalized not in self.phase_enum:
                fallback = self.default_phase.strip().lower().replace("-", "_").replace(" ", "_") if self.default_phase else ""
                phase_normalized = fallback if fallback in self.phase_enum else next(iter(self.phase_enum))
            phase = phase_normalized
        else:
            phase = phase.strip()
            if not phase:
                phase = self.default_phase

        # Description: if missing, synthesize a concise one
        description = raw_desc
        if not isinstance(description, str) or not description.strip():
            # Attempt simple synthesis
            if tools_list and anatomy_list and tools_list != ["none"] and anatomy_list != ["none"]:
                description = f"{tools_list[0]} interacting with {anatomy_list[0]}"
            else:
                description = self.scene_fallback_description

        return {
            "tools": list(dict.fromkeys(tools_list)),
            "anatomy": list(dict.fromkeys(anatomy_list)),
            "surgical_phase": phase,
            "description": description.strip(),
        }
