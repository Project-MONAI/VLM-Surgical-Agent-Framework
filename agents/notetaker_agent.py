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

import json
import os
import time
import logging
import base64
from .base_agent import Agent

class NotetakerAgent(Agent):
    """
    The NotetakerAgent is invoked when the user wants to make a note of something
    happening in surgery.
    """

    def __init__(self, settings_path, response_handler=None, agent_key=None, message_bus=None, procedure_start_str=None):
        super().__init__(settings_path, response_handler, agent_key=agent_key, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)

        # Overwrite the LLM wait with a no-op, as we don't need an LLM here
        self._wait_for_server = self._skip_llm_wait

        # If procedure_start_str is not provided, create one
        if procedure_start_str is None:
            procedure_start_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

        self.procedure_start_str = procedure_start_str

        # Build subfolder in e.g. "procedure_outputs/procedure_YYYY_MM_DD__HH_MM_SS"
        base_output_dir = self.agent_settings.get("notetaker_output_dir", "annotations")
        self.procedure_folder = os.path.join(base_output_dir, f"procedure_{self.procedure_start_str}")
        os.makedirs(self.procedure_folder, exist_ok=True)

        # Path to the notetaker JSON
        self.notes_filepath = os.path.join(self.procedure_folder, "notetaker_notes.json")
        self._logger.info(f"Notetaker notes file: {self.notes_filepath}")

        # Also create a folder to store note images
        self.images_subdir = os.path.join(self.procedure_folder, "note_images")
        os.makedirs(self.images_subdir, exist_ok=True)

        self.notes = []

    def _skip_llm_wait(self, timeout=60):
        self._logger.debug("NotetakerAgent does NOT need an LLM, skipping server wait.")

    def process_request(self, user_text, chat_history, visual_info=None):
        """
        Called by the app whenever the user says "take a note" or "make a note".
        We capture a note with timestamp & optional image, store in memory + JSON file,
        but decode the image from base64 into a .jpg file on disk.
        """
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        image_file = None
        if visual_info and "image_b64" in visual_info and visual_info["image_b64"]:
            # We'll decode and write the image
            image_file = self._save_image(visual_info["image_b64"], timestamp_str)

        note = {
            "timestamp": timestamp_str,
            "text": user_text,
        }
        if image_file:
            note["image_file"] = image_file

        self.notes.append(note)
        self.append_json_to_file(note, self.notes_filepath)

        # Include the actual note content in the response so the UI can display it
        response = (
            f"Note recorded (timestamp={timestamp_str}). "
            f"Total notes so far: {len(self.notes)}. "
            f"Note: {user_text}"
        )
        return {
            "name": "NotetakerAgent",
            "response": response
        }

    def _save_image(self, data_uri, timestamp_str):
        """
        Decodes a data URI (e.g. "data:image/jpeg;base64,<b64>") 
        and writes it to note_images/<unique_filename>.jpg.

        Returns the filename or None if decode failed.
        """
        try:
            # typical format: "data:image/jpeg;base64,ABCD..."
            if not data_uri.startswith("data:image/"):
                self._logger.warning(f"Skipping non-image data URI: {data_uri[:50]}...")
                return None

            header, b64_data = data_uri.split(",", 1)
            # We can guess extension from header if you want. For now, ".jpg"
            # Or if "png" in header -> .png
            extension = ".jpg"
            if "png" in header:
                extension = ".png"

            raw_bytes = base64.b64decode(b64_data)

            # build a unique name for the file
            unique_id = str(int(time.time() * 1000))[-5:]  # last 5 digits
            filename = f"note_{timestamp_str.replace(' ','_')}__{unique_id}{extension}"
            filepath = os.path.join(self.images_subdir, filename)

            with open(filepath, "wb") as f:
                f.write(raw_bytes)

            self._logger.debug(f"Image saved: {filepath}")
            # return just the relative name (or full path):
            return os.path.join("note_images", filename)
        except Exception as e:
            self._logger.error(f"Failed to decode/save image: {e}", exc_info=True)
            return None

    def get_notes(self):
        return self.notes
