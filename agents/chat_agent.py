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
import logging
import base64
from .base_agent import Agent

class ChatAgent(Agent):
    """
    A general chat agent that can optionally handle images 
    """

    def __init__(self, settings_path, response_handler, message_bus=None):
        super().__init__(settings_path, response_handler, message_bus=message_bus)

    def process_request(self, text, chat_history, visual_info=None):
        """
        Process a user request that may have an image in visual_info["image_b64"].
        If there's image data, we call stream_image_response in the base agent,
        otherwise we call stream_response.
        """
        try:
            self._logger.debug("Starting ChatAgent process_request")
            self._logger.debug(f"Input text: {text}")

            if not visual_info:
                visual_info = {}
            image_b64 = visual_info.get("image_b64", None)
            tool_labels = visual_info.get("tool_labels", {})

            # Possibly unify text with tool labels
            final_user_message = self.generate_user_prompt(text, tool_labels)
            prompt = self.generate_full_prompt(final_user_message, chat_history)

            if image_b64:
                self._logger.debug("Received image data, calling stream_image_response.")
                response = self.stream_image_response(
                    prompt=prompt,
                    image_b64=image_b64,
                    temperature=0.0
                )
            else:
                # If no image, just do a normal text-only request
                self._logger.debug("No image data, calling stream_response.")
                response = self.stream_response(
                    prompt=prompt,
                    temperature=0.0
                )
            
            return {"name": "ChatAgent", "response": response}

        except Exception as e:
            self._logger.error(f"Error in ChatAgent.process_request: {e}", exc_info=True)
            return {"name": "ChatAgent", "response": f"Error: {str(e)}"}

    def generate_user_prompt(self, text, tool_labels):
        user_prompt_template = self.agent_settings.get('user_prompt', '')
        
        # Make tool-related queries more explicit to ensure the model understands
        if "tool" in text.lower() or "instrument" in text.lower():
            # Add explicit instruction to ensure model knows to look at the image
            text = f"{text} (refer to the surgical image attached to this message)"
        
        user_prompt_filled = user_prompt_template.replace("{tool_labels}", "").replace("{text}", text)
        self._logger.debug(f"Generated user prompt: {user_prompt_filled}")
        return user_prompt_filled


    def generate_full_prompt(self, final_user_message, chat_history):
        """
        Leverages base class's generate_prompt method.
        """
        return self.generate_prompt(final_user_message, chat_history)

    def save_base64_image(self, base64_str, filename="tmp.png"):
        """
        Helper method if you want to debug images by saving them to disk.
        Not currently used in the logic above, but you can call it if needed.
        """
        try:
            prefix = "data:image/jpeg;base64,"
            if base64_str.startswith(prefix):
                base64_str = base64_str[len(prefix):]
            image_data = base64.b64decode(base64_str)
            with open(filename, 'wb') as f:
                f.write(image_data)
            self._logger.debug(f"Saved debug image to {filename}")
        except Exception as e:
            self._logger.error(f"Failed to save debug image: {e}")