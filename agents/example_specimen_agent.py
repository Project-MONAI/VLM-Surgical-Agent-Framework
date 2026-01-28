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

"""
Example Custom Agent: SpecimenAnalyzerAgent

This is a demonstration of how easy it is to add a new agent to the framework.
Simply create this file and the corresponding config, and it will be automatically
discovered and loaded!

To enable this agent:
1. Set enabled: true in configs/example_specimen_agent.yaml
2. Restart the application
3. The agent will be automatically available in the selector
"""

from agents.base_agent import Agent


class SpecimenAnalyzerAgent(Agent):
    """
    Analyzes specimen bags and gallbladder contents during the extraction phase.

    This agent specializes in:
    - Assessing specimen integrity
    - Identifying stones or pathology
    - Detecting spillage or contamination
    - Providing recommendations for pathology examination
    """

    def __init__(self, settings_path, response_handler):
        super().__init__(settings_path, response_handler)
        self._logger.info("SpecimenAnalyzerAgent initialized")

    def process_request(self, text, chat_history, visual_info=None):
        """
        Process a request about specimen analysis.

        Args:
            text: User query about the specimen
            chat_history: Conversation history
            visual_info: Optional dict with image_b64 and other metadata

        Returns:
            Dict with agent name and response
        """
        try:
            self._logger.debug(f"SpecimenAnalyzerAgent processing: {text[:100]}...")

            # Generate the prompt using base class method
            prompt = self.generate_prompt(text, chat_history)

            # Check if we have an image
            if visual_info and visual_info.get("image_b64"):
                # Use multimodal response with image
                self._logger.debug("Processing with image input")
                response = self.stream_image_response(
                    prompt=prompt,
                    image_b64=visual_info["image_b64"],
                    temperature=0.0,  # Deterministic for medical analysis
                    display_output=True
                )
            else:
                # Text-only response
                self._logger.debug("Processing text-only")
                response = self.stream_response(
                    prompt=prompt,
                    temperature=0.0,
                    display_output=True
                )

            return {
                "name": "SpecimenAnalyzerAgent",
                "response": response
            }

        except Exception as e:
            self._logger.error(f"Error in SpecimenAnalyzerAgent: {e}", exc_info=True)
            return {
                "name": "SpecimenAnalyzerAgent",
                "response": f"Error analyzing specimen: {str(e)}"
            }

