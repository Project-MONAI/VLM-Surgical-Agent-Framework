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
Dynamic Selector Agent

Adapts to available agents automatically. When new agents are added to the
registry, they become available for selection without code changes.
"""

import logging
import json
from typing import Optional, List
from pydantic import BaseModel, Field, create_model
from .base_agent import Agent


class DynamicSelectorAgent(Agent):
    """
    Selector agent that dynamically adapts to available agents.
    Automatically updates its selection options based on the agent registry.
    """

    def __init__(self, settings_path, response_handler, agent_registry=None, message_bus=None):
        super().__init__(settings_path, response_handler, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)
        self.agent_registry = agent_registry

        # Build dynamic Pydantic model for validation
        if agent_registry:
            self._build_dynamic_model()
        else:
            # Fallback to static model if no registry provided
            self._build_static_model()

    def _build_static_model(self):
        """Build static model for backward compatibility"""
        from typing import Literal

        # Static list of agents (fallback)
        agent_names = [
            "ChatAgent",
            "NotetakerAgent",
            "PostOpNoteAgent",
            "EHRAgent",
        ]

        SelectionEnum = Literal[tuple(agent_names)]
        ContextEnum = Literal["procedure", "operating_room"]

        self.SelectorOutput = create_model(
            'SelectorOutput',
            corrected_input=(str, Field(..., description="Corrected user input")),
            selection=(SelectionEnum, Field(..., description="Selected agent")),
            context=(Optional[ContextEnum], Field(None, description="Context"))
        )

        self._logger.warning("DynamicSelectorAgent initialized without registry, using static agent list")

    def _build_dynamic_model(self):
        """Build a dynamic Pydantic model based on available agents"""
        if not self.agent_registry:
            return

        agent_names = self.agent_registry.get_agent_names(enabled_only=True)

        if not agent_names:
            self._logger.warning("No agents found in registry, using static model")
            self._build_static_model()
            return

        # Create dynamic Literal type for agent selection
        from typing import Literal
        SelectionEnum = Literal[tuple(agent_names)]
        ContextEnum = Literal["procedure", "operating_room"]

        # Create dynamic model
        self.SelectorOutput = create_model(
            'SelectorOutput',
            corrected_input=(str, Field(..., description="Corrected user input")),
            selection=(SelectionEnum, Field(..., description="Selected agent")),
            context=(Optional[ContextEnum], Field(None, description="Context"))
        )

        # Update grammar for LLM
        self._update_grammar(agent_names)

        self._logger.info(f"DynamicSelectorAgent configured with {len(agent_names)} agents")

    def _update_grammar(self, agent_names: List[str]):
        """Update the grammar JSON schema with current agent list"""
        schema = {
            "type": "object",
            "properties": {
                "corrected_input": {"type": "string"},
                "selection": {
                    "type": "string",
                    "enum": agent_names
                },
                "context": {
                    "type": ["string", "null"],
                    "enum": ["procedure", "operating_room", None]
                }
            },
            "required": ["corrected_input", "selection"]
        }
        self.grammar = json.dumps(schema)

        # Update agent_prompt to include agent descriptions
        if self.agent_registry:
            self._update_agent_prompt(agent_names)

    def _update_agent_prompt(self, agent_names: List[str]):
        """Update the agent prompt with current agent descriptions"""
        agent_descriptions = []
        for name in agent_names:
            metadata = self.agent_registry.get_metadata(name)
            if metadata and metadata.description:
                agent_descriptions.append(f"    {name}: {metadata.description}")
            else:
                agent_descriptions.append(f"    {name}")

        agents_section = "\n".join(agent_descriptions)

        # Try to update the agent list in the prompt
        # Look for the section that lists agents
        if "The agents you may select from are:" in self.agent_prompt:
            parts = self.agent_prompt.split("The agents you may select from are:")
            if len(parts) > 1:
                # Find the end of the agent list (usually marked by double newline)
                remaining = parts[1]
                # Find the next paragraph (after agent list)
                next_section_idx = remaining.find("\n\n")
                if next_section_idx > 0:
                    after_list = remaining[next_section_idx:]
                    self.agent_prompt = (
                        parts[0] + 
                        "The agents you may select from are:\n" +
                        agents_section + 
                        after_list
                    )
                    self._logger.debug("Updated agent prompt with current agent list")

    def process_request(self, text, chat_history):
        """Process request using dynamic agent list"""
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})

        user_text = (
            f"User said: {text}\n\n"
            "Return JSON matching the schema:"
            '\n{"corrected_input": "...", "selection": "AgentName", "context": "procedure"}'
            "\nOnly include the optional context field when you are confident whether "
            "the request refers to the surgical procedure feed or the operating room webcam."
        )
        messages.append({"role": "user", "content": user_text})

        self._logger.debug(f"DynamicSelectorAgent processing: {text[:100]}...")

        try:
            schema = json.loads(self.grammar) if isinstance(self.grammar, str) else self.grammar
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "selector_output",
                    "schema": schema,
                    "strict": True,
                },
            }
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=self.ctx_length,
                response_format=response_format,
                extra_body={"guided_json": schema},
            )
            raw_json_str = result.choices[0].message.content
            self._logger.debug(f"Raw JSON from vLLM: {raw_json_str}")
            raw_json_str = raw_json_str.replace("\\'", "'")

            parsed = self.SelectorOutput.model_validate_json(raw_json_str)
            selected_agent = parsed.selection
            corrected_text = parsed.corrected_input
            selection_context = parsed.context

            self._logger.info(
                f"Selected: {selected_agent} (context={selection_context})"
            )
            return selected_agent, corrected_text, selection_context

        except Exception as e:
            self._logger.error(f"Error in DynamicSelectorAgent.process_request: {e}", exc_info=True)
            return None, None, None

    def refresh_agents(self):
        """
        Refresh the list of available agents from the registry.
        Useful for hot-reloading new agents without restarting.
        """
        if not self.agent_registry:
            self._logger.warning("Cannot refresh agents: no registry available")
            return

        # Rediscover agents
        discovered = self.agent_registry.discover_agents()
        self._logger.info(f"Refreshed agent list: {len(discovered)} agents")

        # Rebuild the model
        self._build_dynamic_model()
