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

import logging
from typing import Any, Dict, List

from .chat_agent import ChatAgent


class OperatingRoomAgent(ChatAgent):
    """Answers questions about the live operating room webcam feed."""

    def __init__(self, settings_path: str, response_handler=None):
        super().__init__(settings_path, response_handler)
        self._logger = logging.getLogger(__name__)

    def process_request(
        self,
        text: str,
        chat_history: List[Any],
        visual_info: Dict[str, Any] | None = None,
    ):
        response = super().process_request(text, chat_history, visual_info)
        if isinstance(response, dict):
            response["name"] = "OperatingRoomAgent"
        return response
