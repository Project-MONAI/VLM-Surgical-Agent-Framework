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
Surgical Agent Framework - Dynamic Agent Loading Version

This refactored version uses the AgentRegistry for dynamic agent discovery and loading.
Agents can be added by simply creating:
1. A Python file in agents/
2. A YAML config in configs/ (with agent_metadata section)

The system will automatically discover, load, and make them available.
"""

import asyncio
import re
import logging
import os
import sys
import time

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat_history import ChatHistory
from utils.response_handler import ResponseHandler
from utils.agent_registry import AgentRegistry
from agents.dynamic_selector_agent import DynamicSelectorAgent

from servers.web_server import Webserver

logging.basicConfig(level=logging.INFO)
# Reduce verbosity of third‑party libraries to avoid dumping large payloads (e.g., base64 images)
for noisy in [
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "httpcore.http11",
    "websockets",
    "websockets.server",
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Install a redaction filter to scrub base64 images and large data blobs from all logs
class _RedactDataFilter(logging.Filter):
    _patterns = [
        # data:image/...;base64,AAAA
        (re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+"), "data:image/*;base64,[REDACTED]"),
        # JSON fields like "frame_data":"data:image/jpeg;base64,...."
        (re.compile(r'("frame_data"\s*:\s*")data:image/[A-Za-z0-9.+-]+;base64,[^"]+(" )?'), r'\1data:image/*;base64,[REDACTED]"'),
        # Generic very long base64-like strings in JSON fields: "data":"AAAA..."
        (re.compile(r'("data"\s*:\s*")[A-Za-z0-9+/=\r\n]{256,}(" )?'), r'\1[REDACTED]"'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            sanitized = msg
            for pat, repl in self._patterns:
                sanitized = pat.sub(repl, sanitized)
            if sanitized != msg:
                # Overwrite the message safely
                record.msg = sanitized
                record.args = ()
        except Exception:
            pass
        return True

_redact = _RedactDataFilter()
logging.getLogger().addFilter(_redact)
logging.getLogger('websockets').addFilter(_redact)
logging.getLogger('websockets.server').addFilter(_redact)
logging.getLogger('openai').addFilter(_redact)
logging.getLogger('openai._base_client').addFilter(_redact)

async def main():
    logger = logging.getLogger(__name__)
    chat_history = ChatHistory()
    response_handler = ResponseHandler()

    # Create the webserver first so that its frame_queue is available.
    global web
    web = Webserver(web_server='0.0.0.0', web_port=8050, ws_port=49000)

    # Create a directory for uploaded videos if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'uploaded_videos'), exist_ok=True)

    # ============================================================================
    # AGENT MESSAGE BUS: Standardized Inter-Agent Communication
    # ============================================================================

    logger.info("=" * 80)
    logger.info("Initializing Agent Message Bus")
    logger.info("=" * 80)

    from utils.agent_message_bus import AgentMessageBus

    message_bus = AgentMessageBus(web_server=web, message_history_size=1000)
    logger.info("✓ Agent Message Bus created")

    # ============================================================================
    # AGENT REGISTRY: Dynamic Agent Discovery and Loading
    # ============================================================================

    logger.info("=" * 80)
    logger.info("Initializing Agent Registry for Dynamic Agent Loading")
    logger.info("=" * 80)

    registry = AgentRegistry(
        agents_dir="agents",
        configs_dir="configs",
        enable_auto_discovery=True
    )

    # Register dependency resolvers for agent instantiation
    logger.info("Registering dependency resolvers...")

    # Register message bus
    registry.register_dependency_resolver("message_bus", lambda: message_bus)
    logger.info("✓ Message bus registered as dependency")

    # Register frame queues
    registry.register_dependency_resolver("frame_queue", lambda: web.frame_queue)

    # Define annotation callback
    def on_annotation(annotation):
        # Format a simple message for the UI
        surgical_phase = annotation.get("surgical_phase", "unknown")
        tools = ", ".join(t for t in (annotation.get("tools", []) or []) if t and t != "none")
        anatomy = ", ".join(a for a in (annotation.get("anatomy", []) or []) if a and a != "none")

        message = f"Annotation: Phase '{surgical_phase}'"
        if tools:
            message += f" | Tools: {tools}"
        if anatomy:
            message += f" | Anatomy: {anatomy}"
        description = annotation.get("description")
        if description:
            message += f" | {description}"

        # Send to UI
        payload = {"agent_response": message}
        if annotation.get("source") == "operating_room":
            payload["operating_room_annotation"] = True
        web.send_message(payload)

    registry.register_dependency_resolver("on_annotation_callback", lambda: on_annotation)

    # ============================================================================
    # AGENT INSTANTIATION: Create all discovered agents
    # ============================================================================

    logger.info("=" * 80)
    logger.info("Instantiating Agents")
    logger.info("=" * 80)

    agent_names = registry.get_agent_names(enabled_only=True)
    logger.info(f"Found {len(agent_names)} enabled agents to instantiate")

    # Special handling for background agents - they need procedure_start_str
    procedure_start_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    shared_procedure_start = None

    for agent_name in agent_names:
        metadata = registry.get_metadata(agent_name)

        # Build kwargs based on agent type
        kwargs = {}

        # Add procedure_start_str for agents that need it
        if metadata.lifecycle == "background":
            kwargs['procedure_start_str'] = procedure_start_str

        # Instantiate the agent
        agent = registry.instantiate_agent(agent_name, response_handler, **kwargs)

        if agent:
            # Post-instantiation setup for background agents
            if hasattr(agent, 'on_annotation_callback'):
                if agent.on_annotation_callback is None:
                    logger.debug(f"Setting annotation callback for {agent_name} (not injected via dependencies)")
                    agent.on_annotation_callback = on_annotation
                else:
                    logger.debug(f"Annotation callback already set for {agent_name} (injected via dependencies)")

            # Sync procedure_start for multiple background agents
            if hasattr(agent, 'procedure_start'):
                if shared_procedure_start is None:
                    shared_procedure_start = agent.procedure_start
                else:
                    agent.procedure_start = shared_procedure_start
        else:
            logger.warning(f"❌ Failed to instantiate agent: {agent_name}")

    # Get all successfully instantiated agents
    agents = registry.get_all_agents()
    logger.info("=" * 80)
    logger.info(f"Successfully initialized {len(agents)} agents:")
    for name in agents.keys():
        metadata = registry.get_metadata(name)
        logger.info(f"  ✓ {name:30s} [{metadata.category}]")
    logger.info("=" * 80)

    # Print registry statistics
    stats = registry.get_statistics()
    logger.info(f"Registry Statistics: {stats}")

    # Print message bus statistics
    bus_stats = message_bus.get_statistics()
    logger.info("=" * 80)
    logger.info("Message Bus Statistics:")
    logger.info(f"  Registered Agents: {bus_stats['registered_agents']}")
    logger.info(f"  Pending Requests: {bus_stats['pending_requests']}")
    logger.info(f"  Active Workflows: {bus_stats['active_workflows']}")
    logger.info(f"  Message History: {bus_stats['total_messages']}")
    logger.info("=" * 80)

    # ============================================================================
    # SELECTOR AGENT: Create dynamic selector with registry reference
    # ============================================================================

    logger.info("Creating Dynamic Selector Agent...")
    selector_agent = DynamicSelectorAgent(
        "configs/selector.yaml",
        response_handler,
        agent_registry=registry
    )
    logger.info("✓ Dynamic Selector Agent ready")

    # ============================================================================
    # MESSAGE CALLBACK: Handle user input and route to agents
    # ============================================================================

    def msg_callback(payload, msg_type, timestamp):
        """
        Called when the user manually types input or when the webserver passes along an ASR transcript.
        """
        # Special case for summary generation request
        if 'summary_request' in payload and 'user_input' in payload:
            user_text = payload['user_input']
            annotations_data = payload.get('annotations_data', [])
            notes_data = payload.get('notes_data', [])

            logging.debug(f"Processing summary request with {len(annotations_data)} annotations and {len(notes_data)} notes")

            # Build a context-rich prompt for the ChatAgent
            summary_prompt = f"""
Generate a comprehensive procedure summary based on the following data:

ANNOTATIONS:
{annotations_data}

NOTES:
{notes_data}

Format the summary as a structured medical report including:
1. Procedure overview
2. Key phases identified
3. Tools and equipment used
4. Anatomical structures involved
5. Notable observations

Include all relevant clinical details captured in the annotations and notes.

IMPORTANT: This is a TEXT-ONLY SUMMARY request. Do not attempt to identify instruments in any attached image - focus only on summarizing the data provided above.
"""
            # Add to chat history
            chat_history.add_user_message(summary_prompt)
            
            # Check if we have a recent frame to include with the summary request
            frame_data = None
            if hasattr(web, 'lastProcessedFrame') and web.lastProcessedFrame:
                frame_data = web.lastProcessedFrame
                logging.debug("Including last processed frame with summary request")
            
            # Use the chat agent directly for summaries
            chat_agent = agents.get("ChatAgent")
            if chat_agent:
                visual_info = {"image_b64": frame_data, "tool_labels": {}} if frame_data else None
                response_data = chat_agent.process_request(
                    summary_prompt, chat_history.to_list(), visual_info
                )

                # Add response to chat history
                chat_history.add_bot_message(response_data["response"])

                # Send result to UI with special flag for summary
                web.send_message({
                    "agent_response": response_data["response"],
                    "agent_name": response_data.get("name", "AI Assistant"),
                    "summary_response": True
                })
            return
            
        elif 'user_input' in payload:
            user_text = payload['user_input']
            logging.debug(f"Processing user input: {user_text}")
            
            # Also add the user message to chat history if it has the asr_final flag
            # This ensures we record the first message from the microphone
            if payload.get('asr_final', False) and not chat_history.has_message(user_text):
                chat_history.add_user_message(user_text)
                
            try:
                # Let the selector decide which agent to pick
                selected_agent_name, corrected_text, selection_context = selector_agent.process_request(
                    user_text, chat_history.to_list()
                )
                if not selected_agent_name:
                    logging.error("No agent selected by selector for user_input.")
                    return

                # Check for frame data directly in the payload
                frame_data = payload.get('frame_data')
                
                # If not there, try to get it from the frame_queue
                if not frame_data and not web.frame_queue.empty():
                    try:
                        frame_data = web.frame_queue.get_nowait()
                    except Exception as e:
                        logging.error(f"Error retrieving frame data: {e}")
                
                # If still no frame, check if there's a lastProcessedFrame in web
                if not frame_data and hasattr(web, 'lastProcessedFrame') and web.lastProcessedFrame:
                    frame_data = web.lastProcessedFrame
                    logging.debug("Using web's last processed frame")
                
                # If we have a frame, store it for future use
                if frame_data:
                    web.lastProcessedFrame = frame_data

                # Pass the image (if any) along with empty tool labels
                visual_info = {"image_b64": frame_data, "tool_labels": {}}

                # If user input triggers PostOpNoteAgent, do final note generation
                if selected_agent_name == "PostOpNoteAgent":
                    # Stop the background annotation agents
                    for name, agent in agents.items():
                        metadata = registry.get_metadata(name)
                        if metadata and metadata.lifecycle == "background":
                            if hasattr(agent, 'stop'):
                                try:
                                    agent.stop()
                                    logger.info(f"Stopped background agent: {name}")
                                except Exception as e:
                                    logger.error(f"Error stopping {name}: {e}")

                    # Determine the procedure folder from annotation_agent
                    annotation_agent = agents.get("AnnotationAgent")
                    if annotation_agent and hasattr(annotation_agent, 'annotation_filepath'):
                        procedure_folder = os.path.dirname(annotation_agent.annotation_filepath)
                    else:
                        procedure_folder = None

                    post_op_agent = agents.get("PostOpNoteAgent")
                    if post_op_agent and procedure_folder:
                        final_json = post_op_agent.generate_post_op_note(procedure_folder)
                        if final_json is None:
                            response_data = {
                                "name": "PostOpNoteAgent",
                                "response": "Failed to create final post-op note. Check logs."
                            }
                        else:
                            response_data = {
                                "name": "PostOpNoteAgent",
                                "response": "Final post-op note created. See post_op_note.json in the procedure folder."
                            }
                            # Also send the structured note to the UI Summary tab via WebSocket
                            try:
                                web.send_message({
                                    "post_op_note": final_json,
                                    "summary_response": True
                                })
                            except Exception as ws_e:
                                logging.error(f"Failed to send post-op note JSON to UI: {ws_e}")
                    else:
                        response_data = {
                            "name": "PostOpNoteAgent",
                            "response": "PostOpNoteAgent not available or procedure folder not found."
                        }
                else:
                    # Get agent from registry
                    agent = agents.get(selected_agent_name)
                    if agent:
                        response_data = agent.process_request(
                            corrected_text, chat_history.to_list(), visual_info
                        )
                    else:
                        response_data = {
                            "name": selected_agent_name,
                            "response": f"Agent '{selected_agent_name}' not available."
                        }

                # Update chat history - only add user message if it's not already there
                if not chat_history.has_message(corrected_text):
                    chat_history.add_user_message(corrected_text)
                chat_history.add_bot_message(response_data["response"])

                # Check if this is from the NotetakerAgent to tag it for the UI
                if selected_agent_name == "NotetakerAgent":
                    # Pass along the original user input so the UI can infer note content/title
                    web.send_message({
                        "agent_response": response_data["response"],
                        "agent_name": response_data.get("name", "AI Assistant"),
                        "is_note": True,
                        "original_user_input": payload.get('original_user_input', user_text),
                        "user_input": corrected_text,
                    })
                else:
                    # Send result to UI
                    web.send_message({
                        "agent_response": response_data["response"],
                        "agent_name": response_data.get("name", "AI Assistant"),
                    })
            except Exception as e:
                logging.error(f"Error processing user_input: {e}", exc_info=True)

    # Set callback and start web server
    web.msg_callback = msg_callback
    web.start()

    logger.info("=" * 80)
    logger.info("🚀 Surgical Agent Framework Started Successfully")
    logger.info("=" * 80)
    logger.info("Web UI: http://localhost:8050")
    logger.info(f"Available agents: {', '.join(agents.keys())}")
    logger.info("=" * 80)

    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        logging.info("Shutting down gracefully.")

        # Cleanup agents
        for name, agent in agents.items():
            if hasattr(agent, 'stop'):
                try:
                    agent.stop()
                    logger.info(f"Stopped agent: {name}")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())

