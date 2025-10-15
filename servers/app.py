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

import asyncio
import re
import logging
import os
import sys
import time
from threading import Thread

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat_history import ChatHistory
from utils.response_handler import ResponseHandler

from agents.selector_agent import SelectorAgent
from agents.annotation_agent import AnnotationAgent
from agents.chat_agent import ChatAgent
from agents.operating_room_agent import OperatingRoomAgent
from agents.robot_control_agent import RobotControlAgent
from agents.notetaker_agent import NotetakerAgent
from agents.post_op_note_agent import PostOpNoteAgent
from agents.ehr_agent import EHRAgent

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
    chat_history = ChatHistory()
    response_handler = ResponseHandler()

    # Define the callback for messages coming from the WebSocket.
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

                if selection_context is None:
                    selection_context = (
                        "operating_room" if selected_agent_name == "OperatingRoomAgent" else "procedure"
                    )

                def _fetch_procedure_frame():
                    frame = payload.get('frame_data')
                    if not frame and not web.frame_queue.empty():
                        try:
                            frame = web.frame_queue.get_nowait()
                        except Exception as e:
                            logging.error(f"Error retrieving procedure frame data: {e}")
                    if not frame and getattr(web, 'lastProcessedFrame', None):
                        frame = web.lastProcessedFrame
                        logging.debug("Using cached procedure frame")
                    if frame:
                        web.lastProcessedFrame = frame
                    return frame

                def _fetch_operating_room_frame():
                    frame = payload.get('operating_room_frame_data')
                    queue_obj = getattr(web, 'operating_room_frame_queue', None)
                    if not frame and queue_obj is not None and not queue_obj.empty():
                        try:
                            frame = queue_obj.get_nowait()
                        except Exception as e:
                            logging.error(f"Error retrieving operating room frame data: {e}")
                    last_or_frame = getattr(web, 'lastOperatingRoomFrame', None)
                    if not frame and last_or_frame:
                        frame = last_or_frame
                        logging.debug("Using cached operating room frame")
                    if frame:
                        setattr(web, 'lastOperatingRoomFrame', frame)
                    return frame

                if selection_context == "operating_room":
                    frame_data = _fetch_operating_room_frame()
                    if not frame_data:
                        logging.debug(
                            "No operating room frame available; falling back to procedure frame for request."
                        )
                        frame_data = _fetch_procedure_frame()
                else:
                    frame_data = _fetch_procedure_frame()

                # Pass the image (if any) along with metadata about the originating feed
                visual_info = {
                    "image_b64": frame_data,
                    "tool_labels": {},
                    "feed": selection_context,
                }

                if selected_agent_name == "NotetakerAgent":
                    visual_info["note_category"] = (
                        "operating_room" if selection_context == "operating_room" else "procedure"
                    )

                # If user input triggers PostOpNoteAgent, do final note generation
                if selected_agent_name == "PostOpNoteAgent":
                    # Stop the background annotation
                    annotation_agent.stop()
                    operating_room_annotation_agent.stop()

                    # Determine the procedure folder from annotation_agent
                    procedure_folder = os.path.dirname(annotation_agent.annotation_filepath)
                    
                    final_json = post_op_note_agent.generate_post_op_note(procedure_folder)
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
                    agent = agents.get(selected_agent_name)
                    if agent:
                        response_data = agent.process_request(
                            corrected_text, chat_history.to_list(), visual_info
                        )
                    else:
                        response_data = {
                            "name": selected_agent_name,
                            "response": f"Agent '{selected_agent_name}' not implemented."
                        }

                # Update chat history - only add user message if it's not already there
                if not chat_history.has_message(corrected_text):
                    chat_history.add_user_message(corrected_text)
                chat_history.add_bot_message(response_data["response"])

                # Check if this is from the NotetakerAgent to tag it for the UI
                if selected_agent_name == "NotetakerAgent":
                    # Pass along the original user input so the UI can infer note content/title
                    note_category = response_data.get(
                        "note_category", visual_info.get("note_category", "procedure")
                    )
                    web.send_message({
                        "agent_response": response_data["response"],
                        "agent_name": response_data.get("name", "AI Assistant"),
                        "is_note": True,
                        "original_user_input": payload.get('original_user_input', user_text),
                        "user_input": corrected_text,
                        "note_category": note_category,
                    })
                else:
                    # Send result to UI
                    web.send_message({
                        "agent_response": response_data["response"],
                        "agent_name": response_data.get("name", "AI Assistant"),
                    })

            except Exception as e:
                logging.error(f"Error processing user_input: {e}", exc_info=True)

    # Create the webserver first so that its frame_queue is available.
    global web
    web = Webserver(web_server='0.0.0.0', web_port=8050, ws_port=49000, msg_callback=msg_callback)
    
    # Create a directory for uploaded videos if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'uploaded_videos'), exist_ok=True)
    web.start()

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
    
    # Now create agents, passing web.frame_queue to the AnnotationAgent.
    selector_agent = SelectorAgent("configs/selector.yaml", response_handler)
    annotation_agent = AnnotationAgent("configs/annotation_agent.yaml", response_handler, frame_queue=web.frame_queue)
    annotation_agent.on_annotation_callback = on_annotation
    chat_agent = ChatAgent("configs/chat_agent.yaml", response_handler)
    operating_room_agent = OperatingRoomAgent("configs/operating_room_agent.yaml", response_handler)
    robot_control_agent = RobotControlAgent("configs/robot_control_agent.yaml", response_handler)
    notetaker_agent = NotetakerAgent("configs/notetaker_agent.yaml", response_handler)
    post_op_note_agent = PostOpNoteAgent("configs/post_op_note_agent.yaml", response_handler)
    ehr_agent = EHRAgent("configs/ehr_agent.yaml", response_handler)

    operating_room_annotation_agent = AnnotationAgent(
        "configs/operating_room_annotation_agent.yaml",
        response_handler,
        frame_queue=web.operating_room_frame_queue,
        procedure_start_str=annotation_agent.procedure_start_str,
    )
    operating_room_annotation_agent.procedure_start = annotation_agent.procedure_start
    operating_room_annotation_agent.on_annotation_callback = on_annotation

    agents = {
        "ChatAgent": chat_agent,
        "OperatingRoomAgent": operating_room_agent,
        "RobotControlAgent": robot_control_agent,
        "NotetakerAgent": notetaker_agent,
        "PostOpNoteAgent": post_op_note_agent,
        "EHRAgent": ehr_agent,
    }

    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        logging.info("Shutting down gracefully.")

if __name__ == "__main__":
    asyncio.run(main())
