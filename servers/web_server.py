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

import base64
import flask
import json
import time
import socket
import threading
import queue
import websockets
import logging
import requests
import os
import uuid
import sys

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websockets.sync.server import serve as websocket_serve
from flask import request, jsonify, redirect, url_for
from agents.post_op_note_agent import PostOpNoteAgent

class Webserver(threading.Thread):
    def __init__(self, web_server='0.0.0.0', web_port=8050, ws_port=49000,
                 audio_ws_port=49001, msg_callback=None):
        super().__init__(daemon=True)
        self.host = web_server
        self.port = web_port
        self.msg_callback = msg_callback
        self.audio_ws_port = audio_ws_port
        self.frame_queue = queue.Queue()
        
        # Create videos directory if it doesn't exist
        self.videos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploaded_videos')
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # Current video path
        self.current_video_path = None
        
        # Store the most recent frame for follow-up questions
        self.lastProcessedFrame = None
        
        # Initialize the post-op note agent
        try:
            post_op_note_settings = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                               'configs/post_op_note_agent.yaml')
            self.post_op_note_agent = PostOpNoteAgent(post_op_note_settings)
            self._logger = logging.getLogger(__name__)
            self._logger.info("Post-op note agent initialized successfully")
        except Exception as e:
            self._logger = logging.getLogger(__name__)
            self._logger.error(f"Failed to initialize post-op note agent: {e}", exc_info=True)
            self.post_op_note_agent = None

        self.app = flask.Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web/templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web/static'))
        self.app.add_url_rule('/', view_func=self.on_index, methods=['GET'])
        self.app.add_url_rule('/api/tts', view_func=self.tts_route, methods=['POST'])
        self.app.add_url_rule('/api/upload_video', view_func=self.upload_video_route, methods=['POST'])
        self.app.add_url_rule('/api/videos', view_func=self.list_videos_route, methods=['GET'])
        self.app.add_url_rule('/api/select_video', view_func=self.select_video_route, methods=['POST'])
        self.app.add_url_rule('/api/generate_post_op_note', view_func=self.generate_post_op_note_route, methods=['POST'])
        self.app.add_url_rule('/videos/<path:filename>', view_func=self.serve_video, methods=['GET'])

        # For text messages from WebSocket - make sure we listen on all interfaces
        self.ws_queue = queue.Queue()
        # Configure WebSocket with longer ping timeout and interval for more reliability
        self.ws_server = websocket_serve(
            self.on_websocket, 
            host='0.0.0.0', 
            port=ws_port,
            ping_interval=30,  # Send ping every 30 seconds (default is 20)
            ping_timeout=60,   # Wait 60 seconds for pong response (default is 20)
            max_size=10485760  # Increase max message size to 10MB for frame data
        )
        self._logger.info(f"WebSocket server listening on 0.0.0.0:{ws_port}")
        self.ws_thread = threading.Thread(target=lambda: self.ws_server.serve_forever(), daemon=True)

        # For single-chunk audio - make sure we listen on all interfaces
        self.audio_ws_server = websocket_serve(
            self.on_audio_websocket, 
            host='0.0.0.0', 
            port=self.audio_ws_port,
            ping_interval=30,  # Send ping every 30 seconds
            ping_timeout=60    # Wait 60 seconds for pong response
        )
        self._logger.info(f"Audio WebSocket server listening on 0.0.0.0:{self.audio_ws_port}")
        self.audio_ws_thread = threading.Thread(target=lambda: self.audio_ws_server.serve_forever(), daemon=True)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

        # Create a socket to talk to Whisper. We'll reconnect for each new audio session.
        self.create_whisper_socket()

    def create_whisper_socket(self):
        """Create a fresh socket to the Whisper server (port 43001)."""
        self.whisper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.whisper_socket.connect(('localhost', 43001))
            self._logger.debug("Created + connected fresh socket to Whisper server.")
        except ConnectionRefusedError:
            self._logger.error("Connection to Whisper server refused. Is the Whisper server running?")
            raise
        except Exception as e:
            self._logger.error(f"Error connecting to Whisper server: {e}")
            raise
            
    def delayed_socket_recreation(self, delay=5):
        """Try to recreate the whisper socket after a delay"""
        time.sleep(delay)
        try:
            self._logger.info(f"Attempting to reconnect to Whisper server after {delay}s delay")
            self.create_whisper_socket()
            self._logger.info("Successfully reconnected to Whisper server")
        except Exception as e:
            self._logger.error(f"Failed to reconnect to Whisper server: {e}")
            # Try again with a longer delay
            threading.Thread(target=lambda: self.delayed_socket_recreation(delay*2), daemon=True).start()

    def on_index(self):
        video_src = None  # Will use default in template
        if self.current_video_path:
            video_filename = os.path.basename(self.current_video_path)
            video_src = f'/videos/{video_filename}'
        return flask.render_template('index.html', video_src=video_src)
        
    def serve_video(self, filename):
        """Serve uploaded videos"""
        return flask.send_from_directory(self.videos_dir, filename)
        
    def upload_video_route(self):
        """Handle video upload"""
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Get original filename and extension
        original_filename = os.path.basename(video_file.filename)
        base_name, file_ext = os.path.splitext(original_filename)
        
        # Create a safe filename (handle duplicates)
        final_filename = original_filename
        counter = 1
        
        # Check if file exists, if so, add incremental suffix
        while os.path.exists(os.path.join(self.videos_dir, final_filename)):
            final_filename = f"{base_name}_{counter}{file_ext}"
            counter += 1
        
        # Save the video file with the final name
        video_path = os.path.join(self.videos_dir, final_filename)
        video_file.save(video_path)
        
        # Update current video path
        self.current_video_path = video_path
        
        # Send message to client to update video source
        self.send_message({
            "video_updated": True,
            "video_src": f"/videos/{final_filename}"
        })
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "video_src": f"/videos/{final_filename}",
            "filename": final_filename
        })
        
    def list_videos_route(self):
        """List all uploaded videos"""
        videos = []
        for filename in os.listdir(self.videos_dir):
            if os.path.isfile(os.path.join(self.videos_dir, filename)) and filename.lower().endswith(('.mp4', '.webm', '.mov', '.avi')):
                videos.append({
                    "filename": filename,
                    "video_src": f"/videos/{filename}",
                    "size": os.path.getsize(os.path.join(self.videos_dir, filename)),
                    "modified": os.path.getmtime(os.path.join(self.videos_dir, filename))
                })
        
        # Sort by most recently modified
        videos.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({"videos": videos})
        
    def select_video_route(self):
        """Select a video from the uploaded videos"""
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({"error": "No filename provided"}), 400
            
        video_path = os.path.join(self.videos_dir, filename)
        if not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 404
            
        # Update current video path
        self.current_video_path = video_path
        
        # Send message to client to update video source
        self.send_message({
            "video_updated": True,
            "video_src": f"/videos/{filename}"
        })
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "video_src": f"/videos/{filename}"
        })

    def on_websocket(self, websocket):
        listener_thread = threading.Thread(target=self.websocket_listener, args=[websocket], daemon=True)
        listener_thread.start()
        # Send queued messages
        try:
            while True:
                msg = self.ws_queue.get()
                self._logger.debug(f"Sending message to client: {msg}")
                try:
                    websocket.send(msg)
                except websockets.exceptions.ConnectionClosedOK:
                    self._logger.info("WebSocket connection closed by client")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    self._logger.error(f"WebSocket connection error: {e}")
                    break
                except Exception as e:
                    self._logger.error(f"WebSocket send error: {e}", exc_info=True)
                    # Don't break here, try to continue sending other messages
        except Exception as e:
            self._logger.error(f"WebSocket message queue processing error: {e}", exc_info=True)

    def websocket_listener(self, websocket):
        try:
            while True:
                try:
                    msg = websocket.recv()
                    self._logger.debug(f"Received message from client (len={len(msg)}).")
                    try:
                        data = json.loads(msg)
                        # Handle heartbeat messages from client
                        if data.get('type') == 'heartbeat':
                            self._logger.debug("Received heartbeat from client")
                            continue
                            
                        # If auto_frame flag is present, push frame_data into the frame_queue
                        if data.get('auto_frame') == True:
                            frame_data = data.get('frame_data')
                            if frame_data:
                                self._logger.debug("Got auto_frame data from client.")
                                self.frame_queue.put(frame_data)
                                # Also store it for future use
                                self.lastProcessedFrame = frame_data
                            continue
                        # Also check for 'frame_data' in non-auto messages
                        frame_data = data.pop('frame_data', None)
                        if frame_data:
                            self._logger.debug("Got frame_data from client.")
                            self.frame_queue.put(frame_data)
                            # Store the frame for future use
                            self.lastProcessedFrame = frame_data
                        if 'user_input' in data and self.msg_callback:
                            self._logger.debug(f"Sending user_input to msg_callback: {data}")
                            self.msg_callback(data, 0, int(time.time() * 1000))
                    except json.JSONDecodeError:
                        self._logger.warning("Invalid JSON from client.")
                        continue
                except websockets.exceptions.ConnectionClosedOK:
                    self._logger.info("WebSocket connection closed by client (listener)")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    self._logger.error(f"WebSocket connection error (listener): {e}")
                    # Attempt to gracefully clean up the connection
                    try:
                        websocket.close()
                    except:
                        pass
                    break
                except TimeoutError as e:
                    self._logger.warning(f"WebSocket timeout error: {e}")
                    # Attempt to gracefully close the connection
                    try:
                        websocket.close()
                    except:
                        pass
                    break
                except Exception as e:
                    self._logger.error(f"WebSocket receive error: {e}", exc_info=True)
                    # Log specific details for debugging
                    if "Bad file descriptor" in str(e):
                        self._logger.warning("Bad file descriptor error - possible connection timeout")
                    break
        except Exception as e:
            self._logger.error(f"WebSocket listener thread error: {e}", exc_info=True)
            # Ensure we clean up the websocket
            try:
                websocket.close()
            except:
                pass

    def on_audio_websocket(self, websocket):
        self._logger.info("Audio websocket connected (one-shot).")
        audio_data = bytearray()
        try:
            while True:
                try:
                    chunk = websocket.recv()
                    if isinstance(chunk, bytes):
                        audio_data.extend(chunk)
                    else:
                        break
                except websockets.exceptions.ConnectionClosedOK:
                    self._logger.info("Audio websocket connection closed normally by client.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    self._logger.warning(f"Audio websocket connection closed with error: {e}")
                    # Attempt to gracefully clean up
                    try:
                        websocket.close()
                    except:
                        pass
                    break
                except TimeoutError as e:
                    self._logger.warning(f"Audio WebSocket timeout error: {e}")
                    # Attempt to gracefully close the connection
                    try:
                        websocket.close()
                    except:
                        pass
                    break
                except Exception as e:
                    self._logger.error(f"Error receiving audio data: {e}", exc_info=True)
                    if "Bad file descriptor" in str(e):
                        self._logger.warning("Bad file descriptor error in audio socket - possible connection timeout")
                    break
        except Exception as e:
            self._logger.error(f"Audio websocket processing error: {e}", exc_info=True)
            # Clean up
            try:
                websocket.close()
            except:
                pass
        
        if len(audio_data) > 0:
            try:
                self._logger.debug(f"Forwarding final chunk of size {len(audio_data)} bytes to whisper server")
                self.whisper_socket.sendall(audio_data)
                self._logger.debug("Shutting down write side so Whisper sees EOF")
                
                try:
                    self.whisper_socket.shutdown(socket.SHUT_WR)
                    recognized_text = self.read_whisper_result()
                    self._logger.debug(f"Got recognized_text from whisper: {recognized_text}")
                    
                    if recognized_text.strip():
                        self._logger.debug("Requesting a frame from browser for final transcript.")
                        # Also directly add the user message to the UI
                        self.send_message({
                            "request_frame": True,
                            "recognized_text": recognized_text,
                            "user_input": recognized_text,
                            "asr_final": True
                        })
                    else:
                        self._logger.debug("No recognized text found from whisper.")
                except Exception as e:
                    self._logger.error(f"Error processing whisper results: {e}", exc_info=True)
                
                self._logger.debug("Closing whisper socket entirely. Re-initializing for next time.")
                try:
                    self.whisper_socket.close()
                except Exception as e:
                    self._logger.error(f"Error closing whisper socket: {e}")
                
                try:
                    self.create_whisper_socket()
                except Exception as e:
                    self._logger.error(f"Error recreating whisper socket: {e}", exc_info=True)
                    # Try to create a new socket after a delay in a separate thread
                    threading.Thread(target=self.delayed_socket_recreation, daemon=True).start()
            except Exception as e:
                self._logger.error(f"Error processing audio data: {e}", exc_info=True)
        else:
            self._logger.debug("No audio data received from client.")

    def read_whisper_result(self):
        result_buffer = b""
        try:
            # Set a timeout on the socket to avoid hanging indefinitely
            self.whisper_socket.settimeout(10.0)  # 10 seconds timeout
            
            while True:
                try:
                    chunk = self.whisper_socket.recv(1024)
                    if not chunk:
                        break
                    result_buffer += chunk
                except socket.timeout:
                    self._logger.warning("Timeout while reading from whisper socket")
                    break
                except Exception as e:
                    self._logger.error(f"Error reading from whisper socket: {e}")
                    break
                    
            # Reset timeout to default
            self.whisper_socket.settimeout(None)
            
            lines = result_buffer.decode('utf-8', errors='replace').split('\n')
            recognized_text = ""
            for line in lines:
                line = line.strip()
                if line.startswith("0 0 "):
                    recognized_text = line[4:]
                    
            return recognized_text
        except Exception as e:
            self._logger.error(f"Error processing whisper results: {e}", exc_info=True)
            return ""
    
    def tts_route(self):
        data = request.json
        text = data.get('text', '').strip()
        tts_service = data.get('tts_service', 'local')  # Default to local TTS
        api_key = data.get('api_key', None)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if tts_service == 'elevenlabs':
            return self._handle_elevenlabs_tts(text, api_key)
        else:  # local TTS service
            return self._handle_local_tts(text, data)

    def _handle_elevenlabs_tts(self, text, api_key):
        """Handle ElevenLabs TTS requests"""
        if not api_key:
            return jsonify({"error": "No API key provided for ElevenLabs"}), 400

        voice_id = "TX3LPaxmHKxFdv7VOQHJ"
        model_id = "eleven_multilingual_v2"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,        
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code != 200:
                return jsonify({"error": f"ElevenLabs API error {r.status_code}: {r.text[:500]}"}), 400
            audio_base64 = base64.b64encode(r.content).decode('utf-8')
            return jsonify({"tts_base64": audio_base64})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def _handle_local_tts(self, text, data):
        """Handle local TTS service requests"""
        try:
            tts_host = os.getenv('TTS_HOST', 'localhost')
            tts_port = os.getenv('TTS_PORT', '8082')
            tts_url = f"http://{tts_host}:{tts_port}/api/tts"

            # Get TTS model from request, default to a good quality model
            tts_model = data.get('tts_model', 'tts_models/en/ljspeech/vits')

            payload = {
                "text": text,
                "model_name": tts_model,
                "speaker_name": data.get('speaker_name'),
                "language": data.get('language')
            }

            # Try to connect to local TTS service
            r = requests.post(tts_url, json=payload, params={"format": "json"}, timeout=30)

            if r.status_code == 200:
                response_data = r.json()
                # The local TTS service returns base64 audio data
                return jsonify({"tts_base64": response_data.get('audio')})
            else:
                self._logger.error(f"Local TTS service error {r.status_code}: {r.text[:500]}")
                return jsonify({"error": f"Local TTS service error {r.status_code}: {r.text[:500]}"}), 400

        except requests.exceptions.ConnectionError:
            self._logger.error("Could not connect to local TTS service")
            return jsonify({"error": "Local TTS service is not available. Please check if it's running."}), 503
        except requests.exceptions.Timeout:
            self._logger.error("Local TTS service request timed out")
            return jsonify({"error": "TTS request timed out"}), 504
        except Exception as e:
            self._logger.error(f"Local TTS service error: {e}")
            return jsonify({"error": str(e)}), 500

    def send_message(self, payload):
        try:
            if not isinstance(payload, str):
                payload = json.dumps(payload)
            self._logger.debug(f"Queueing message for client: {payload}")
            self.ws_queue.put(payload)
        except Exception as e:
            self._logger.error(f"Error queueing message for client: {e}", exc_info=True)

    def generate_post_op_note_route(self):
        """Generate a post-op note summary using annotations and notes"""
        if not self.post_op_note_agent:
            self._logger.error("Post-op note agent not initialized")
            return jsonify({"error": "Post-op note agent not initialized"}), 500
            
        try:
            # Check if we have direct data from the frontend
            data = request.json
            self._logger.debug(f"Received post-op note request: {data}")
            
            if data and 'notes' in data and 'annotations' in data:
                self._logger.info("Using frontend-provided data for post-op note generation")
                
                # Create a temporary folder for this data
                import datetime
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="temp_procedure_")
                
                # Save the annotation data
                annotation_json = os.path.join(temp_dir, "annotation.json")
                with open(annotation_json, 'w') as f:
                    json.dump(data['annotations'], f, indent=2)
                
                # Save the notes data
                notes_json = os.path.join(temp_dir, "notetaker_notes.json")
                with open(notes_json, 'w') as f:
                    json.dump(data['notes'], f, indent=2)
                
                procedure_folder = temp_dir
                self._logger.info(f"Created temporary procedure folder: {procedure_folder}")
                
                # If we have frontend data but it's empty, return a basic note
                if not data['notes'] and not data['annotations']:
                    basic_note = {
                        "procedure_information": {
                            "procedure_type": "Unknown procedure",
                            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "duration": data.get('video_duration', 'Unknown'),
                            "surgeon": "Not specified"
                        },
                        "findings": ["No findings recorded"],
                        "procedure_timeline": [],
                        "complications": []
                    }
                    return jsonify({
                        "success": True, 
                        "post_op_note": basic_note
                    })
                
            else:
                # No frontend data, look for annotations directory
                self._logger.info("No frontend data, using stored annotations")
                annotations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'annotations')
                os.makedirs(annotations_dir, exist_ok=True)
                
                # Find the most recent procedure folder
                procedure_folders = []
                for folder in os.listdir(annotations_dir):
                    if folder.startswith('procedure_'):
                        folder_path = os.path.join(annotations_dir, folder)
                        if os.path.isdir(folder_path):
                            procedure_folders.append(folder_path)
                
                if not procedure_folders:
                    self._logger.warning("No procedure annotations found")
                    return jsonify({"error": "No procedure annotations found"}), 404
                    
                # Get the most recent folder by modification time
                procedure_folder = max(procedure_folders, key=os.path.getmtime)
            
            # Generate the post-op note using the agent
            self._logger.info(f"Generating post-op note from folder: {procedure_folder}")
            post_op_note = self.post_op_note_agent.generate_post_op_note(procedure_folder)
            
            if post_op_note:
                return jsonify({
                    "success": True, 
                    "post_op_note": post_op_note
                })
            else:
                self._logger.error("Post-op note agent returned empty result")
                # Return a basic note structure instead of error
                basic_note = {
                    "procedure_information": {
                        "procedure_type": "Laparoscopic procedure",
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "duration": data.get('video_duration', 'Unknown') if data else 'Unknown',
                        "surgeon": "Not specified"
                    },
                    "findings": ["Insufficient data for detailed findings"],
                    "procedure_timeline": [],
                    "complications": []
                }
                return jsonify({
                    "success": True, 
                    "post_op_note": basic_note
                })
                
        except Exception as e:
            self._logger.error(f"Error generating post-op note: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    def format_post_op_note_html(self, post_op_note):
        """Format the post-op note JSON into HTML for display"""
        try:
            # Extract key information from the note
            procedure_info = post_op_note.get("procedure_information", {})
            patient_info = post_op_note.get("patient_information", {})
            findings = post_op_note.get("findings", [])
            timeline = post_op_note.get("procedure_timeline", [])
            complications = post_op_note.get("complications", [])
            
            # Format the HTML - always include procedure info
            html = f"""
            <div class="p-4 border border-dark-700 rounded-lg">
                <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
                <div class="space-y-2">
                    <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> {procedure_info.get('procedure_type', 'Not specified')}</p>
                    <p class="text-sm"><span class="font-medium text-gray-400">Date:</span> {procedure_info.get('date', 'Not specified')}</p>
                    <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> {procedure_info.get('duration', 'Not specified')}</p>
                    <p class="text-sm"><span class="font-medium text-gray-400">Surgeon:</span> {procedure_info.get('surgeon', 'Not specified')}</p>
                </div>
            </div>
            """
            
            # Only add sections with content to avoid empty bullet lists
            # Add findings section if available and not empty
            if findings and len(findings) > 0:
                html += f"""
                <div class="p-4 border border-dark-700 rounded-lg mt-4">
                    <h3 class="text-lg font-semibold mb-2 text-primary-400">Key Findings</h3>
                    <ul class="list-disc list-inside space-y-1 text-sm">
                """
                
                for finding in findings:
                    if finding and finding.strip():  # Only add non-empty findings
                        html += f"<li>{finding}</li>"
                
                html += """
                    </ul>
                </div>
                """
            
            # Add timeline section if available and not empty
            if timeline and len(timeline) > 0:
                # Filter out events with no description
                valid_events = [event for event in timeline if event.get('description') and event['description'].strip()]
                
                if valid_events:
                    html += f"""
                    <div class="p-4 border border-dark-700 rounded-lg mt-4">
                        <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Timeline</h3>
                        <ul class="list-disc list-inside space-y-1 text-sm">
                    """
                    
                    for event in valid_events:
                        time = event.get('time', 'Unknown')
                        description = event.get('description', 'No description')
                        html += f"<li><span class='font-medium text-primary-300'>{time}</span>: {description}</li>"
                    
                    html += """
                        </ul>
                    </div>
                    """
            
            # Add complications section if available and not empty
            if complications and len(complications) > 0:
                # Filter out empty complications
                valid_complications = [comp for comp in complications if comp and comp.strip()]
                
                if valid_complications:
                    html += f"""
                    <div class="p-4 border border-dark-700 rounded-lg mt-4">
                        <h3 class="text-lg font-semibold mb-2 text-primary-400">Complications</h3>
                        <ul class="list-disc list-inside space-y-1 text-sm">
                    """
                    
                    for complication in valid_complications:
                        html += f"<li>{complication}</li>"
                    
                    html += """
                        </ul>
                    </div>
                    """
            
            # Add a message about annotations if no substantive content was found
            if not (findings or timeline or complications):
                html += """
                <div class="p-4 border border-dark-700 rounded-lg mt-4">
                    <h3 class="text-lg font-semibold mb-2 text-primary-400">Additional Information</h3>
                    <p class="text-sm text-gray-300">
                        Insufficient procedure data is available for a detailed summary. 
                        For better results, add more annotations and notes during the procedure.
                    </p>
                </div>
                """
            
            return html
        except Exception as e:
            self._logger.error(f"Error formatting post-op note HTML: {e}", exc_info=True)
            return f"""
            <div class='p-4 border border-dark-700 rounded-lg'>
                <h3 class="text-lg font-semibold mb-2 text-red-400">Error Generating Summary</h3>
                <p class="text-sm text-gray-300">An error occurred while formatting the procedure summary: {str(e)}</p>
                <p class="text-sm text-gray-400 mt-2">Try adding more annotations and notes to improve summary generation.</p>
            </div>
            """
    
    def run(self):
        self.ws_thread.start()
        self.audio_ws_thread.start()
        self._logger.info(f"Starting Flask app on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the web server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8050, help='Port to listen on')
    args = parser.parse_args()
    
    print(f"Starting web server on {args.host}:{args.port}...")
    server = Webserver(web_server=args.host, web_port=args.port)
    server.start()
    try:
        server.join()
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
        sys.exit(0)
