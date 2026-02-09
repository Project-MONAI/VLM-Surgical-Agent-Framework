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
USB Camera WebRTC Test Utility

Simple utility to capture video from USB camera and stream via WebRTC.
Provides WebRTC offer/answer and ICE server configuration endpoints.

Usage:
    python test_video_from_usb_cam.py [--camera-index 0] [--port 8080] [--fps 30]

Endpoints:
    POST /offer      - WebRTC offer/answer exchange
    GET /iceServers  - Retrieve ICE server configuration for clients
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from typing import Set

import cv2
import numpy as np
from aiohttp import web
from aiohttp.web import middleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state for peer connections
webrtc_pcs: Set[RTCPeerConnection] = set()
active_video_tracks: Set['USBCameraTrack'] = set()


@middleware
async def cors_middleware(request, handler):
    """Add CORS headers to all responses."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


class USBCameraTrack(VideoStreamTrack):
    """
    A video track that captures frames from a USB camera.

    This class runs a background thread to continuously capture frames from
    the camera, and serves them via WebRTC.
    """

    def __init__(self, camera_index: int = 0, fps: int = 30):
        """
        Initialize the USB camera track.

        Args:
            camera_index: Index of the USB camera device (0 for default camera)
            fps: Target frames per second for the stream
        """
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self.frame_count = 0
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True

        # Start camera capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info(f"Started USB camera capture thread for device {camera_index}")

    def _capture_loop(self):
        """Continuously capture frames from the USB camera in a background thread."""
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            logger.error(f"Failed to open camera at index {self.camera_index}")
            return

        # Set camera properties for optimal streaming
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info("Camera opened successfully:")
        logger.info(f"  Resolution: {actual_width}x{actual_height}")
        logger.info(f"  FPS: {actual_fps}")

        frame_time = 1.0 / self.fps
        last_capture = time.time()

        while self._running:
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV format) to RGB (WebRTC format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self._lock:
                    self._frame = frame_rgb

                # Maintain target FPS
                elapsed = time.time() - last_capture
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_capture = time.time()
            else:
                logger.warning("Failed to capture frame from camera")
                time.sleep(0.1)

        cap.release()
        logger.info("Camera released")

    def stop(self):
        """Stop the camera capture thread."""
        logger.info("Stopping camera capture")
        self._running = False
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        super().stop()

    async def recv(self):
        """
        Generate video frames for WebRTC from the latest camera image.

        Returns:
            VideoFrame: The current frame from the camera
        """
        pts, time_base = await self.next_timestamp()

        with self._lock:
            if self._frame is None:
                # Black frame until camera provides first frame
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                img = self._frame.copy()

        # Create VideoFrame for WebRTC
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        self.frame_count += 1
        if self.frame_count % 100 == 0:
            logger.debug(f"Sent {self.frame_count} frames on WebRTC stream")

        return frame


# ---------------------------------------------------------------------------
# WebRTC Server Handlers
# ---------------------------------------------------------------------------


async def ice_servers(request):
    """Return ICE server configuration for WebRTC."""
    # Empty ICE servers for local-only connections
    # Add STUN/TURN servers here if you need to connect across networks
    return web.Response(
        content_type="application/json",
        text=json.dumps([]),
    )


async def offer(request):
    """Handle WebRTC offer from client and return answer."""
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create new peer connection
    pc = RTCPeerConnection()
    webrtc_pcs.add(pc)
    logger.info(f"Created WebRTC peer connection (total: {len(webrtc_pcs)})")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            webrtc_pcs.discard(pc)

            # Stop and remove video track
            video_track = getattr(pc, "_video_track", None)
            if video_track:
                video_track.stop()
                active_video_tracks.discard(video_track)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"WebRTC ICE connection state: {pc.iceConnectionState}")

    # Get camera configuration from app
    camera_index = request.app["camera_index"]
    fps = request.app["fps"]

    # Create and add video track
    video_track = USBCameraTrack(camera_index=camera_index, fps=fps)
    pc.addTrack(video_track)
    pc._video_track = video_track  # type: ignore[attr-defined]
    active_video_tracks.add(video_track)
    logger.info(f"Added WebRTC video track (total active: {len(active_video_tracks)})")

    # Handle the offer
    await pc.setRemoteDescription(offer_desc)
    logger.info("Set remote description (offer)")

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Created and set local description (answer)")

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }),
    )


async def on_shutdown(app):
    """Clean up peer connections and camera tracks on shutdown."""
    logger.info("Shutting down server...")

    # Stop all video tracks
    for track in list(active_video_tracks):
        track.stop()

    # Close all peer connections
    coros = [pc.close() for pc in webrtc_pcs]
    await asyncio.gather(*coros, return_exceptions=True)

    webrtc_pcs.clear()
    active_video_tracks.clear()
    logger.info("Cleanup complete")


async def main(args):
    """Main entry point for the test utility."""
    logger.info("=" * 60)
    logger.info("USB Camera WebRTC Test Utility")
    logger.info("=" * 60)
    logger.info(f"Camera index: {args.camera_index}")
    logger.info(f"Target FPS: {args.fps}")
    logger.info(f"Server port: {args.port}")
    logger.info("")

    # Create aiohttp application with CORS middleware
    app = web.Application(middlewares=[cors_middleware])
    app["camera_index"] = args.camera_index
    app["fps"] = args.fps
    app.on_shutdown.append(on_shutdown)

    # Add WebRTC endpoints
    app.router.add_get("/iceServers", ice_servers)
    app.router.add_post("/offer", offer)

    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    logger.info(f"✅ Server running at http://{args.host}:{args.port}/")
    logger.info("   GET  /iceServers - ICE server configuration")
    logger.info("   POST /offer      - WebRTC offer/answer endpoint")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        # Keep running until interrupted
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="USB Camera WebRTC Test Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="USB camera device index"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port"
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
