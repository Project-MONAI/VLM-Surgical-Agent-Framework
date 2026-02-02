// Add this at the beginning of the script or in the DOMContentLoaded event listener
const sessionId = generateUniqueId();

// UI timing constants
const OVERLAY_HIDE_DELAY_MS = 1500;

// Function to generate a unique ID
function generateUniqueId() {
  return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

// Global video source configuration (must be in global scope for frame capture to access)
let videoSourceConfig = null;
let currentVideoSourceMode = 'surgical';  // Default fallback
let isWebRTCConnected = false;

// Fetch video source configuration from backend
let videoConfigPromise = null;  // Promise to track loading state
async function loadVideoSourceConfig() {
  try {
    const response = await fetch('/api/video_sources');
    if (response.ok) {
      videoSourceConfig = await response.json();
      currentVideoSourceMode = videoSourceConfig.default_source || 'surgical';
      console.log('Video source configuration loaded:', videoSourceConfig);
    } else {
      console.warn('Failed to load video source configuration, using defaults');
    }
  } catch (error) {
    console.error('Error loading video source configuration:', error);
  }
}

// Helper function to create frame message payload with correct flags
// Must be in global scope so startAutoFrameCapture can access it
function createFrameMessagePayload(frameData) {
  if (!videoSourceConfig || !videoSourceConfig.sources) {
    // Fallback to defaults if configuration not loaded
    console.warn('Video source config not loaded yet - using legacy format (auto_frame/frame_data)');
    return {
      auto_frame: true,
      frame_data: frameData
    };
  }

  // Determine which video source mode we're in based on source_type
  let sourceMode = currentVideoSourceMode;
  
  // If WebRTC is connected, find the source with source_type='livestream'
  if (isWebRTCConnected) {
    // Find the source mode that has source_type='livestream'
    for (const [mode, config] of Object.entries(videoSourceConfig.sources)) {
      if (config.source_type === 'livestream') {
        sourceMode = mode;
        break;
      }
    }
  } else {
    // For uploaded videos, find the source with source_type='uploaded'
    for (const [mode, config] of Object.entries(videoSourceConfig.sources)) {
      if (config.source_type === 'uploaded') {
        sourceMode = mode;
        break;
      }
    }
  }

  // Get the configuration for this source mode
  const sourceConf = videoSourceConfig.sources[sourceMode];
  if (!sourceConf) {
    console.warn(`No configuration found for source mode '${sourceMode}', using defaults`);
    return {
      auto_frame: true,
      frame_data: frameData
    };
  }

  // Build payload with correct flags from configuration
  const payload = {};
  payload[sourceConf.websocket_flag] = true;
  payload[sourceConf.frame_data_key] = frameData;

  console.log(`✓ Using source mode '${sourceMode}' (type: ${sourceConf.source_type || 'unknown'})`);
  console.log(`  Flags: ${sourceConf.websocket_flag}=true, ${sourceConf.frame_data_key}=<frame>`);
  return payload;
}

// Load configuration on page load (store promise for checking later)
videoConfigPromise = loadVideoSourceConfig();

// Panel Collapsible Functionality
document.addEventListener('DOMContentLoaded', function() {
  // Initialize save note button
  const saveNoteBtn = document.getElementById('save-note-btn');
  if (saveNoteBtn) {
    saveNoteBtn.onclick = saveManualNote;
  }

  // Get all collapsible panels
  const panels = document.querySelectorAll('.panel-toggle');

  // Add click event listeners to each panel toggle button
  panels.forEach(panel => {
    panel.addEventListener('click', function(e) {
      e.preventDefault();

      // Get the target content element
      const targetId = this.getAttribute('data-target').substring(1);
      const contentElement = document.getElementById(targetId);

      // Toggle the content visibility
      contentElement.classList.toggle('hidden');

      // Update the expanded state
      const isExpanded = !contentElement.classList.contains('hidden');
      this.setAttribute('aria-expanded', isExpanded);

      // Update the chevron icon
      const icon = this.querySelector('.fa-chevron-down');
      if (isExpanded) {
        icon.style.transform = 'rotate(0deg)';
      } else {
        icon.style.transform = 'rotate(-90deg)';
      }
    });
  });

  // Initialize all panels as expanded by default
  panels.forEach(panel => {
    // Set aria-expanded to true
    panel.setAttribute('aria-expanded', 'true');

    // Make sure the chevron is pointing down
    const icon = panel.querySelector('.fa-chevron-down');
    icon.style.transform = 'rotate(0deg)';

    // Make sure the content is visible
    const targetId = panel.getAttribute('data-target').substring(1);
    const contentElement = document.getElementById(targetId);
    if (contentElement) {
      contentElement.classList.remove('hidden');
    }
  });

  // Tab functionality
  const tabButtons = document.querySelectorAll('[role="tab"]');
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabId = button.getAttribute('data-tab-target');

      // Hide all tab panes
      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
        pane.style.display = 'none';
      });

      // Deactivate all tab buttons
      document.querySelectorAll('[role="tab"]').forEach(btn => {
        btn.classList.remove('active');
        btn.classList.remove('border-b-2');
        btn.classList.remove('border-primary-500');
        btn.classList.remove('bg-dark-800');
      });

      // Show the selected tab pane
      const selectedPane = document.getElementById(tabId);
      selectedPane.classList.add('active');
      selectedPane.style.display = 'block';

      // Activate the clicked tab button
      button.classList.add('active');
      button.classList.add('border-b-2');
      button.classList.add('border-primary-500');
      button.classList.add('bg-dark-800');
    });
  });

  // Make sure the first tab is active on page load
  const firstTabPane = document.querySelector('.tab-pane');
  if (firstTabPane) {
    firstTabPane.style.display = 'block';
  }

  // Setup video browser modal
  const showVideosBtn = document.getElementById('show-videos-btn');
  if (showVideosBtn) {
    showVideosBtn.addEventListener('click', () => {
      // Open modal
      const modal = document.getElementById('videoSelectModal');
      if (modal) {
        modal.classList.add('show');
        modal.style.display = 'flex'; // Change from block to flex to center properly
        document.body.classList.add('overflow-hidden');

        // Load videos
        loadVideos();
      }
    });
  }

  // Setup refresh videos button
  const refreshVideosBtn = document.getElementById('refresh-videos-btn');
  if (refreshVideosBtn) {
    refreshVideosBtn.addEventListener('click', loadVideos);
  }

  // Setup search videos functionality
  const videoSearch = document.getElementById('video-search');
  if (videoSearch) {
    videoSearch.addEventListener('input', function() {
      const searchText = this.value.toLowerCase();
      const videoItems = document.querySelectorAll('.video-item');

      // For each item, check if the filename contains the search text
      videoItems.forEach(item => {
        const filename = item.querySelector('h4')?.textContent.toLowerCase() || '';
        if (searchText === '' || filename.includes(searchText)) {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      });

      // Show or hide the "no videos" message based on search results
      const visibleItems = Array.from(videoItems).filter(item => item.style.display !== 'none');
      const noVideos = document.getElementById('no-videos');

      if (visibleItems.length === 0 && searchText !== '' && noVideos) {
        noVideos.style.display = 'block';
        noVideos.innerHTML = `
          <div class="bg-dark-800 rounded-xl p-6 border border-dark-600">
            <i class="fas fa-search text-4xl mb-4 text-gray-500"></i>
            <p class="text-gray-300 text-lg">No videos matching "${searchText}"</p>
            <p class="text-gray-400 text-sm mt-2">Try a different search term</p>
          </div>
        `;
      } else if (noVideos && visibleItems.length > 0) {
        noVideos.style.display = 'none';
      }
    });
  }

  // Setup file input display
  const fileInput = document.getElementById('video-upload');
  if (fileInput) {
    fileInput.addEventListener('change', function() {
      const fileName = this.files[0]?.name || 'Select video file...';
      const fileNameElement = this.closest('.relative').querySelector('.file-name');
      if (fileNameElement) {
        fileNameElement.textContent = fileName;
      }
    });
  }

  // Setup auto-resizing textarea
  const chatInput = document.getElementById('chat-message-input');
  if (chatInput) {
    // Function to adjust height based on content
    function autoResizeTextarea() {
      chatInput.style.height = 'auto'; // Reset height to recalculate

      // Set a minimum height
      const minHeight = 48; // 3rem

      // Calculate new height based on scroll height (content)
      const newHeight = Math.min(chatInput.scrollHeight, 128); // 128px = 8rem (max-h-32)

      // Apply the new height, but not less than minimum
      chatInput.style.height = Math.max(newHeight, minHeight) + 'px';
    }

    // Initialize on load
    autoResizeTextarea();

    // Update on input
    chatInput.addEventListener('input', autoResizeTextarea);

    // Reset on submit
    document.querySelector('button[onclick="onChatMessageSubmit()"]').addEventListener('click', function() {
      setTimeout(() => {
        chatInput.style.height = '48px'; // Reset to minimum height after submit
      }, 10);
    });
  }

  // Modal functionality
  const modalTriggers = document.querySelectorAll('[data-modal-target]');
  const modalCloseButtons = document.querySelectorAll('[data-close-modal]');

  modalTriggers.forEach(trigger => {
    trigger.addEventListener('click', () => {
      const modalId = trigger.getAttribute('data-modal-target');
      const modal = document.getElementById(modalId);

      // Show modal with fade effect
      modal.classList.add('show');
      modal.style.display = 'flex'; // Use flex for centering
      document.body.classList.add('overflow-hidden');
    });
  });

  modalCloseButtons.forEach(button => {
    button.addEventListener('click', () => {
      const modal = button.closest('.modal');
      closeModal(modal);
    });
  });

  // Close modal when clicking on backdrop
  document.addEventListener('click', (e) => {
    const modals = document.querySelectorAll('.modal.show');
    modals.forEach(modal => {
      // If click is directly on the modal (the backdrop) but not on the dialog
      if (e.target === modal || e.target.classList.contains('fixed')) {
        if (!e.target.closest('.modal-dialog')) {
          closeModal(modal);
        }
      }
    });
  });

  function closeModal(modal) {
    if (!modal) return;

    modal.classList.remove('show');
    modal.classList.add('closing');

    setTimeout(() => {
      modal.style.display = 'none'; // Reset display property
      modal.classList.remove('closing');
      document.body.classList.remove('overflow-hidden');
    }, 300);
  }

  // Video source toggle logic
  const uploadedRadio = document.getElementById('videoSourceUploaded');
  const liveRadio = document.getElementById('videoSourceLive');
  const liveStreamUrlContainer = document.getElementById('liveStreamUrlContainer');
  const uploadedVideoWrapper = document.getElementById('uploadedVideoWrapper');
  const liveStreamWrapper = document.getElementById('liveStreamWrapper');
  const liveStreamIframe = document.getElementById('liveStreamIframe');
  const loadLiveStreamBtn = document.getElementById('loadLiveStreamBtn');
  const surgeryVideo = document.getElementById('surgery-video');
  const fullscreenBtn = document.getElementById('fullscreen-btn');
  const captureFrameBtn = document.getElementById('capture-frame-btn');

  function setVideoSourceMode(mode) {
    if (mode === 'uploaded') {
      if (liveStreamUrlContainer) liveStreamUrlContainer.style.display = 'none';
      if (fullscreenBtn) fullscreenBtn.disabled = false;
      if (captureFrameBtn) captureFrameBtn.disabled = false;
      // Disconnect WebRTC if connected
      disconnectWebRTC();
      // Clear srcObject and use src for uploaded videos
      if (surgeryVideo) {
        surgeryVideo.srcObject = null;
        surgeryVideo.style.display = '';
      }
    } else if (mode === 'live') {
      if (liveStreamUrlContainer) liveStreamUrlContainer.style.display = '';
      if (fullscreenBtn) fullscreenBtn.disabled = false; // Enable fullscreen for live streams
      if (captureFrameBtn) captureFrameBtn.disabled = false; // Enable frame capture for live streams
      if (surgeryVideo) {
        // Clear src and prepare for srcObject from WebRTC
        surgeryVideo.src = '';
        surgeryVideo.style.display = '';
      }
    }
  }

  if (uploadedRadio && liveRadio) {
    uploadedRadio.addEventListener('change', function() {
      if (this.checked) setVideoSourceMode('uploaded');
    });
    liveRadio.addEventListener('change', function() {
      if (this.checked) setVideoSourceMode('live');
    });
  }

  // WebRTC connection management
  let webrtcConnection = null;

  // WebRTC event handlers
  const connectWebRTCBtn = document.getElementById('connectWebRTCBtn');
  const disconnectWebRTCBtn = document.getElementById('disconnectWebRTCBtn');
  const webrtcStatus = document.getElementById('webrtcStatus');

  if (connectWebRTCBtn) {
    connectWebRTCBtn.addEventListener('click', connectWebRTC);
  }

  if (disconnectWebRTCBtn) {
    disconnectWebRTCBtn.addEventListener('click', disconnectWebRTC);
  }

  // WebRTC connection function
  async function connectWebRTC() {
    const urlInput = document.getElementById('liveStreamUrl');
    const serverUrl = urlInput ? urlInput.value.trim() : 'http://localhost:8080';
    const videoElement = document.getElementById('surgery-video');

    if (!videoElement) {
      console.error('Video element not found');
      updateWebRTCStatus('Error: Video element not found', 'error');
      return;
    }

    if (isWebRTCConnected) {
      console.log('WebRTC already connected');
      return;
    }

    try {
      updateWebRTCStatus('Connecting...', 'connecting');
      setWebRTCButtonsState(true, false);

      // Connect to WebRTC stream using the provided template
      const connection = await connectToWebRTCStream(serverUrl, videoElement);

      webrtcConnection = connection;
      isWebRTCConnected = true;

      updateWebRTCStatus('Connected', 'connected');
      setWebRTCButtonsState(false, true);

      console.log('WebRTC connection established successfully');

      // Start auto frame capture for live stream
      startAutoFrameCapture();

      // Enable the microphone button for live stream
      enableMicButton();

    } catch (error) {
      console.error('Failed to connect to WebRTC stream:', error);
      updateWebRTCStatus(`Error: ${error.message}`, 'error');
      setWebRTCButtonsState(false, false);
      isWebRTCConnected = false;
      webrtcConnection = null;
    }
  }

  // WebRTC disconnection function (improved cleanup like working example)
  function disconnectWebRTC() {
    if (webrtcConnection) {
      try {
        // Stop transceivers first (like working example)
        if (webrtcConnection.getTransceivers) {
          webrtcConnection.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
              transceiver.stop();
            }
          });
        }

        // Close peer connection
        webrtcConnection.close();
        console.log('WebRTC connection closed');
      } catch (error) {
        console.error('Error closing WebRTC connection:', error);
      }
      webrtcConnection = null;
    }

    // Clear video stream
    const videoElement = document.getElementById('surgery-video');
    if (videoElement) {
      videoElement.srcObject = null;
    }

    isWebRTCConnected = false;
    updateWebRTCStatus('Disconnected', 'disconnected');
    setWebRTCButtonsState(false, false);

    // Stop auto frame capture
    stopAutoFrameCapture();

    // Disable mic button only if there's no uploaded video playing
    const hasUploadedVideo = videoElement && videoElement.src && videoElement.src.trim() !== '';
    if (!hasUploadedVideo) {
      disableMicButton();
    }

    console.log('Disconnected from WebRTC stream');
  }

  // Make disconnectWebRTC globally accessible
  window.disconnectWebRTC = disconnectWebRTC;

  // Helper function to update WebRTC status
  function updateWebRTCStatus(message, status) {
    if (webrtcStatus) {
      webrtcStatus.textContent = message;
      webrtcStatus.className = 'ml-2 text-xs';

      switch (status) {
        case 'connecting':
          webrtcStatus.className += ' text-yellow-400';
          break;
        case 'connected':
          webrtcStatus.className += ' text-green-400';
          break;
        case 'error':
          webrtcStatus.className += ' text-red-400';
          break;
        case 'disconnected':
        default:
          webrtcStatus.className += ' text-gray-400';
          break;
      }
    }
  }

  // Helper function to manage button states
  function setWebRTCButtonsState(connectDisabled, disconnectVisible) {
    if (connectWebRTCBtn) {
      connectWebRTCBtn.disabled = connectDisabled;
    }
    if (disconnectWebRTCBtn) {
      disconnectWebRTCBtn.style.display = disconnectVisible ? 'inline-block' : 'none';
    }
  }

  // WebRTC connection implementation based on provided template
  async function connectToWebRTCStream(serverUrl, videoElement) {
    try {
      // 1. Get ICE servers
      console.log(`Fetching ICE servers from ${serverUrl}/iceServers`);
      const iceServersResponse = await fetch(`${serverUrl}/iceServers`);

      if (!iceServersResponse.ok) {
        throw new Error(`Failed to fetch ICE servers: ${iceServersResponse.status}`);
      }

      const serverIceServers = await iceServersResponse.json();

      // Combine with default STUN server as fallback (like working example)
      const iceServers = [
        { urls: 'stun:stun.l.google.com:19302' },
        ...(Array.isArray(serverIceServers) ? serverIceServers : [])
      ];
      console.log('Using ICE servers:', iceServers);

      // 2. Create peer connection
      const pc = new RTCPeerConnection({
        sdpSemantics: 'unified-plan',
        iceServers: iceServers
      });

      // 3. Handle video stream (using addEventListener like working example)
      pc.addEventListener('track', (event) => {
        console.log('Received track:', event.track.kind);
        if (event.track.kind === 'video') {
          console.log('Setting video stream to video element');
          videoElement.srcObject = event.streams[0];
          console.log('Video stream received and set');
        }
      });

      // Handle ICE connection state changes
      pc.addEventListener('iceconnectionstatechange', () => {
        console.log('ICE connection state:', pc.iceConnectionState);
        switch (pc.iceConnectionState) {
          case 'connected':
          case 'completed':
            updateWebRTCStatus('Connected', 'connected');
            break;
          case 'disconnected':
          case 'failed':
            updateWebRTCStatus('Connection failed', 'error');
            break;
          case 'connecting':
            updateWebRTCStatus('Connecting...', 'connecting');
            break;
        }
      });

      // Handle connection state changes (like working example)
      pc.addEventListener('connectionstatechange', () => {
        console.log('Connection state:', pc.connectionState);
        if (pc.connectionState === 'failed') {
          updateWebRTCStatus('Connection failed', 'error');
        }
      });

      // 4. Setup and negotiate
      pc.addTransceiver('video', { direction: 'recvonly' });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      console.log('Created offer, sending to server');

      // 5. Exchange with server
      const response = await fetch(`${serverUrl}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const answer = await response.json();
      console.log('Received answer from server');

      // Validate answer before setting
      if (!answer.sdp || !answer.type) {
        throw new Error('Invalid SDP answer received from server');
      }

      await pc.setRemoteDescription(answer);
      console.log('WebRTC negotiation completed successfully');

      return pc;

    } catch (error) {
      console.error('WebRTC connection error:', error);
      throw error;
    }
  }

  // Default to uploaded video mode on load
  setVideoSourceMode('uploaded');
});

// Set the current time in the welcome message and connect websocket
document.addEventListener('DOMContentLoaded', function() {
  const welcomeTimeElement = document.getElementById('welcome-time');
  if (welcomeTimeElement) {
    const now = new Date();
    welcomeTimeElement.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // Connect to main application WebSocket server (for chat, video updates, etc.)
  if (typeof connectWebsocket === 'function') {
    connectWebsocket(49000, handleServerMessage);
    console.log("Connected to WebSocket server");
  } else {
    console.error("WebSocket connection function not available");
    showToast("WebSocket connection not available", "error");
  }

  // Initialize video element
  const videoElement = document.getElementById('surgery-video');

  if (videoElement) {
    // Ensure autoplay is disabled on page load
    videoElement.autoplay = false;

    // Reset video source to empty state on page load
    videoElement.src = '';
    videoElement.load();

    // Pause any playing video
    if (!videoElement.paused) {
      videoElement.pause();
    }

    // Show placeholder message
    showVideoPlaceholder();

    // Start automatic frame capture for annotations when video plays
    videoElement.addEventListener('play', () => {
      startAutoFrameCapture();
    }, { once: false });
  }

  // Set up video event listeners for frame capture if not already done
  if (videoElement) {
    // Clean up existing event listeners to avoid duplicates
    const existingListeners = videoElement._hasFrameCaptureListeners;
    if (!existingListeners) {
      videoElement.addEventListener('play', () => {
        console.log("Video playback started, enabling auto frame capture");
        startAutoFrameCapture();
      });

      videoElement.addEventListener('pause', () => {
        console.log("Video playback paused, disabling auto frame capture");
        stopAutoFrameCapture();
      });

      videoElement.addEventListener('ended', () => {
        console.log("Video playback ended, disabling auto frame capture");
        stopAutoFrameCapture();
      });

      // Mark that we've added these listeners
      videoElement._hasFrameCaptureListeners = true;
    }

    // Allow video autoplay if the attribute is set in HTML
    // videoElement.pause();
  }
});

// Handle messages from the server
function handleServerMessage(message) {
  console.log("Received message from server:", message);

  // Check if this message is for another session
  if (message.session_id && message.session_id !== sessionId && 
      message.recognized_text && message.asr_final) {
    console.log("Ignoring ASR message from different session:", message.session_id);
    return; // Skip processing this message as it's for another session
  }

  // Handle recognized text from audio
  if (message.recognized_text && message.asr_final) {
    // Display the transcribed text in the chat stream as a user message
    addMessageToChat(message.recognized_text, 'user');
    // Do NOT populate the input bar; keep it clear like typed-submit behavior
    const chatInput = document.getElementById('chat-message-input');
    if (chatInput) {
      chatInput.value = '';
      chatInput.style.height = '48px';
    }
  }

  // Handle AI responses
  if (message.message) {
    addMessageToChat(message.message, 'agent');
  }

  // Handle agent responses (new format)
  if (message.agent_response) {
    // Check if this is an annotation (contains marker text)
    if (message.agent_response.startsWith('Annotation:')) {
      // Add to annotations panel
      addAnnotation(message.agent_response);
    }
    // Check if this is a note (or has is_note flag)
    else if (message.is_note || message.agent_response.toLowerCase().includes('note:')) {
      // Add to notes panel with user's original message if available
      addNote(message.agent_response, message.original_user_input || message.user_input || '');
    }
    else {
      // Regular response - just add to chat; include agent name if provided
      addMessageToChat(message.agent_response, 'agent', message.agent_name);

      // Handle TTS for agent responses if enabled
      if (window.isTtsEnabled) {
        generateSpeech(message.agent_response);
      }

    }

    // Update phase if annotation includes phase info
    if (message.agent_response.includes("Phase '") || message.agent_response.includes("phase '")) {
      updatePhaseFromAnnotation(message.agent_response);
    }
  }

  // Handle structured post-op note sent over WebSocket (voice-triggered flow)
  if (message.post_op_note) {
    try {
      renderPostOpNote(message.post_op_note);
      // Switch to Summary tab to show the result
      const summaryTabBtn = document.getElementById('summary-tab');
      if (summaryTabBtn) summaryTabBtn.click();
      showToast('Post‑op note generated and rendered in Summary tab', 'success');
    } catch (e) {
      console.error('Error rendering post-op note from WebSocket:', e);
      showToast('Error displaying post‑op note', 'error');
    }
  }

  // Handle video updates
  if (message.video_updated && message.video_src) {
    // Stop only TTS audio playback when video is updated (don't reset connection)
    stopCurrentTTS();

    const videoElement = document.getElementById('surgery-video');
    if (videoElement) {
      // Pause current video first
      try {
        videoElement.pause();
      } catch (e) {
        console.warn("Could not pause video:", e);
      }

      // Set new source with autoplay
      videoElement.src = message.video_src;
      videoElement.load();
      videoElement.autoplay = true;

      // Reset UI elements for new video
      const phaseElement = document.getElementById('current-phase');
      if (phaseElement) {
        phaseElement.textContent = 'Undefined';
      }

      // Clear annotations for new video
      const annotationsContainer = document.getElementById('annotations-container');
      if (annotationsContainer) {
        annotationsContainer.innerHTML = `
          <div class="text-center text-gray-400 p-5">
            <i class="fas fa-tag fa-3x mb-3"></i>
            <p>No annotations available yet. Annotations will appear here as they are generated.</p>
          </div>
        `;
      }

      // Reset annotation count
      const annotationCount = document.querySelector('.annotation-count');
      if (annotationCount) {
        annotationCount.textContent = '0';
      }
    }
  }

  // Handle request for frame
  if (message.request_frame) {
    sendFrameWithText(message.recognized_text);
  }
}

// Render a post‑op note JSON into the Summary tab. Supports both new and legacy shapes.
function renderPostOpNote(postOpNote) {
  const container = document.getElementById('summary-container');
  if (!container) return;

  // Helper to format duration seconds
  function fmtDuration(sec) {
    if (sec == null) return 'Not specified';
    const s = Math.max(0, parseInt(sec, 10));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    return `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${r.toString().padStart(2,'0')}`;
  }

  // Detect new grammar (flat keys) vs legacy structure
  const isNew = !!postOpNote.date_time || !!postOpNote.personnel || !!postOpNote.timeline;

  let html = '';
  if (isNew) {
    const procType = postOpNote.procedure_type || 'Not specified';
    const dateTime = postOpNote.date_time || 'Not specified';
    const nature = postOpNote.procedure_nature || 'unknown';
    const personnel = postOpNote.personnel || {};
    const surgeon = personnel.surgeon || 'Not specified';
    const assistant = personnel.assistant || 'Not specified';
    const anaesthetist = personnel.anaesthetist || 'Not specified';

    // Derive a duration if phase_summary present
    let durationStr = 'Not specified';
    const phaseSummary = postOpNote.phase_summary || [];
    if (Array.isArray(phaseSummary) && phaseSummary.length) {
      const total = phaseSummary.reduce((acc, p) => acc + (typeof p.duration_seconds === 'number' ? p.duration_seconds : 0), 0);
      if (total > 0) durationStr = fmtDuration(total);
    }

    html += `
      <div class="p-4 border border-dark-700 rounded-lg">
        <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
        <div class="space-y-2">
          <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> ${procType}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Date/Time:</span> ${dateTime}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Nature:</span> ${nature}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> ${durationStr}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Surgeon:</span> ${surgeon}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Assistant:</span> ${assistant}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Anaesthetist:</span> ${anaesthetist}</p>
        </div>
      </div>
    `;

    // Findings (string)
    if (postOpNote.findings && String(postOpNote.findings).trim()) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Findings</h3>
          <p class="text-sm text-gray-300">${postOpNote.findings}</p>
        </div>
      `;
    }

    // Complications (string)
    if (postOpNote.complications && String(postOpNote.complications).trim()) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Complications</h3>
          <p class="text-sm text-gray-300">${postOpNote.complications}</p>
        </div>
      `;
    }

    // Prophylaxis and EBL
    html += `
      <div class="p-4 border border-dark-700 rounded-lg mt-4">
        <h3 class="text-lg font-semibold mb-2 text-primary-400">Perioperative Details</h3>
        <div class="space-y-1 text-sm">
          <p><span class="font-medium text-gray-400">Estimated blood loss:</span> ${postOpNote.blood_loss_estimate || 'Not specified'}</p>
          <p><span class="font-medium text-gray-400">DVT prophylaxis:</span> ${postOpNote.dvt_prophylaxis || 'Not specified'}</p>
          <p><span class="font-medium text-gray-400">Antibiotic prophylaxis:</span> ${postOpNote.antibiotic_prophylaxis || 'Not specified'}</p>
          <p><span class="font-medium text-gray-400">Post‑op instructions:</span> ${postOpNote.postoperative_instructions || 'Not specified'}</p>
        </div>
      </div>
    `;

    // Phase summary
    if (Array.isArray(phaseSummary) && phaseSummary.length) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Phase Summary</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${phaseSummary.map(p => `<li>${p.phase}: ${p.start_time || 'Unknown'} (${p.duration || 'Not specified'})</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Timeline (event)
    const timeline = Array.isArray(postOpNote.timeline) ? postOpNote.timeline : [];
    if (timeline.length) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Timeline</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${timeline.map(ev => `<li><span class='font-medium text-primary-300'>${ev.time || 'Unknown'}</span>: ${ev.event || ''}</li>`).join('')}
          </ul>
        </div>
      `;
    }
  } else {
    // Legacy shape (procedure_information, findings[], procedure_timeline[], complications[])
    const info = postOpNote.procedure_information || {};
    html += `
      <div class="p-4 border border-dark-700 rounded-lg">
        <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
        <div class="space-y-2">
          <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> ${info.procedure_type || 'Not specified'}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Date:</span> ${info.date || 'Not specified'}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> ${info.duration || 'Not specified'}</p>
          <p class="text-sm"><span class="font-medium text-gray-400">Surgeon:</span> ${info.surgeon || 'Not specified'}</p>
        </div>
      </div>
    `;
    const findings = Array.isArray(postOpNote.findings) ? postOpNote.findings : [];
    if (findings.length) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Key Findings</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${findings.map(f => `<li>${f}</li>`).join('')}
          </ul>
        </div>
      `;
    }
    const timeline = Array.isArray(postOpNote.procedure_timeline) ? postOpNote.procedure_timeline : [];
    if (timeline.length) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Timeline</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${timeline.map(ev => `<li><span class='font-medium text-primary-300'>${ev.time || 'Unknown'}</span>: ${ev.description || ''}</li>`).join('')}
          </ul>
        </div>
      `;
    }
    const complications = Array.isArray(postOpNote.complications) ? postOpNote.complications : [];
    if (complications.length) {
      html += `
        <div class="p-4 border border-dark-700 rounded-lg mt-4">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Complications</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${complications.map(c => `<li>${c}</li>`).join('')}
          </ul>
        </div>
      `;
    }
  }

  container.innerHTML = html;
}

// Function to add annotation to the annotations panel
function addAnnotation(annotationText) {
  const annotationsContainer = document.getElementById('annotations-container');
  const noAnnotationsMsg = annotationsContainer.querySelector('.text-center');

  // Remove "no annotations" message if it exists
  if (noAnnotationsMsg) {
    noAnnotationsMsg.remove();
  }

  // Create annotation element
  const annotationElement = document.createElement('div');
  annotationElement.className = 'bg-dark-800 rounded-lg p-3 border border-dark-700 mb-3';

  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Parse the annotation text to extract phase, tools, anatomy
  let phaseText = '';
  let toolsText = '';
  let anatomyText = '';

  if (annotationText.includes('Phase')) {
    const phaseMatch = annotationText.match(/Phase [\'"]([^'"]+)[\'"]/) || 
                       annotationText.match(/phase [\'"]([^'"]+)[\'"]/) || 
                       annotationText.match(/Phase: ([^|]+)/);
    if (phaseMatch) {
      phaseText = phaseMatch[1];
    }
  }

  if (annotationText.includes('Tools:')) {
    const toolsMatch = annotationText.match(/Tools: ([^|]+)/);
    if (toolsMatch) {
      toolsText = toolsMatch[1].trim();
    }
  }

  if (annotationText.includes('Anatomy:')) {
    const anatomyMatch = annotationText.match(/Anatomy: ([^$]+)/);
    if (anatomyMatch) {
      anatomyText = anatomyMatch[1].trim();
    }
  }

  annotationElement.innerHTML = `
    <div class="flex justify-between items-start mb-2">
      <div>
        <h3 class="text-lg font-semibold text-success-400">${phaseText || 'Annotation'}</h3>
        <div class="flex flex-wrap gap-1 mt-1">
          ${toolsText ? `<span class="badge bg-dark-700 text-primary-300 px-2 py-0.5"><i class="fas fa-tools mr-1"></i>${toolsText}</span>` : ''}
          ${anatomyText ? `<span class="badge bg-dark-700 text-yellow-300 px-2 py-0.5"><i class="fas fa-heart mr-1"></i>${anatomyText}</span>` : ''}
        </div>
      </div>
      <span class="text-xs text-gray-400">${timeString}</span>
    </div>
    <p class="text-sm text-gray-300">${annotationText}</p>
  `;

  // Add to container
  annotationsContainer.prepend(annotationElement);

  // Update the count
  const annotationCount = document.querySelector('.annotation-count');
  if (annotationCount) {
    annotationCount.textContent = parseInt(annotationCount.textContent || '0') + 1;
  }
}

// Function to add note to the notes panel
function addNote(noteText, userMessage) {
  const notesContainer = document.getElementById('notes-container');
  const noNotesMsg = notesContainer.querySelector('.text-center');

  // Duplicate detection will happen after extracting the final content

  // Remove "no notes" message if it exists
  if (noNotesMsg) {
    noNotesMsg.remove();
  }

  // Create note element with modern styling
  const noteElement = document.createElement('div');
  noteElement.className = 'bg-dark-800 rounded-lg p-3 border border-dark-700 mb-3 hover:shadow-md transition-all duration-200';

  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const dateString = now.toLocaleDateString([], { month: 'short', day: 'numeric' });

  // Get current video time if available
  const currentVideo = document.getElementById('surgery-video');

  // Robust extraction: always put the actual note text into the content field
  let title = 'Note';
  let content = '';
  let category = 'General';

  // 1) Prefer the user's original message (strip the directive)
  if (userMessage) {
    const cleanedUser = userMessage
      .replace(/^\s*(?:take|make)\s+(?:a\s+)?note\b\s*(?:about|on|regarding|that)?\s*[:,-]?\s*/i, '')
      .trim();
    if (cleanedUser) {
      content = cleanedUser;
    }
  }

  // 2) If still empty, prefer explicit "Note:" content from the agent response
  if (!content) {
    const agentNoteMatch = (noteText || '').match(/note\s*:\s*([\s\S]+)/i);
    if (agentNoteMatch && agentNoteMatch[1]) {
      content = agentNoteMatch[1].trim();
    }
  }

  // 3) If still empty, clean meta wrappers from noteText
  if (!content) {
    let cleaned = (noteText || '').replace(/Note recorded[^.]*\./i, '').trim();
    cleaned = cleaned.replace(/Total notes[^.]*\./i, '').trim();
    cleaned = cleaned.replace(/^Note\s*:\s*/i, '').trim();
    if (cleaned) {
      content = cleaned;
    }
  }

  // 4) Final fallback: synthesize from time if we have nothing
  if (!content) {
    const currentTime = currentVideo ? formatTime(currentVideo.currentTime) : timeString;
    if (userMessage) {
      const fallbackClean = userMessage.replace(/^\s*(?:take|make)\s+(?:a\s+)?note\b\s*/i, '').trim();
      content = fallbackClean ? `Observation at ${currentTime}: ${fallbackClean}` : `Observation at ${currentTime}.`;
    } else {
      content = `Observation at ${currentTime}.`;
    }
  }

  // Compute a concise title but keep full content
  const firstSentence = content.split(/[.!?]/, 1)[0].trim();
  if (firstSentence) {
    title = firstSentence.length <= 60 ? firstSentence : (firstSentence.slice(0, 57) + '…');
  }

  // Determine category based on content keywords
  if (content.toLowerCase().includes('bleed') || content.toLowerCase().includes('blood')) {
    category = 'Bleeding';
  } else if (content.toLowerCase().includes('tool') || content.toLowerCase().includes('instrument')) {
    category = 'Tools';
  } else if (content.toLowerCase().includes('anatomy') || content.toLowerCase().includes('organ')) {
    category = 'Anatomy';
  } else if (content.toLowerCase().includes('procedure') || content.toLowerCase().includes('technique')) {
    category = 'Procedure';
  }

  // Second duplicate guard based on the extracted content (more reliable)
  const existingByContent = notesContainer.querySelectorAll('.note-content');
  for (let i = 0; i < existingByContent.length; i++) {
    if (existingByContent[i].textContent.trim() === content.trim()) {
      console.log("Duplicate note content detected, not adding again");
      return;
    }
  }

  noteElement.innerHTML = `
    <div class="flex justify-between items-start mb-2">
      <div class="flex-1">
        <div class="flex items-center">
          <h3 class="text-sm font-semibold text-primary-400 truncate">${title}</h3>
          <span class="ml-2 text-[10px] font-medium bg-primary-900/50 text-primary-300 px-1.5 py-0.5 rounded-full">${category}</span>
        </div>
        <div class="text-xs text-gray-400 mt-0.5 mb-1.5">${timeString} · Video: ${currentVideo ? formatTime(currentVideo.currentTime) : 'N/A'}</div>
      </div>
      <div class="flex ml-2 mt-1 space-x-1">
        <button class="edit-note-btn w-6 h-6 flex items-center justify-center text-xs text-gray-400 hover:text-primary-300 transition-colors rounded-full bg-dark-800 hover:bg-dark-700">
          <i class="fas fa-edit"></i>
        </button>
        <button class="delete-note-btn w-6 h-6 flex items-center justify-center text-xs text-gray-400 hover:text-red-400 transition-colors rounded-full bg-dark-800 hover:bg-dark-700">
          <i class="fas fa-trash-alt"></i>
        </button>
      </div>
    </div>
    <div class="text-xs text-gray-300 note-content leading-relaxed border-l-2 border-primary-800/30 pl-2 mb-1 max-h-24 overflow-y-auto">
      ${content}
    </div>
  `;

  // Add to container
  notesContainer.prepend(noteElement);

  // Add event listeners for the buttons
  const editBtn = noteElement.querySelector('.edit-note-btn');
  if (editBtn) {
    editBtn.addEventListener('click', function() {
      editNote(noteElement, title, content);
    });
  }

  const deleteBtn = noteElement.querySelector('.delete-note-btn');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', function() {
      deleteNote(noteElement);
    });
  }

  // Update the count
  const notesCount = document.querySelector('.notes-count');
  if (notesCount) {
    notesCount.textContent = parseInt(notesCount.textContent || '0') + 1;
  }

  // Also reflect the note in the chat window for visibility
  try {
    const chatSnippet = (content || '').trim();
    if (chatSnippet) {
      const maxLen = 300;
      const displayText = chatSnippet.length > maxLen ? chatSnippet.slice(0, maxLen - 1) + '…' : chatSnippet;
      addMessageToChat(`📝 Note saved: ${displayText}`, 'agent');
    } else {
      addMessageToChat('📝 Note saved.', 'agent');
    }
  } catch (e) {
    console.warn('Failed to add note echo to chat:', e);
  }
}

// Function to edit an existing note
function editNote(noteElement, title, content) {
  // Get the note modal elements
  const modal = document.getElementById('addNoteModal');
  const modalTitle = modal.querySelector('.modal-title');
  const titleInput = document.getElementById('note-title');
  const contentInput = document.getElementById('note-content');
  const saveButton = document.getElementById('save-note-btn');

  // Change modal title to indicate editing
  if (modalTitle) {
    modalTitle.innerHTML = '<i class="fas fa-edit text-primary-400 mr-2"></i> Edit Note';
  }

  // Pre-fill the form with existing values
  if (titleInput) titleInput.value = title;
  if (contentInput) contentInput.value = content;

  // Temporarily store the note element to edit
  saveButton.setAttribute('data-editing', 'true');
  saveButton.setAttribute('data-note-id', Date.now().toString()); // Use timestamp as a makeshift ID
  noteElement.id = saveButton.getAttribute('data-note-id');

  // Change save button text
  saveButton.innerHTML = '<i class="fas fa-save mr-1.5"></i> Update Note';

  // Update the save handler
  saveButton.onclick = function() {
    // Get the updated values
    const newTitle = titleInput.value.trim();
    const newContent = contentInput.value.trim();
    const message = document.getElementById('note-message').value.trim();

    if (!newTitle || !newContent) {
      showToast('Please enter a title and content for your note', 'error');
      return;
    }

    // Update the note HTML
    const noteToUpdate = document.getElementById(this.getAttribute('data-note-id'));
    if (noteToUpdate) {
      const titleEl = noteToUpdate.querySelector('h3');
      const contentEl = noteToUpdate.querySelector('.note-content');

      if (titleEl) titleEl.textContent = newTitle;
      if (contentEl) contentEl.innerHTML = newContent;

      // Close modal and clean up
      closeModal(modal);
      resetNoteForm();

      showToast('Note updated successfully', 'success');

      // Send any message to chat if provided
      if (message) {
        addMessageToChat(message, 'user');
        sendMessageToBackend(message);
      }
    }
  };

  // Open the modal
  modal.classList.add('show');
  modal.style.display = 'flex';
  document.body.classList.add('overflow-hidden');
}

// Function to delete a note
function deleteNote(noteElement) {
  // Ask for confirmation
  if (confirm('Are you sure you want to delete this note?')) {
    // Remove the note element
    noteElement.classList.add('opacity-0', 'scale-95');
    noteElement.style.transition = 'all 0.3s ease-in-out';

    // Add a slight delay before actually removing the element
    setTimeout(() => {
      noteElement.remove();

      // Update the count
      const notesCount = document.querySelector('.notes-count');
      if (notesCount) {
        const currentCount = parseInt(notesCount.textContent || '0');
        notesCount.textContent = Math.max(0, currentCount - 1);
      }

      // Check if there are no more notes and show the empty message
      const notesContainer = document.getElementById('notes-container');
      if (notesContainer && notesContainer.children.length === 0) {
        notesContainer.innerHTML = `
          <div class="text-center text-gray-400 p-4">
            <div class="flex items-center justify-center mb-2">
              <i class="fas fa-sticky-note text-xl mr-2 text-primary-700 opacity-60"></i>
              <span class="text-sm">No notes available</span>
            </div>
            <p class="text-xs">Ask the assistant to "take a note about..." something you observe</p>
          </div>
        `;
      }

      showToast('Note deleted successfully', 'success');
    }, 300);
  }
}

// Function to reset the note form
function resetNoteForm() {
  const modal = document.getElementById('addNoteModal');
  const modalTitle = modal.querySelector('.modal-title');
  const titleInput = document.getElementById('note-title');
  const contentInput = document.getElementById('note-content');
  const messageInput = document.getElementById('note-message');
  const saveButton = document.getElementById('save-note-btn');

  // Reset title
  if (modalTitle) {
    modalTitle.innerHTML = '<i class="fas fa-sticky-note text-primary-400 mr-2"></i> Add Note';
  }

  // Clear form inputs
  if (titleInput) titleInput.value = '';
  if (contentInput) contentInput.value = '';
  if (messageInput) messageInput.value = '';

  // Reset save button
  saveButton.removeAttribute('data-editing');
  saveButton.removeAttribute('data-note-id');
  saveButton.innerHTML = '<i class="fas fa-save mr-1.5"></i> Save Note';

  // Reset the onclick handler
  saveButton.onclick = saveManualNote;

  // Hide image preview if any
  const previewContainer = document.getElementById('note-image-preview-container');
  if (previewContainer) previewContainer.classList.add('hidden');
}

// Update the phase tag under the video
function updatePhaseFromAnnotation(annotationText) {
  const phaseElement = document.getElementById('current-phase');
  if (!phaseElement) return;

  const phaseMatch = annotationText.match(/Phase [\'"]([^'"]+)[\'"]/) || 
                     annotationText.match(/phase [\'"]([^'"]+)[\'"]/) || 
                     annotationText.match(/Phase: ([^|]+)/);

  if (phaseMatch) {
    const phaseName = phaseMatch[1].trim();
    phaseElement.textContent = phaseName;

    // Add animation to highlight the change
    phaseElement.style.transition = 'all 0.3s ease';
    phaseElement.style.backgroundColor = '#16a34a'; // success-600
    phaseElement.style.color = 'white';

    setTimeout(() => {
      phaseElement.style.backgroundColor = '';
      phaseElement.style.color = '';
    }, 1500);
  }
}

// Format time in MM:SS format (shared utility function)
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
}

// Video time tracking
document.addEventListener('DOMContentLoaded', function() {
  const video = document.getElementById('surgery-video');
  const currentTimeDisplay = document.getElementById('video-current-time');
  const durationDisplay = document.getElementById('video-duration');

  if (video && currentTimeDisplay && durationDisplay) {
    // Update time displays when metadata is loaded
    video.addEventListener('loadedmetadata', function() {
      durationDisplay.textContent = formatTime(video.duration);
    });

    // Update current time during playback
    video.addEventListener('timeupdate', function() {
      currentTimeDisplay.textContent = formatTime(video.currentTime);
    });
  }
});

// Fullscreen functionality for video
function toggleFullscreen() {
  const videoContainer = document.getElementById('video-container');

  if (!document.fullscreenElement) {
    if (videoContainer.requestFullscreen) {
      videoContainer.requestFullscreen();
    } else if (videoContainer.webkitRequestFullscreen) { /* Safari */
      videoContainer.webkitRequestFullscreen();
    } else if (videoContainer.msRequestFullscreen) { /* IE11 */
      videoContainer.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) { /* Safari */
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) { /* IE11 */
      document.msExitFullscreen();
    }
  }
}

// Custom toast notifications
function showToast(message, type = 'info', duration = 4000) {
  // Create toast container if it doesn't exist
  let toastContainer = document.querySelector('.toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    // Use inline styles to ensure positioning works
    Object.assign(toastContainer.style, {
      position: 'fixed',
      top: '1rem',
      right: '1rem',
      zIndex: '9999',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-end',
      gap: '0.5rem',
      maxWidth: '20rem'
    });
    document.body.appendChild(toastContainer);
  }

  // Create toast element
  const toast = document.createElement('div');
  toast.className = `custom-toast ${type}`;

  // Apply base toast styles inline to ensure they're applied
  Object.assign(toast.style, {
    display: 'flex',
    alignItems: 'center',
    padding: '0.75rem 1rem',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
    transform: 'translateX(100%)',
    opacity: '0',
    transition: 'all 0.3s ease',
    color: 'white',
    width: '100%',
    marginBottom: '0.5rem',
    borderLeft: '4px solid'
  });

  // Add appropriate icon and styling based on type
  let icon = 'info-circle';
  let bgColor, borderColor;

  if (type === 'success') {
    icon = 'check-circle';
    bgColor = 'linear-gradient(to right, #059669, #047857)'; // green-600 to green-700
    borderColor = '#34d399'; // green-400
  } else if (type === 'error') {
    icon = 'exclamation-triangle';
    bgColor = 'linear-gradient(to right, #dc2626, #b91c1c)'; // red-600 to red-700
    borderColor = '#f87171'; // red-400
  } else {
    // info default
    bgColor = 'linear-gradient(to right, #0369a1, #075985)'; // primary-700 to primary-800
    borderColor = '#38bdf8'; // primary-400
  }

  toast.style.background = bgColor;
  toast.style.borderLeftColor = borderColor;

  // Add close button and improved styling
  toast.innerHTML = `
    <i class="fas fa-${icon}" style="margin-right: 0.75rem; font-size: 1.1rem;"></i>
    <span style="flex-grow: 1; font-size: 0.9rem;">${message}</span>
    <button style="margin-left: 0.5rem; opacity: 0.8; cursor: pointer; background: none; border: none; color: white;" 
            onmouseover="this.style.opacity='1'" 
            onmouseout="this.style.opacity='0.8'" 
            onclick="this.parentNode.remove()">
      <i class="fas fa-times"></i>
    </button>
  `;

  // Add to container
  toastContainer.appendChild(toast);

  // Trigger animation after a small delay
  setTimeout(() => {
    toast.style.transform = 'translateX(0)';
    toast.style.opacity = '1';
  }, 10);

  // Remove after duration
  const timeoutId = setTimeout(() => {
    toast.style.transform = 'translateX(100%)';
    toast.style.opacity = '0';
    setTimeout(() => {
      if (toast.parentNode) {
        toast.remove();
      }
    }, 300);
  }, duration);

  // Cancel timeout when manually closed
  toast.querySelector('button').addEventListener('click', () => {
    clearTimeout(timeoutId);
  });
}

// Simple recent-duplicate suppression for chat messages
const _recentMsgs = { user: [], agent: [] }; // keep last few messages with timestamps
function _pushRecent(sender, text) {
  const arr = _recentMsgs[sender] || (_recentMsgs[sender] = []);
  const now = Date.now();
  arr.push({ text, t: now });
  // keep last 10 and trim older than 8 seconds
  while (arr.length > 10) arr.shift();
  _recentMsgs[sender] = arr.filter(it => now - it.t < 8000);
}
function _isRecentDuplicate(sender, text) {
  const arr = _recentMsgs[sender] || [];
  const now = Date.now();
  // consider duplicate if same text seen in last 4 seconds
  return arr.some(it => it.text === text && (now - it.t) < 4000);
}

// Chat functions
function onChatMessageKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    onChatMessageSubmit();
  }
}

function onChatMessageSubmit() {
  const inputElement = document.getElementById('chat-message-input');
  const message = inputElement.value.trim();

  if (message) {
    // Clear input
    inputElement.value = '';

    // Add to chat history
    addMessageToChat(message, 'user');

    // Send to backend with current video frame
    sendMessageToBackend(message);
  }
}

// Function to send a message to the backend with a video frame
function sendMessageToBackend(message) {
  try {
    // Prepare payload with message and session ID
    const payload = {
      user_input: message,
      original_user_input: message, // Store original message for note processing
      session_id: sessionId // Add session ID to track messages
    };

    // First try to get a new frame by capturing the current video
    let frameData = captureVideoFrame();

    // If we couldn't get a new frame, try the previously stored frame
    if (!frameData) {
      console.warn("Could not capture current frame, trying to use last captured frame");
      frameData = sessionStorage.getItem('lastCapturedFrame');
    }

    // If we still don't have a frame but have a video loaded, try seeking and capturing
    if (!frameData) {
      const videoElement = document.getElementById('surgery-video');
      if (videoElement && videoElement.readyState >= 2) {
        // Try to seek to first frame to ensure we can capture something
        try {
          console.warn("Attempting frame capture by seeking to beginning of video");

          // Store current playback state and position
          const wasPlaying = !videoElement.paused;
          const currentTime = videoElement.currentTime;

          // Pause if playing
          if (wasPlaying) {
            videoElement.pause();
          }

          // Seek to a small offset (0.1 second) to ensure a frame is available
          videoElement.currentTime = 0.1;

          // Wait a tiny bit for the frame to load
          setTimeout(() => {
            // Try to capture again
            frameData = captureVideoFrame();

            if (frameData) {
              console.log("Successfully captured frame after seeking");
              sessionStorage.setItem('lastCapturedFrame', frameData);

              // Send the message with the new frame
              completeMessageSend(message, frameData);
            } else {
              // Still couldn't get a frame, use placeholders
              fallbackFrameCapture(message);
            }

            // Restore playback state
            try {
              videoElement.currentTime = currentTime;
              if (wasPlaying) {
                videoElement.play().catch(e => console.warn("Could not resume playback:", e));
              }
            } catch (e) {
              console.warn("Error restoring video state:", e);
            }
          }, 50);

          // Return early as we'll send the message in the callback
          return;
        } catch (e) {
          console.warn("Error while trying to seek for frame capture:", e);
        }
      }

      // If we're still here, try placeholder
      fallbackFrameCapture(message);
      return;
    }

    // If we got here with a frame, send it
    completeMessageSend(message, frameData);
  } catch (err) {
    console.error("Error sending message to backend:", err);
    showToast("Error sending message: " + err.message, "error");

    // Try to send just the message without a frame
    if (typeof sendJSON === 'function') {
      sendJSON({
        user_input: message,
        original_user_input: message
      });
      console.log("Sent message WITHOUT frame data due to error");
    }
  }
}

// Function to create a placeholder image frame
function createPlaceholderFrame() {
  // Check if we already have a placeholder frame
  let frameData = sessionStorage.getItem('placeholderFrame');
  if (frameData) {
    console.log("Using existing placeholder frame");
    return frameData;
  }

  try {
    // Create a placeholder canvas with text
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');

    // Fill with dark background
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add text indicating this is a placeholder
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Surgical Agentic Framework Demo', canvas.width/2, canvas.height/2 - 20);

    ctx.font = '16px Arial';
    ctx.fillText('A frame will be captured when a video is playing', canvas.width/2, canvas.height/2 + 20);

    // Convert to data URL and store it
    frameData = canvas.toDataURL('image/jpeg', 0.8);
    sessionStorage.setItem('placeholderFrame', frameData);
    console.log("Created new placeholder frame");
    return frameData;
  } catch (err) {
    console.error("Error creating placeholder frame:", err);
    return null;
  }
}

// Create placeholder frame when all other methods fail
function fallbackFrameCapture(message) {
  console.warn("No previously captured frame available, using fallback");
  let frameData = createPlaceholderFrame();

  // If we still don't have a frame, create a message-specific one
  if (!frameData) {
    try {
      // Create a placeholder canvas with text
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 360;
      const ctx = canvas.getContext('2d');

      // Fill with dark background
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add text indicating no frame is available
      ctx.fillStyle = '#ffffff';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No video frame available', canvas.width/2, canvas.height/2);
      ctx.font = '16px Arial';
      ctx.fillText('Question: "' + message.substring(0, 30) + (message.length > 30 ? '...' : '') + '"', canvas.width/2, canvas.height/2 + 30);

      // Convert to data URL
      frameData = canvas.toDataURL('image/jpeg', 0.8);

      // Store this placeholder for potential future use
      sessionStorage.setItem('placeholderFrame', frameData);
    } catch (canvasErr) {
      console.error("Error creating placeholder frame:", canvasErr);
    }
  }

  completeMessageSend(message, frameData);
}

// Final step to send message with whatever frame we have
function completeMessageSend(message, frameData) {
  // Prepare payload with message
  const payload = {
    user_input: message,
    original_user_input: message
  };

  // Always try to provide some frame data, even if it's a placeholder
  if (!frameData) {
    console.warn("No frame data provided, creating placeholder");
    frameData = createPlaceholderFrame();
  }

  // Add frame data to payload
  if (frameData) {
    payload.frame_data = frameData;
    console.log("Frame data added to message payload");
  } else {
    // This should almost never happen since we generate placeholders
    console.warn("Failed to create any frame data to send with message");
  }

  // Send payload to server
  if (typeof sendJSON === 'function') {
    // Show sending status if we have frame data
    if (frameData) {
      showFrameSendingStatus();
    }

    sendJSON(payload);
    console.log("Message sent to backend" + (frameData ? " with frame data" : " WITHOUT frame data"));
  } else {
    console.error("sendJSON function not available");
    showToast("Unable to send message to server", "error");
  }
}

// Function to send frame data with text
function sendFrameWithText(text) {
  try {
    // Delegate to the common send function with force capture enabled
    sendTextWithMaxCapture(text);
  } catch (err) {
    console.error("Error sending frame with text:", err);

    // Fallback - just send the text without a frame
    if (typeof sendJSON === 'function') {
      sendJSON({
        user_input: text,
        asr_final: true
      });
      console.log("Text sent without frame (error recovery): " + text);
    }
  }
}

// Enhanced function that tries harder to get a frame from the video
function sendTextWithMaxCapture(text) {
  // Try to get a frame directly
  let frameData = captureVideoFrame();

  // If that worked, use it
  if (frameData) {
    sessionStorage.setItem('lastCapturedFrame', frameData);
    completeFrameWithTextSend(text, frameData);
    return;
  }

  // Otherwise try getting the last stored frame
  frameData = sessionStorage.getItem('lastCapturedFrame');
  if (frameData) {
    completeFrameWithTextSend(text, frameData);
    return;
  }

  // If we still don't have a frame but video is loaded, try seek and capture
  const videoElement = document.getElementById('surgery-video');
  if (videoElement && videoElement.readyState >= 2) {
    // Try to force a frame capture
    const wasPlaying = !videoElement.paused;
    const currentTime = videoElement.currentTime;

    // Pause playback
    if (wasPlaying) {
      videoElement.pause();
    }

    try {
      // Seek to start if needed to ensure we get a frame
      if (videoElement.currentTime > 10) {
        // If we're far into the video, seek to 0.1s
        videoElement.currentTime = 0.1;
      } else {
        // Otherwise just seek a tiny bit forward to force frame update
        videoElement.currentTime = Math.max(0.1, videoElement.currentTime + 0.1);
      }

      // Wait a brief moment for the frame to load
      setTimeout(() => {
        // Try to capture again
        frameData = captureVideoFrame();

        if (frameData) {
          // Success! Store and send
          sessionStorage.setItem('lastCapturedFrame', frameData);
          completeFrameWithTextSend(text, frameData);
        } else {
          // Still no frame - use placeholder
          createPlaceholderFrameWithText(text);
        }

        // Restore video state
        try {
          videoElement.currentTime = currentTime;
          if (wasPlaying) {
            videoElement.play().catch(e => {});
          }
        } catch (e) {}
      }, 100);

      return; // Will continue in the timeout
    } catch (e) {
      console.warn("Error during forced frame capture:", e);
      // Continue to placeholder
    }
  }

  // If all else failed, use a placeholder
  createPlaceholderFrameWithText(text);
}

// Create a placeholder frame with the specified text
function createPlaceholderFrameWithText(text) {
  // Check if we already have a placeholder
  let frameData = sessionStorage.getItem('placeholderFrame');

  // If not, create one
  if (!frameData) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 360;
      const ctx = canvas.getContext('2d');

      // Dark background
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add helpful text
      ctx.fillStyle = '#ffffff';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No video frame available', canvas.width/2, canvas.height/2);
      ctx.font = '16px Arial';
      ctx.fillText('Voice command: "' + text.substring(0, 30) + (text.length > 30 ? '...' : '') + '"', canvas.width/2, canvas.height/2 + 30);

      // Convert to data URL
      frameData = canvas.toDataURL('image/jpeg', 0.8);

      // Store for future use
      sessionStorage.setItem('placeholderFrame', frameData);
    } catch (canvasErr) {
      console.error("Error creating placeholder frame:", canvasErr);
    }
  }

  // Send whatever we have
  completeFrameWithTextSend(text, frameData);
}

// Complete the send operation with whatever frame we've managed to get
function completeFrameWithTextSend(text, frameData) {
  // Prepare payload with required fields
  const payload = {
    user_input: text,
    asr_final: true
  };

  // Add frame data if we have it
  if (frameData) {
    payload.frame_data = frameData;
  }

  // Send to server
  if (typeof sendJSON === 'function') {
    sendJSON(payload);
    console.log("Voice input sent" + (frameData ? " with frame" : " WITHOUT frame") + ": " + text);

    // Show a warning to the user if no frame was available
    if (!frameData) {
      showToast("No video frame available - AI response may be limited", "warning");
    }
  } else {
    console.error("sendJSON function not available");
    showToast("Unable to send message", "error");
  }
}

function addMessageToChat(message, sender = 'user', agentName) {
  const chatHistoryContainer = document.getElementById('chat-history-container');

  // Check if this message already exists (to prevent duplicates)
  const lastMessage = chatHistoryContainer.lastElementChild;
  if (lastMessage && 
      lastMessage.classList.contains(sender === 'user' ? 'user-message' : 'agent-message') &&
      lastMessage.querySelector('.message-content').textContent === message) {
    console.log("Duplicate message detected, not adding again");
    return;
  }
  // Broader recent duplicate suppression (covers race-y double inserts)
  if (_isRecentDuplicate(sender, message)) {
    console.log("Recent duplicate suppressed for", sender);
    return;
  }

  const messageDiv = document.createElement('div');
  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (sender === 'user') {
    messageDiv.className = 'user-message';
    messageDiv.innerHTML = `
      <div class="flex items-center justify-between mb-0 -mt-1 -mx-1">
        <span class="flex items-center">
          <span class="avatar-icon bg-primary-700">
            <i class="fas fa-user"></i>
          </span>
          <span class="text-[11px] text-white/90">You</span>
        </span>
        <span class="text-[10px] text-primary-200/80">${timeString}</span>
      </div>
      <div class="message-content">
        ${message}
      </div>
    `;
  } else {
    messageDiv.className = 'agent-message';
    const displayName = agentName || 'AI Assistant';
    messageDiv.innerHTML = `
      <div class="flex items-center justify-between mb-0 -mt-1 -mx-1">
        <span class="flex items-center">
          <span class="avatar-icon bg-success-600">
            <i class="fas fa-robot"></i>
          </span>
          <span class="text-[11px] text-success-300/90">${displayName}</span>
        </span>
        <span class="text-[10px] text-gray-400/80">${timeString}</span>
      </div>
      <div class="message-content">
        ${message}
      </div>
    `;

    // Add entry animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(10px)';

    setTimeout(() => {
      messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
      messageDiv.style.opacity = '1';
      messageDiv.style.transform = 'translateY(0)';
    }, 10);
  }

  chatHistoryContainer.appendChild(messageDiv);
  _pushRecent(sender, message);
  chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

function onChatHistoryReset() {
  const chatHistoryContainer = document.getElementById('chat-history-container');

  // Stop any playing TTS audio
  resetTTSState();

  // Remove all messages except the welcome message
  while (chatHistoryContainer.childNodes.length > 1) {
    chatHistoryContainer.removeChild(chatHistoryContainer.lastChild);
  }
}

// Functions for mic control
function toggleMic() {
  const micBtn = document.getElementById('mic-btn');

  // Check if button is disabled
  if (micBtn.disabled) {
    showToast('Please load a video or connect to a live stream before using the microphone', 'error');
    return;
  }

  const isRecording = micBtn.classList.contains('recording');

  if (isRecording) {
    // Stop recording - Return to blue
    micBtn.classList.remove('recording');
    micBtn.classList.remove('from-red-600', 'to-red-700');
    micBtn.classList.remove('hover:from-red-500', 'hover:to-red-600');
    micBtn.classList.add('from-primary-600', 'to-primary-700');
    micBtn.classList.add('hover:from-primary-500', 'hover:to-primary-600');
    micBtn.innerHTML = '<i class="fas fa-microphone mr-2"></i> <span>Start Mic</span>';

    // Stop the recording
    stopRecording();
  } else {
    // Start recording - Make button red
    micBtn.classList.add('recording');
    micBtn.classList.remove('from-primary-600', 'to-primary-700');
    micBtn.classList.remove('hover:from-primary-500', 'hover:to-primary-600');
    micBtn.classList.add('from-red-600', 'to-red-700');
    micBtn.classList.add('hover:from-red-500', 'hover:to-red-600');
    micBtn.innerHTML = '<i class="fas fa-stop-circle mr-2"></i> <span>Stop Mic</span>';

    // Start the recording
    startRecording();
  }
}

// Function to enable the mic button when a video is loaded
function enableMicButton() {
  const micBtn = document.getElementById('mic-btn');
  if (micBtn) {
    micBtn.disabled = false;
    micBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    console.log('Microphone button enabled');
  }
}

// Function to disable the mic button when no video is available
function disableMicButton() {
  const micBtn = document.getElementById('mic-btn');
  if (micBtn) {
    micBtn.disabled = true;
    micBtn.classList.add('opacity-50', 'cursor-not-allowed');
    console.log('Microphone button disabled');
  }
}

// Function to show video placeholder when no video is selected
function showVideoPlaceholder() {
  const videoElement = document.getElementById('surgery-video');
  if (videoElement) {
    // Create a canvas to show "No Video Selected" message
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');

    // Fill background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw border
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);

    // Draw text
    ctx.fillStyle = '#999';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('No Video Selected', canvas.width / 2, canvas.height / 2 - 10);

    ctx.font = '16px Arial';
    ctx.fillText('Upload or select a video to begin', canvas.width / 2, canvas.height / 2 + 20);

    // Convert canvas to data URL and set as poster
    const dataURL = canvas.toDataURL();
    videoElement.poster = dataURL;

    // Clear any existing source
    videoElement.removeAttribute('src');
    const sources = videoElement.querySelectorAll('source');
    sources.forEach(source => source.removeAttribute('src'));
    videoElement.load();
  }
}

// Connect to the audio.js functions for recording
function startRecording() {
  try {
    // Capture a frame at the start of recording to ensure we have one
    const frameData = captureVideoFrame();
    if (frameData) {
      sessionStorage.setItem('lastCapturedFrame', frameData);
      console.log("Captured initial frame for voice recording");
    }

    if (typeof startAudio === 'function') {
      // Pass session ID to the audio module
      startAudio(sessionId);
      showToast('Recording started', 'info');
    } else {
      console.error('startAudio function not found!');
      showToast('Error starting recording - audio.js not loaded correctly', 'error');
    }
  } catch (err) {
    console.error('Error starting recording:', err);
    showToast('Failed to start recording: ' + err.message, 'error');
  }
}

function stopRecording() {
  try {
    // Capture a final frame before stopping
    const frameData = captureVideoFrame();
    if (frameData) {
      sessionStorage.setItem('lastCapturedFrame', frameData);
      console.log("Captured final frame for voice recording");
    }

    if (typeof stopAudio === 'function') {
      stopAudio();
      showToast('Processing your voice input...', 'info');
    } else {
      console.error('stopAudio function not found!');
      showToast('Error stopping recording - audio.js not loaded correctly', 'error');
    }
  } catch (err) {
    console.error('Error stopping recording:', err);
    showToast('Failed to stop recording: ' + err.message, 'error');
  }
}

// Image capture function
function captureAndStoreFrame() {
  try {
    const frameData = captureVideoFrame();
    if (frameData) {
      // Store the captured frame in sessionStorage
      sessionStorage.setItem('lastCapturedFrame', frameData);
      // Update the display with manual capture status
      updateCapturedFrameDisplay(frameData, 'Manually captured', 'manual');
      showToast('Frame captured successfully!', 'success');
    } else {
      updateFrameStatus('Failed to capture frame', 'text-xs text-red-400');
      showToast('Failed to capture frame - video not playing', 'error');
    }
  } catch (err) {
    console.error('Error capturing frame:', err);
    updateFrameStatus('Capture error: ' + err.message, 'text-xs text-red-400');
    showToast('Error capturing frame: ' + err.message, 'error');
  }
}

// Auto frame capture for annotations
let frameCapture = null;
const FRAME_CAPTURE_INTERVAL = 10000; // 10 seconds

function startAutoFrameCapture() {
  // Clear any existing interval
  stopAutoFrameCapture();

  // Force a capture now to ensure we have a frame, even if the video is paused
  let initialFrame = captureVideoFrame();

  // If we couldn't capture, but a video is loaded, try to seek to the first frame
  if (!initialFrame) {
    const videoElement = document.getElementById('surgery-video');
    if (videoElement && videoElement.readyState >= 2) {
      // Try to seek to first frame to ensure we can capture something
      try {
        // Store current playback state
        const wasPlaying = !videoElement.paused;

        // Pause if playing
        if (wasPlaying) {
          videoElement.pause();
        }

        // Seek to a small offset (0.1 second) to ensure a frame is available
        videoElement.currentTime = 0.1;

        // Try to capture again
        console.log("Attempting to capture after seeking to first frame");
        initialFrame = captureVideoFrame();

        // Resume playback if it was playing before
        if (wasPlaying) {
          videoElement.play().catch(e => console.warn("Could not resume playback:", e));
        }
      } catch (e) {
        console.warn("Error while trying to seek for frame capture:", e);
      }
    }
  }

  // If we now have a frame, store and send it
  if (initialFrame) {
    // Store the successfully captured frame
    sessionStorage.setItem('lastCapturedFrame', initialFrame);
    console.log("Initial frame captured and stored successfully");

    // Update the display
    updateCapturedFrameDisplay(initialFrame, 'Initial frame captured', 'auto');

    // Send to server for annotation
    if (typeof sendJSON === 'function') {
      const payload = createFrameMessagePayload(initialFrame);
      sendJSON(payload);
      console.log("Initial frame sent for annotation:", payload);
    }
  } else {
    console.warn("Failed to capture initial frame - ChatBot responses may be limited");
    showToast("Could not capture video frame - AI responses may be limited", "warning");
  }

  // Start regular interval for frame capture
  frameCapture = setInterval(() => {
    // Try to capture current frame
    const frameData = captureVideoFrame();

    if (frameData) {
      // Store the frame for future use
      sessionStorage.setItem('lastCapturedFrame', frameData);

      // Send frame to server with auto_frame flag for annotation
      if (typeof sendJSON === 'function') {
        const payload = createFrameMessagePayload(frameData);
        sendJSON(payload);
        console.log("Auto-captured frame sent for annotation:", payload);
      }
    } else {
      // We couldn't capture a frame on this interval
      console.warn("Failed to capture automatic frame");

      // Try to use any previously stored frame for annotation
      const lastFrame = sessionStorage.getItem('lastCapturedFrame');
      if (lastFrame && typeof sendJSON === 'function') {
        updateCapturedFrameDisplay(lastFrame, 'Auto-capture failed - using previous', 'fallback');
        const payload = createFrameMessagePayload(lastFrame);
        sendJSON(payload);
        console.log("Using previously captured frame for annotation:", payload);
      }
    }
  }, FRAME_CAPTURE_INTERVAL);

  console.log("Auto frame capture started");
}

function stopAutoFrameCapture() {
  if (frameCapture) {
    clearInterval(frameCapture);
    frameCapture = null;
    console.log("Auto frame capture stopped");
  }
}

// Function to capture the current video frame
function captureVideoFrame() {
  const videoElement = document.getElementById('surgery-video');

  if (!videoElement) {
    console.warn('Video element not found');
    return null;
  }

  // Check if we have either a src (uploaded video) or srcObject (WebRTC stream)
  const hasUploadedVideo = videoElement.src && videoElement.src !== window.location.href;
  const hasWebRTCStream = videoElement.srcObject !== null;

  if (!hasUploadedVideo && !hasWebRTCStream) {
    console.warn('Video source is not set - no src or srcObject');
    return null;
  }

  // Check if the video element has a valid size
  const hasValidSize = videoElement.videoWidth > 0 && videoElement.videoHeight > 0;

  // You can capture frames from paused videos as well, as long as they've loaded
  // Only require the video to be playing if we don't have a current frame (initial load)
  const hasCurrentFrame = sessionStorage.getItem('lastCapturedFrame') !== null;

  if (videoElement.readyState < 2) { // HAVE_CURRENT_DATA (2) or higher needed
    console.warn('Video not ready for frame capture, readyState:', videoElement.readyState);
    return null;
  }

  // Allow capturing from paused videos if they've loaded a frame
  const canCapture = hasValidSize && videoElement.readyState >= 2;
  if (!canCapture) {
    console.warn('Video not ready for capture - either not loaded or no dimensions');

    // Return last captured frame if available
    const lastFrame = sessionStorage.getItem('lastCapturedFrame');
    if (lastFrame) {
      console.log("Using previous frame since video isn't ready for capture");
      updateCapturedFrameDisplay(lastFrame, 'Video not ready - using previous', 'fallback');
      return lastFrame;
    }

    // Create a placeholder frame if we can't get a real one
    const placeholder = createPlaceholderFrame();
    updateCapturedFrameDisplay(placeholder, 'Video not ready - using placeholder', 'placeholder');
    return placeholder;
  }

  try {
    // Create a canvas element
    const canvas = document.createElement('canvas');

    // Check if video dimensions are available
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      console.warn('Video dimensions are not available yet');
      // Return a placeholder frame instead of trying with default dimensions
      const placeholder = createPlaceholderFrame();
      updateCapturedFrameDisplay(placeholder, 'No video dimensions - using placeholder', 'placeholder');
      return placeholder;
    } else {
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
    }

    // Draw the current frame to the canvas
    const ctx = canvas.getContext('2d');

    // This try/catch specifically focuses on the drawing operation
    try {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    } catch (drawErr) {
      console.error('Error drawing video frame to canvas:', drawErr);

      // Return the last frame we captured if available
      const lastFrame = sessionStorage.getItem('lastCapturedFrame');
      if (lastFrame) {
        console.log("Using last captured frame after drawing error");
        updateCapturedFrameDisplay(lastFrame, 'Drawing error - using previous', 'fallback');
        return lastFrame;
      }

      // Fallback to placeholder
      const placeholder = createPlaceholderFrame();
      updateCapturedFrameDisplay(placeholder, 'Drawing error - using placeholder', 'placeholder');
      return placeholder;
    }

    // Convert to base64 data URL
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);

    // Verify we got a valid data URL (should start with 'data:image/jpeg;base64,')
    if (!dataURL || !dataURL.startsWith('data:image/jpeg;base64,')) {
      console.error('Invalid data URL generated from canvas');

      // Try to use previous frame
      const lastFrame = sessionStorage.getItem('lastCapturedFrame');
      if (lastFrame) {
        console.log("Invalid data URL - using previous frame instead");
        updateCapturedFrameDisplay(lastFrame, 'Invalid data - using previous', 'fallback');
        return lastFrame;
      }

      const placeholder = createPlaceholderFrame();
      updateCapturedFrameDisplay(placeholder, 'Invalid data - using placeholder', 'placeholder');
      return placeholder;
    }

    // Store successfully captured frame in session storage for fallback
    sessionStorage.setItem('lastCapturedFrame', dataURL);

    // Update the captured frame display in the UI
    updateCapturedFrameDisplay(dataURL, 'Frame captured', 'video');

    console.log('Frame captured successfully');
    return dataURL;
  } catch (err) {
    console.error('Error capturing video frame:', err);

    // Try to get the last stored frame
    const lastFrame = sessionStorage.getItem('lastCapturedFrame');
    if (lastFrame) {
      console.log("Using previously captured frame after error");
      updateCapturedFrameDisplay(lastFrame, 'Using previous frame', 'fallback');
      return lastFrame;
    }

    // If nothing else works, create a placeholder
    const placeholder = createPlaceholderFrame();
    updateCapturedFrameDisplay(placeholder, 'Placeholder frame', 'placeholder');
    return placeholder;
  }
}

// Generate summary function
function generateSummary() {
  const summaryContainer = document.getElementById('summary-container');

  // Show loading state
  summaryContainer.innerHTML = `
    <div class="flex justify-center items-center h-full">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  `;

  // Prevent multiple clicks
  const generateButton = document.getElementById('generate-summary-btn');
  if (generateButton) {
    generateButton.disabled = true;
    generateButton.classList.add('opacity-50');
  }

  // Add this function if it doesn't exist yet
  if (typeof displayFormattedPostOpNote !== 'function') {
    // Function to display formatted post-op note from JSON data
    window.displayFormattedPostOpNote = function(postOpNote, container) {
      if (!postOpNote) {
        container.innerHTML = `
          <div class="p-4 border border-red-700 rounded-lg">
            <h3 class="text-lg font-semibold mb-2 text-red-400">Error</h3>
            <p class="text-sm text-gray-300">No post-op note data was provided.</p>
          </div>
        `;
        return;
      }

      // Format procedure information
      const procInfo = postOpNote.procedure_information || {};

      // Create HTML content
      let html = `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
          <div class="space-y-2">
            <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> ${procInfo.procedure_type || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Date:</span> ${procInfo.date || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> ${procInfo.duration || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Surgeon:</span> ${procInfo.surgeon || 'Not specified'}</p>
          </div>
        </div>
      `;

      // Add findings section if available
      const findings = postOpNote.findings || [];
      if (findings.length > 0) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Key Findings</h3>
            <ul class="list-disc list-inside space-y-1 text-sm">
        `;

        findings.forEach(finding => {
          if (finding && finding.trim()) {
            html += `<li>${finding}</li>`;
          }
        });

        html += `
            </ul>
          </div>
        `;
      }

      // Add timeline section if available
      const timeline = postOpNote.procedure_timeline || [];
      if (timeline.length > 0) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Timeline</h3>
            <ul class="list-disc list-inside space-y-1 text-sm">
        `;

        timeline.forEach(event => {
          if (event.description && event.description.trim()) {
            html += `<li><span class='font-medium text-primary-300'>${event.time || 'Unknown'}</span>: ${event.description}</li>`;
          }
        });

        html += `
            </ul>
          </div>
        `;
      }

      // Add complications section if available
      const complications = postOpNote.complications || [];
      if (complications.length > 0) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Complications</h3>
            <ul class="list-disc list-inside space-y-1 text-sm">
        `;

        complications.forEach(complication => {
          if (complication && complication.trim()) {
            html += `<li>${complication}</li>`;
          }
        });

        html += `
            </ul>
          </div>
        `;
      }

      // If no substantive content, show a message
      if (!findings.length && !timeline.length && !complications.length) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Additional Information</h3>
            <p class="text-sm text-gray-300">
              Insufficient procedure data is available for a detailed summary. 
              For better results, add more annotations and notes during the procedure.
            </p>
          </div>
        `;
      }

      // Set the HTML
      container.innerHTML = html;
    };
  }

  // Gather data from notes
  const notesContainer = document.getElementById('notes-container');
  const notes = notesContainer ? Array.from(notesContainer.querySelectorAll('.bg-dark-800:not(.text-center)')) : [];

  // Gather data from annotations
  const annotationsContainer = document.getElementById('annotations-container');
  const annotations = annotationsContainer ? Array.from(annotationsContainer.querySelectorAll('.bg-dark-800:not(.text-center)')) : [];

  // Get current video duration
  const video = document.getElementById('surgery-video');
  const videoDuration = video ? formatTime(video.duration) : 'Unknown';

  // Extract note data to send to backend
  const noteData = notes.map(note => {
    const contentEl = note.querySelector('.note-content');
    const content = contentEl ? contentEl.textContent.trim() : '';
    const timeEl = note.querySelector('.text-xs.text-gray-500') || note.querySelector('[class*="text-"].text-gray-500');
    const time = timeEl ? timeEl.textContent.trim() : '';
    return { text: content, timestamp: time };
  }).filter(note => note.text && note.text !== 'Take a note');

  // Extract annotation data to send to backend
  const annotationData = annotations.map(ann => {
    // For annotations, the content is in a p tag with text-sm text-gray-300 classes
    const contentEl = ann.querySelector('p.text-sm.text-gray-300');
    const content = contentEl ? contentEl.textContent.trim() : '';

    // Get timestamp from the span
    const timeEl = ann.querySelector('.text-xs.text-gray-400');
    const time = timeEl ? timeEl.textContent.trim() : '';

    // Get surgical phase from the h3 tag
    const phaseEl = ann.querySelector('h3.text-lg.font-semibold');
    const phase = phaseEl ? phaseEl.textContent.trim() : '';

    // Get tools and anatomy from the badges
    const toolsEl = ann.querySelector('.badge.text-primary-300');
    const tools = toolsEl ? toolsEl.textContent.trim() : '';

    const anatomyEl = ann.querySelector('.badge.text-yellow-300');
    const anatomy = anatomyEl ? anatomyEl.textContent.trim() : '';

    return { 
      description: content, 
      timestamp: time,
      surgical_phase: phase,
      tools: tools ? [tools] : [],
      anatomy: anatomy ? [anatomy] : []
    };
  }).filter(ann => ann.description);

  // Log what we're sending to help with debugging
  console.log("Generating summary with:");
  console.log("- Notes:", noteData.length, noteData);
  console.log("- Annotations:", annotationData.length, annotationData);

  // Generate request to backend for summary generation with data
  fetch('/api/generate_post_op_note', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      request_type: 'generate_summary',
      notes: noteData,
      annotations: annotationData,
      video_duration: videoDuration
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Failed to generate summary');
    }
    return response.json();
  })
  .then(data => {
    console.log("Summary response:", data);

    if (data && data.post_op_note) {
      // Render structured note directly in the Summary tab
      renderPostOpNote(data.post_op_note);
    } else if (data && data.summary) {
      // Backward compatibility with old API
      summaryContainer.innerHTML = `
        <div class="space-y-4">
          ${data.summary}
        </div>
      `;
    } else {
      // Fallback to client-side summary if the server doesn't return one
      fallbackGenerateSummary(notes, annotations, videoDuration, summaryContainer);
    }

    // Re-enable the button
    if (generateButton) {
      generateButton.disabled = false;
      generateButton.classList.remove('opacity-50');
    }
  })
  .catch(error => {
    console.error('Error generating summary:', error);

    // Fallback to client-side generation
    fallbackGenerateSummary(notes, annotations, videoDuration, summaryContainer);

    // Re-enable the button
    if (generateButton) {
      generateButton.disabled = false;
      generateButton.classList.remove('opacity-50');
    }
  });
}

// Fallback client-side summary generation
function fallbackGenerateSummary(notes, annotations, videoDuration, summaryContainer) {
  // Process after a small delay to show loading animation
  setTimeout(() => {
    // Extract procedure information
    const procedureType = determineTypeFromAnnotations(annotations);
    const phases = extractPhasesFromAnnotations(annotations);
    const keyEventsFromNotes = extractKeyEventsFromNotes(notes);
    const keyEventsFromAnnotations = extractKeyEventsFromAnnotations(annotations);

    // Combine key events and sort by timestamp
    const allKeyEvents = [...keyEventsFromNotes, ...keyEventsFromAnnotations]
      .filter(event => event && event.trim()) // Filter out empty events
      .sort((a, b) => {
        // Extract time values for comparison (assuming format like "00:12:18")
        const timeA = a.match(/(\d+:\d+)/);
        const timeB = b.match(/(\d+:\d+)/);
        if (!timeA || !timeB) return 0;

        const [minsA, secsA] = timeA[1].split(':').map(Number);
        const [minsB, secsB] = timeB[1].split(':').map(Number);

        return (minsA * 60 + secsA) - (minsB * 60 + secsB);
      });

    // Get actual notes content (that has meaningful data)
    const noteSummary = summarizeNotesByCategory(notes);
    const hasValidNotes = noteSummary && noteSummary.trim() && !noteSummary.includes('undefined');

    // Generate the HTML for the procedure info - always show this section
    let summaryHTML = `
      <div class="space-y-4">
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
          <div class="space-y-2">
            <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> ${procedureType}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> ${videoDuration}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Phases:</span> ${phases.join(', ') || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Notes:</span> ${notes.length} | <span class="font-medium text-gray-400">Annotations:</span> ${annotations.length}</p>
          </div>
        </div>
    `;

    // Only add key events section if we have actual events
    if (allKeyEvents.length > 0) {
      summaryHTML += `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Key Events</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${allKeyEvents.map(event => `<li>${event}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Only add notes summary if we have actual notes with content
    if (hasValidNotes) {
      summaryHTML += `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Notes Summary</h3>
          <div class="space-y-3">
            ${noteSummary}
          </div>
        </div>
      `;
    }

    // Add a default message if we don't have much information to show
    if (!allKeyEvents.length && !hasValidNotes) {
      summaryHTML += `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Additional Information</h3>
          <p class="text-sm text-gray-300">
            Not enough procedure data available for a detailed summary.
            Try adding more annotations and notes during the procedure for a more comprehensive summary.
          </p>
        </div>
      `;
    }

    // Close the main div
    summaryHTML += `</div>`;

    // Update the container with our generated HTML
    summaryContainer.innerHTML = summaryHTML;
  }, 1000);
}

// Helper function to determine procedure type from annotations
function determineTypeFromAnnotations(annotations) {
  // Default procedure type
  let procedureType = "Surgical Procedure";

  // Try to extract procedure type from annotations
  for (const annotation of annotations) {
    const content = annotation.textContent || '';
    if (content.toLowerCase().includes('laparoscopic')) {
      procedureType = "Laparoscopic Procedure";
      break;
    } else if (content.toLowerCase().includes('robotic')) {
      procedureType = "Robotic-Assisted Procedure";
      break;
    } else if (content.toLowerCase().includes('endoscopic')) {
      procedureType = "Endoscopic Procedure";
      break;
    } else if (content.toLowerCase().includes('open surgery')) {
      procedureType = "Open Surgery";
      break;
    }
  }

  return procedureType;
}

// Helper function to extract phases from annotations
function extractPhasesFromAnnotations(annotations) {
  const phases = new Set();

  for (const annotation of annotations) {
    const content = annotation.textContent || '';
    const phaseMatch = content.match(/Phase[:\s'"]+([^'"|\n]+)/i) || content.match(/phase[:\s'"]+([^'"|\n]+)/i);

    if (phaseMatch && phaseMatch[1]) {
      phases.add(phaseMatch[1].trim());
    }
  }

  return Array.from(phases);
}

// Helper function to extract key events from notes
function extractKeyEventsFromNotes(notes) {
  const keyEvents = [];

  for (const note of notes) {
    const title = note.querySelector('h3')?.textContent || '';
    const content = note.querySelector('.note-content')?.textContent || '';
    const timeElement = note.querySelector('.text-gray-400')?.textContent || '';
    const timeMatch = timeElement.match(/Video:\s*(\d+:\d+)/);
    const time = timeMatch ? timeMatch[1] : '';

    if (time && (title || content)) {
      let eventType = '';

      // Determine the type of event
      if (content.toLowerCase().includes('bleed') || title.toLowerCase().includes('bleed')) {
        eventType = 'Bleeding observed';
      } else if (content.toLowerCase().includes('tool') || title.toLowerCase().includes('tool') || 
                content.toLowerCase().includes('instrument') || title.toLowerCase().includes('instrument')) {
        eventType = 'Tool usage';
      } else if (content.toLowerCase().includes('incision') || title.toLowerCase().includes('incision') ||
                content.toLowerCase().includes('cut') || title.toLowerCase().includes('cut')) {
        eventType = 'Incision made';
      } else if (content.toLowerCase().includes('suture') || title.toLowerCase().includes('suture') ||
                content.toLowerCase().includes('stitch') || title.toLowerCase().includes('stitch')) {
        eventType = 'Suture placed';
      } else {
        eventType = 'Observation';
      }

      keyEvents.push(`${eventType} at ${time}`);
    }
  }

  return keyEvents;
}

// Helper function to extract key events from annotations
function extractKeyEventsFromAnnotations(annotations) {
  const keyEvents = [];

  for (const annotation of annotations) {
    const content = annotation.textContent || '';
    const timeMatch = content.match(/at time (\d+:\d+)/i);
    const time = timeMatch ? timeMatch[1] : '';

    // Look for key events in the annotation
    if (time || content.includes(':')) {
      let eventDesc = '';

      if (content.toLowerCase().includes('phase change')) {
        const phaseMatch = content.match(/Phase[:\s'"]+([^'"|\n]+)/i) || content.match(/phase[:\s'"]+([^'"|\n]+)/i);
        if (phaseMatch && phaseMatch[1]) {
          eventDesc = `Phase changed to '${phaseMatch[1].trim()}'`;
        } else {
          eventDesc = 'Phase changed';
        }
      } else if (content.toLowerCase().includes('tool')) {
        const toolMatch = content.match(/tool[s]?[:\s'"]+([^'"|\n]+)/i);
        if (toolMatch && toolMatch[1]) {
          eventDesc = `Tool used: ${toolMatch[1].trim()}`;
        } else {
          eventDesc = 'Tool usage noted';
        }
      } else if (content.toLowerCase().includes('anatomy')) {
        const anatomyMatch = content.match(/anatomy[:\s'"]+([^'"|\n]+)/i);
        if (anatomyMatch && anatomyMatch[1]) {
          eventDesc = `Anatomy identified: ${anatomyMatch[1].trim()}`;
        } else {
          eventDesc = 'Anatomy identified';
        }
      }

      if (eventDesc && time) {
        keyEvents.push(`${eventDesc} at ${time}`);
      }
    }
  }

  return keyEvents;
}

// Helper function to summarize notes by category
function summarizeNotesByCategory(notes) {
  const categories = {};

  // Group notes by category
  for (const note of notes) {
    // Get the category from the note
    const categoryElement = note.querySelector('.bg-primary-900\\/50') || 
                            note.querySelector('.text-xs.font-medium') ||
                            note.querySelector('[class*="text-"].font-medium');
    const category = categoryElement ? categoryElement.textContent.trim() : 'General';

    // Get the content - make sure it exists and isn't empty
    const contentElement = note.querySelector('.note-content');
    const content = contentElement?.textContent?.trim() || '';

    // Skip empty content
    if (!content) continue;

    // Initialize the category array if needed
    if (!categories[category]) {
      categories[category] = [];
    }

    // Add the content to the appropriate category
    categories[category].push(content);
  }

  // If we have no categories with content, return empty string
  if (Object.keys(categories).length === 0) {
    return '';
  }

  // Generate summary HTML
  let summary = '';
  for (const [category, contents] of Object.entries(categories)) {
    // Skip categories with no valid contents
    if (!contents.length) continue;

    // Filter out any undefined or empty contents
    const validContents = contents.filter(content => content && content.trim());

    // Skip if we have no valid contents after filtering
    if (!validContents.length) continue;

    summary += `
      <div class="mb-2">
        <h4 class="text-sm font-medium text-primary-300 mb-1">${category} (${validContents.length})</h4>
        <ul class="list-disc list-inside pl-2 text-xs text-gray-300 space-y-1">
          ${validContents.map(content => {
            // Truncate long content with ellipsis
            const displayContent = content.substring(0, 100) + (content.length > 100 ? '...' : '');
            return `<li>${displayContent}</li>`;
          }).join('')}
        </ul>
      </div>
    `;
  }

  return summary;
}

// Capture for note function (placeholder)
function captureForNote() {
  // In a real implementation, this would capture the current video frame

  // Show preview
  const previewContainer = document.getElementById('note-image-preview-container');
  const previewImage = document.getElementById('note-image-preview');

  // This would be replaced with the actual captured image
  previewImage.src = 'https://via.placeholder.com/640x360?text=Captured+Frame';
  previewContainer.classList.remove('hidden');

  showToast('Frame captured for note', 'success');
}

// Save manual note function
function saveManualNote() {
  const title = document.getElementById('note-title').value.trim();
  const content = document.getElementById('note-content').value.trim();
  const message = document.getElementById('note-message').value.trim();

  if (!title || !content) {
    showToast('Please enter a title and content for your note', 'error');
    return;
  }

  // Format the note for processing
  const noteText = `Note: ${title}. ${content}`;

  // Use our enhanced addNote function to create the note
  addNote(noteText, `Take a note about ${title}`);

  // Close modal
  const modal = document.getElementById('addNoteModal');
  closeModal(modal);

  // Reset the form
  resetNoteForm();

  showToast('Note saved successfully', 'success');

  // If there's a message, send it to the chat
  if (message) {
    addMessageToChat(message, 'user');
    // Send the message to the backend if needed
    sendMessageToBackend(message);
  }
}

// Function to fetch and display videos
function loadVideos() {
  const videoList = document.getElementById('video-list');
  const videoLoading = document.getElementById('video-loading');
  const noVideos = document.getElementById('no-videos');

  // Show loading state
  if (videoLoading) {
    videoLoading.style.display = 'block';
  }
  if (noVideos) {
    noVideos.style.display = 'none';
  }

  // Clear existing videos
  const existingVideos = videoList.querySelectorAll('.video-item');
  existingVideos.forEach(item => item.remove());

  // Fetch videos from the server
  fetch('/api/videos')
    .then(response => {
      if (!response.ok) {
        throw new Error('Failed to fetch videos');
      }
      return response.json();
    })
    .then(data => {
      if (videoLoading) {
        videoLoading.style.display = 'none';
      }

      if (data.videos && data.videos.length > 0) {
        // Display videos
        data.videos.forEach((video, index) => {
          const videoItem = document.createElement('div');
          videoItem.className = 'video-item p-0 bg-dark-800 rounded-xl border border-dark-600 hover:border-primary-500 transition-all duration-200 cursor-pointer overflow-hidden group';

          // Format file size
          const sizeInMB = (video.size / (1024 * 1024)).toFixed(2);

          // Format date
          const date = new Date(video.modified * 1000);
          const formattedDate = date.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
          });

          // Get a random duration for preview purposes (in real app this would come from the video metadata)
          const duration = `${Math.floor(Math.random() * 10) + 1}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`;

          // Create a unique ID for this video item
          const videoItemId = `video-item-${index}`;

          videoItem.innerHTML = `
            <div class="flex flex-col sm:flex-row sm:items-center">
              <div class="bg-dark-900 h-full min-h-24 sm:w-40 flex items-center justify-center p-3 relative">
                <div class="absolute inset-0 bg-gradient-to-r from-primary-900/30 to-primary-700/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <i class="fas fa-film text-primary-400 text-4xl relative z-10"></i>
                <span class="absolute bottom-2 right-2 text-xs text-white bg-dark-900/80 px-2 py-1 rounded">${duration}</span>
              </div>
              <div class="flex-grow p-4">
                <div class="flex items-start justify-between">
                  <h4 class="text-lg font-medium text-white group-hover:text-primary-300 transition-colors duration-200 truncate max-w-[80%]">${video.filename}</h4>
                  <div class="badge bg-dark-700 text-xs text-gray-300 ml-2">${sizeInMB} MB</div>
                </div>
                <div class="flex items-center mt-2 text-sm text-gray-400">
                  <i class="fas fa-calendar-alt mr-1 text-primary-500"></i>
                  <span>${formattedDate}</span>
                </div>
              </div>
              <div class="flex items-center justify-end p-3 pr-4 bg-dark-800 sm:bg-transparent">
                <button class="btn px-4 py-2 rounded-lg bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-500 hover:to-primary-600 text-white transform hover:scale-105 transition-all duration-200 shadow-md select-video-btn" data-filename="${video.filename}" data-item-id="${videoItemId}">
                  <i class="fas fa-play mr-2"></i> Play
                </button>
              </div>
            </div>
          `;

          videoItem.id = videoItemId;
          videoList.appendChild(videoItem);

          // Add click event to select button
          const selectBtn = videoItem.querySelector('.select-video-btn');
          if (selectBtn) {
            selectBtn.addEventListener('click', (e) => {
              e.preventDefault();
              // Add loading indicator to button
              selectBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Loading...';
              selectBtn.disabled = true;
              // Call select function
              selectVideo(video.filename);
            });
          }

          // Add click event to entire item (except the button)
          videoItem.addEventListener('click', (e) => {
            // If not clicking the button itself
            if (!e.target.closest('.select-video-btn')) {
              // Find and click the select button
              const btn = videoItem.querySelector('.select-video-btn');
              if (btn && !btn.disabled) {
                btn.click();
              }
            }
          });
        });
      } else {
        // Show no videos message
        if (noVideos) {
          noVideos.style.display = 'block';
        }
      }
    })
    .catch(error => {
      console.error('Error fetching videos:', error);

      if (videoLoading) {
        videoLoading.style.display = 'none';
      }

      if (noVideos) {
        noVideos.style.display = 'block';
        noVideos.innerHTML = `
          <div class="bg-dark-800 rounded-xl p-6 border border-red-900">
            <i class="fas fa-exclamation-circle text-5xl mb-4 text-red-500"></i>
            <p class="text-lg text-red-300">Error loading videos</p>
            <p class="text-sm text-gray-400 mt-2">${error.message}</p>
          </div>
        `;
      }
    });
}

// Function to select a video
function selectVideo(filename) {
  // Stop any playing TTS audio when switching videos (don't reset entire TTS state)
  stopCurrentTTS();

  // Close the modal
  const modal = document.getElementById('videoSelectModal');
  closeModal(modal);

  // Show loading state
  showToast('Loading video...', 'info');

  // Send request to server
  fetch('/api/select_video', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ filename })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Failed to select video');
    }
    return response.json();
  })
  .then(data => {
    // Update video source
    const videoElement = document.getElementById('surgery-video');
    if (videoElement && data.video_src) {
      // Stop any current playback
      try {
        videoElement.pause();
      } catch (e) {
        console.warn("Could not pause video:", e);
      }

      // Set new source and load (but don't play automatically)
      videoElement.src = data.video_src;

      // Clear any placeholder poster
      videoElement.removeAttribute('poster');

      videoElement.load();

      // Explicitly play the video (replaces autoplay)
      videoElement.play().catch(e => {
        console.warn("Could not start video playback:", e);
      });

      // Show success message
      showToast('Video loaded successfully!', 'success');

      // Enable the microphone button
      enableMicButton();

      // Reset the current phase 
      const phaseElement = document.getElementById('current-phase');
      if (phaseElement) {
        phaseElement.textContent = 'Undefined';
      }

      // Clear any existing annotations
      const annotationsContainer = document.getElementById('annotations-container');
      if (annotationsContainer) {
        annotationsContainer.innerHTML = `
          <div class="text-center text-gray-400 p-5">
            <i class="fas fa-tag fa-3x mb-3"></i>
            <p>No annotations available yet. Annotations will appear here as they are generated.</p>
          </div>
        `;
      }

      // Update the count
      const annotationCount = document.querySelector('.annotation-count');
      if (annotationCount) {
        annotationCount.textContent = '0';
      }
    }
  })
  .catch(error => {
    console.error('Error selecting video:', error);
    showToast('Failed to load video: ' + error.message, 'error');
  });
}

function closeModal(modal) {
  if (!modal) return;

  modal.classList.remove('show');
  modal.classList.add('closing');

  setTimeout(() => {
    modal.style.display = 'none'; // Reset display property
    modal.classList.remove('closing');
    document.body.classList.remove('overflow-hidden');
  }, 300);
}

// Video upload functionality
function uploadVideo() {
  const fileInput = document.getElementById('video-upload');
  const file = fileInput.files[0];
  const uploadForm = document.getElementById('video-upload-form');
  const uploadButton = uploadForm?.querySelector('button[type="button"]');

  if (!file) {
    showToast('Please select a video file first', 'error');
    return;
  }

  // Stop any playing TTS audio when uploading new video (don't reset entire TTS state)
  stopCurrentTTS();

  // Create FormData object
  const formData = new FormData();
  formData.append('video', file);

  // Show loading state
  if (uploadButton) {
    uploadButton.disabled = true;
    uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Uploading...';
  }
  showToast('Uploading video...', 'info');

  // Use fetch to upload the file
  fetch('/api/upload_video', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // Handle successful upload with the actual filename
    const displayName = data.filename || 'video';
    showToast(`"${displayName}" uploaded successfully!`, 'success');

    // Reset file name display
    const fileNameElement = fileInput.closest('.relative')?.querySelector('.file-name');
    if (fileNameElement) {
      fileNameElement.textContent = 'Select video file...';
    }

    // Reset upload button
    if (uploadButton) {
      uploadButton.disabled = false;
      uploadButton.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload';
    }

    // Clear the file input
    fileInput.value = '';

    // Update the button to show "Uploaded!" temporarily
    if (uploadButton) {
      uploadButton.classList.add('bg-green-600', 'border-green-700');
      uploadButton.innerHTML = '<i class="fas fa-check mr-2"></i> Uploaded!';
      setTimeout(() => {
        uploadButton.classList.remove('bg-green-600', 'border-green-700');
        uploadButton.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload';
      }, 2000);
    }

    // Load the video but don't autoplay
    if (data && data.video_src) {
      const videoElement = document.getElementById('surgery-video');
      if (videoElement) {
        // Pause any current video first
        try {
          videoElement.pause();
        } catch (e) {
          console.warn("Could not pause video:", e);
        }

        // Set new source and load
        videoElement.src = data.video_src;

        // Clear any placeholder poster
        videoElement.removeAttribute('poster');

        videoElement.load();

        // Explicitly play the video (replaces autoplay)
        videoElement.play().catch(e => {
          console.warn("Could not start video playback:", e);
        });

        // Enable the microphone button
        enableMicButton();

        // Reset the current phase display
        const phaseElement = document.getElementById('current-phase');
        if (phaseElement) {
          phaseElement.textContent = 'Undefined';
        }

        // Clear existing annotations
        const annotationsContainer = document.getElementById('annotations-container');
        if (annotationsContainer) {
          annotationsContainer.innerHTML = `
            <div class="text-center text-gray-400 p-5">
              <i class="fas fa-tag fa-3x mb-3"></i>
              <p>No annotations available yet. Annotations will appear here as they are generated.</p>
            </div>
          `;
        }

        // Reset annotation count
        const annotationCount = document.querySelector('.annotation-count');
        if (annotationCount) {
          annotationCount.textContent = '0';
        }

        // Clear existing notes
        const notesContainer = document.getElementById('notes-container');
        if (notesContainer) {
          notesContainer.innerHTML = `
            <div class="text-center text-gray-400 p-5">
              <i class="fas fa-sticky-note fa-3x mb-3"></i>
              <p>No notes available yet. You can add notes manually or ask the assistant to take notes for you.</p>
            </div>
          `;
        }

        // Reset notes count
        const notesCount = document.querySelector('.notes-count');
        if (notesCount) {
          notesCount.textContent = '0';
        }
      }
    }
  })
  .catch(error => {
    console.error('Error uploading video:', error);
    showToast('Failed to upload video. Please try again.', 'error');

    // Reset upload button
    if (uploadButton) {
      uploadButton.disabled = false;
      uploadButton.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload';
    }
  });
}

// ==============================
// TTS (Text-to-Speech) Functions
// ==============================

// Global TTS state (simplified POC pattern)
window.isTtsEnabled = false;
window.currentTtsAudio = null;  // Track currently playing TTS audio
window.ttsDebounceTimer = null; // Timer reference for cleanup (legacy)
window.ttsWebSocket = null;     // WebSocket connection for TTS
window.audioContext = null;     // Web Audio API context
window.audioQueue = [];         // Queue for audio chunks
window.isPlayingTTS = false;    // Flag to track TTS playback state
window.reconnectAttempts = 0;   // Track reconnection attempts
window.currentChunkIndex = 0;   // Track current chunk being processed
window.totalChunks = 0;         // Total number of chunks in current request
window.isProcessingChunks = false; // Flag to track if we're processing chunks
window.pendingChunks = new Map(); // Track pending chunks waiting for audio
window.chunkTimeouts = new Map(); // Track chunk timeouts
const MAX_RECONNECT_ATTEMPTS = 3;
const TTS_WS_URL = 'ws://localhost:8082/ws/tts'; // Direct connection to TTS service
const CHUNK_TIMEOUT_MS = 30000; // 30 seconds timeout per chunk

// Text chunking configuration
const CHUNK_CONFIG = {
  maxChunkLength: 150,  // Maximum characters per chunk
  sentenceEnders: ['.', '!', '?'],
  phraseBreaks: [',', ';', ':'],
  minChunkLength: 20    // Minimum characters per chunk
};

// Split text into chunks for TTS processing
function splitTextIntoChunks(text) {
  // Handle empty or invalid text
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    return [];
  }

  // If text is short enough, return as single chunk
  if (text.length <= CHUNK_CONFIG.maxChunkLength) {
    return [text.trim()];
  }

  const chunks = [];
  let currentChunk = '';
  let sentences = [];

  // First, split by sentence endings
  let currentSentence = '';
  for (let i = 0; i < text.length; i++) {
    currentSentence += text[i];

    if (CHUNK_CONFIG.sentenceEnders.includes(text[i])) {
      // Look ahead for potential quotes or closing punctuation
      let j = i + 1;
      while (j < text.length && /[\s"')\]}>]/.test(text[j])) {
        currentSentence += text[j];
        j++;
      }
      i = j - 1; // Adjust index

      const trimmedSentence = currentSentence.trim();
      if (trimmedSentence) {
        sentences.push(trimmedSentence);
      }
      currentSentence = '';
    }
  }

  // Add remaining text as final sentence
  if (currentSentence.trim()) {
    sentences.push(currentSentence.trim());
  }

  // Now group sentences into chunks
  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length <= CHUNK_CONFIG.maxChunkLength) {
      currentChunk += (currentChunk ? ' ' : '') + sentence;
    } else {
      // If current chunk has content, add it
      if (currentChunk) {
        chunks.push(currentChunk);
      }

      // If sentence is too long, split it further
      if (sentence.length > CHUNK_CONFIG.maxChunkLength) {
        const subChunks = splitLongSentence(sentence);
        chunks.push(...subChunks);
        currentChunk = '';
      } else {
        currentChunk = sentence;
      }
    }
  }

  // Add remaining chunk
  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks.filter(chunk => chunk.trim().length > 0);
}

// Split a long sentence into smaller chunks
function splitLongSentence(sentence) {
  const chunks = [];
  let currentChunk = '';
  const words = sentence.split(' ');

  for (const word of words) {
    // Handle extremely long words that exceed chunk size
    if (word.length > CHUNK_CONFIG.maxChunkLength) {
      console.warn(`Word "${word.substring(0, 20)}..." exceeds chunk length, truncating`);
      const truncatedWord = word.substring(0, CHUNK_CONFIG.maxChunkLength - 3) + '...';

      if (currentChunk) {
        chunks.push(currentChunk);
        currentChunk = '';
      }
      chunks.push(truncatedWord);
      continue;
    }

    if (currentChunk.length + word.length + 1 <= CHUNK_CONFIG.maxChunkLength) {
      currentChunk += (currentChunk ? ' ' : '') + word;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk);
      }
      currentChunk = word;
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks.filter(chunk => chunk.trim().length > 0);
}

// Initialize TTS based on selected service
function initializeTTS(event) {
  const ttsToggle = document.getElementById('ttsEnable');
  const ttsService = document.getElementById('ttsService');

  if (!ttsToggle || !ttsService) {
    console.warn('TTS elements not found');
    return;
  }

  const isEnabled = ttsToggle.checked;
  const serviceType = ttsService.value;

  // Only reset if TTS is being disabled
  if (!isEnabled) {
    resetTTSConnection();
    window.isTtsEnabled = isEnabled;
    return;
  }

  window.isTtsEnabled = isEnabled;

  // Update service options UI
  updateTtsServiceOptions();

  // Only connect to WebSocket for local TTS service
  if (serviceType === 'local') {
    setTimeout(() => {
      connectTTSWebSocket();
    }, 100);
  } else {
    updateTTSConnectionStatus(true, 'REST API Ready');
  }
}

// Initialize Web Audio API context
function initializeAudioContext() {
  try {
    window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  } catch (error) {
    console.error('Error initializing Web Audio API:', error);
    // Fallback to regular Audio API will be used
  }
}

// Update TTS WebSocket connection status in UI
function updateTTSConnectionStatus(isConnected, message = '') {
  const statusDot = document.getElementById('ttsStatusDot');
  const statusText = document.getElementById('ttsStatusText');

  if (statusDot && statusText) {
    if (isConnected) {
      statusDot.className = 'w-2 h-2 rounded-full bg-green-500 mr-2';
      statusText.textContent = 'WebSocket Connected';
      statusText.className = 'text-green-400';
    } else {
      statusDot.className = 'w-2 h-2 rounded-full bg-red-500 mr-2';
      statusText.textContent = message || 'WebSocket Disconnected';
      statusText.className = 'text-red-400';
    }
  }
}

// Connect to TTS WebSocket
// Connect to TTS WebSocket (simplified POC pattern)
function connectTTSWebSocket() {
  // Check if already connected
  if (window.ttsWebSocket && window.ttsWebSocket.readyState === WebSocket.OPEN) {
    return;
  }

  updateTTSConnectionStatus(false, 'Connecting...');

  try {
    window.ttsWebSocket = new WebSocket(TTS_WS_URL);

    window.ttsWebSocket.onopen = () => {
      console.log('✅ TTS WebSocket connected');
      window.reconnectAttempts = 0;
      updateTTSConnectionStatus(true);
      showToast('TTS WebSocket connected', 'success');
    };

    window.ttsWebSocket.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        // Binary audio data received
        const arrayBuffer = await event.data.arrayBuffer();

        // If we're processing chunks, add to queue; otherwise play directly
        if (window.isProcessingChunks) {
          await addToAudioQueue(arrayBuffer);

          // Check if we have a pending chunk that's waiting for this audio
          // Use the current chunk index to find the pending chunk
          const currentChunkKey = `chunk_${window.currentChunkIndex}`;
          if (window.pendingChunks.has(currentChunkKey)) {
            const chunkData = window.pendingChunks.get(currentChunkKey);
            window.pendingChunks.delete(currentChunkKey);

            // Clear timeout for this chunk
            if (window.chunkTimeouts.has(currentChunkKey)) {
              clearTimeout(window.chunkTimeouts.get(currentChunkKey));
              window.chunkTimeouts.delete(currentChunkKey);
            }

            // Move to next chunk
            window.currentChunkIndex++;

            // Process next chunk
            processNextChunk(chunkData.chunks);
          }
        } else {
          await playAudioData(arrayBuffer);
        }
      } else {
        // JSON status message
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'error') {
            console.error('❌ TTS WebSocket error:', data.message);
            showToast('TTS Error: ' + data.message, 'error');
          }
        } catch (e) {
          // Ignore non-JSON messages
        }
      }
    };

    window.ttsWebSocket.onerror = (error) => {
      console.error('❌ TTS WebSocket error:', error);
      updateTTSConnectionStatus(false, 'Connection Error');
      showToast('TTS WebSocket connection error', 'error');
    };

    window.ttsWebSocket.onclose = (event) => {
      console.log(`🔌 TTS WebSocket closed: ${event.code} - ${event.reason}`);
      updateTTSConnectionStatus(false, 'Disconnected');

      // Only auto-reconnect if TTS is still enabled, we haven't exceeded max attempts,
      // and the closure wasn't intentional (code 1000 = normal closure)
      if (window.isTtsEnabled && window.reconnectAttempts < MAX_RECONNECT_ATTEMPTS && event.code !== 1000) {
        window.reconnectAttempts++;
        console.log(`Reconnecting TTS WebSocket... (attempt ${window.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);
        setTimeout(() => {
          if (window.isTtsEnabled) { // Double-check TTS is still enabled
            connectTTSWebSocket();
          }
        }, 2000);
      } else if (window.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.error('TTS WebSocket max reconnection attempts exceeded');
        showToast('TTS service unavailable - please check if the service is running', 'error');
      }
    };

  } catch (error) {
    console.error('❌ Error creating TTS WebSocket:', error);
    updateTTSConnectionStatus(false, 'Failed to Connect');
    showToast('Failed to create TTS WebSocket connection', 'error');
  }
}

// Disconnect TTS WebSocket
function disconnectTTSWebSocket() {
  if (window.ttsWebSocket) {
    window.ttsWebSocket.close();
    window.ttsWebSocket = null;
    updateTTSConnectionStatus(false, 'Disconnected');
    console.log('TTS WebSocket disconnected');
  }
}

// Play audio data using Web Audio API or fallback to regular Audio
async function playAudioData(arrayBuffer) {
  try {
    if (window.audioContext && window.audioContext.state !== 'closed') {
      // Use Web Audio API for better performance
      const audioBuffer = await window.audioContext.decodeAudioData(arrayBuffer.slice(0));
      const source = window.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(window.audioContext.destination);

      // Track the audio source for cleanup
      window.currentTtsAudio = source;
      window.isPlayingTTS = true;

      source.onended = () => {
        window.currentTtsAudio = null;
        window.isPlayingTTS = false;
      };

      // Resume audio context if suspended (Chrome autoplay policy)
      if (window.audioContext.state === 'suspended') {
        await window.audioContext.resume();
      }

      source.start();

    } else {
      // Fallback to regular Audio API
      const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      window.currentTtsAudio = audio;
      window.isPlayingTTS = true;

      audio.addEventListener('ended', () => {
        URL.revokeObjectURL(audioUrl);
        window.currentTtsAudio = null;
        window.isPlayingTTS = false;
      });

      audio.addEventListener('error', (error) => {
        console.error('Error playing TTS audio:', error);
        URL.revokeObjectURL(audioUrl);
        window.currentTtsAudio = null;
        window.isPlayingTTS = false;
      });

      await audio.play();
    }

  } catch (error) {
    console.error('Error playing TTS audio:', error);
    window.currentTtsAudio = null;
    window.isPlayingTTS = false;
  }
}

// Audio queue management for chunked TTS
async function addToAudioQueue(arrayBuffer) {
  // Manage memory before adding new item
  manageAudioQueueMemory();

  const audioItem = {
    arrayBuffer: arrayBuffer,
    timestamp: Date.now()
  };

  window.audioQueue.push(audioItem);

  // Start playing if not already playing
  if (!window.isPlayingTTS) {
    await playNextFromQueue();
  }
}

async function playNextFromQueue() {
  if (window.audioQueue.length === 0) {
    window.isPlayingTTS = false;
    return;
  }

  const audioItem = window.audioQueue.shift();
  window.isPlayingTTS = true;

  try {
    if (window.audioContext && window.audioContext.state !== 'closed') {
      // Use Web Audio API for better performance
      const audioBuffer = await window.audioContext.decodeAudioData(audioItem.arrayBuffer.slice(0));
      const source = window.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(window.audioContext.destination);

      // Track the audio source for cleanup
      window.currentTtsAudio = source;

      source.onended = async () => {
        window.currentTtsAudio = null;
        // Play next item in queue
        await playNextFromQueue();
      };

      // Resume audio context if suspended (Chrome autoplay policy)
      if (window.audioContext.state === 'suspended') {
        await window.audioContext.resume();
      }

      source.start();

    } else {
      // Fallback to regular Audio API
      const blob = new Blob([audioItem.arrayBuffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      window.currentTtsAudio = audio;

      audio.addEventListener('ended', async () => {
        URL.revokeObjectURL(audioUrl);
        window.currentTtsAudio = null;
        // Play next item in queue
        await playNextFromQueue();
      });

      audio.addEventListener('error', async (error) => {
        console.error('Error playing TTS audio:', error);
        URL.revokeObjectURL(audioUrl);
        window.currentTtsAudio = null;
        // Continue with next item even if this one failed
        await playNextFromQueue();
      });

      await audio.play();
    }

  } catch (error) {
    console.error('Error playing queued TTS audio:', error);
    window.currentTtsAudio = null;
    // Continue with next item even if this one failed
    await playNextFromQueue();
  }
}

// Clear audio queue
function clearAudioQueue() {
  // Clean up any audio URLs to prevent memory leaks
  window.audioQueue.forEach(item => {
    if (item.audioUrl) {
      URL.revokeObjectURL(item.audioUrl);
    }
  });

  window.audioQueue = [];
  window.isPlayingTTS = false;
  window.currentTtsAudio = null;
}

// Add memory management for audio queue
function manageAudioQueueMemory() {
  const maxQueueSize = 10; // Maximum number of audio chunks to keep in memory
  const maxAge = 60000; // Maximum age of audio chunks (1 minute)
  const now = Date.now();

  // Remove old items
  window.audioQueue = window.audioQueue.filter(item => {
    if (now - item.timestamp > maxAge) {
      if (item.audioUrl) {
        URL.revokeObjectURL(item.audioUrl);
      }
      return false;
    }
    return true;
  });

  // Remove excess items (keep only the latest ones)
  while (window.audioQueue.length > maxQueueSize) {
    const oldItem = window.audioQueue.shift();
    if (oldItem.audioUrl) {
      URL.revokeObjectURL(oldItem.audioUrl);
    }
  }
}

// Update TTS service options UI
function updateTtsServiceOptions() {
  const ttsService = document.getElementById('ttsService');
  const localOptions = document.getElementById('localTtsOptions');
  const elevenlabsOptions = document.getElementById('elevenlabsOptions');
  const ttsDescription = document.getElementById('ttsDescription');

  if (ttsService && localOptions && elevenlabsOptions && ttsDescription) {
    if (ttsService.value === 'local') {
      localOptions.style.display = 'block';
      elevenlabsOptions.style.display = 'none';
      ttsDescription.textContent = 'Voice output using local TTS via WebSocket connection for real-time audio streaming';
    } else {
      localOptions.style.display = 'none';
      elevenlabsOptions.style.display = 'block';
      ttsDescription.textContent = 'Voice output using ElevenLabs Text-to-Speech API via REST API';
    }
  }
}

// Stop current TTS audio if playing
function stopCurrentTTS() {
  // Stop chunk processing
  window.isProcessingChunks = false;

  // Clear pending chunks and timeouts
  window.pendingChunks.clear();
  window.chunkTimeouts.forEach(timeoutId => clearTimeout(timeoutId));
  window.chunkTimeouts.clear();

  // Clear audio queue
  clearAudioQueue();

  // Stop currently playing audio
  if (window.currentTtsAudio) {
    try {
      if (window.currentTtsAudio.stop) {
        // Web Audio API source
        window.currentTtsAudio.stop();
      } else {
        // Regular Audio element
        window.currentTtsAudio.pause();
        window.currentTtsAudio.currentTime = 0;
      }
      window.currentTtsAudio = null;
      window.isPlayingTTS = false;
    } catch (error) {
      console.warn('Error stopping TTS audio:', error);
    }
  }
}

// Reset all TTS state and stop any audio
function resetTTSState() {
  // Stop current audio
  stopCurrentTTS();

  // Clear debounce timer
  if (window.ttsDebounceTimer) {
    clearTimeout(window.ttsDebounceTimer);
    window.ttsDebounceTimer = null;
  }

  // Reset chunked processing state
  window.currentChunkIndex = 0;
  window.totalChunks = 0;
  window.isProcessingChunks = false;

  // Clear pending chunks and timeouts
  window.pendingChunks.clear();
  window.chunkTimeouts.forEach(timeoutId => clearTimeout(timeoutId));
  window.chunkTimeouts.clear();

  // Clear audio queue
  clearAudioQueue();
}

// Generate speech from text with chunked processing for better latency
function generateSpeech(text) {
  if (!text || !window.isTtsEnabled) return;

  const ttsService = document.getElementById('ttsService');
  const ttsServiceValue = ttsService ? ttsService.value : 'local';

  // Check connection for local TTS service
  if (ttsServiceValue === 'local') {
    if (!window.ttsWebSocket || window.ttsWebSocket.readyState !== WebSocket.OPEN) {
      console.warn('TTS WebSocket not connected. Please check connection status.');
      showToast('TTS not connected. Please enable TTS to connect.', 'warning');
      return;
    }
  }

  // Clear any existing debounce timer (for cleanup)
  if (window.ttsDebounceTimer) {
    clearTimeout(window.ttsDebounceTimer);
    window.ttsDebounceTimer = null;
  }

  // Stop any currently playing TTS audio and clear queue
  stopCurrentTTS();
  clearAudioQueue();

  // Split text into chunks for better latency
  const chunks = splitTextIntoChunks(text);

  if (chunks.length === 0) return;

  // Initialize chunk processing state
  window.currentChunkIndex = 0;
  window.totalChunks = chunks.length;
  window.isProcessingChunks = true;

  console.log(`Starting chunked TTS processing: ${chunks.length} chunks`);

  // Process chunks sequentially
  processNextChunk(chunks);
}

// Process the next chunk in the sequence
function processNextChunk(chunks) {
  // Safety check: ensure we have valid chunks array
  if (!chunks || !Array.isArray(chunks)) {
    console.error('Invalid chunks array provided to processNextChunk');
    window.isProcessingChunks = false;
    return;
  }

  if (!window.isProcessingChunks || window.currentChunkIndex >= chunks.length) {
    // All chunks processed
    window.isProcessingChunks = false;
    console.log('All TTS chunks processed');
    return;
  }

  const chunk = chunks[window.currentChunkIndex];

  // Safety check: ensure chunk is valid
  if (!chunk || typeof chunk !== 'string' || chunk.trim().length === 0) {
    console.warn(`Skipping invalid chunk at index ${window.currentChunkIndex}`);
    window.currentChunkIndex++;
    processNextChunk(chunks);
    return;
  }

  console.log(`Processing chunk ${window.currentChunkIndex + 1}/${window.totalChunks}: "${chunk.substring(0, 50)}..."`);

  const ttsService = document.getElementById('ttsService');
  const ttsServiceValue = ttsService ? ttsService.value : 'local';

  try {
    if (ttsServiceValue === 'local') {
      performChunkedWebSocketTTSRequest(chunk, chunks);
    } else {
      performChunkedRestTTSRequest(chunk, chunks);
    }
  } catch (error) {
    console.error('Error processing chunk:', error);

    // Continue with next chunk even if this one failed
    window.currentChunkIndex++;
    processNextChunk(chunks);
  }
}

// Perform chunked WebSocket TTS request
function performChunkedWebSocketTTSRequest(chunk, chunks) {
  if (!window.ttsWebSocket || window.ttsWebSocket.readyState !== WebSocket.OPEN) {
    console.error('TTS WebSocket not connected');
    showToast('TTS service not connected', 'error');
    window.isProcessingChunks = false;
    return;
  }

  // Get TTS model
  const ttsModel = document.getElementById('ttsModel');
  const modelName = ttsModel ? ttsModel.value : 'tts_models/en/ljspeech/vits';

  // Format message to match TTS service WebSocket API
  const request = {
    text: chunk,
    model: modelName,
    chunk_index: window.currentChunkIndex,
    total_chunks: window.totalChunks
  };

  try {
    // Mark this chunk as pending
    const chunkKey = `chunk_${window.currentChunkIndex}`;
    window.pendingChunks.set(chunkKey, { chunks: chunks, timestamp: Date.now() });

    // Set timeout for this chunk
    const timeoutId = setTimeout(() => {
      console.error(`Chunk ${window.currentChunkIndex} timed out`);

      // Remove from pending chunks
      window.pendingChunks.delete(chunkKey);
      window.chunkTimeouts.delete(chunkKey);

      // Move to next chunk even if this one failed
      window.currentChunkIndex++;
      processNextChunk(chunks);
    }, CHUNK_TIMEOUT_MS);

    window.chunkTimeouts.set(chunkKey, timeoutId);

    // Send the request
    window.ttsWebSocket.send(JSON.stringify(request));

    // Note: We don't immediately process the next chunk here.
    // The next chunk will be processed when the audio for this chunk is received
    // in the WebSocket onmessage handler.

  } catch (error) {
    console.error('Error sending chunked TTS WebSocket request:', error);
    showToast('Failed to send TTS request', 'error');
    window.isProcessingChunks = false;

    // Clean up pending chunk
    const chunkKey = `chunk_${window.currentChunkIndex}`;
    window.pendingChunks.delete(chunkKey);
    if (window.chunkTimeouts.has(chunkKey)) {
      clearTimeout(window.chunkTimeouts.get(chunkKey));
      window.chunkTimeouts.delete(chunkKey);
    }
  }
}

// Perform chunked REST API TTS request
function performChunkedRestTTSRequest(chunk, chunks) {
  const ttsService = document.getElementById('ttsService');
  const ttsServiceValue = ttsService ? ttsService.value : 'local';

  let requestData = {
    text: chunk,
    tts_service: ttsServiceValue,
    chunk_index: window.currentChunkIndex,
    total_chunks: window.totalChunks
  };

  if (ttsServiceValue === 'elevenlabs') {
    const apiKeyInput = document.getElementById('ttsApiKey');
    requestData.api_key = apiKeyInput ? apiKeyInput.value : '';
  }

  // Request TTS from server
  fetch('/api/tts', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestData)
  })
  .then(response => response.json())
  .then(async (data) => {
    if (data.tts_base64) {
      // Convert base64 to arrayBuffer and add to queue
      const audioFormat = ttsServiceValue === 'elevenlabs' ? 'audio/mp3' : 'audio/wav';
      const binaryString = atob(data.tts_base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      await addToAudioQueue(bytes.buffer);

      // Move to next chunk
      window.currentChunkIndex++;

      // Process next chunk
      processNextChunk(chunks);
    } else {
      console.error('No audio data received from TTS service for chunk', window.currentChunkIndex);
      // Continue with next chunk even if this one failed
      window.currentChunkIndex++;
      processNextChunk(chunks);
    }
  })
  .catch(error => {
    console.error('Error with chunked TTS request:', error);
    // Continue with next chunk even if this one failed
    window.currentChunkIndex++;
    processNextChunk(chunks);
  });
}

// Initialize TTS system when page loads
document.addEventListener('DOMContentLoaded', function() {
  // Initialize Web Audio API context
  initializeAudioContext();

  // Setup TTS event listeners (remove any existing ones first)
  const ttsToggle = document.getElementById('ttsEnable');
  const ttsService = document.getElementById('ttsService');

  if (ttsToggle) {
    // Remove any existing listeners to prevent duplicates
    ttsToggle.removeEventListener('change', initializeTTS);
    ttsToggle.addEventListener('change', initializeTTS);
  }

  if (ttsService) {
    // Remove any existing listeners to prevent duplicates
    ttsService.removeEventListener('change', initializeTTS);
    ttsService.addEventListener('change', initializeTTS);
  }
});

// Stop TTS and cleanup when page is being unloaded
window.addEventListener('beforeunload', function() {
  resetTTSState();
  disconnectTTSWebSocket();
  if (window.audioContext && window.audioContext.state !== 'closed') {
    window.audioContext.close();
  }
});

// Reset TTS connection state (useful for debugging and manual reset)
function resetTTSConnection() {
  // Close existing connection
  if (window.ttsWebSocket) {
    try {
      window.ttsWebSocket.close(1000, 'Manual reset'); // Normal closure
    } catch (e) {
      console.warn('⚠️ Error closing WebSocket:', e);
    }
    window.ttsWebSocket = null;
  }

  // Reset connection attempts
  window.reconnectAttempts = 0;

  // Reset TTS state
  resetTTSState();

  // Update UI
  updateTTSConnectionStatus(false, 'Disconnected');
}

// Test TTS WebSocket connection (for debugging - call from browser console)
function testTTSConnection() {
  resetTTSConnection();

  // Enable TTS temporarily for testing
  const originalTtsEnabled = window.isTtsEnabled;
  window.isTtsEnabled = true;

  // Attempt connection
  connectTTSWebSocket();

  // Test message after a delay
  setTimeout(() => {
    if (window.ttsWebSocket && window.ttsWebSocket.readyState === WebSocket.OPEN) {
      generateSpeech('This is a test of the WebSocket TTS system.');
    }

    // Restore original TTS state
    window.isTtsEnabled = originalTtsEnabled;
  }, 2000);
}

// Helper function to get readable WebSocket state
function getWebSocketStateText(state) {
  switch(state) {
    case WebSocket.CONNECTING: return 'CONNECTING (0)';
    case WebSocket.OPEN: return 'OPEN (1)';
    case WebSocket.CLOSING: return 'CLOSING (2)';
    case WebSocket.CLOSED: return 'CLOSED (3)';
    default: return `UNKNOWN (${state})`;
  }
}

// ============================================================================
// CAPTURED FRAME DISPLAY FUNCTIONS
// ============================================================================

// Update the captured frame display in the UI
function updateCapturedFrameDisplay(frameData, status = 'Captured', source = 'manual') {
  const frameImage = document.getElementById('captured-frame-image');
  const framePlaceholder = document.getElementById('captured-frame-placeholder');
  const frameStatus = document.getElementById('frame-status');
  const frameTimestamp = document.getElementById('frame-timestamp');

  if (!frameImage || !framePlaceholder || !frameStatus || !frameTimestamp) {
    const missingElements = [];
    if (!frameImage) missingElements.push('captured-frame-image');
    if (!framePlaceholder) missingElements.push('captured-frame-placeholder');
    if (!frameStatus) missingElements.push('frame-status');
    if (!frameTimestamp) missingElements.push('frame-timestamp');

    console.warn(`Captured frame display elements not found: ${missingElements.join(', ')}`);
    return;
  }

  if (frameData) {
    // Show the captured frame
    frameImage.src = frameData;
    frameImage.classList.remove('hidden');
    framePlaceholder.classList.add('hidden');

    // Update status and timestamp
    frameStatus.textContent = status;
    frameStatus.className = 'text-xs text-green-400';

    const now = new Date();
    const timeString = now.toLocaleTimeString();
    frameTimestamp.textContent = `${timeString} (${source})`;

    console.log(`Frame display updated: ${status} from ${source}`);
  } else {
    // Show placeholder
    frameImage.classList.add('hidden');
    framePlaceholder.classList.remove('hidden');

    frameStatus.textContent = 'No frame available';
    frameStatus.className = 'text-xs text-gray-400';
    frameTimestamp.textContent = '';
  }
}

// Show the sending overlay when frame is being sent to vLLM
function showFrameSendingStatus() {
  const overlay = document.getElementById('frame-sending-overlay');
  const frameStatus = document.getElementById('frame-status');

  if (overlay) {
    overlay.classList.remove('hidden');
    overlay.classList.add('flex');
  }

  if (frameStatus) {
    frameStatus.textContent = 'Sending for AI Analysis...';
    frameStatus.className = 'text-xs text-primary-400';
  }

  // Hide the overlay after a short time
  setTimeout(() => {
    if (overlay) {
      overlay.classList.add('hidden');
      overlay.classList.remove('flex');
    }
    if (frameStatus) {
      frameStatus.textContent = 'Sent for AI Analysis';
      frameStatus.className = 'text-xs text-green-400';
    }
  }, OVERLAY_HIDE_DELAY_MS);
}

// Update frame status without changing the image
function updateFrameStatus(status, className = 'text-xs text-gray-400') {
  const frameStatus = document.getElementById('frame-status');
  if (frameStatus) {
    frameStatus.textContent = status;
    frameStatus.className = className;
  }
}
