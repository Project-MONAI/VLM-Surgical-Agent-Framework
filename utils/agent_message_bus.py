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
Agent Message Bus - Central communication hub for agent-to-agent messaging.

Features:
- Request/Response pattern with timeouts
- Publish/Subscribe for events
- Human-in-the-loop approval flows
- Workflow state management
- Message persistence and replay
"""

import logging
import queue
import threading
import uuid
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Types of messages that can be sent between agents"""
    REQUEST = "request"           # Request with expected response
    RESPONSE = "response"         # Response to a request
    EVENT = "event"               # Fire-and-forget notification
    APPROVAL_REQUEST = "approval_request"  # Request user approval
    APPROVAL_RESPONSE = "approval_response"  # User approval result


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    message_id: str
    message_type: MessageType
    sender_agent: str
    target_agent: Optional[str]  # None for broadcast events
    action: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For linking request/response
    priority: MessagePriority = MessagePriority.NORMAL
    requires_user_approval: bool = False
    timeout_seconds: Optional[int] = None
    success: bool = True  # For response messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_agent": self.sender_agent,
            "target_agent": self.target_agent,
            "action": self.action,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "requires_user_approval": self.requires_user_approval,
            "timeout_seconds": self.timeout_seconds,
            "success": self.success,
        }


class AgentMessageBus:
    """
    Central message bus for agent-to-agent communication.

    Features:
    - Request/Response pattern with timeouts
    - Publish/Subscribe for events
    - Human-in-the-loop approval flows
    - Workflow state management
    - Message persistence and replay
    """

    def __init__(self, web_server=None, message_history_size: int = 1000):
        self._logger = logging.getLogger(__name__)
        self.web_server = web_server

        # Message routing
        self._message_queues: Dict[str, queue.Queue] = {}  # agent_name -> queue
        self._subscribers: Dict[str, List[Callable]] = {}  # event_type -> callbacks

        # Request/Response tracking
        self._pending_requests: Dict[str, threading.Event] = {}  # message_id -> event
        self._responses: Dict[str, AgentMessage] = {}  # message_id -> response

        # User approval tracking
        self._pending_approvals: Dict[str, threading.Event] = {}  # message_id -> event
        self._approval_responses: Dict[str, bool] = {}  # message_id -> approved

        # Message history for debugging and replay
        self._message_history: List[AgentMessage] = []
        self._message_history_size = message_history_size

        # Workflow state
        self._active_workflows: Dict[str, Dict[str, Any]] = {}  # workflow_id -> state

        self._lock = threading.Lock()
        self._logger.info("Agent Message Bus initialized")

    # ========================================================================
    # AGENT REGISTRATION
    # ========================================================================

    def register_agent(self, agent_name: str) -> queue.Queue:
        """
        Register an agent to receive messages.
        Returns a queue that the agent should poll for messages.
        """
        with self._lock:
            if agent_name not in self._message_queues:
                self._message_queues[agent_name] = queue.Queue()
                self._logger.info(f"Registered agent: {agent_name}")
            return self._message_queues[agent_name]

    def unregister_agent(self, agent_name: str):
        """Unregister an agent"""
        with self._lock:
            if agent_name in self._message_queues:
                del self._message_queues[agent_name]
                self._logger.info(f"Unregistered agent: {agent_name}")

    # ========================================================================
    # REQUEST/RESPONSE PATTERN
    # ========================================================================

    def send_request(
        self,
        sender_agent: str,
        target_agent: str,
        action: str,
        payload: Dict[str, Any],
        timeout_seconds: int = 30,
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_user_approval: bool = False
    ) -> Optional[AgentMessage]:
        """
        Send a request to another agent and wait for response.

        Args:
            sender_agent: Name of the requesting agent
            target_agent: Name of the target agent
            action: Action to perform (e.g., "fetch_tool", "check_status")
            payload: Data for the request
            timeout_seconds: How long to wait for response
            priority: Message priority
            requires_user_approval: If True, request user approval first

        Returns:
            Response message or None if timeout
        """
        message_id = str(uuid.uuid4())

        message = AgentMessage(
            message_id=message_id,
            message_type=MessageType.REQUEST,
            sender_agent=sender_agent,
            target_agent=target_agent,
            action=action,
            payload=payload,
            priority=priority,
            requires_user_approval=requires_user_approval,
            timeout_seconds=timeout_seconds,
        )

        # Set up response tracking
        response_event = threading.Event()
        with self._lock:
            self._pending_requests[message_id] = response_event

        # Handle user approval if required
        if requires_user_approval:
            self._logger.info(f"Request {message_id} requires user approval")
            approved = self._request_user_approval(message)
            if not approved:
                self._logger.info(f"Request {message_id} denied by user")
                with self._lock:
                    del self._pending_requests[message_id]
                return None

        # Send message
        self._route_message(message)

        # Wait for response
        self._logger.debug(f"Waiting for response to {message_id} (timeout={timeout_seconds}s)")
        response_received = response_event.wait(timeout=timeout_seconds)

        with self._lock:
            if response_received and message_id in self._responses:
                response = self._responses.pop(message_id)
                del self._pending_requests[message_id]
                self._logger.info(f"Received response to {message_id}")
                return response
            else:
                # Timeout
                if message_id in self._pending_requests:
                    del self._pending_requests[message_id]
                self._logger.warning(f"Timeout waiting for response to {message_id}")
                return None

    def send_response(
        self,
        responding_agent: str,
        original_request: AgentMessage,
        payload: Dict[str, Any],
        success: bool = True
    ):
        """
        Send a response to a previous request.

        Args:
            responding_agent: Name of the agent responding
            original_request: The original request message
            payload: Response data
            success: Whether the request was successful
        """
        response = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_agent=responding_agent,
            target_agent=original_request.sender_agent,
            action=f"{original_request.action}_response",
            payload={**payload, "success": success},
            correlation_id=original_request.message_id,
            success=success,
        )

        # Store response and signal waiting thread
        with self._lock:
            if original_request.message_id in self._pending_requests:
                self._responses[original_request.message_id] = response
                self._pending_requests[original_request.message_id].set()

        # Also route to sender's queue for async handling
        self._route_message(response)

        self._logger.info(
            f"Sent response from {responding_agent} to {original_request.sender_agent}"
        )

    # ========================================================================
    # PUBLISH/SUBSCRIBE PATTERN (EVENTS)
    # ========================================================================

    def publish_event(
        self,
        sender_agent: str,
        event_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ):
        """
        Publish an event that any agent can subscribe to.
        Fire-and-forget, no response expected.

        Args:
            sender_agent: Name of the publishing agent
            event_type: Type of event (e.g., "tool_retrieved", "phase_changed")
            payload: Event data
            priority: Message priority
        """
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVENT,
            sender_agent=sender_agent,
            target_agent=None,  # Broadcast
            action=event_type,
            payload=payload,
            priority=priority,
        )

        self._route_message(message)

        # Notify subscribers
        with self._lock:
            if event_type in self._subscribers:
                subscribers = list(self._subscribers[event_type])

        # Call subscribers outside lock to avoid deadlock
        if event_type in self._subscribers:
            for callback in subscribers:
                try:
                    # Call in separate thread to avoid blocking
                    threading.Thread(
                        target=callback,
                        args=(message,),
                        daemon=True
                    ).start()
                except Exception as e:
                    self._logger.error(f"Error in event subscriber: {e}", exc_info=True)

        self._logger.debug(f"Published event: {event_type} from {sender_agent}")

    def subscribe_to_event(self, event_type: str, callback: Callable[[AgentMessage], None]):
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

        self._logger.info(f"Subscribed to event: {event_type}")

    def unsubscribe_from_event(self, event_type: str, callback: Callable[[AgentMessage], None]):
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to stop listening for
            callback: The callback function to remove
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]
                    self._logger.info(f"Unsubscribed from event: {event_type}")
                except ValueError:
                    self._logger.warning(f"Callback not found for event: {event_type}")

    # ========================================================================
    # USER APPROVAL FLOW
    # ========================================================================

    def _request_user_approval(self, message: AgentMessage) -> bool:
        """
        Request user approval via UI and wait for response.

        Args:
            message: The message requiring approval

        Returns:
            True if approved, False if denied
        """
        approval_id = message.message_id
        approval_event = threading.Event()

        with self._lock:
            self._pending_approvals[approval_id] = approval_event

        # Send approval request to UI via WebSocket
        if self.web_server:
            approval_request = {
                "type": "approval_request",
                "approval_id": approval_id,
                "sender_agent": message.sender_agent,
                "target_agent": message.target_agent,
                "action": message.action,
                "description": message.payload.get("description", ""),
                "message": message.payload.get("approval_message", 
                    f"Agent {message.sender_agent} requests permission to execute: {message.action}"),
            }
            self.web_server.send_message(approval_request)
            self._logger.info(f"Sent approval request {approval_id} to UI")
        else:
            self._logger.error("Cannot request approval: web_server not configured")
            return False

        # Wait for user response (default: 60 seconds)
        timeout = message.timeout_seconds if message.timeout_seconds else 60
        approved = approval_event.wait(timeout=timeout)

        with self._lock:
            if approved and approval_id in self._approval_responses:
                result = self._approval_responses.pop(approval_id)
                del self._pending_approvals[approval_id]
                return result
            else:
                # Timeout = denial
                if approval_id in self._pending_approvals:
                    del self._pending_approvals[approval_id]
                return False

    def handle_user_approval(self, approval_id: str, approved: bool):
        """
        Called by web server when user responds to approval request.

        Args:
            approval_id: ID of the approval request
            approved: True if approved, False if denied
        """
        with self._lock:
            if approval_id in self._pending_approvals:
                self._approval_responses[approval_id] = approved
                self._pending_approvals[approval_id].set()
                self._logger.info(f"User approval {approval_id}: {'APPROVED' if approved else 'DENIED'}")

    # ========================================================================
    # WORKFLOW MANAGEMENT
    # ========================================================================

    def start_workflow(self, workflow_id: str, initiator_agent: str, workflow_data: Dict[str, Any]):
        """
        Start a multi-step workflow.

        Args:
            workflow_id: Unique workflow identifier
            initiator_agent: Agent starting the workflow
            workflow_data: Initial workflow state
        """
        with self._lock:
            self._active_workflows[workflow_id] = {
                "id": workflow_id,
                "initiator": initiator_agent,
                "status": "active",
                "started_at": datetime.now(),
                "data": workflow_data,
                "steps": [],
            }

        self._logger.info(f"Started workflow {workflow_id} by {initiator_agent}")

    def update_workflow(self, workflow_id: str, step_name: str, step_data: Dict[str, Any]):
        """
        Update workflow with a completed step.

        Args:
            workflow_id: Workflow identifier
            step_name: Name of the completed step
            step_data: Step results
        """
        with self._lock:
            if workflow_id in self._active_workflows:
                self._active_workflows[workflow_id]["steps"].append({
                    "name": step_name,
                    "timestamp": datetime.now(),
                    "data": step_data,
                })
                self._logger.info(f"Updated workflow {workflow_id}: step {step_name}")

    def complete_workflow(self, workflow_id: str, result: Dict[str, Any]):
        """
        Mark a workflow as complete.

        Args:
            workflow_id: Workflow identifier
            result: Final workflow result
        """
        with self._lock:
            if workflow_id in self._active_workflows:
                self._active_workflows[workflow_id]["status"] = "completed"
                self._active_workflows[workflow_id]["completed_at"] = datetime.now()
                self._active_workflows[workflow_id]["result"] = result
                self._logger.info(f"Completed workflow {workflow_id}")

    def fail_workflow(self, workflow_id: str, error: str):
        """
        Mark a workflow as failed.

        Args:
            workflow_id: Workflow identifier
            error: Error message
        """
        with self._lock:
            if workflow_id in self._active_workflows:
                self._active_workflows[workflow_id]["status"] = "failed"
                self._active_workflows[workflow_id]["failed_at"] = datetime.now()
                self._active_workflows[workflow_id]["error"] = error
                self._logger.error(f"Failed workflow {workflow_id}: {error}")

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow state"""
        with self._lock:
            return self._active_workflows.get(workflow_id)

    # ========================================================================
    # MESSAGE ROUTING
    # ========================================================================

    def _route_message(self, message: AgentMessage):
        """Route a message to the appropriate agent queue(s)"""
        # Add to history
        with self._lock:
            self._message_history.append(message)
            if len(self._message_history) > self._message_history_size:
                self._message_history.pop(0)

        # Route to specific agent or broadcast
        if message.target_agent:
            # Targeted message
            with self._lock:
                if message.target_agent in self._message_queues:
                    self._message_queues[message.target_agent].put(message)
                else:
                    self._logger.warning(
                        f"Target agent {message.target_agent} not registered"
                    )
        else:
            # Broadcast to all agents
            with self._lock:
                for agent_queue in self._message_queues.values():
                    agent_queue.put(message)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_message_history(self, limit: int = 100) -> List[AgentMessage]:
        """Get recent message history"""
        with self._lock:
            return list(self._message_history[-limit:])

    def get_pending_requests(self) -> List[str]:
        """Get list of pending request IDs"""
        with self._lock:
            return list(self._pending_requests.keys())

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workflows"""
        with self._lock:
            return {
                wf_id: wf for wf_id, wf in self._active_workflows.items()
                if wf["status"] == "active"
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        with self._lock:
            return {
                "registered_agents": len(self._message_queues),
                "pending_requests": len(self._pending_requests),
                "pending_approvals": len(self._pending_approvals),
                "active_workflows": len([
                    wf for wf in self._active_workflows.values() 
                    if wf["status"] == "active"
                ]),
                "total_messages": len(self._message_history),
                "event_subscribers": {
                    event_type: len(callbacks)
                    for event_type, callbacks in self._subscribers.items()
                },
            }

    def clear_history(self):
        """Clear message history (for testing/debugging)"""
        with self._lock:
            self._message_history.clear()
            self._logger.info("Message history cleared")
