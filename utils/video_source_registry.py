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
Generic Video Source Registry
=============================

Manages multiple video sources and their associated selectors dynamically.
Enables adding new video sources without code changes - just configuration.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from glob import glob


@dataclass
class VideoSourceConfig:
    """
    Configuration for a single video source.
    Attributes:
        mode_name: Unique identifier for this video source (e.g., 'surgical', 'operating_room')
        enabled: Whether this video source is active
        display_name: Human-readable name for UI display
        description: Detailed description of the video source
        selector_config: Path to default selector configuration
        plugin_selector_pattern: Glob pattern to find plugin-specific selector
        frame_queue_name: Name of the frame queue for this source
        context_name: Context identifier used by agents (e.g., 'procedure', 'operating_room')
        auto_detect_flags: WebSocket message flags for auto-detection
        priority: Priority level (higher = preferred default), default 0
        source_type: Type of source ('uploaded' or 'livestream'), optional for backwards compatibility
    """
    mode_name: str
    enabled: bool
    display_name: str
    description: str
    selector_config: str
    plugin_selector_pattern: str
    frame_queue_name: str
    context_name: str
    auto_detect_flags: Dict[str, str] = field(default_factory=dict)
    priority: int = 0
    source_type: Optional[str] = None  # 'uploaded' or 'livestream'


class VideoSourceRegistry:
    """
    Registry for managing multiple video sources dynamically.
    This class enables a generic, configuration-driven approach to handling
    multiple video sources without hardcoding logic for each source.
    Key Features:
        - Configuration-driven: Define sources in YAML
        - Auto-discovery: Finds plugin selectors automatically
        - Zero-code extension: Add new sources without touching Python
        - Validation: Ensures modes are valid before use
        - Introspection: Query available sources at runtime
    Example:
        >>> registry = VideoSourceRegistry("configs/video_sources.yaml", plugin_dirs)
        >>> registry.register_all_sources(agent_registry, response_handler)
        >>>
        >>> # At runtime
        >>> selector = registry.get_selector("surgical")
        >>> context = registry.get_context("surgical")
        >>> frame_queue_name = registry.get_frame_queue_name("surgical")
    Configuration File Format:
        See configs/video_sources.yaml for complete example.
    """
    def __init__(self, config_path: str, plugin_dirs: List[Path] = None):
        """
        Initialize the video source registry.
        Args:
            config_path: Path to video_sources.yaml configuration file
            plugin_dirs: List of plugin directories to search for selector configs
        """
        self._logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.plugin_dirs = plugin_dirs or []
        # Registries
        self.sources: Dict[str, VideoSourceConfig] = {}
        self.selectors: Dict[str, Any] = {}  # mode_name -> selector instance
        # Settings
        self.default_source = "surgical"
        self.auto_detect_enabled = True
        self.allow_manual_switching = True
        # Load configuration
        self._load_config()
    def _load_config(self):
        """
        Load video sources from configuration file.
        Creates VideoSourceConfig objects for each enabled source and
        stores them in the registry.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config:
                self._logger.error("Empty or invalid configuration file")
                self._create_fallback_config()
                return
            # Load global settings
            settings = config.get('settings', {})
            self.default_source = settings.get('default_source', 'surgical')
            self.auto_detect_enabled = settings.get('auto_detect_enabled', True)
            self.allow_manual_switching = settings.get('allow_manual_switching', True)
            self._logger.info(f"Global settings: default_source={self.default_source}, "
                            f"auto_detect={self.auto_detect_enabled}")
            # Load video sources
            video_sources = config.get('video_sources', {})
            if not video_sources:
                self._logger.warning("No video sources defined in configuration")
                self._create_fallback_config()
                return
            for mode_name, source_config in video_sources.items():
                if not source_config.get('enabled', True):
                    self._logger.info(f"Video source '{mode_name}' is disabled, skipping")
                    continue
                # Create VideoSourceConfig
                try:
                    source = VideoSourceConfig(
                        mode_name=mode_name,
                        enabled=source_config.get('enabled', True),
                        display_name=source_config.get('display_name', mode_name.replace('_', ' ').title()),
                        description=source_config.get('description', ''),
                        selector_config=source_config.get('selector_config', 'configs/selector.yaml'),
                        plugin_selector_pattern=source_config.get('plugin_selector_pattern', ''),
                        frame_queue_name=source_config.get('frame_queue_name', f'{mode_name}_frame_queue'),
                        context_name=source_config.get('context_name', mode_name),
                        auto_detect_flags=source_config.get('auto_detect', {}),
                        priority=source_config.get('priority', 0),
                        source_type=source_config.get('source_type')
                    )
                    self.sources[mode_name] = source
                    self._logger.info(f"✓ Registered video source: '{mode_name}' "
                                    f"({source.display_name}, priority={source.priority})")
                except Exception as e:
                    self._logger.error(f"Failed to create config for source '{mode_name}': {e}")
            if not self.sources:
                self._logger.warning("No enabled video sources found")
                self._create_fallback_config()
                return
            self._logger.info(f"Successfully loaded {len(self.sources)} video sources")
        except FileNotFoundError:
            self._logger.error(f"Configuration file not found: {self.config_path}")
            self._create_fallback_config()
        except yaml.YAMLError as e:
            self._logger.error(f"Failed to parse YAML configuration: {e}")
            self._create_fallback_config()
        except Exception as e:
            self._logger.error(f"Unexpected error loading configuration: {e}", exc_info=True)
            self._create_fallback_config()
    def _create_fallback_config(self):
        """
        Create minimal fallback configuration when main config fails.
        Creates a basic 'surgical' video source to ensure system can still operate.
        """
        self._logger.warning("Using fallback video source configuration")
        self.sources = {
            'surgical': VideoSourceConfig(
                mode_name='surgical',
                enabled=True,
                display_name='Surgical Camera',
                description='Default surgical feed',
                selector_config='configs/selector.yaml',
                plugin_selector_pattern='',
                frame_queue_name='frame_queue',
                context_name='procedure',
                auto_detect_flags={'websocket_flag': 'auto_frame', 'frame_data_key': 'frame_data'},
                priority=10
            )
        }
        self.default_source = 'surgical'
    def register_all_sources(self, agent_registry, response_handler):
        """
        Register selectors for all enabled video sources.
        Searches for plugin-specific selectors first, falls back to default
        selector config if not found. Creates DynamicSelectorAgent instances
        for each source.
        Args:
            agent_registry: Agent registry instance for agent discovery
            response_handler: Response handler for agent output
        Raises:
            ImportError: If DynamicSelectorAgent cannot be imported
        """
        try:
            from agents.dynamic_selector_agent import DynamicSelectorAgent
        except ImportError as e:
            self._logger.error(f"Failed to import DynamicSelectorAgent: {e}")
            return
        for mode_name, source in self.sources.items():
            selector_path = self._find_selector_config(source)
            if not selector_path:
                selector_path = source.selector_config
                self._logger.debug(f"Using default selector for '{mode_name}': {selector_path}")
            try:
                selector = DynamicSelectorAgent(
                    selector_path,
                    response_handler,
                    agent_registry=agent_registry
                )
                self.selectors[mode_name] = selector
                self._logger.info(f"✓ Selector ready for '{mode_name}': {selector_path}")
            except FileNotFoundError:
                self._logger.error(f"Selector config not found for '{mode_name}': {selector_path}")
            except Exception as e:
                self._logger.error(f"Failed to create selector for '{mode_name}': {e}", exc_info=True)
        if not self.selectors:
            self._logger.error("No selectors were successfully registered!")
        else:
            self._logger.info(f"✓ Registered {len(self.selectors)} selectors for {len(self.sources)} sources")
    def _find_selector_config(self, source: VideoSourceConfig) -> Optional[str]:
        """
        Find selector configuration using plugin pattern.
        Searches plugin directories for selector configs matching the pattern
        specified in the video source configuration.
        Args:
            source: Video source configuration with plugin_selector_pattern
        Returns:
            Path to selector config, or None if not found
        """
        if not source.plugin_selector_pattern:
            return None
        # Search in plugin directories
        for plugin_dir in self.plugin_dirs:
            # Clean up pattern - remove leading wildcards if absolute path will be used
            pattern = source.plugin_selector_pattern.lstrip('*/')
            full_pattern = str(plugin_dir / pattern)
            try:
                matches = glob(full_pattern, recursive=True)
                if matches:
                    self._logger.info(f"📦 Found plugin selector for '{source.mode_name}': {matches[0]}")
                    return matches[0]
            except Exception as e:
                self._logger.debug(f"Error searching for plugin selector: {e}")
        return None
    def get_selector(self, mode_name: str):
        """
        Get selector for a video source mode.
        Args:
            mode_name: Video source mode name
        Returns:
            Selector instance, or default selector if mode not found
        """
        selector = self.selectors.get(mode_name)
        if not selector:
            self._logger.warning(f"No selector found for mode '{mode_name}', using default")
            selector = self.selectors.get(self.default_source)
            if not selector and self.selectors:
                # Return any available selector as last resort
                selector = next(iter(self.selectors.values()))
        return selector
    def get_context(self, mode_name: str) -> str:
        """
        Get context name for a video source mode.
        The context name is used by agents to understand which type of content
        they're analyzing (e.g., 'procedure' for surgical video, 'operating_room'
        for OR webcam).
        Args:
            mode_name: Video source mode name
        Returns:
            Context name (e.g., 'procedure', 'operating_room')
        """
        source = self.sources.get(mode_name)
        if source:
            return source.context_name
        # Fallback to default source's context
        default = self.sources.get(self.default_source)
        if default:
            return default.context_name
        # Last resort: return mode name itself
        return mode_name
    def get_frame_queue_name(self, mode_name: str) -> str:
        """
        Get frame queue name for a video source.
        Args:
            mode_name: Video source mode name
        Returns:
            Frame queue name (e.g., 'frame_queue', 'operating_room_frame_queue')
        """
        source = self.sources.get(mode_name)
        if source:
            return source.frame_queue_name
        # Fallback to default
        default = self.sources.get(self.default_source)
        if default:
            return default.frame_queue_name
        return 'frame_queue'
    def get_auto_detect_flags(self, mode_name: str) -> Dict[str, str]:
        """
        Get auto-detection flags for a video source.
        These flags are used to automatically detect which video source is active
        based on WebSocket message contents.
        Args:
            mode_name: Video source mode name
        Returns:
            Dictionary with 'websocket_flag' and 'frame_data_key' entries
        """
        source = self.sources.get(mode_name)
        if source:
            return source.auto_detect_flags
        return {}
    def detect_mode_from_message(self, message: dict) -> Optional[str]:
        """
        Auto-detect video mode from WebSocket message.
        Checks message against auto-detection flags for all registered sources
        to determine which video source is active.
        Args:
            message: WebSocket message data (dictionary)
        Returns:
            Detected mode name, or None if no match found
        """
        if not self.auto_detect_enabled:
            return None
        # Sort by priority (higher priority checked first)
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        for mode_name, source in sorted_sources:
            flags = source.auto_detect_flags
            websocket_flag = flags.get('websocket_flag')
            if websocket_flag and message.get(websocket_flag):
                self._logger.debug(f"Auto-detected mode '{mode_name}' from flag '{websocket_flag}'")
                return mode_name
        return None
    def get_all_modes(self) -> List[str]:
        """
        Get list of all registered video source modes.
        Returns:
            List of mode names (e.g., ['surgical', 'operating_room'])
        """
        return list(self.sources.keys())
    def get_source_info(self, mode_name: str) -> Optional[VideoSourceConfig]:
        """
        Get full configuration for a video source.
        Args:
            mode_name: Video source mode name
        Returns:
            VideoSourceConfig object, or None if not found
        """
        return self.sources.get(mode_name)
    def validate_mode(self, mode_name: str) -> bool:
        """
        Check if a mode name is valid and enabled.
        Args:
            mode_name: Video source mode name to validate
        Returns:
            True if mode exists and is enabled, False otherwise
        """
        return mode_name in self.sources
    def get_default_mode(self) -> str:
        """
        Get the default video source mode.
        Returns:
            Default mode name as configured in settings
        """
        return self.default_source
    def list_sources(self) -> Dict[str, dict]:
        """
        Get a summary of all video sources.
        Useful for API endpoints and UI generation.
        Returns:
            Dictionary mapping mode names to their display information:
            {
                'surgical': {
                    'display_name': 'Surgical Camera',
                    'description': '...',
                    'context': 'procedure',
                    'enabled': True,
                    'priority': 10,
                    'has_selector': True
                },
                ...
            }
        """
        return {
            mode: {
                'display_name': source.display_name,
                'description': source.description,
                'context': source.context_name,
                'enabled': source.enabled,
                'priority': source.priority,
                'has_selector': mode in self.selectors,
                'frame_queue': source.frame_queue_name
            }
            for mode, source in self.sources.items()
        }
    def get_mode_by_context(self, context_name: str) -> Optional[str]:
        """
        Find video mode by context name.
        Useful for reverse lookups when you have a context and need the mode.
        Args:
            context_name: Context identifier (e.g., 'procedure', 'operating_room')
        Returns:
            Mode name that matches the context, or None if not found
        """
        for mode, source in self.sources.items():
            if source.context_name == context_name:
                return mode
        return None
    def __repr__(self) -> str:
        """String representation of registry."""
        return (f"VideoSourceRegistry(sources={len(self.sources)}, "
                f"selectors={len(self.selectors)}, "
                f"default='{self.default_source}')")
    def __str__(self) -> str:
        """Human-readable string representation."""
        sources_list = ', '.join(f"'{m}'" for m in self.sources.keys())
        return f"VideoSourceRegistry with {len(self.sources)} sources: {sources_list}"
