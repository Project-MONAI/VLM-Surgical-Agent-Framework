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
Dynamic Agent Registry and Loader

Enables plugin-style agent architecture where agents can be added by simply:
1. Creating an agent Python file in agents/
2. Creating a config YAML file in configs/
3. (Optional) Adding agent_metadata to the config

The registry will automatically discover, load, and instantiate agents.
"""

import os
import sys
import yaml
import logging
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Type, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class AgentMetadata:
    """Metadata describing an agent's requirements and capabilities"""
    name: str
    class_name: str
    module: str
    config_path: str
    enabled: bool = True
    category: str = "general"
    priority: int = 10
    requires_llm: bool = True
    requires_visual: bool = False
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle: str = "singleton"  # singleton, per_request, background
    description: str = ""


class AgentRegistry:
    """
    Discovers, loads, and manages agents dynamically from a directory structure.

    Directory structure examples:
        Standard (current):
            agents/
                chat_agent.py (contains ChatAgent class)
            configs/
                chat_agent.yaml (contains agent configuration)

        Plugin-style (future):
            agents/
                custom_agents/
                    my_agent.py (contains MyAgent class)
            configs/
                custom_agents/
                    my_agent.yaml

    Agent Discovery:
        The registry automatically identifies agent configs by checking if they have:
        1. An 'agent_metadata' section with 'module' field, OR
        2. A corresponding Python file in the agents/ directory

        System configs (like global.yaml, selector.yaml) are automatically skipped
        because they don't match these criteria - no hardcoded skip lists needed!
    """

    def __init__(self,
                 agents_dir: str = "agents",
                 configs_dir: str = "configs",
                 enable_auto_discovery: bool = True,
                 global_config_path: str = "configs/global.yaml"):
        self._logger = logging.getLogger(__name__)
        self.agents_dir = Path(agents_dir)
        self.configs_dir = Path(configs_dir)
        self.enable_auto_discovery = enable_auto_discovery

        # Registry stores
        self._metadata: Dict[str, AgentMetadata] = {}
        self._agent_classes: Dict[str, Type] = {}
        self._agent_instances: Dict[str, Any] = {}
        self._dependency_resolvers: Dict[str, Callable] = {}
        self._agent_module_paths: Dict[str, Path] = {}  # For plugin imports

        # Special agents that should not be auto-loaded
        self._excluded_agents = {"SelectorAgent", "DynamicSelectorAgent"}

        # Load plugin directories from global config and environment
        self.plugin_dirs = self._load_plugin_directories(global_config_path)
        if self.plugin_dirs:
            self._logger.info(f"📦 Plugin directories configured: {self.plugin_dirs}")

        if enable_auto_discovery:
            self.discover_agents()

    def _load_plugin_directories(self, global_config_path: str) -> List[Path]:
        """Load plugin directory paths from global config and environment variable"""
        plugin_dirs = []

        # 1. Load from global config
        if os.path.exists(global_config_path):
            try:
                with open(global_config_path, 'r') as f:
                    global_config = yaml.safe_load(f) or {}
                config_dirs = global_config.get('plugin_directories', [])
                if config_dirs:
                    plugin_dirs.extend([Path(d) for d in config_dirs])
                    self._logger.debug(f"Loaded {len(config_dirs)} plugin dirs from global.yaml")
            except Exception as e:
                self._logger.warning(f"Failed to load plugin directories from {global_config_path}: {e}")

        # 2. Load from environment variable (overrides/extends config)
        env_dirs = os.environ.get('AGENT_PLUGIN_DIRS', '').strip()
        if env_dirs:
            # Support colon-separated paths (Linux style)
            env_paths = [Path(d.strip()) for d in env_dirs.split(':') if d.strip()]
            plugin_dirs.extend(env_paths)
            self._logger.debug(f"Loaded {len(env_paths)} plugin dirs from AGENT_PLUGIN_DIRS")

        # Validate and expand paths
        validated_dirs = []
        for plugin_dir in plugin_dirs:
            # Expand user home directory
            plugin_dir = plugin_dir.expanduser()

            # Make absolute if relative
            if not plugin_dir.is_absolute():
                plugin_dir = Path.cwd() / plugin_dir

            if plugin_dir.exists():
                validated_dirs.append(plugin_dir)
                self._logger.debug(f"✓ Valid plugin directory: {plugin_dir}")
                with open("configs" / "global.yaml", 'r') as f:
                    plugin_config = yaml.safe_load(f) or {}
                    excluded_agents = plugin_config.get('excluded_agents', [])
                    if excluded_agents:
                        self._excluded_agents.update(excluded_agents)  # set uses update(), not extend()
                        self._logger.info(f"📦 Plugin excludes: {excluded_agents}")
            else:
                self._logger.warning(f"✗ Plugin directory not found: {plugin_dir}")

        return validated_dirs

    def register_dependency_resolver(self, dep_type: str, resolver: Callable):
        """
        Register a function to resolve specific dependency types.

        Example:
            registry.register_dependency_resolver("frame_queue",
                lambda: webserver.frame_queue)
        """
        self._dependency_resolvers[dep_type] = resolver
        self._logger.debug(f"Registered dependency resolver for '{dep_type}'")

    def discover_agents(self) -> List[str]:
        """
        Discover all agents by scanning config files from core and plugin directories.
        Plugin agents will OVERRIDE core agents if names match.
        Returns list of discovered agent names.
        """
        discovered = []

        # 1. Discover core agents first
        self._logger.info("🔍 Discovering core agents...")
        core_agents = self._discover_from_directory(self.configs_dir, self.agents_dir, source="core")
        discovered.extend(core_agents)

        # 2. Discover plugin agents (these can override core agents)
        if self.plugin_dirs:
            self._logger.info(f"🔍 Discovering plugin agents from {len(self.plugin_dirs)} directories...")
            for plugin_dir in self.plugin_dirs:
                plugin_agents = self._discover_plugin(plugin_dir)
                for agent_name in plugin_agents:
                    if agent_name in discovered:
                        self._logger.info(f"🔄 Plugin agent '{agent_name}' overriding core agent")
                    else:
                        discovered.append(agent_name)

        self._logger.info(f"✓ Discovered {len(discovered)} agents total ({len(core_agents)} core, {len(discovered) - len(core_agents)} plugin overrides)")
        return discovered

    def _discover_from_directory(self, configs_dir: Path, agents_dir: Path, source: str = "core") -> List[str]:
        """
        Discover agents from a specific config and agents directory pair.

        Args:
            configs_dir: Directory containing YAML config files
            agents_dir: Directory containing Python agent files
            source: Source identifier for logging (e.g., "core", "plugin")

        Returns:
            List of discovered agent names
        """
        discovered = []

        if not configs_dir.exists():
            self._logger.warning(f"{source.capitalize()} configs directory not found: {configs_dir}")
            return discovered

        # Scan all YAML files in configs directory
        for config_file in configs_dir.glob("*.yaml"):
            try:
                metadata = self._load_agent_metadata(config_file, agents_dir)
                if metadata and metadata.enabled:
                    if metadata.name not in self._excluded_agents:
                        self._metadata[metadata.name] = metadata
                        # Store the agents directory path for this agent (for plugin imports)
                        if agents_dir != self.agents_dir:
                            self._agent_module_paths[metadata.name] = agents_dir
                        discovered.append(metadata.name)
                        source_label = f"[{source}]" if source != "core" else ""
                        self._logger.info(f"✓ Discovered agent: {metadata.name} ({metadata.category}) {source_label}")
            except Exception as e:
                self._logger.error(f"Error loading metadata from {config_file}: {e}", exc_info=True)

        return discovered

    def _discover_plugin(self, plugin_dir: Path) -> List[str]:
        """
        Discover agents from a plugin directory.

        Expected structure:
            plugin_dir/
                agents/
                    my_agent.py
                configs/
                    my_agent.yaml

        Args:
            plugin_dir: Root directory of the plugin

        Returns:
            List of discovered agent names from this plugin
        """
        agents_subdir = plugin_dir / "agents"
        configs_subdir = plugin_dir / "configs"

        # Validate plugin structure
        if not agents_subdir.exists():
            self._logger.warning(f"Plugin directory {plugin_dir} missing 'agents' subfolder - skipping")
            return []

        if not configs_subdir.exists():
            self._logger.warning(f"Plugin directory {plugin_dir} missing 'configs' subfolder - skipping")
            return []

        # Validate matching .py and .yaml files
        config_files = {f.stem for f in configs_subdir.glob("*.yaml")}
        agent_files = {f.stem for f in agents_subdir.glob("*.py") 
                      if f.name != "__init__.py"}

        # Only load agents that have both config and Python file
        matched_agents = config_files & agent_files

        if not matched_agents:
            self._logger.warning(f"Plugin directory {plugin_dir} has no matching agent/config pairs - skipping")
            return []

        unmatched = (config_files | agent_files) - matched_agents
        if unmatched:
            self._logger.warning(f"Plugin directory {plugin_dir} has unmatched files: {unmatched}")

        # Discover agents from this plugin
        plugin_name = plugin_dir.name
        return self._discover_from_directory(configs_subdir, agents_subdir, source=f"plugin:{plugin_name}")

    def _load_agent_metadata(self, config_path: Path, agents_dir: Optional[Path] = None) -> Optional[AgentMetadata]:
        """
        Load agent metadata from a config file.

        Returns None if the config is not a valid agent config (e.g., system configs like global.yaml).
        A valid agent config must have either:
        1. An 'agent_metadata' section with 'module' field, OR
        2. A corresponding Python file in agents/ directory

        Args:
            config_path: Path to the YAML config file
            agents_dir: Optional custom agents directory (for plugins)
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        metadata_dict = config.get('agent_metadata', {})

        # Use provided agents_dir or fall back to default
        if agents_dir is None:
            agents_dir = self.agents_dir

        # Determine unique agent name (can differ from class_name for multiple instances)
        # Priority: metadata.name > config.agent_name > infer from filename
        agent_name = None
        if metadata_dict:
            agent_name = metadata_dict.get('name') or config.get('agent_name')

        if not agent_name:
            agent_name = config.get('agent_name')

        if not agent_name:
            # Infer from filename: chat_agent.yaml -> ChatAgent
            name_parts = config_path.stem.split('_')
            agent_name = ''.join(word.capitalize() for word in name_parts)

        # If no metadata section, try to infer from config (backward compatibility)
        if not metadata_dict:
            # Check if a corresponding Python agent file exists
            module_name = config_path.stem  # e.g., chat_agent
            expected_agent_file = agents_dir / f"{module_name}.py"

            if not expected_agent_file.exists():
                # Not a valid agent config - likely a system config like global.yaml or selector.yaml
                self._logger.debug(f"Skipping {config_path.name}: no agent_metadata and no matching Python file at {expected_agent_file}")
                return None

            metadata_dict = {
                'class_name': agent_name,
                'module': f'agents.{module_name}',
                'enabled': True,
                'description': config.get('description', ''),
            }

        # Create metadata object
        # Use agent_name as unique identifier, class_name for the Python class
        metadata = AgentMetadata(
            name=agent_name,  # Unique identifier for this agent instance
            class_name=metadata_dict.get('class_name', agent_name),  # Python class to instantiate
            module=metadata_dict.get('module', ''),
            config_path=str(config_path),
            enabled=metadata_dict.get('enabled', True),
            category=metadata_dict.get('category', 'general'),
            priority=metadata_dict.get('priority', 10),
            requires_llm=metadata_dict.get('requires_llm', True),
            requires_visual=metadata_dict.get('requires_visual', False),
            dependencies=metadata_dict.get('dependencies', []),
            lifecycle=metadata_dict.get('lifecycle', 'singleton'),
            description=metadata_dict.get('description', config.get('description', '')),
        )

        if not metadata.name or not metadata.module:
            self._logger.warning(f"Invalid metadata in {config_path}: name={metadata.name}, module={metadata.module}")
            return None

        return metadata
    def load_agent_class(self, agent_name: str) -> Optional[Type]:
        """Dynamically import and return the agent class"""
        if agent_name in self._agent_classes:
            return self._agent_classes[agent_name]

        metadata = self._metadata.get(agent_name)
        if not metadata:
            self._logger.error(f"No metadata found for agent: {agent_name}")
            return None

        try:
            # Check if this is a plugin agent (has custom module path)
            if agent_name in self._agent_module_paths:
                # Plugin agent - load from file path
                agents_dir = self._agent_module_paths[agent_name]
                module_file = agents_dir / f"{metadata.module.split('.')[-1]}.py"

                if not module_file.exists():
                    self._logger.error(f"Plugin module file not found: {module_file}")
                    return None

                # Load module from file
                spec = importlib.util.spec_from_file_location(metadata.module, module_file)
                if spec is None or spec.loader is None:
                    self._logger.error(f"Failed to create module spec for {module_file}")
                    return None

                module = importlib.util.module_from_spec(spec)
                sys.modules[metadata.module] = module  # Add to sys.modules for imports
                spec.loader.exec_module(module)

                self._logger.debug(f"Loaded plugin module: {metadata.module} from {module_file}")
            else:
                # Core agent - standard import
                module = importlib.import_module(metadata.module)

            # Get the agent class from the module
            agent_class = getattr(module, metadata.class_name)

            # Verify it's a class
            if not inspect.isclass(agent_class):
                self._logger.error(f"{metadata.class_name} is not a class")
                return None

            # Cache it
            self._agent_classes[agent_name] = agent_class
            self._logger.debug(f"✓ Loaded agent class: {agent_name} from {metadata.module}")
            return agent_class

        except ImportError as e:
            self._logger.error(f"Failed to import module {metadata.module} for agent {agent_name}: {e}")
            return None
        except AttributeError as e:
            self._logger.error(f"Failed to find class {metadata.class_name} in module {metadata.module}: {e}")
            return None
        except Exception as e:
            self._logger.error(f"Failed to load agent {agent_name}: {e}", exc_info=True)
            return None

    def instantiate_agent(self, 
                         agent_name: str, 
                         response_handler,
                         **override_kwargs) -> Optional[Any]:
        """
        Instantiate an agent with dependency injection.

        Args:
            agent_name: Name of the agent to instantiate
            response_handler: Response handler (required for all agents)
            **override_kwargs: Additional kwargs to override dependencies

        Returns:
            Agent instance or None if instantiation fails
        """
        # Check if already instantiated (singleton or background)
        metadata = self._metadata.get(agent_name)
        if not metadata:
            self._logger.error(f"Agent not registered: {agent_name}")
            return None

        if metadata.lifecycle in ("singleton", "background") and agent_name in self._agent_instances:
            self._logger.debug(f"Returning cached instance of {agent_name}")
            return self._agent_instances[agent_name]

        # Load the class
        agent_class = self.load_agent_class(agent_name)
        if not agent_class:
            return None

        try:
            # Build constructor arguments
            kwargs = {
                'settings_path': metadata.config_path,
                'response_handler': response_handler,
            }

            # Resolve dependencies
            for dep in metadata.dependencies:
                dep_type = dep.get('type')
                dep_name = dep.get('name', dep_type)  # Use name if provided, else use type

                if dep_type in override_kwargs:
                    # Use override if provided
                    kwargs[dep_name] = override_kwargs[dep_type]
                elif dep_type in self._dependency_resolvers:
                    # Use registered resolver
                    resolver = self._dependency_resolvers[dep_type]
                    kwargs[dep_name] = resolver()  # Use dep_name as parameter name
                else:
                    self._logger.warning(
                        f"No resolver for dependency '{dep_type}' of {agent_name}"
                    )

            # Add any additional overrides
            for key, value in override_kwargs.items():
                if key not in kwargs:
                    kwargs[key] = value

            # Inspect __init__ to only pass supported parameters
            sig = inspect.signature(agent_class.__init__)
            filtered_kwargs = {
                k: v for k, v in kwargs.items() 
                if k in sig.parameters
            }

            # Instantiate
            self._logger.info(f"Instantiating {agent_name}...")
            instance = agent_class(**filtered_kwargs)

            # Store the instance (singletons and background agents need to persist)
            if metadata.lifecycle in ("singleton", "background"):
                self._agent_instances[agent_name] = instance

            self._logger.info(f"✓ Successfully instantiated {agent_name}")
            return instance

        except Exception as e:
            self._logger.error(f"Failed to instantiate {agent_name}: {e}", exc_info=True)
            return None

    def get_agent(self, agent_name: str) -> Optional[Any]:
        """Get an already-instantiated agent"""
        return self._agent_instances.get(agent_name)

    def get_all_agents(self) -> Dict[str, Any]:
        """Get all instantiated agents"""
        return self._agent_instances.copy()

    def get_agent_names(self, category: Optional[str] = None, 
                       enabled_only: bool = True) -> List[str]:
        """
        Get list of agent names, optionally filtered.

        Args:
            category: Filter by category (conversational, analysis, etc.)
            enabled_only: Only return enabled agents

        Returns:
            List of agent names sorted by priority
        """
        names = []
        for name, metadata in self._metadata.items():
            if enabled_only and not metadata.enabled:
                continue
            if category and metadata.category != category:
                continue
            names.append(name)

        # Sort by priority (lower number = higher priority)
        names.sort(key=lambda n: self._metadata[n].priority)
        return names

    def get_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """Get metadata for an agent"""
        return self._metadata.get(agent_name)

    def get_all_metadata(self) -> Dict[str, AgentMetadata]:
        """Get metadata for all discovered agents"""
        return self._metadata.copy()

    def reload_agent(self, agent_name: str):
        """
        Hot-reload an agent (useful for development).
        Clears cached class and instance, forcing reload on next access.
        """
        if agent_name in self._agent_classes:
            del self._agent_classes[agent_name]
            self._logger.info(f"Cleared cached class for {agent_name}")

        if agent_name in self._agent_instances:
            # Cleanup if needed
            old_instance = self._agent_instances[agent_name]
            if hasattr(old_instance, 'stop'):
                try:
                    old_instance.stop()
                    self._logger.info(f"Stopped old instance of {agent_name}")
                except Exception as e:
                    self._logger.warning(f"Error stopping {agent_name}: {e}")
            del self._agent_instances[agent_name]

        # Reload metadata
        metadata = self._metadata.get(agent_name)
        if metadata:
            config_path = Path(metadata.config_path)
            if config_path.exists():
                new_metadata = self._load_agent_metadata(config_path)
                if new_metadata:
                    self._metadata[agent_name] = new_metadata

        self._logger.info(f"Reloaded agent: {agent_name}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics for debugging/monitoring"""
        return {
            'total_discovered': len(self._metadata),
            'total_loaded_classes': len(self._agent_classes),
            'total_instantiated': len(self._agent_instances),
            'agents_by_category': self._get_agents_by_category(),
            'enabled_agents': len([m for m in self._metadata.values() if m.enabled]),
        }

    def _get_agents_by_category(self) -> Dict[str, List[str]]:
        """Group agents by category"""
        by_category = {}
        for name, metadata in self._metadata.items():
            category = metadata.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(name)
        return by_category

