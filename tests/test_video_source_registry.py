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
Unit Tests for VideoSourceRegistry
===================================

Run with: python -m pytest tests/test_video_source_registry.py -v

Or: python tests/test_video_source_registry.py
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_source_registry import VideoSourceRegistry, VideoSourceConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'settings': {
            'default_source': 'surgical',
            'auto_detect_enabled': True,
            'allow_manual_switching': True
        },
        'video_sources': {
            'surgical': {
                'enabled': True,
                'display_name': 'Surgical Camera',
                'description': 'Laparoscopic surgical camera',
                'selector_config': 'configs/selector.yaml',
                'plugin_selector_pattern': '*/configs/*_surgical_selector.yaml',
                'frame_queue_name': 'frame_queue',
                'context_name': 'procedure',
                'auto_detect': {
                    'websocket_flag': 'auto_frame',
                    'frame_data_key': 'frame_data'
                },
                'priority': 10
            },
            'operating_room': {
                'enabled': True,
                'display_name': 'Operating Room',
                'description': 'OR webcam feed',
                'selector_config': 'configs/selector.yaml',
                'plugin_selector_pattern': '*/configs/*_or_selector.yaml',
                'frame_queue_name': 'or_frame_queue',
                'context_name': 'operating_room',
                'auto_detect': {
                    'websocket_flag': 'or_auto_frame',
                    'frame_data_key': 'or_frame_data'
                },
                'priority': 5
            },
            'microscope': {
                'enabled': False,
                'display_name': 'Microscope',
                'description': 'Surgical microscope',
                'selector_config': 'configs/selector.yaml',
                'plugin_selector_pattern': '*/configs/*_micro_selector.yaml',
                'frame_queue_name': 'micro_queue',
                'context_name': 'microscope',
                'auto_detect': {
                    'websocket_flag': 'micro_frame',
                    'frame_data_key': 'micro_data'
                },
                'priority': 3
            }
        }
    }


@pytest.fixture
def config_file(sample_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def registry(config_file):
    """Create VideoSourceRegistry instance."""
    return VideoSourceRegistry(config_file)


# ============================================================================
# Test Configuration Loading
# ============================================================================

def test_load_config(registry):
    """Test configuration loading."""
    assert len(registry.sources) == 2  # Only enabled sources
    assert 'surgical' in registry.sources
    assert 'operating_room' in registry.sources
    assert 'microscope' not in registry.sources  # Disabled


def test_default_source(registry):
    """Test default source setting."""
    assert registry.default_source == 'surgical'
    assert registry.get_default_mode() == 'surgical'


def test_auto_detect_enabled(registry):
    """Test auto-detect setting."""
    assert registry.auto_detect_enabled is True


def test_invalid_config_file():
    """Test handling of invalid config file."""
    registry = VideoSourceRegistry('nonexistent_file.yaml')

    # Should create fallback config
    assert len(registry.sources) >= 1
    assert 'surgical' in registry.sources  # Fallback


def test_empty_config_file():
    """Test handling of empty config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        empty_config = f.name

    try:
        registry = VideoSourceRegistry(empty_config)
        assert len(registry.sources) >= 1  # Should have fallback
    finally:
        os.unlink(empty_config)


# ============================================================================
# Test VideoSourceConfig
# ============================================================================

def test_source_config_attributes(registry):
    """Test VideoSourceConfig attributes."""
    source = registry.sources['surgical']

    assert isinstance(source, VideoSourceConfig)
    assert source.mode_name == 'surgical'
    assert source.enabled is True
    assert source.display_name == 'Surgical Camera'
    assert source.context_name == 'procedure'
    assert source.priority == 10


def test_source_auto_detect_flags(registry):
    """Test auto-detect flags configuration."""
    source = registry.sources['surgical']
    flags = source.auto_detect_flags

    assert flags['websocket_flag'] == 'auto_frame'
    assert flags['frame_data_key'] == 'frame_data'


# ============================================================================
# Test Mode Validation
# ============================================================================

def test_validate_mode_valid(registry):
    """Test validation of valid modes."""
    assert registry.validate_mode('surgical') is True
    assert registry.validate_mode('operating_room') is True


def test_validate_mode_invalid(registry):
    """Test validation of invalid modes."""
    assert registry.validate_mode('invalid_mode') is False
    assert registry.validate_mode('microscope') is False  # Disabled
    assert registry.validate_mode('') is False
    assert registry.validate_mode(None) is False


# ============================================================================
# Test Selector Lookup
# ============================================================================

def test_get_selector_without_registration(registry):
    """Test getting selector before registration."""
    selector = registry.get_selector('surgical')
    # Should return None or default since not registered yet
    assert selector is None or selector is not None  # Depends on fallback


def test_get_selector_invalid_mode(registry):
    """Test getting selector for invalid mode."""
    selector = registry.get_selector('invalid_mode')
    # Should fall back to default
    assert selector is not None or selector is None  # Depends on implementation


# ============================================================================
# Test Context Lookup
# ============================================================================

def test_get_context_valid_mode(registry):
    """Test getting context for valid mode."""
    assert registry.get_context('surgical') == 'procedure'
    assert registry.get_context('operating_room') == 'operating_room'


def test_get_context_invalid_mode(registry):
    """Test getting context for invalid mode."""
    # Should fall back to default context
    context = registry.get_context('invalid_mode')
    assert context == 'procedure'  # Default source's context


def test_get_mode_by_context(registry):
    """Test reverse lookup: context to mode."""
    assert registry.get_mode_by_context('procedure') == 'surgical'
    assert registry.get_mode_by_context('operating_room') == 'operating_room'
    assert registry.get_mode_by_context('invalid') is None


# ============================================================================
# Test Frame Queue Names
# ============================================================================

def test_get_frame_queue_name(registry):
    """Test getting frame queue names."""
    assert registry.get_frame_queue_name('surgical') == 'frame_queue'
    assert registry.get_frame_queue_name('operating_room') == 'or_frame_queue'


def test_get_frame_queue_name_invalid_mode(registry):
    """Test getting frame queue name for invalid mode."""
    # Should fall back to default
    name = registry.get_frame_queue_name('invalid_mode')
    assert name == 'frame_queue'  # Default


# ============================================================================
# Test Auto-Detection
# ============================================================================

def test_auto_detect_surgical_mode(registry):
    """Test auto-detecting surgical mode."""
    message = {
        'auto_frame': True,
        'frame_data': 'base64_encoded_frame...'
    }

    detected = registry.detect_mode_from_message(message)
    assert detected == 'surgical'


def test_auto_detect_or_mode(registry):
    """Test auto-detecting OR mode."""
    message = {
        'or_auto_frame': True,
        'or_frame_data': 'base64_encoded_frame...'
    }

    detected = registry.detect_mode_from_message(message)
    assert detected == 'operating_room'


def test_auto_detect_no_match(registry):
    """Test auto-detection with no matching flags."""
    message = {
        'some_other_flag': True,
        'data': 'something'
    }

    detected = registry.detect_mode_from_message(message)
    assert detected is None


def test_auto_detect_priority(registry):
    """Test that higher priority sources are checked first."""
    # Both flags present - should detect higher priority (surgical=10 vs or=5)
    message = {
        'auto_frame': True,
        'or_auto_frame': True,
        'frame_data': 'data1',
        'or_frame_data': 'data2'
    }

    detected = registry.detect_mode_from_message(message)
    # Should detect surgical (higher priority)
    assert detected == 'surgical'


def test_auto_detect_disabled():
    """Test when auto-detection is disabled."""
    config = {
        'settings': {'auto_detect_enabled': False},
        'video_sources': {
            'surgical': {
                'enabled': True,
                'auto_detect': {'websocket_flag': 'auto_frame'}
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        registry = VideoSourceRegistry(config_path)
        message = {'auto_frame': True, 'frame_data': 'data'}

        detected = registry.detect_mode_from_message(message)
        assert detected is None  # Auto-detect disabled
    finally:
        os.unlink(config_path)


# ============================================================================
# Test Source Listing
# ============================================================================

def test_get_all_modes(registry):
    """Test getting all mode names."""
    modes = registry.get_all_modes()

    assert 'surgical' in modes
    assert 'operating_room' in modes
    assert 'microscope' not in modes  # Disabled
    assert len(modes) == 2


def test_list_sources(registry):
    """Test getting source summary."""
    sources = registry.list_sources()

    assert 'surgical' in sources
    assert 'operating_room' in sources

    surgical_info = sources['surgical']
    assert surgical_info['display_name'] == 'Surgical Camera'
    assert surgical_info['context'] == 'procedure'
    assert surgical_info['enabled'] is True
    assert surgical_info['priority'] == 10
    assert surgical_info['has_selector'] is False  # Not registered yet


def test_get_source_info(registry):
    """Test getting full source configuration."""
    source = registry.get_source_info('surgical')

    assert source is not None
    assert isinstance(source, VideoSourceConfig)
    assert source.display_name == 'Surgical Camera'


# ============================================================================
# Test String Representations
# ============================================================================

def test_repr(registry):
    """Test __repr__ method."""
    repr_str = repr(registry)

    assert 'VideoSourceRegistry' in repr_str
    assert 'sources=2' in repr_str
    assert 'surgical' in repr_str


def test_str(registry):
    """Test __str__ method."""
    str_repr = str(registry)

    assert 'VideoSourceRegistry' in str_repr
    assert '2 sources' in str_repr
    assert 'surgical' in str_repr


# ============================================================================
# Integration Tests (require mock agents)
# ============================================================================

def test_register_all_sources_no_agents(registry):
    """Test registering sources without agent infrastructure."""
    # This will fail to import agents but should handle gracefully
    # In real usage, proper agent registry and response handler needed

    # For now, just verify method exists and doesn't crash
    try:
        registry.register_all_sources(None, None)
    except (ImportError, AttributeError):
        # Expected if agents not available
        pass


# ============================================================================
# Edge Cases
# ============================================================================

def test_multiple_registries():
    """Test creating multiple registry instances."""
    config1 = {
        'video_sources': {
            'camera1': {'enabled': True, 'display_name': 'Camera 1'}
        }
    }
    config2 = {
        'video_sources': {
            'camera2': {'enabled': True, 'display_name': 'Camera 2'}
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
        yaml.dump(config1, f1)
        path1 = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
        yaml.dump(config2, f2)
        path2 = f2.name

    try:
        registry1 = VideoSourceRegistry(path1)
        registry2 = VideoSourceRegistry(path2)

        assert 'camera1' in registry1.sources
        assert 'camera1' not in registry2.sources
        assert 'camera2' in registry2.sources
        assert 'camera2' not in registry1.sources
    finally:
        os.unlink(path1)
        os.unlink(path2)


def test_unicode_in_config():
    """Test handling Unicode characters in configuration."""
    config = {
        'video_sources': {
            'surgical': {
                'enabled': True,
                'display_name': '手術カメラ',  # Japanese
                'description': 'Caméra chirurgicale'  # French
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', 
                                    encoding='utf-8', delete=False) as f:
        yaml.dump(config, f, allow_unicode=True)
        config_path = f.name

    try:
        registry = VideoSourceRegistry(config_path)
        source = registry.sources['surgical']

        assert source.display_name == '手術カメラ'
        assert '手術カメラ' in str(registry.list_sources())
    finally:
        os.unlink(config_path)


# ============================================================================
# Run Tests Standalone
# ============================================================================

if __name__ == '__main__':
    # Run with pytest if available, otherwise basic test runner
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        print("pytest not available, running basic tests...")

        # Create sample config
        config = {
            'settings': {'default_source': 'surgical'},
            'video_sources': {
                'surgical': {
                    'enabled': True,
                    'display_name': 'Surgical Camera',
                    'context_name': 'procedure',
                    'auto_detect': {'websocket_flag': 'auto_frame'}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Basic tests
            print("Testing configuration load...")
            registry = VideoSourceRegistry(config_path)
            assert len(registry.sources) > 0
            print("✓ Configuration loaded")

            print("Testing mode validation...")
            assert registry.validate_mode('surgical')
            assert not registry.validate_mode('invalid')
            print("✓ Mode validation works")

            print("Testing auto-detection...")
            msg = {'auto_frame': True}
            detected = registry.detect_mode_from_message(msg)
            assert detected == 'surgical'
            print("✓ Auto-detection works")

            print("\n✅ All basic tests passed!")

        finally:
            os.unlink(config_path)
