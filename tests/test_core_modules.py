"""
Tests for SC-NeuroCore Core Modules
===================================

Comprehensive test suite for:
- TensorStream: Unified data structure with domain conversions
- CognitiveOrchestrator: Central pipeline orchestration

Author: Claude (Session 2026-01-31)
"""

import pytest
import numpy as np
import os
import sys

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sc_neurocore.core.tensor_stream import TensorStream
from sc_neurocore.core.orchestrator import CognitiveOrchestrator


# =============================================================================
# TensorStream Tests
# =============================================================================

class TestTensorStream:
    """Tests for TensorStream data structure."""

    def test_create_from_prob(self):
        """Test creating TensorStream from probability values."""
        probs = np.array([0.2, 0.5, 0.8])
        stream = TensorStream.from_prob(probs)

        assert stream.domain == 'prob'
        assert np.array_equal(stream.data, probs)

    def test_to_prob_identity(self):
        """Test to_prob returns identity for prob domain."""
        probs = np.array([0.1, 0.9, 0.5])
        stream = TensorStream.from_prob(probs)

        result = stream.to_prob()
        assert np.allclose(result, probs)

    def test_to_bitstream_shape(self):
        """Test bitstream conversion produces correct shape."""
        probs = np.array([0.3, 0.7])
        stream = TensorStream.from_prob(probs)

        bitstream = stream.to_bitstream(length=512)

        assert bitstream.shape == (2, 512)
        assert bitstream.dtype == np.uint8

    def test_bitstream_probability_approximation(self):
        """Test that bitstream mean approximates original probability."""
        np.random.seed(42)
        probs = np.array([0.25, 0.75])
        stream = TensorStream.from_prob(probs)

        bitstream = stream.to_bitstream(length=10000)
        recovered_probs = np.mean(bitstream, axis=-1)

        # Should be close with high sample count
        assert np.allclose(recovered_probs, probs, atol=0.02)

    def test_to_quantum_shape(self):
        """Test quantum conversion produces correct state vector shape."""
        probs = np.array([0.0, 0.5, 1.0])
        stream = TensorStream.from_prob(probs)

        quantum = stream.to_quantum()

        assert quantum.shape == (3, 2)
        assert np.iscomplexobj(quantum)

    def test_quantum_normalization(self):
        """Test quantum states are properly normalized."""
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        stream = TensorStream.from_prob(probs)

        quantum = stream.to_quantum()

        # |alpha|^2 + |beta|^2 should equal 1
        norms = np.abs(quantum[..., 0])**2 + np.abs(quantum[..., 1])**2
        assert np.allclose(norms, 1.0)

    def test_bitstream_to_prob_conversion(self):
        """Test converting bitstream back to probability."""
        bitstream = np.array([
            [1, 1, 0, 0, 1, 1, 0, 0],  # 0.5
            [1, 1, 1, 1, 1, 1, 0, 0],  # 0.75
        ], dtype=np.uint8)

        stream = TensorStream(data=bitstream, domain='bitstream')
        probs = stream.to_prob()

        assert np.allclose(probs, [0.5, 0.75])

    def test_quantum_to_prob_born_rule(self):
        """Test quantum to probability uses Born rule."""
        # Create quantum state directly
        # |psi> = alpha|0> + beta|1>  -> p = |beta|^2
        quantum = np.array([
            [1.0, 0.0],  # p = 0
            [np.sqrt(0.5), np.sqrt(0.5)],  # p = 0.5
            [0.0, 1.0],  # p = 1
        ], dtype=complex)

        stream = TensorStream(data=quantum, domain='quantum')
        probs = stream.to_prob()

        assert np.allclose(probs, [0.0, 0.5, 1.0])


# =============================================================================
# CognitiveOrchestrator Tests
# =============================================================================

class MockModule:
    """Mock module for testing orchestrator."""
    def __init__(self, transform_fn):
        self.transform_fn = transform_fn
        self.weights = np.array([1.0, 2.0])

    def forward(self, x):
        return self.transform_fn(x)

    def get_weights(self):
        return self.weights


class MockStepModule:
    """Mock module using step interface."""
    def __init__(self, factor=2.0):
        self.factor = factor
        self.v = 0.0

    def step(self, x):
        return x * self.factor

    def get_state(self):
        return {'v': self.v}


class TestCognitiveOrchestrator:
    """Tests for CognitiveOrchestrator."""

    def test_register_module(self):
        """Test registering modules."""
        orch = CognitiveOrchestrator()
        module = MockModule(lambda x: x)

        orch.register_module('test', module)

        assert 'test' in orch.modules
        assert orch.modules['test'] is module

    def test_set_attention(self):
        """Test setting attention focus."""
        orch = CognitiveOrchestrator()
        orch.register_module('sensor', MockModule(lambda x: x))

        orch.set_attention('sensor')

        assert orch.attention_focus == 'sensor'

    def test_set_attention_invalid_module(self):
        """Test setting attention on non-existent module."""
        orch = CognitiveOrchestrator()

        orch.set_attention('nonexistent')

        assert orch.attention_focus is None

    def test_execute_pipeline_single_module(self):
        """Test executing pipeline with single module."""
        orch = CognitiveOrchestrator()
        orch.register_module('double', MockModule(lambda x: x * 2))

        input_stream = TensorStream.from_prob(np.array([0.25, 0.5]))
        output_stream = orch.execute_pipeline(['double'], input_stream)

        expected = np.array([0.5, 1.0])
        assert np.allclose(output_stream.to_prob(), expected)

    def test_execute_pipeline_multiple_modules(self):
        """Test executing pipeline with multiple modules."""
        orch = CognitiveOrchestrator()
        orch.register_module('add_half', MockModule(lambda x: x + 0.1))
        orch.register_module('double', MockModule(lambda x: x * 2))

        input_stream = TensorStream.from_prob(np.array([0.2]))
        output_stream = orch.execute_pipeline(['add_half', 'double'], input_stream)

        # (0.2 + 0.1) * 2 = 0.6
        expected = np.array([0.6])
        assert np.allclose(output_stream.to_prob(), expected)

    def test_execute_pipeline_missing_module(self):
        """Test pipeline skips missing modules."""
        orch = CognitiveOrchestrator()
        orch.register_module('exists', MockModule(lambda x: x * 2))

        input_stream = TensorStream.from_prob(np.array([0.3]))
        output_stream = orch.execute_pipeline(['missing', 'exists'], input_stream)

        # Only 'exists' should run
        expected = np.array([0.6])
        assert np.allclose(output_stream.to_prob(), expected)

    def test_execute_pipeline_step_module(self):
        """Test pipeline with step-based modules."""
        orch = CognitiveOrchestrator()
        orch.register_module('stepper', MockStepModule(factor=3.0))

        input_stream = TensorStream.from_prob(np.array([0.1, 0.2]))
        output_stream = orch.execute_pipeline(['stepper'], input_stream)

        expected = np.array([0.3, 0.6])
        assert np.allclose(output_stream.to_prob(), expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
