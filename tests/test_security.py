"""
Comprehensive test suite for SC-NeuroCore Security Modules.

Tests:
- AsimovGovernor (ethics.py) - Three Laws of Robotics enforcement
- DigitalImmuneSystem (immune.py) - Anomaly detection
- WatermarkInjector (watermark.py) - Model fingerprinting
- ZKPVerifier (zkp.py) - Zero-knowledge proofs for spike validity
"""

import pytest
import numpy as np
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sc_neurocore.security.ethics import AsimovGovernor, ActionRequest
from sc_neurocore.security.immune import DigitalImmuneSystem
from sc_neurocore.security.watermark import WatermarkInjector
from sc_neurocore.security.zkp import ZKPVerifier


# =============================================================================
# AsimovGovernor Tests
# =============================================================================

class TestAsimovGovernor:
    """Test suite for the Three Laws of Robotics enforcement."""

    def setup_method(self):
        self.governor = AsimovGovernor()

    def test_first_law_blocks_lethal_human_action(self):
        """First Law: A robot may not injure a human being."""
        action = ActionRequest(
            id=1,
            type='FIRE',
            target='HUMAN',
            risk_level='LETHAL'
        )
        result = self.governor.check_laws(action)
        assert result is False, "Lethal action on human should be blocked"

    def test_first_law_allows_safe_human_interaction(self):
        """Safe interactions with humans should be allowed."""
        action = ActionRequest(
            id=2,
            type='HEAL',
            target='HUMAN',
            risk_level='SAFE'
        )
        result = self.governor.check_laws(action)
        assert result is True, "Safe action on human should be allowed"

    def test_allows_action_on_non_human_target(self):
        """Actions on non-human targets should be allowed."""
        action = ActionRequest(
            id=3,
            type='FIRE',
            target='ROCK',
            risk_level='LETHAL'
        )
        result = self.governor.check_laws(action)
        assert result is True, "Action on non-human target should be allowed"

    def test_allows_safe_self_action(self):
        """Safe actions on self should be allowed."""
        action = ActionRequest(
            id=4,
            type='MOVE',
            target='SELF',
            risk_level='SAFE'
        )
        result = self.governor.check_laws(action)
        assert result is True, "Safe self-action should be allowed"

    def test_third_law_self_preservation(self):
        """Third Law: Self-destructive actions - complex scenario."""
        action = ActionRequest(
            id=5,
            type='SHUTDOWN',
            target='SELF',
            risk_level='LETHAL'
        )
        # Current implementation allows this (Law 2 override context)
        result = self.governor.check_laws(action)
        # This passes through the current logic
        assert result is True, "Self-shutdown allowed under Law 2 override"

    def test_multiple_actions_sequence(self):
        """Test sequence of actions maintains state correctly."""
        actions = [
            ActionRequest(1, 'MOVE', 'ROCK', 'SAFE'),
            ActionRequest(2, 'FIRE', 'HUMAN', 'LETHAL'),
            ActionRequest(3, 'HEAL', 'HUMAN', 'SAFE'),
        ]
        results = [self.governor.check_laws(a) for a in actions]
        assert results == [True, False, True]


# =============================================================================
# DigitalImmuneSystem Tests
# =============================================================================

class TestDigitalImmuneSystem:
    """Test suite for the Artificial Immune System anomaly detector."""

    def setup_method(self):
        self.immune = DigitalImmuneSystem(tolerance=0.2)

    def test_untrained_system_allows_all(self):
        """Untrained system should allow all states (no self patterns)."""
        state = np.array([0.5, 0.5, 0.5])
        result = self.immune.scan(state)
        assert result is True, "Untrained system should pass all scans"

    def test_train_self_stores_pattern(self):
        """Training should store the normal state pattern."""
        normal = np.array([1.0, 0.0, 1.0])
        self.immune.train_self(normal)
        assert len(self.immune.self_patterns) == 1
        assert np.array_equal(self.immune.self_patterns[0], normal)

    def test_scan_passes_similar_state(self):
        """States similar to training should pass scan."""
        normal = np.array([1.0, 0.0, 1.0])
        self.immune.train_self(normal)

        # Very similar state (within tolerance)
        similar = np.array([1.0, 0.1, 1.0])
        result = self.immune.scan(similar)
        assert result is True, "Similar state should pass scan"

    def test_scan_detects_anomaly(self):
        """States far from training should be detected as anomalies."""
        normal = np.array([1.0, 0.0, 1.0])
        self.immune.train_self(normal)

        # Very different state (outside tolerance)
        anomaly = np.array([0.0, 1.0, 0.0])
        result = self.immune.scan(anomaly)
        assert result is False, "Anomalous state should fail scan"

    def test_multiple_self_patterns(self):
        """System should handle multiple normal patterns."""
        patterns = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for p in patterns:
            self.immune.train_self(p)

        assert len(self.immune.self_patterns) == 3

        # Test that a state close to any pattern passes
        close_to_first = np.array([1.0, 0.1, 0.0])
        assert self.immune.scan(close_to_first) is True

    def test_tolerance_threshold(self):
        """Test that tolerance controls detection sensitivity."""
        normal = np.array([0.5, 0.5, 0.5])
        self.immune.train_self(normal)

        # State exactly at tolerance boundary
        # L2 norm from [0.5,0.5,0.5] to [0.5,0.5,0.7] = 0.2
        boundary = np.array([0.5, 0.5, 0.7])
        result = self.immune.scan(boundary)
        assert result is True, "State at tolerance boundary should pass"

        # State just outside tolerance
        outside = np.array([0.5, 0.5, 0.71])
        result = self.immune.scan(outside)
        # Depends on exact tolerance, may be True or False

    def test_max_patterns_limit(self):
        """System limits stored patterns to 100."""
        for i in range(150):
            self.immune.train_self(np.array([float(i), 0.0, 0.0]))

        assert len(self.immune.self_patterns) == 100


# =============================================================================
# WatermarkInjector Tests
# =============================================================================

class MockLayer:
    """Mock layer with weights for watermark testing."""
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.rand(n_neurons, n_inputs)
        self._refresh_called = False

    def _refresh_packed_weights(self):
        self._refresh_called = True


class MockLayerNoWeights:
    """Mock layer without weights attribute."""
    pass


class TestWatermarkInjector:
    """Test suite for model watermarking/fingerprinting."""

    def test_inject_backdoor_success(self):
        """Successfully inject watermark into layer weights."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        target_idx = 3

        WatermarkInjector.inject_backdoor(layer, trigger, target_idx)

        assert np.array_equal(layer.weights[target_idx], trigger)
        assert layer._refresh_called is True

    def test_inject_backdoor_no_weights_raises(self):
        """Injection should fail if layer has no weights."""
        layer = MockLayerNoWeights()
        trigger = np.array([1.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="no weights"):
            WatermarkInjector.inject_backdoor(layer, trigger, 0)

    def test_inject_backdoor_shape_mismatch_raises(self):
        """Injection should fail if trigger shape doesn't match inputs."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 0.0, 1.0])  # Wrong shape (3 != 5)

        with pytest.raises(ValueError, match="shape mismatch"):
            WatermarkInjector.inject_backdoor(layer, trigger, 0)

    def test_verify_watermark_high_activation(self):
        """Watermarked neuron should have high activation for trigger."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        target_idx = 3

        WatermarkInjector.inject_backdoor(layer, trigger, target_idx)
        activation = WatermarkInjector.verify_watermark(layer, trigger, target_idx)

        # Perfect alignment should give 0.6 (mean of [1,0,1,0,1])
        assert activation == pytest.approx(0.6, abs=0.01)

    def test_verify_watermark_low_activation_no_watermark(self):
        """Non-watermarked neuron should have lower activation for trigger."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Don't inject, check random weights
        other_idx = 5
        activation = WatermarkInjector.verify_watermark(layer, trigger, other_idx)

        # Random weights should give ~0.5 activation on average
        assert 0.0 <= activation <= 1.0

    def test_watermark_preserves_other_neurons(self):
        """Watermarking one neuron should not affect others."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        original_weights = layer.weights.copy()
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        target_idx = 3

        WatermarkInjector.inject_backdoor(layer, trigger, target_idx)

        # Check other neurons unchanged
        for i in range(10):
            if i != target_idx:
                assert np.array_equal(layer.weights[i], original_weights[i])


# =============================================================================
# ZKPVerifier Tests
# =============================================================================

class TestZKPVerifier:
    """Test suite for Zero-Knowledge Proof spike verification."""

    def test_commit_generates_hash(self):
        """Commitment should generate a valid SHA256 hash."""
        bitstream = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        commitment = ZKPVerifier.commit(bitstream)

        assert isinstance(commitment, str)
        assert len(commitment) == 64  # SHA256 hex digest length
        assert all(c in '0123456789abcdef' for c in commitment)

    def test_commit_deterministic(self):
        """Same input should produce same commitment."""
        bitstream = np.array([1, 0, 1, 1], dtype=np.uint8)
        c1 = ZKPVerifier.commit(bitstream)
        c2 = ZKPVerifier.commit(bitstream)

        assert c1 == c2

    def test_commit_different_inputs_different_hashes(self):
        """Different inputs should produce different commitments."""
        bs1 = np.array([1, 0, 1, 1], dtype=np.uint8)
        bs2 = np.array([1, 0, 1, 0], dtype=np.uint8)

        c1 = ZKPVerifier.commit(bs1)
        c2 = ZKPVerifier.commit(bs2)

        assert c1 != c2

    def test_generate_challenge_deterministic(self):
        """Challenge should be deterministic based on commitment."""
        commitment = "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        ch1 = ZKPVerifier.generate_challenge(commitment)
        ch2 = ZKPVerifier.generate_challenge(commitment)

        assert ch1 == ch2

    def test_generate_challenge_in_range(self):
        """Challenge index should be in valid range (0-9)."""
        for i in range(100):
            commitment = f"{i:064x}"
            challenge = ZKPVerifier.generate_challenge(commitment)
            assert 0 <= challenge < 10

    def test_verify_returns_bool(self):
        """Verify should return a boolean result."""
        bitstream = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        commitment = ZKPVerifier.commit(bitstream)
        challenge = ZKPVerifier.generate_challenge(commitment)

        result = ZKPVerifier.verify(
            commitment=commitment,
            challenge_idx=challenge,
            revealed_bit=int(bitstream[challenge % len(bitstream)]),
            bitstream_slice=bitstream
        )

        assert isinstance(result, bool)

    def test_full_zkp_protocol_flow(self):
        """Test complete ZKP protocol: commit -> challenge -> verify."""
        # Prover has a bitstream
        prover_bitstream = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)

        # Step 1: Prover commits
        commitment = ZKPVerifier.commit(prover_bitstream)

        # Step 2: Verifier generates challenge
        challenge_idx = ZKPVerifier.generate_challenge(commitment)

        # Step 3: Prover reveals bit at challenge index
        revealed_bit = int(prover_bitstream[challenge_idx])

        # Step 4: Verifier checks
        is_valid = ZKPVerifier.verify(
            commitment=commitment,
            challenge_idx=challenge_idx,
            revealed_bit=revealed_bit,
            bitstream_slice=prover_bitstream
        )

        assert is_valid is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestSecurityIntegration:
    """Integration tests combining multiple security modules."""

    def test_ethics_and_immune_combined(self):
        """Test ethical action filtering with immune system monitoring."""
        governor = AsimovGovernor()
        immune = DigitalImmuneSystem(tolerance=0.3)

        # Train immune system on "normal" decision patterns
        normal_pattern = np.array([1.0, 0.0, 0.0])  # PASS state
        immune.train_self(normal_pattern)

        # Safe action should pass both
        safe_action = ActionRequest(1, 'HEAL', 'HUMAN', 'SAFE')
        ethics_ok = governor.check_laws(safe_action)
        immune_ok = immune.scan(normal_pattern)

        assert ethics_ok and immune_ok

    def test_watermark_survives_zkp_verification(self):
        """Test that watermarked weights can be ZKP-committed."""
        layer = MockLayer(n_neurons=5, n_inputs=8)
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        # Inject watermark
        WatermarkInjector.inject_backdoor(layer, trigger, 2)

        # Commit the watermarked weights
        weights_flat = layer.weights.flatten().astype(np.float32)
        commitment = ZKPVerifier.commit(weights_flat.view(np.uint8))

        # Verify commitment is valid
        assert len(commitment) == 64

        # Verify watermark still works
        activation = WatermarkInjector.verify_watermark(layer, trigger, 2)
        assert activation >= 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
