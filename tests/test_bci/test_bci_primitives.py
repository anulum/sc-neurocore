# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI Primitives Tests

import unittest

import numpy as np
import pytest

from sc_neurocore.bci_studio.bci_primitives import (
    BCIFeedbackCommand,
    BCIFrame,
    BCIPrimitiveConfig,
    BCIClosedLoopEngine,
    BCIClosedLoopPrimitive,
    SCHEMA_VERSION,
)


class TestBCIClosedLoopEngine(unittest.TestCase):
    def setUp(self):
        self.engine = BCIClosedLoopEngine(channels=64)

    def test_process_returns_dict(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertIn("command", result)
        self.assertIn("latency_ms", result)
        self.assertIn("spikes", result)

    def test_command_is_binary(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertIn(result["command"], (0, 1))

    def test_latency_positive(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertGreater(result["latency_ms"], 0.0)

    def test_spikes_count_non_negative(self):
        raw = np.random.randn(64).astype(np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.5)
        self.assertGreaterEqual(result["spikes"], 0)

    def test_zero_input_produces_no_spikes(self):
        raw = np.zeros(64, dtype=np.float32)
        result = self.engine.process_bci_frame(raw, reward=0.0)
        self.assertEqual(result["spikes"], 0)
        self.assertEqual(result["command"], 0)

    def test_channels_attribute(self):
        self.assertEqual(self.engine.channels, 64)

    def test_weights_shape(self):
        self.assertEqual(self.engine.weights.shape, (64,))


def test_primitive_processes_matrix_frame_with_trace_and_packet() -> None:
    primitive = BCIClosedLoopPrimitive(
        BCIPrimitiveConfig(
            channels=4,
            sampling_rate_hz=1_000,
            threshold_sigma=4.0,
            refractory_samples=4,
            command_threshold_hz=10.0,
            weight_decay=1.0,
            latency_budget_ms=1_000.0,
            enable_native_learning=False,
        )
    )
    samples = np.zeros((40, 4), dtype=np.float32)
    samples[10, 0] = -25.0
    samples[20, 1] = -25.0
    samples[30, 3] = -25.0

    result = primitive.process_frame(BCIFrame(samples=samples, reward=0.0, timestamp_us=123))
    decoded = BCIFeedbackCommand.from_packet(result.feedback_packet)

    assert result.trace.schema_version == SCHEMA_VERSION
    assert result.trace.input_shape == (40, 4)
    assert result.trace.spike_count == 3
    assert result.trace.active_channels == 3
    assert result.trace.latency_budget_met
    assert result.command.command == BCIFeedbackCommand.COMMAND_STIM
    assert result.command.safety_limited is True
    assert decoded.command == result.command.command
    assert decoded.timestamp_us == 123
    assert len(result.feedback_packet) == 24


def test_primitive_reward_updates_weights_without_unbounded_growth() -> None:
    primitive = BCIClosedLoopPrimitive(
        BCIPrimitiveConfig(
            channels=4,
            learning_rate=0.5,
            weight_decay=1.0,
            min_weight=0.5,
            max_weight=1.25,
            enable_native_learning=False,
        )
    )
    raw = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float32)

    primitive.process_frame(BCIFrame(samples=raw, reward=1.0))

    assert np.all(primitive.weights <= 1.25)
    assert np.all(primitive.weights >= 0.5)
    assert primitive.weights[1] > primitive.weights[0]


def test_primitive_rejects_bad_shape_and_non_finite_values() -> None:
    primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=4))

    with pytest.raises(ValueError, match="expected 4"):
        primitive.process_frame(BCIFrame(samples=np.zeros(3, dtype=np.float32)))

    bad = np.zeros((4, 4), dtype=np.float32)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        primitive.process_frame(BCIFrame(samples=bad))


def test_config_rejects_invalid_latency_budget() -> None:
    with pytest.raises(ValueError, match="latency_budget_ms"):
        BCIPrimitiveConfig(latency_budget_ms=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0}, "channels must be positive"),
        ({"sampling_rate_hz": 0}, "sampling_rate_hz must be positive"),
        ({"threshold_sigma": 0.0}, "threshold_sigma must be positive"),
        ({"legacy_derivative_threshold": 0.0}, "legacy_derivative_threshold must be positive"),
        ({"refractory_samples": 0}, "refractory_samples must be >= 1"),
        ({"command_threshold_hz": -1.0}, "command_threshold_hz must be non-negative"),
        ({"legacy_active_fraction_threshold": 1.5}, "legacy_active_fraction_threshold must be in"),
        ({"learning_rate": -0.1}, "learning_rate must be non-negative"),
        ({"weight_decay": 0.0}, "weight_decay must be in"),
        ({"min_weight": 0.0}, "min_weight must be positive"),
        ({"max_feedback_amplitude": 0.0}, "max_feedback_amplitude must be positive"),
    ],
)
def test_config_rejects_each_invalid_field(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BCIPrimitiveConfig(**kwargs)


def test_from_packet_rejects_short_packet() -> None:
    with pytest.raises(ValueError, match="at least 24 bytes"):
        BCIFeedbackCommand.from_packet(b"\x00" * 10)


def test_from_packet_rejects_unknown_schema() -> None:
    cmd = BCIFeedbackCommand(command=1, channel=0, amplitude=0.5, timestamp_us=1, score=0.5)
    tampered = bytearray(cmd.to_packet())
    tampered[0] = 7  # overwrite the schema field with an unsupported version
    with pytest.raises(ValueError, match="unsupported BCI feedback packet schema"):
        BCIFeedbackCommand.from_packet(bytes(tampered))


def test_primitive_accepts_matching_initial_weights() -> None:
    weights = np.full(4, 2.0, dtype=np.float32)
    primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=4), initial_weights=weights)
    assert primitive.weights.shape == (4,)
    assert np.all(primitive.weights == 2.0)


def test_primitive_rejects_mismatched_initial_weights() -> None:
    with pytest.raises(ValueError, match="does not match"):
        BCIClosedLoopPrimitive(
            BCIPrimitiveConfig(channels=4),
            initial_weights=np.ones(3, dtype=np.float32),
        )


def test_primitive_honours_explicit_frame_id() -> None:
    primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=4))
    result = primitive.process_frame(
        BCIFrame(samples=np.zeros((2, 4), dtype=np.float32), frame_id=42)
    )
    assert result.trace.frame_id == 42
    # The counter advances past the explicit id so the next auto id does not collide.
    nxt = primitive.process_frame(BCIFrame(samples=np.zeros((2, 4), dtype=np.float32)))
    assert nxt.trace.frame_id == 43


def test_validate_samples_rejects_three_dimensional_frame() -> None:
    primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=4))
    with pytest.raises(ValueError, match=r"shape \(channels,\) or \(samples, channels\)"):
        primitive.process_frame(BCIFrame(samples=np.zeros((2, 2, 4), dtype=np.float32)))


def test_validate_samples_rejects_empty_matrix() -> None:
    primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=4))
    with pytest.raises(ValueError, match="at least one sample"):
        primitive.process_frame(BCIFrame(samples=np.zeros((0, 4), dtype=np.float32)))


def test_validate_samples_rejects_wrong_channel_count_matrix() -> None:
    primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=4))
    with pytest.raises(ValueError, match="expected 4"):
        primitive.process_frame(BCIFrame(samples=np.zeros((2, 5), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
