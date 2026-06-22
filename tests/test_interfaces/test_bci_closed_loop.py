# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Closed-loop BCI HIL template tests

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.interfaces.bci_closed_loop import (
    ClosedLoopBCIConfig,
    ClosedLoopBCITemplate,
    ImplantEmulator,
    RateSpikeDecoder,
)


def _waveform() -> np.ndarray:
    data = np.zeros((96, 4), dtype=np.float32)
    data[12, 0] = -25.0
    data[40, 1] = -30.0
    data[72, 3] = -35.0
    return data


def test_closed_loop_processes_waveform_to_aer_feedback_and_telemetry() -> None:
    template = ClosedLoopBCITemplate(
        ClosedLoopBCIConfig(
            n_channels=4,
            sampling_rate_hz=30_000,
            threshold_sigma=4.0,
            snippet_samples=16,
            waveform_mode="spike",
        )
    )

    result = template.process_window(_waveform(), window_start_us=500)

    assert result.waveform.n_channels == 4
    assert result.waveform.n_spikes_detected == 3
    assert result.spike_raster.shape == (96, 4)
    assert int(result.spike_raster.sum()) == 3
    assert result.aer.n_events == 3
    assert result.aer_payload.startswith(b"AERX")
    assert result.feedback.timestamp_us == 500
    assert result.feedback.active_count == 3
    assert result.telemetry["total_ticks"] == 2
    assert result.telemetry["layers"]["implant_input"]["spike_count"] == 3
    assert result.telemetry["layers"]["implant_feedback"]["spike_count"] == 3


def test_rate_decoder_returns_rates_per_channel() -> None:
    decoder = RateSpikeDecoder(sampling_rate_hz=1_000)
    raster = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int8)

    rates = decoder.decode(raster)

    assert np.allclose(rates, np.array([500.0, 250.0]))


def test_implant_emulator_clips_feedback_and_records_frames() -> None:
    emulator = ImplantEmulator(gain=2.0, max_feedback=1.0)

    frame = emulator.apply_feedback(np.array([0.25, 2.0, -2.0]), timestamp_us=10)

    assert frame.values == (0.5, 1.0, -1.0)
    assert frame.active_count == 3
    assert emulator.frames == [frame]


def test_closed_loop_rejects_channel_mismatch() -> None:
    template = ClosedLoopBCITemplate(ClosedLoopBCIConfig(n_channels=3))

    with pytest.raises(ValueError, match="expected 3"):
        template.process_window(_waveform())


def test_closed_loop_rejects_non_matrix_waveform() -> None:
    template = ClosedLoopBCITemplate(ClosedLoopBCIConfig(n_channels=4))

    with pytest.raises(ValueError, match="shape"):
        template.process_window(np.zeros(4, dtype=np.float32))


def test_closed_loop_rejects_empty_window() -> None:
    template = ClosedLoopBCITemplate(ClosedLoopBCIConfig(n_channels=4))

    with pytest.raises(ValueError, match="at least one sample"):
        template.process_window(np.zeros((0, 4), dtype=np.float32))


def test_closed_loop_fails_closed_when_decoder_is_cleared() -> None:
    # __post_init__ always installs a decoder and feedback sink; if either is
    # externally nulled the template must fail closed rather than dereference None.
    template = ClosedLoopBCITemplate(ClosedLoopBCIConfig(n_channels=4))
    template.decoder = None

    with pytest.raises(RuntimeError, match="was not initialised"):
        template.process_window(_waveform())


def test_rate_decoder_rejects_non_matrix_raster() -> None:
    decoder = RateSpikeDecoder(sampling_rate_hz=1_000)

    with pytest.raises(ValueError, match="samples, channels"):
        decoder.decode(np.zeros(5, dtype=np.float32))
