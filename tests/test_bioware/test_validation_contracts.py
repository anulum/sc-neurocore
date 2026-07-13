# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware boundary-validation tests

"""Fail-closed tests for Bioware value and component boundaries."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    AEREvent,
    AERToSCConverter,
    ArtifactRejector,
    BioHybridFrameResult,
    DetectedSpike,
    LatencyBudget,
    LFPBand,
    MEAConfig,
    MEAToAERTranscoder,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
    CultureHealth,
    extract_lfp_power,
)
from sc_neurocore.bioware.bioware_validation import (
    require_finite,
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
    require_positive_int,
    validate_binary_bitstream,
    validate_voltage_matrix,
)


class TestScalarValidation:
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_require_finite_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            require_finite(value, "value")

    def test_scalar_sign_and_integer_guards(self) -> None:
        with pytest.raises(ValueError, match="value must be >= 0"):
            require_nonnegative(-1.0, "value")
        with pytest.raises(ValueError, match="value must be > 0"):
            require_positive(0.0, "value")
        with pytest.raises(TypeError, match="value must be an integer"):
            require_nonnegative_int(cast(Any, True), "value")
        with pytest.raises(ValueError, match="value must be >= 0"):
            require_nonnegative_int(-1, "value")
        with pytest.raises(ValueError, match="value must be > 0"):
            require_positive_int(0, "value")

    @pytest.mark.parametrize("value", [True, np.bool_(False), "1.0"])
    def test_scalar_guards_reject_non_real_values(self, value: Any) -> None:
        with pytest.raises(TypeError, match="value must be a real number"):
            require_finite(cast(Any, value), "value")

    @pytest.mark.parametrize(
        ("matrix", "error"),
        [
            (cast(Any, [[1.0]]), TypeError),
            (np.array([["x"]], dtype=object), TypeError),
            (np.ones(2), ValueError),
            (np.empty((0, 1)), ValueError),
            (np.empty((1, 0)), ValueError),
            (np.array([[float("inf")]]), ValueError),
        ],
    )
    def test_voltage_matrix_rejects_invalid_inputs(
        self,
        matrix: Any,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            validate_voltage_matrix(matrix)

    def test_voltage_matrix_rejects_channel_mismatch(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            validate_voltage_matrix(np.ones((2, 1)), expected_channels=2)

    @pytest.mark.parametrize(
        ("bitstream", "allow_empty", "error"),
        [
            (cast(Any, [0, 1]), False, TypeError),
            (np.ones((1, 2)), False, ValueError),
            (np.array([], dtype=np.uint8), False, ValueError),
            (np.array(["x"]), False, TypeError),
            (np.array([float("nan")]), False, ValueError),
            (np.array([0, 2]), False, ValueError),
        ],
    )
    def test_bitstream_rejects_invalid_inputs(
        self,
        bitstream: Any,
        allow_empty: bool,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            validate_binary_bitstream(bitstream, name="bits", allow_empty=allow_empty)


class TestDataContractValidation:
    def test_mea_layout_type_is_explicit(self) -> None:
        with pytest.raises(TypeError, match="layout must be a MEALayout"):
            MEAConfig(layout=cast(Any, "60ch"))
        with pytest.raises(TypeError, match="layout must be a MEALayout"):
            MEAConfig.from_layout(cast(Any, "60ch"))

    @pytest.mark.parametrize(
        "waveform",
        [
            cast(Any, [1.0]),
            np.array([], dtype=float),
            np.ones((1, 1)),
            np.array(["x"]),
            np.array([float("inf")]),
        ],
    )
    def test_detected_spike_rejects_invalid_waveforms(self, waveform: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            DetectedSpike(0, 0.0, -1.0, waveform=waveform)

    def test_aer_packet_rejects_out_of_range_fields(self) -> None:
        with pytest.raises(ValueError, match="timestamp must fit"):
            AEREvent(0, 0x10000)
        with pytest.raises(TypeError, match="valid must be a bool"):
            AEREvent(0, 0, valid=cast(Any, 1))
        with pytest.raises(ValueError, match="weight must fit"):
            AEREvent(0, 0, weight=0x10000)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_spikes", 1),
            ("num_aer_events", 1),
            ("num_bitstreams", 1),
            ("num_opto_pulses", 1),
        ],
    )
    def test_frame_result_rejects_payload_count_mismatch(
        self,
        field: str,
        value: int,
    ) -> None:
        payload: dict[str, Any] = {
            "round": 1,
            "num_spikes": 0,
            "num_aer_events": 0,
            "num_bitstreams": 0,
            "num_opto_pulses": 0,
            "latency_us": 1.0,
            "health": {},
            "spikes": [],
            "aer_events": [],
            "bitstreams": {},
            "opto_pulses": [],
        }
        payload[field] = value
        with pytest.raises(ValueError, match="must equal"):
            BioHybridFrameResult(**payload)


class TestComponentBoundaryValidation:
    def test_acquisition_rejects_invalid_configuration_and_frames(self) -> None:
        with pytest.raises(TypeError, match="config must be an MEAConfig"):
            SpikeDetector(cast(Any, object()))
        detector = SpikeDetector(MEAConfig(num_channels=1, sample_rate_hz=1_000.0))
        with pytest.raises(ValueError, match="shorter than one sample"):
            detector.detect(np.ones((10, 1)), snippet_ms=0.1)

    def test_sorter_rejects_inconsistent_and_assignment_waveforms(self) -> None:
        spikes = [
            DetectedSpike(0, 0.0, -1.0, waveform=np.ones(2)),
            DetectedSpike(0, 0.1, -1.0, waveform=np.ones(3)),
        ]
        with pytest.raises(ValueError, match="same shape"):
            SpikeSorter(num_units=2).fit(spikes)

        class _PCA:
            n_features_in_ = 2

        sorter = SpikeSorter()
        sorter._pca = _PCA()
        sorter._kmeans = object()
        with pytest.raises(ValueError, match="expected 2"):
            sorter.assign([DetectedSpike(0, 0.0, -1.0, waveform=np.ones(1))])

    def test_artifact_rejector_requires_stimulus_inside_frame(self) -> None:
        with pytest.raises(ValueError, match="inside the voltage frame"):
            ArtifactRejector().blank(np.ones((10, 1)), [1.0], 10.0)

    def test_analysis_rejects_invalid_health_inputs(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            CultureHealth(min_firing_rate_hz=1.0, max_firing_rate_hz=1.0)
        health = CultureHealth()
        for counts in (
            cast(Any, [1]),
            np.empty(0),
            np.ones((1, 1)),
            np.array(["x"]),
            np.array([-1.0]),
            np.array([float("nan")]),
        ):
            with pytest.raises((TypeError, ValueError)):
                health.assess(counts, 1.0)

    def test_lfp_and_latency_configuration_is_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            LFPBand(" ", 1.0, 2.0)
        with pytest.raises(ValueError, match="must exceed"):
            LFPBand("bad", 2.0, 2.0)
        data = np.ones((8, 1))
        with pytest.raises(ValueError, match="bands must not be empty"):
            extract_lfp_power(data, 100.0, bands=[])
        with pytest.raises(TypeError, match="LFPBand"):
            extract_lfp_power(data, 100.0, bands=cast(Any, [object()]))
        band = LFPBand("same", 1.0, 2.0)
        with pytest.raises(ValueError, match="duplicate"):
            extract_lfp_power(data, 100.0, bands=[band, band])
        with pytest.raises(ValueError, match="history latency_us"):
            LatencyBudget(history=[-1.0])
        with pytest.raises(ValueError, match="violations cannot exceed"):
            LatencyBudget(history=[], violations=1)

    def test_encoding_rejects_epoch_window_and_lfsr_violations(self) -> None:
        transcoder = MEAToAERTranscoder()
        with pytest.raises(ValueError, match="precedes"):
            transcoder.transcode([DetectedSpike(0, 0.0, -1.0)], t_start_s=0.1)
        with pytest.raises(ValueError, match="16-bit"):
            transcoder.transcode([DetectedSpike(0, 1.0, -1.0)])
        with pytest.raises(ValueError, match="lfsr_seed"):
            AERToSCConverter(lfsr_seed=0x10000)
        converter = AERToSCConverter(window_ticks=10, num_neurons=1)
        with pytest.raises(ValueError, match="neuron_id"):
            converter.convert([AEREvent(1, 0)])
        with pytest.raises(ValueError, match="timestamp"):
            converter.convert([AEREvent(0, 10)])
        with pytest.raises(ValueError, match="probability"):
            converter._lfsr_encode(2.0, 0)
        with pytest.raises(ValueError, match="neuron_id"):
            converter._lfsr_encode(0.5, 1)
        with pytest.raises(ValueError, match="max_pulse_ms"):
            SCToOptoEncoder(min_pulse_ms=2.0, max_pulse_ms=1.0)
