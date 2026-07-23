# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComponentBoundaryValidation from former test_validation_contracts.py

"""Focused suite: TestComponentBoundaryValidation from former test_validation_contracts.py."""

from __future__ import annotations

from tests.test_bioware.validation_contracts_support import *  # noqa: F403

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
