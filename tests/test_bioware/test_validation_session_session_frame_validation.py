# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSessionFrameValidation from former test_validation_session.py

"""Focused suite: TestSessionFrameValidation from former test_validation_session.py."""

from __future__ import annotations

from tests.test_bioware.validation_session_support import *  # noqa: F403


class TestSessionFrameValidation:
    def test_frame_must_fit_aer_epoch_without_advancing_round(self) -> None:
        session = BioHybridSession(**_parts())
        long_frame = np.ones((2_000, 2))
        with pytest.raises(ValueError, match="16-bit AER"):
            session.process_frame(long_frame)
        assert session.round_count == 0

    def test_frame_must_fit_converter_window_without_advancing_round(self) -> None:
        parts = _parts()
        parts["sc_converter"] = AERToSCConverter(window_ticks=10, num_neurons=2)
        session = BioHybridSession(**parts)
        with pytest.raises(ValueError, match="window_ticks"):
            session.process_frame(np.ones((2, 2)))
        assert session.round_count == 0

    def test_experiment_time_must_be_nonnegative(self) -> None:
        session = BioHybridSession(**_parts())
        with pytest.raises(ValueError, match="t_start_s"):
            session.process_frame(np.ones((1, 2)), t_start_s=-1.0)

    def test_health_accounting_ignores_mapped_non_mea_source_channel(self) -> None:
        class _MappedDetector(SpikeDetector):
            def detect(
                self,
                voltage_data: np.ndarray[Any, Any],
                snippet_ms: float = 2.0,
            ) -> list[DetectedSpike]:
                del voltage_data, snippet_ms
                return [DetectedSpike(channel=2, timestamp_s=0.0, amplitude_uv=-1.0)]

        parts = _parts()
        config = cast(MEAConfig, parts["mea_config"])
        parts["detector"] = _MappedDetector(config)
        parts["transcoder"] = MEAToAERTranscoder(channel_map={2: 0})
        result = BioHybridSession(**parts).process_frame(np.ones((1, 2)))

        assert result.health["active_channels"] == 0
