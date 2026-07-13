# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware session-validation tests

"""Fail-closed and transactional tests for session orchestration."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    AERToSCConverter,
    ArtifactRejector,
    BioHybridSession,
    BiologicalSTDP,
    CultureHealth,
    DetectedSpike,
    HomeostaticPlasticity,
    LatencyBudget,
    MEAConfig,
    MEAToAERTranscoder,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
)


def _parts() -> dict[str, Any]:
    config = MEAConfig(num_channels=2)
    return {
        "mea_config": config,
        "detector": SpikeDetector(config),
        "transcoder": MEAToAERTranscoder(),
        "sc_converter": AERToSCConverter(num_neurons=2),
        "opto_encoder": SCToOptoEncoder(),
    }


class TestSessionConstructionValidation:
    @pytest.mark.parametrize(
        "field",
        ["mea_config", "detector", "transcoder", "sc_converter", "opto_encoder"],
    )
    def test_core_component_types_are_enforced(self, field: str) -> None:
        parts = _parts()
        parts[field] = object()
        with pytest.raises(TypeError):
            BioHybridSession(**parts)

    @pytest.mark.parametrize(
        "field",
        [
            "stdp",
            "health_monitor",
            "artifact_rejector",
            "pharm_model",
            "latency_budget",
            "homeostatic",
            "sorter",
            "zenith_core",
        ],
    )
    def test_optional_component_types_are_enforced(self, field: str) -> None:
        parts = _parts()
        parts.update(
            {
                "stdp": BiologicalSTDP(),
                "health_monitor": CultureHealth(),
                "artifact_rejector": ArtifactRejector(),
                "pharm_model": PharmModel(),
                "latency_budget": LatencyBudget(),
                "homeostatic": HomeostaticPlasticity(),
                "sorter": SpikeSorter(),
                "zenith_core": cast(
                    Any, type("Zenith", (), {"step_from_bio_rates": lambda *_: None})()
                ),
            }
        )
        parts[field] = object()
        with pytest.raises(TypeError):
            BioHybridSession(**parts)

    def test_detector_configuration_must_match(self) -> None:
        parts = _parts()
        parts["detector"] = SpikeDetector(MEAConfig(num_channels=3))
        with pytest.raises(ValueError, match="detector.config"):
            BioHybridSession(**parts)

    def test_converter_and_channel_map_must_cover_mea(self) -> None:
        parts = _parts()
        parts["sc_converter"] = AERToSCConverter(num_neurons=1)
        with pytest.raises(ValueError, match="cover every MEA channel"):
            BioHybridSession(**parts)

        parts = _parts()
        parts["transcoder"] = MEAToAERTranscoder(channel_map={0: 3})
        with pytest.raises(ValueError, match="channel_map targets"):
            BioHybridSession(**parts)

        parts = _parts()
        parts["transcoder"] = MEAToAERTranscoder(channel_map={0: 1})
        BioHybridSession(**parts)


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


def test_type_casts_do_not_weaken_runtime_checks() -> None:
    """Keep wrong-type construction explicit under strict static typing."""
    parts = _parts()
    parts["mea_config"] = cast(Any, object())
    with pytest.raises(TypeError):
        BioHybridSession(**parts)
