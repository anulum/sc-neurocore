# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSessionConstructionValidation from former test_validation_session.py

"""Focused suite: TestSessionConstructionValidation from former test_validation_session.py."""

from __future__ import annotations

from tests.test_bioware.validation_session_support import *  # noqa: F403

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
