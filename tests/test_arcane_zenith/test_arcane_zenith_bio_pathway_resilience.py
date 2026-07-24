# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBioPathwayResilience from former test_arcane_zenith.py

"""Focused suite: TestBioPathwayResilience from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403


class TestBioPathwayResilience:
    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_resilience_payload_contains_pathway_identity(self, core):
        payload = core.evaluate_bio_pathway_resilience(
            {2: 20.0, 0: 5.0, 1: 10.0},
            pathway_name="visual-cortex",
            bitstream_length=64,
            radiation_profile=RadiationProfile("test", 0.01, "pathway stress"),
            seed=12,
        )

        assert payload["layer_id"] == "bio:visual-cortex"
        assert payload["pathway_name"] == "visual-cortex"
        assert payload["pathway_channels"] == [0, 1, 2]
        assert payload["input_shape"] == [3, 64]
        assert payload["seed"] == 12

    def test_resilience_is_deterministic_for_same_seed(self, core):
        rates = {0: 8.0, 1: 16.0}
        first = core.evaluate_bio_pathway_resilience(
            rates,
            pathway_name="motor",
            bitstream_length=32,
            seed=99,
        )
        second = core.evaluate_bio_pathway_resilience(
            rates,
            pathway_name="motor",
            bitstream_length=32,
            seed=99,
        )

        assert first == second

    def test_resilience_empty_rates_falls_back_to_single_channel(self, core):
        payload = core.evaluate_bio_pathway_resilience(
            {},
            pathway_name="silent",
            bitstream_length=16,
            seed=7,
        )
        assert payload["input_shape"] == [1, 16]
        assert payload["nominal_probability"] == 0.0

    def test_resilience_rejects_invalid_arguments(self, core):
        with pytest.raises(ValueError, match="pathway_name"):
            core.evaluate_bio_pathway_resilience({0: 1.0}, pathway_name="")
        with pytest.raises(ValueError, match="bitstream_length"):
            core.evaluate_bio_pathway_resilience(
                {0: 1.0},
                pathway_name="ok",
                bitstream_length=0,
            )
