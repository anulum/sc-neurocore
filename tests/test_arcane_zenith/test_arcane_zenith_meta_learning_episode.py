# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetaLearningEpisode from former test_arcane_zenith.py

"""Focused suite: TestMetaLearningEpisode from former test_arcane_zenith.py."""

from __future__ import annotations

import pytest
from sc_neurocore.arcane_zenith import (
    create_arcane_neuron_with_zenith_plasticity,
)

pytest.importorskip("torch")


class TestMetaLearningEpisode:
    """Meta-learning episodes and their reasoning traces."""

    def test_episode_returns_deterministic_summary_and_trace(self) -> None:
        """An episode reports its steps, spikes, rate and one trace row per current."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        currents = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        result = core.run_meta_learning_episode(currents, reset_before=True)

        assert result["steps"] == len(currents)
        assert 0 <= result["spike_count"] <= len(currents)
        assert 0.0 <= result["spike_rate"] <= 1.0
        assert len(result["trace"]) == len(currents)
        assert result["trace"][0]["current"] == pytest.approx(currents[0])
        assert result["trace"][-1]["current"] == pytest.approx(currents[-1])

    def test_episode_trace_contains_required_contract_fields(self) -> None:
        """Every trace row has the contract fields within their ranges."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        result = core.run_meta_learning_episode([0.25, 0.5, 0.75], reset_before=True)

        required = {
            "current",
            "spike",
            "tau_deep",
            "surprise_baseline",
            "delta_conf",
            "lr_base",
            "novelty",
            "confidence",
            "identity_drift",
        }
        for item in result["trace"]:
            assert required.issubset(item.keys())
            assert 1000.0 <= float(item["tau_deep"]) <= 50000.0
            assert 0.01 <= float(item["surprise_baseline"]) <= 0.5
            assert 0.0 <= float(item["delta_conf"]) <= 1.0
            assert 0.001 <= float(item["lr_base"]) <= 0.1

    def test_episode_rejects_empty_currents(self) -> None:
        """An episode without currents is refused."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        with pytest.raises(ValueError, match="currents must be non-empty"):
            core.run_meta_learning_episode([])

    def test_compact_reasoning_trace_export(self) -> None:
        """The compact trace holds its fields within their ranges."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        core.run_meta_learning_episode([1.0, 2.0, 3.0], reset_before=True)
        trace = core.export_reasoning_trace()

        assert set(trace.keys()) == {
            "novelty",
            "confidence",
            "identity_drift",
            "tau_deep",
            "surprise_baseline",
            "delta_conf",
            "lr_base",
        }
        assert 1000.0 <= trace["tau_deep"] <= 50000.0
        assert 0.01 <= trace["surprise_baseline"] <= 0.5
        assert 0.0 <= trace["delta_conf"] <= 1.0
        assert 0.001 <= trace["lr_base"] <= 0.1
