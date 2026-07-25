# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour probe models contracts

"""Focused behavior-probe models contracts."""

from .behavior_probe_support import *


def test_probe_undrivable_model_is_resilient() -> None:
    """A model needing synaptic input records errors and is not drivable."""

    profile = probe_model_behavior("DendriticNMDANeuron")
    assert profile.drivable is False
    assert profile.behavior_tags == ()
    assert all(obs.error is not None for obs in profile.observations)


def test_probe_deterministic_model_reproduces() -> None:
    """A deterministic model yields a stable, non-empty, valid tag set."""

    first = probe_model_behavior("ThetaNeuron")
    second = probe_model_behavior("ThetaNeuron")
    assert first.behavior_tags == second.behavior_tags
    assert first.result_sha256 == second.result_sha256
    assert first.behavior_tags
    assert set(first.behavior_tags) <= BEHAVIOR_TAGS
    assert first.stochastic is False


def test_probe_strong_drive_reveals_tonic_firing() -> None:
    """The wide sweep reaches the strong-drive tonic regime of AdEx."""

    profile = probe_model_behavior("AdExNeuron")
    assert "excitable" in profile.behavior_tags
    assert "tonic" in profile.behavior_tags


def test_probe_records_seeded_poisson_default_as_reproducible() -> None:
    """The fixed replay seed keeps the measured default Poisson trace stable."""

    profile = probe_model_behavior("PoissonNeuron")
    assert profile.stochastic is False
    assert "stochastic" not in profile.behavior_tags
    assert "excitable" in profile.behavior_tags
    assert all(observation.reproducible for observation in profile.observations)
