# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour probe tests

"""Tests for the measured behaviour probe.

The tag-derivation predicates are exercised purely on synthetic observations
(no simulation); a small sample of fast, deterministic models is probed live to
prove the sweep reproduces; the manifest shape and digests are checked directly.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.behavior_taxonomy import BEHAVIOR_TAGS
from sc_neurocore.studio.behavior_probe import (
    BEHAVIOR_SWEEP_CURRENTS,
    BehaviorObservation,
    behavior_tags_for,
    derive_behavior_tags,
    probe_all_models,
    probe_model_behavior,
)


def _obs(
    current: float,
    pattern: str,
    *,
    rate_hz: float = 0.0,
    spike_count: int = 0,
    reproducible: bool = True,
    error: str | None = None,
) -> BehaviorObservation:
    """Construct a synthetic observation."""

    return BehaviorObservation(
        current=current,
        pattern=pattern,
        rate_hz=rate_hz,
        spike_count=spike_count,
        reproducible=reproducible,
        error=error,
    )


def test_derive_empty_when_all_drives_error() -> None:
    """A model that cannot be driven at any current yields no tags."""

    observations = [_obs(c, "error", error="TypeError: needs glutamate") for c in (0.0, 1.0, 4.0)]
    assert derive_behavior_tags(observations, stochastic=False) == ()


def test_derive_quiescent_when_all_silent() -> None:
    """A model that never spikes is quiescent and nothing else."""

    observations = [_obs(c, "silent") for c in (0.0, 1.0, 4.0, 16.0)]
    assert derive_behavior_tags(observations, stochastic=False) == ("quiescent",)


def test_derive_excitable_and_tonic() -> None:
    """A silent-then-tonic sweep yields excitable + tonic (+ rate-coded)."""

    observations = [
        _obs(0.0, "silent"),
        _obs(16.0, "tonic", rate_hz=40.0, spike_count=8),
        _obs(64.0, "tonic", rate_hz=90.0, spike_count=18),
    ]
    tags = derive_behavior_tags(observations, stochastic=False)
    assert "excitable" in tags
    assert "tonic" in tags
    assert "rate-coded" in tags
    assert "quiescent" not in tags


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("adapting", "adapting"),
        ("bursting", "bursting"),
        ("irregular", "irregular"),
        ("chaotic", "chaotic"),
        ("single_spike", "phasic"),
    ],
)
def test_derive_maps_each_pattern_to_its_tag(pattern: str, expected: str) -> None:
    """Each reproducible firing pattern contributes its behaviour tag."""

    observations = [_obs(0.0, "silent"), _obs(16.0, pattern, rate_hz=30.0, spike_count=6)]
    tags = derive_behavior_tags(observations, stochastic=False)
    assert expected in tags
    assert "excitable" in tags


def test_derive_rate_coded_requires_monotone_rise() -> None:
    """A rate that rises then falls is not rate-coded (depolarisation block)."""

    rising = [
        _obs(0.0, "silent"),
        _obs(16.0, "tonic", rate_hz=20.0, spike_count=4),
        _obs(64.0, "tonic", rate_hz=80.0, spike_count=16),
    ]
    assert "rate-coded" in derive_behavior_tags(rising, stochastic=False)

    blocked = [
        _obs(0.0, "silent"),
        _obs(16.0, "tonic", rate_hz=80.0, spike_count=16),
        _obs(64.0, "single_spike", rate_hz=5.0, spike_count=1),
    ]
    assert "rate-coded" not in derive_behavior_tags(blocked, stochastic=False)


def test_derive_rate_coded_needs_a_real_rise() -> None:
    """A flat f-I curve (constant rate) is not rate-coded."""

    flat = [
        _obs(16.0, "tonic", rate_hz=40.0, spike_count=8),
        _obs(64.0, "tonic", rate_hz=40.0, spike_count=8),
    ]
    assert "rate-coded" not in derive_behavior_tags(flat, stochastic=False)


def test_derive_stochastic_withholds_fine_tags() -> None:
    """A stochastic model keeps only excitability and the stochastic flag."""

    observations = [
        _obs(0.0, "silent", reproducible=False),
        _obs(16.0, "bursting", rate_hz=100.0, spike_count=20, reproducible=False),
        _obs(64.0, "chaotic", rate_hz=300.0, spike_count=60, reproducible=False),
    ]
    tags = derive_behavior_tags(observations, stochastic=True)
    assert tags == ("excitable", "stochastic")
    assert "bursting" not in tags
    assert "chaotic" not in tags


def test_derive_stochastic_quiescent() -> None:
    """A stochastic model that never spikes is quiescent + stochastic."""

    observations = [_obs(c, "silent", reproducible=False) for c in (0.0, 16.0, 64.0)]
    tags = derive_behavior_tags(observations, stochastic=True)
    assert tags == ("quiescent", "stochastic")


def test_derive_ignores_error_observations_for_excitability() -> None:
    """Error drives do not count as silent; a real spike still wins."""

    observations = [
        _obs(0.0, "error", error="ValueError: unstable"),
        _obs(16.0, "tonic", rate_hz=40.0, spike_count=8),
    ]
    tags = derive_behavior_tags(observations, stochastic=False)
    assert "excitable" in tags
    assert "quiescent" not in tags


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


def test_probe_all_models_manifest_shape() -> None:
    """The manifest carries the sweep config, digests and per-model entries."""

    manifest = probe_all_models(names=["ThetaNeuron", "DendriticNMDANeuron"])
    assert manifest["schema_version"] == "studio.behavior-probe.v1"
    assert tuple(manifest["sweep"]["currents"]) == BEHAVIOR_SWEEP_CURRENTS
    assert len(manifest["sweep_sha256"]) == 64
    assert len(manifest["result_sha256"]) == 64
    assert set(manifest["models"]) == {"ThetaNeuron", "DendriticNMDANeuron"}
    assert manifest["models"]["DendriticNMDANeuron"]["drivable"] is False


def test_behavior_tags_for_reads_manifest_entry() -> None:
    """The helper extracts a model's recorded tags and is empty for unknowns."""

    manifest = probe_all_models(names=["ThetaNeuron"])
    assert behavior_tags_for("ThetaNeuron", manifest)
    assert behavior_tags_for("NoSuchModel", manifest) == ()
