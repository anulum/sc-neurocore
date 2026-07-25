# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour probe derivation contracts

"""Focused behavior-probe derivation contracts."""

from .behavior_probe_support import *


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
