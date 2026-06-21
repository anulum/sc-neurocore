# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for edge Izhikevich neuron and online-O1 adaptation benchmark

"""Contracts for the edge Izhikevich neuron and the online-O1 adaptation benchmark."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.benchmarks import online_o1_adaptation as o1
from sc_neurocore.benchmarks.online_o1_adaptation import build_online_o1_adaptation_benchmark
from sc_neurocore.edge.neuron import IzhikevichNeuron


def test_izhikevich_neuron_spikes_resets_and_exposes_variants() -> None:
    """An Izhikevich neuron spikes under strong drive, resets, and exposes firing variants."""
    neuron = IzhikevichNeuron()

    spiked = any(neuron.tick(1000 << 16) for _ in range(100))
    assert spiked

    neuron.reset()
    assert neuron.spike_count == 0

    assert isinstance(IzhikevichNeuron.chattering(), IzhikevichNeuron)
    assert isinstance(IzhikevichNeuron.intrinsic_burst(), IzhikevichNeuron)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_synapses": 0}, {"target_weight": -1}, {"max_pairings": 0}],
)
def test_build_benchmark_rejects_invalid_arguments(kwargs: dict[str, Any]) -> None:
    """The benchmark builder validates synapse count, target weight and pairing bounds."""
    with pytest.raises(ValueError):
        build_online_o1_adaptation_benchmark(**kwargs)


def test_build_benchmark_reports_rust_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the Rust backend is unavailable, the report flags it without failing."""
    monkeypatch.setattr(o1, "is_available", lambda: False)

    report = build_online_o1_adaptation_benchmark(max_pairings=4)

    assert report["rust"]["available"] is False


def test_build_benchmark_handles_rust_construction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Rust synapse construction failure is reported as unavailable, not raised."""

    def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("native init failed")

    monkeypatch.setattr(o1, "is_available", lambda: True)
    monkeypatch.setattr(o1, "RustOnlineO1Synapse", _boom)

    report = build_online_o1_adaptation_benchmark(max_pairings=4)

    assert report["rust"]["available"] is False
