# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing energy-landscape contracts

from __future__ import annotations

import math

import pytest

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.quantum_annealing import (
    EnergyLandscape,
    IsingModel,
)
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


def test_energy_landscape_exhaustive_and_supplied_samples() -> None:
    """Small models enumerate exactly while supplied samples remain bounded."""
    model = simple_ising()
    exhaustive = EnergyLandscape(backend="python").analyze(model)
    assert exhaustive["n_samples"] == 8
    assert exhaustive["min_energy"] <= exhaustive["max_energy"]
    assert exhaustive["degeneracy"] >= 1
    assert exhaustive["n_unique_energies"] >= 1

    supplied = EnergyLandscape().analyze(
        model,
        [{0: 1, 1: 1, 2: 1}, {0: -1, 1: -1, 2: -1}],
    )
    assert supplied["n_samples"] == 2
    assert math.isfinite(supplied["mean_energy"])


def test_energy_landscape_large_sampling_is_deterministic() -> None:
    """Large-model fallback uses the configured finite sample count and seed."""
    model = IsingModel(h={0: -1.0}, n_qubits=21)
    first = EnergyLandscape(backend="python", random_sample_count=101, seed=7).analyze(model)
    second = EnergyLandscape(backend="python", random_sample_count=101, seed=7).analyze(model)
    assert first == second
    assert first["n_samples"] == 101
    assert first["min_energy"] == -1.0
    assert first["max_energy"] == 1.0
    assert first["spectral_gap"] == 2.0


def test_energy_landscape_native_batch_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native batch dispatch receives canonical matrices and validates its result."""
    captured: tuple[object, ...] = ()
    samples = [{0: 1 if index % 2 == 0 else -1, 1: -1, 2: 1} for index in range(101)]

    def fake_batch(*args: object) -> list[float]:
        nonlocal captured
        captured = args
        return [-1.0, -1.0, 0.5] + [1.0] * 98

    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_batch_energy", fake_batch)
    model = IsingModel(h={0: 0.5}, J={(1, 2): -0.25}, n_qubits=3)
    result = EnergyLandscape(backend="rust").analyze(model, samples)
    assert result["degeneracy"] == 2
    assert result["spectral_gap"] == 1.5
    assert result["n_unique_energies"] == 3
    assert captured == (
        [0],
        [0.5],
        [1],
        [2],
        [-0.25],
        [[sample.get(index, 1) for index in range(3)] for sample in samples],
        0.0,
    )


def test_energy_landscape_rejects_bad_native_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native batch results must align one-to-one with samples."""
    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_batch_energy", lambda *args: [0.0])
    with pytest.raises(RuntimeError, match="wrong energy count"):
        EnergyLandscape(backend="rust").analyze(simple_ising(), [{}, {}])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: EnergyLandscape(backend=unsafe("gpu")), "backend"),
        (lambda: EnergyLandscape(random_sample_count=unsafe(True)), "positive"),
        (lambda: EnergyLandscape(seed=unsafe(1.5)), "seed"),
        (lambda: EnergyLandscape().analyze(unsafe("bad")), "non-empty"),
        (lambda: EnergyLandscape().analyze(IsingModel()), "non-empty"),
        (lambda: EnergyLandscape().analyze(simple_ising(), unsafe("bad")), "sequence"),
        (lambda: EnergyLandscape().analyze(simple_ising(), []), "must not be empty"),
        (lambda: EnergyLandscape().analyze(simple_ising(), [{0: 0}]), "supported domain"),
        (lambda: EnergyLandscape._enumerate_all(-1), "between"),
        (lambda: EnergyLandscape._enumerate_all(21), "between"),
    ],
)
def test_energy_landscape_rejects_invalid_inputs(call: object, match: str) -> None:
    """Landscape configuration and samples fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
