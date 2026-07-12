# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing analysis and hardware tests

"""Exercise landscape, aggregation, TTS, topology, and chain contracts."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.quantum_annealing import (
    ChainBreakResolver,
    EmbeddingAnalyzer,
    EnergyLandscape,
    HardwareGraph,
    IsingModel,
    SampleAggregator,
    TTSAnalyzer,
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


def test_embedding_analyzer_sparse_and_dense_models() -> None:
    """Embedding estimates report graph density and chain capacity."""
    sparse = EmbeddingAnalyzer().analyze(simple_ising())
    assert sparse["n_logical_qubits"] == 3
    assert sparse["n_couplers"] == 2
    assert sparse["min_chain_estimate"] == 1
    dense = IsingModel(
        J={(first, second): -1.0 for first in range(5) for second in range(first + 1, 5)},
        n_qubits=5,
    )
    assert EmbeddingAnalyzer().analyze(dense)["density"] == 1.0
    with pytest.raises(ValueError, match="non-empty"):
        EmbeddingAnalyzer().analyze(IsingModel())


def test_sample_aggregator_statistics() -> None:
    """Aggregation deduplicates, bins, and Boltzmann-weights aligned samples."""
    samples = [{0: 1, 1: -1}, {0: 1, 1: -1}, {0: -1, 1: 1}]
    result = SampleAggregator().aggregate(samples, [-2.0, -2.0, 0.0], temperature=0.5)
    assert result["unique_samples"] == 2
    assert result["best_sample"] == samples[0]
    assert result["best_energy"] == -2.0
    assert result["success_probability"] == pytest.approx(2 / 3)
    assert result["gs_degeneracy"] == 2
    assert len(result["histogram"]["counts"]) == 2
    assert result["boltzmann_avg_energy"] < result["mean_energy"]
    assert SampleAggregator().aggregate([], []) == {
        "unique_samples": 0,
        "best": {},
        "histogram": {},
    }


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SampleAggregator().aggregate(unsafe("bad"), []), "sequences"),
        (lambda: SampleAggregator().aggregate([{}], []), "equal lengths"),
        (lambda: SampleAggregator().aggregate([{}], [0.0], 0.0), "greater than zero"),
        (lambda: SampleAggregator().aggregate([{unsafe(-1): 1}], [0.0]), "indices"),
        (lambda: SampleAggregator().aggregate([{0: 2}], [0.0]), "domain"),
        (lambda: SampleAggregator().aggregate([{}], [float("nan")]), "finite"),
    ],
)
def test_sample_aggregator_rejects_invalid_inputs(call: object, match: str) -> None:
    """Misaligned, malformed, and non-finite sample sets are rejected."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


def test_tts_compute_boundary_cases() -> None:
    """TTS handles zero, perfect, and interior success probabilities."""
    analyzer = TTSAnalyzer()
    impossible = analyzer.compute(0.0, 20.0)
    assert math.isinf(impossible["tts_us"])
    perfect = analyzer.compute(1.0, 20.0)
    assert perfect["n_runs_needed"] == 1.0
    interior = analyzer.compute(0.5, 20.0)
    assert interior["tts_us"] > 20.0
    assert interior["tts_ms"] == interior["tts_us"] / 1000.0


def test_tts_from_samples_and_comparison() -> None:
    """Observed energy counts feed comparable named solver rows."""
    analyzer = TTSAnalyzer()
    row = analyzer.from_samples([-2.0, -2.0, 0.0], -2.0, tolerance=1e-9)
    assert row["p_success"] == pytest.approx(2 / 3)
    empty = analyzer.from_samples([], -2.0)
    assert math.isinf(empty["tts_us"])
    comparison = analyzer.compare_solvers(
        {
            "python": {"energies": [-2.0, 0.0], "t_anneal_us": 40.0},
            "native": {"energies": [-2.0, -2.0]},
        },
        -2.0,
    )
    assert set(comparison) == {"python", "native"}
    assert comparison["native"]["p_success"] == 1.0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: TTSAnalyzer().compute(float("nan"), 20.0), "finite"),
        (lambda: TTSAnalyzer().compute(-0.1, 20.0), "between"),
        (lambda: TTSAnalyzer().compute(1.1, 20.0), "between"),
        (lambda: TTSAnalyzer().compute(0.5, 20.0, 0.0), "strictly"),
        (lambda: TTSAnalyzer().compute(0.5, 20.0, 1.0), "strictly"),
        (lambda: TTSAnalyzer().compute(0.5, 0.0), "greater than zero"),
        (lambda: TTSAnalyzer().from_samples(unsafe("bad"), -1.0), "sequence"),
        (lambda: TTSAnalyzer().from_samples([0.0], float("inf")), "finite"),
        (lambda: TTSAnalyzer().from_samples([0.0], 0.0, tolerance=0.0), "tolerance"),
        (lambda: TTSAnalyzer().from_samples([float("nan")], 0.0), "finite"),
        (lambda: TTSAnalyzer().compare_solvers({"": {"energies": []}}, 0.0), "names"),
        (lambda: TTSAnalyzer().compare_solvers({"x": {"energies": "bad"}}, 0.0), "energy sequence"),
        (
            lambda: TTSAnalyzer().compare_solvers(
                {"x": {"energies": [], "t_anneal_us": True}}, 0.0
            ),
            "numeric",
        ),
    ],
)
def test_tts_rejects_invalid_inputs(call: object, match: str) -> None:
    """TTS probabilities, times, energies, and solver payloads are validated."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


def test_hardware_graph_capacities_and_embedding() -> None:
    """Each supported topology reports its documented idealized capacity."""
    assert HardwareGraph("chimera", 2).n_physical_qubits == 32
    assert HardwareGraph("pegasus", 16).n_physical_qubits == 5760
    assert HardwareGraph("zephyr", 2).n_physical_qubits == 192
    graph = HardwareGraph("pegasus", 2)
    result = graph.can_embed(simple_ising())
    assert result["embeddable"] is True
    assert graph.connectivity == 15
    assert result["utilization_pct"] > 0.0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: HardwareGraph("unknown"), "Unknown topology"),
        (lambda: HardwareGraph("chimera", 0), "positive"),
        (lambda: HardwareGraph("pegasus", 1), "at least two"),
        (lambda: HardwareGraph().can_embed(unsafe("bad")), "non-empty"),
    ],
)
def test_hardware_graph_rejects_invalid_inputs(call: object, match: str) -> None:
    """Unknown topology, invalid size, and empty models are rejected."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


def test_chain_resolution_and_break_statistics() -> None:
    """Majority voting and break analysis handle ties and single-qubit chains."""
    samples = [{0: 1, 1: 1, 2: -1}, {0: 1, 1: -1, 2: -1}]
    chains = {0: [0, 1], 1: [2]}
    resolved = ChainBreakResolver().resolve(samples, chains)
    assert resolved == [{0: 1, 1: -1}, {0: 1, 1: -1}]
    stats = ChainBreakResolver().analyze_breaks(samples, chains)
    assert stats["total_breaks"] == 1
    assert stats["break_rate"] == 0.5
    assert stats["per_chain"] == {0: 0.5, 1: 0.0}
    assert ChainBreakResolver().analyze_breaks([], chains)["break_rate"] == 0.0


def test_chain_energy_minimization_refines_vote() -> None:
    """Energy minimization flips a voted spin only when energy decreases."""
    model = IsingModel(h={0: 2.0, 1: -2.0}, n_qubits=2)
    result = ChainBreakResolver("minimize_energy").resolve(
        [{0: 1, 1: 1}],
        {0: [0], 1: [1]},
        model,
    )
    assert result == [{0: -1, 1: 1}]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: ChainBreakResolver("bad"), "Unknown method"),
        (lambda: ChainBreakResolver("minimize_energy").resolve([], {}), "model is required"),
        (lambda: ChainBreakResolver().resolve(unsafe("bad"), {}), "sequence"),
        (lambda: ChainBreakResolver().resolve([{unsafe(-1): 1}], {}), "indices"),
        (lambda: ChainBreakResolver().resolve([{0: 0}], {}), "spins"),
        (lambda: ChainBreakResolver().resolve([], {unsafe(-1): [0]}), "logical"),
        (lambda: ChainBreakResolver().resolve([], {0: []}), "non-empty"),
        (lambda: ChainBreakResolver().resolve([], {0: [unsafe("x")]}), "physical"),
        (lambda: ChainBreakResolver().resolve([], {0: [1, 1]}), "duplicate"),
        (lambda: ChainBreakResolver().resolve([], {0: [1], 1: [1]}), "multiple"),
        (lambda: ChainBreakResolver().resolve([], {0: [1]}, unsafe("bad")), "IsingModel"),
        (lambda: ChainBreakResolver().resolve([], {2: [1]}, IsingModel(h={0: 0.0})), "fit within"),
    ],
)
def test_chain_resolution_rejects_invalid_inputs(call: object, match: str) -> None:
    """Malformed samples, chains, and model mappings fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
