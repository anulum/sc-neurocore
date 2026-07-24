# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (dwave_backend) from former test_quantum_annealing_solvers_backends.py

from __future__ import annotations

from quantum_annealing_solvers_backends_support import *  # noqa: F403


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chain_strength": 0.0}, "chain_strength"),
        ({"num_reads": unsafe(False)}, "num_reads"),
        ({"annealing_time_us": float("inf")}, "annealing_time_us"),
    ],
)
def test_dwave_interface_rejects_invalid_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    """QPU parameters must be finite and positive."""
    with pytest.raises(ValueError, match=match):
        DWaveInterface(**unsafe(kwargs))


def test_dwave_fallback_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Ocean dependencies use no more than 20 local reads."""
    monkeypatch.setattr(backends, "HAS_DWAVE", False)
    monkeypatch.setattr(backends, "HAS_DIMOD", False)
    interface = DWaveInterface(num_reads=50)
    assert interface.available is False
    result = interface.solve_ising(simple_ising())
    assert result["backend"] == "simulated_annealing_fallback"
    assert result["num_reads"] == 20
    with pytest.raises(ValueError, match="non-empty"):
        interface.solve_ising(unsafe("bad"))


def test_dwave_qpu_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """QPU submission forwards the BQM and reports validated timing."""
    captured: dict[str, object] = {}

    class FakeDimod:
        class BinaryQuadraticModel:
            def __init__(self, *args: object) -> None:
                captured["bqm_args"] = args

    class FakeSampler:
        pass

    class FakeBest:
        sample = {0: 1, 1: -1, 2: 1}
        energy = -1.25

    class FakeResponse:
        first = FakeBest()
        info = {"timing": {"qpu_access_time": 123}}

    class FakeComposite:
        def __init__(self, sampler: FakeSampler) -> None:
            captured["sampler"] = sampler

        def sample(self, bqm: object, **kwargs: object) -> FakeResponse:
            captured["bqm"] = bqm
            captured["kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(backends, "HAS_DWAVE", True)
    monkeypatch.setattr(backends, "HAS_DIMOD", True)
    monkeypatch.setattr(backends, "dimod", FakeDimod)
    monkeypatch.setattr(backends, "DWaveSampler", FakeSampler)
    monkeypatch.setattr(backends, "EmbeddingComposite", FakeComposite)
    model = simple_ising()
    result = DWaveInterface(1.7, 31, 23.0).solve_ising(model)
    assert result == {
        "best_spins": {0: 1, 1: -1, 2: 1},
        "best_energy": -1.25,
        "num_reads": 31,
        "backend": "dwave_qpu",
        "timing": {"qpu_access_time": 123},
    }
    assert captured["bqm_args"] == (model.h, model.J, model.offset, "SPIN")
    assert captured["kwargs"] == {
        "num_reads": 31,
        "chain_strength": 1.7,
        "annealing_time": 23.0,
    }


@pytest.mark.parametrize(
    ("first", "info", "match"),
    [
        (unsafe(None), {}, "best sample"),
        (unsafe(types.SimpleNamespace(sample="bad", energy=-1.0)), {}, "best sample"),
        (unsafe(types.SimpleNamespace(sample={0: 1}, energy=float("inf"))), {}, "non-finite"),
    ],
)
def test_dwave_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
    first: object,
    info: object,
    match: str,
) -> None:
    """Incomplete QPU responses raise instead of fabricating results."""

    class FakeDimod:
        class BinaryQuadraticModel:
            def __init__(self, *args: object) -> None:
                self.args = args

    class FakeSampler:
        pass

    class FakeResponse:
        def __init__(self) -> None:
            self.first = first
            self.info = info

    class FakeComposite:
        def __init__(self, sampler: object) -> None:
            self.sampler = sampler

        def sample(self, bqm: object, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(backends, "HAS_DWAVE", True)
    monkeypatch.setattr(backends, "HAS_DIMOD", True)
    monkeypatch.setattr(backends, "dimod", FakeDimod)
    monkeypatch.setattr(backends, "DWaveSampler", FakeSampler)
    monkeypatch.setattr(backends, "EmbeddingComposite", FakeComposite)
    with pytest.raises(RuntimeError, match=match):
        DWaveInterface().solve_ising(IsingModel(h={0: 0.0}))
