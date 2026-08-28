# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust-Python neuron binding coverage and parity map

"""Rust-Python neuron binding coverage and focused runtime parity checks."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import NamedTuple, Protocol, cast

import pytest

from sc_neurocore.neurons import models as py_models


class StepModel(Protocol):
    """Minimal public stepping surface shared by Python and Rust neuron objects."""

    def step(self, *args: object) -> object:
        """Advance one model step and return the model-specific spike value."""


class StateModel(StepModel, Protocol):
    """Stepping model that also exposes its public dynamic state."""

    def get_state(self) -> dict[str, float]:
        """Return the model's named dynamic state."""


class _RecordingStepModel:
    """Small step model used to cover generic stimulus routing."""

    def __init__(self, result: object) -> None:
        """Store the deterministic result returned by every step."""

        self.result = result
        self.calls: list[tuple[object, ...]] = []

    def step(self, *args: object) -> object:
        """Record one step call and return the configured result."""

        self.calls.append(args)
        return self.result


class PythonOnlyBoundary(NamedTuple):
    """Documented reason one registry name has no same-name PyO3 constructor."""

    name: str
    source_path: Path
    source_token: str
    reason_token: str


_ENGINE: ModuleType | None
try:
    _ENGINE = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    HAS_ENGINE = True
except ImportError:  # pragma: no cover - exercised only when the optional engine is absent.
    _ENGINE = None
    HAS_ENGINE = False

_DUAL_FLOAT = frozenset(
    {
        "AlphaNeuron",
        "COBALIFNeuron",
        "PinskyRinzelNeuron",
        "HayL5PyramidalNeuron",
        "SCLeakyTwoCompartmentLIFNeuron",
        "SCExponentialTwoCompartmentLIFNeuron",
    }
)
_BOOL_PARAM = frozenset({"TsodyksMarkramNeuron", "CompteWMNeuron"})
_INT_INPUT = frozenset(
    {
        "LoihiCUBANeuron",
        "Loihi2Neuron",
        "TrueNorthNeuron",
        "SpiNNaker2Neuron",
        "IntegerQIFNeuron",
        "AkidaNeuron",
    }
)
_VEC_INPUT = frozenset({"AmariNeuralField", "LeakyCompeteFireNeuron"})
_RATE_OVERRIDE = frozenset({"PoissonNeuron", "InhomogeneousPoissonNeuron", "GammaRenewalNeuron"})

_PYTHON_ONLY_MODELS = frozenset(
    {
        "AstrocyteNeuron",
        "ChayKeizerMinimalNeuron",
        "HybridFisherPosnerLIFNeuron",
        "Izhikevich2007Neuron",
        "SCLeakyTwoCompartmentLIFNeuron",
        "SCResettingParallelSpikingNeuron",
        "SRM0Neuron",
    }
)

_PYTHON_ONLY_BOUNDARIES = (
    PythonOnlyBoundary(
        name="AstrocyteNeuron",
        source_path=Path("src/sc_neurocore/neurons/models/astrocyte_adapter.py"),
        source_token="Adapter that wraps :class:`AstrocyteModel`",
        reason_token="Population adapter over `AstrocyteModel`",
    ),
    PythonOnlyBoundary(
        name="ChayKeizerMinimalNeuron",
        source_path=Path("src/sc_neurocore/neurons/models/chay_keizer_minimal.py"),
        source_token="Reduced three-state Chay-Keizer",
        reason_token="reduced three-state pancreatic beta-cell model",
    ),
    PythonOnlyBoundary(
        name="HybridFisherPosnerLIFNeuron",
        source_path=Path("src/sc_neurocore/neurons/models/hybrid_fisher_posner_lif.py"),
        source_token="quantum_cognition",
        reason_token="depends on the Python `SpinPoolMPS` quantum-metabolic state",
    ),
    PythonOnlyBoundary(
        name="Izhikevich2007Neuron",
        source_path=Path("src/sc_neurocore/neurons/models/izhikevich2007.py"),
        source_token="py_izhikevich2007_simulate",
        reason_token="function-level compiled accelerator",
    ),
    PythonOnlyBoundary(
        name="SCLeakyTwoCompartmentLIFNeuron",
        source_path=Path("src/sc_neurocore/neurons/models/sc_leaky_tc_lif.py"),
        source_token="preserved repository recurrence",
        reason_token="count-neutral preserved leaky recurrence",
    ),
    PythonOnlyBoundary(
        name="SCResettingParallelSpikingNeuron",
        source_path=Path("src/sc_neurocore/neurons/models/sc_resetting_psn.py"),
        source_token="preserved repository recurrence",
        reason_token="count-neutral preserved resetting recurrence",
    ),
    PythonOnlyBoundary(
        name="SRM0Neuron",
        source_path=Path("src/sc_neurocore/neurons/models/srm0.py"),
        source_token="coupled linear system",
        reason_token="exact-flow SRM0 membrane accumulator",
    ),
)

_RUST_NAME_MAP = {
    "AdaptiveThresholdMoENeuron": "RustAdaptiveThresholdMoENeuron",
    "AstrocyteLIFNeuron": "RustAstrocyteLIFNeuron",
    "CochlearHairCell": "RustCochlearHairCell",
    "ContinuousAttractorNeuron": "RustContinuousAttractorNeuron",
    "DendriticNMDANeuron": "RustDendriticNMDANeuron",
    "DirectionSelectiveRGC": "RustDirectionSelectiveRGC",
    "HybridLinearAttentionNeuron": "RustHybridLinearAttentionNeuron",
    "MulticompartmentMCNNeuron": "RustMulticompartmentMCNNeuron",
    "QuantumInspiredLIFNeuron": "RustQuantumInspiredLIFNeuron",
    "SCExponentialTwoCompartmentLIFNeuron": "SCExponentialTwoCompartmentLIF",
}

_GENERIC_PARITY_UNSUPPORTED = frozenset(_RUST_NAME_MAP) | _PYTHON_ONLY_MODELS
_STOCHASTIC = frozenset(
    {
        "EscapeRateNeuron",
        "SCStochasticRateAdaptationNeuron",
        "PoissonNeuron",
        "InhomogeneousPoissonNeuron",
        "GammaRenewalNeuron",
        "StochasticIFNeuron",
        "StochasticLIFNeuron",
        "GalvesLocherbachNeuron",
        "GLMNeuron",
        "GIFPopulationNeuron",
        "WongWangUnit",
    }
)
_DOC_PATH = Path("docs/api/neuron_models.md")
_RUST_AGGREGATE_SOURCE_PATHS = (
    Path("engine/src/pyo3_neurons.rs"),
    Path("engine/src/lib.rs"),
)
_DOC_TOKENS = (
    "177 public Python registry names",
    "160 same-name Rust constructors",
    "10 Rust-prefixed or core-only constructors",
    "7 Python-only registry names",
    "Python-only boundary rationale",
    "HybridFisherPosnerLIFNeuron",
)


def _all_model_names() -> tuple[str, ...]:
    """Return public Python model registry names in deterministic order."""

    return tuple(str(name) for name in py_models.__all__)


def _repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[1]


def _engine_module() -> ModuleType:
    """Return the compiled Rust engine module or skip parity-only tests."""

    if _ENGINE is None:
        pytest.skip("Rust engine not built")
    return _ENGINE


def _rust_name(name: str) -> str:
    """Return the current Rust constructor name for a Python registry name."""

    return _RUST_NAME_MAP.get(name, name)


def _make_py(name: str) -> StepModel:
    """Instantiate a Python neuron model by public registry name."""

    constructor = getattr(py_models, name)
    return cast(StepModel, constructor())


def _make_rs(name: str) -> StepModel:
    """Instantiate a Rust neuron model by mapped PyO3 constructor name."""

    module = _engine_module()
    rust_name = _rust_name(name)
    if not hasattr(module, rust_name):
        pytest.skip(f"No Rust binding for {name}")
    constructor = getattr(module, rust_name)
    return cast(StepModel, constructor())


def _rust_source_names() -> set[str]:
    """Return PyO3 class names declared by the committed Rust sources."""

    root = _repo_root()
    binding_paths = sorted((root / "engine/src/bindings").rglob("*.rs"))
    aggregate_paths = tuple(root / path for path in _RUST_AGGREGATE_SOURCE_PATHS)
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (*aggregate_paths, *binding_paths)
    )
    macro_names = re.findall(
        r'(?m)^[ \t]*py_neuron_default!\(\s*"([^"]+)"',
        source_text,
    )
    explicit_names = re.findall(
        r'(?m)^[ \t]*#\[pyclass\(\s*name\s*=\s*"([^"]+)"',
        source_text,
    )
    return set(macro_names + explicit_names)


def _spike_count_from_sequence(values: Sequence[object]) -> int:
    """Return an integer spike count from a vector-like model result."""

    total = 0
    for value in values:
        if isinstance(value, bool):
            total += int(value)
        elif isinstance(value, int):
            total += value
        elif isinstance(value, float):
            total += 1 if abs(value) > 0.001 else 0
        else:
            raise TypeError(f"Unsupported vector spike value: {value!r}")
    return total


def _spike_count(result: object) -> int:
    """Convert a model-specific step result into an integer spike count."""

    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result
    if isinstance(result, float):
        return 1 if abs(result) > 0.001 else 0
    if isinstance(result, tuple):
        return (
            1
            if any(isinstance(value, (int, float)) and abs(value) > 0.001 for value in result)
            else 0
        )
    if isinstance(result, list):
        return _spike_count_from_sequence(result)
    raise TypeError(f"Unsupported spike result: {result!r}")


def _run_steps(model: StepModel, name: str, n: int = 500) -> list[int]:
    """Run a model through the generic parity stimulus and collect spikes."""

    spikes: list[int] = []
    for _ in range(n):
        if name in _VEC_INPUT:
            stimulus = [5.0] * 4 if name == "LeakyCompeteFireNeuron" else [0.5] * 64
            spikes.append(_spike_count(model.step(stimulus)))
        elif name in _INT_INPUT:
            spikes.append(_spike_count(model.step(50)))
        elif name in _DUAL_FLOAT:
            spikes.append(_spike_count(model.step(5.0, 0.0)))
        elif name in _BOOL_PARAM:
            spikes.append(_spike_count(model.step(5.0, False)))
        elif name in _RATE_OVERRIDE:
            current = 200.0 if name == "InhomogeneousPoissonNeuron" else -1.0
            spikes.append(_spike_count(model.step(current)))
        else:
            spikes.append(_spike_count(model.step(5.0)))
    return spikes


def test_rust_binding_coverage_map_classifies_every_python_model() -> None:
    """Keep explicit binding decisions aligned with the Python model registry."""

    model_names = set(_all_model_names())
    mapped_names = set(_RUST_NAME_MAP)
    boundary_names = {boundary.name for boundary in _PYTHON_ONLY_BOUNDARIES}

    assert model_names >= _PYTHON_ONLY_MODELS
    assert boundary_names == _PYTHON_ONLY_MODELS
    assert mapped_names <= model_names
    assert model_names >= _STOCHASTIC
    assert model_names >= _GENERIC_PARITY_UNSUPPORTED
    assert not (_PYTHON_ONLY_MODELS & mapped_names)
    assert len(model_names) == 177
    assert len(model_names - mapped_names - _PYTHON_ONLY_MODELS) == 160


def test_rust_binding_coverage_map_matches_committed_rust_sources() -> None:
    """Verify every non-Python-only registry name has a committed PyO3 class."""

    rust_names = _rust_source_names()
    missing = {
        name: _rust_name(name)
        for name in _all_model_names()
        if name not in _PYTHON_ONLY_MODELS and _rust_name(name) not in rust_names
    }

    assert not missing


def test_python_only_boundaries_are_documented_from_live_sources() -> None:
    """Lock durable Python-only decisions to source evidence and public docs."""

    doc_text = (_repo_root() / _DOC_PATH).read_text(encoding="utf-8")
    rust_names = _rust_source_names()

    for boundary in _PYTHON_ONLY_BOUNDARIES:
        source_text = (_repo_root() / boundary.source_path).read_text(encoding="utf-8")
        assert boundary.source_token in source_text
        assert boundary.name in doc_text
        assert boundary.reason_token in doc_text
        assert boundary.name not in rust_names


@pytest.mark.skipif(not HAS_ENGINE, reason="Rust engine not built")
def test_rust_binding_coverage_map_matches_built_engine() -> None:
    """Verify the installed engine exposes the documented binding map."""

    module = _engine_module()
    rust_exports = set(dir(module))
    missing = {
        name: _rust_name(name)
        for name in _all_model_names()
        if name not in _PYTHON_ONLY_MODELS and _rust_name(name) not in rust_exports
    }

    assert not missing


def test_public_docs_describe_rust_binding_coverage_map() -> None:
    """Keep public model docs aligned with the live Rust binding map."""

    text = (_repo_root() / _DOC_PATH).read_text(encoding="utf-8")

    for token in _DOC_TOKENS:
        assert token in text


def test_optional_engine_absence_skip_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the explicit skip path used when the optional engine is absent."""

    monkeypatch.setattr(sys.modules[__name__], "_ENGINE", None)

    with pytest.raises(pytest.skip.Exception):
        _engine_module()


def test_missing_rust_constructor_skip_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the explicit skip path for stale or absent Rust constructors."""

    monkeypatch.setattr(sys.modules[__name__], "_ENGINE", ModuleType("empty_engine"))

    with pytest.raises(pytest.skip.Exception):
        _make_rs("QuadraticIFNeuron")


def test_spike_count_conversion_branches() -> None:
    """Exercise scalar, vector, tuple, and invalid spike-result conversion."""

    assert _spike_count(True) == 1
    assert _spike_count(2) == 2
    assert _spike_count(0.01) == 1
    assert _spike_count((0.0, 0.01)) == 1
    assert _spike_count([True, 2, 0.01]) == 4

    with pytest.raises(TypeError):
        _spike_count(object())
    with pytest.raises(TypeError):
        _spike_count_from_sequence([object()])


def test_rate_override_stimulus_routing() -> None:
    """Exercise the rate-model current overrides used by the parity harness."""

    poisson = _RecordingStepModel(False)
    inhomogeneous = _RecordingStepModel(False)

    assert _run_steps(poisson, "PoissonNeuron", 1) == [0]
    assert _run_steps(inhomogeneous, "InhomogeneousPoissonNeuron", 1) == [0]
    assert poisson.calls == [(-1.0,)]
    assert inhomogeneous.calls == [(200.0,)]


@pytest.mark.skipif(not HAS_ENGINE, reason="Rust engine not built")
def test_chay_keizer_five_state_trajectory_parity() -> None:
    """Lock the faithful Rust five-state burster to the Python reference trace."""

    py_model = py_models.ChayKeizerNeuron()
    constructor = _engine_module().ChayKeizerNeuron
    rs_model = cast(StateModel, constructor())

    for _ in range(500):
        assert rs_model.step(5.0) == py_model.step(5.0)

    rust_state = rs_model.get_state()
    assert set(rust_state) == {"v", "m", "h", "n", "ca"}
    for name in rust_state:
        assert rust_state[name] == pytest.approx(getattr(py_model, name), abs=1e-12)


@pytest.mark.parametrize("name", _all_model_names())
@pytest.mark.skipif(not HAS_ENGINE, reason="Rust engine not built")
def test_parity(name: str) -> None:
    """Compare generic scalar-current spike counts where contracts match."""

    if name in _PYTHON_ONLY_MODELS:
        pytest.skip(f"{name} is currently Python-only")
    if name in _RUST_NAME_MAP:
        pytest.skip(f"{name} uses a mapped Rust constructor outside this generic parity harness")
    if name in _STOCHASTIC:
        pytest.skip(f"{name} is RNG-dependent, skip exact parity")
    py_model = _make_py(name)
    rs_model = _make_rs(name)

    n = 2000 if "Butera" in name or "Bertram" in name else 500
    py_spikes = _run_steps(py_model, name, n)
    rs_spikes = _run_steps(rs_model, name, n)

    py_count = sum(py_spikes)
    rs_count = sum(rs_spikes)

    if py_count == 0 and rs_count == 0:
        return

    max_delta = max(3, int(max(py_count, rs_count) * 0.15))
    assert abs(py_count - rs_count) <= max_delta, (
        f"{name}: Python={py_count}, Rust={rs_count}, "
        f"delta={abs(py_count - rs_count)}, max_delta={max_delta}"
    )


@pytest.mark.parametrize(
    "name", ["LoihiCUBANeuron", "TrueNorthNeuron", "SigmaDeltaNeuron", "McCullochPittsNeuron"]
)
@pytest.mark.skipif(not HAS_ENGINE, reason="Rust engine not built")
def test_exact_parity(name: str) -> None:
    """Check integer and deterministic hardware-style models exactly."""

    py_model = _make_py(name)
    rs_model = _make_rs(name)

    py_spikes = _run_steps(py_model, name, 200)
    rs_spikes = _run_steps(rs_model, name, 200)

    assert py_spikes == rs_spikes, f"{name}: spike trains differ"
