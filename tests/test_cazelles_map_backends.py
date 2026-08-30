# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cazelles source-map runtime parity

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models import cazelles_map
from sc_neurocore.neurons.models.cazelles_map import CazellesMapNeuron


_ROOT = Path(__file__).resolve().parents[1]


def _run(backend: str, *, n_steps: int = 600) -> tuple[npt.NDArray[np.float64], int, float]:
    neuron = CazellesMapNeuron()
    trace, events = neuron.simulate(n_steps, 0.0, backend=backend)
    return trace, events, neuron.x


@pytest.mark.parametrize("backend", ["rust", "julia", "go", "mojo"])
def test_exact_runtime_parity(backend: str) -> None:
    assert {
        "rust": cazelles_map._HAS_RUST,
        "julia": cazelles_map._ensure_julia_loaded(),
        "go": cazelles_map._ensure_go_loaded(),
        "mojo": cazelles_map._ensure_mojo_loaded(),
    }[backend]
    expected = _run("python")
    observed = _run(backend)
    np.testing.assert_array_equal(observed[0], expected[0])
    assert observed[1:] == expected[1:]


def test_ci_builds_the_required_go_and_mojo_libraries() -> None:
    """Keep both strict runtime loaders enrolled in the aggregate CI build."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "go/neurons/cazelles_map" in workflow
    assert "go build -buildmode=c-shared -o libcazelles.so cazelles_map.go" in workflow
    assert "-o src/sc_neurocore/accel/mojo/neurons/libcazelles.so" in workflow
    assert "src/sc_neurocore/accel/mojo/neurons/cazelles_map.mojo" in workflow


def test_mojo_source_orbit_event_parity_and_one_step_ulp_bound() -> None:
    assert cazelles_map._ensure_mojo_loaded()
    reference, expected_events, _xf = _run("python")
    observed, events, _mojo_xf = _run("mojo")
    assert expected_events == 7
    assert events == expected_events
    np.testing.assert_array_equal(observed, reference)

    rng = np.random.default_rng(2026)
    for _ in range(2000):
        x = float(rng.uniform(0.001, 0.999))
        if min(abs(x - point) for point in (0.4, 0.6, 0.7)) < 1.0e-6:
            continue
        expected, expected_event = CazellesMapNeuron(x=x).simulate(1, backend="python")
        observed, event = CazellesMapNeuron(x=x).simulate(1, backend="mojo")
        assert event == expected_event
        assert float(observed[0]) == float(expected[0])


def test_figure_one_parameters_and_all_four_branches() -> None:
    probes = (
        (0.2, 0.21),
        (0.5, 0.875),
        (0.65, 0.07500000000000007),
        (0.8, 0.5999999999999999),
    )
    for x, expected in probes:
        neuron = CazellesMapNeuron(x=x)
        assert neuron.step(0.0) == int(x >= 0.4 and expected < 0.4)
        assert neuron.x == pytest.approx(expected, abs=2.0e-16)


@pytest.mark.parametrize(
    ("x", "expected"),
    ((0.0, 0.0), (0.4, 1.0), (0.6, 0.0), (0.7, 0.7), (1.0, 0.3999999999999999)),
)
def test_disclosed_right_continuous_breakpoints(x: float, expected: float) -> None:
    neuron = CazellesMapNeuron(x=x)
    neuron.step(0.0)
    assert neuron.x == expected


def test_alpha_exponent_and_additive_input_extensions() -> None:
    linear = CazellesMapNeuron(x=0.2, alpha=0.1, exponent=1)
    quadratic = CazellesMapNeuron(x=0.2, alpha=0.1, exponent=2)
    linear.step(0.01)
    quadratic.step(0.01)
    assert linear.x == pytest.approx(0.24)
    assert quadratic.x == pytest.approx(0.224)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"x": float("nan")},
        {"alpha": 1.0},
        {"exponent": 3},
        {"exponent": True},
        {"x1": 0.6, "x2": 0.4},
        {"x": -0.1},
    ),
)
def test_invalid_construction_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CazellesMapNeuron(**kwargs)


def test_nonfinite_and_out_of_domain_updates_are_atomic() -> None:
    neuron = CazellesMapNeuron()
    before = neuron.x
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert neuron.x == before
    with pytest.raises(FloatingPointError, match="left its configured domain"):
        neuron.step(2.0)
    assert neuron.x == before


def test_simulation_validation_and_state_commit() -> None:
    with pytest.raises(ValueError, match="integer"):
        CazellesMapNeuron().simulate(True)
    with pytest.raises(ValueError, match="between"):
        CazellesMapNeuron().simulate(-1)
    with pytest.raises(ValueError, match="backend"):
        CazellesMapNeuron().simulate(1, backend="cuda")
    neuron = CazellesMapNeuron()
    trace, events = neuron.simulate(600, backend="python")
    assert events == 7
    assert neuron.x == trace[-1]
    assert np.all((trace >= 0.0) & (trace <= 1.0))
