# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC reference graphs exported to real snnTorch cells

"""Compare SC graph execution with independently constructed real snnTorch cells."""

from typing import Any
import json

import nir
import numpy as np
import pytest

torch = pytest.importorskip("torch")
snn = pytest.importorskip("snntorch")
pytest.importorskip("nirtorch")

from sc_neurocore.nir_bridge import from_nir, to_snntorch


@pytest.mark.parametrize("kind", ["if", "lif", "cubalif"])
def test_independent_sc_definition_exports_real_cells_with_nonunit_gains(kind: str) -> None:
    """Compare independently authored SC dynamics with real cells and nonunit input gains."""
    width, dt = 3, 0.125
    r = np.array([0.5, 3.0, 7.0])
    threshold = np.array([0.3, 0.8, 1.2])
    if kind == "if":
        neuron = nir.IF(r=r, v_threshold=threshold)
    elif kind == "lif":
        neuron = nir.LIF(
            tau=np.array([0.4, 0.8, 1.5]), r=r, v_leak=np.zeros(width), v_threshold=threshold
        )
    else:
        neuron = nir.CubaLIF(
            tau_syn=np.array([0.5, 0.7, 1.3]),
            tau_mem=np.array([0.4, 0.8, 1.5]),
            r=r,
            v_leak=np.zeros(width),
            v_threshold=threshold,
            w_in=np.array([0.5, 1.5, 2.5]),
        )
    graph = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([4])),
            "weight": nir.Affine(
                np.arange(12, dtype=float).reshape(3, 4) / 12, np.array([0.1, -0.1, 0.2])
            ),
            "cell": neuron,
            "out": nir.Output(np.array([width])),
        },
        edges=[("in", "weight"), ("weight", "cell"), ("cell", "out")],
    )
    network = from_nir(graph, dt=dt)
    exported = to_snntorch(network)
    spikes = 0
    with torch.no_grad():
        for x in np.random.default_rng(8).uniform(-0.5, 2.0, (80, 4)):
            a = network.step({"in": x})["out"]
            b = exported(torch.from_numpy(x)).numpy()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(
                network.nodes["cell"].v, exported[1].cell.mem.numpy(), rtol=0, atol=1e-12
            )
            spikes += int(a.sum())
    assert spikes > 20


def _single_neuron_graph(**overrides: Any) -> Any:
    values = dict(
        tau=np.ones(3),
        r=np.ones(3),
        v_leak=np.zeros(3),
        v_threshold=np.ones(3),
        v_reset=np.zeros(3),
    )
    values.update(overrides)
    return nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "cell": nir.LIF(**values),
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "cell"), ("cell", "out")],
    )


@pytest.mark.parametrize(
    "values,message",
    [
        ({"v_leak": np.ones(3)}, "Nonzero leak"),
        ({"v_reset": np.ones(3)}, "reset potentials"),
        ({"v_threshold": -np.ones(3)}, "threshold"),
        ({"r": np.full(3, float("nan"))}, "finite"),
        ({"tau": np.ones(3) * 0.1}, "clamping"),
        ({"tau": np.full(3, float("inf"))}, "time constants"),
        ({"tau": np.zeros(3)}, "time constants"),
    ],
)
def test_sc_dynamics_refused_before_creating_inexact_torch_cells(values: Any, message: str) -> None:
    """Refuse physical parameters that the target cells would clamp or reinterpret."""
    network = from_nir(_single_neuron_graph(**values), dt=0.5)
    with pytest.raises(ValueError, match=message):
        to_snntorch(network)


def test_sc_subtract_reset_is_refused_instead_of_using_a_different_spike_rule() -> None:
    """Refuse an SC reset convention whose overshoot decision differs from snnTorch."""
    network = from_nir(_single_neuron_graph(), reset_mode="subtract")
    with pytest.raises(ValueError, match="differ after overshoot"):
        to_snntorch(network)


@pytest.mark.parametrize(
    "contract",
    [
        {"schema_version": "sc-neurocore.scnir.v0.7", "streams": []},
        {"bitstream_length": 256},
        {"encoding": "bipolar"},
        {"q_format": "Q8.8"},
        {"precision": {"fractional_bits": 8}},
        [{"encoding": "deterministic_replay"}],
        {"rounding": "stochastic"},
        {"overflow": "wrap"},
        {"kind": "lfsr", "seed": 1},
    ],
)
@pytest.mark.parametrize("as_text", [False, True])
def test_embedded_stream_semantics_cannot_disappear_into_float_torch(
    contract: Any, as_text: bool
) -> None:
    """Refuse declared SC semantics in root, nested or JSON-text NIR metadata."""
    network = from_nir(_single_neuron_graph())
    network.nir_metadata["portable_contract"] = json.dumps(contract) if as_text else contract
    with pytest.raises(ValueError, match="bitstream length, encoding, Q-format"):
        to_snntorch(network)


@pytest.mark.parametrize(
    "case,message",
    [
        ("empty", "one input and one output"),
        ("branch", "Branches"),
        ("merge", "Merging edges"),
        ("disconnected", "Disconnected NIR nodes"),
        ("scale", "Unsupported snnTorch export primitive"),
        ("matrix", "one vector"),
        ("fractional-delay", "Approximated NIR imports"),
        ("complex", "Complex NIR parameters"),
    ],
)
def test_real_graph_structures_that_a_torch_sequence_cannot_preserve_are_refused(
    case: str, message: str
) -> None:
    """Refuse actual branches, hidden components and unsupported tensor semantics."""
    nodes = {
        "in": nir.Input(np.array([3])),
        "linear": nir.Linear(np.eye(3)),
        "out": nir.Output(np.array([3])),
    }
    edges = [("in", "linear"), ("linear", "out")]
    if case == "empty":
        nodes, edges = {}, []
    elif case == "branch":
        nodes["parallel"] = nir.Linear(np.eye(3))
        edges += [("in", "parallel"), ("parallel", "out")]
    elif case in ("merge", "disconnected"):
        nodes["unreachable"] = nir.Linear(np.eye(3))
        edges += [("unreachable", "unreachable")]
        if case == "merge":
            edges += [("unreachable", "out")]
    elif case == "matrix":
        nodes = {
            "in": nir.Input(np.array([2, 3])),
            "linear": nir.Scale(np.ones((2, 3))),
            "out": nir.Output(np.array([2, 3])),
        }
    elif case == "scale":
        nodes["linear"] = nir.Scale(np.ones(3))
    elif case == "fractional-delay":
        nodes["linear"] = nir.Delay(np.full(3, 0.5))
    else:
        nodes["linear"] = nir.Linear(np.eye(3, dtype=np.complex128) * (1 + 2j))
    network = from_nir(nir.NIRGraph(nodes=nodes, edges=edges))
    with pytest.raises(ValueError, match=message):
        to_snntorch(network)


@pytest.mark.parametrize("bias", [None, np.zeros(3), np.full(3, float("nan"))])
def test_linear_and_affine_precision_and_finiteness_are_preserved(bias: Any) -> None:
    """Export true bias-free layers and refuse nonfinite affine parameters."""
    node = nir.Linear(np.eye(3)) if bias is None else nir.Affine(np.eye(3), bias)
    graph = nir.NIRGraph(
        nodes={"in": nir.Input(np.array([3])), "weight": node, "out": nir.Output(np.array([3]))},
        edges=[("in", "weight"), ("weight", "out")],
    )
    network = from_nir(graph)
    if bias is not None and not np.isfinite(bias).all():
        with pytest.raises(ValueError, match="linear parameters must be finite"):
            to_snntorch(network)
    else:
        exported = to_snntorch(network)
        assert (exported[0].bias is None) == (bias is None)
        with torch.no_grad():
            np.testing.assert_array_equal(
                exported(torch.ones(3, dtype=torch.float64)).numpy(), np.ones(3)
            )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("tau_syn", 0.0, "synaptic time constants"),
        ("tau_syn", 0.1, "synaptic decay"),
        ("w_in", float("nan"), "input gain must be finite"),
    ],
)
def test_current_based_parameters_cannot_be_clamped_or_replaced_silently(
    field: str, value: float, message: str
) -> None:
    """Refuse actual CubaLIF definitions outside the finite decay domain."""
    values = dict(
        tau_syn=np.full(3, 2.0),
        tau_mem=np.full(3, 2.0),
        r=np.ones(3),
        v_leak=np.zeros(3),
        v_threshold=np.ones(3),
        w_in=np.ones(3),
    )
    values[field] = np.full(3, value)
    graph = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([3])),
            "cell": nir.CubaLIF(**values),
            "out": nir.Output(np.array([3])),
        },
        edges=[("in", "cell"), ("cell", "out")],
    )
    with pytest.raises(ValueError, match=message):
        to_snntorch(from_nir(graph, dt=0.5))
