# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Numeric NIR round trip through a real file

"""An exported network, read back from its ``.nir`` file, computes the same thing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nir
import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_nir, to_nir

_STEPS = 40


def _feedforward() -> Any:
    rng = np.random.default_rng(7)
    return nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type={"input": np.array([3])}),
            "affine": nir.Affine(
                weight=rng.uniform(0.2, 1.0, size=(2, 3)), bias=rng.uniform(0.0, 0.2, size=2)
            ),
            "lif": nir.LIF(
                tau=np.array([4.0, 9.0]),
                r=np.array([1.0, 1.5]),
                v_leak=np.array([0.0, 0.1]),
                v_threshold=np.array([0.8, 1.2]),
                v_reset=np.array([0.0, -0.2]),
            ),
            "out": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("in", "affine"), ("affine", "lif"), ("lif", "out")],
    )


def _current_based_with_delay_and_recurrence() -> Any:
    return nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type={"input": np.array([2])}),
            "linear": nir.Linear(weight=np.array([[0.9, -0.3], [0.2, 0.7]])),
            "delay": nir.Delay(delay=np.array([2.0, 3.0])),
            "cuba": nir.CubaLIF(
                tau_syn=np.array([2.0, 3.0]),
                tau_mem=np.array([5.0, 6.0]),
                r=np.array([1.0, 1.0]),
                v_leak=np.array([0.0, 0.0]),
                v_threshold=np.array([0.5, 0.6]),
                v_reset=np.array([0.0, 0.0]),
                w_in=np.array([1.0, 1.0]),
            ),
            "scale": nir.Scale(scale=np.array([0.5, 0.5])),
            "out": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("in", "linear"),
            ("linear", "delay"),
            ("delay", "cuba"),
            ("cuba", "scale"),
            ("scale", "cuba"),
            ("cuba", "out"),
        ],
        type_check=False,
    )


def _drive(width: int) -> list[np.ndarray[Any, Any]]:
    rng = np.random.default_rng(11)
    return [rng.uniform(0.0, 3.0, size=width) for _ in range(_STEPS)]


def _trace(network: Any, drive: list[np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
    (source,) = network.input_nodes
    (sink,) = network.output_nodes
    return [network.step({source: value})[sink].copy() for value in drive]


@pytest.mark.parametrize(
    ("build", "width", "dt"),
    [(_feedforward, 3, 1.0), (_current_based_with_delay_and_recurrence, 2, 0.5)],
    ids=["feedforward-lif", "cuba-delay-recurrent"],
)
def test_a_network_read_back_from_its_nir_file_computes_identically(
    tmp_path: Path, build: Any, width: int, dt: float
) -> None:
    """Same inputs, same outputs, shapes and parameters after a file round trip."""
    original = from_nir(build(), dt=dt)
    path = tmp_path / "round_trip.nir"
    to_nir(original, path)
    restored = from_nir(path, dt=dt)

    assert restored.nir_version == nir.version
    # Both networks realise recurrent edges with their internal delay nodes.
    assert restored.topo_order == original.topo_order
    assert set(restored.nodes) == set(original.nodes)
    for name, node in original.nodes.items():
        for field_name, value in vars(node).items():
            if isinstance(value, np.ndarray) and not field_name.startswith("_"):
                np.testing.assert_array_equal(getattr(restored.nodes[name], field_name), value)

    drive = _drive(width)
    expected = _trace(original, drive)
    actual = _trace(restored, drive)
    assert [trace.shape for trace in actual] == [trace.shape for trace in expected]
    for step, (left, right) in enumerate(zip(expected, actual, strict=True)):
        np.testing.assert_array_equal(right, left, err_msg=f"step {step}")
    # A silent network would agree trivially; both cases spike repeatedly.
    assert sum(int(np.count_nonzero(trace)) for trace in expected) >= 5
