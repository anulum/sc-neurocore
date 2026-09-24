# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A Delay after the weight delays the connection

"""NIR allows a ``Delay`` on either side of a ``Linear``; both must delay the connection.

A post-weight delay used to be passed through as if absent, so a file written
``source -> Linear -> Delay -> target`` compiled to hardware with no delay.
Each case writes a real NIR file and reads it through the bridge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir, from_scnetwork


def _lif(count: int):
    return nir.LIF(
        tau=np.full(count, 20.0),
        r=np.ones(count),
        v_leak=np.zeros(count),
        v_threshold=np.ones(count),
        v_reset=np.zeros(count),
    )


def _graph(tmp_path: Path, nodes: dict, edges: list[tuple[str, str]]):
    nodes = {
        "input": nir.Input(input_type={"input": np.array([2])}),
        "output": nir.Output(output_type={"output": np.array([3])}),
        **nodes,
    }
    path = tmp_path / "graph.nir"
    nir.write(path, nir.NIRGraph(nodes=nodes, edges=[("input", "src"), *edges, ("dst", "output")]))
    return from_scnetwork(from_nir(str(path), dt=1.0))


def test_a_delay_after_the_weight_delays_the_connection(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        {
            "src": _lif(2),
            "w": nir.Linear(weight=np.full((3, 2), 0.5)),
            "d": nir.Delay(delay=np.full(3, 3.0)),
            "dst": _lif(3),
        },
        [("src", "w"), ("w", "d"), ("d", "dst")],
    )
    (connection,) = [c for c in graph.connections if c.src == "src"]
    assert connection.dst == "dst"
    assert connection.delay_steps == 3


def test_delays_on_both_sides_of_the_weight_add(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        {
            "src": _lif(2),
            "d_in": nir.Delay(delay=np.full(2, 1.0)),
            "w": nir.Linear(weight=np.full((3, 2), 0.5)),
            "d_out": nir.Delay(delay=np.full(3, 2.0)),
            "dst": _lif(3),
        },
        [("src", "d_in"), ("d_in", "w"), ("w", "d_out"), ("d_out", "dst")],
    )
    (connection,) = [c for c in graph.connections if c.src == "src"]
    assert connection.delay_steps == 3


def test_a_post_weight_delay_that_differs_between_targets_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="delays destination neurons differently"):
        _graph(
            tmp_path,
            {
                "src": _lif(2),
                "w": nir.Linear(weight=np.full((3, 2), 0.5)),
                "d": nir.Delay(delay=np.array([1.0, 2.0, 2.0])),
                "dst": _lif(3),
            },
            [("src", "w"), ("w", "d"), ("d", "dst")],
        )
