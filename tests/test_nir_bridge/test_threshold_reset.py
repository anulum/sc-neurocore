# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir
from sc_neurocore.nir_bridge.node_map import (
    map_node,
)


class TestThresholdAndReset:
    def test_strict_threshold(self) -> None:
        """NIR spec: z=1 when v > v_threshold (strict, not >=)."""
        node = nir.LIF(
            tau=np.array([100.0]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
            v_threshold=np.array([1.0]),
        )
        sc = map_node("lif", node, dt=1.0)
        # Feed exactly threshold worth of current: v should reach 1.0 but not spike
        # dv = (0 - 0 + 1*100) * 1/100 = 1.0, so v = 1.0
        sc.forward(np.array([100.0]))
        assert sc.v[0] == 1.0  # strict >: v == threshold means no spike
        # Larger input to push v above threshold
        # dv = (0 - 1.0 + 1*200) * 0.01 = 199*0.01 = 1.99, v = 1.0 + 1.99 = 2.99
        out = sc.forward(np.array([200.0]))
        assert out[0] == 1.0  # now fires (v > threshold)

    def test_reset_mode_default(self) -> None:
        """Default reset: v = v_reset after spike."""

        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "lif": nir.LIF(
                tau=np.array([1.0]),
                r=np.array([1.0]),
                v_leak=np.array([0.0]),
                v_threshold=np.array([0.5]),
                v_reset=np.array([0.1]),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="reset")
        net.run({"input": np.array([10.0])}, steps=3)
        lif = net.nodes["lif"]
        # After spiking, v should be near v_reset (0.1), not near v-threshold
        assert lif.v[0] < 1.0

    def test_reset_mode_subtract(self) -> None:
        """Subtract reset: v = v - v_threshold after spike."""

        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "lif": nir.LIF(
                tau=np.array([1.0]),
                r=np.array([1.0]),
                v_leak=np.array([0.0]),
                v_threshold=np.array([0.5]),
                v_reset=np.array([0.0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="subtract")
        # Large input -> spike -> v should be v_at_spike - threshold, not v_reset
        out = net.step({"input": np.array([2.0])})
        assert out["output"][0] == 1.0  # spiked
        lif = net.nodes["lif"]
        # v was 2.0 (from r*I*dt/tau = 1*2*1/1 = 2), spiked, subtract: 2.0-0.5=1.5
        assert lif.v[0] == pytest.approx(1.5)

    def test_cubalif_subtract_reset(self) -> None:
        """CubaLIF subtract reset mode."""

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "cubalif": nir.CubaLIF(
                tau_syn=np.array([1.0, 1.0]),
                tau_mem=np.array([1.0, 1.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
                w_in=np.ones(2),
                v_reset=np.zeros(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "cubalif"), ("cubalif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="subtract")
        out = net.run({"input": np.array([5.0, 5.0])}, steps=3)
        # Should produce spikes with subtract reset
        total = sum(r.sum() for r in out["output"])
        assert total > 0

    def test_if_subtract_reset(self) -> None:
        """IF neuron subtract reset mode."""
        node = nir.IF(
            r=np.array([1.0]),
            v_threshold=np.array([0.5]),
        )
        sc = map_node("if", node, dt=1.0, reset_mode="subtract")
        out = sc.forward(np.array([2.0]))
        assert out[0] == 1.0
        # v was 2.0 (r*I*dt = 1*2*1 = 2.0), spiked, subtract: 2.0 - 0.5 = 1.5
        assert sc.v[0] == pytest.approx(1.5)

    def test_invalid_reset_mode_still_works(self) -> None:
        """Unknown reset_mode falls through to default (reset) behavior."""

        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "lif": nir.LIF(
                tau=np.array([1.0]),
                r=np.array([1.0]),
                v_leak=np.array([0.0]),
                v_threshold=np.array([0.5]),
                v_reset=np.array([0.0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="unknown")
        out = net.step({"input": np.array([2.0])})
        assert out["output"][0] == 1.0
        # Unknown mode falls through to else (reset mode)
        assert net.nodes["lif"].v[0] == pytest.approx(0.0)
