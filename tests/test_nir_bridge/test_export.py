# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

from pathlib import Path

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir

from tests.test_nir_bridge.support import make_lif_affine_graph


class TestExport:
    def test_roundtrip_lif_affine(self) -> None:
        from sc_neurocore.nir_bridge import to_nir

        graph = make_lif_affine_graph(n_in=3, n_out=2)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported, nir.NIRGraph)
        assert len(exported.nodes) == 4
        assert len(exported.edges) == 3
        assert isinstance(exported.nodes["lif"], nir.LIF)
        assert isinstance(exported.nodes["affine"], nir.Affine)

    def test_roundtrip_file_io(self, tmp_path: Path) -> None:
        from sc_neurocore.nir_bridge import to_nir

        graph = make_lif_affine_graph()
        network = from_nir(graph)
        path = tmp_path / "exported.nir"
        to_nir(network, path=str(path))
        assert path.exists()
        reloaded = from_nir(str(path))
        assert len(reloaded.nodes) == 4

    def test_export_type_error(self) -> None:
        from sc_neurocore.nir_bridge import to_nir

        with pytest.raises(TypeError, match="Expected SCNetwork"):
            to_nir("not a network")

    def test_export_linear_chain(self) -> None:
        from sc_neurocore.nir_bridge import to_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "linear": nir.Linear(weight=np.eye(2, dtype=np.float32)),
            "scale": nir.Scale(scale=np.array([2.0, 2.0])),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "linear"), ("linear", "scale"), ("scale", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["linear"], nir.Linear)
        assert isinstance(exported.nodes["scale"], nir.Scale)

    def test_export_all_basic_types(self) -> None:
        """Roundtrip every exportable node type through from_nir -> to_nir."""
        from sc_neurocore.nir_bridge import to_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "lif": nir.LIF(
                tau=np.array([10.0, 10.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "if": nir.IF(r=np.array([1.0, 1.0]), v_threshold=np.array([0.5, 0.5])),
            "li": nir.LI(tau=np.array([10.0, 10.0]), r=np.ones(2), v_leak=np.zeros(2)),
            "integ": nir.I(r=np.array([1.0, 1.0])),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32), bias=np.zeros(2, dtype=np.float32)
            ),
            "lin": nir.Linear(weight=np.eye(2, dtype=np.float32)),
            "scale": nir.Scale(scale=np.array([2.0, 2.0])),
            "thr": nir.Threshold(threshold=np.array([0.5, 0.5])),
            "flat": nir.Flatten(start_dim=0, end_dim=-1),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [
            ("input", "lif"),
            ("lif", "if"),
            ("if", "li"),
            ("li", "integ"),
            ("integ", "aff"),
            ("aff", "lin"),
            ("lin", "scale"),
            ("scale", "thr"),
            ("thr", "flat"),
            ("flat", "output"),
        ]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["lif"], nir.LIF)
        assert isinstance(exported.nodes["if"], nir.IF)
        assert isinstance(exported.nodes["li"], nir.LI)
        assert isinstance(exported.nodes["integ"], nir.I)
        assert isinstance(exported.nodes["aff"], nir.Affine)
        assert isinstance(exported.nodes["lin"], nir.Linear)
        assert isinstance(exported.nodes["scale"], nir.Scale)
        assert isinstance(exported.nodes["thr"], nir.Threshold)
        assert isinstance(exported.nodes["flat"], nir.Flatten)

    def test_export_cubalif_roundtrip(self) -> None:
        """CubaLIF roundtrip preserves all 7 parameters exactly."""
        from sc_neurocore.nir_bridge import to_nir

        orig = nir.CubaLIF(
            tau_syn=np.array([5.0, 5.0]),
            tau_mem=np.array([10.0, 10.0]),
            r=np.array([0.8, 0.8]),
            v_leak=np.array([-0.1, -0.1]),
            v_threshold=np.ones(2),
            w_in=np.array([1.2, 1.2]),
            v_reset=np.array([-0.25, -0.25]),
        )
        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "cubalif": orig,
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "cubalif"), ("cubalif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph, dt=1.0)
        exported = to_nir(network)
        out = exported.nodes["cubalif"]
        assert isinstance(out, nir.CubaLIF)
        for param in ["tau_syn", "tau_mem", "r", "v_leak", "v_threshold", "w_in", "v_reset"]:
            np.testing.assert_array_equal(getattr(out, param), getattr(orig, param))

    def test_export_cubali_roundtrip(self) -> None:
        """CubaLI roundtrip preserves parameters."""
        from sc_neurocore.nir_bridge import to_nir

        orig = nir.CubaLI(
            tau_syn=np.array([5.0, 5.0]),
            tau_mem=np.array([10.0, 10.0]),
            r=np.ones(2),
            v_leak=np.zeros(2),
            w_in=np.array([1.5, 1.5]),
        )
        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "cubali": orig,
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "cubali"), ("cubali", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        out = exported.nodes["cubali"]
        assert isinstance(out, nir.CubaLI)
        for param in ["tau_syn", "tau_mem", "r", "v_leak", "w_in"]:
            np.testing.assert_array_equal(getattr(out, param), getattr(orig, param))

    def test_export_delay_roundtrip(self) -> None:
        """Delay node roundtrip."""
        from sc_neurocore.nir_bridge import to_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "lif": nir.LIF(
                tau=np.array([10.0, 10.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "delay": nir.Delay(delay=np.array([1.0, 1.0])),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "lif"), ("lif", "delay"), ("delay", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["delay"], nir.Delay)

    def test_export_conv_pool_roundtrip(self) -> None:
        """Conv1d, Conv2d, SumPool2d, AvgPool2d roundtrip."""
        from sc_neurocore.nir_bridge import to_nir

        w1d = np.random.randn(2, 1, 3).astype(np.float32)
        conv1d_node = nir.Conv1d(
            input_shape=8,
            weight=w1d,
            bias=np.zeros(2, dtype=np.float32),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        nodes = {
            "input": nir.Input(input_type=conv1d_node.input_type),
            "conv1d": conv1d_node,
            "output": nir.Output(output_type=conv1d_node.output_type),
        }
        edges = [("input", "conv1d"), ("conv1d", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["conv1d"], nir.Conv1d)
        np.testing.assert_array_equal(exported.nodes["conv1d"].weight, w1d)

        w2d = np.random.randn(2, 1, 3, 3).astype(np.float32)
        conv2d_node = nir.Conv2d(
            input_shape=np.array([4, 4]),
            weight=w2d,
            bias=np.zeros(2, dtype=np.float32),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        nodes2 = {
            "input": nir.Input(input_type=conv2d_node.input_type),
            "conv2d": conv2d_node,
            "output": nir.Output(output_type=conv2d_node.output_type),
        }
        edges2 = [("input", "conv2d"), ("conv2d", "output")]
        graph2 = nir.NIRGraph(nodes=nodes2, edges=edges2)
        network2 = from_nir(graph2)
        exported2 = to_nir(network2)
        assert isinstance(exported2.nodes["conv2d"], nir.Conv2d)
        np.testing.assert_array_equal(exported2.nodes["conv2d"].weight, w2d)

        nodes3 = {
            "input": nir.Input(input_type={"input": np.array([1, 4, 4])}),
            "spool": nir.SumPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([2, 2]),
                padding=np.array([0, 0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1, 2, 2])}),
        }
        edges3 = [("input", "spool"), ("spool", "output")]
        graph3 = nir.NIRGraph(nodes=nodes3, edges=edges3)
        network3 = from_nir(graph3)
        exported3 = to_nir(network3)
        assert isinstance(exported3.nodes["spool"], nir.SumPool2d)

        nodes4 = {
            "input": nir.Input(input_type={"input": np.array([1, 4, 4])}),
            "apool": nir.AvgPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([2, 2]),
                padding=np.array([0, 0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1, 2, 2])}),
        }
        edges4 = [("input", "apool"), ("apool", "output")]
        graph4 = nir.NIRGraph(nodes=nodes4, edges=edges4)
        network4 = from_nir(graph4)
        exported4 = to_nir(network4)
        assert isinstance(exported4.nodes["apool"], nir.AvgPool2d)


# --- Sinabs interop tests ---
# Sinabs exports: nir.LIF (r=1, v_leak=0, tau=physical), nir.IF, nir.LI,
# nir.Affine (always, even bias=0). No CubaLIF. No dt baked into params.
# Internal dynamics use exponential decay: alpha = exp(-1/tau_mem).
