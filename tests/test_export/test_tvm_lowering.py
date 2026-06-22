# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TVM Lowering Tests

from __future__ import annotations

import unittest
from sc_neurocore.export.tvm_lowering import (
    TVMLowering,
    TargetSchedule,
    TargetDevice,
    RelayFunction,
)


class MockNode:
    def __init__(self, t, i, ins, out, **kwargs):
        self.type, self.id, self.inputs, self.output = t, i, ins, out
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGraph:
    def __init__(self, nodes):
        self.nodes = nodes


def simple_graph():
    return MockGraph(
        [
            MockNode("SC_AND", "m1", ["input_a", "input_b"], "mac_1"),
            MockNode("LIF_MEMBRANE", "n1", ["mac_1"], "spike_out", threshold=0.75, leak=0.9),
        ]
    )


class TestTargetSchedule(unittest.TestCase):
    def test_cpu_defaults(self):
        s = TargetSchedule.for_cpu()
        self.assertEqual(s.device, TargetDevice.CPU)
        self.assertEqual(s.opt_level, 3)

    def test_gpu_schedule(self):
        s = TargetSchedule.for_gpu()
        self.assertEqual(s.device, TargetDevice.CUDA)
        self.assertIn("warp_level_popcount", s.sc_specific)

    def test_fpga_xilinx(self):
        s = TargetSchedule.for_fpga("xilinx")
        self.assertEqual(s.device, TargetDevice.FPGA_XILINX)
        self.assertTrue(s.sc_specific["bitstream_packing"])

    def test_fpga_intel(self):
        s = TargetSchedule.for_fpga("intel")
        self.assertEqual(s.device, TargetDevice.FPGA_INTEL)


class TestRelayFunction(unittest.TestCase):
    def test_to_relay_text(self):
        func = RelayFunction(
            name="test_fn",
            params=[("x", "(128,), dtype=bool")],
            body_lines=["let %y = %x;"],
            ret_var="%y",
            ret_type="(128,), dtype=bool",
        )
        text = func.to_relay_text()
        self.assertIn("def @test_fn", text)
        self.assertIn("%y", text)


class TestTVMLowering(unittest.TestCase):
    def test_lower_produces_relay_text(self):
        lowering = TVMLowering()
        relay_text = lowering.lower(
            simple_graph(), {"input_a": (128, 1024), "input_b": (128, 1024)}
        )
        self.assertIn("def @sc_forward", relay_text)

    def test_contains_and_op(self):
        lowering = TVMLowering()
        relay_text = lowering.lower(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        self.assertIn("bitwise_and", relay_text)

    def test_contains_lif_op(self):
        lowering = TVMLowering()
        relay_text = lowering.lower(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        self.assertIn("@scpn.lif", relay_text)
        self.assertIn("threshold=0.75", relay_text)

    def test_unknown_node_type_lowers_to_passthrough(self):
        # A node whose type matches none of the SC primitives is emitted as a
        # scalar-shaped passthrough binding rather than dropped.
        lowering = TVMLowering()
        graph = MockGraph([MockNode("UNSUPPORTED_OP", "p1", ["input_a"], "out_p")])
        relay_text = lowering.lower(graph, {"input_a": (128,)})
        self.assertIn("passthrough", relay_text)

    def test_target_header(self):
        schedule = TargetSchedule.for_gpu()
        lowering = TVMLowering(schedule)
        relay_text = lowering.lower(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        self.assertIn("// Target: cuda", relay_text)

    def test_fpga_sc_config_in_header(self):
        schedule = TargetSchedule.for_fpga("xilinx")
        lowering = TVMLowering(schedule)
        relay_text = lowering.lower(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        self.assertIn("bitstream_packing", relay_text)

    def test_popcount_lowering(self):
        graph = MockGraph(
            [
                MockNode("SC_POPCOUNT", "pc1", ["input_a"], "count_out"),
            ]
        )
        lowering = TVMLowering()
        relay_text = lowering.lower(graph, {"input_a": (128, 1024)})
        self.assertIn("sum(cast(", relay_text)
        self.assertIn("int32", relay_text)

    def test_mux_lowering(self):
        graph = MockGraph(
            [
                MockNode("SC_MUX", "mx1", ["sel", "a", "b"], "mux_out"),
            ]
        )
        lowering = TVMLowering()
        relay_text = lowering.lower(graph, {"sel": (128,), "a": (128,), "b": (128,)})
        self.assertIn("where(", relay_text)

    def test_custom_function_name(self):
        lowering = TVMLowering()
        relay_text = lowering.lower(
            simple_graph(), {"input_a": (128,), "input_b": (128,)}, func_name="my_network"
        )
        self.assertIn("def @my_network", relay_text)

    def test_build_script(self):
        lowering = TVMLowering(TargetSchedule.for_gpu())
        relay_text = "def @main(%x: Tensor[(1), dtype=bool]) -> Tensor[(1), dtype=bool] {\n  %x\n}"
        script = lowering.emit_build_script(relay_text)
        self.assertIn("import tvm", script)
        self.assertIn("target", script)
        self.assertIn("cuda", script)
        self.assertIn("relay_ir =", script)
        self.assertIn(repr(relay_text), script)
        self.assertIn("relay.build(mod, target=target)", script)
        self.assertIn("lib.export_library(output_path)", script)
        self.assertNotIn("stub", script.lower())

    def test_empty_graph(self):
        graph = MockGraph([])
        lowering = TVMLowering()
        relay_text = lowering.lower(graph, {"input_a": (128,)})
        self.assertIn("def @sc_forward", relay_text)


if __name__ == "__main__":
    unittest.main()
