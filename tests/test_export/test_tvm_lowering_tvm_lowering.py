# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTVMLowering from former test_tvm_lowering.py

"""Focused suite: TestTVMLowering from former test_tvm_lowering.py."""

from __future__ import annotations

from tvm_lowering_support import *  # noqa: F403


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
