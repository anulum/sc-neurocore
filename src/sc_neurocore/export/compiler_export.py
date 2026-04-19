# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SSA Compiler Export

"""SSA-based TVM/MLIR compiler frontend for SC-NeuroCore IR graphs.

Exports SNN dataflow graphs to MLIR text via topological traversal
with strict SSA register allocation and shape inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SSAEnvironment:
    """Manages Static Single Assignment (SSA) registers for MLIR/Relay."""

    def __init__(self):
        self.registers: Dict[str, str] = {}
        self.counter: int = 0

    def allocate(self, edge_name: str) -> str:
        reg = f"%{self.counter}"
        self.counter += 1
        self.registers[edge_name] = reg
        return reg

    def get(self, edge_name: str) -> str:
        if edge_name not in self.registers:
            # Assume it's a global input if not defined internally
            return f"%{edge_name}"
        return self.registers[edge_name]


class ShapeInference:
    """Infers tensor shapes dynamically across the SNN graph."""

    def __init__(self, input_shapes: Dict[str, Tuple[int, ...]]):
        self.shapes = input_shapes.copy()

    def infer(self, node: Any):
        if node.type == "SC_AND":
            # AND gate preserves shape (element-wise)
            self.shapes[node.output] = self.shapes[node.inputs[0]]
        elif node.type == "SC_MUX":
            self.shapes[node.output] = self.shapes[node.inputs[0]]
        elif node.type == "SC_POPCOUNT":
            in_shape = self.shapes[node.inputs[0]]
            self.shapes[node.output] = in_shape[:-1] + (1,)
        elif node.type == "LIF_MEMBRANE":
            self.shapes[node.output] = self.shapes[node.inputs[0]]


class CompilerExporter:
    def __init__(self, target: str = "mlir"):
        self.target = target

    def _topological_sort(self, nodes: List[Any]) -> List[Any]:
        """Kahn's algorithm for topological sorting of the DAG."""
        in_degree = {n.id: 0 for n in nodes}
        node_map = {n.id: n for n in nodes}
        adj_list = {n.id: [] for n in nodes}
        output_to_node_id = {n.output: n.id for n in nodes}

        # Build adjacency and degrees based on data flow (output -> input)
        for n in nodes:
            for inp in n.inputs:
                if inp in output_to_node_id:
                    src_id = output_to_node_id[inp]
                    adj_list[src_id].append(n.id)
                    in_degree[n.id] += 1

        queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            curr_id = queue.pop(0)
            curr_node = node_map[curr_id]
            sorted_nodes.append(curr_node)

            for neighbor in adj_list[curr_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(nodes):
            raise ValueError("Cycle detected in SNN IR graph. Cannot lower to SSA.")

        return sorted_nodes

    def _format_mlir_type(self, shape: Tuple[int, ...], dtype: str = "i1") -> str:
        if not shape or shape == (1,):
            return dtype
        dims = "x".join(map(str, shape))
        return f"tensor<{dims}x{dtype}>"

    def export_to_mlir(self, ir_graph: Any, input_shapes: Dict[str, Tuple[int, ...]]) -> str:
        """Emits strict SSA MLIR text via topological traversal."""
        sorted_nodes = self._topological_sort(ir_graph.nodes)
        ssa = SSAEnvironment()
        shape_inf = ShapeInference(input_shapes)

        mlir_lines = ["module {"]

        sig_args = ", ".join(
            [f"%{inp}: {self._format_mlir_type(shape)}" for inp, shape in input_shapes.items()]
        )
        mlir_lines.append(f"  func.func @sc_network_forward({sig_args}) {{")

        last_reg = ""
        last_shape = None

        for node in sorted_nodes:
            shape_inf.infer(node)
            out_shape = shape_inf.shapes[node.output]
            out_type = self._format_mlir_type(
                out_shape, "i1" if "POPCOUNT" not in node.type else "i32"
            )

            # Map input edges to SSA registers BEFORE allocating the output
            # (Ensures correct dependency tracking)
            in_regs = [ssa.get(inp) for inp in node.inputs]
            out_reg = ssa.allocate(node.output)

            last_reg = out_reg
            last_shape = out_type

            if node.type == "SC_AND":
                mlir_lines.append(
                    f"    {out_reg} = scpn.and {in_regs[0]}, {in_regs[1]} : {out_type}"
                )
            elif node.type == "SC_MUX":
                mlir_lines.append(
                    f"    {out_reg} = scpn.mux {in_regs[0]}, {in_regs[1]}, {in_regs[2]} : {out_type}"
                )
            elif node.type == "SC_POPCOUNT":
                in_type = self._format_mlir_type(shape_inf.shapes[node.inputs[0]], "i1")
                mlir_lines.append(
                    f"    {out_reg} = scpn.popcount {in_regs[0]} : ({in_type}) -> {out_type}"
                )
            elif node.type == "LIF_MEMBRANE":
                th = getattr(node, "threshold", 1.0)
                lk = getattr(node, "leak", 0.9)
                mlir_lines.append(
                    f"    {out_reg} = scpn.lif {in_regs[0]} {{threshold={th}, leak={lk}}} : {out_type}"
                )

        mlir_lines.append(f"    return {last_reg} : {last_shape}")
        mlir_lines.append("  }")
        mlir_lines.append("}")
        return "\n".join(mlir_lines)


if __name__ == "__main__":

    class MockNode:
        def __init__(self, t, i, ins, out, **kwargs):
            self.type = t
            self.id = i
            self.inputs = ins
            self.output = out
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockGraph:
        def __init__(self):
            # Deliberately out of order: LIF is defined before AND,
            # but LIF depends on AND's output ("mac_1").
            self.nodes = [
                MockNode("LIF_MEMBRANE", "n1", ["mac_1"], "spike_out", threshold=0.75, leak=0.9),
                MockNode("SC_AND", "m1", ["input_a", "input_b"], "mac_1"),
            ]

    exporter = CompilerExporter()
    inputs = {"input_a": (128, 1024), "input_b": (128, 1024)}

    print("--- Real SSA MLIR Export ---")
    print(exporter.export_to_mlir(MockGraph(), inputs))
