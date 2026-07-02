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

from typing import TYPE_CHECKING, Mapping

from sc_neurocore.hdl_gen._ident import sanitize_ident


_SUPPORTED_TARGETS = frozenset({"mlir"})
_NODE_INPUT_ARITY = {
    "SC_AND": 2,
    "SC_MUX": 3,
    "SC_POPCOUNT": 1,
    "LIF_MEMBRANE": 1,
}


if TYPE_CHECKING:
    from typing import Protocol, Sequence

    class _IRNode(Protocol):
        @property
        def id(self) -> str: ...

        @property
        def type(self) -> str: ...

        @property
        def inputs(self) -> Sequence[str]: ...

        @property
        def output(self) -> str: ...

    class _IRGraph(Protocol):
        @property
        def nodes(self) -> Sequence[_IRNode]: ...


class SSAEnvironment:
    """Manages Static Single Assignment (SSA) registers for MLIR/Relay."""

    def __init__(self) -> None:
        self.registers: dict[str, str] = {}
        self.counter: int = 0

    def allocate(self, edge_name: str) -> str:
        """Allocate and bind the next SSA register for an SC-IR edge."""
        reg = f"%{self.counter}"
        self.counter += 1
        self.registers[edge_name] = reg
        return reg

    def get(self, edge_name: str) -> str:
        """Return an allocated register or validate an external input register."""
        if edge_name not in self.registers:
            return f"%{sanitize_ident(edge_name, context='input name')}"
        return self.registers[edge_name]


class ShapeInference:
    """Infers tensor shapes dynamically across the SNN graph."""

    def __init__(self, input_shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(input_shapes)

    def infer(self, node: _IRNode) -> None:
        """Infer and store the output shape for one supported SC-IR node."""
        expected_arity = _NODE_INPUT_ARITY.get(node.type)
        if expected_arity is None:
            raise ValueError(f"Unsupported SC-IR node type {node.type!r} for MLIR export.")
        if len(node.inputs) != expected_arity:
            raise ValueError(
                f"SC-IR node {node.id!r} of type {node.type!r} expects "
                f"{expected_arity} input edge(s), got {len(node.inputs)}."
            )

        input_shapes = [self._shape_for(node, edge_name) for edge_name in node.inputs]
        if node.type == "SC_AND" or node.type == "SC_MUX":
            self.shapes[node.output] = input_shapes[0]
        elif node.type == "SC_POPCOUNT":
            self.shapes[node.output] = input_shapes[0][:-1] + (1,)
        elif node.type == "LIF_MEMBRANE":
            self.shapes[node.output] = input_shapes[0]

    def _shape_for(self, node: _IRNode, edge_name: str) -> tuple[int, ...]:
        try:
            return self.shapes[edge_name]
        except KeyError as exc:
            raise ValueError(
                f"Missing input shape for edge {edge_name!r} consumed by node {node.id!r}."
            ) from exc


class CompilerExporter:
    """Export SC-IR graph-like objects to strict SSA MLIR text."""

    def __init__(self, target: str = "mlir") -> None:
        """Create an exporter for a supported compiler backend target."""
        if target not in _SUPPORTED_TARGETS:
            supported = ", ".join(sorted(_SUPPORTED_TARGETS))
            raise ValueError(
                f"Unsupported compiler export target {target!r}; supported targets: {supported}."
            )
        self.target = target

    def _topological_sort(self, nodes: Sequence[_IRNode]) -> list[_IRNode]:
        """Kahn's algorithm for topological sorting of the DAG."""
        self._validate_unique_edges(nodes)
        in_degree = {n.id: 0 for n in nodes}
        node_map = {n.id: n for n in nodes}
        adj_list: dict[str, list[str]] = {n.id: [] for n in nodes}
        output_to_node_id = {n.output: n.id for n in nodes}

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

    def _format_mlir_type(self, shape: tuple[int, ...], dtype: str = "i1") -> str:
        """Render an MLIR scalar or tensor type for a validated static shape."""
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"MLIR tensor dimensions must be positive; got {shape!r}.")
        if not shape or shape == (1,):
            return dtype
        dims = "x".join(map(str, shape))
        return f"tensor<{dims}x{dtype}>"

    def export_to_mlir(
        self, ir_graph: _IRGraph, input_shapes: Mapping[str, tuple[int, ...]]
    ) -> str:
        """Emit strict SSA MLIR text via topological traversal."""
        input_shape_map = dict(input_shapes)
        nodes = tuple(ir_graph.nodes)
        self._validate_output_input_collisions(nodes, input_shape_map)
        sorted_nodes = self._topological_sort(nodes)
        if not sorted_nodes:
            raise ValueError(
                "Cannot export MLIR for a graph with no nodes; expected at least one node."
            )
        self._validate_export_contract(sorted_nodes, input_shape_map)

        ssa = SSAEnvironment()
        shape_inf = ShapeInference(input_shape_map)
        safe_input_names = {
            inp: sanitize_ident(inp, context="input name") for inp in input_shape_map
        }

        mlir_lines = ["module {"]

        sig_args = ", ".join(
            [
                f"%{safe_input_names[inp]}: {self._format_mlir_type(shape)}"
                for inp, shape in input_shape_map.items()
            ]
        )
        mlir_lines.append(f"  func.func @sc_network_forward({sig_args}) {{")

        last_reg = ""
        last_shape = ""

        for node in sorted_nodes:
            shape_inf.infer(node)
            out_shape = shape_inf.shapes[node.output]
            out_type = self._format_mlir_type(
                out_shape, "i1" if "POPCOUNT" not in node.type else "i32"
            )

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

    def _validate_unique_edges(self, nodes: Sequence[_IRNode]) -> None:
        seen_node_ids: set[str] = set()
        seen_outputs: set[str] = set()
        for node in nodes:
            if node.id in seen_node_ids:
                raise ValueError(f"Duplicate node id {node.id!r} in SC-IR graph.")
            if node.output in seen_outputs:
                raise ValueError(f"Duplicate output edge {node.output!r} in SC-IR graph.")
            seen_node_ids.add(node.id)
            seen_outputs.add(node.output)

    def _validate_output_input_collisions(
        self, nodes: Sequence[_IRNode], input_shapes: Mapping[str, tuple[int, ...]]
    ) -> None:
        graph_inputs = set(input_shapes)
        for node in nodes:
            if node.output in graph_inputs:
                raise ValueError(f"SC-IR output edge {node.output!r} collides with graph input.")

    def _validate_export_contract(
        self, nodes: Sequence[_IRNode], input_shapes: Mapping[str, tuple[int, ...]]
    ) -> None:
        produced_edges = {node.output for node in nodes}
        graph_inputs = set(input_shapes)
        for node in nodes:
            for edge_name in node.inputs:
                if edge_name not in produced_edges and edge_name not in graph_inputs:
                    raise ValueError(
                        f"Missing input shape for external edge {edge_name!r} "
                        f"consumed by node {node.id!r}."
                    )
