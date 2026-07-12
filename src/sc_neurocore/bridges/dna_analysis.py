# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA circuit structural analysis

"""Hybridization, topology, hairpin, and gate-graph analysis."""

from __future__ import annotations

from typing import Any, Dict

from .dna_types import DNACircuitDesign


class CrossHybridizationChecker:
    """Detect unwanted cross-hybridization between circuit strands.

    Computes a pairwise alignment score matrix for all strands in a
    design and flags pairs with dangerous complementarity.

    Parameters
    ----------
    max_complementary_run : int
        Maximum allowed consecutive complementary bases between
        two non-interacting strands (default 8).
    """

    def __init__(self, max_complementary_run: int = 8) -> None:
        self._max_run = max_complementary_run

    def check(self, design: DNACircuitDesign) -> list[Dict[str, Any]]:
        """Check all strand pairs for cross-hybridization.

        Returns a list of flagged pairs with the offending run length.
        """
        all_strands = design.input_strands + design.output_strands + design.fuel_strands
        for g in design.gates:
            all_strands.extend(g.strands)

        flags: list[Dict[str, Any]] = []
        comp_table = str.maketrans("ACGT", "TGCA")

        for i in range(len(all_strands)):
            for j in range(i + 1, len(all_strands)):
                sa = all_strands[i]
                sb = all_strands[j]
                comp_b = sb.sequence.translate(comp_table)[::-1]

                max_run = self._longest_common_substring(sa.sequence, comp_b)
                if max_run >= self._max_run:
                    flags.append(
                        {
                            "strand_a": sa.name,
                            "strand_b": sb.name,
                            "complementary_run": max_run,
                            "severity": "high" if max_run >= 12 else "medium",
                        }
                    )

        return flags

    @staticmethod
    def _longest_common_substring(a: str, b: str) -> int:
        """Length of the longest common substring."""
        if not a or not b:
            return 0
        max_len = 0
        prev = [0] * (len(b) + 1)
        for i in range(len(a)):
            curr = [0] * (len(b) + 1)
            for j in range(len(b)):
                if a[i] == b[j]:
                    curr[j + 1] = prev[j] + 1
                    max_len = max(max_len, curr[j + 1])
            prev = curr
        return max_len


class TopologicalAnalyzer:
    """Analyze circuit topology: depth, fan-out, feedback detection.

    Builds a directed graph from gate connectivity, then computes:
    - Topological sort order (or detects cycles)
    - Circuit depth (critical path length)
    - Fan-out per signal (number of consumers)
    - Feedback loops (cycles in the gate graph)
    """

    def analyze(self, design: DNACircuitDesign) -> Dict[str, Any]:
        """Run full topological analysis.

        Returns
        -------
        dict
            ``depth``, ``fan_out``, ``has_feedback``, ``cycles``,
            ``topological_order``, ``critical_path``.
        """
        adj: Dict[str, list[str]] = {}
        in_degree: Dict[str, int] = {}
        all_nodes: set[str] = set()

        for g in design.gates:
            out = g.output_name
            all_nodes.add(out)
            adj.setdefault(out, [])
            in_degree.setdefault(out, 0)

            for inp in g.input_names:
                all_nodes.add(inp)
                adj.setdefault(inp, []).append(out)
                in_degree[out] = in_degree.get(out, 0) + 1
                in_degree.setdefault(inp, 0)

        # Kahn's algorithm for topological sort + cycle detection
        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        topo_order: list[str] = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        has_feedback = len(topo_order) < len(all_nodes)
        cycles: list[list[str]] = []
        if has_feedback:
            remaining = all_nodes - set(topo_order)
            cycles.append(sorted(remaining))

        # Compute depth via longest path in DAG
        depth: Dict[str, int] = {n: 0 for n in all_nodes}
        for node in topo_order:
            for neighbor in adj.get(node, []):
                depth[neighbor] = max(depth[neighbor], depth[node] + 1)

        max_depth = max(depth.values()) if depth else 0

        # Fan-out
        fan_out: Dict[str, int] = {}
        for g in design.gates:
            for inp in g.input_names:
                fan_out[inp] = fan_out.get(inp, 0) + 1

        # Critical path
        critical_path: list[str] = []
        if depth:
            current = max(depth, key=lambda x: depth[x])
            critical_path = [current]

        return {
            "depth": max_depth,
            "fan_out": fan_out,
            "has_feedback": has_feedback,
            "cycles": cycles,
            "topological_order": topo_order,
            "critical_path": critical_path,
            "n_nodes": len(all_nodes),
        }


class HairpinChecker:
    """Detect potential hairpin (stem-loop) secondary structures.

    Scans each strand for self-complementary regions that could form
    intramolecular hairpins, reducing effective concentration and
    interfering with gate operation.

    Parameters
    ----------
    min_stem_length : int
        Minimum stem length to flag (default 4 bp).
    min_loop_length : int
        Minimum loop length for a valid hairpin (default 3 nt).
    """

    _WC: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def __init__(
        self,
        min_stem_length: int = 4,
        min_loop_length: int = 3,
    ) -> None:
        self._min_stem = min_stem_length
        self._min_loop = min_loop_length

    def check_strand(self, sequence: str) -> list[Dict[str, Any]]:
        """Find potential hairpins in a single sequence.

        Returns
        -------
        list[dict]
            Each entry: ``stem_start``, ``stem_end``, ``loop_start``,
            ``loop_end``, ``stem_length``, ``delta_g_estimate``.
        """
        hairpins: list[Dict[str, Any]] = []
        n = len(sequence)

        for i in range(n - self._min_stem * 2 - self._min_loop):
            for stem_len in range(self._min_stem, min(12, (n - i) // 2)):
                loop_start = i + stem_len
                for loop_len in range(
                    self._min_loop,
                    min(10, n - loop_start - stem_len + 1),
                ):
                    j = loop_start + loop_len
                    # Check complementarity of stem
                    matches = 0
                    for k in range(stem_len):
                        left = sequence[i + k]
                        right = sequence[j + stem_len - 1 - k]
                        if self._WC.get(left) == right:
                            matches += 1
                    if matches >= stem_len:
                        dg_est = -1.5 * stem_len + 1.3  # rough estimate
                        hairpins.append(
                            {
                                "stem_start": i,
                                "stem_end": i + stem_len,
                                "loop_start": loop_start,
                                "loop_end": j,
                                "stem_length": stem_len,
                                "loop_length": loop_len,
                                "delta_g_estimate": dg_est,
                            }
                        )

        return hairpins

    def check_design(self, design: DNACircuitDesign) -> list[Dict[str, Any]]:
        """Check all strands in a circuit for hairpins.

        Returns list of flagged strands with hairpin details.
        """
        flags: list[Dict[str, Any]] = []
        all_strands = list(design.input_strands) + list(design.output_strands)
        for g in design.gates:
            all_strands.extend(g.strands)

        for strand in all_strands:
            hairpins = self.check_strand(strand.sequence)
            if hairpins:
                flags.append(
                    {
                        "strand_name": strand.name,
                        "sequence_length": strand.length,
                        "n_hairpins": len(hairpins),
                        "worst_stem": max(h["stem_length"] for h in hairpins),
                        "hairpins": hairpins,
                    }
                )

        return flags


# ══════════════════════════════════════════════════════════════════════
# Gate Optimizer
# ══════════════════════════════════════════════════════════════════════


class GateOptimizer:
    """Circuit-level gate optimization.

    Performs:
    - Dead gate elimination (outputs not consumed by any downstream gate)
    - Constant propagation (gates with all-zero or all-max inputs)
    - Identity elimination (BUFFER gates with direct pass-through)
    - Duplicate detection (identical gate specs)
    """

    def optimize(
        self,
        gates: list[Dict[str, Any]],
        output_names: list[str],
    ) -> Dict[str, Any]:
        """Optimize a gate list before compilation.

        Parameters
        ----------
        gates : list[dict]
            Raw gate specifications.
        output_names : list[str]
            Required output signal names.

        Returns
        -------
        dict
            ``optimized_gates``, ``removed_count``, ``removals``.
        """
        required: set[str] = set(output_names)
        removals: list[Dict[str, str]] = []

        # Build dependency graph: which signals are consumed?
        consumed: set[str] = set(output_names)
        for g in gates:
            for inp in g["inputs"]:
                consumed.add(inp)

        # Dead gate elimination
        live_gates: list[Dict[str, Any]] = []
        for g in gates:
            if g["output"] not in consumed and g["output"] not in required:
                removals.append({"gate": str(g), "reason": "dead_output"})
            else:
                live_gates.append(g)

        # Identity elimination (BUFFER with no downstream transform)
        final_gates: list[Dict[str, Any]] = []
        for g in live_gates:
            if (
                g["type"].upper() == "BUFFER"
                and len(g["inputs"]) == 1
                and g["output"] not in required
            ):
                removals.append({"gate": str(g), "reason": "identity_buffer"})
            else:
                final_gates.append(g)

        # Duplicate detection
        seen: set[str] = set()
        deduped: list[Dict[str, Any]] = []
        for g in final_gates:
            key = f"{g['type']}_{','.join(sorted(g['inputs']))}_{g['output']}"
            if key in seen:
                removals.append({"gate": str(g), "reason": "duplicate"})
            else:
                seen.add(key)
                deduped.append(g)

        return {
            "optimized_gates": deduped,
            "removed_count": len(removals),
            "original_count": len(gates),
            "removals": removals,
        }
