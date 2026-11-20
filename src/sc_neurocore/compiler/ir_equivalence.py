# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded equivalence for compiler-emitted user IR

"""Prove the emitted graph boundary against an independent fixed-point reference."""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._ir_equivalence_reference import MODEL_OPERATIONS, build_reference
from .equivalence_check import formal_tools_available, prove_equivalence
from .equivalence_miter import parse_module_interface
from .guarantee_summary import summarize_ir_text

# Small graph checks must bound the caller's wait because solver cost can
# grow rapidly with the graph and the requested number of frames.
DEFAULT_GRAPH_PROOF_TIMEOUT_SECONDS = 300.0
_RESET_CYCLES = 2
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


@dataclass(frozen=True)
class GraphEquivalence:
    """A graph proof outcome bound to the exact reference and DUT sources."""

    depth: int
    minimum_depth: int
    verdict: str
    dut_sha256: str
    reference_sha256: str
    reason: str | None = None
    counterexample: str | None = None

    @property
    def proven(self) -> bool:
        """Return whether the asserted graph outputs passed the full requested bound."""
        return self.verdict == "PASS"

    def to_json(self) -> dict[str, object]:
        """Return the proof record stored in the guarantee summary."""
        return {
            "mode": "bmc",
            "engine": "abc bmc3",
            "flattened": True,
            "depth": self.depth,
            "minimum_depth": self.minimum_depth,
            "reset_cycles": _RESET_CYCLES,
            "verdict": self.verdict,
            "dut_sha256": self.dut_sha256,
            "reference_sha256": self.reference_sha256,
            "reason": self.reason,
            "counterexample": self.counterexample,
        }


def _admit_operations(summary: dict[str, Any]) -> None:
    """Refuse any computation outside the exact parameters of its library evidence."""
    for entry in summary["operations"]:
        operation = entry["operation"]
        if operation in MODEL_OPERATIONS and not any(
            evidence["class"] == "FORMAL" for evidence in entry["guarantees"]
        ):
            raise ValueError(f"{operation} is outside the library's exact formal parameters")


def _digest(text: str) -> str:
    """Return the source identity exactly as passed to the proof runner."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_graph_equivalence(
    name: str,
    ir: str,
    verilog: str,
    *,
    depth: int,
    timeout_s: float = DEFAULT_GRAPH_PROOF_TIMEOUT_SECONDS,
) -> GraphEquivalence:
    """Check all visible outputs of a verified canonical IR graph with SymbiYosys.

    Parameters
    ----------
    name : str
        The verified graph's module name.
    ir : str
        Canonical IR text from the native graph printer.
    verilog : str
        The exact SystemVerilog to check and emit as the graph top.
    depth : int
        Number of frames the bounded model checker examines, including reset.
    timeout_s : float
        Limit for external proof execution, in seconds. Defaults to 300.

    Returns
    -------
    GraphEquivalence
        A proof, counterexample, unavailable tool, or inconclusive result.

    Raises
    ------
    ValueError
        If the graph or bounds cannot establish an observable comparison.
    """
    if not _IDENTIFIER.fullmatch(name) or name == "equiv_miter":
        raise ValueError("bounded IR equivalence requires an unreserved SystemVerilog module name")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("the equivalence timeout must be positive and finite")
    _admit_operations(summarize_ir_text(name, ir, verilog))
    reference = build_reference(name, ir)
    found = parse_module_interface(verilog, name)
    if found != list(reference.ports):
        raise ValueError("the emitted port list differs from the independent graph interface")
    # The miter samples registered outputs after the transition has taken effect.
    minimum = _RESET_CYCLES + 2 + reference.delay
    if depth < minimum:
        raise ValueError(
            f"equivalence depth {depth} is too short to observe this graph; need {minimum}"
        )

    from sc_neurocore.studio.resource_feedback import with_instanced_primitives

    source = with_instanced_primitives(verilog)
    source_digest, reference_digest = _digest(source), _digest(reference.verilog)
    if not formal_tools_available("yosys-abc"):
        return GraphEquivalence(
            depth,
            minimum,
            "UNAVAILABLE",
            source_digest,
            reference_digest,
            reason="SymbiYosys, Yosys and Yosys ABC (yosys-abc) are required on PATH",
        )
    with tempfile.TemporaryDirectory(prefix="scn-ir-equivalence-") as workdir:
        try:
            result = prove_equivalence(
                source,
                reference.verilog,
                found,
                dut_top=name,
                ref_top=reference.top,
                depth=depth,
                reset_cycles=_RESET_CYCLES,
                timeout_s=timeout_s,
                workdir=Path(workdir),
                flatten=True,
                abc_bmc=True,
            )
        except (OSError, RuntimeError) as error:
            reason = str(error).split("\n", 1)[0] or type(error).__name__
            return GraphEquivalence(
                depth, minimum, "INCONCLUSIVE", source_digest, reference_digest, reason=reason
            )
    if result.verdict not in {"PASS", "FAIL"} or (
        result.verdict == "PASS" and result.returncode != 0
    ):
        return GraphEquivalence(
            depth,
            minimum,
            "INCONCLUSIVE",
            source_digest,
            reference_digest,
            reason=f"solver returned {result.verdict} with exit status {result.returncode}",
        )
    return GraphEquivalence(
        depth,
        minimum,
        result.verdict,
        source_digest,
        reference_digest,
        counterexample=result.counterexample,
    )
