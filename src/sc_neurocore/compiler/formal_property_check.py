# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Machine-checked RTL property runner

"""Machine-check that an RTL module satisfies a bound SystemVerilog assertion set.

Where :mod:`sc_neurocore.compiler.equivalence_check` proves two modules produce
identical outputs, this runner proves a *single* module satisfies its own safety
obligations: the SVA file carries ``assume`` constraints on the environment and
``assert`` obligations bound (via SystemVerilog ``bind``) onto the design's
internal state. Given the RTL source, the assertion source, and the top module
name, it emits a ``.sby`` script, invokes ``sby`` through the shared runner
(:mod:`sc_neurocore.compiler._sby_runner`), and parses the verdict into a
:class:`PropertyProofResult`.

A ``PASS`` is a real proof — for every environment behaviour allowed by the
``assume`` constraints, the ``assert`` obligations hold to the checked depth. A
``FAIL`` returns the failing assertion and the counterexample-trace path.

Bounded model checking (``mode="bmc"``) is the default and is *complete* for a
design that reaches a stationary state within the checked depth — for a bounded
accumulator that stops updating after a fixed number of steps, a BMC depth past
that step count exhausts the reachable state space. k-induction (``mode="prove"``)
is offered for genuinely unbounded designs but is not the default: an accumulator
invariant such as ``acc <= BOUND`` is not inductive on its own (the successor
state ``acc + step`` can exceed ``BOUND`` from an unreachable predecessor), so it
reports spurious counterexamples without a strengthening reachable-state invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ._sby_runner import formal_tools_available, raise_for_incomplete, run_sby_task

__all__ = ["PropertyProofResult", "formal_tools_available", "prove_property"]


@dataclass
class PropertyProofResult:
    """Outcome of a machine-checked RTL property proof.

    Attributes
    ----------
    proven : bool
        ``True`` when the checker proved every bound assertion holds to ``depth``
        cycles under the SVA environment assumptions.
    verdict : str
        The raw SymbiYosys verdict (``"PASS"``, ``"FAIL"``, ``"ERROR"``, ...).
    mode : str
        ``"bmc"`` (bounded) or ``"prove"`` (k-induction).
    depth : int
        Checked depth in clock cycles.
    engine : str
        SMT engine used (e.g. ``"z3"``).
    returncode : int
        ``sby`` process exit code.
    counterexample : str or None
        Failing-assertion description on a ``FAIL`` verdict, else ``None``.
    trace_path : str or None
        Path to the counterexample VCD trace on ``FAIL``, else ``None``.
    summary : list[str]
        The ``sby`` summary lines, retained for diagnostics.
    """

    proven: bool
    verdict: str
    mode: str
    depth: int
    engine: str
    returncode: int
    counterexample: str | None = None
    trace_path: str | None = None
    summary: list[str] = field(default_factory=list)


def _generate_property_sby(
    top: str,
    rtl_file: str,
    sva_file: str,
    *,
    depth: int,
    mode: str,
    engine: str,
) -> str:
    """Render a ``.sby`` script that reads the RTL and its bound SVA and checks it.

    The RTL is read first so the ``bind`` statement in the SystemVerilog assertion
    file resolves against an already-elaborated target module; ``prep -top`` then
    keeps the bound checker instance in the design handed to ``smtbmc``.
    """
    return (
        "[tasks]\n"
        f"{mode}\n"
        "\n"
        "[options]\n"
        f"{mode}: mode {mode}\n"
        f"{mode}: depth {depth}\n"
        "\n"
        "[engines]\n"
        f"smtbmc {engine}\n"
        "\n"
        "[script]\n"
        f"read -formal {rtl_file}\n"
        f"read -sv -formal {sva_file}\n"
        f"prep -top {top}\n"
        "\n"
        "[files]\n"
        f"{rtl_file}\n"
        f"{sva_file}\n"
    )


def prove_property(
    rtl_verilog: str,
    sva_verilog: str,
    *,
    top: str,
    mode: Literal["bmc", "prove"] = "bmc",
    depth: int = 32,
    engine: str = "z3",
    timeout_s: float = 300.0,
    workdir: str | Path | None = None,
) -> PropertyProofResult:
    """Prove ``rtl_verilog`` satisfies the assertions in ``sva_verilog`` via SymbiYosys.

    Parameters
    ----------
    rtl_verilog : str
        Synthesisable Verilog source defining ``top``.
    sva_verilog : str
        SystemVerilog source carrying the environment ``assume`` constraints and
        the ``assert`` obligations, bound onto ``top`` with a ``bind`` statement.
    top : str
        The RTL module name under verification.
    mode : {"bmc", "prove"}
        Bounded model checking (default, complete for state-stationary designs) or
        k-induction.
    depth : int
        BMC / induction depth in clock cycles.
    engine : str
        SMT engine (``"z3"`` by default; the ``smtbmc`` backend).
    timeout_s : float
        Wall-clock limit for the ``sby`` process.
    workdir : str or Path, optional
        Directory for the generated sources and ``sby`` run tree. A temporary
        directory is created and left in place if omitted (caller cleans up).

    Returns
    -------
    PropertyProofResult
        The parsed verdict.

    Raises
    ------
    RuntimeError
        If the formal tools are absent, the ``sby`` run errors out (a tool or
        setup failure, distinct from a ``FAIL`` disproof), or times out.
    """
    if not formal_tools_available(engine):
        raise RuntimeError(
            f"SymbiYosys ('sby'), Yosys ('yosys') and the '{engine}' SMT solver must be on PATH"
        )

    work = Path(workdir) if workdir is not None else Path.cwd() / f"{top}_property_work"
    work.mkdir(parents=True, exist_ok=True)

    rtl_file, sva_file = f"{top}.v", f"{top}_sva.sv"
    (work / rtl_file).write_text(rtl_verilog, encoding="utf-8")
    (work / sva_file).write_text(sva_verilog, encoding="utf-8")
    sby_name = f"{top}.sby"
    (work / sby_name).write_text(
        _generate_property_sby(top, rtl_file, sva_file, depth=depth, mode=mode, engine=engine),
        encoding="utf-8",
    )

    run = run_sby_task(work, sby_name, timeout_s=timeout_s)
    raise_for_incomplete(run, what="property proof")
    if run.verdict == "PASS":
        return PropertyProofResult(
            proven=True,
            verdict=run.verdict,
            mode=mode,
            depth=depth,
            engine=engine,
            returncode=run.returncode,
            summary=run.summary,
        )
    return PropertyProofResult(
        proven=False,
        verdict=run.verdict,
        mode=mode,
        depth=depth,
        engine=engine,
        returncode=run.returncode,
        counterexample=run.counterexample or "assertion failed",
        trace_path=run.trace_path,
        summary=run.summary,
    )
