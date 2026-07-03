# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Machine-checked Python↔RTL equivalence runner

"""Run a machine-checked equivalence proof between a compiled module and a reference.

Wraps the SymbiYosys bounded-model-checking flow into a single call: given the
device-under-test Verilog (the compiler's generated RTL), an independent
reference module, and their shared interface, this builds a sequential miter
(see :mod:`sc_neurocore.compiler.equivalence_miter`), emits the ``.sby`` script,
invokes ``sby`` via the shared runner (:mod:`sc_neurocore.compiler._sby_runner`),
and parses the verdict into an :class:`EquivalenceResult`.

A ``PASS`` verdict is a real proof — for *every* input sequence up to the checked
depth the two modules produce identical outputs — not a sampled simulation. A
``FAIL`` returns the failing assertion and the counterexample-trace path.

Bounded model checking establishes equivalence only up to ``depth`` cycles from
reset. Unbounded proof by ``mode="prove"`` (k-induction) is offered but is not
the default: for datapaths with wide signed multipliers (e.g. the fixed-point
neuron update) the induction step reports spurious counterexamples from
unreachable mid-states unless the reachable-state invariant is supplied, so a
bounded proof to a solver-tractable depth is the honest default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ._sby_runner import (
    SbyRun,
    formal_tools_available,
    parse_verdict,
    raise_for_incomplete,
    run_sby_task,
)
from .equivalence_miter import MiterPort, build_equivalence_miter

__all__ = [
    "EquivalenceResult",
    "formal_tools_available",
    "prove_equivalence",
]

# Re-exported for callers and tests that reach the shared verdict parser through
# this module; the canonical definition lives in ``_sby_runner``.
_parse_verdict = parse_verdict


@dataclass
class EquivalenceResult:
    """Outcome of a machine-checked equivalence proof.

    Attributes
    ----------
    proven : bool
        ``True`` when the checker proved output equivalence to ``depth`` cycles.
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


def _generate_sby(
    miter_top: str,
    source_files: list[str],
    *,
    depth: int,
    mode: str,
    engine: str,
) -> str:
    """Render a ``.sby`` script that reads the sources and checks the miter."""
    reads = "\n".join(f"read -formal {name}" for name in source_files)
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
        f"{reads}\n"
        f"prep -top {miter_top}\n"
        "\n"
        "[files]\n" + "\n".join(source_files) + "\n"
    )


def _result_from_run(run: SbyRun, *, mode: str, depth: int, engine: str) -> EquivalenceResult:
    """Map a raw :class:`SbyRun` onto an :class:`EquivalenceResult`.

    A ``PASS`` yields ``proven=True``; a ``FAIL`` yields ``proven=False`` with the
    counterexample; an inconclusive k-induction result (``mode="prove"`` whose
    induction step did not converge) yields ``proven=False`` with the ``UNKNOWN``
    verdict and *no* counterexample — the modules may still be equivalent.

    Raises
    ------
    RuntimeError
        On a tool or setup failure (``ERROR`` / crash), which is not a verdict
        about equivalence.
    """
    raise_for_incomplete(run, what="equivalence proof")
    if run.verdict == "PASS":
        return EquivalenceResult(
            proven=True,
            verdict=run.verdict,
            mode=mode,
            depth=depth,
            engine=engine,
            returncode=run.returncode,
            summary=run.summary,
        )
    if run.verdict == "FAIL":
        return EquivalenceResult(
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
    return EquivalenceResult(
        proven=False,
        verdict=run.verdict,
        mode=mode,
        depth=depth,
        engine=engine,
        returncode=run.returncode,
        summary=run.summary,
    )


def prove_equivalence(
    dut_verilog: str,
    ref_verilog: str,
    io_ports: list[MiterPort],
    *,
    dut_top: str,
    ref_top: str,
    dut_params: dict[str, int] | None = None,
    ref_params: dict[str, int] | None = None,
    depth: int = 6,
    engine: str = "z3",
    mode: Literal["bmc", "prove"] = "bmc",
    reset_cycles: int = 2,
    clock: str = "clk",
    reset_n: str = "rst_n",
    timeout_s: float = 300.0,
    workdir: str | Path | None = None,
) -> EquivalenceResult:
    """Prove ``dut_verilog`` equivalent to ``ref_verilog`` via SymbiYosys.

    Parameters
    ----------
    dut_verilog, ref_verilog : str
        Verilog sources defining ``dut_top`` and ``ref_top`` respectively.
    io_ports : list[MiterPort]
        Shared interface passed to :func:`build_equivalence_miter`.
    dut_top, ref_top : str
        Module names under verification (must differ).
    dut_params, ref_params : dict[str, int], optional
        Per-instance parameter overrides.
    depth : int
        BMC / induction depth in clock cycles.
    engine : str
        SMT engine (``"z3"`` by default; the ``smtbmc`` backend).
    mode : {"bmc", "prove"}
        Bounded model checking or k-induction.
    reset_cycles : int
        Leading clocks to hold reset before comparing.
    clock, reset_n : str
        Clock and active-low reset port names.
    timeout_s : float
        Wall-clock limit for the ``sby`` process.
    workdir : str or Path, optional
        Directory for the generated sources and ``sby`` run tree. A temporary
        directory is created and left in place if omitted (caller cleans up).

    Returns
    -------
    EquivalenceResult
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

    miter_top = "equiv_miter"
    miter = build_equivalence_miter(
        dut_top,
        ref_top,
        io_ports,
        miter_name=miter_top,
        dut_params=dut_params,
        ref_params=ref_params,
        reset_cycles=reset_cycles,
        clock=clock,
        reset_n=reset_n,
    )

    work = Path(workdir) if workdir is not None else Path.cwd() / f"{miter_top}_work"
    work.mkdir(parents=True, exist_ok=True)

    dut_file, ref_file, miter_file = f"{dut_top}.v", f"{ref_top}.v", f"{miter_top}.v"
    (work / dut_file).write_text(dut_verilog, encoding="utf-8")
    (work / ref_file).write_text(ref_verilog, encoding="utf-8")
    (work / miter_file).write_text(miter, encoding="utf-8")
    sby_name = f"{miter_top}.sby"
    (work / sby_name).write_text(
        _generate_sby(
            miter_top,
            [miter_file, ref_file, dut_file],
            depth=depth,
            mode=mode,
            engine=engine,
        ),
        encoding="utf-8",
    )

    run = run_sby_task(work, sby_name, timeout_s=timeout_s)
    return _result_from_run(run, mode=mode, depth=depth, engine=engine)
