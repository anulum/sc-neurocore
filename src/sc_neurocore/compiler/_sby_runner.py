# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared SymbiYosys task runner

"""Run a single SymbiYosys task and read its verdict.

This is the one place that shells out to ``sby``. Both the Python↔RTL
equivalence runner (:mod:`sc_neurocore.compiler.equivalence_check`) and the
RTL-property runner (:mod:`sc_neurocore.compiler.formal_property_check`) build
their own ``.sby`` script — a sequential miter for the former, an RTL-plus-bound-
assertions check for the latter — write their sources into a work directory, and
then hand the finished task to :func:`run_sby_task`. Keeping the invocation,
verdict parse, and counterexample extraction here means those callers share one
audited subprocess boundary rather than each re-implementing it.

Responsibility boundary: this module knows *how to run one already-written ``.sby``
task and interpret its output*. It does not know what is being proved, so it never
builds a miter, emits a monitor, or decides pass/fail semantics — the callers map
the raw :class:`SbyRun` onto their own result type.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "SbyRun",
    "formal_tools_available",
    "is_inconclusive",
    "parse_verdict",
    "raise_for_incomplete",
    "run_sby_task",
]

# SymbiYosys return code for an inconclusive task: the outcome is neither a proof
# nor a counterexample. In particular a k-induction (``mode prove``) run whose base
# case holds but whose induction step does not converge reports ``UNKNOWN`` with
# this code — the property may hold yet was not proved.
_INCONCLUSIVE_RC = 4

_SUMMARY_RE = re.compile(
    r"DONE \((?P<verdict>PASS|FAIL|ERROR|UNKNOWN|TIMEOUT|[A-Z]+), rc=(?P<rc>\d+)\)"
)
_ASSERT_RE = re.compile(r"failed assertion .*", re.IGNORECASE)
_TRACE_RE = re.compile(r"counterexample trace:\s*(?P<path>\S+)")


def formal_tools_available(engine: str = "z3") -> bool:
    """Return ``True`` when the full proof toolchain is on ``PATH``.

    Checks ``sby`` and ``yosys`` plus the SMT solver binary for ``engine`` — for
    the ``smtbmc`` backend the engine name is also the solver executable name
    (``z3``, ``boolector``, ``yices`` …). A runner may have ``sby`` and ``yosys``
    installed yet lack the solver (as on a CI image that ships only the HDL
    toolchain), in which case a proof would error out at the engine stage; this
    guard reports that case as unavailable so callers and tests can skip cleanly.

    Parameters
    ----------
    engine : str
        The ``smtbmc`` SMT engine whose solver binary must also be present.

    Returns
    -------
    bool
        ``True`` only when ``sby``, ``yosys``, and the ``engine`` solver all
        resolve on ``PATH``.
    """
    return (
        shutil.which("sby") is not None
        and shutil.which("yosys") is not None
        and shutil.which(engine) is not None
    )


def parse_verdict(stdout: str) -> tuple[str, int]:
    """Extract the ``(verdict, rc)`` from the ``sby`` summary.

    SymbiYosys prints one ``DONE (<verdict>, rc=<n>)`` line per finished task; the
    last such line is the authoritative outcome (an earlier ``DONE (ERROR, ...)``
    can precede a retried task). Returns ``("UNKNOWN", -1)`` when no ``DONE`` line
    is present at all — a truncated or crashed run.

    Parameters
    ----------
    stdout : str
        The captured ``sby`` standard output.

    Returns
    -------
    tuple[str, int]
        The verdict token and the ``sby`` return code it reported.
    """
    match = None
    for match in _SUMMARY_RE.finditer(stdout):
        pass  # keep the last DONE line
    if match is None:
        return "UNKNOWN", -1
    return match.group("verdict"), int(match.group("rc"))


@dataclass
class SbyRun:
    """The raw outcome of one :func:`run_sby_task` invocation.

    Attributes
    ----------
    verdict : str
        The parsed SymbiYosys verdict (``"PASS"``, ``"FAIL"``, ``"ERROR"``, ...).
    rc : int
        The return code reported inside the ``DONE (..., rc=N)`` summary line.
    returncode : int
        The ``sby`` process exit code (distinct from ``rc``: the process may exit
        non-zero on a ``FAIL`` while ``rc`` classifies the proof outcome).
    counterexample : str or None
        The failing-assertion description when the output carries one, else
        ``None``.
    trace_path : str or None
        Absolute path to the counterexample trace (resolved against the work
        directory) when the output names one, else ``None``.
    summary : list[str]
        The ``sby`` ``summary:`` lines, retained for diagnostics.
    stdout : str
        The full captured standard output, retained for error reporting.
    """

    verdict: str
    rc: int
    returncode: int
    counterexample: str | None = None
    trace_path: str | None = None
    summary: list[str] = field(default_factory=list)
    stdout: str = ""


def run_sby_task(workdir: Path, sby_name: str, *, timeout_s: float) -> SbyRun:
    """Run one already-written ``.sby`` task in ``workdir`` and read its verdict.

    The ``.sby`` script and every source file it reads must already exist in
    ``workdir``; this call only invokes ``sby -f <sby_name>`` there, captures the
    output, and parses it into an :class:`SbyRun`. Counterexample text and trace
    path are extracted opportunistically — they are populated regardless of
    verdict when present, and the caller decides which fields matter for its
    result type.

    Parameters
    ----------
    workdir : Path
        Directory holding the ``.sby`` script and its sources; also the ``sby``
        run tree and the base for resolving a relative counterexample trace path.
    sby_name : str
        File name of the ``.sby`` script within ``workdir``.
    timeout_s : float
        Wall-clock limit for the ``sby`` process.

    Returns
    -------
    SbyRun
        The parsed run outcome.

    Raises
    ------
    RuntimeError
        If the ``sby`` process exceeds ``timeout_s``.
    """
    try:
        proc = subprocess.run(
            ["sby", "-f", sby_name],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"sby task '{sby_name}' timed out after {timeout_s}s") from exc

    stdout = proc.stdout or ""
    verdict, rc = parse_verdict(stdout)
    summary = [line for line in stdout.splitlines() if "summary:" in line]

    assert_match = _ASSERT_RE.search(stdout)
    trace_match = _TRACE_RE.search(stdout)
    trace_path = str(workdir / trace_match.group("path")) if trace_match is not None else None

    return SbyRun(
        verdict=verdict,
        rc=rc,
        returncode=proc.returncode,
        counterexample=assert_match.group(0) if assert_match is not None else None,
        trace_path=trace_path,
        summary=summary,
        stdout=stdout,
    )


def is_inconclusive(run: SbyRun) -> bool:
    """Return ``True`` for an inconclusive result — proved nothing, disproved nothing.

    A k-induction (``mode prove``) run whose base case holds but whose induction
    step does not converge reports ``UNKNOWN`` with :data:`_INCONCLUSIVE_RC`: the
    property may well be true, but was not proved and no counterexample was found
    (the induction step reached the goal from an *unreachable* predecessor). This
    is a real, honest outcome distinct from both a disproof and a tool failure.

    Parameters
    ----------
    run : SbyRun
        The raw run to inspect.

    Returns
    -------
    bool
        ``True`` only for the ``UNKNOWN`` / inconclusive-return-code signature.
    """
    return run.verdict == "UNKNOWN" and run.rc == _INCONCLUSIVE_RC


def raise_for_incomplete(run: SbyRun, *, what: str) -> None:
    """Raise when the run is a tool or setup failure, not a verdict about the design.

    A ``PASS`` (proved) or ``FAIL`` (disproved with a counterexample) is a
    conclusive outcome, and an inconclusive k-induction result
    (:func:`is_inconclusive`) is a real — if unhelpful — outcome; none of these
    raise. Any *other* verdict — ``ERROR``, a crash with no ``DONE`` line, a
    timeout — is a tool or setup failure that must not be read as a claim about
    the design, so callers invoke this before interpreting a result.

    Parameters
    ----------
    run : SbyRun
        The raw run to inspect.
    what : str
        Short label for the check (``"equivalence proof"``, ``"property proof"``)
        used in the raised message.

    Raises
    ------
    RuntimeError
        When the run neither proved, disproved, nor came back inconclusive.
    """
    if run.verdict in ("PASS", "FAIL") or is_inconclusive(run):
        return
    tail = "\n".join(run.stdout.splitlines()[-15:])
    raise RuntimeError(f"{what} did not complete (verdict={run.verdict}, rc={run.rc}):\n{tail}")
