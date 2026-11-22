# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — External toolchain runs behind an evidence packet

"""Run the external toolchains an evidence packet records, and identify the tools and the source.

:func:`run_formal_job` runs one committed SymbiYosys job in a scratch copy, :func:`synthesize`
runs Yosys through the Studio synthesis backend and :func:`place_and_route` places and routes a
graph out of context. :func:`tool_versions` and :func:`source_identity` record what ran and from
which source. The packet's format, manifest and verification live in
:mod:`sc_neurocore.safety_cert.evidence_packet`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.compiler._sby_runner import run_sby_task
from sc_neurocore.hdl.resources import baseline_primitive_text
from sc_neurocore.safety_cert.dense_cosim import PRIMITIVES
from sc_neurocore.studio.bit_true_execution import native_tool_version
from sc_neurocore.studio.resource_feedback import FeedbackGraph

TOOLS = (
    "iverilog",
    "vvp",
    "yosys",
    "yosys-smtbmc",
    "sby",
    "nextpnr-ice40",
    "yices-smt2",
    "z3",
    "cvc5",
    "boolector",
    "bitwuzla",
)
"""External tools whose versions every packet records."""


_SOLVER_BINARIES = {"yices": "yices-smt2"}


_ASSUME = re.compile(r"\bassume\s*(?:property)?\s*\(")


@dataclass(frozen=True)
class FormalTarget:
    """One library module whose committed formal job the packet runs.

    Attributes
    ----------
    module : str
        Verilog module name.
    job : str
        Repository path of its ``.sby`` job.
    origin : str
        Why the packet runs it: named by the guarantee summary, or instantiated
        inside a co-simulated-only module.
    summary : Mapping[str, Any] or None
        The guarantee summary entry, when the summary names the module.
    """

    module: str
    job: str
    origin: str
    summary: Mapping[str, Any] | None = None


def _sha256(data: bytes) -> str:
    """Return the hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Write ``value`` as indented, key-sorted JSON with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkout_root(module_file: Path) -> Path | None:
    """Return the source checkout a package module lies in, or ``None`` when installed.

    Parameters
    ----------
    module_file : Path
        A module file of the ``sc_neurocore.safety_cert`` package.

    Returns
    -------
    Path or None
        The directory three levels above the package directory (``src/sc_neurocore``
        sits below it) when it holds ``hdl/formal`` and ``src/sc_neurocore``; ``None``
        for an installed package.
    """
    root = module_file.resolve().parents[3]
    if (root / "hdl" / "formal").is_dir() and (root / "src" / "sc_neurocore").is_dir():
        return root
    return None


def _sby_sections(text: str) -> dict[str, list[str]]:
    """Return the non-empty, non-comment lines of each section of a ``.sby`` job."""
    sections: dict[str, list[str]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return sections


def _job_sources(repository: Path, job: Path, files: Sequence[str]) -> tuple[Path, list[Path]]:
    """Resolve the ``[files]`` of a job relative to its directory or to the checkout.

    The committed jobs use both conventions; the one under which every file exists
    inside the checkout is taken, as ``tools/formal_job_inputs.py`` does.

    Raises
    ------
    ValueError
        No convention resolves every file inside the checkout.
    """
    names = [line.split()[-1] for line in files]
    for directory in (job.parent, repository):
        sources = [(directory / name).resolve() for name in names]
        if all(path.is_file() and path.is_relative_to(repository) for path in sources):
            return directory, sources
    raise ValueError("its sources are missing or outside the checkout")


def _solver_binary(engine: str) -> str | None:
    """Return the solver executable an ``smtbmc`` engine line needs, else ``None``."""
    tokens = engine.split()
    if not tokens or tokens[0] != "smtbmc":
        return None
    solvers = [token for token in tokens[1:] if not token.startswith("-")]
    solver = solvers[0] if solvers else "yices"
    return _SOLVER_BINARIES.get(solver, solver)


def run_formal_job(
    repository: Path | None,
    target: FormalTarget,
    out_dir: Path,
    *,
    timeout_s: float = 1800.0,
    scratch_parent: Path | None = None,
) -> dict[str, Any]:
    """Run one committed SymbiYosys job in a scratch copy and record its outcome.

    The job and every source it reads are copied, with their relative layout, into a
    fresh temporary directory, and ``sby`` runs there, so no run directory is ever
    written into the checkout. The log of a completed run is written to
    ``formal/<module>.log`` under ``out_dir``.

    Parameters
    ----------
    repository : Path or None
        Source checkout holding ``hdl/formal``, or ``None`` when not run from one.
    target : FormalTarget
        The module and its job.
    out_dir : Path
        Packet directory that receives the log.
    timeout_s : float, default 1800.0
        Wall-clock limit of the ``sby`` run.
    scratch_parent : Path or None, optional
        Directory for the scratch copy; the system temporary directory when omitted.

    Returns
    -------
    dict
        The record: ``status`` ``run`` with the verdict, the log path and its SHA-256,
        or ``unavailable`` with the reason; mode, depth, engine, assumptions and the
        SHA-256 of every job source whenever the job could be read.

    Raises
    ------
    ValueError
        The scratch directory lies inside the checkout, where a run would write into
        the committed job tree.
    """
    record: dict[str, Any] = {"module": target.module, "job": target.job, "origin": target.origin}
    if target.summary is not None:
        for key in ("class", "proved_parameters", "instance_parameters", "differs"):
            if key in target.summary:
                record[key] = target.summary[key]
    if repository is None or not (repository / target.job).is_file():
        record.update(
            status="unavailable",
            reason=f"{target.job} is not present: the packet was not exported from a "
            "source checkout holding the committed formal jobs",
        )
        return record
    job = repository / target.job
    sections = _sby_sections(job.read_text(encoding="utf-8"))
    options = dict(line.split(None, 1) for line in sections.get("options", []))
    engine = sections.get("engines", [""])[0]
    record.update(mode=options.get("mode"), depth=int(options.get("depth", "0")), engine=engine)
    try:
        directory, sources = _job_sources(repository, job, sections.get("files", []))
    except ValueError as error:
        record.update(status="unavailable", reason=f"{target.job}: {error}")
        return record
    texts = {path: path.read_bytes() for path in sorted({job, *sources})}
    record["sources_sha256"] = {
        path.relative_to(repository).as_posix(): _sha256(data) for path, data in texts.items()
    }
    record["assumptions"] = [
        line.strip()
        for data in texts.values()
        for line in data.decode("utf-8").splitlines()
        if _ASSUME.search(line)
    ]
    module_source = repository / "hdl" / f"{target.module}.v"
    if f"{target.module}.v" in PRIMITIVES and module_source.is_file():
        packaged = baseline_primitive_text(f"{target.module}.v").encode("utf-8")
        record["proved_source_equals_simulated_primitive"] = module_source.read_bytes() == packaged
    solver = _solver_binary(engine)
    missing = [
        name
        for name in ("sby", "yosys", *([] if solver is None else [solver]))
        if shutil.which(name) is None
    ]
    if missing:
        record.update(status="unavailable", reason=f"not installed: {', '.join(missing)}")
        return record
    with tempfile.TemporaryDirectory(prefix="sc_evidence_formal_", dir=scratch_parent) as scratch:
        root = Path(scratch).resolve()
        if root.is_relative_to(repository.resolve()):
            raise ValueError("the formal scratch directory must lie outside the checkout")
        for path, data in texts.items():
            copy = root / path.relative_to(repository)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(data)
        workdir = root / directory.relative_to(repository)
        argument = job.relative_to(directory).as_posix()
        try:
            run = run_sby_task(workdir, argument, timeout_s=timeout_s)
        except RuntimeError as error:
            record.update(status="run", verdict="TIMEOUT", reason=str(error))
            return record
    log = out_dir / "formal" / f"{target.module}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(run.stdout, encoding="utf-8")
    record.update(
        status="run",
        verdict=run.verdict,
        rc=run.rc,
        returncode=run.returncode,
        log=f"formal/{target.module}.log",
        log_sha256=_sha256(log.read_bytes()),
    )
    return record


def synthesize(verilog: str, out_dir: Path, target: str = "ice40") -> dict[str, Any]:
    """Synthesise a design with Yosys through the Studio synthesis backend.

    A report Yosys produced, successful or not, is written to
    ``synthesis/yosys_report.json`` under ``out_dir``.

    Parameters
    ----------
    verilog : str
        The top level with every library source it needs.
    out_dir : Path
        Packet directory that receives the report.
    target : str, default "ice40"
        Synthesis target identifier of :func:`sc_neurocore.studio.synthesis.run_synthesis`.

    Returns
    -------
    dict
        The record: ``completed``, ``failed`` or ``unavailable`` with the reason, and
        the report path and its SHA-256 when a report exists.
    """
    from sc_neurocore.studio.synthesis import run_synthesis

    record: dict[str, Any] = {"tool": "yosys", "target": target}
    if shutil.which("yosys") is None:
        record.update(status="unavailable", reason="yosys is not installed")
        return record
    try:
        report = run_synthesis(verilog, target)
    except ValueError as error:
        record.update(status="failed", reason=str(error))
        return record
    if report.get("success") is True:
        record.update(status="completed", reason=None)
    else:
        record.update(status="failed", reason=report.get("error"))
    path = out_dir / "synthesis" / "yosys_report.json"
    _write_json(path, report)
    record.update(report="synthesis/yosys_report.json", report_sha256=_sha256(path.read_bytes()))
    return record


def place_and_route(
    graph: FeedbackGraph,
    out_dir: Path,
    target: str = "ice40",
    clock_mhz: int = 12,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Place and route a graph out of context and set the routed result against its estimate.

    For a target :func:`~sc_neurocore.studio.resource_feedback.run_resource_feedback`
    supports, its full report is written to ``synthesis/pnr_report.json`` and the
    estimate, the routed core and the estimate check to
    ``synthesis/estimate_vs_real.json`` under ``out_dir``.

    Parameters
    ----------
    graph : FeedbackGraph
        The verified graph.
    out_dir : Path
        Packet directory that receives the reports.
    target : str, default "ice40"
        Studio target.
    clock_mhz : int, default 12
        Clock the timing is judged at; must be positive.
    timeout_seconds : float
        Limit for each Yosys and nextpnr run, set for the executing host.

    Returns
    -------
    dict
        The record: ``run`` with the overall ``verdict``, the verdict of every check
        and the paths and SHA-256 of both reports, or ``unavailable`` with the reason.
    """
    from sc_neurocore.studio.resource_feedback import ENGINE_TARGETS, run_resource_feedback

    record: dict[str, Any] = {"tool": "nextpnr-ice40", "target": target, "clock_mhz": clock_mhz}
    if target not in ENGINE_TARGETS:
        record.update(
            status="unavailable",
            reason=f"place and route is supported for {sorted(ENGINE_TARGETS)}, not {target!r}",
        )
        return record
    report = run_resource_feedback(
        graph, clock_mhz=clock_mhz, target=target, timeout_seconds=timeout_seconds
    )
    toolchain = next((c for c in report["checks"] if c["name"] == "toolchain"), None)
    if toolchain is not None:
        record.update(status="unavailable", reason=str(toolchain["detail"]))
        return record
    checks = {check["name"]: check["verdict"] for check in report["checks"]}
    comparison = {
        "estimate": report["estimate"],
        "core_logic_cells": report.get("core_logic_cells"),
        "fmax": (report.get("place_and_route") or {}).get("fmax"),
        "estimate_comparison": report.get("estimate_comparison"),
        "estimate_check": next((c for c in report["checks"] if c["name"] == "estimate"), None),
    }
    pnr_path = out_dir / "synthesis" / "pnr_report.json"
    comparison_path = out_dir / "synthesis" / "estimate_vs_real.json"
    _write_json(pnr_path, report)
    _write_json(comparison_path, comparison)
    record.update(
        status="run",
        verdict=report["verdict"],
        checks=checks,
        report="synthesis/pnr_report.json",
        report_sha256=_sha256(pnr_path.read_bytes()),
        estimate_vs_real="synthesis/estimate_vs_real.json",
        estimate_vs_real_sha256=_sha256(comparison_path.read_bytes()),
    )
    return record


def tool_versions() -> dict[str, Any]:
    """Return the version of every external tool and of Python and both packages.

    Returns
    -------
    dict
        ``tools`` (availability and first version line of each of :data:`TOOLS`),
        ``python``, ``sc_neurocore`` and ``sc_neurocore_engine``.

    Raises
    ------
    ImportError
        The ``sc_neurocore_engine`` package is not installed.
    """
    from sc_neurocore._version import VERSION

    tools: dict[str, dict[str, Any]] = {}
    for name in TOOLS:
        path = shutil.which(name)
        tools[name] = {
            "available": path is not None,
            "version": None if path is None else native_tool_version(path, name),
        }
    import sc_neurocore_engine

    engine = str(getattr(sc_neurocore_engine, "__version__", "unreported"))
    return {
        "tools": tools,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "sc_neurocore": VERSION,
        "sc_neurocore_engine": engine,
    }


def source_identity(repository: Path | None) -> dict[str, Any]:
    """Return the package version and, from a Git checkout, its commit and dirty flag.

    Parameters
    ----------
    repository : Path or None
        Directory the package runs from, or ``None`` when it is installed.

    Returns
    -------
    dict
        ``package_version`` and ``checkout``; with a checkout, ``commit`` and ``dirty``
        (uncommitted or untracked files), otherwise the ``reason``.
    """
    from sc_neurocore._version import VERSION

    identity: dict[str, Any] = {"package_version": VERSION}
    git = shutil.which("git")
    if repository is None or git is None:
        reason = "git is not installed" if git is None else "not run from a source checkout"
        identity.update(checkout=False, reason=reason)
        return identity

    def ask(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # nosec B603
            [git, "-C", str(repository), *arguments],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )

    top = ask("rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != repository.resolve():
        identity.update(checkout=False, reason="not run from a Git checkout")
        return identity
    status = ask("status", "--porcelain")
    identity.update(
        checkout=True,
        commit=ask("rev-parse", "HEAD").stdout.strip(),
        dirty=bool(status.stdout.strip()),
    )
    return identity
