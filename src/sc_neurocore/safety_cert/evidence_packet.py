# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hash-bound evidence packet of one generated dense design

"""Export and verify a hash-bound evidence packet for one generated dense layer design.

:func:`export_evidence_packet` takes the IR text of a dense layer graph (the shape
:mod:`sc_neurocore.safety_cert.dense_cosim` supports) and writes into one directory:

``design.sc``, ``design.sv``
    The IR text exactly as given and the SystemVerilog the engine emits from it.
``design.guarantees.json``, ``design.guarantees.txt``
    The guarantee summary of :mod:`sc_neurocore.compiler.guarantee_summary`.
``parameters.json``
    Q formats, seeds of every encoder, geometry and constants read from the IR.
``cosim/``
    The stimulus, the Python reference trace, the RTL trace, the testbench and the
    verdict with the first differing cycle.
``formal/``
    For every library module the summary classifies ``FORMAL`` or ``PROVEN AT OTHER
    PARAMETERS``, and for every library module inside a ``CO-SIMULATED ONLY`` module,
    its committed SymbiYosys job run in a scratch copy outside the checkout, with
    verdict, mode, depth, engine, assumptions and digests of the log and job sources;
    a job that cannot run here is recorded as unavailable with the reason. Each
    co-simulated-only module is recorded with that boundary.
``synthesis/``
    The Yosys report of :func:`sc_neurocore.studio.synthesis.run_synthesis` for the
    top level with its primitives, or the reason it is unavailable.
``tools.json``, ``source.json``
    Tool, Python, package and engine versions; the package version and, from a Git
    checkout, its commit and whether the tree is dirty.
``evidence.json``, ``evidence.txt``
    The verdicts and the boundary statement: what the packet establishes and what it
    does not.
``manifest.json``
    Every other member with its SHA-256 and size, and the content digest of that list
    computed by :class:`sc_neurocore.safety_cert.evidence.EvidenceBag`.

:func:`verify_evidence_packet` re-reads and re-hashes every member and refuses a missing
member, an extra file, an altered member, a symbolic link or a manifest whose digest
does not match its list.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sc_neurocore.compiler._sby_runner import run_sby_task
from sc_neurocore.safety_cert.dense_cosim import (
    PRIMITIVES,
    CosimResult,
    cosimulate,
    parse_dense_design,
)
from sc_neurocore.compiler.guarantee_summary import render_guarantee_text, summarize_graph
from sc_neurocore.hdl.resources import baseline_primitive_text
from sc_neurocore.safety_cert.evidence import EvidenceBag, EvidenceItem
from sc_neurocore.studio.bit_true_execution import native_tool_version

__all__ = [
    "CONTAINED_MODULES",
    "MANIFEST",
    "SCHEMA",
    "FormalTarget",
    "PacketExport",
    "PacketVerification",
    "boundary_statement",
    "checkout_root",
    "export_evidence_packet",
    "formal_targets",
    "run_formal_job",
    "source_identity",
    "synthesize",
    "tool_versions",
    "verify_evidence_packet",
]

SCHEMA = "sc-neurocore.evidence-packet.v1"
"""Schema identifier of the manifest and the evidence summary."""

MANIFEST = "manifest.json"
"""File name of the manifest inside a packet."""

CONTAINED_MODULES: Mapping[str, tuple[str, ...]] = {
    # sc_dense_matrix_layer.v lines 98, 119, 134, 143 and 155.
    "sc_dense_matrix_layer": (
        "sc_bitstream_encoder",
        "sc_bitstream_synapse",
        "sc_dotproduct_to_current",
        "sc_lif_neuron",
    ),
}
"""Library modules each co-simulated-only module instantiates inside itself."""

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
_CATEGORIES = (
    ("design.", "design", "Design artefact"),
    ("parameters.json", "design", "Q formats, seeds and constants read from the IR"),
    ("cosim/", "test", "Co-simulation of the emitted top level against the Python reference"),
    ("formal/", "formal", "Formal job record or log"),
    ("synthesis/", "report", "Synthesis record or report"),
    ("tools.json", "report", "Tool, Python, package and engine versions"),
    ("source.json", "report", "Package version and source checkout identity"),
    ("evidence.", "report", "Verdicts and boundary statement"),
)


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


@dataclass(frozen=True)
class PacketExport:
    """Outcome of :func:`export_evidence_packet`.

    Attributes
    ----------
    directory : Path
        The packet directory.
    manifest_sha256 : str
        Content digest of the manifest's member list.
    members : int
        Number of members listed in the manifest.
    cosim_agrees : bool
        Whether the RTL trace equalled the reference on every cycle.
    first_difference : int or None
        First differing cycle, or ``None``.
    """

    directory: Path
    manifest_sha256: str
    members: int
    cosim_agrees: bool
    first_difference: int | None


@dataclass(frozen=True)
class PacketVerification:
    """Outcome of :func:`verify_evidence_packet`.

    Attributes
    ----------
    directory : Path
        The checked directory.
    ok : bool
        Whether no problem was found.
    problems : tuple of str
        Every problem found, one per line.
    members : int
        Number of members the manifest lists.
    manifest_sha256 : str or None
        The digest the manifest records, when it could be read.
    """

    directory: Path
    ok: bool
    problems: tuple[str, ...]
    members: int
    manifest_sha256: str | None


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


def formal_targets(summary: Mapping[str, Any]) -> tuple[list[FormalTarget], list[dict[str, Any]]]:
    """Return the formal jobs a summary calls for and the co-simulated-only boundaries.

    Parameters
    ----------
    summary : Mapping[str, Any]
        A guarantee summary of :func:`sc_neurocore.compiler.guarantee_summary.summarize_ir_text`.

    Returns
    -------
    tuple
        The jobs to run, one per module, and one boundary record per co-simulated-only
        operation.
    """
    targets: dict[str, FormalTarget] = {}
    boundaries: list[dict[str, Any]] = []
    for entry in summary["operations"]:
        for guarantee in entry["guarantees"]:
            kind = guarantee["class"]
            if kind in {"FORMAL", "PROVEN AT OTHER PARAMETERS"}:
                targets.setdefault(
                    entry["module"],
                    FormalTarget(
                        entry["module"],
                        guarantee["job"],
                        f"named by the guarantee summary for {entry['id']} {entry['operation']}",
                        guarantee,
                    ),
                )
            elif kind == "CO-SIMULATED ONLY":
                module = entry["module"]
                contained = CONTAINED_MODULES.get(module, ())
                boundaries.append(
                    {
                        "module": module,
                        "operation": f"{entry['id']} {entry['operation']}",
                        "class": kind,
                        "test": guarantee["test"],
                        "contained_modules": list(contained),
                        "statement": (
                            f"{module} has no formal job of its own. Its composition is "
                            "established only by co-simulation; the formal jobs of the "
                            "modules it contains prove each of them alone, in its own "
                            "harness."
                        ),
                    }
                )
                for name in contained:
                    targets.setdefault(
                        name,
                        FormalTarget(
                            name, f"hdl/formal/{name}.sby", f"instantiated inside {module}"
                        ),
                    )
    return list(targets.values()), boundaries


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

    record: dict[str, Any] = {
        "tool": "yosys",
        "target": target,
        "place_and_route": {
            "status": "not run",
            "reason": "run_synthesis performs Yosys synthesis only; no place and route "
            "was run, so no routed resource or timing figures exist",
        },
    }
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
    import sc_neurocore

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
        "sc_neurocore": sc_neurocore.__version__,
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
    import sc_neurocore

    identity: dict[str, Any] = {"package_version": sc_neurocore.__version__}
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


def boundary_statement(
    cosim: Mapping[str, Any],
    formal: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    synthesis: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Say what a packet establishes and what it does not.

    Parameters
    ----------
    cosim : Mapping[str, Any]
        Co-simulation verdict of :meth:`CosimResult.verdict`.
    formal : Sequence[Mapping[str, Any]]
        Formal job records of :func:`run_formal_job`.
    boundaries : Sequence[Mapping[str, Any]]
        Co-simulated-only records of :func:`formal_targets`.
    synthesis : Mapping[str, Any]
        Synthesis record of :func:`synthesize`.

    Returns
    -------
    dict
        ``establishes`` and ``does_not_establish``, each a list of sentences.
    """
    establishes = [
        "design.sc is the IR text given to the exporter and design.sv is the SystemVerilog "
        "this installation's engine emitted from it; manifest.json binds every member by "
        "its SHA-256 and size.",
    ]
    if cosim["verdict"] == "AGREE":
        establishes.append(
            "Simulated by Icarus Verilog with the packaged library primitives, the emitted "
            "top level matched the Python bit-true reference exactly (spikes, every "
            f"membrane word, running and run_done) on all {cosim['cycles']} cycles of the "
            "recorded stimulus."
        )
    else:
        establishes.append(
            "The emitted top level and the Python bit-true reference DIFFER, first at "
            f"cycle {cosim['first_difference']} of the recorded stimulus."
        )
    missing: list[str] = []
    for record in formal:
        if record.get("status") != "run":
            missing.append(
                f"No formal result for {record['module']}: {record['job']} was not run "
                f"({record['reason']})."
            )
        elif record.get("verdict") != "PASS":
            missing.append(
                f"No formal result for {record['module']}: {record['job']} ran with verdict "
                f"{record.get('verdict')}, which proves nothing."
            )
        else:
            establishes.append(
                f"{record['job']} ran for {record['module']} with verdict PASS "
                f"({record['mode']}, depth {record['depth']}, engine "
                f"{record['engine'] or 'default'}), in its own harness at the harness's "
                "parameters, under the recorded assumptions."
            )
        if record.get("proved_source_equals_simulated_primitive") is False:
            missing.append(
                f"No formal result for the simulated {record['module']}: {record['job']} "
                f"reads hdl/{record['module']}.v, which differs from the packaged "
                f"{record['module']}.v primitive the co-simulation used."
            )
    if synthesis["status"] == "completed":
        establishes.append(
            f"Yosys synthesis for the {synthesis['target']} target completed; its resource "
            "counts are in synthesis/yosys_report.json."
        )
    else:
        missing.append(
            f"No synthesis result: synthesis {synthesis['status']} ({synthesis['reason']})."
        )
    excluded = [
        *missing,
        "No unbounded proof: each formal verdict holds only to its stated depth, for the "
        "module in its own harness at the harness's parameters and under its assumptions; "
        "a passing prove-mode job is a k-induction proof of that harness only.",
        "No formal proof of the generated top level, nor of a library module at "
        "parameters other than those its harness proves.",
        "No statement about inputs, cycles or seeds other than the recorded stimulus, "
        "and no error bound on the stochastic bitstream estimates.",
        "No measured power, timing or behaviour on hardware, and no place and route.",
        "Synthesis figures are estimates of the named tool and version for the named "
        "target, not measurements.",
        "No authenticity: the manifest is a digest list, not a signature. Verification "
        "detects a changed, missing or added member, but anyone can rewrite a packet "
        "together with its manifest; compare manifest_sha256 with a copy held "
        "independently of the packet.",
    ]
    for boundary in boundaries:
        excluded.append(boundary["statement"])
    return {"establishes": establishes, "does_not_establish": excluded}


def _render_text(summary: Mapping[str, Any]) -> str:
    """Render the evidence summary for people."""
    lines = [
        f"Evidence packet of {summary['graph']} ({summary['schema']})",
        f"Co-simulation: {summary['cosim']['verdict']} over {summary['cosim']['cycles']} cycles",
    ]
    for record in summary["formal"]:
        outcome = record.get("verdict") or f"unavailable ({record.get('reason')})"
        lines.append(f"Formal {record['module']}: {outcome}")
    synthesis = summary["synthesis"]
    lines.append(f"Synthesis ({synthesis['target']}): {synthesis['status']}")
    lines.append("")
    lines.append("This packet establishes:")
    lines += [f"- {sentence}" for sentence in summary["boundary"]["establishes"]]
    lines.append("")
    lines.append("This packet does not establish:")
    lines += [f"- {sentence}" for sentence in summary["boundary"]["does_not_establish"]]
    return "\n".join(lines) + "\n"


def _category(path: str) -> tuple[str, str]:
    """Return the evidence category and description of a member path."""
    return next((c, d) for prefix, c, d in _CATEGORIES if path.startswith(prefix))


def _write_cosim(directory: Path, result: CosimResult) -> dict[str, Any]:
    """Write the co-simulation members and return the verdict."""
    cosim = directory / "cosim"
    _write_json(cosim / "stimulus.json", [list(words) for words in result.stimulus])
    _write_json(cosim / "reference_trace.json", [row.to_json() for row in result.reference])
    _write_json(cosim / "rtl_trace.json", [row.to_json() for row in result.rtl])
    (cosim / "testbench.sv").write_text(result.testbench, encoding="utf-8")
    verdict = result.verdict()
    _write_json(cosim / "verdict.json", verdict)
    return verdict


def export_evidence_packet(
    graph_text: str,
    out_dir: Path,
    *,
    cycles: int = 256,
    synthesis_target: str = "ice40",
    formal_timeout_s: float = 1800.0,
    repository: Path | None = None,
) -> PacketExport:
    """Write the evidence packet of one dense layer design into a new directory.

    Parameters
    ----------
    graph_text : str
        IR text of a supported dense layer graph.
    out_dir : Path
        Directory to create; an existing one must be empty.
    cycles : int, default 256
        Cycles of the co-simulation stimulus.
    synthesis_target : str, default "ice40"
        Yosys synthesis target.
    formal_timeout_s : float, default 1800.0
        Wall-clock limit of each formal job.
    repository : Path or None, optional
        Source checkout holding the formal jobs and the Git identity; the checkout this
        package runs from when omitted, if any.

    Returns
    -------
    PacketExport
        Directory, manifest digest, member count and co-simulation outcome.

    Raises
    ------
    ValueError
        The text is not valid IR, fails verification, is not a supported dense design
        (:class:`~sc_neurocore.safety_cert.dense_cosim.UnsupportedDesignError` names the
        unsupported operations), or ``out_dir`` exists and is not an empty directory.
    sc_neurocore.safety_cert.dense_cosim.SimulatorUnavailableError
        Icarus Verilog is not installed.
    sc_neurocore.safety_cert.dense_cosim.CosimError
        The simulator fails.
    """
    from sc_neurocore_engine import parse_ir

    graph = parse_ir(graph_text)
    problems = graph.verify()
    if problems:
        raise ValueError("the graph fails IR verification: " + "; ".join(problems))
    design = parse_dense_design(graph.to_text())
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} exists and is not an empty directory")
    verilog = graph.emit_sv()
    result = cosimulate(design, verilog, cycles=cycles)
    repository = checkout_root(Path(__file__)) if repository is None else repository.resolve()
    # Read before any member exists, so a packet written inside the checkout does not
    # mark it dirty.
    identity = source_identity(repository)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "design.sc").write_text(graph_text, encoding="utf-8")
    (out_dir / "design.sv").write_text(verilog, encoding="utf-8")
    summary = summarize_graph(graph)
    _write_json(out_dir / "design.guarantees.json", summary)
    (out_dir / "design.guarantees.txt").write_text(render_guarantee_text(summary), encoding="utf-8")
    _write_json(out_dir / "parameters.json", design.to_parameters())
    cosim = _write_cosim(out_dir, result)

    targets, boundaries = formal_targets(summary)
    formal_records = [
        run_formal_job(repository, target, out_dir, timeout_s=formal_timeout_s)
        for target in targets
    ]
    _write_json(
        out_dir / "formal" / "formal.json",
        {"jobs": formal_records, "co_simulated_only": boundaries},
    )

    source = verilog + "".join(baseline_primitive_text(name) for name in PRIMITIVES)
    synthesis = synthesize(source, out_dir, synthesis_target)
    _write_json(out_dir / "synthesis" / "synthesis.json", synthesis)

    _write_json(out_dir / "tools.json", tool_versions())
    _write_json(out_dir / "source.json", identity)
    evidence = {
        "schema": SCHEMA,
        "graph": design.name,
        "cosim": cosim,
        "formal": formal_records,
        "co_simulated_only": boundaries,
        "synthesis": synthesis,
        "boundary": boundary_statement(cosim, formal_records, boundaries, synthesis),
    }
    _write_json(out_dir / "evidence.json", evidence)
    (out_dir / "evidence.txt").write_text(_render_text(evidence), encoding="utf-8")

    bag = EvidenceBag()
    members: list[dict[str, Any]] = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        name = path.relative_to(out_dir).as_posix()
        category, description = _category(name)
        data = path.read_bytes()
        bag.add(EvidenceItem(name, category, description, _sha256(data)))
        members.append(
            {
                "path": name,
                "category": category,
                "description": description,
                "sha256": _sha256(data),
                "size": len(data),
            }
        )
    digest = bag.content_sha256()
    _write_json(
        out_dir / MANIFEST,
        {"schema": SCHEMA, "graph": design.name, "members": members, "manifest_sha256": digest},
    )
    return PacketExport(out_dir, digest, len(members), result.agrees, result.first_difference)


def _read_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return the parsed manifest, or the reason it cannot be used."""
    if path.is_symlink() or not path.is_file():
        return None, f"{MANIFEST} is missing or not a regular file"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"{MANIFEST} is not valid JSON: {error}"
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or not isinstance(manifest.get("members"), list)
    ):
        return None, f"{MANIFEST} does not follow schema {SCHEMA}"
    return manifest, None


def verify_evidence_packet(directory: Path) -> PacketVerification:
    """Re-read and re-hash every member of an evidence packet.

    Parameters
    ----------
    directory : Path
        The packet directory.

    Returns
    -------
    PacketVerification
        ``ok`` only when the manifest's digest matches its member list, every listed
        member is a regular file with the listed SHA-256 and size, and the directory
        holds no other file, no directory that holds no member, no entry that is not a
        regular file (such as a FIFO or a socket) and no symbolic link. No such entry is
        ever opened.
    """
    if directory.is_symlink() or not directory.is_dir():
        return PacketVerification(directory, False, (f"{directory} is not a directory",), 0, None)
    manifest, reason = _read_manifest(directory / MANIFEST)
    if manifest is None:
        return PacketVerification(directory, False, (str(reason),), 0, None)
    problems: list[str] = []
    recorded = manifest.get("manifest_sha256")
    bag = EvidenceBag()
    listed: dict[str, Mapping[str, Any]] = {}
    for entry in manifest["members"]:
        try:
            bag.add(
                EvidenceItem(
                    entry["path"], entry["category"], entry["description"], entry["sha256"]
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"invalid manifest entry {entry!r}: {error}")
            continue
        listed[entry["path"]] = entry
    if bag.content_sha256() != recorded:
        problems.append(f"{MANIFEST} digest does not match its member list")
    folders_of_members = {
        parent.as_posix()
        for name in listed
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    present: set[str] = set()
    for current, folders, files in os.walk(directory, followlinks=False):
        for name in folders:
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                problems.append(f"symbolic link: {relative}")
            elif relative not in folders_of_members:
                problems.append(f"extra directory: {relative}")
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                problems.append(f"symbolic link: {relative}")
            elif not path.is_file():
                problems.append(f"extra entry that is not a regular file: {relative}")
            elif relative != MANIFEST:
                present.add(relative)
    for name in sorted(present - set(listed)):
        problems.append(f"extra file: {name}")
    for name, entry in sorted(listed.items()):
        path = directory.joinpath(*PurePosixPath(name).parts)
        if name not in present:
            if not path.is_symlink():
                problems.append(f"missing member: {name}")
            continue
        data = path.read_bytes()
        if _sha256(data) != entry["sha256"] or len(data) != entry.get("size"):
            problems.append(f"altered member: {name}")
    return PacketVerification(
        directory, not problems, tuple(problems), len(manifest["members"]), recorded
    )
