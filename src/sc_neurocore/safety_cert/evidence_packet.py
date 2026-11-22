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
    top level with its primitives, or the reason it is unavailable; for the iCE40
    target, the place-and-route report of
    :func:`sc_neurocore.studio.resource_feedback.run_resource_feedback`
    (``pnr_report.json``) and the engine's static estimate set against the routed
    result (``estimate_vs_real.json``).
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

The toolchain runs a packet records live in
:mod:`sc_neurocore.safety_cert.evidence_toolchain`; this module re-exports their public
names as the same objects.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sc_neurocore.safety_cert.dense_cosim import (
    PRIMITIVES,
    CosimResult,
    cosimulate,
    parse_dense_design,
)
from sc_neurocore.compiler.guarantee_summary import render_guarantee_text, summarize_graph
from sc_neurocore.hdl.resources import baseline_primitive_text
from sc_neurocore.safety_cert.evidence import EvidenceBag, EvidenceItem
from sc_neurocore.safety_cert.evidence_toolchain import (
    TOOLS as TOOLS,
    FormalTarget as FormalTarget,
    _sha256,
    _write_json,
    checkout_root as checkout_root,
    place_and_route as place_and_route,
    run_formal_job as run_formal_job,
    source_identity as source_identity,
    synthesize as synthesize,
    tool_versions as tool_versions,
)

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
    "place_and_route",
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


_CATEGORIES = (
    ("design.", "design", "Design artefact"),
    ("parameters.json", "design", "Q formats, seeds and constants read from the IR"),
    ("cosim/", "test", "Co-simulation of the emitted top level against the Python reference"),
    ("formal/", "formal", "Formal job record or log"),
    ("synthesis/pnr_report.json", "report", "Place-and-route resource and timing feedback"),
    (
        "synthesis/estimate_vs_real.json",
        "report",
        "Engine static estimate against the routed result",
    ),
    ("synthesis/", "report", "Synthesis record or report"),
    ("tools.json", "report", "Tool, Python, package and engine versions"),
    ("source.json", "report", "Package version and source checkout identity"),
    ("evidence.", "report", "Verdicts and boundary statement"),
)


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
        Synthesis record of :func:`synthesize`, with the record of
        :func:`place_and_route` under ``place_and_route`` when one was made.

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
    routed = synthesis.get("place_and_route")
    if routed is None:
        missing.append("No place and route: none was run for this packet.")
    elif routed["status"] != "run":
        missing.append(f"No place and route: {routed['reason']}.")
    else:
        checks = ", ".join(f"{name} {verdict}" for name, verdict in routed["checks"].items())
        establishes.append(
            f"nextpnr-ice40 placed and routed the design out of context, in a five-pin "
            f"shift-register harness, for the {routed['target']} target at "
            f"{routed['clock_mhz']} MHz with resource-feedback verdict {routed['verdict']} "
            f"({checks}); the routed figures and the engine's static estimate set against "
            "them are in synthesis/pnr_report.json and synthesis/estimate_vs_real.json."
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
        "No measured power, timing or behaviour on hardware. Routed timing is the "
        "place-and-route tool's static analysis of the core in its harness, and package "
        "I/O is judged by the core's port count, not on a board.",
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
    routed = synthesis["place_and_route"]
    outcome = routed.get("verdict") or f"unavailable ({routed.get('reason')})"
    lines.append(f"Place and route ({routed['target']}, {routed['clock_mhz']} MHz): {outcome}")
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
    clock_mhz: int = 12,
    formal_timeout_s: float = 1800.0,
    repository: Path | None = None,
    pnr_timeout_s: float,
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
        Yosys synthesis and place-and-route target.
    clock_mhz : int, default 12
        Clock the routed timing is judged at; must be positive.
    formal_timeout_s : float, default 1800.0
        Wall-clock limit of each formal job.
    repository : Path or None, optional
        Source checkout holding the formal jobs and the Git identity; the checkout this
        package runs from when omitted, if any.
    pnr_timeout_s : float
        Limit for each Yosys and nextpnr run of the place and route, set for the
        executing host.

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
    if clock_mhz <= 0:
        raise ValueError("clock_mhz must be positive")
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
    synthesis["place_and_route"] = place_and_route(
        graph, out_dir, synthesis_target, clock_mhz, timeout_seconds=pnr_timeout_s
    )
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
