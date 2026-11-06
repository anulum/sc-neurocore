# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Guarantee summary of an emitted IR top level

"""State, for every operation of an IR graph, what its emitted hardware is guaranteed by.

The IR compiler writes a SystemVerilog top level that instantiates library modules
from ``hdl/`` for some operations and writes the logic of the others inline. This
module classifies each operation of a graph into the classes below and writes the
result next to the emitted top level, as JSON for tools and as short text for people.

``FORMAL``
    The operation instantiates a library module that has a tracked SymbiYosys job of
    its own, and the instance has exactly the parameters the job's harness proves.
    The entry names the job, its mode and depth, the engine, how many assertions the
    harness checks, the assumptions it makes and the parameters.
``PROVEN AT OTHER PARAMETERS``
    The module has its own formal job, but the instance differs from the harness in
    at least one parameter, so the proof does not cover this instance. The entry
    gives both parameter sets and the names that differ.
``CO-SIMULATED ONLY``
    The operation instantiates a library module with no formal job of its own, but a
    committed test simulates the emitted RTL against its Python reference. The entry
    names that test; no facet receipt exists for library modules.
``PROBABILISTIC``
    The operation produces or consumes a stochastic bitstream, so its value is an
    estimate whose error depends on the stream length and seed recorded in the entry.
    No error bound is established yet for LFSR streams, so ``bound`` stays empty.
``NONE``
    Inline logic or port wiring written by the compiler, with no separate proof or
    receipt of its own.

An operation can carry two classes: an encoder instantiates a formally checked module
and produces a stochastic estimate. The parameters of each instance are read from the
emitted SystemVerilog itself; a parameter the instance does not set takes the
module's default. The evidence below is static package data; the repository's tests
check every entry against the tracked jobs, harnesses and modules, so a missing job,
a changed depth or parameter, or an unclassified operation fails continuous
integration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "GUARANTEE_CLASSES",
    "LIBRARY_MODULES",
    "OPERATION_RULES",
    "SCHEMA",
    "CosimEvidence",
    "FormalEvidence",
    "IrGraph",
    "LibraryModule",
    "OperationRule",
    "parse_instances",
    "render_guarantee_text",
    "summarize_graph",
    "summarize_ir_text",
    "verilog_value",
    "write_guarantee_summary",
]

SCHEMA = "sc-neurocore.guarantee-summary.v2"
"""Schema identifier written into every summary."""

GUARANTEE_CLASSES = (
    "FORMAL",
    "PROVEN AT OTHER PARAMETERS",
    "CO-SIMULATED ONLY",
    "PROBABILISTIC",
    "NONE",
)
"""Guarantee classes in the order the summary counts them."""


@dataclass(frozen=True)
class FormalEvidence:
    """A tracked SymbiYosys job that checks one library module on its own.

    Attributes
    ----------
    job : str
        Repository path of the ``.sby`` job.
    mode : str
        SymbiYosys mode, ``prove`` (induction) or ``bmc`` (bounded).
    depth : int
        Induction or bounded-model-check depth in clock cycles.
    engine : str
        Engine line of the job, including an explicit solver when it names one.
    assertions : int
        Assertions in the job's harness.
    assumptions : tuple of str
        Assumption statements of the harness, as written.
    parameters : Mapping[str, str]
        Every module parameter as the harness instantiates it, as written.
    defaults : Mapping[str, str]
        The module's own parameter defaults, as written in its declaration.
    """

    job: str
    mode: str
    depth: int
    engine: str
    assertions: int
    assumptions: tuple[str, ...]
    parameters: Mapping[str, str]
    defaults: Mapping[str, str]


@dataclass(frozen=True)
class CosimEvidence:
    """A committed test that simulates a library module's emitted RTL against Python.

    Attributes
    ----------
    test : str
        Repository path of the co-simulation test.
    """

    test: str


@dataclass(frozen=True)
class LibraryModule:
    """A library module the IR compiler instantiates, with the evidence it has.

    Attributes
    ----------
    name : str
        Verilog module name.
    evidence : FormalEvidence or CosimEvidence
        Its own formal job, or, when it has none, its co-simulation test.
    """

    name: str
    evidence: FormalEvidence | CosimEvidence


@dataclass(frozen=True)
class OperationRule:
    """How the IR compiler emits one operation keyword.

    Attributes
    ----------
    module : str or None
        Library module the operation instantiates, or ``None`` for inline logic.
    stochastic : bool
        Whether the operation's value is a stochastic-bitstream estimate.
    emitted : str
        What the compiler writes into the top level for it.
    """

    module: str | None
    stochastic: bool
    emitted: str


LIBRARY_MODULES: Mapping[str, LibraryModule] = {
    "sc_bitstream_encoder": LibraryModule(
        "sc_bitstream_encoder",
        FormalEvidence(
            job="hdl/formal/sc_bitstream_encoder.sby",
            mode="prove",
            depth=20,
            engine="smtbmc",
            assertions=2,
            assumptions=("if (init) assume(!rst_n);",),
            parameters={"DATA_WIDTH": "16", "LFSR_WIDTH": "16", "SEED_INIT": "16'hACE1"},
            defaults={"DATA_WIDTH": "16", "LFSR_WIDTH": "16", "SEED_INIT": "16'hACE1"},
        ),
    ),
    "sc_lif_neuron": LibraryModule(
        "sc_lif_neuron",
        FormalEvidence(
            job="hdl/formal/sc_lif_neuron.sby",
            mode="prove",
            depth=25,
            engine="smtbmc",
            assertions=5,
            assumptions=("if (!past_valid) assume(!rst_n);",),
            parameters={
                "DATA_WIDTH": "16",
                "FRACTION": "8",
                "V_REST": "0",
                "V_RESET": "0",
                "V_THRESHOLD": "16'sd256",
                "REFRACTORY_PERIOD": "2",
            },
            defaults={
                "DATA_WIDTH": "16",
                "FRACTION": "8",
                "V_REST": "0",
                "V_RESET": "0",
                "V_THRESHOLD": "(1 << FRACTION)",
                "REFRACTORY_PERIOD": "0",
            },
        ),
    ),
    "sc_dcls_layer_core": LibraryModule(
        "sc_dcls_layer_core",
        FormalEvidence(
            job="hdl/formal/sc_dcls_layer_core.sby",
            mode="bmc",
            depth=16,
            engine="smtbmc cvc5",
            assertions=5,
            assumptions=(),
            parameters={
                "N_TAPS": "3",
                "DATA_WIDTH": "16",
                "FRACTION": "8",
                "ACC_WIDTH": "32",
                "DELAY_DEPTH": "4",
                "PTR_WIDTH": "2",
            },
            defaults={
                "N_TAPS": "3",
                "DATA_WIDTH": "16",
                "FRACTION": "8",
                "ACC_WIDTH": "32",
                "DELAY_DEPTH": "31",
                "PTR_WIDTH": "5",
            },
        ),
    ),
    "sc_dense_matrix_layer": LibraryModule(
        "sc_dense_matrix_layer", CosimEvidence("tests/test_ir_dense_rtl.py")
    ),
    "sc_graph_forward": LibraryModule(
        "sc_graph_forward", CosimEvidence("tests/test_graph_forward_rtl_cosim.py")
    ),
    "sc_softmax_attention": LibraryModule(
        "sc_softmax_attention", CosimEvidence("tests/test_softmax_attention_rtl_cosim.py")
    ),
    "sc_kuramoto_step": LibraryModule(
        "sc_kuramoto_step", CosimEvidence("tests/test_kuramoto_step_rtl_cosim.py")
    ),
}
"""Library modules the IR compiler instantiates, keyed by module name."""

OPERATION_RULES: Mapping[str, OperationRule] = {
    "input": OperationRule(None, False, "input port"),
    "output": OperationRule(None, False, "output port assignment"),
    "constant": OperationRule(None, False, "constant literal"),
    "state": OperationRule(None, False, "register with asynchronous reset"),
    "store": OperationRule(None, False, "next-state commit of a register"),
    "check": OperationRule(None, False, "simulation check, excluded from synthesis"),
    "word": OperationRule(None, False, "inline fixed-point arithmetic"),
    "scale": OperationRule(None, False, "inline fixed-point multiply"),
    "offset": OperationRule(None, False, "inline fixed-point add"),
    "div_const": OperationRule(None, False, "inline constant divider"),
    "reduce": OperationRule(None, False, "inline vector reduction"),
    "encode": OperationRule("sc_bitstream_encoder", True, "instance of sc_bitstream_encoder"),
    "and": OperationRule(None, True, "inline bitwise AND (stochastic multiply)"),
    "xor": OperationRule(None, True, "inline bitwise XOR"),
    "popcount": OperationRule(None, True, "inline bit counter"),
    "decode": OperationRule(None, True, "inline counter scaled to Q8.8"),
    "lif_step": OperationRule("sc_lif_neuron", False, "instance of sc_lif_neuron"),
    "dense_forward": OperationRule(
        "sc_dense_matrix_layer", False, "instance of sc_dense_matrix_layer"
    ),
    "dcls_layer": OperationRule("sc_dcls_layer_core", False, "instance of sc_dcls_layer_core"),
    "graph_forward": OperationRule("sc_graph_forward", False, "instance of sc_graph_forward"),
    "softmax_attention": OperationRule(
        "sc_softmax_attention", False, "instance of sc_softmax_attention"
    ),
    "kuramoto_step": OperationRule("sc_kuramoto_step", False, "instance of sc_kuramoto_step"),
}
"""Every IR operation keyword the compiler emits, with how it is emitted."""

_LINE = re.compile(r"^\s*(?:%(?P<id>\d+)\s*=\s*)?sc\.(?P<op>[a-z_]+)\b(?P<rest>.*)$")
_STREAM = re.compile(r"bitstream<(?P<length>\d+)>")
_SEED = re.compile(r"\bseed=(?P<seed>0x[0-9A-Fa-f]+|\d+)")
_OPERAND = re.compile(r"%(\d+)")
_INSTANCE = re.compile(
    r"^\s*(?P<module>sc_[a-z0-9_]+)\s*#\((?P<body>.*?)\)\s*\w+\s*\(", re.M | re.S
)
_ASSIGNMENT = re.compile(r"\.(?P<name>\w+)\s*\((?P<value>[^()]*(?:\([^()]*\)[^()]*)*)\)")
_LITERAL = re.compile(
    r"^(?P<sign>-?)\s*(?:(?P<size>\d+)?\s*'\s*(?P<signed>[sS]?)(?P<base>[bBoOdDhH])\s*)?"
    r"(?P<digits>[0-9a-fA-F_]+)$"
)
_BASES = {"b": 2, "o": 8, "d": 10, "h": 16}


class IrGraph(Protocol):
    """The part of an engine ``ScGraph`` the summary reads."""

    @property
    def name(self) -> str:
        """Graph name, also the emitted module name."""

    def to_text(self) -> str:
        """Return the graph's IR text."""

    def emit_sv(self) -> str:
        """Return the emitted SystemVerilog top level."""


def verilog_value(text: str) -> int | str:
    """Return the integer a Verilog parameter value denotes, or its text when not a literal.

    Parameters
    ----------
    text : str
        Parameter value as written, such as ``256``, ``16'sd256`` or ``16'hACE1``.

    Returns
    -------
    int or str
        The literal's value; an expression is returned unchanged, stripped of spaces.
    """
    match = _LITERAL.match(text.strip())
    if match is None:
        return "".join(text.split())
    base = _BASES[(match.group("base") or "d").lower()]
    try:
        value = int(match.group("digits").replace("_", ""), base)
    except ValueError:
        return "".join(text.split())
    return -value if match.group("sign") else value


def parse_instances(verilog: str) -> list[tuple[str, dict[str, str]]]:
    """Return every library-module instance of a top level, with its parameter assignments.

    Parameters
    ----------
    verilog : str
        Emitted SystemVerilog.

    Returns
    -------
    list of tuple
        ``(module, {parameter: value as written})`` in emission order.
    """
    return [
        (
            match.group("module"),
            {
                assignment.group("name"): assignment.group("value").strip()
                for assignment in _ASSIGNMENT.finditer(match.group("body"))
            },
        )
        for match in _INSTANCE.finditer(verilog)
    ]


def _probabilistic(
    operation: str, rest: str, operands: list[str], streams: Mapping[str, int]
) -> dict[str, Any]:
    """Return the PROBABILISTIC entry of a stochastic operation.

    Parameters
    ----------
    operation : str
        Operation keyword.
    rest : str
        Text of the IR line after the keyword.
    operands : list of str
        Value identifiers the operation reads.
    streams : Mapping[str, int]
        Bitstream length of each value identifier produced so far.

    Returns
    -------
    dict
        The class, the stream parameters found in the graph and an empty bound.
    """
    lengths = sorted({streams[operand] for operand in operands if operand in streams})
    own = _STREAM.search(rest)
    if operation == "encode" and own is not None:
        lengths = [int(own.group("length"))]
    parameters: dict[str, Any] = {"stream_lengths": lengths}
    seed = _SEED.search(rest)
    if seed is not None:
        parameters["seed"] = seed.group("seed")
    return {
        "class": "PROBABILISTIC",
        "parameters": parameters,
        "bound": None,
        "basis": "no error bound is established for LFSR bitstreams yet",
    }


def _module_guarantee(module: LibraryModule, instance: Mapping[str, str]) -> dict[str, Any]:
    """Return the module entry of one instance, judged by its effective parameters.

    Parameters
    ----------
    module : LibraryModule
        The instantiated library module.
    instance : Mapping[str, str]
        Parameter assignments of the instance in the emitted top level.

    Returns
    -------
    dict
        A FORMAL, PROVEN AT OTHER PARAMETERS or CO-SIMULATED ONLY entry.
    """
    evidence = module.evidence
    if isinstance(evidence, CosimEvidence):
        return {"class": "CO-SIMULATED ONLY", "test": evidence.test, "receipt": None}
    effective = {**evidence.defaults, **instance}
    differs = sorted(
        name
        for name in evidence.parameters.keys() | effective.keys()
        if verilog_value(evidence.parameters.get(name, ""))
        != verilog_value(effective.get(name, ""))
    )
    entry: dict[str, Any] = {
        "class": "FORMAL" if not differs else "PROVEN AT OTHER PARAMETERS",
        "job": evidence.job,
        "mode": evidence.mode,
        "depth": evidence.depth,
        "engine": evidence.engine,
        "assertions": evidence.assertions,
        "assumptions": list(evidence.assumptions),
        "proved_parameters": dict(evidence.parameters),
        "instance_parameters": effective,
    }
    if differs:
        entry["differs"] = differs
    return entry


def summarize_ir_text(name: str, text: str, verilog: str) -> dict[str, Any]:
    """Classify every operation of a graph from its IR text and emitted top level.

    Parameters
    ----------
    name : str
        Graph name, which is also the emitted module name.
    text : str
        The graph's IR text, as ``ScGraph.to_text`` writes it.
    verilog : str
        The SystemVerilog emitted for that graph, whose instances give the parameters.

    Returns
    -------
    dict
        The summary: schema, graph name, SHA-256 of the IR text and of the top level,
        one entry per operation in graph order, and the number of entries per class.

    Raises
    ------
    ValueError
        An operation keyword has no classification, or an operation's module has no
        matching instance in the top level.
    """
    instances: dict[str, list[dict[str, str]]] = {}
    for module, parameters in parse_instances(verilog):
        instances.setdefault(module, []).append(parameters)
    operations: list[dict[str, Any]] = []
    streams: dict[str, int] = {}
    for line in text.splitlines():
        match = _LINE.match(line)
        if match is None or match.group("op") == "graph":
            continue
        operation, rest, value = match.group("op"), match.group("rest"), match.group("id")
        rule = OPERATION_RULES.get(operation)
        if rule is None:
            raise ValueError(f"IR operation sc.{operation} has no guarantee classification")
        operands = _OPERAND.findall(rest)
        guarantees: list[dict[str, Any]] = []
        if rule.module is not None:
            remaining = instances.get(rule.module, [])
            if not remaining:
                raise ValueError(
                    f"the top level has no instance of {rule.module} for sc.{operation}"
                )
            guarantees.append(_module_guarantee(LIBRARY_MODULES[rule.module], remaining.pop(0)))
        if rule.stochastic:
            guarantees.append(_probabilistic(operation, rest, operands, streams))
        if not guarantees:
            guarantees.append({"class": "NONE", "reason": rule.emitted})
        stream = _STREAM.search(rest)
        if value is not None:
            if stream is not None:
                streams[value] = int(stream.group("length"))
            elif operation in {"and", "xor"}:
                inherited = [streams[operand] for operand in operands if operand in streams]
                if inherited:
                    streams[value] = inherited[0]
        operations.append(
            {
                "id": None if value is None else f"%{value}",
                "operation": f"sc.{operation}",
                "module": rule.module,
                "emitted": rule.emitted,
                "guarantees": guarantees,
            }
        )
    counts = {name_: 0 for name_ in GUARANTEE_CLASSES}
    for entry in operations:
        for guarantee in entry["guarantees"]:
            counts[guarantee["class"]] += 1
    return {
        "schema": SCHEMA,
        "graph": name,
        "ir_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sv_sha256": hashlib.sha256(verilog.encode("utf-8")).hexdigest(),
        "operations": operations,
        "classes": counts,
    }


def summarize_graph(graph: IrGraph) -> dict[str, Any]:
    """Classify every operation of an engine IR graph.

    Parameters
    ----------
    graph : IrGraph
        A built or parsed ``sc_neurocore_engine.ScGraph``.

    Returns
    -------
    dict
        The summary of :func:`summarize_ir_text` for the graph's IR text and its
        emitted top level.
    """
    return summarize_ir_text(graph.name, graph.to_text(), graph.emit_sv())


def _parameters(values: Mapping[str, str]) -> str:
    """Return parameter assignments as ``NAME=value`` pairs."""
    return ", ".join(f"{name}={value}" for name, value in values.items())


def _describe(guarantee: Mapping[str, Any]) -> str:
    """Return one guarantee entry as a short phrase."""
    kind = guarantee["class"]
    if kind in {"FORMAL", "PROVEN AT OTHER PARAMETERS"}:
        assumptions = "; ".join(guarantee["assumptions"]) or "none"
        job = (
            f"{guarantee['job']} ({guarantee['mode']}, depth {guarantee['depth']}, "
            f"{guarantee['engine']}; {guarantee['assertions']} assertions; "
            f"assumptions: {assumptions})"
        )
        if kind == "FORMAL":
            return f"FORMAL {job} at {_parameters(guarantee['proved_parameters'])}"
        differs = ", ".join(guarantee["differs"])
        return (
            f"PROVEN AT OTHER PARAMETERS {job}: proved at "
            f"{_parameters(guarantee['proved_parameters'])}, instance has "
            f"{_parameters(guarantee['instance_parameters'])} (differs: {differs})"
        )
    if kind == "CO-SIMULATED ONLY":
        return f"CO-SIMULATED ONLY {guarantee['test']} (no receipt)"
    if kind == "PROBABILISTIC":
        parameters = guarantee["parameters"]
        lengths = ", ".join(str(length) for length in parameters["stream_lengths"]) or "unknown"
        seed = f", seed {parameters['seed']}" if "seed" in parameters else ""
        return f"PROBABILISTIC (stream length {lengths}{seed}; no error bound established)"
    return f"NONE ({guarantee['reason']})"


def render_guarantee_text(summary: Mapping[str, Any]) -> str:
    """Render a summary as short text, one line per operation.

    Parameters
    ----------
    summary : Mapping[str, Any]
        A summary returned by :func:`summarize_ir_text`.

    Returns
    -------
    str
        Header, one line per operation and the class counts.
    """
    lines = [f"Guarantee summary of {summary['graph']} ({summary['schema']})"]
    for entry in summary["operations"]:
        label = entry["id"] or "-"
        phrases = "; ".join(_describe(guarantee) for guarantee in entry["guarantees"])
        lines.append(f"{label} {entry['operation']}: {phrases}")
    counts = ", ".join(f"{name} {count}" for name, count in summary["classes"].items())
    lines.append(f"Classes: {counts}")
    return "\n".join(lines) + "\n"


def write_guarantee_summary(graph: IrGraph, sv_path: Path) -> tuple[Path, Path]:
    """Write the summary of a graph next to its emitted top level.

    Parameters
    ----------
    graph : IrGraph
        The graph whose emitted SystemVerilog is at ``sv_path``.
    sv_path : Path
        Path of the emitted ``.sv`` file.

    Returns
    -------
    tuple of Path
        The JSON and text files, ``<stem>.guarantees.json`` and ``<stem>.guarantees.txt``.
    """
    summary = summarize_graph(graph)
    json_path = sv_path.with_name(f"{sv_path.stem}.guarantees.json")
    text_path = sv_path.with_name(f"{sv_path.stem}.guarantees.txt")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_guarantee_text(summary), encoding="utf-8")
    return json_path, text_path
