# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NotImplemented guard audit

"""Audit executable ``NotImplementedError`` raises in tracked Python sources."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True, order=True)
class _RaiseSite:
    """Executable ``NotImplementedError`` raise location and message fingerprint."""

    path: str
    qualname: str
    message_fragment: str


_APPROVED_NOTIMPLEMENTED_GUARDS = {
    _RaiseSite(
        "src/sc_neurocore/drivers/sc_neurocore_driver.py",
        "SC_NeuroCore_Driver.run_step",
        "HARDWARE DMA transfer requires PYNQ overlay",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "_build_population_spec",
        "noise_std == 0.0",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "_build_population_spec",
        "refractory_period == 0",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "_build_population_spec",
        "external entropy_source",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "_build_population_spec",
        "v_reset == v_rest",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "_build_population_spec",
        "does not support model",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "NetworkTorchBridge.__init__",
        "plastic projections",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/_torch_bridge.py",
        "NetworkTorchBridge.__init__",
        "delayed projections",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network._raise_for_rust_incompatibilities",
        "Rust backend does not support",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network._run_mpi",
        "spike_gating is not supported",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network._run_mpi",
        "fim_lambda > 0",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network._run_mpi",
        "embedded stimuli",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network._run_mpi",
        "state monitors",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network._run_mpi",
        "synaptic plasticity",
    ),
    _RaiseSite(
        "src/sc_neurocore/network/network.py",
        "Network.to_torch",
        "embedded stimuli",
    ),
    _RaiseSite(
        "src/sc_neurocore/neurons/base.py",
        "BaseNeuron.get_state",
        "<abstract>",
    ),
    _RaiseSite(
        "src/sc_neurocore/neurons/base.py",
        "BaseNeuron.reset_state",
        "<abstract>",
    ),
    _RaiseSite(
        "src/sc_neurocore/neurons/base.py",
        "BaseNeuron.step",
        "<abstract>",
    ),
    _RaiseSite(
        "src/sc_neurocore/nir_bridge/node_map.py",
        "map_node",
        "not yet supported",
    ),
    _RaiseSite(
        "src/sc_neurocore/optics/photonic_emitter.py",
        "CompilationResult.to_gdsii",
        "num_modulators > 0",
    ),
}


def _tracked_python_sources() -> list[Path]:
    """Return tracked package Python files, excluding generated dependency trees."""
    result = subprocess.run(
        ["git", "ls-files", "src/sc_neurocore/**/*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def _qualname_for(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Return the dotted class/function path enclosing ``node``."""
    names: list[str] = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(current.name)
    return ".".join(reversed(names))


def _message_fragment(exc: ast.expr | None) -> str:
    """Return the stable literal fragment carried by a NotImplementedError raise."""
    if isinstance(exc, ast.Name):
        return "<abstract>"
    if not isinstance(exc, ast.Call):
        return "<unknown>"
    if not exc.args:
        return "<abstract>"
    first_arg = exc.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    if isinstance(first_arg, ast.JoinedStr):
        return "".join(
            value.value
            for value in first_arg.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    return "<dynamic>"


def _is_notimplemented_raise(node: ast.Raise) -> bool:
    """Return whether ``node`` raises ``NotImplementedError`` directly."""
    exc = node.exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def _notimplemented_sites(path: Path) -> list[_RaiseSite]:
    """Return approved-site fingerprints for one tracked Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    sites: list[_RaiseSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and _is_notimplemented_raise(node):
            message = _message_fragment(node.exc)
            sites.append(
                _RaiseSite(
                    path.as_posix(),
                    _qualname_for(node, parents),
                    message,
                )
            )
    return sites


def _raise_expression(source: str) -> ast.expr | None:
    """Return the exception expression from a one-line raise statement."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            return node.exc
    raise AssertionError(f"source does not contain a raise statement: {source}")


def test_message_fragment_fallbacks_are_explicit() -> None:
    """The audit helper exposes stable labels for nonliteral raise arguments."""
    assert _message_fragment(ast.parse("errors[0]", mode="eval").body) == "<unknown>"
    assert _message_fragment(_raise_expression("raise NotImplementedError()")) == "<abstract>"
    assert _message_fragment(
        _raise_expression("raise NotImplementedError(reason)")
    ) == "<dynamic>"


def test_raise_expression_requires_raise_statement() -> None:
    """Raise-expression fixtures fail loudly when the input has no raise."""
    with pytest.raises(AssertionError, match="does not contain a raise statement"):
        _raise_expression("value = 1")


def test_notimplemented_errors_are_approved_fail_fast_guards() -> None:
    """Every executable NotImplementedError is an approved explicit guard."""
    observed = {
        site
        for path in _tracked_python_sources()
        for site in _notimplemented_sites(path)
        if any(
            site.path == approved.path
            and site.qualname == approved.qualname
            and approved.message_fragment in site.message_fragment
            for approved in _APPROVED_NOTIMPLEMENTED_GUARDS
        )
    }
    all_sites = {
        site for path in _tracked_python_sources() for site in _notimplemented_sites(path)
    }

    assert all_sites == observed
    assert len(all_sites) == len(_APPROVED_NOTIMPLEMENTED_GUARDS)
