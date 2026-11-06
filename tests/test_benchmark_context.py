# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for benchmark measurement context

"""Verify the actual host context recorded with benchmark measurements."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from benchmarks._benchmark_context import measurement_context


def test_measurement_context_records_reproducibility_contract() -> None:
    """Benchmark artefacts must preserve isolation and host-load evidence."""
    load_before = [1.0, 2.0, 3.0]
    context = measurement_context(load_before)

    assert context["load_average_before"] == load_before
    assert "taskset affinity only" in str(context["cpu_isolation"])
    assert "non-exclusive workstation run" in str(context["concurrent_load_status"])
    assert "local regression context" in str(context["timing_interpretation"])
    assert "reserved isolated cores" in str(context["production_rerun_requirement"])
    assert "cpu_affinity" in context
    assert "load_average_after" in context
    assert "cpu_governor" in context
    assert "cpu_frequency_mhz" in context

    assert "host" not in context
    assert context["architecture"]
    assert context["cpu_model"]
    assert context["logical_cpu_count"]
    assert context["operating_system"]
    assert context["python_version"]
    assert context["numpy_version"]


_HOST_NAME_CALLS = {
    ("platform", "node"),
    ("socket", "gethostname"),
    ("socket", "getfqdn"),
}
_UNAME_HOST_FIELDS = {("os", "uname"): "nodename", ("platform", "uname"): "node"}


def _module_names(tree: ast.AST) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Map the names a module binds to the modules and functions they stand for.

    Parameters
    ----------
    tree : ast.AST
        Parsed module.

    Returns
    -------
    tuple of dict
        Local module names, including aliases, to module names; and local names
        bound by ``from module import function`` to that module and function.
    """
    modules: dict[str, str] = {}
    functions: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                functions[alias.asname or alias.name] = (node.module, alias.name)
    return modules, functions


def _host_name_reads(tree: ast.AST) -> list[int]:
    """Return the lines of ``tree`` that read the machine host name.

    Module aliases and ``from`` imports are resolved, so ``import socket as s``
    followed by ``s.gethostname()`` is found like ``socket.gethostname()``.

    Parameters
    ----------
    tree : ast.AST
        Parsed module.

    Returns
    -------
    list of int
        Lines calling ``platform.node``, ``socket.gethostname`` or ``socket.getfqdn``,
        importing one of them by name, or reading the host field of an
        ``os.uname()`` or ``platform.uname()`` result by attribute or by index 1.

    Notes
    -----
    A static scan cannot follow a host name read in two steps (``u = os.uname()``
    then ``u.nodename``), through ``getattr`` with a computed name, through
    ``__import__`` or from the ``HOSTNAME`` environment variable.
    """
    modules, functions = _module_names(tree)

    def target(function: ast.expr) -> tuple[str, str] | None:
        if isinstance(function, ast.Name):
            return functions.get(function.id)
        if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            module = modules.get(function.value.id)
            return None if module is None else (module, function.attr)
        return None

    lines = []
    for node in ast.walk(tree):
        called = isinstance(node, ast.Call) and target(node.func) in _HOST_NAME_CALLS
        imported = isinstance(node, ast.ImportFrom) and any(
            (node.module, alias.name) in _HOST_NAME_CALLS for alias in node.names
        )
        uname_host = False
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.value, ast.Call):
            field = _UNAME_HOST_FIELDS.get(target(node.value.func) or ("", ""))
            by_name = isinstance(node, ast.Attribute) and node.attr == field
            by_index = (
                isinstance(node, ast.Subscript)
                and field is not None
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 1
            )
            uname_host = by_name or by_index
        if (called or imported or uname_host) and isinstance(node, (ast.expr, ast.stmt)):
            lines.append(node.lineno)
    return lines


def test_no_benchmark_producer_records_the_machine_host_name() -> None:
    """Benchmark records keep platform and hardware context, never the host name.

    Results are committed and published, so a host name would identify the
    machine. Every module under ``benchmarks/`` is parsed, and none reads the
    host name through ``platform``, ``socket`` or ``os.uname``.
    """
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    offenders = [
        f"{path.relative_to(root.parent)}:{line}"
        for path in sorted(root.rglob("*.py"))
        for line in _host_name_reads(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "import platform\nHOST = platform.node()\n",
        "import socket\nHOST = socket.gethostname()\n",
        "import socket\nHOST = socket.getfqdn()\n",
        "import os\nHOST = os.uname().nodename\n",
        "import platform\nHOST = platform.uname().node\n",
        "from platform import node\n",
        "from socket import gethostname as name\n",
        "import socket as s\nHOST = s.gethostname()\n",
        "import platform as p\nHOST = p.node()\n",
        "from os import uname\nHOST = uname().nodename\n",
        "import os\nHOST = os.uname()[1]\n",
        "import platform\nHOST = platform.uname()[1]\n",
    ],
)
def test_every_way_of_reading_the_host_name_is_recognised(source: str) -> None:
    """The scan recognises each standard-library route to the host name.

    Parameters
    ----------
    source : str
        Module source that reads the host name one way.
    """
    assert _host_name_reads(ast.parse(source)) != []


def test_platform_and_hardware_context_is_not_mistaken_for_the_host_name() -> None:
    """Platform, architecture and CPU readings stay allowed."""
    source = (
        "import os, platform\nA = platform.platform()\nB = platform.machine()\n"
        "C = os.uname().machine\nD = os.uname()[4]\nE = platform.uname()[0]\n"
    )
    assert _host_name_reads(ast.parse(source)) == []
