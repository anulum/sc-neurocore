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
_HOST_NAME_IMPORTS = {("platform", "node"), ("socket", "gethostname"), ("socket", "getfqdn")}


def _host_name_reads(tree: ast.AST) -> list[int]:
    """Return the lines of ``tree`` that read the machine host name.

    Parameters
    ----------
    tree : ast.AST
        Parsed module.

    Returns
    -------
    list of int
        Lines calling ``platform.node``, ``socket.gethostname`` or ``socket.getfqdn``,
        reading ``nodename`` from ``os.uname()`` or ``node`` from ``platform.uname()``,
        or importing one of those functions by name.
    """
    lines = []
    for node in ast.walk(tree):
        called = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) in _HOST_NAME_CALLS
        )
        read_from_uname = (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and (node.value.func.value.id, node.value.func.attr, node.attr)
            in {("os", "uname", "nodename"), ("platform", "uname", "node")}
        )
        imported = isinstance(node, ast.ImportFrom) and any(
            (node.module, alias.name) in _HOST_NAME_IMPORTS for alias in node.names
        )
        if (called or read_from_uname or imported) and isinstance(node, (ast.expr, ast.stmt)):
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
    source = "import os, platform\nA = platform.platform()\nB = platform.machine()\nC = os.uname().machine\n"
    assert _host_name_reads(ast.parse(source)) == []
