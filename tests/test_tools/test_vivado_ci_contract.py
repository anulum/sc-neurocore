# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vivado CI gate contract tests

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple


class VivadoGate(NamedTuple):
    """Vivado-gated pytest surface discovered from the live test tree."""

    test_path: str
    test_functions: tuple[str, ...]
    skip_reasons: tuple[str, ...]


EXPECTED_VIVADO_GATES = {
    "tests/test_adc_to_spike_quantiser_synth.py",
    "tests/test_dcls_synth_zu3eg.py",
    "tests/test_ultrascale_plus_flow.py",
}


def _repo_root() -> Path:
    """Return the repository root containing the Vivado-gated tests."""

    return Path(__file__).resolve().parents[2]


def _source_for_node(source: str, node: ast.AST) -> str:
    """Return the source segment for an AST node."""

    return ast.get_source_segment(source, node) or ""


def _vivado_gated_functions(source: str) -> tuple[str, ...]:
    """Return test functions that inspect the MIF_VIVADO_CI environment gate."""

    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if "MIF_VIVADO_CI" in _source_for_node(source, node):
            names.append(node.name)
    return tuple(sorted(names))


def _skip_reasons(source: str) -> tuple[str, ...]:
    """Return pytest.skip reason strings from one Vivado-gated test file."""

    tree = ast.parse(source)
    reasons = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_pytest_skip_call(node):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                reasons.append(arg.value)
    return tuple(sorted(reasons))


def _is_pytest_skip_call(node: ast.Call) -> bool:
    """Return whether the AST call is pytest.skip(...)."""

    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "skip"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
    )


def _vivado_gates() -> list[VivadoGate]:
    """Discover Vivado opt-in test files from the live pytest tree."""

    gates = []
    this_file = Path(__file__).resolve()
    for path in sorted((_repo_root() / "tests").rglob("test_*.py")):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        if 'os.environ.get("MIF_VIVADO_CI")' not in text:
            continue
        relative = path.relative_to(_repo_root()).as_posix()
        gates.append(
            VivadoGate(
                test_path=relative,
                test_functions=_vivado_gated_functions(text),
                skip_reasons=_skip_reasons(text),
            )
        )
    return gates


def test_vivado_ci_gates_are_documented() -> None:
    """Ensure every Vivado opt-in pytest gate appears in public hardware docs."""

    docs = (_repo_root() / "docs" / "hardware" / "vivado_ci_gates.md").read_text(
        encoding="utf-8"
    )
    gates = _vivado_gates()
    gate_paths = {gate.test_path for gate in gates}

    assert gate_paths == EXPECTED_VIVADO_GATES
    assert "MIF_VIVADO_CI=1" in docs
    assert "Vivado 2024.2" in docs
    assert "xczu3eg-sbva484-1-e" in docs

    for gate in gates:
        assert gate.test_functions
        assert gate.test_path in docs
        for reason in gate.skip_reasons:
            if "set MIF_VIVADO_CI=1" in reason:
                assert "Vivado 2024.2 runner" in reason


def test_vivado_ci_gate_is_reachable_from_docs_nav() -> None:
    """Keep the Vivado gate guide reachable from hardware docs and MkDocs."""

    repo = _repo_root()
    toolchain_guide = (repo / "docs" / "hardware" / "FPGA_TOOLCHAIN_GUIDE.md").read_text(
        encoding="utf-8"
    )
    mkdocs = (repo / "mkdocs.yml").read_text(encoding="utf-8")

    assert "vivado_ci_gates.md" in toolchain_guide
    assert "hardware/vivado_ci_gates.md" in mkdocs
