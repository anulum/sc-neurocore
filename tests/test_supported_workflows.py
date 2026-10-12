# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The documented workflows print what the page says they print

"""Run every code block of the supported-workflows page and compare its output with the page."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs/guides/supported_workflows.md"
_BLOCK = re.compile(
    r"<!-- workflow: (?P<name>[a-z0-9-]+) -->\n```python\n(?P<code>.*?)```\n\n"
    r"Expected output:\n\n```text\n(?P<output>.*?)```\n",
    re.DOTALL,
)


def documented_workflows(page: Path = PAGE) -> dict[str, tuple[str, str]]:
    """Read the workflows of the page.

    Parameters
    ----------
    page : Path
        The supported-workflows page.

    Returns
    -------
    dict
        Workflow name to its code and its expected output, in page order.
    """
    text = page.read_text(encoding="utf-8")
    return {match["name"]: (match["code"], match["output"]) for match in _BLOCK.finditer(text)}


def run_workflow(python: str, code: str, directory: Path) -> subprocess.CompletedProcess[str]:
    """Run one code block with the given interpreter in the given directory.

    Parameters
    ----------
    python : str
        Interpreter whose ``sc_neurocore`` is under test.
    code : str
        Code block of the page.
    directory : Path
        Working directory of the run.

    Returns
    -------
    subprocess.CompletedProcess
        Status and captured output.
    """
    return subprocess.run(
        [python, "-c", code],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


WORKFLOWS = documented_workflows()


def test_the_page_documents_four_workflows_with_code_and_output() -> None:
    """Every workflow heading has one marked block and one expected output."""
    text = PAGE.read_text(encoding="utf-8")
    assert list(WORKFLOWS) == [
        "stochastic-neuron",
        "bitstream-product",
        "equation-neuron",
        "equation-to-verilog",
    ]
    assert text.count("<!-- workflow: ") == len(WORKFLOWS) == text.count("Expected output:")
    assert all(code.strip() and output.strip() for code, output in WORKFLOWS.values())


@pytest.mark.parametrize("name", list(WORKFLOWS))
def test_a_documented_workflow_prints_its_documented_output(name: str, tmp_path: Path) -> None:
    """The block runs outside the repository directory and prints exactly the documented lines.

    Parameters
    ----------
    name : str
        Workflow name on the page.
    tmp_path : Path
        Working directory of the run.
    """
    code, expected = WORKFLOWS[name]
    result = run_workflow(sys.executable, code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_the_page_names_exactly_the_subpackages_the_wheel_leaves_out() -> None:
    """The list of source-only subpackages is the exclusion list of the build configuration."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = {
        item.removeprefix("sc_neurocore.")
        for item in configuration["tool"]["setuptools"]["packages"]["find"]["exclude"]
        if not item.endswith(".*")
    }
    listing = PAGE.read_text(encoding="utf-8").split("<!-- source-only-subpackages -->\n", 1)[1]
    assert set(re.findall(r"`([a-z_]+)`", listing)) == excluded
    assert len(excluded) == 16


def test_the_readme_leads_with_what_runs_today() -> None:
    """The quick start and the link to this page stand before the roadmap in the README."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Quick Start")
    assert start < readme.index("## Public Development Workbench and the 4.0.0 Roadmap")
    assert "docs/guides/supported_workflows.md" in readme[start : start + 2500]
