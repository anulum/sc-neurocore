# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real repository engine gate provenance regressions

"""Exercise real repository engine-gate provenance through public pytest sweeps."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.test_engine_requirement_gate import _GUARDED_BINDING_FILES

_ROOT = Path(__file__).resolve().parents[1]


def _copy_engine_inventory(destination: Path) -> None:
    """Copy every maintained source without changing the active checkout."""
    copied = set(_GUARDED_BINDING_FILES) | {
        "tests/test_engine_requirement_gate.py",
        "tests/engine_requirement.py",
        "tests/engine_gate_provenance.py",
        "tests/__init__.py",
    }
    for relative in copied:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)


def _run_source_sweeps(directory: Path) -> subprocess.CompletedProcess[str]:
    """Execute the actual inventory and dominance tests in the copied repository."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(directory)
    entry = str(directory / "tests/test_engine_requirement_gate.py")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--confcutdir",
            str(directory),
            entry + "::test_pinned_inventory_equals_the_derived_require_engine_set",
            entry + "::test_no_test_module_imports_the_engine_before_a_gate",
            "-q",
            "--no-cov",
        ],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize(
    "mutation", ["pickle-noop", "default-noop", "import-shadow", "attribute", "wrong-target"]
)
def test_real_repository_sweeps_refuse_replaced_or_shadowed_engine_gates(
    tmp_path: Path, mutation: str
) -> None:
    """Reject altered real consumer sources through both actual pytest sweeps.

    Parameters
    ----------
    tmp_path
        Private copy of the maintained source inventory, never the live checkout.
    mutation
        Replacement of a reviewed local body or of a maintained imported binding.
    """
    _copy_engine_inventory(tmp_path)
    if mutation.endswith("noop"):
        filename = (
            "test_engine_global_pickle.py"
            if mutation == "pickle-noop"
            else "test_default_neuron_engine_binding.py"
        )
        target = tmp_path / "tests" / filename
        text = target.read_text()
        tree = ast.parse(text)
        function = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "require_engine"
        )
        assert function.end_lineno is not None
        lines = text.splitlines(keepends=True)
        indent = " " * function.col_offset
        lines[function.lineno - 1 : function.end_lineno] = [
            f"{indent}def require_engine():\n{indent}    return None\n"
        ]
        target.write_text("".join(lines))
    else:
        target = tmp_path / "tests/test_bitstream_averager.py"
        text = target.read_text()
        replacement = {
            "import-shadow": "require_engine = lambda: None\nrequire_engine()",
            "attribute": "class UntrustedGate:\n    def require_engine(self):\n        return None\n"
            "untrusted = UntrustedGate()\nuntrusted.require_engine()",
            "wrong-target": 'require_engine("math")',
        }[mutation]
        assert "\nrequire_engine()\n" in text
        target.write_text(text.replace("\nrequire_engine()\n", f"\n{replacement}\n", 1))
    result = _run_source_sweeps(tmp_path)
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "test_pinned_inventory_equals_the_derived_require_engine_set" in combined
    assert "require_engine-guarded set drifted" in combined


@pytest.mark.parametrize("style", ["module-alias", "module-qualified", "module-keyword", "literal"])
def test_real_repository_sweeps_accept_maintained_gate_import_forms(
    tmp_path: Path, style: str
) -> None:
    """Preserve legitimate compiled-engine gates through the real repository sweeps.

    Parameters
    ----------
    tmp_path
        Private complete copy of the maintained engine test inventory.
    style
        Canonical module alias, qualified import, keyword or literal engine target.
    """
    _copy_engine_inventory(tmp_path)
    target = tmp_path / "tests/test_bitstream_averager.py"
    text = target.read_text()
    imports, call = {
        "module-alias": ("import tests.engine_requirement as gates", "gates.require_engine()"),
        "module-qualified": (
            "import tests.engine_requirement",
            "tests.engine_requirement.require_engine()",
        ),
        "module-keyword": (
            "from tests.engine_requirement import require_engine, ENGINE_EXTENSION_MODULE",
            "require_engine(module=ENGINE_EXTENSION_MODULE)",
        ),
        "literal": (
            "from tests.engine_requirement import require_engine",
            'require_engine("sc_neurocore_engine.sc_neurocore_engine")',
        ),
    }[style]
    text = text.replace("from tests.engine_requirement import require_engine", imports, 1)
    target.write_text(text.replace("\nrequire_engine()\n", f"\n{call}\n", 1))
    result = _run_source_sweeps(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


@pytest.mark.parametrize(
    ("prefix", "call", "accepted"),
    [
        ("del require_engine", "require_engine()", False),
        ("from math import *", "require_engine()", False),
        (
            "import tests.engine_requirement as gates\ngates.require_engine = lambda: None",
            "gates.require_engine()",
            False,
        ),
        ("", "object().require_engine()", False),
        (
            "async def configure(value=(require_engine := lambda: None)):\n    pass",
            "require_engine()",
            False,
        ),
        (
            "configure = lambda value=(require_engine := lambda: None): value",
            "require_engine()",
            False,
        ),
        (
            "@((require_engine := lambda value: value))\ndef configure():\n    pass",
            "require_engine()",
            False,
        ),
        (
            "try:\n    raise ValueError\nexcept ValueError as require_engine:\n    pass",
            "require_engine()",
            False,
        ),
        ("", 'require_engine(target="sc_neurocore_engine.sc_neurocore_engine")', False),
        ("", 'require_engine("sc_neurocore_engine.sc_neurocore_engine", "math")', False),
        (
            "from tests.engine_requirement import ENGINE_EXTENSION_MODULE",
            'require_engine(ENGINE_EXTENSION_MODULE + "")',
            False,
        ),
        (
            "@staticmethod\n"
            "def configure(a: int = 1, /, *args: float, b: int = 2, c: int, **kwargs: str) -> int:\n"
            "    global require_engine\n    require_engine = lambda: None\n    return a\n"
            "async def configure_async():\n    require_engine = lambda: None\n"
            "configure_lambda = lambda: None\n"
            "class Configuration(int, metaclass=type):\n    require_engine = None\n"
            "try:\n    pass\nexcept ValueError:\n    pass",
            "require_engine()",
            True,
        ),
        (
            "import tests.engine_requirement as gates\n"
            "class Configuration((gates := object())):\n    pass",
            "gates.require_engine()",
            False,
        ),
        pytest.param(
            "class Configuration:\n    global require_engine\n    require_engine = lambda: None",
            "require_engine()",
            False,
            id="class-global-store",
        ),
        pytest.param(
            "import tests.engine_requirement as gates\n"
            "class Configuration:\n    gates.require_engine = lambda: None",
            "gates.require_engine()",
            False,
            id="class-namespace-store",
        ),
        pytest.param(
            "class Configuration:\n    class Nested:\n"
            "        global require_engine\n        require_engine = lambda: None",
            "require_engine()",
            False,
            id="nested-class-global-store",
        ),
    ],
)
def test_real_repository_sweeps_track_import_time_binding_changes(
    tmp_path: Path, prefix: str, call: str, accepted: bool
) -> None:
    """Distinguish module binding changes from local scopes in real test sources.

    Parameters
    ----------
    tmp_path
        Private complete copy of the maintained engine test inventory.
    prefix
        Source inserted before the actual binding module's engine gate.
    call
        Actual top-level gate expression subjected to the repository sweeps.
    accepted
        Whether the maintained binding and compiled-module target survive.
    """
    _copy_engine_inventory(tmp_path)
    target = tmp_path / "tests/test_bitstream_averager.py"
    text = target.read_text()
    target.write_text(text.replace("\nrequire_engine()\n", f"\n{prefix}\n{call}\n", 1))
    result = _run_source_sweeps(tmp_path)
    combined = result.stdout + result.stderr
    assert result.returncode == (0 if accepted else 1), combined
    assert ("2 passed" if accepted else "require_engine-guarded set drifted") in combined
    if not accepted:
        assert "engine imports not dominated by an unconditional gate" in combined
        assert "2 failed" in combined


def test_real_repository_sweeps_accept_explicit_pytest_gate_inventory_change(
    tmp_path: Path,
) -> None:
    """Accept a maintained pytest engine gate only with the matching inventory edit.

    Parameters
    ----------
    tmp_path
        Complete copied source inventory retaining every other engine consumer.
    """
    _copy_engine_inventory(tmp_path)
    target = tmp_path / "tests/test_bitstream_averager.py"
    target.write_text(
        target.read_text().replace(
            "\nrequire_engine()\n",
            '\npytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")\n',
            1,
        )
    )
    inventory = tmp_path / "tests/test_engine_requirement_gate.py"
    inventory.write_text(
        inventory.read_text().replace('    "tests/test_bitstream_averager.py",\n', "", 1)
    )
    result = _run_source_sweeps(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
