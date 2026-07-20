# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for test inventory audit

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/test_inventory_audit.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_inventory_audit", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "tests"], cwd=root, check=True)


def test_parse_collect_only_output_extracts_files_and_summary() -> None:
    tool = _load_tool()

    files, count = tool.parse_collect_only_output(
        "\n".join(
            [
                "tests/test_alpha.py::test_one",
                "tests/nested/test_beta.py::TestBeta::test_two",
                "tests/nested/test_beta.py::test_three",
                "3 tests collected in 0.12s",
            ]
        )
    )

    assert files == ("tests/nested/test_beta.py", "tests/test_alpha.py")
    assert count == 3


def test_parse_collect_only_output_requires_summary() -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="collected-tests summary"):
        tool.parse_collect_only_output("tests/test_alpha.py::test_one\n")


def test_module_level_importorskip_dependencies_ignores_function_calls(tmp_path: Path) -> None:
    tool = _load_tool()
    test_file = tmp_path / "test_optional.py"
    _write(
        test_file,
        "\n".join(
            [
                "import pytest",
                "onnx = pytest.importorskip('onnx')",
                "def test_late_skip():",
                "    pytest.importorskip('not_a_module_level_gate')",
            ]
        ),
    )

    assert tool.module_level_importorskip_dependencies(test_file) == ("onnx",)


def test_module_level_importorskip_dependencies_accepts_expr_and_annassign(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    test_file = tmp_path / "test_optional.py"
    _write(
        test_file,
        "\n".join(
            [
                "import pytest",
                "pytest.importorskip()",
                "pytest.importorskip('cupy')",
                "mpi4py: object = pytest.importorskip('mpi4py')",
            ]
        ),
    )

    assert tool.module_level_importorskip_dependencies(test_file) == ("cupy", "mpi4py")


def test_module_test_opt_out_requires_top_level_false_assignment(tmp_path: Path) -> None:
    tool = _load_tool()
    opted_out = tmp_path / "test_support.py"
    enabled = tmp_path / "test_enabled.py"
    nested = tmp_path / "test_nested.py"
    _write(opted_out, "__test__ = False\n")
    _write(enabled, "__test__ = True\n")
    _write(nested, "def configure():\n    __test__ = False\n")

    assert tool.has_module_test_opt_out(opted_out) is True
    assert tool.has_module_test_opt_out(enabled) is False
    assert tool.has_module_test_opt_out(nested) is False


def test_build_inventory_audit_allows_only_module_level_optional_skips(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    _write(tmp_path / "tests/test_collected.py", "def test_ok():\n    assert True\n")
    _write(
        tmp_path / "tests/test_optional.py",
        "import pytest\nonnx = pytest.importorskip('onnx')\n",
    )
    _write(tmp_path / "tests/test_support.py", "__test__ = False\n")
    _write(tmp_path / "tests/test_missing.py", "def test_missing():\n    assert True\n")
    _git_repo(tmp_path)

    audit = tool.build_inventory_audit(
        tmp_path,
        "tests/test_collected.py::test_ok\n1 tests collected in 0.01s\n",
    )

    assert audit.passed is False
    assert audit.collected_tests == 1
    assert [item.path for item in audit.optional_import_skips] == ["tests/test_optional.py"]
    assert audit.optional_import_skips[0].dependencies == ("onnx",)
    assert [item.path for item in audit.intentional_non_test_modules] == ["tests/test_support.py"]
    assert audit.unexpected_uncollected == ("tests/test_missing.py",)
    payload = audit.to_json()
    assert payload["intentional_non_test_modules"] == [
        {"marker": "__test__ = False", "path": "tests/test_support.py"}
    ]
    assert payload["unexpected_uncollected"] == ["tests/test_missing.py"]


def test_main_writes_json_and_returns_success_for_optional_skips(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tests/test_collected.py", "def test_ok():\n    assert True\n")
    _write(
        tmp_path / "tests/test_optional.py",
        "import pytest\nmpi4py = pytest.importorskip('mpi4py')\n",
    )
    _git_repo(tmp_path)
    collect_output = tmp_path / "collect.txt"
    collect_output.write_text(
        "tests/test_collected.py::test_ok\n1 tests collected in 0.01s\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    tool = _load_tool()

    exit_code = tool.main(
        [
            "--repo",
            str(tmp_path),
            "--collect-output",
            str(collect_output),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["tracked_test_file_count"] == 2
    assert payload["collected_test_file_count"] == 1
    assert payload["optional_import_skip_count"] == 1


def test_main_prints_json_and_returns_failure_for_unexpected_gap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    _write(tmp_path / "tests/test_collected.py", "def test_ok():\n    assert True\n")
    _write(tmp_path / "tests/test_missing.py", "def test_missing():\n    assert True\n")
    _git_repo(tmp_path)
    collect_output = tmp_path / "collect.txt"
    collect_output.write_text(
        "tests/test_collected.py::test_ok\n1 tests collected in 0.01s\n",
        encoding="utf-8",
    )

    exit_code = tool.main(["--repo", str(tmp_path), "--collect-output", str(collect_output)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["passed"] is False
    assert payload["unexpected_uncollected"] == ["tests/test_missing.py"]


def test_script_entrypoint_uses_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / "tests/test_collected.py", "def test_ok():\n    assert True\n")
    _git_repo(tmp_path)
    collect_output = tmp_path / "collect.txt"
    collect_output.write_text(
        "tests/test_collected.py::test_ok\n1 tests collected in 0.01s\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--repo",
            str(tmp_path),
            "--collect-output",
            str(collect_output),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(TOOL), run_name="__main__")

    assert exc.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
