# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Test inventory audit

"""Account for every tracked test module in a native pytest collection transcript."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = 1
COLLECTED_TESTS_RE = re.compile(r"(?P<count>\d+) tests collected")
COLLECTED_FILE_RE = re.compile(r"^(tests/.+?\.py)::")


@dataclass(frozen=True)
class OptionalImportSkip:
    """A tracked test file skipped during collection by an optional dependency."""

    path: str
    dependencies: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        """Return a stable JSON object for audit artefacts."""
        return {"path": self.path, "dependencies": list(self.dependencies)}


@dataclass(frozen=True)
class IntentionalNonTestModule:
    """A ``test_*.py`` compatibility or support module excluded explicitly."""

    path: str

    def to_json(self) -> dict[str, str]:
        """Return a stable JSON object for audit artefacts."""
        return {"path": self.path, "marker": "__test__ = False"}


@dataclass(frozen=True)
class TestInventoryAudit:
    """Repository test inventory compared with a pytest collect-only transcript."""

    schema_version: int
    tracked_test_files: tuple[str, ...]
    collected_test_files: tuple[str, ...]
    collected_tests: int
    optional_import_skips: tuple[OptionalImportSkip, ...]
    intentional_non_test_modules: tuple[IntentionalNonTestModule, ...]
    unexpected_uncollected: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every uncollected tracked file has an optional gate."""
        return self.collected_tests > 0 and not self.unexpected_uncollected

    def to_json(self) -> dict[str, object]:
        """Return a stable JSON-compatible audit payload."""
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "tracked_test_file_count": len(self.tracked_test_files),
            "collected_test_file_count": len(self.collected_test_files),
            "collected_tests": self.collected_tests,
            "optional_import_skip_count": len(self.optional_import_skips),
            "optional_import_skips": [item.to_json() for item in self.optional_import_skips],
            "intentional_non_test_module_count": len(self.intentional_non_test_modules),
            "intentional_non_test_modules": [
                item.to_json() for item in self.intentional_non_test_modules
            ],
            "unexpected_uncollected": list(self.unexpected_uncollected),
        }


def tracked_test_files(repo: Path) -> tuple[str, ...]:
    """Return tracked test files whose basename follows pytest's test-file pattern."""
    result = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        path = Path(line)
        if path.name.startswith("test_") and path.suffix == ".py":
            files.append(path.as_posix())
    return tuple(sorted(files))


def parse_collect_only_output(text: str) -> tuple[tuple[str, ...], int]:
    """Parse pytest collect-only output into collected files and test count."""
    files = {
        match.group(1)
        for line in text.splitlines()
        if (match := COLLECTED_FILE_RE.match(line)) is not None
    }
    summary_match = None
    for match in COLLECTED_TESTS_RE.finditer(text):
        summary_match = match
    if summary_match is None:
        raise ValueError("pytest collect-only output did not include a collected-tests summary")
    return tuple(sorted(files)), int(summary_match.group("count"))


def module_level_optional_dependencies(path: Path) -> tuple[str, ...]:
    """Return imports guarded by explicit module-level optional-dependency skips."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dependencies: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.If):
            dependency = _missing_module_skip_dependency(node)
            if dependency is not None:
                dependencies.append(dependency)
        call = _module_level_call(node)
        if call is None or not _is_pytest_importorskip(call):
            continue
        if not call.args:
            continue
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            dependencies.append(first_arg.value)
    return tuple(sorted(set(dependencies)))


def _missing_module_skip_dependency(node: ast.If) -> str | None:
    """Identify a module skip that requires an absent import specification."""
    if not any(_is_module_skip(_module_level_call(statement)) for statement in node.body):
        return None
    conditions = (
        node.test.values
        if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And)
        else [node.test]
    )
    for condition in conditions:
        if not (
            isinstance(condition, ast.Compare)
            and len(condition.ops) == len(condition.comparators) == 1
            and isinstance(condition.ops[0], ast.Is)
            and isinstance(condition.comparators[0], ast.Constant)
            and condition.comparators[0].value is None
            and isinstance(condition.left, ast.Call)
        ):
            continue
        call = condition.left
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "find_spec"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "util"
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "importlib"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and call.args[0].value
        ):
            return call.args[0].value
    return None


def _is_module_skip(call: ast.Call | None) -> bool:
    """Require an explicit pytest module skip instead of a function-only skip."""
    return (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "skip"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "pytest"
        and any(
            keyword.arg == "allow_module_level"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    )


def has_module_test_opt_out(path: Path) -> bool:
    """Return whether a module explicitly declares pytest's non-test marker."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "__test__"
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        ):
            return True
    return False


def build_inventory_audit(repo: Path, collect_output: str) -> TestInventoryAudit:
    """Build a test-inventory audit from git state and pytest collection output."""
    tracked = tracked_test_files(repo)
    collected, collected_tests = parse_collect_only_output(collect_output)
    collected_set = set(collected)
    optional_skips: list[OptionalImportSkip] = []
    intentional_non_tests: list[IntentionalNonTestModule] = []
    unexpected: list[str] = []
    for path in tracked:
        if path in collected_set:
            continue
        if has_module_test_opt_out(repo / path):
            intentional_non_tests.append(IntentionalNonTestModule(path=path))
            continue
        dependencies = module_level_optional_dependencies(repo / path)
        if dependencies:
            optional_skips.append(OptionalImportSkip(path=path, dependencies=dependencies))
        else:
            unexpected.append(path)

    return TestInventoryAudit(
        schema_version=SCHEMA_VERSION,
        tracked_test_files=tracked,
        collected_test_files=collected,
        collected_tests=collected_tests,
        optional_import_skips=tuple(optional_skips),
        intentional_non_test_modules=tuple(intentional_non_tests),
        unexpected_uncollected=tuple(unexpected),
    )


def _module_level_call(node: ast.stmt) -> ast.Call | None:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return node.value
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        return node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
        return node.value
    return None


def _is_pytest_importorskip(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "importorskip"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "pytest"
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--collect-output",
        type=Path,
        required=True,
        help="Path to pytest --collect-only output.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the test-inventory audit CLI."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo.resolve()
    collect_output_path = args.collect_output.resolve()
    audit = build_inventory_audit(repo, collect_output_path.read_text(encoding="utf-8"))
    payload = audit.to_json()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    if not audit.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
