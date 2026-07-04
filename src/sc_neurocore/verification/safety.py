# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AST blocklist screen for auto-generated code

"""Static safety screening for generated Python snippets.

The verifier rejects AST-visible file, process, network, dynamic import,
dynamic execution, and reflection escape routes before generated code can be
accepted by higher-level verification workflows. It is a conservative screening
step, not a sandbox or proof of semantic safety.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
import logging
from dataclasses import dataclass
from typing import ClassVar, TypeVar

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class CodeSafetyVerifier:
    """AST blocklist screen for auto-generated code.

    Walks the AST and rejects code containing known-dangerous patterns:
    filesystem mutation, process spawning, network access, code execution,
    and unrestricted imports.

    Limitations: this is a static blocklist, not a sandbox. It catches
    common dangerous patterns but cannot prove semantic safety, model data
    flow, or reason about values assembled before screening. Do not use as a
    security boundary without additional sandboxing.
    """

    _RELATIVE_IMPORT: ClassVar[str] = "<relative>"

    _BLOCKED_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {
            # Process
            "system",
            "popen",
            "spawn",
            "spawnl",
            "spawnle",
            "kill",
            "fork",
            "forkpty",
            "execv",
            "execve",
            # Filesystem mutation
            "rmtree",
            "unlink",
            "remove",
            "rmdir",
            "rename",
            "replace",
            "truncate",
            "makedirs",
            "mkdir",
            "write_text",
            "write_bytes",
            "touch",
            "chmod",
            "chown",
            "symlink",
            "symlink_to",
            "hardlink_to",
            "open",
            # Network
            "socket",
            "connect",
            "bind",
            "listen",
            "accept",
            "urlopen",
            "urlretrieve",
            # Subprocess
            "Popen",
            "call",
            "check_call",
            "check_output",
            "run",
            # Reflection (write)
            "setattr",
            "delattr",
            "getattr",
            # Dynamic execution through namespaces such as __builtins__.eval
            "exec",
            "eval",
            "compile",
            "__import__",
            "import_module",
        }
    )

    _BLOCKED_CALL_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "exec",
            "eval",
            "compile",
            "__import__",
            "breakpoint",
            "open",
            "getattr",
            "setattr",
            "delattr",
            "globals",
            "locals",
            "vars",
            "__builtins__",
            "remove",
            "unlink",
            "rmdir",
            "rename",
            "replace",
            "import_module",
        }
    )

    _BLOCKED_IMPORTS: ClassVar[frozenset[str]] = frozenset(
        {
            _RELATIVE_IMPORT,
            "builtins",
            "os",
            "pathlib",
            "subprocess",
            "shutil",
            "socket",
            "sys",
            "http",
            "urllib",
            "requests",
            "importlib",
            "ctypes",
            "signal",
            "multiprocessing",
            "tempfile",
            "glob",
            "ftplib",
        }
    )

    def verify_code_safety(self, source_code: str) -> bool:
        """Return whether ``source_code`` passes the static blocklist.

        Parameters
        ----------
        source_code:
            Python source text to parse and inspect without executing.

        Returns
        -------
        bool
            ``True`` when no blocked import or call is visible in the AST;
            ``False`` when parsing fails or a blocked pattern is found.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            logger.error("Safety Violation: Syntax Error in generated code.")
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                blocked_call = self._blocked_call_name(node.func)
                if blocked_call is not None:
                    if isinstance(node.func, ast.Attribute):
                        logger.warning(
                            "Safety Violation: blocked call '%s'.",
                            blocked_call,
                        )
                        return False
                    logger.warning(
                        "Safety Violation: blocked builtin '%s'.",
                        blocked_call,
                    )
                    return False

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in self._import_roots(node):
                    if name in self._BLOCKED_IMPORTS:
                        logger.warning(
                            "Safety Violation: blocked import '%s'.",
                            name,
                        )
                        return False

        return True

    def verify_logic_invariant(
        self,
        func: Callable[[InputT], OutputT],
        input_sample: InputT,
        expected_condition: Callable[[OutputT], bool],
    ) -> bool:
        """Return whether a dynamic invariant holds for one input sample.

        Parameters
        ----------
        func:
            Callable under verification.
        input_sample:
            Sample passed to ``func``.
        expected_condition:
            Predicate that must return ``True`` for the callable output.

        Returns
        -------
        bool
            ``True`` if ``func(input_sample)`` satisfies
            ``expected_condition``; otherwise ``False``.
        """
        try:
            res = func(input_sample)
            if expected_condition(res):
                return True

            logger.error(
                "Safety Violation: Logic invariant failed. Output %s invalid.",
                res,
            )
            return False
        except Exception as exc:
            logger.error("Safety Violation: Runtime Error - %s", exc)
            return False

    def _blocked_call_name(self, func: ast.expr) -> str | None:
        """Return the blocked call name exposed by ``func``, if any."""
        if isinstance(func, ast.Attribute):
            if func.attr in self._BLOCKED_ATTRS:
                return func.attr
            return None
        if isinstance(func, ast.Name):
            if func.id in self._BLOCKED_CALL_NAMES:
                return func.id
            return None
        if isinstance(func, ast.Subscript):
            return self._blocked_call_name(func.value)
        return None

    @staticmethod
    def _import_roots(node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
        """Return top-level module names imported by ``node``."""
        if isinstance(node, ast.Import):
            return tuple(alias.name.split(".")[0] for alias in node.names)
        if node.module:
            return (node.module.split(".")[0],)
        return (CodeSafetyVerifier._RELATIVE_IMPORT,)
