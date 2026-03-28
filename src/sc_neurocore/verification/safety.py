# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AST blocklist screen for auto-generated code

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeSafetyVerifier:
    """AST blocklist screen for auto-generated code.

    Walks the AST and rejects code containing known-dangerous patterns:
    filesystem mutation, process spawning, network access, code execution,
    and unrestricted imports.

    Limitations: this is a static blocklist, not a sandbox. It catches
    common dangerous patterns but cannot prevent all malicious code.
    Obfuscated calls (getattr chains, importlib indirection) may bypass it.
    Do not use as a security boundary without additional sandboxing.
    """

    _BLOCKED_ATTRS = frozenset(
        {
            # Process
            "system",
            "popen",
            "spawn",
            "spawnl",
            "spawnle",
            "kill",
            "fork",
            # Filesystem mutation
            "rmtree",
            "unlink",
            "remove",
            "rmdir",
            "rename",
            "truncate",
            "makedirs",
            # Network
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
        }
    )

    _BLOCKED_BUILTINS = frozenset(
        {
            "exec",
            "eval",
            "compile",
            "__import__",
            "breakpoint",
        }
    )

    _BLOCKED_IMPORTS = frozenset(
        {
            "subprocess",
            "shutil",
            "socket",
            "http",
            "urllib",
            "requests",
            "importlib",
            "ctypes",
            "signal",
            "multiprocessing",
        }
    )

    def verify_code_safety(self, source_code: str) -> bool:
        """Static analysis of source code for dangerous patterns.

        Returns True if no blocked patterns found, False otherwise.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            logger.error("Safety Violation: Syntax Error in generated code.")
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in self._BLOCKED_ATTRS:
                        logger.warning(
                            "Safety Violation: blocked call '%s'.",
                            node.func.attr,
                        )
                        return False
                elif isinstance(node.func, ast.Name):
                    if node.func.id in self._BLOCKED_BUILTINS:
                        logger.warning(
                            "Safety Violation: blocked builtin '%s'.",
                            node.func.id,
                        )
                        return False

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in self._BLOCKED_IMPORTS:
                        logger.warning(
                            "Safety Violation: blocked import '%s'.",
                            name,
                        )
                        return False

        return True

    def verify_logic_invariant(self, func: Any, input_sample: Any, expected_condition: Any) -> bool:
        """Dynamic verification: run func and check output against condition."""
        try:
            res = func(input_sample)
            if expected_condition(res):
                return True
            else:
                logger.error(
                    "Safety Violation: Logic invariant failed. Output %s invalid.",
                    res,
                )
                return False
        except Exception as e:
            logger.error("Safety Violation: Runtime Error - %s", e)
            return False
