# SPDX-License-Identifier: AGPL-3.0-or-later
import ast
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeSafetyVerifier:
    """
    Formal Verification for Self-Modifying Code.
    Analyzes AST to prevent catastrophic bugs in auto-generated updates.
    """

    _BLOCKED_ATTRS = frozenset({"system", "popen", "rmtree"})
    _BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__"})

    def verify_code_safety(self, source_code: str) -> bool:
        """Static analysis of source code for dangerous patterns."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            logger.error("Safety Violation: Syntax Error in generated code.")
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in self._BLOCKED_ATTRS:
                        logger.warning("Safety Violation: blocked call '%s'.", node.func.attr)
                        return False
                elif isinstance(node.func, ast.Name):
                    if node.func.id in self._BLOCKED_BUILTINS:
                        logger.warning("Safety Violation: blocked builtin '%s'.", node.func.id)
                        return False

        return True

    def verify_logic_invariant(self, func, input_sample, expected_condition):  # type: ignore
        """
        Dynamic verification (Unit Test on the fly).
        """
        try:
            res = func(input_sample)
            if expected_condition(res):
                return True
            else:
                logger.error("Safety Violation: Logic invariant failed. Output %s invalid.", res)
                return False
        except Exception as e:
            logger.error("Safety Violation: Runtime Error - %s", e)
            return False
