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

    def verify_code_safety(self, source_code: str) -> bool:
        """
        Static analysis of source code for dangerous patterns.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            logger.error("Safety Violation: Syntax Error in generated code.")
            return False

        BLOCKED_ATTRS = {"system", "popen", "rmtree", "call", "Popen"}
        BLOCKED_NAMES = {"eval", "exec", "compile", "__import__"}
        violations: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRS:
                    violations.append(f"line {node.lineno}: blocked call '{node.func.attr}'")
                elif isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                    violations.append(f"line {node.lineno}: blocked builtin '{node.func.id}'")

        if violations:
            for v in violations:
                logger.warning("Safety violation: %s", v)
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
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as e:
            logger.error("Safety Violation: Runtime Error - %s", e)
            return False
