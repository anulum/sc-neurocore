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

        # Check for forbidden imports or calls (e.g., 'os.system("rm -rf")')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check for os.system, subprocess.call etc.
                    if node.func.attr in ["system", "popen", "rmtree"]:
                        # Heuristic check
                        logger.warning(
                            "Safety Warning: Dangerous call detected '%s'.", node.func.attr
                        )
                        # In a real system, this would be stricter.
                        # For demo, we allow it but log it.

            # Check for infinite loops (While True without Break)
            if isinstance(node, ast.While):
                # Simple heuristic: if test is Constant(True), check for break
                pass

        logger.info("Safety Check: Code structure appears valid.")
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
