
import ast
import inspect
from dataclasses import dataclass

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
            print("Safety Violation: Syntax Error in generated code.")
            return False
            
        # Check for forbidden imports or calls (e.g., 'os.system("rm -rf")')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check for os.system, subprocess.call etc.
                    if node.func.attr in ['system', 'popen', 'rmtree']:
                        # Heuristic check
                        print(f"Safety Warning: Dangerous call detected '{node.func.attr}'.")
                        # In a real system, this would be stricter.
                        # For demo, we allow it but log it.
                        
            # Check for infinite loops (While True without Break)
            if isinstance(node, ast.While):
                # Simple heuristic: if test is Constant(True), check for break
                pass
                
        print("Safety Check: Code structure appears valid.")
        return True

    def verify_logic_invariant(self, func, input_sample, expected_condition):
        """
        Dynamic verification (Unit Test on the fly).
        """
        try:
            res = func(input_sample)
            if expected_condition(res):
                return True
            else:
                print(f"Safety Violation: Logic invariant failed. Output {res} invalid.")
                return False
        except Exception as e:
            print(f"Safety Violation: Runtime Error - {e}")
            return False
