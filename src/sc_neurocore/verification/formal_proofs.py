
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Interval:
    min_val: float
    max_val: float
    
    def __add__(self, other):
        return Interval(self.min_val + other.min_val, self.max_val + other.max_val)
        
    def __mul__(self, other):
        # Interval multiplication
        vals = [
            self.min_val * other.min_val,
            self.min_val * other.max_val,
            self.max_val * other.min_val,
            self.max_val * other.max_val
        ]
        return Interval(min(vals), max(vals))
        
    def __repr__(self):
        return f"[{self.min_val:.4f}, {self.max_val:.4f}]"

class FormalVerifier:
    """
    Simulated SMT Solver using Interval Arithmetic.
    Proves properties of Stochastic Functions.
    """
    
    @staticmethod
    def verify_probability_bounds(input_interval: Interval, weight_interval: Interval) -> bool:
        """
        Prove that Output Probability is always in [0, 1].
        Logic: Out = Input * Weight (AND gate)
        """
        # Logic: P(A & B) = P(A) * P(B) assuming independence
        out = input_interval * weight_interval
        
        is_safe = out.min_val >= 0.0 and out.max_val <= 1.0
        print(f"Verification: Input {input_interval} * Weight {weight_interval} -> Output {out}")
        print(f"Property (0 <= p <= 1): {'HELD' if is_safe else 'VIOLATED'}")
        return is_safe

    @staticmethod
    def verify_energy_safety(energy: float, cost: float) -> bool:
        """
        Prove that operation will not consume more energy than available.
        """
        # Symbolic check
        # Precondition: Energy >= Cost
        # Postcondition: NewEnergy >= 0
        if energy >= cost:
            new_e = energy - cost
            print(f"Verification: {energy} - {cost} = {new_e} >= 0. HELD.")
            return True
        else:
            print(f"Verification: {energy} < {cost}. VIOLATED (Halt).")
            return False
