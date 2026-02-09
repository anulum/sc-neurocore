
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class AdaptiveInference:
    """
    Manages Progressive Precision / Early Exit for SC.
    """
    check_interval: int = 64
    tolerance: float = 0.05 # 5% stability
    min_length: int = 128
    max_length: int = 2048
    
    def run_adaptive(self, step_func: Callable[[], float]) -> float:
        """
        Runs the SC process step-by-step until convergence or max_length.
        
        Args:
            step_func: Function that executes one step and returns current estimate.
        """
        history: List[float] = []
        
        current_val = 0.0
        
        for t in range(self.max_length):
            current_val = step_func()
            
            if t >= self.min_length and t % self.check_interval == 0:
                # Check stability over last 3 checks
                history.append(current_val)
                if len(history) >= 3:
                    # If variance is low, exit
                    recent = history[-3:]
                    if (max(recent) - min(recent)) < self.tolerance:
                        return current_val
                        
        return current_val
