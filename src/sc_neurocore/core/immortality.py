
import pickle
import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DigitalSoul:
    """
    Handles the persistence and 'immortality' of an SC Agent.
    Captures full state (weights, traces, parameters) for restoration.
    """
    agent_id: str
    state_data: Dict[str, Any] = field(default_factory=dict)
    
    def capture_agent(self, orchestrator):
        """
        Extracts state from all modules registered in the orchestrator.
        """
        print(f"Soul: Capturing state for Agent '{self.agent_id}'...")
        for name, module in orchestrator.modules.items():
            if hasattr(module, 'get_weights'):
                self.state_data[f"{name}_weights"] = module.get_weights()
            elif hasattr(module, 'weights'):
                self.state_data[f"{name}_weights"] = module.weights
                
            if hasattr(module, 'get_state'):
                self.state_data[f"{name}_state"] = module.get_state()
                
        print(f"Soul: Captured {len(self.state_data)} state components.")

    def save_soul(self, filepath: str):
        """
        Serializes the soul to a file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Soul: Saved to {filepath}")

    @classmethod
    def load_soul(cls, filepath: str) -> 'DigitalSoul':
        """
        Restores a soul from a file.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
            
    def reincarnate(self, orchestrator):
        """
        Injects the soul data back into an existing orchestrator's modules.
        """
        print(f"Soul: Reincarnating Agent '{self.agent_id}'...")
        for name, module in orchestrator.modules.items():
            w_key = f"{name}_weights"
            s_key = f"{name}_state"
            
            if w_key in self.state_data:
                if hasattr(module, 'weights'):
                    module.weights = self.state_data[w_key]
                if hasattr(module, '_refresh_packed_weights'):
                    module._refresh_packed_weights()
                    
            if s_key in self.state_data:
                if hasattr(module, 'v'): # Typical for neurons
                    module.v = self.state_data[s_key].get('v', 0.0)
        print("Soul: Reincarnation complete.")
