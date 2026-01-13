
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from ..core.orchestrator import CognitiveOrchestrator

@dataclass
class EnsembleOrchestrator:
    """
    Manages a collective of SC-NeuroCore Agents.
    Implements ensemble consensus and coordinated action.
    """
    agents: Dict[str, CognitiveOrchestrator] = field(default_factory=dict)
    
    def add_agent(self, name: str, agent: CognitiveOrchestrator):
        self.agents[name] = agent
        
    def run_consensus(self, pipeline: List[str], initial_input: Any) -> np.ndarray:
        """
        Runs the same pipeline on all agents and averages results.
        """
        results = []
        for name, agent in self.agents.items():
            out = agent.execute_pipeline(pipeline, initial_input)
            results.append(out.to_prob())
            
        # Majority vote / Average
        return np.mean(results, axis=0)

    def coordinated_mission(self, goal: str):
        """
        Assigns sub-tasks to agents based on their capabilities.
        """
        print(f"Ensemble: Initiating mission '{goal}'...")
        for name, agent in self.agents.items():
            print(f"  Agent '{name}': Assigned sub-task.")
            agent.active_goals = [f"{goal}_subtask"]
