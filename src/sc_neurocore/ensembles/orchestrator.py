# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any

"""Multi-agent ensemble orchestration with consensus and coordinated missions."""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from ..core.orchestrator import CognitiveOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class EnsembleOrchestrator:
    """
    Manages a collective of SC-NeuroCore Agents.
    Implements ensemble consensus and coordinated action.
    """

    agents: Dict[str, CognitiveOrchestrator] = field(default_factory=dict)

    def add_agent(self, name: str, agent: CognitiveOrchestrator):  # type: ignore
        self.agents[name] = agent

    def run_consensus(self, pipeline: List[str], initial_input: Any) -> np.ndarray[Any, Any]:
        """
        Runs the same pipeline on all agents and averages results.
        """
        results = []
        for name, agent in self.agents.items():
            out = agent.execute_pipeline(pipeline, initial_input)
            results.append(out.to_prob())

        # Majority vote / Average
        return np.mean(results, axis=0)  # type: ignore

    def coordinated_mission(self, goal: str):  # type: ignore
        """
        Assigns sub-tasks to agents based on their capabilities.
        """
        logger.info("Ensemble: Initiating mission '%s'...", goal)
        for name, agent in self.agents.items():
            logger.info("  Agent '%s': Assigned sub-task.", name)
            agent.active_goals = [f"{goal}_subtask"]
