# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-agent ensemble orchestration with consensus and

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

    def add_agent(self, name: str, agent: CognitiveOrchestrator) -> None:
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
        ensemble_output: np.ndarray[Any, Any] = np.mean(results, axis=0)
        return ensemble_output

    def coordinated_mission(self, goal: str) -> None:
        """
        Assigns sub-tasks to agents based on their capabilities.
        """
        logger.info("Ensemble: Initiating mission '%s'...", goal)
        for name, agent in self.agents.items():
            logger.info("  Agent '%s': Assigned sub-task.", name)
            agent.active_goals = [f"{goal}_subtask"]
