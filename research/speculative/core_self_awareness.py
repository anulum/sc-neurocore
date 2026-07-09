# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SelfModel:
    capabilities: List[str] = field(default_factory=list)
    current_goals: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class MetaCognitionLoop:
    """
    Implements Computational Self-Awareness.
    Observes the Orchestrator and maintains a dynamic Self-Model.
    """

    self_model: SelfModel = field(default_factory=SelfModel)

    def observe(self, orchestrator):
        """
        Introspection step. Reads internal state of the executive.
        """
        # 1. Update Capabilities
        self.self_model.capabilities = list(orchestrator.modules.keys())

        # 2. Update Goals
        if hasattr(orchestrator, "active_goals"):
            self.self_model.current_goals = orchestrator.active_goals

        # 3. Assess Confidence (Simulated)
        # In a real system, this would analyze prediction error rates.
        # Here we simulate confidence fluctuation based on module count (complexity).
        complexity = len(self.self_model.capabilities)
        self.self_model.confidence = 1.0 / (1.0 + 0.1 * complexity)

        logger.info(
            "Meta-Cognition: I am aware of %d modules. Confidence: %.2f",
            complexity,
            self.self_model.confidence,
        )

    def reflect(self) -> str:
        """
        Returns a linguistic summary of the self-state.
        """
        return f"I am an agent with {len(self.self_model.capabilities)} capabilities. My primary goal is {self.self_model.current_goals}."
