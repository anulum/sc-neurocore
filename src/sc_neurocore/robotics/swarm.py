import logging
import numpy as np
from dataclasses import dataclass
from ..layers.sc_learning_layer import SCLearningLayer

logger = logging.getLogger(__name__)


@dataclass
class SwarmCoupling:
    """
    Brain-to-Brain coupling for SC Networks (Swarm Intelligence).
    Synchronizes two agents via mutual spike injection.
    """

    coupling_strength: float = 0.1

    def synchronize(self, agent_a: SCLearningLayer, agent_b: SCLearningLayer):
        """
        Adjust weights of both agents to align their firing patterns.
        (Simplified: Hebbian cross-correlation update)
        """
        # We assume both agents have same number of neurons
        if agent_a.n_neurons != agent_b.n_neurons:
            raise ValueError("Agents must have same size for direct coupling.")

        # Extract weights
        wa = agent_a.get_weights()
        wb = agent_b.get_weights()

        # Mutual Attraction: Weights shift toward each other
        # W_new = W + alpha * (W_other - W)
        delta = self.coupling_strength * (wb - wa)

        # Update Agent A
        new_wa = wa + delta
        for i in range(agent_a.n_neurons):
            for j in range(agent_a.n_inputs):
                agent_a.synapses[i][j].update_weight(new_wa[i, j])

        # Update Agent B (Reciprocal)
        new_wb = wb - delta
        for i in range(agent_b.n_neurons):
            for j in range(agent_b.n_inputs):
                agent_b.synapses[i][j].update_weight(new_wb[i, j])

        logger.info(
            "Swarm Synchronization: Shifted weights by magnitude %.6f", np.mean(np.abs(delta))
        )
