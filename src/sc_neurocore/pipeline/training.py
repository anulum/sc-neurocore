# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training loops for stochastic computing networks (RL and

"""Training loops for stochastic computing networks (RL and multimodal fusion)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ..layers.sc_learning_layer import SCLearningLayer
from ..synapses.r_stdp import RewardModulatedSTDPSynapse

logger = logging.getLogger(__name__)


class SCTrainingLoop:
    """
    Standard and Reinforcement Learning loops for SC Networks.
    """

    @staticmethod
    def run_rl_epoch(
        agent: SCLearningLayer,
        env_step_func: Callable[[np.ndarray], float],
        input_data: np.ndarray,
        generations: int = 10,
    ) -> None:
        """
        Runs a reinforcement learning epoch.
        Uses RewardModulatedSTDPSynapse logic.
        """
        for gen in range(generations):
            # 1. Run forward pass
            spikes = agent.run_epoch(input_data)  # type: ignore[arg-type]

            # 2. Get reward from environment
            reward = env_step_func(spikes)

            # 3. Apply reward to all synapses
            for i in range(agent.n_neurons):
                for j in range(agent.n_inputs):
                    syn = agent.synapses[i][j]
                    if isinstance(syn, RewardModulatedSTDPSynapse):
                        syn.apply_reward(reward)

            logger.info("RL Epoch %d: Reward = %.4f", gen, reward)

    @staticmethod
    def train_multimodal_fusion(fusion_layer: object, dataset: object, epochs: int = 5) -> None:
        """
        Stub for training weights in a fusion layer.
        """
        raise NotImplementedError("multimodal fusion training not implemented")
