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
from typing import Any

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
        env_step_func: Callable[[np.ndarray[Any, Any]], float],
        input_data: np.ndarray[Any, Any],
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
    def train_multimodal_fusion(fusion_layer: Any, dataset: Any, epochs: int = 5) -> None:
        """Train weights in a multimodal fusion layer via per-sample updates.

        Iterates over the dataset for ``epochs`` rounds, calling
        ``fusion_layer.train_step(sample)`` on each sample returned by
        ``dataset.get_sample(i)``.  The fusion layer is responsible for
        its own weight update rule (Hebbian, LMS, etc.).
        """
        n_samples = getattr(dataset, "n_samples", len(getattr(dataset, "labels", [])))
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(n_samples):
                sample = dataset.get_sample(i)
                output = fusion_layer.train_step(sample)
                if output is not None:
                    total_loss += float(np.sum(np.abs(output)))
            avg_loss = total_loss / max(n_samples, 1)
            logger.info("Fusion Epoch %d/%d: avg_loss=%.4f", epoch + 1, epochs, avg_loss)
