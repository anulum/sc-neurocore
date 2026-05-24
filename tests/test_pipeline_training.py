# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import numpy as np

from sc_neurocore.pipeline.training import SCTrainingLoop
from sc_neurocore.synapses.r_stdp import RewardModulatedSTDPSynapse


class _Agent:
    n_neurons = 1
    n_inputs = 2

    def __init__(self) -> None:
        self.synapses = [
            [RewardModulatedSTDPSynapse(w_min=0.0, w_max=1.0, eligibility_trace=1.0), object()]
        ]
        self.epochs: list[np.ndarray] = []

    def run_epoch(self, input_data: np.ndarray) -> np.ndarray:
        self.epochs.append(input_data.copy())
        return np.array([1.0, 0.0])


class _FusionLayer:
    def __init__(self) -> None:
        self.samples: list[np.ndarray] = []

    def train_step(self, sample: np.ndarray) -> np.ndarray | None:
        self.samples.append(sample)
        return sample if sample.sum() > 0.0 else None


class _DatasetWithSamples:
    n_samples = 2

    def get_sample(self, index: int) -> np.ndarray:
        return np.array([float(index), 1.0])


class _DatasetWithLabelsOnly:
    labels = [0, 1, 2]

    def get_sample(self, index: int) -> np.ndarray:
        return np.array([float(index)])


def test_reinforcement_epoch_applies_reward_only_to_reward_modulated_synapses():
    agent = _Agent()
    input_data = np.array([0.25, 0.75])
    rewards: list[np.ndarray] = []

    def env_step(spikes: np.ndarray) -> float:
        rewards.append(spikes.copy())
        return 0.5

    reward_synapse = agent.synapses[0][0]
    before_weight = reward_synapse.w

    SCTrainingLoop.run_rl_epoch(agent, env_step, input_data, generations=3)

    assert len(agent.epochs) == 3
    assert len(rewards) == 3
    assert reward_synapse.w > before_weight


def test_multimodal_fusion_uses_declared_sample_count_and_accumulates_outputs():
    layer = _FusionLayer()

    SCTrainingLoop.train_multimodal_fusion(layer, _DatasetWithSamples(), epochs=2)

    assert [sample.tolist() for sample in layer.samples] == [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]


def test_multimodal_fusion_falls_back_to_label_count_when_sample_count_is_absent():
    layer = _FusionLayer()

    SCTrainingLoop.train_multimodal_fusion(layer, _DatasetWithLabelsOnly(), epochs=1)

    assert [sample.tolist() for sample in layer.samples] == [[0.0], [1.0], [2.0]]
