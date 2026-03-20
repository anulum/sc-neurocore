# Learning

Training paradigms beyond single-node STDP: BPTT, truncated BPTT,
eligibility traces, reward-modulated learning, meta-learning,
homeostatic scaling, short-term plasticity, structural plasticity,
federated learning, lifelong/continual learning, and neuroevolution.

## BPTT Learner

::: sc_neurocore.learning.advanced.BPTTLearner

## Truncated BPTT (Williams & Peng 1990)

Chunks long sequences into windows of k timesteps, backpropagating
gradients within each chunk while carrying membrane state forward.
Memory O(k) instead of O(T).

::: sc_neurocore.learning.advanced.TBPTTLearner

## Eligibility Traces (e-prop, Bellec et al. 2020)

::: sc_neurocore.learning.advanced.EligibilityTrace

## Reward-Modulated STDP

::: sc_neurocore.learning.advanced.RewardModulatedLearner

## Meta-Learning (MAML, Finn et al. 2017)

::: sc_neurocore.learning.advanced.MetaLearner

## Homeostatic Plasticity (Turrigiano 2008)

::: sc_neurocore.learning.advanced.HomeostaticPlasticity

## Short-Term Plasticity (Tsodyks-Markram 1997)

::: sc_neurocore.learning.advanced.ShortTermPlasticity

## Structural Plasticity

::: sc_neurocore.learning.advanced.StructuralPlasticity

## Federated

::: sc_neurocore.learning.federated

## Lifelong (EWC)

Elastic Weight Consolidation with active penalty: pushes drifted weights
back toward consolidated values, weighted by Fisher information.

::: sc_neurocore.learning.lifelong

## Neuroevolution

::: sc_neurocore.learning.neuroevolution
