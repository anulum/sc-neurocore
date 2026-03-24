<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 28: 13 Learning Rules — Complete Plasticity Catalog

SC-NeuroCore ships 13 learning rules spanning Hebbian, gradient-based,
reward-modulated, and meta-learning paradigms. This tutorial explains
each rule, when to use it, and how it maps to the code.

## Decision Guide

| If you need... | Use | Category |
|---|---|---|
| Unsupervised feature extraction | Pair/Triplet STDP | Hebbian |
| Biologically realistic learning | Voltage STDP (Clopath) | Hebbian |
| Homeostatic rate regulation | BCM | Hebbian |
| Supervised classification (small networks) | BPTT + surrogate | Gradient |
| Supervised classification (long sequences) | TBPTT | Gradient |
| GPU training + FPGA deployment | PyTorch surrogate → `to_sc_weights()` | Gradient |
| Reward-based RL | R-STDP or Eligibility traces | Three-factor |
| Online RL without backprop | e-prop | Three-factor |
| Few-shot learning / task adaptation | MAML | Meta |
| Continual learning (no forgetting) | EWC | Continual |
| Firing rate stabilization | Homeostatic plasticity | Homeostatic |
| Short-term facilitation/depression | STP | Homeostatic |
| Structural rewiring | Structural plasticity | Structural |

## Spike-Timing Dependent Plasticity (3 variants)

### 1. Pair STDP (Bi & Poo 1998)

The classic: if pre fires before post (causal), strengthen. If post fires
before pre (anti-causal), weaken.

```python
from sc_neurocore.synapses.stochastic_stdp import StochasticSTDPSynapse

syn = StochasticSTDPSynapse(
    learning_rate=0.01,  # STDP learning rate
    window_size=20,      # STDP window (timesteps)
    ltd_ratio=1.2,       # LTD/LTP ratio (>1 for stability)
)

# Simulate causal pairing: pre at t=10, post at t=15
for t in range(100):
    pre_bit = 1 if t == 10 else 0
    post_bit = 1 if t == 15 else 0
    syn.process_step(pre_bit=pre_bit, post_bit=post_bit)

print(f"Weight after causal pairing: {syn.w:.4f}")
```

**When to use**: Unsupervised feature extraction, receptive field development,
temporal sequence learning.

### 2. Triplet STDP (Pfister & Gerstner 2006)

Extends pair STDP to capture rate-dependence. Uses 4 traces (r1, r2, o1, o2)
to distinguish high-frequency from low-frequency pairings:

```python
from sc_neurocore.synapses.triplet_stdp import TripletSTDP

syn = TripletSTDP(tau_plus=16.8, tau_minus=33.7, tau_x=101.0, tau_y=125.0)

# High-frequency burst: pre and post at 50 Hz
for t in range(200):
    pre_spike = (t % 20 == 0)
    post_spike = (t % 20 == 5)
    syn.step(pre_spike=pre_spike, post_spike=post_spike, dt=1.0)

print(f"Weight after burst pairing: {syn.w:.4f}")
```

**When to use**: When pair STDP doesn't match experimental data — triplet STDP
reproduces BCM-like rate-dependence and frequency effects seen in slice experiments.

### 3. Voltage-Based STDP (Clopath et al. 2010)

Unifies rate and timing plasticity. Uses the postsynaptic voltage trace
instead of post-spike timing:

```python
from sc_neurocore.synapses.clopath_stdp import ClopathSTDP

syn = ClopathSTDP(a_ltp=8e-5, a_ltd=14e-5, theta_minus=-70.6, theta_plus=-45.3)

# Drive with voltage trace
for t in range(500):
    pre_spike = (t % 50 == 0)
    u_post = -65.0 + 30.0 * np.exp(-((t % 50) - 5)**2 / 10.0)  # EPSP
    syn.step(pre_spike=pre_spike, u_post=u_post, dt=0.5)

print(f"Weight: {syn.w:.6f}")
```

**When to use**: Most biologically realistic. Reproduces STDP, BCM, and
rate-coding results from a single mechanism.

## 4. BCM Metaplasticity (Bienenstock-Cooper-Munro 1982)

The sliding threshold prevents runaway potentiation. High-rate neurons
become harder to potentiate:

```python
from sc_neurocore.synapses.bcm import BCMSynapse

syn = BCMSynapse(eta=0.01, tau_theta=1000.0)

# Theta adapts to the neuron's recent activity
for t in range(5000):
    post_rate = 20.0 + 10.0 * np.sin(t / 500.0)  # oscillating rate
    syn.step(pre_rate=15.0, post_rate=post_rate, dt=1.0)

print(f"Weight: {syn.w:.4f}, Threshold: {syn.theta_m:.1f}")
```

**When to use**: Self-stabilizing networks, preventing epileptic-like runaway activity.

## Gradient-Based (3 variants)

### 5. BPTT with Surrogate Gradients

Backpropagation through time for SNNs, using differentiable approximations
of the spike function:

```python
from sc_neurocore.learning.advanced import BPTTLearner

# learner = BPTTLearner(network, loss_fn=mse, lr=1e-3)
# learner.train(inputs, targets, n_epochs=100)
```

**When to use**: Small supervised classification tasks where full temporal backprop fits in memory.

### 6. Truncated BPTT (Williams & Peng 1990)

Backprop through only the last $k$ timesteps. Memory $O(k)$ instead of $O(T)$:

```python
from sc_neurocore.learning.advanced import TBPTTLearner

# learner = TBPTTLearner(network, loss_fn=mse, lr=1e-3, k=50)
# Only backprop through last 50 steps — handles arbitrarily long sequences
```

**When to use**: Long temporal sequences where full BPTT is infeasible.

### 7. PyTorch Surrogate Training

6 surrogate gradient functions + learnable beta/threshold on all 10 SNN cell types.
This is the path to 99.49% MNIST:

```python
from sc_neurocore.training.snn_modules import SpikingNet

net = SpikingNet(784, 128, 10, learn_beta=True, learn_threshold=True)
# Train with standard PyTorch optimizer
# After training: net.to_sc_weights() exports to SC-NeuroCore format
```

**When to use**: Production classification. Train on GPU, deploy on FPGA via `to_sc_weights()`.

## Three-Factor Rules

### 8. Eligibility Traces / e-prop (Bellec et al. 2020)

Local learning rule: eligibility trace marks "who did what," then a global
reward/error signal decides "was it good?"

```python
from sc_neurocore.learning.advanced import EligibilityTrace

et = EligibilityTrace(tau_e=20.0)
# delta_w = et.update(pre_spike, post_spike, error_signal)
# Weight change = eligibility × error (three factors: pre, post, error)
```

**When to use**: Online learning without backprop. Biologically plausible RL.

### 9. Reward-Modulated STDP (R-STDP)

STDP with a global reward signal that gates weight changes:

```python
from sc_neurocore.learning.advanced import RewardModulatedLearner

# learner = RewardModulatedLearner(network, tau_reward=100.0)
# learner.step(reward=1.0)  # positive reward → reinforce recent STDP
# learner.step(reward=-1.0)  # negative → reverse recent STDP
```

**When to use**: Reinforcement learning in spiking networks. Robotics, decision-making.

## 10. Meta-Learning (MAML, Finn et al. 2017)

Learn to learn: optimize for fast adaptation to new tasks:

```python
from sc_neurocore.learning.advanced import MetaLearner

# ml = MetaLearner(network, inner_lr=0.01, outer_lr=0.001)
# ml.outer_step(tasks)  # each task: few-shot adapt + evaluate
```

**When to use**: Few-shot learning. When the network must quickly adapt to new data.

## 11. Continual Learning (EWC, Kirkpatrick et al. 2017)

Elastic Weight Consolidation: remember old tasks while learning new ones:

```python
from sc_neurocore.learning.lifelong import EWC_SCLayer

layer = EWC_SCLayer(n_inputs=10, n_neurons=5, ewc_lambda=10.0)

# Train on task A
# ...
layer.consolidate_task()  # compute Fisher information

# Train on task B — EWC penalty prevents forgetting A
layer.apply_ewc_penalty(step_size=0.01)
```

**When to use**: Sequential task learning without catastrophic forgetting.

## Homeostatic + Structural (3 rules)

### 12. Homeostatic Plasticity

Stabilizes firing rates by adjusting intrinsic excitability:

```python
from sc_neurocore.learning.advanced import HomeostaticPlasticity

hp = HomeostaticPlasticity(target_rate=10.0, tau=1000.0)
# Neurons firing above target → threshold increases
# Neurons firing below target → threshold decreases
```

### 13a. Short-Term Plasticity (STP)

Synaptic facilitation and depression on the timescale of 10-1000 ms:

```python
from sc_neurocore.learning.advanced import ShortTermPlasticity

stp = ShortTermPlasticity(tau_d=200.0, tau_f=600.0, u_se=0.2)
# High-frequency → depression (resources depleted)
# Low-frequency → facilitation (resources accumulate)
```

### 13b. Structural Plasticity

Create and destroy synapses based on activity:

```python
from sc_neurocore.learning.advanced import StructuralPlasticity

sp = StructuralPlasticity(growth_rate=0.001, prune_threshold=0.01)
# Low-weight synapses are pruned
# New synapses grow between co-active neurons
```

## Further Reading

- [Tutorial 03: Surrogate Gradient Training](03_surrogate_gradient_training.md) — GPU training pipeline
- [Tutorial 08: Online Learning](08_online_learning_stdp.md) — STDP in depth
- [Tutorial 34: ArcaneNeuron](34_arcane_neuron.md) — meta-learning rate dynamics
- [API: Learning](../api/learning.md) — auto-generated API docs
