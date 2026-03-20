# Tutorial 28: 13 Learning Rules — Complete Plasticity Catalog

SC-NeuroCore ships 13 learning rules spanning Hebbian, gradient-based,
reward-modulated, and meta-learning paradigms.

## Spike-Timing Dependent Plasticity (3 variants)

### Pair STDP (Bi & Poo 1998)
```python
from sc_neurocore.synapses.stochastic_stdp import StochasticSTDPSynapse
```

### Triplet STDP (Pfister & Gerstner 2006)
Captures rate-dependence of LTP/LTD via 4 traces (r1, r2, o1, o2):
```python
from sc_neurocore.synapses.triplet_stdp import TripletSTDP
syn = TripletSTDP(tau_plus=16.8, tau_minus=33.7)
for t in range(1000):
    syn.step(pre_spike=t%20==0, post_spike=t%25==0, dt=1.0)
```

### Voltage-Based STDP (Clopath et al. 2010)
Unifies rate and timing plasticity via membrane voltage traces:
```python
from sc_neurocore.synapses.clopath_stdp import ClopathSTDP
syn = ClopathSTDP(a_ltp=8e-5, a_ltd=14e-5)
syn.step(pre_spike=True, u_post=-30.0, dt=0.5)
```

## BCM Metaplasticity (Bienenstock-Cooper-Munro 1982)
Sliding threshold: high-rate neurons become harder to potentiate:
```python
from sc_neurocore.synapses.bcm import BCMSynapse
syn = BCMSynapse(eta=0.01, tau_theta=1000.0)
```

## Gradient-Based (3 variants)

### BPTT with Surrogate Gradients
```python
from sc_neurocore.learning.advanced import BPTTLearner
learner = BPTTLearner(network, loss_fn=mse, lr=1e-3)
```

### Truncated BPTT (Williams & Peng 1990)
Memory O(k) instead of O(T):
```python
from sc_neurocore.learning.advanced import TBPTTLearner
learner = TBPTTLearner(network, loss_fn=mse, lr=1e-3, k=50)
```

### PyTorch Surrogate Training
6 surrogates + learnable beta/threshold on all 10 cell types:
```python
from sc_neurocore.training.snn_modules import SpikingNet
net = SpikingNet(784, 128, 10, learn_beta=True, learn_threshold=True)
```

## Three-Factor Rules

### Eligibility Traces (e-prop, Bellec et al. 2020)
```python
from sc_neurocore.learning.advanced import EligibilityTrace
et = EligibilityTrace(tau_e=20.0)
delta = et.update(pre_spike, post_spike, error_signal)
```

### Reward-Modulated STDP
```python
from sc_neurocore.learning.advanced import RewardModulatedLearner
learner = RewardModulatedLearner(network, tau_reward=100.0)
learner.step(reward=1.0)
```

## Meta-Learning (MAML, Finn et al. 2017)
```python
from sc_neurocore.learning.advanced import MetaLearner
ml = MetaLearner(network, inner_lr=0.01, outer_lr=0.001)
ml.outer_step(tasks)
```

## Continual Learning (EWC, Kirkpatrick et al. 2017)
```python
from sc_neurocore.learning.lifelong import EWC_SCLayer
layer = EWC_SCLayer(n_inputs=10, n_neurons=5, ewc_lambda=10.0)
layer.consolidate_task()
layer.apply_ewc_penalty(step_size=0.01)
```

## Homeostatic + Structural
```python
from sc_neurocore.learning.advanced import (
    HomeostaticPlasticity, ShortTermPlasticity, StructuralPlasticity,
)
```
