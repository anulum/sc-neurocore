# Neuron Integrator Paths

SC-NeuroCore now exposes explicit baseline and higher-order integration paths
for selected neuron models where integrator choice materially affects the
numerics.

## Purpose

The goal is clarity, not silent replacement.

- the historical path remains the default
- the alternative path is explicit and opt-in
- tests compare both paths immediately

This avoids confusion between:

- preserving established behaviour
- evaluating a more accurate integrator on the same model

## Current Models

| Model | Default path | Alternative path | Why it exists |
|---|---|---|---|
| `SCIzhikevichNeuron` | `baseline_half_euler` | `rk4` | quadratic voltage term benefits from a clearer explicit higher-order reference |
| `HodgkinHuxleyNeuron` | `baseline_euler` | `rk4` | four coupled ion-channel ODEs are sensitive to step method |
| `AdExNeuron` | `baseline_euler` | `rk4` | exponential spike-initiation term benefits from a higher-order alternative |

## How To Use

```python
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.models.adex import AdExNeuron

izh = SCIzhikevichNeuron(integrator="rk4")
hh = HodgkinHuxleyNeuron(integrator="rk4")
adex = AdExNeuron(integrator="rk4")
```

Baseline-preserving construction:

```python
izh = SCIzhikevichNeuron()          # baseline_half_euler
hh = HodgkinHuxleyNeuron()          # baseline_euler
adex = AdExNeuron()                 # baseline_euler
```

## Design Rules

- default construction must preserve historical behaviour
- alternative paths must be named explicitly in the constructor
- tests must cover default preservation and candidate-path stability
- docs must state what each path means

## What This Does Not Claim

- it does not claim that RK4 is universally the best method for every neuron
- it does not claim that every model in `neurons/models/` has already been
  migrated
- it does not claim that a stiffness-specific Rosenbrock path exists today

The current state is explicit:

- baseline path kept
- RK4 path added for the first three priority models
- further integrator work should follow the same pattern
