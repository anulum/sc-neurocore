<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 47: Auto-Fit -- Neuron Model Fitting

Record from a real neuron. Auto-Fit sweeps 13 models from the library,
optimizes parameters, ranks by fit quality. The best model is ready
for simulation and FPGA deployment.

## 1. Prepare Recording

```python
import numpy as np
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

hh = HodgkinHuxleyNeuron()
current = np.zeros(500)
current[50:450] = 10.0
voltage = np.zeros(500)
for t in range(500):
    hh.step(float(current[t]))
    voltage[t] = hh.v
```

## 2. Fit Models

```python
from sc_neurocore.autofit import fit

results = fit(voltage, current, dt=0.1, top_k=5)
for r in results:
    print(f"{r.model_name}: RMSE={r.rmse:.2f}, score={r.combined_score:.3f}")
```

## 3. Extract Features

```python
from sc_neurocore.autofit.features import extract_features
feats = extract_features(voltage, dt=0.1)
print(f"Spikes: {feats['spike_count']}, Rate: {feats['firing_rate']:.1f} Hz")
```

## Fittable Models

LIF, HH, Izhikevich, AdEx, FitzHugh-Nagumo, Morris-Lecar,
Hindmarsh-Rose, Lapicque, QIF, ExpIF, Alpha, Theta, Resonate-and-Fire.

## Further Reading

- [API: Auto-Fit](../api/autofit.md)
