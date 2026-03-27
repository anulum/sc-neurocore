<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 47: Auto-Fit — Neuron Model Fitting

Record from a real neuron (or generate synthetic data). Auto-Fit sweeps
13 models from SC-NeuroCore's library, optimises parameters for each,
and ranks by fit quality. The best model is ready for simulation and
FPGA deployment — no manual tuning needed.

## Why Auto-Fit

Choosing the right neuron model is the first decision in any SNN
project. LIF is simple but misses adaptation. HH is accurate but
expensive on FPGA. Auto-Fit makes this decision data-driven:

| Manual Approach | Auto-Fit |
|----------------|----------|
| Pick a model by intuition | Sweep 13 models systematically |
| Tune parameters by hand | Optimise via scipy.minimize |
| Hope it matches data | Quantify fit with RMSE + spike timing |

## 1. Prepare a Recording

Generate a target voltage trace (or load real electrophysiology data):

```python
import numpy as np
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

# Simulate a "ground truth" HH recording
hh = HodgkinHuxleyNeuron()
current = np.zeros(500, dtype=np.float32)
current[50:450] = 10.0  # 400ms step current

voltage = np.zeros(500, dtype=np.float32)
for t in range(500):
    hh.step(float(current[t]))
    voltage[t] = hh.v

print(f"Recording: {len(voltage)} timesteps, "
      f"range [{voltage.min():.1f}, {voltage.max():.1f}] mV")
```

For real data: load from `.abf`, `.nwb`, or CSV. The only requirement
is a voltage trace and corresponding current injection.

## 2. Fit Models

```python
from sc_neurocore.autofit import fit

results = fit(
    voltage=voltage,
    current=current,
    dt=0.1,       # timestep in ms
    top_k=5,      # return best 5 models
)

print("Top 5 models:")
for i, r in enumerate(results):
    print(f"  {i+1}. {r.model_name:25s} RMSE={r.rmse:>7.2f} mV, "
          f"spike_timing={r.spike_timing_error:>6.2f} ms, "
          f"score={r.combined_score:.4f}")
```

The combined score balances voltage RMSE and spike timing accuracy.
Lower is better.

## 3. Extract Features

Before fitting, inspect the recording's electrophysiological features:

```python
from sc_neurocore.autofit.features import extract_features

feats = extract_features(voltage, dt=0.1)
print(f"Spikes: {feats['spike_count']}")
print(f"Rate: {feats['firing_rate']:.1f} Hz")
print(f"ISI CV: {feats['isi_cv']:.3f}")
print(f"Adaptation ratio: {feats['adaptation_ratio']:.3f}")
print(f"Rheobase estimate: {feats['rheobase']:.1f} nA")
```

Features guide model selection: high adaptation ratio → AdEx or ALIF,
high ISI CV → bursting model, regular firing → LIF.

## 4. Use the Best Model

```python
best = results[0]
print(f"Best model: {best.model_name}")
print(f"Optimal parameters: {best.params}")

# Create an instance with fitted parameters
model = best.create_model()

# Simulate with the fitted model
fitted_voltage = np.zeros(500, dtype=np.float32)
for t in range(500):
    model.step(float(current[t]))
    fitted_voltage[t] = model.v

# Verify fit quality
rmse = np.sqrt(np.mean((voltage - fitted_voltage) ** 2))
print(f"Verified RMSE: {rmse:.2f} mV")
```

## Fittable Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| LIFNeuron | 4 | Regular spiking, fast FPGA |
| IzhikevichNeuron | 6 | Diverse patterns |
| AdExNeuron | 7 | Adaptation, bursting |
| HodgkinHuxleyNeuron | 12 | Biophysical accuracy |
| FitzHughNagumoNeuron | 5 | Oscillations |
| MorrisLecarNeuron | 8 | Type I/II excitability |
| HindmarshRoseNeuron | 7 | Chaotic bursting |
| LapicqueNeuron | 3 | Simplest model |
| QuadraticIFNeuron | 4 | Near-threshold dynamics |
| ExpIFNeuron | 5 | Sharp spike initiation |
| AlphaNeuron | 4 | Synaptic conductance |
| ThetaNeuron | 3 | Phase oscillator |
| ResonateAndFireNeuron | 5 | Subthreshold resonance |

## Integration with Studio

In the Visual Studio:
1. Import a voltage trace (CSV) via the Import button
2. The trace appears as an overlay on the Trace view
3. Browse models and visually compare against the imported data
4. Auto-Fit ranks models automatically

## FPGA Deployment

The fitted model deploys directly to FPGA:
```python
# Fit → compile → synthesise
best_model = results[0].create_model()
# Export to Verilog via equation compiler or Studio pipeline
```

## References

- Pospischil et al. (2008). "Minimal Hodgkin-Huxley type models for
  different classes of cortical and thalamic neurons." Biological
  Cybernetics 99(4):427-441.
- Rossant et al. (2011). "Fitting Neuron Models to Spike Trains."
  Front. Neurosci. 5:9.
