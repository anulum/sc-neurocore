# MontbrioMeanField

**Module:** `engine/src/neurons/population.rs`
**Reference:** Montbrio, Pazo & Roxin, Phys Rev X 5:021028, 2015
**Family:** Exact mean-field reduction of QIF neuron population
**State variables:** `r` (population firing rate), `v` (mean membrane potential)

---

## Biological Context

The Montbrio-Pazo-Roxin (MPR) model is the exact mean-field reduction of an infinite population of quadratic integrate-and-fire (QIF) neurons with Lorentzian-distributed heterogeneity. Unlike phenomenological mean-field models, this derivation is mathematically exact, capturing the collective dynamics (synchrony, oscillations, bistability) with just 2 ODEs.

Key features:
- **Exact reduction**: not an approximation — captures full population dynamics
- **2 ODEs only**: r (firing rate) and v (mean voltage) suffice
- **Lorentzian heterogeneity**: parameter delta controls distribution width
- **Recurrent coupling**: parameter J models excitatory self-coupling
- **Bifurcation structure**: saddle-node, Hopf, and SNIC bifurcations accessible

---

## Equations

$$\tau \frac{dr}{dt} = \frac{\Delta}{\pi \tau} + 2rv$$

$$\tau \frac{dv}{dt} = v^2 + \eta + I + J\tau r - (\pi \tau r)^2$$

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 0.01 | Population firing rate (Hz) |
| `v` | -2.0 | Mean membrane potential |
| `delta` | 1.0 | Heterogeneity width |
| `eta` | -5.0 | Mean excitability |
| `tau` | 1.0 | Membrane time constant (ms) |
| `j` | 15.0 | Synaptic coupling |
| `dt` | 0.01 | Integration step |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/population.rs` |
| PyO3 wrapper | Yes (state: r, v) |
| NetworkRunner wired | `NeuronVariant::MontbrioMPR` |
| `create_neuron("MontbrioMeanField")` | Yes |
| `supported_models()` | Includes "MontbrioMeanField" |
| STRONG tests | 10 |
| Benchmark | `montbrio_100k_steps`: **3.92 ms** (39.2 ns/step), i5-11600K |
