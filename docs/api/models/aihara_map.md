# AiharaMapNeuron

**Module:** `engine/src/neurons/maps.rs`
**Reference:** Aihara et al., Phys Lett A 144:333, 1990
**Family:** 2D discrete chaotic neuron map
**State variables:** `x` (fast/output), `y` (slow/recovery)

---

## Biological Context

The Aihara chaotic neuron map is one of the earliest discrete-time models capable of generating chaotic spiking dynamics. It uses a sigmoid nonlinearity to model the relationship between membrane potential and output, with a slow recovery variable providing negative feedback.

Key features:
- **Chaotic dynamics**: depending on parameters, produces periodic, quasi-periodic, or chaotic spiking
- **Bursting**: parameter regimes exist for burst-like patterns
- **Fast computation**: no differential equations, pure arithmetic per step
- **Associative memory**: used in chaotic neural networks for pattern retrieval

---

## Equations

$$x(n+1) = k_f \cdot x(n) \cdot \sigma(x(n) + \alpha) - y(n) + I$$
$$y(n+1) = k_s \cdot y(n) + \delta \cdot x(n)$$

where $\sigma(z) = 1/(1 + \exp(-z))$.

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `x` | 0.0 | Fast variable |
| `y` | 0.0 | Slow variable |
| `k_f` | 0.7 | Fast decay |
| `k_s` | 0.95 | Slow decay |
| `alpha` | 2.0 | Sigmoid offset |
| `delta` | 0.05 | Slow coupling |
| `x_threshold` | 0.5 | Spike detection |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/maps.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: x, y) |
| NetworkRunner wired | `NeuronVariant::AiharaMap` |
| `create_neuron("AiharaMap")` | Yes |
| `supported_models()` | Includes "AiharaMap" |
| STRONG tests | 9 (fire, silent, chaos, negative, NaN, extreme, reset, rate-input, performance) |
| Benchmark | `aihara_100k_steps`: **1.97 ms** (19.7 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| aihara_100k_steps | 1.97 ms |
| Per step | **19.7 ns** |

Discrete map, no sub-stepping. Measured 2026-04-04.
