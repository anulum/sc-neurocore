# GolgiCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Solinas et al., Front Cell Neurosci 1:2, 2007; D'Angelo & De Schutter, Cerebellum 8:399, 2009
**Family:** WB Na+/K+ core + A-type K+ + Ca2+-dependent AHP
**State variables:** `v`, `m`, `h`, `n` (WB gating), `a`, `b` (A-type K+), `ca` (intracellular Ca2+)

---

## Biological Context

Golgi cells are large inhibitory interneurons in the cerebellar granular layer. They provide the primary source of tonic and phasic GABAergic (and glycinergic) inhibition to granule cells at glomeruli, forming a feedforward and feedback inhibitory loop that controls granule cell excitability.

Key features:
- **Spontaneous pacemaker firing** (3-10 Hz): intrinsic membrane properties enable autonomous rhythmic firing without synaptic input
- **Wide dendritic arbour**: extends into the molecular layer to receive parallel fibre input (feedback from granule cells)
- **A-type K+ current**: produces onset delay and phasic pause, contributing to temporal filtering
- **Ca2+-dependent AHP**: slow afterhyperpolarisation limits sustained firing rate (spike frequency adaptation)
- **Mixed GABA/glycine release**: both inhibitory neurotransmitters released at glomerular synapses

---

## Equations

### Wang-Buzsáki Na+/K+ gating with phi scaling

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_A - I_{AHP} - I_L + I_{ext}$$

$$I_{Na} = g_{Na} m^3 h (V - E_{Na})$$
$$I_K = g_K n^4 (V - E_K)$$
$$I_A = g_A a^3 b (V - E_K)$$
$$I_{AHP} = g_{AHP} \frac{[Ca^{2+}]}{[Ca^{2+}] + 0.5} (V - E_K)$$

Gate kinetics: WB alpha/beta rates with phi=5 scaling (via `safe_rate`).

### A-type K+ gating (Connor-Stevens style)

$$a_\infty = \frac{1}{1 + \exp(-(V+50)/20)}, \quad \tau_a = 2 \text{ ms}$$
$$b_\infty = \frac{1}{1 + \exp((V+70)/6)}, \quad \tau_b = 50 \text{ ms}$$

### Ca2+ dynamics

$$\frac{d[Ca^{2+}]}{dt} = -\frac{[Ca^{2+}]}{\tau_{Ca}}$$

On spike: $[Ca^{2+}] \leftarrow [Ca^{2+}] + 0.2$, $\tau_{Ca} = 200$ ms.

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -60.0 | mV | Membrane potential |
| `m` | 0.05 | — | Na+ activation |
| `h` | 0.6 | — | Na+ inactivation |
| `n` | 0.32 | — | Kdr activation |
| `a` | 0.1 | — | A-type K+ activation |
| `b` | 0.8 | — | A-type K+ inactivation |
| `ca` | 0.0 | mM | Intracellular Ca2+ |
| `g_na` | 35.0 | mS/cm² | Na+ conductance |
| `g_k` | 9.0 | mS/cm² | Kdr conductance |
| `g_a` | 2.0 | mS/cm² | A-type K+ conductance |
| `g_ahp` | 0.5 | mS/cm² | Ca2+-dependent K+ (AHP) |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `phi` | 5.0 | — | Kinetic scaling factor |
| `dt` | 0.5 | ms | Integration timestep (4 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, m, h, n, a, b, ca) |
| NetworkRunner wired | `NeuronVariant::Golgi` |
| `create_neuron("GolgiCell")` | Yes |
| `supported_models()` | Includes "GolgiCell" |
| STRONG tests | 11 (fire, spontaneous, adaptation, A-type delay, Ca2+, negative, NaN, extreme, reset, gates, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `golgi_1k_steps`: **396 µs** (396 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| golgi_1k_steps | 396 µs |
| Per step | **396 ns** |

WB gating with 4 sub-steps (dt=0.125 ms) + A-type K+ + Ca2+-dependent AHP. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=15. Verified.
2. **Near-spontaneous with minimal input.** Fires with very low current due to depolarised leak. Verified.
3. **Adaptation via Ca2+-AHP.** Spike frequency decreases during sustained drive. Verified.
4. **A-type K+ delays onset.** Removing g_a shortens latency to first spike. Verified.
5. **Ca2+ accumulates during spiking.** Ca2+ > 0 after sustained firing. Verified.
6. **Reset clears all state.** All variables return to initial values. Verified.
