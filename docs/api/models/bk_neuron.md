# BKNeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Bhatt & Storm, J Physiol 557:329, 2003; Faber & Bhatt, PNAS 100:2813, 2003
**Family:** WB Na+/K+ base + BK (big conductance Ca2+-activated K+)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `ca` (intracellular Ca2+)

---

## Biological Context

BK (big conductance, MaxiK, KCa1.1) channels have the largest single-channel conductance (~250 pS) of any K+ channel. They require both membrane depolarisation and intracellular Ca2+ for full activation. During action potentials, Ca2+ influx through voltage-gated Ca2+ channels activates BK, producing fast repolarisation and a prominent fast afterhyperpolarisation (fAHP).

Key features:
- **Fast AHP**: BK activation during/after spikes produces rapid deep AHP
- **AP narrowing**: BK shortens action potential duration
- **Burst termination**: Ca2+ accumulation during bursts progressively activates BK
- **Frequency control**: stronger BK → longer interspike intervals

---

## Equations

### WB base + BK

$$I_{BK} = g_{BK} \cdot m_{BK,\infty}(V, [Ca^{2+}]) \cdot (V - E_K)$$

$$m_{BK,\infty} = \frac{1}{1 + \exp(-(V - V_{1/2}^{BK})/15)}$$

$$V_{1/2}^{BK} = 10 - 30 \cdot \frac{[Ca^{2+}]}{[Ca^{2+}] + 0.5}$$

On spike: $[Ca^{2+}] \leftarrow [Ca^{2+}] + 0.3$. Decay: $\tau_{Ca} = 50$ ms.

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `ca` | 0.0 | mM | Intracellular Ca2+ |
| `g_bk` | 3.0 | mS/cm² | BK conductance |
| `tau_ca` | 50.0 | ms | Ca2+ decay time constant |
| `phi` | 5.0 | — | Kinetic scaling |
| `dt` | 0.5 | ms | Timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, ca) |
| NetworkRunner wired | `NeuronVariant::BK` |
| `create_neuron("BK")` | Yes |
| `supported_models()` | Includes "BK" |
| STRONG tests | 10 (fire, silent, Ca2+ accumulation, AHP deepening, rate reduction, negative, NaN, extreme, reset, performance) |
| Benchmark | `bk_1k_steps`: **3.16 ms** (3.16 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| bk_1k_steps | 3.16 ms |
| Per step | **3.16 µs** |

WB gating with 50 sub-steps + BK + Ca2+ dynamics. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=3. Verified.
2. **Silent without input.** No spontaneous firing. Verified.
3. **Ca2+ accumulates during spiking.** ca > 0 after sustained firing. Verified.
4. **BK deepens AHP.** Ca2+ builds during spiking, activating BK. Verified.
5. **BK reduces firing rate.** Removing g_bk increases spike count. Verified.
6. **Reset clears state.** v=-65, ca=0 after reset. Verified.
