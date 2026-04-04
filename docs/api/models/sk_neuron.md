# SKNeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Bhatt & Storm, J Physiol 557:329, 2003; Stocker, Nat Rev Neurosci 5:758, 2004
**Family:** WB Na+/K+ base + SK (small conductance Ca2+-activated K+)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `ca` (intracellular Ca2+)

---

## Biological Context

SK (KCa2.x) channels are activated solely by intracellular Ca2+ without voltage dependence. They produce the medium afterhyperpolarisation (mAHP) lasting 50-200 ms, which is the primary mechanism for spike frequency adaptation in many cortical and hippocampal neurons.

Key features:
- **Medium AHP (mAHP)**: Ca2+ accumulation during firing activates SK → hyperpolarisation
- **Spike frequency adaptation**: progressive SK activation limits sustained firing rate
- **No voltage dependence**: activation purely via Hill function of [Ca2+]
- **Synaptic plasticity**: SK in dendritic spines gates NMDA receptor activation

---

## Equations

$$I_{SK} = g_{SK} \cdot \frac{[Ca^{2+}]^2}{[Ca^{2+}]^2 + K_d^2} \cdot (V - E_K)$$

$K_d = 0.5$ mM (half-activation). On spike: $[Ca^{2+}] += 0.2$, $\tau_{Ca} = 150$ ms.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, ca) |
| NetworkRunner wired | `NeuronVariant::SK` |
| `create_neuron("SK")` | Yes |
| `supported_models()` | Includes "SK" |
| STRONG tests | 10 (fire, silent, adaptation, Ca2+-only, rate reduction, negative, NaN, extreme, reset, performance) |
| Benchmark | `sk_1k_steps`: **2.79 ms** (2.79 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| sk_1k_steps | 2.79 ms |
| Per step | **2.79 µs** |

WB gating with 50 sub-steps + SK + Ca2+ dynamics. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=2. Verified.
2. **Silent without input.** No spontaneous firing. Verified.
3. **Spike frequency adaptation.** Early epochs fire more than late. Verified.
4. **Ca2+-only dependence.** SK inactive at ca=0 (sk_inf < 0.001). Verified.
5. **SK reduces firing rate.** Removing g_sk increases spike count. Verified.
6. **Reset clears state.** v=-65, ca=0 after reset. Verified.
