# TasteReceptorCell

**Module:** `engine/src/neurons/sensory/taste_receptor_cell.rs`
**Reference:** Chaudhari & Roper 2010; Liman et al. 2014
**Family:** Graded sensory receptor, gustatory IP3/Ca2+/ATP signalling
**State variables:** `v` (receptor potential), `ca` (intracellular Ca2+), `ip3` (IP3 concentration), `atp_release` (ATP output)

---

## Biological Context

Taste receptor cells (Type II cells) transduce sweet, bitter, and umami stimuli in taste buds. They lack conventional synapses and instead release ATP through CALHM1/3 channels as a paracrine transmitter to gustatory afferent nerves.

Key features:
- Tastant binding to T1R/T2R GPCRs activates gustducin -> PLC-beta2 -> IP3 production
- IP3 triggers Ca2+ release from endoplasmic reticulum (ER) stores via IP3R3
- Ca2+ activates TRPM5 channels -> depolarisation -> CALHM1/3 opening -> ATP release
- ATP acts on P2X2/P2X3 receptors on gustatory afferent fibres
- Graded output: ATP release is proportional to intracellular Ca2+
- No conventional action potentials in Type II cells (though some show non-conventional spikes)

The model implements the IP3 -> Ca2+ -> ATP cascade with first-order kinetics and a Hill-type tastant binding function. The primary output is the receptor potential (mV), with `atp_release` as a secondary readable state.

---

## Equations

### GPCR -> IP3 production (Hill function)

$$IP3_{target} = \frac{C}{C + 0.5}$$

where $C$ is tastant concentration (clamped $\geq 0$).

### IP3 dynamics

$$\frac{d[IP3]}{dt} = \frac{IP3_{target} - [IP3]}{\tau_{IP3}}$$

IP3 is clamped to $[0, 1]$.

### Ca2+ release from ER

$$Ca_{release} = [IP3]^2 \cdot (1 - [Ca^{2+}])$$

The quadratic IP3 dependence models cooperative IP3R gating. The $(1 - Ca)$ term provides store depletion feedback.

### Ca2+ dynamics

$$\frac{d[Ca^{2+}]}{dt} = Ca_{release} - \frac{[Ca^{2+}]}{\tau_{Ca}}$$

Ca2+ is clamped to $[0, 1]$.

### TRPM5 current and membrane potential

$$I_{TRPM5} = gain \cdot [Ca^{2+}] \cdot 20$$

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + I_{TRPM5}}{\tau}$$

### ATP release

$$ATP_{release} = [Ca^{2+}]$$

ATP release is directly proportional to intracellular Ca2+.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -50.0 | mV | Receptor potential |
| `v_rest` | -50.0 | mV | Resting potential |
| `tau` | 10.0 | ms | Membrane time constant |
| `ca` | 0.0 | — | Normalised intracellular Ca2+ [0, 1] |
| `ip3` | 0.0 | — | Normalised IP3 concentration [0, 1] |
| `tau_ip3` | 100.0 | ms | IP3 dynamics time constant |
| `tau_ca` | 200.0 | ms | Ca2+ decay (pump) time constant |
| `gain` | 1.0 | — | TRPM5 current gain |
| `atp_release` | 0.0 | — | ATP release rate (output state) |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/taste_receptor_cell.rs` |
| PyO3 wrapper | `py_sensory_graded!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | **No** — graded model, `step()` returns `f64` |
| `create_neuron("TasteReceptorCell")` | No (not in NetworkRunner variant enum) |
| coverage tests | 6 (depolarise, ATP release, no-response, Ca²⁺ bounds, reset, constructor/default equivalence) |
| NaN/extreme input test | Covered by `ca_bounded` test (10k steps at extreme input) |
| Benchmark | Not benchmarked (shares arch with olfactory, ~100 ns/step) |

Graded sensory models are accessed directly via PyO3 (`TasteReceptorCell` class) or Rust. They are not routed through `NetworkRunner` because the network step loop expects `i32` spike outputs.

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| 10k steps | ~1.0 ms (estimated) |
| Per step | ~100 ns |

The step function evaluates one division (Hill), one `powi(2)`, three linear ODEs, and two `clamp` calls. No `exp()`. Expected cost under 500 ns/step.

---

## Findings

1. **Quadratic IP3 dependence models cooperative IP3R gating.** The $[IP3]^2$ term means Ca2+ release increases steeply with IP3 concentration, producing a threshold-like response curve.
2. **Store depletion term (1 - Ca) prevents Ca2+ runaway.** As Ca2+ rises toward 1.0, the release rate drops to zero. Combined with the pump term ($-Ca/\tau_{Ca}$), Ca2+ is guaranteed to stay in $[0, 1]$.
3. **ATP release directly tracks Ca2+.** The `atp_release` state is set equal to Ca2+ each step, providing a simple readout of the cell's excitation level.
4. **Slow time constants (tau_ip3 = 100 ms, tau_ca = 200 ms).** Taste responses are slow compared to auditory or somatosensory transduction, matching the perceptual timescale of gustation.
5. **No response at zero tastant concentration.** The Hill function produces IP3_target = 0 at C = 0, confirmed by the `taste_no_response_without_tastant` test.
