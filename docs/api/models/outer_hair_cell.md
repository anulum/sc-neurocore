# OuterHairCell

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Dallos 2008; Hudspeth 2008
**Family:** Graded sensory receptor, auditory electromotility
**State variables:** `v` (receptor potential), `motility` (somatic length change)

---

## Biological Context

Outer hair cells (OHCs) are the cochlear amplifier. Roughly 12,000 OHCs per human cochlea provide active mechanical feedback to the basilar membrane via prestin-driven somatic electromotility. This amplifies weak sounds by up to 40-60 dB and sharpens frequency tuning.

Key features:
- MET channels on stereocilia open with basilar membrane displacement (same Boltzmann gating as IHC)
- More sensitive than IHCs: lower x_half, steeper slope
- Prestin (SLC26A5) in the lateral wall changes cell length in response to membrane potential
- Motility is a nonlinear (Boltzmann) function of V, centred around resting potential
- Compressive nonlinearity: gain decreases with increasing sound level
- Graded output, no action potentials

The model captures MET transduction and the voltage-dependent prestin motility function. The motility output can drive basilar membrane feedback in a cochlear model.

---

## Equations

### MET channel gating

$$p_{open} = \frac{1}{1 + \exp\left(-\frac{x - x_{half}}{s}\right)}$$

### Receptor potential dynamics

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + g_{MET} \cdot p_{open} \cdot (0 - V)}{\tau}$$

### Prestin electromotility

$$motility = \frac{gain}{1 + \exp\left(-\frac{V + 40}{10}\right)} - \frac{gain}{2}$$

The motility output is centred at zero when V = -40 mV (the Boltzmann midpoint) and ranges from $-gain/2$ to $+gain/2$.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -70.0 | mV | Receptor potential |
| `v_rest` | -70.0 | mV | Resting potential |
| `tau` | 0.3 | ms | Membrane time constant |
| `g_met` | 15.0 | — | MET channel max conductance |
| `x_half` | 20.0 | nm | Boltzmann half-activation displacement |
| `s` | 6.0 | nm | Boltzmann slope factor |
| `motility` | 0.0 | — | Normalised somatic length change |
| `gain` | 5.0 | — | Prestin gain factor |
| `dt` | 0.025 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_sensory_graded!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | **No** — graded model, `step()` returns `f64` |
| `create_neuron("OuterHairCell")` | No (not in NetworkRunner variant enum) |
| STRONG tests | 3 (depolarise + motility, reset, bounded) |
| NaN/extreme input test | Covered by `bounded` test (10k steps at max input) |
| Benchmark | Not benchmarked (shares arch with IHC, ~20 ns/step) |

Graded sensory models are accessed directly via PyO3 (`OuterHairCell` class) or Rust. They are not routed through `NetworkRunner` because the network step loop expects `i32` spike outputs.

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| 10k steps | ~195 µs (estimated) |
| Per step | ~20 ns |

Step function evaluates two `exp()` calls (MET Boltzmann + prestin Boltzmann) and one linear ODE. Expected cost comparable to IHC.

---

## Findings

1. **Lower x_half (20 nm) makes OHC more sensitive than IHC (50 nm).** This matches biology: OHCs respond to smaller displacements to provide amplification.
2. **Prestin motility is bipolar.** Depolarisation causes cell shortening (positive motility), hyperpolarisation causes elongation (negative motility), centred at V = -40 mV.
3. **Faster tau (0.3 ms) than IHC (0.5 ms).** OHCs need to follow cycle-by-cycle basilar membrane motion at high frequencies.
4. **Motility resets to zero on `reset()`.** Receptor potential returns to v_rest = -70 mV.
