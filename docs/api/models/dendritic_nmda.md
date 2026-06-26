# DendriticNMDANeuron

**Module:** `sc_neurocore.neurons.models.dendritic_nmda`
**Rust path:** `sc_neurocore_engine::neurons::multi_compartment::DendriticNMDANeuron`
**Reference:** Jahr & Stevens (1990), Schiller et al. (2000)
**Family:** Multi-compartment biophysical neurons
**State variables:** `v_soma` (somatic potential), `v_dend` (dendritic potential)

---

## 1. Mathematical Formalism

### Core equations

Two-compartment neuron with a soma and dendrite connected by coupling conductance $g_c$.
The dendrite contains NMDA receptors with the classical voltage-dependent magnesium block.

**Mg²⁺ block (Jahr & Stevens 1990, Equation 1):**

$$B(V) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]}{3.57} \cdot \exp(-0.062 \cdot V)}$$

where $[\text{Mg}^{2+}]$ is the extracellular magnesium concentration (default: 1.0 mM)
and $V$ is the dendritic membrane potential in mV. The constants 3.57 mM and 0.062 mV⁻¹
are empirical fits from voltage-clamp recordings in hippocampal neurons.

At rest ($V = -65$ mV): $B(-65) \approx 0.06$ — the channel is ~94% blocked.
At depolarised state ($V = 0$ mV): $B(0) \approx 0.78$ — the block is largely relieved.
At strong depolarisation ($V = +20$ mV): $B(20) \approx 0.93$ — nearly fully open.

This voltage dependence creates a **coincidence detector**: the NMDA channel requires
both glutamate binding (presynaptic) AND postsynaptic depolarisation to conduct.

**NMDA current:**

$$I_{\text{NMDA}} = g_{\text{NMDA}} \cdot \text{glutamate} \cdot B(V_d) \cdot (V_d - E_{\text{NMDA}})$$

where:
- $g_{\text{NMDA}}$ is the NMDA conductance (default: 1.5)
- glutamate is the presynaptic neurotransmitter concentration [0, 1]
- $B(V_d)$ is the Mg²⁺ block factor at dendritic potential
- $E_{\text{NMDA}} = 0$ mV is the NMDA reversal potential

**Dendritic dynamics:**

$$\tau_d \frac{dV_d}{dt} = -(V_d - E_L) + I_{\text{NMDA}} + g_c \cdot (V_s - V_d)$$

where $E_L = -65$ mV is the leak reversal and $g_c = 0.5$ is the coupling conductance.

**Somatic dynamics:**

$$\tau_s \frac{dV_s}{dt} = -(V_s - E_L) + I_{\text{ext}} + g_c \cdot (V_d - V_s)$$

**Spike condition:**

$$\text{spike} = \begin{cases} 1 & \text{if } V_s \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

with hard reset $V_s \leftarrow E_L$ after spike.

### Mg²⁺ block derivation

The Jahr & Stevens (1990) Mg²⁺ block arises from the open-channel blocking mechanism:
Mg²⁺ ions sit in the channel pore at resting potential, blocking ion flow. Depolarisation
expels the Mg²⁺ by electrostatic force, relieving the block.

The Boltzmann-like form $B(V)$ comes from modelling the Mg²⁺ as a charged blocker
with a single binding site at fractional electrical distance $\delta \approx 0.8$
through the membrane electric field:

$$B(V) = \frac{1}{1 + [\text{Mg}^{2+}] \cdot K_0 \cdot \exp(-z \delta F V / RT)}$$

With $z = 2$ (Mg²⁺ charge), $\delta = 0.8$, $F/RT \approx 0.0389$ mV⁻¹ at 37°C,
and $K_0 = 1/3.57$ mM⁻¹, the exponent becomes:

$$z \delta F / RT \approx 2 \times 0.8 \times 0.0389 \approx 0.062 \text{ mV}^{-1}$$

This is the exact value used in our implementation, confirming the physical derivation.

### Coincidence detection

The NMDA channel operates as an AND gate:

| Glutamate | V_dend | B(V) | I_NMDA | Biological meaning |
|-----------|--------|------|--------|-------------------|
| 0 | -65 mV | 0.06 | 0 | No presynaptic input |
| 1 | -65 mV | 0.06 | small | Pre but no post (blocked) |
| 0 | 0 mV | 0.78 | 0 | Post but no pre |
| 1 | 0 mV | 0.78 | large | **Both pre and post (AND)** |

This coincidence detection is the biophysical basis of Hebbian learning
and is fundamental to long-term potentiation (LTP) in the hippocampus.

### Two-compartment coupling

The soma-dendrite coupling is bidirectional:

$$I_{d \to s} = g_c \cdot (V_d - V_s)$$
$$I_{s \to d} = g_c \cdot (V_s - V_d) = -I_{d \to s}$$

Current conservation is maintained. The coupling conductance $g_c$ controls
how strongly the two compartments are electrically coupled. Higher $g_c$
makes them more isopotential; lower $g_c$ allows larger voltage differences
between soma and dendrite, enabling local dendritic computation.

---

## 2. Theoretical Context

### Problem statement

NMDA receptors play a critical role in synaptic plasticity, learning, and memory.
Their voltage-dependent magnesium block creates a coincidence detection mechanism
that is essential for Hebbian learning. Single-compartment neuron models cannot
capture this dendritic computation because the NMDA block depends on *local*
dendritic voltage, not somatic voltage.

### Historical significance

The Jahr & Stevens (1990) paper established the quantitative form of the Mg²⁺ block
that is now used universally in computational neuroscience. Key contributions:

1. **First quantitative measurement** of the voltage dependence of Mg²⁺ block
   in physiological conditions (1 mM external Mg²⁺, 37°C)
2. **Simple analytical formula** $B(V) = 1/(1 + [Mg]/3.57 \cdot \exp(-0.062V))$
   that fits experimental data with R² > 0.99
3. **Physical interpretation** as single-site open-channel block consistent
   with the Woodhull (1973) blocking model

### Role in synaptic plasticity

The NMDA receptor is the coincidence detector for Hebbian LTP:

1. **Presynaptic activity** releases glutamate → binds to NMDA receptor
2. **Postsynaptic depolarisation** → relieves Mg²⁺ block
3. **Both conditions met** → Ca²⁺ flows through NMDA channel
4. **Ca²⁺ influx** → triggers CaMKII → LTP induction

Our model captures steps 1-3. Step 4 (calcium-dependent plasticity) would
require a separate plasticity rule (e.g., calcium-dependent STDP).

### Relationship to existing models

| Model | Compartments | NMDA | Mg block | Coincidence |
|-------|-------------|------|----------|-------------|
| Standard LIF | 1 (soma) | No | No | No |
| Pinsky-Rinzel | 2 (soma + dend) | Yes | Simplified | Yes |
| Hay L5 | 3+ (detailed) | Yes | Full | Yes |
| **DendriticNMDA** | **2 (soma + dend)** | **Yes** | **Jahr & Stevens exact** | **Yes** |
| Dendrify | N (arbitrary) | Optional | Configurable | Yes |

Our model sits between the simple LIF and detailed multi-compartment models,
providing NMDA coincidence detection with minimal computational cost.

### Applications

1. **Hebbian learning circuits:** Pre-post coincidence detection for STDP
2. **Working memory:** NMDA's slow kinetics (~100ms) enable sustained activity
3. **Dendritic computation:** Local nonlinear integration in dendrites
4. **Calcium signalling:** The NMDA current is a proxy for Ca²⁺ influx
5. **Pharmacological modelling:** Mg²⁺ concentration can model drug effects
   (e.g., memantine blocks NMDA in Alzheimer's treatment)
6. **Disease modelling:** NMDA hypofunction models schizophrenia (Olney & Farber 1995);
   NMDA overactivation models excitotoxicity in stroke and neurodegeneration

### Mg²⁺ block across species and temperatures

The constants 3.57 mM and 0.062 mV⁻¹ are for hippocampal neurons at 37°C.
Variations across preparations:

| Preparation | K₀ (mM) | δ (mV⁻¹) | Source |
|-------------|---------|-----------|--------|
| Rat hippocampus, 37°C | 3.57 | 0.062 | Jahr & Stevens 1990 |
| Rat hippocampus, 24°C | 3.57 | 0.062 | Same (temp-independent) |
| Mouse cortex, 37°C | ~3.6 | ~0.06 | Kampa et al. 2004 |
| NR2A subunit | 3.0 | 0.062 | Monyer et al. 1994 |
| NR2B subunit | 4.5 | 0.060 | Monyer et al. 1994 |

The default values are appropriate for most mammalian cortical/hippocampal modelling.
For NR2B-dominant synapses (prefrontal cortex), consider mg_conc * 1.26 correction.

### NMDA channel properties not modelled

For completeness, properties present in real NMDA channels but not in this model:

| Property | Timescale | Effect | Why omitted |
|----------|-----------|--------|-------------|
| Rise time (~10ms) | Fast | Delayed current onset | Adds ODE, minor effect at dt=0.1 |
| Decay time (~100ms) | Slow | Prolonged current | Would need glutamate kinetics |
| GluN2 subunit composition | — | Different kinetics | Model-specific, not universal |
| Glycine co-agonist | — | Required for activation | Assumed saturating |
| Ca²⁺ permeability | — | Triggers plasticity | Separate Ca²⁺ model needed |
| Desensitisation | Seconds | Reduced current | Rare in physiological conditions |

---

## 3. Pipeline Position

```
Presynaptic neuron                    External input
     │                                      │
     ▼ (glutamate)                          ▼ (i_soma)
┌──────────────────────────────────────────────┐
│              DendriticNMDANeuron              │
│                                              │
│  ┌───────────┐    g_c    ┌──────────┐       │
│  │ Dendrite  │◀────────▶│  Soma    │       │
│  │ V_d       │           │  V_s     │       │
│  │ + NMDA    │           │  + leak  │       │
│  │ + Mg block│           │  + I_ext │       │
│  └───────────┘           └────┬─────┘       │
│                               │              │
│                         V_s ≥ θ → spike     │
└──────────────────────────────────────────────┘
     │
     ▼
Binary spike (0 or 1)
```

### Inputs

| Input | Type | Range | Description |
|-------|------|-------|-------------|
| `i_soma` | `float` | $(-\infty, +\infty)$ | External current to soma |
| `glutamate` | `float` | $[0, 1]$ | Presynaptic glutamate (0=none, 1=saturated) |

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `spike` | `int` | $\{0, 1\}$ | Binary somatic spike |

---

## 4. Features

| Feature | Description |
|---------|-------------|
| **Exact Mg²⁺ block** | Jahr & Stevens (1990) formula with exact constants |
| **Two compartments** | Soma + dendrite with bidirectional coupling |
| **Coincidence detection** | NMDA requires both glutamate AND depolarisation |
| **Configurable Mg²⁺** | `mg_conc` parameter for pharmacological studies |
| **Separate time constants** | tau_soma (fast, 20ms) and tau_dend (slow, 50ms) |
| **Adjustable coupling** | `g_coupling` controls soma-dendrite interaction |
| **Hard spike reset** | Soma resets to E_L on spike |
| **Candidate-first RK4** | Production path integrates `(v_soma, v_dend)` with RK4 before committing state |
| **Baseline comparison** | `integrator="baseline_euler"` preserves the historical dendrite-first Euler path |
| **Polyglot parity** | Python, Rust engine, Rust safety, Go, Julia, and Mojo share the 253-spike anchor |

---

## 5. Usage Examples

### Basic usage with somatic input only

```python
from sc_neurocore.neurons.models import DendriticNMDANeuron

neuron = DendriticNMDANeuron()
spikes = sum(neuron.step(50.0, 0.0) for _ in range(2000))
print(f"Somatic input only: {spikes} spikes in 2000 steps")
```

### Coincidence detection

```python
# Only soma input.
n1 = DendriticNMDANeuron()
for _ in range(500): n1.step(30.0, 0.0)
print(f"Soma only — V_dend={n1.v_dend:.2f}")

# Soma + glutamate.
n2 = DendriticNMDANeuron()
for _ in range(500): n2.step(30.0, 1.0)
print(f"Soma + glut — V_dend={n2.v_dend:.2f}")
```

### Mg²⁺ block curve

```python
import math
n = DendriticNMDANeuron(mg_conc=1.0)
for v in range(-80, 30, 10):
    b = n.mg_block(float(v))
    bar = '█' * int(b * 50)
    print(f"V={v:+4d} mV: B={b:.3f} {bar}")
```

### Pharmacological variation

```python
# Normal Mg²⁺ (1 mM).
n_normal = DendriticNMDANeuron(mg_conc=1.0)
# Low Mg²⁺ (mimics Mg-free ACSF in experiments).
n_low_mg = DendriticNMDANeuron(mg_conc=0.1)
# High Mg²⁺ (extra block).
n_high_mg = DendriticNMDANeuron(mg_conc=3.0)

for name, neuron in [("normal", n_normal), ("low_Mg", n_low_mg), ("high_Mg", n_high_mg)]:
    spikes = sum(neuron.step(40.0, 0.8) for _ in range(2000))
    print(f"{name}: {spikes} spikes")
```

### Dendritic NMDA current monitoring

```python
neuron = DendriticNMDANeuron()
for t in range(200):
    neuron.step(30.0, 1.0)
    if t % 50 == 0:
        b = neuron.mg_block(neuron.v_dend)
        i_nmda = neuron.g_nmda * 1.0 * b * (neuron.v_dend - neuron.e_nmda)
        print(f"t={t}: V_d={neuron.v_dend:.1f}, B={b:.3f}, I_nmda={i_nmda:.2f}")
```

### Coupling strength effect

```python
for g in [0.1, 0.5, 1.0, 2.0, 5.0]:
    n = DendriticNMDANeuron(g_coupling=g)
    for _ in range(500):
        n.step(30.0, 1.0)
    diff = abs(n.v_soma - n.v_dend)
    print(f"g_c={g:.1f}: V_soma-V_dend = {diff:.2f} mV")
```

---

## 6. Technical Reference

### Class: `DendriticNMDANeuron`

Decorated with `@dataclass`. Defined in
`src/sc_neurocore/neurons/models/dendritic_nmda.py`.

#### Constructor Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `g_nmda` | `float` | `1.5` | $\geq 0$ | NMDA conductance |
| `e_nmda` | `float` | `0.0` | Any | NMDA reversal potential (mV) |
| `mg_conc` | `float` | `1.0` | $\geq 0$ | Extracellular Mg²⁺ concentration (mM) |
| `g_coupling` | `float` | `0.5` | $\geq 0$ | Soma-dendrite coupling conductance |
| `tau_soma` | `float` | `20.0` | $> 0$ | Soma time constant (ms) |
| `tau_dend` | `float` | `50.0` | $> 0$ | Dendrite time constant (ms) |
| `theta` | `float` | `-50.0` | Any | Spike threshold (mV) |
| `dt` | `float` | `0.1` | $> 0$ | Integration timestep (ms) |
| `integrator` | `str` | `"rk4"` | `"rk4"` or `"baseline_euler"` | Numerical integration path |

#### State Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `v_soma` | `float` | `-65.0` | Somatic membrane potential (mV) |
| `v_dend` | `float` | `-65.0` | Dendritic membrane potential (mV) |

#### Methods

**`mg_block(v: float) -> float`**

Compute Mg²⁺ block factor B(V) = 1/(1 + [Mg]/3.57 · exp(-0.062·V)).

**`step(i_soma: float, glutamate: float) -> int`**

Advance one timestep. Returns 1 if somatic spike, 0 otherwise.

**`reset() -> None`**

Reset v_soma and v_dend to -65.0 mV.

### Rust implementation parity

| Operation | Python | Rust |
|-----------|--------|------|
| Mg block | `1/(1 + (mg/3.57)*exp(-0.062*v))` | `1.0/(1.0 + (mg/3.57)*(-0.062*v).exp())` |
| I_NMDA | `g*glut*B*(V_d - E)` | `g*glut*b*(v_dend - e_nmda)` |
| dV_d/dt | `(-V_d - 65 + I_nmda + g_c*(V_s-V_d))/tau_d` | identical |
| dV_s/dt | `(-V_s - 65 + I_ext + g_c*(V_d-V_s))/tau_s` | identical |
| Integrator | Candidate-first RK4 over `(v_soma, v_dend)` | identical |
| Spike | `next_v_soma >= theta → v_soma = -65` | identical |

### Edge cases

| Condition | Behaviour |
|-----------|-----------|
| `mg_conc = 0` | No Mg block: B(V) = 1.0 everywhere |
| `glutamate = 0` | No NMDA current, regardless of B(V) |
| `g_coupling = 0` | Soma and dendrite fully decoupled |
| `V at -100mV` | B(-100) ≈ 0.005 — nearly complete block |
| `V at +40mV` | B(40) ≈ 0.97 — nearly no block |

---

## 7. Performance Benchmarks

Measured local regression results from
`benchmarks/results/local_python_2026-06-26_dendritic_nmda_rk4.json`.
The run used 20,000 steps, five repeats, `i_soma=50.0`, `glutamate=0.5`,
and the expected 253-spike anchor. The evidence class is
`local_regression_non_isolated`; it verifies parity and regression timing on the
local workstation, not isolated CPU-core benchmark performance.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes | Command |
|---------|----------------|-------------|-------------|--------|---------|
| Python | 3602.653 | 3565.111 | 3756.433 | 253 | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_dendritic_nmda.py` |
| Rust engine | 101.743 | 100.656 | 106.196 | 253 | `cargo run --release --manifest-path engine/Cargo.toml --example bench_dendritic_nmda_rk4` |
| Go service | 188.400 | 180.800 | 196.500 | 253 | `go test ... -bench BenchmarkDendriticNMDARK4` |
| Julia mirror | 127.686 | 122.129 | 128.417 | 253 | `julia --project=. -e <dendritic_nmda rk4 benchmark>` |
| Mojo kernel | Smoke only | Smoke only | Smoke only | 253 | `mojo run --disable-warnings src/sc_neurocore/accel/mojo/kernels/dendritic_nmda.mojo` |

### Benchmark artefacts

| Surface | Path |
|---------|------|
| Python benchmark driver | `benchmarks/bench_model_dendritic_nmda.py` |
| JSON result | `benchmarks/results/local_python_2026-06-26_dendritic_nmda_rk4.json` |
| Rust benchmark example | `engine/examples/bench_dendritic_nmda_rk4.rs` |
| Go benchmark | `src/sc_neurocore/accel/go/services/dendritic_nmda_test.go` |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/dendritic_nmda.jl` |
| Mojo smoke kernel | `src/sc_neurocore/accel/mojo/kernels/dendritic_nmda.mojo` |

### Memory

| Implementation | Per-neuron |
|---------------|------------|
| Python | ~200 bytes |
| Rust | 80 bytes (10× f64) |

---

## 8. Citations

1. **Jahr, C. E. & Stevens, C. F.** "A quantitative description of NMDA
   receptor-channel kinetic behavior." Journal of Neuroscience 10(6):1830-1837, 1990.
   — Source of Mg²⁺ block formula: B(V) = 1/(1 + [Mg]/3.57·exp(-0.062V)).

2. **Schiller, J. et al.** "NMDA spikes in basal dendrites of cortical
   pyramidal neurons." Nature 404:285-289, 2000.
   — Dendritic NMDA spike phenomenon justifying two-compartment model.

3. **Woodhull, A. M.** "Ionic blockage of sodium channels in nerve."
   Journal of General Physiology 61(6):687-708, 1973.
   — Original open-channel block model underlying the Mg²⁺ block formula.

4. **Pinsky, P. F. & Rinzel, J.** "Intrinsic and network rhythmogenesis
   in a reduced Traub model for CA3 neurons." Journal of Computational
   Neuroscience 1:39-60, 1994.
   — Two-compartment model with NMDA, predecessor to our simplified version.

5. **Mainen, Z. F. & Sejnowski, T. J.** "Influence of dendritic structure
   on firing pattern in model neocortical neurons." Nature 382:363-366, 1996.
   — Demonstration that dendritic morphology shapes spike output.

6. **Larkum, M. E. et al.** "A new cellular mechanism for coupling inputs
   arriving at different cortical layers." Nature 398:338-341, 1999.
   — BAC firing: dendritic Ca²⁺ spikes coupled to somatic Na⁺ spikes.

---

## Validation

### Test suite results

| Test | What it verifies | Status |
|------|-----------------|--------|
| `tests/test_model_dendritic_nmda.py` | Default RK4 path, Euler comparison, public import wiring, invalid-input preservation | PASS |
| `tests/test_gap_models.py::TestDendriticNMDANeuron` | Legacy public behavior checks for Mg²⁺ block, spiking, coincidence detection, reset | PASS |
| `cargo test --manifest-path engine/Cargo.toml --lib nmda_` | Rust engine NMDA unit anchor and related NMDA library checks | PASS |
| `go test src/sc_neurocore/accel/go/services/dendritic_nmda.go src/sc_neurocore/accel/go/services/dendritic_nmda_test.go` | Go mirror anchor and invalid-state preservation | PASS |
| `rustc --test src/sc_neurocore/accel/rust/safety/dendritic_nmda.rs` | Standalone Rust safety mirror anchor and invalid-state preservation | PASS |
| `julia --project=. -e <dendritic_nmda anchor>` | Julia mirror 253-spike anchor | PASS |
| `mojo run --disable-warnings src/sc_neurocore/accel/mojo/kernels/dendritic_nmda.mojo` | Mojo smoke anchor | PASS |

### Equation-to-code traceability

| Equation | Python location | Rust location |
|----------|----------------|---------------|
| $B(V) = 1/(1 + [Mg]/3.57 \cdot e^{-0.062V})$ | `_mg_block_value` | `mg_block` |
| $I_{NMDA} = g \cdot \text{glut} \cdot B \cdot (V_d - E)$ | `_derivatives` | `derivatives` |
| $\tau_d \, dV_d/dt$ | `_derivatives` | `derivatives` |
| $\tau_s \, dV_s/dt$ | `_derivatives` | `derivatives` |
| RK4 candidate | `_rk4_substep` | `rk4_substep` |

---

## Design Decisions

### Why separate time constants for soma and dendrite?

The soma ($\tau_s = 20$ ms) has faster dynamics than the dendrite ($\tau_d = 50$ ms),
reflecting the biophysical reality that dendrites have higher membrane resistance
and capacitance per unit area. This separation allows the dendrite to integrate
NMDA-mediated inputs over a longer timescale (~50ms) while the soma responds
rapidly to dendritic current and generates fast action potentials.

### Why hard-coded E_L = -65 mV in the ODE?

The leak reversal is embedded in the ODE as `(-V - 65.0)` rather than
`(-(V - E_L))` because it matches the Rust implementation exactly and
avoids adding another configurable parameter for a value that is nearly
universal across cortical neuron models. To change E_L, subclass and override.

### Why not include AMPA channels?

AMPA channels provide fast excitation (decay ~2ms) complementary to NMDA's
slow excitation (decay ~100ms). We omit AMPA because:
1. The model focuses on NMDA coincidence detection specifically
2. The `i_soma` input can represent AMPA-like fast excitation externally
3. Adding AMPA would double the number of parameters without adding new qualitative behaviour

---

## Implementation Notes

### RK4 integration and Euler comparison

The production path integrates the coupled soma-dendrite state with a
candidate-first fourth-order Runge-Kutta step. The model validates finite
configuration, finite state, finite somatic current, and finite non-negative
glutamate before computing any candidate. If the candidate leaves the finite
domain, Python raises and the non-throwing low-level mirrors return no spike
without committing a partial state.

`integrator="baseline_euler"` keeps the historical dendrite-first forward Euler
update for regression comparisons. The default path remains `"rk4"` in Python,
Rust engine, Rust safety, Go, Julia, and Mojo.

### Leak reversal hard-coded at -65 mV

Both compartments use $E_L = -65$ mV hard-coded in the ODE:
`-self.v_soma - 65.0` rather than `-(self.v_soma - self.e_l)`.

This matches the Rust implementation exactly and simplifies the parameter space.
The value -65 mV is the standard resting potential for cortical pyramidal neurons.
For other cell types (e.g., interneurons at -55 mV), modify the source directly
or use a different neuron model.

### Glutamate as a normalised variable

The `glutamate` input is treated as a dimensionless concentration in [0, 1], where:
- 0 = no glutamate release
- 1 = maximal (saturating) glutamate concentration

This avoids modelling vesicle release dynamics, re-uptake, and diffusion. For
realistic glutamate transients, preprocess the input through a synapse model
(e.g., an alpha function: $g(t) = t/\tau \cdot \exp(1 - t/\tau)$).

---

## Known Limitations

1. **No NMDA kinetics:** The model uses instantaneous Mg²⁺ block without
   NMDA rise/decay time constants (~10ms rise, ~100ms decay). For temporal
   NMDA dynamics, use the Destexhe (1994) NMDA synapse model.

2. **No calcium tracking:** The NMDA current is computed but calcium influx
   is not tracked. For calcium-dependent plasticity, add explicit Ca²⁺ dynamics.

3. **No dendritic spikes:** The model does not support regenerative dendritic
   events (NMDA spikes, Ca²⁺ spikes). Only somatic spikes are generated.

4. **Linear coupling:** The soma-dendrite coupling is ohmic (linear). Real
   dendrites have voltage-dependent conductances that make coupling nonlinear.

5. **No spatial extent:** The dendrite is a single electrical compartment.
   For spatially extended dendrites, use multi-compartment models (Hay L5).

6. **No voltage-gated channels:** Neither compartment has active conductances
   (Na⁺, K⁺, Ca²⁺). Somatic spikes are generated by threshold crossing, not
   by Hodgkin-Huxley dynamics. For biophysically realistic spike shapes, use
   the HodgkinHuxleyNeuron or HayL5PyramidalNeuron models.

7. **Single dendrite:** Real pyramidal neurons have basal, apical oblique, and
   apical tuft dendrites with distinct NMDA properties. Use MulticompartmentMCNNeuron
   for dual-dendrite (basal + apical) models.

---

*SC-NeuroCore v3.14.0 — Stochastic Computing Spiking Neural Network Framework*
*© 2020–2026 Miroslav Šotek. AGPL-3.0-or-later.*
