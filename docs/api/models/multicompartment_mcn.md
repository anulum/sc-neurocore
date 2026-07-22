# MulticompartmentMCNNeuron

**Module:** `sc_neurocore.neurons.models.multicompartment_mcn`
**Rust path:** `sc_neurocore_engine::neurons::multi_compartment::MulticompartmentMCNNeuron`
**Reference:** Brain-Cog-Lab, arXiv:2503.00713, PNAS 2025
**Family:** Multi-compartment spiking neurons for working memory
**State variables:** `u` (soma), `v_basal` (basal dendrite), `v_apical` (apical dendrite)

---

## 1. Mathematical Formalism

### Core equations (arXiv:2503.00713, Spiking-WM, PNAS 2025)

Dual-dendrite model where the apical dendrite **gates** how strongly basal information
influences the soma, enabling nonlinear integration for long-term temporal memory.

**Basal dendrite:**

$$\tau_b \frac{dV_b}{dt} = -V_b + x_b$$

Simple leaky integrator receiving bottom-up sensory input $x_b$.

**Apical dendrite:**

$$\tau_a \frac{dV_a}{dt} = -V_a + x_a$$

Simple leaky integrator receiving top-down contextual input $x_a$.

**Soma (σ-gated):**

$$\tau \frac{dU}{dt} = -U + \sigma(V_a) \cdot \left[\frac{g_B}{g_L} (V_b - U) + W_s \cdot I\right]$$

where $\sigma(x) = 1/(1 + \exp(-\beta x))$ is the sigmoid gating function with
steepness $\beta$, $g_B/g_L$ is the basal-to-soma conductance ratio, and $I$ is
direct somatic input.

The key innovation is the **σ-gating**: the apical dendrite controls a sigmoid
gate that modulates the basal-to-soma current. When $V_a$ is large (strong
top-down input), $\sigma(V_a) \approx 1$ and basal input passes freely to the soma.
When $V_a$ is small or negative, $\sigma(V_a) \approx 0$ and basal input is suppressed.

**Spike generation:**

$$S[t] = \Theta(U[t] - V_{th})$$

where $\Theta$ is the Heaviside step function.

**Soft reset:**

$$U[t] \leftarrow U[t] \cdot (1 - S[t])$$

When $S = 1$ (spike), $U$ is reset to 0. When $S = 0$ (no spike), $U$ is unchanged.
This is equivalent to $U \leftarrow 0$ on spike, but the multiplicative form
has better gradient properties for surrogate gradient training.

### Default parameters (Table II of arXiv:2503.00713)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `tau` | $\tau$ | 2.0 | Soma time constant |
| `tau_b` | $\tau_b$ | 2.0 | Basal dendrite time constant |
| `tau_a` | $\tau_a$ | 2.0 | Apical dendrite time constant |
| `g_ratio` | $g_B/g_L$ | 1.0 | Basal-to-soma conductance ratio |
| `beta` | $\beta$ | 1.0 | Sigmoid steepness |
| `v_th` | $V_{th}$ | 1.0 | Spike threshold |
| `dt` | $dt$ | 1.0 | Integration timestep |

### Sigmoid gating analysis

The sigmoid gate $\sigma(V_a) = 1/(1 + \exp(-\beta V_a))$ has the following properties:

| $V_a$ | $\sigma(V_a)$ with $\beta = 1$ | Effect |
|-------|-------------------------------|--------|
| $-5$ | 0.0067 | Basal input nearly blocked |
| $-2$ | 0.119 | Basal input strongly attenuated |
| $0$ | 0.5 | 50% of basal input passes |
| $+2$ | 0.881 | Most basal input passes |
| $+5$ | 0.993 | Nearly full basal-soma coupling |

With $\beta > 1$, the gate becomes sharper (more binary-like).
With $\beta < 1$, the gate becomes softer (more graded).

### Steady-state analysis

For constant inputs $(x_b, x_a, I)$, the steady states are:

$$V_b^* = x_b, \quad V_a^* = x_a$$

$$U^* = \sigma(x_a) \cdot \left[\frac{g_B}{g_L} \cdot (x_b - U^*) + I\right]$$

Solving for $U^*$:

$$U^* = \frac{\sigma(x_a) \cdot (g_B/g_L \cdot x_b + I)}{1 + \sigma(x_a) \cdot g_B/g_L}$$

The neuron fires when $U^* \geq V_{th}$, i.e., when the gated combination of
basal input and somatic current exceeds threshold.

### Energy interpretation

The gating mechanism can be interpreted through an energy framework.
The soma potential $U$ follows a gradient descent on the energy:

$$E(U) = \frac{1}{2}U^2 - \sigma(V_a) \cdot \left[g_{B/L} \cdot V_b \cdot U + I \cdot U - \frac{g_{B/L}}{2} U^2\right]$$

The minimum of $E$ corresponds to the steady-state $U^*$. The sigmoid gate
$\sigma(V_a)$ controls the depth of the energy well: when the gate is open,
the well is deeper (stronger attractor), enabling sustained activity.

This energy interpretation connects to attractor network models of working
memory (Hopfield 1982, Amit & Brunel 1997), where sustained activity corresponds
to a stable fixed point in a high-dimensional energy landscape.

### Discretisation (candidate-first RK4)

The production implementation advances the three-state vector
$(U, V_b, V_a)$ with classical RK4. Basal, apical, and direct somatic drives are
held constant across the four stages, and every stage evaluates the soma gate
from the stage-local apical voltage. The candidate state is checked for finite
values before commit; the soma is then reset to 0 if the candidate crosses
$V_{th}$.

For regression comparison only, Python accepts
`integrator="baseline_euler"`, which evaluates the same coupled right-hand side
with one explicit Euler increment. The default and all production backend
surfaces use `integrator="rk4"`.

---

## 2. Theoretical Context

### Problem statement

Standard spiking neurons (LIF, Izhikevich) have a single compartment and cannot
distinguish between bottom-up (sensory) and top-down (contextual) inputs. For
tasks requiring long-term temporal memory (e.g., reinforcement learning with
delayed rewards), the network needs a mechanism to gate information flow based
on context.

### The Spiking-WM solution

Spiking-WM (arXiv:2503.00713, published in PNAS 2025) introduces a multi-compartment
neuron where:

1. **Basal dendrite** receives sensory/bottom-up input
2. **Apical dendrite** receives contextual/top-down input
3. **Apical gate** controls how much basal information reaches the soma
4. **Soma** integrates gated input and generates spikes

This architecture enables:
- **Selective attention:** Top-down signals gate relevant bottom-up information
- **Working memory:** Sustained apical input maintains the gate open, allowing
  continuous integration of basal input over long timescales
- **Temporal credit assignment:** The gate can learn which temporal contexts
  are relevant for reward prediction

### Biological motivation

The dual-dendrite architecture is inspired by cortical pyramidal neurons, which have:

- **Basal dendrites** (close to soma) receiving local sensory input
- **Apical tuft dendrites** (in layer 1) receiving long-range feedback from
  higher cortical areas

The apical-basal interaction is mediated by active conductances (Ca²⁺ channels)
that create a nonlinear gate. Our sigmoid function $\sigma(V_a)$ is a simplified
model of this BAC (backpropagation-activated calcium) firing mechanism
(Larkum et al. 1999).

### Three-compartment vs two-compartment

The MCN has three compartments (soma + basal + apical) while simpler models like
DendriticNMDANeuron have two (soma + dendrite). The third compartment (apical) adds:

1. **Separate top-down channel:** Context and sensory inputs are processed by
   independent integrators before combining at the soma
2. **Nonlinear gate:** The sigmoid gate creates a multiplicative interaction
   between basal and apical, enabling XOR-like computation impossible with
   linear summation
3. **Credit assignment:** During training, gradients for sensory weights
   (through basal) and context weights (through apical) are separable,
   simplifying optimisation

The cost is one additional ODE per neuron — negligible compared to the
computational benefit.

### Spiking-WM results

The paper reports that MCN neurons outperform standard LIF on:

| Task | LIF accuracy | MCN accuracy | Improvement |
|------|-------------|-------------|-------------|
| T-Maze (memory) | 52% | 95% | +43% |
| Delayed reward RL | 65% | 92% | +27% |
| Sequential MNIST | 91% | 97% | +6% |

The improvement is largest on tasks requiring long-term temporal memory,
where the apical gate provides the mechanism for credit assignment.

### Relationship to existing models

| Model | Compartments | Gating | Working memory | Reference |
|-------|-------------|--------|----------------|-----------|
| Standard LIF | 1 | None | No | Lapicque 1907 |
| Pinsky-Rinzel | 2 (soma + dend) | Passive | Limited | Pinsky & Rinzel 1994 |
| Hay L5 | 3+ (detailed) | Ca²⁺ spikes | Yes (slow) | Hay et al. 2011 |
| **MCN** | **3 (soma + basal + apical)** | **σ-gate** | **Yes (explicit)** | **arXiv:2503.00713** |
| DendriticNMDA | 2 (soma + dend) | Mg²⁺ block | Limited | Jahr & Stevens 1990 |

---

## 3. Pipeline Position

```
Bottom-up input (x_b)          Top-down context (x_a)
     │                                │
     ▼                                ▼
┌──────────┐                   ┌──────────┐
│  Basal   │                   │  Apical  │
│  dendrite│                   │  dendrite│
│  V_b     │                   │  V_a     │
└────┬─────┘                   └────┬─────┘
     │                              │
     │    ┌─────────────────────┐   │
     │    │    Soma             │   │
     └───▶│    U                │◀──┘ (σ-gate)
          │    + direct I       │
          │    spike: Θ(U-V_th) │
          └─────────┬───────────┘
                    │
                    ▼
            Binary spike (0 or 1)
```

### Inputs

| Input | Type | Range | Description |
|-------|------|-------|-------------|
| `x_basal` | `float` | $(-\infty, +\infty)$ | Bottom-up / sensory input |
| `x_apical` | `float` | $(-\infty, +\infty)$ | Top-down / contextual input |
| `i_soma` | `float` | $(-\infty, +\infty)$ | Direct somatic current |

For the simple `step(current)` interface: `x_basal = current`, `x_apical = 0`, `i_soma = 0`.

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `spike` | `int` | $\{0, 1\}$ | Binary somatic spike |

---

## 4. Features

| Feature | Description |
|---------|-------------|
| **Dual dendrites** | Separate basal (sensory) and apical (contextual) inputs |
| **Sigmoid gate** | Apical controls basal-to-soma current via σ(V_a) |
| **Configurable steepness** | β parameter controls gate sharpness |
| **Soft reset** | U ← U·(1-S), preserves sub-threshold dynamics |
| **Table II defaults** | All parameters match PNAS 2025 publication |
| **Three time constants** | Independent τ for soma, basal, apical |
| **Direct soma input** | Bypass dendrites via i_soma parameter |
| **Simple API** | `step(current)` for single-input use |
| **Full API** | `step_compartments(x_b, x_a, I)` for triple-input use |
| **Candidate-first RK4** | Python, Rust, Julia, Go, and Mojo advance the same coupled three-state RHS |
| **Fail-closed validation** | Non-finite inputs, parameters, states, and candidates are rejected before state commit |
| **Baseline comparison path** | Python keeps `integrator="baseline_euler"` for explicit regression comparisons |

---

## 5. Usage Examples

### Basic basal-only input

```python
from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

neuron = MulticompartmentMCNNeuron()
spikes = sum(neuron.step(3.2) for _ in range(100))
print(f"Basal-only: {spikes} spikes in 100 steps")
# Note: with step(), x_apical=0 → σ(0)=0.5 → half-gated
```

### Apical gating demonstration

```python
# No apical input: gate = σ(0) = 0.5.
n1 = MulticompartmentMCNNeuron()
s1 = sum(n1.step_compartments(2.5, 0.0, 0.0) for _ in range(200))

# Strong apical: gate → 1.0, full basal-soma coupling.
n2 = MulticompartmentMCNNeuron()
s2 = sum(n2.step_compartments(2.5, 5.0, 0.0) for _ in range(200))

# Inhibitory apical: gate → 0.0, basal blocked.
n3 = MulticompartmentMCNNeuron()
s3 = sum(n3.step_compartments(2.5, -5.0, 0.0) for _ in range(200))

print(f"No apical: {s1} spikes")
print(f"Strong apical: {s2} spikes")
print(f"Inhibitory apical: {s3} spikes")
```

### Working memory task

```python
# Phase 1: store stimulus (x_b=3, x_a=2 to open gate).
n = MulticompartmentMCNNeuron()
for _ in range(50):
    n.step_compartments(3.0, 2.0, 0.0)
print(f"After store: U={n.u:.3f}, V_b={n.v_basal:.3f}")

# Phase 2: delay period (no input, gate closed).
delay_spikes = sum(n.step_compartments(0.0, -2.0, 0.0) for _ in range(100))
print(f"During delay: {delay_spikes} spikes, U={n.u:.3f}")

# Phase 3: recall (re-open gate with apical).
recall_spikes = sum(n.step_compartments(0.0, 3.0, 0.0) for _ in range(50))
print(f"Recall: {recall_spikes} spikes")
```

### Threshold sensitivity

```python
for v_th in [0.5, 1.0, 2.0, 5.0]:
    n = MulticompartmentMCNNeuron(v_th=v_th)
    spikes = sum(n.step_compartments(2.0, 1.0, 0.0) for _ in range(200))
    print(f"V_th={v_th:.1f}: {spikes} spikes/200 steps")
```

### Conductance ratio sweep

```python
for g in [0.0, 0.5, 1.0, 2.0, 5.0]:
    n = MulticompartmentMCNNeuron(g_ratio=g)
    spikes = sum(n.step_compartments(2.0, 2.0, 0.0) for _ in range(200))
    print(f"g_B/g_L={g:.1f}: {spikes} spikes (higher g → more basal influence)")
```

### Beta steepness sweep

```python
for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
    n = MulticompartmentMCNNeuron(beta=beta)
    # Check gate at V_a = 0.5.
    gate = n._sigma(0.5)
    print(f"beta={beta:5.1f}: σ(0.5)={gate:.4f}")
```

---

## 6. Technical Reference

### Class: `MulticompartmentMCNNeuron`

Decorated with `@dataclass`. Defined in
`src/sc_neurocore/neurons/models/multicompartment_mcn.py`.

#### Constructor Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `tau` | `float` | `2.0` | $> 0$ | Soma time constant |
| `tau_b` | `float` | `2.0` | $> 0$ | Basal dendrite time constant |
| `tau_a` | `float` | `2.0` | $> 0$ | Apical dendrite time constant |
| `g_ratio` | `float` | `1.0` | $\geq 0$ | Basal-to-soma conductance ratio ($g_B/g_L$) |
| `beta` | `float` | `1.0` | $> 0$ | Sigmoid gate steepness |
| `v_th` | `float` | `1.0` | $> 0$ | Spike threshold |
| `dt` | `float` | `1.0` | $> 0$ | Integration timestep |
| `integrator` | `"rk4"` or `"baseline_euler"` | `"rk4"` | supported literal | Production RK4 or explicit Euler regression path |

#### State Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `u` | `float` | `0.0` | Somatic membrane potential |
| `v_basal` | `float` | `0.0` | Basal dendrite potential |
| `v_apical` | `float` | `0.0` | Apical dendrite potential |

#### Methods

**`step_compartments(x_basal: float, x_apical: float, i_soma: float) -> int`**

Full step with three inputs. Returns 1 if spike, 0 otherwise.

**`step(current: float) -> int`**

Simple step: x_basal = current, x_apical = 0, i_soma = 0.

**`reset() -> None`**

Reset u, v_basal, v_apical to 0.0.

**`_sigma(x: float) -> float`**

Sigmoid gate: 1/(1 + exp(-β·x)).

### Polyglot implementation parity

| Surface | Path | Role |
|---------|------|------|
| Python reference | `src/sc_neurocore/neurons/models/multicompartment_mcn.py` | Public model, validation, `baseline_euler` comparison mode |
| Rust engine | `engine/src/neurons/multi_compartment/multicompartment_mcn.rs` | Compiled production backend and Rust benchmark path |
| Rust safety mirror | `src/sc_neurocore/accel/rust/safety/multicompartment_mcn.rs` | Standalone safety-surface parity check |
| Go service | `src/sc_neurocore/accel/go/services/multicompartment_mcn.go` | Native Go RK4 service and benchmark hook |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/multicompartment_mcn.jl` | Julia RK4 parity mirror |
| Mojo kernel | `src/sc_neurocore/accel/mojo/kernels/multicompartment_mcn.mojo` | SIMD-shaped RK4 kernel |

All five production language surfaces use the same `(U, V_b, V_a)` derivative
order, the same `1/(1+exp(-beta*x))` gate, the same threshold-reset rule, and the
same `49,999` spike anchor at `200,000` steps with basal current `3.2`.

### Edge cases

| Condition | Behaviour |
|-----------|-----------|
| `x_apical = 0` always | Gate = σ(0) = 0.5 — half coupling |
| `x_apical >> 0` always | Gate ≈ 1.0 — full coupling (equivalent to 2-compartment) |
| `x_apical << 0` always | Gate ≈ 0.0 — decoupled (soma only receives i_soma) |
| `g_ratio = 0` | No basal-to-soma current even with gate open |
| `beta = 0` | rejected; use a small positive value for a nearly flat gate |
| `dt > tau` | accepted if finite/positive; RK4 is more stable than Euler, but large steps still alter dynamics |
| non-finite input/state/candidate | Python raises before mutation; Go/Rust safety/Rust engine return no spike and preserve state |

---

## 7. Performance Benchmarks

### Five-backend local regression benchmark

Command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_model_multicompartment_mcn.py
```

Artefact:
`benchmarks/results/local_python_2026-06-26_multicompartment_mcn_rk4.json`.

This is a local non-isolated workstation run for regression context only, not a
published throughput claim. The benchmark fails closed unless Python, Rust, Go,
Julia, and Mojo all report the same spike count.

Measured local regression results from
`benchmarks/results/local_python_2026-06-26_multicompartment_mcn_rk4.json`:

| Backend | Median ns/step | Min ns/step | Max ns/step | Spike anchor |
|---------|---------------:|------------:|------------:|-------------:|
| Python | 12,644.231 | 11,561.743 | 14,075.619 | 49,999 |
| Rust engine | 54.171 | 52.303 | 57.929 | 49,999 |
| Go | 107.500 | 106.400 | 117.000 | 49,999 |
| Julia | 77.278 | 76.022 | 80.008 | 49,999 |
| Mojo | 124.208 | 123.075 | 128.090 | 49,999 |

### Memory

| Implementation | Per-neuron |
|---------------|------------|
| Python | ~200 bytes |
| Rust | 80 bytes (10× f64) |

---

## 8. Citations

1. **Spiking-WM.** Brain-Cog-Lab. "Spiking Working Memory for Temporal Credit
   Assignment in Reinforcement Learning." arXiv:2503.00713, PNAS 2025.
   — Source of all equations, Table II parameters, and performance results.

2. **BAC firing.** Larkum, M. E. et al. "A new cellular mechanism for coupling
   inputs arriving at different cortical layers." Nature 398:338-341, 1999.
   — Biological basis for apical-basal gating in pyramidal neurons.

3. **Dendritic computation.** Poirazi, P. et al. "Pyramidal neuron as a
   two-layer neural network." Neuron 37(6):989-999, 2003.
   — Dendrites as computational subunits (motivates multi-compartment models).

4. **Surrogate gradients.** Neftci, E. O. et al. "Surrogate Gradient Learning
   in Spiking Neural Networks." IEEE Signal Processing Magazine 36(6), 2019.
   — Training framework compatible with the soft reset mechanism.

5. **Temporal credit assignment.** Bellec, G. et al. "A solution to the learning
   dilemma for recurrent networks of spiking neurons." Nature Communications
   11(1):3625, 2020.
   — E-prop: eligibility traces for temporal credit, related to MCN's gate mechanism.

6. **Predictive processing.** Rao, R. P. & Ballard, D. H. "Predictive coding
   in the visual cortex: a functional interpretation of some extra-classical
   receptive-field effects." Nature Neuroscience 2(1):79-87, 1999.
   — Top-down predictions via apical dendrites, consistent with MCN architecture.

7. **Attractor networks.** Amit, D. J. & Brunel, N. "Model of global spontaneous
   activity and local structured activity during delay periods in the cerebral
   cortex." Cerebral Cortex 7(3):237-252, 1997.
   — Attractor dynamics for working memory, related to the energy interpretation.

8. **Dendritic gating.** Gidon, A. et al. "Dendritic action potentials and
   computation in human layer 2/3 cortical neurons." Science 367(6473):83-87, 2020.
   — Experimental evidence for dendritic gating mechanisms in human cortex.

---

## Validation

### Test suite results

| Test | What it verifies | Status |
|------|-----------------|--------|
| `test_defaults_match_table_ii` | tau=tau_b=tau_a=2.0, g_ratio=1.0, beta=1.0, v_th=1.0 | PASS |
| `test_step_returns_binary` | Output in {0, 1} | PASS |
| `test_sigma_gating` | σ(0)=0.5, σ(10)>0.99, σ(-10)<0.01 | PASS |
| `test_default_integrator_is_rk4` | RK4 is the default production path | PASS |
| `test_unknown_integrator_rejected` | unsupported integrator literals fail closed | PASS |
| `test_rk4_and_baseline_euler_paths_diverge` | comparison path remains explicit and observable | PASS |
| `test_cross_backend_spike_anchor` | `49,999` spikes at 200k steps / current 3.2 | PASS |
| `test_non_finite_current_rejected_without_mutation` | invalid input is rejected before state commit | PASS |
| `test_non_finite_runtime_state_rejected_before_mutation` | corrupted runtime state is rejected before mutation | PASS |
| `test_apical_drive_increases_firing_anchor` | strong apical drive increases firing at basal drive 2.5 | PASS |
| `test_soft_reset_to_zero` | U=0 after spike | PASS |
| `test_step_compartments_api` | 3-arg API works | PASS |
| `test_reset` | u=v_basal=v_apical=0 | PASS |

### Equation-to-code traceability

| Equation | Python location | Rust location |
|----------|----------------|---------------|
| $\tau_b \, dV_b/dt = -V_b + x_b$ | `multicompartment_mcn.py:202` | `multi_compartment/multicompartment_mcn.rs:107` |
| $\tau_a \, dV_a/dt = -V_a + x_a$ | `multicompartment_mcn.py:203` | `multi_compartment/multicompartment_mcn.rs:108` |
| $\sigma(V_a) = 1/(1+e^{-\beta V_a})$ | `multicompartment_mcn.py:157-171` | `multi_compartment/multicompartment_mcn.rs:72-74` |
| $\tau \, dU/dt = -U + \sigma \cdot [g(V_b-U)+I]$ | `multicompartment_mcn.py:200-201` | `multi_compartment/multicompartment_mcn.rs:106` |
| RK4 candidate | `multicompartment_mcn.py:209-254` | `multi_compartment/multicompartment_mcn.rs:112-144` |
| $S = \Theta(U - V_{th})$, reset $U \leftarrow 0$ | `multicompartment_mcn.py:333-339` | `multi_compartment/multicompartment_mcn.rs:165-169` |

---

## Design Decisions

### Why unitless potentials (default 0, not -65 mV)?

Unlike biophysical models (HodgkinHuxley, DendriticNMDA), the MCN uses abstract
unitless potentials starting at 0. This follows the machine learning convention
where activations are normalised to [0, 1] or [-1, 1] ranges. The model is designed
for integration into deep spiking networks, not biophysical simulation.

### Why equal time constants (τ = τ_b = τ_a = 2.0)?

Table II of arXiv:2503.00713 uses equal time constants for all compartments.
The paper explores different values but finds equal τ = 2.0 optimal for the
tasks tested. Different values can be set via constructor parameters for
application-specific tuning.

### Why soft reset (U·(1-S)) instead of hard reset?

The multiplicative form U·(1-S) is mathematically equivalent to "U = 0 on spike"
but has a well-defined derivative: $dU'/dU = 1 - S$, which is either 1 (no spike)
or 0 (spike). This makes the reset compatible with the straight-through estimator
for surrogate gradient training. Hard reset creates a discontinuity that requires
special handling during backpropagation.

---

## Known Limitations

1. **No dendritic spikes:** The dendrites are passive leaky integrators without
   active conductances. Real apical dendrites can generate Ca²⁺ spikes.

2. **Fixed gate function:** The sigmoid gate cannot be learned end-to-end in
   the current implementation. The paper learns the full network weights but
   keeps the gate function fixed.

3. **No lateral inhibition:** The model has no mechanism for cross-neuron
   competition. Lateral inhibition must be implemented at the network level.

4. **No adaptation:** There is no spike-frequency adaptation (w variable).
   For tasks requiring adaptation, combine with the SFA mechanism.

5. **Unitless:** The potentials are abstract quantities, not millivolts.
   This limits biophysical interpretability but simplifies ML integration.

6. **No heterogeneity:** All parameters are shared across the neuron. For
   heterogeneous populations, create neurons with different parameter sets.

7. **Fixed-step RK4:** No adaptive timestepping. For stiff dynamics
   (very different τ values), reduce `dt` and re-run the parity benchmark.

---

*SC-NeuroCore v3.16.0 — Stochastic Computing Spiking Neural Network Framework*

*© 2020–2026 Miroslav Šotek. AGPL-3.0-or-later.*
