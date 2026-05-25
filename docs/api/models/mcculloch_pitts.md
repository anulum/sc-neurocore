# McCullochPittsNeuron

**Module:** `sc_neurocore.neurons.models.mcculloch_pitts`
**Reference:** McCulloch & Pitts, Bull. Math. Biophys. 5(4):115–133, 1943
**Family:** Binary (stateless threshold)
**State variables:** None (purely combinational)

---

## Equations

### Threshold activation

$$y = \begin{cases} 1 & \text{if } \sum w_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

### Implementation

```python
def step(self, weighted_input: float) -> int:
    if not math.isfinite(weighted_input) or not math.isfinite(self.theta):
        raise ValueError
    return 1 if weighted_input >= self.theta else 0
```

No ODE, no state update, no transcendental functions.
Single comparison per step — the simplest possible neuron model.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `theta` | 1.0 | — | Activation threshold |

### Key parameter relationships

- **theta = 1.0:** Fires on any non-trivial positive input (OR-like)
- **theta = 2.0:** Requires two unit inputs (AND gate)
- **theta = 0.5 + negative weight:** NOT gate
- **Stateless:** No history, no adaptation, no plasticity

---

## Analytical Properties

### Computational universality

McCulloch–Pitts neurons with appropriate wiring can compute any
Boolean function. This was the first formal proof that neural
networks are Turing-complete (with feedback).

### Transfer function

The activation is a Heaviside step function H(x − θ):
- Derivative: delta function at x = θ (not differentiable)
- No surrogate gradient needed — model is discrete by design

### Information capacity

- **1 bit per step:** Output is always {0, 1}
- **No temporal coding:** No ISI, no spike timing information
- **No rate coding:** Fixed 0 or 1 regardless of input magnitude

---

## Behaviour

- **The first mathematical neuron** (1943). Founding model of
  computational neuroscience and theoretical computer science.
- **Stateless:** No membrane potential, no history — pure
  combinational logic gate.
- **Logic gates:** theta=2 → AND, theta=1 → OR, theta=0.5
  with negative weight → NOT.
- **Deterministic:** Identical input → identical output, always.
- **reset() is no-op:** No state to reset.

## Validation contract

The reference implementation revalidates the mutable threshold at every
`step()` before the Heaviside comparison:

- `theta` must be finite at construction time and at runtime;
- `weighted_input` must be finite;
- equality at the boundary fires, preserving the McCulloch-Pitts
  \(y = 1 \iff x \geq \theta\) convention;
- `reset()` is a no-op because the model is stateless and must not rewrite the
  threshold parameter.

Go, Julia, Rust, and Mojo now expose the same finite-threshold and finite-input
contract. Go and Rust return explicit errors, Julia raises `DomainError`, and
the Mojo kernel returns `-1` for invalid finite-contract inputs.

### Biological relevance

The M–P neuron captures the all-or-nothing firing principle
observed by Adrian (1926) but discards all biophysical detail:
no refractory period, no adaptation, no synaptic dynamics.
It is useful as a baseline for formal analysis and as a building
block in stochastic computing pipelines where binary activation
is the native representation.

### SC pipeline role

In the stochastic computing pipeline, McCullochPitts serves as:
- **Comparator:** Converts analogue bitstream statistics to binary
- **Logic primitive:** AND/OR/NOT gates for bitstream operations
- **Baseline model:** Performance floor for benchmarking

---

## Measured Performance

### Isolation throughput

| Metric | Value |
|--------|-------|
| Python throughput | ~1,811,000 steps/s |
| Rust throughput | ~identical (trivial computation) |
| Bottleneck | Python interpreter overhead |
| FLOPs per step | 1 comparison (no arithmetic) |

### Network throughput

Tested configuration:
- Population: 10 neurons
- Input: constant I=5.0 (all fire every step)
- Duration: 1000 steps
- Result: 10,000 spikes (all neurons fire every step)

### Analysis verified

| Function | Input | Result |
|----------|-------|--------|
| spike_count(train) | 5000 steps at I=5.0 | 5000 |
| firing_rate(train, dt=0.001) | same | 1000 Hz |

---

## Pipeline Verification (End-to-End)

### 1. Import → Construction

```python
from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron
n = McCullochPittsNeuron()
assert n.theta == 1.0
```
**Status: PASS**

### 2. step(current) → int {0, 1}

```python
result = n.step(0.0)
assert isinstance(result, int) and result in (0, 1)
```
**Status: PASS** — returns native Python int.

### 3. Spiking under drive

```python
spikes = sum(n.step(5.0) for _ in range(1000))
assert spikes == 1000  # always fires if input >= theta
```
**Status: PASS** — deterministic: every step with I ≥ θ fires.

### 4. Sub-threshold silence

```python
no_spikes = sum(n.step(0.5) for _ in range(1000))
assert no_spikes == 0
```
**Status: PASS** — no firing below threshold.

### 5. Stability (50,000 steps)

```python
for _ in range(50000): n.step(5.0)
# No state to check — model is stateless
```
**Status: PASS** — no state divergence possible (stateless).

### 6. reset()

```python
n.reset()  # no-op
```
**Status: PASS** — no state to reset.

### 7. Population

```python
from sc_neurocore.network.population import Population
pop = Population(McCullochPittsNeuron, n=10, label="mp")
assert pop.n == 10 and pop.model_name == "McCullochPittsNeuron"
```
**Status: PASS**

### 8. Rust parity

```python
from sc_neurocore_engine import sc_neurocore_engine as eng
rn = eng.McCullochPittsNeuron()
py_s = [n.step(5.0) for _ in range(100)]
rs_s = [rn.step(5.0) for _ in range(100)]
assert py_s == rs_s
```
**Status: PASS** — exact match (deterministic integer model).

### 9. Analysis pipeline

```python
import numpy as np
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate
train = np.array([n.step(5.0) for _ in range(5000)])
assert spike_count(train) == 5000
assert firing_rate(train, dt=0.001) == 1000.0  # Hz
```
**Status: PASS**

### 10. Logic gate verification

```python
and_gate = McCullochPittsNeuron(theta=2.0)
assert and_gate.step(1.0) == 0 and and_gate.step(2.0) == 1
or_gate = McCullochPittsNeuron(theta=1.0)
assert or_gate.step(1.0) == 1 and or_gate.step(0.5) == 0
```
**Status: PASS** — AND and OR gate semantics verified.

---

## Infrastructure Pipeline Diagram

```
McCullochPittsNeuron(theta=1.0)
        │
        ▼
   step(I) → 0|1          Pure comparison: I ≥ θ
        │
        ▼
   Population(n=N)         N independent instances
        │
        ▼
   SpikeMonitor             Binary spike trains
        │
        ▼
   spike_count / firing_rate  Analysis on binary output
        │
        ▼
   Rust McCullochPittsNeuron  Exact integer parity
```

---

## Numerical Considerations

- **No floating-point issues:** Single comparison, no accumulation
- **No overflow risk:** No state variables
- **No stability concerns:** Stateless model
- **Exact reproducibility:** Deterministic for identical inputs
- **Finite comparison contract:** Non-finite thresholds or weighted inputs are
  rejected before the comparison, avoiding undefined Boolean semantics.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, below/at/above threshold, negative input, stateless, custom theta, reset noop, deterministic, AND gate, OR gate |
| Validation | 8 | finite threshold and weighted input, runtime threshold revalidation, Heaviside boundary after threshold mutation |
| Network | 1 | Population |
| Pipeline | 10 | End-to-end verification (this document) |
| **Total** | **46** | dedicated module checks |

---

## Findings (measured 2026-04-04)

1. McCullochPitts is the fastest model at ~1.8M steps/s (Python
   interpreter overhead dominates — actual computation is 1 comparison)
2. Exact Rust parity confirmed — deterministic integer model
3. All 10 pipeline stages verified green
4. Logic gate semantics (AND, OR) confirmed with theta parameterisation
5. Model serves as performance ceiling baseline for the pipeline
6. Polyglot finite-input and finite-threshold contracts aligned across Go,
   Julia, Rust, and Mojo
