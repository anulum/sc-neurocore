# ParallelSpikingNeuron (k-order sliding PSN)

**Module:** `sc_neurocore.neurons.models.psn`
**Reference:** Fang, W., Yu, Z., Zhou, Z., Chen, D., Chen, Y., Ma, Z.,
Masquelier, T. & Tian, Y. (2023). Parallel Spiking Neurons with High
Efficiency and Ability to Learn Long-term Dependencies. NeurIPS 2023.
DOI [10.48550/arXiv.2304.12760](https://doi.org/10.48550/arXiv.2304.12760)
**Family:** Hardware / neuromorphic (learnable temporal filter)
**State variables:** `hidden` (H[t]), retained inputs (last `kernel_size` values)

---

## Equations

Streaming form of the PSN family (paper Eqs. 14–15):

$$H[t] = \sum_{i=0}^{k-1} W_i \cdot X[t-k+1+i], \qquad X[j] = 0 \;\text{for}\; j < 0$$

$$S[t] = \Theta\!\left(H[t] - V_{th}\right), \qquad \Theta(0) = 1$$

`weights[i]` is $W_i$, so `weights[k-1]` multiplies the newest input
$X[t]$ and `weights[0]` the oldest retained input $X[t-k+1]$. The sum
accumulates sequentially from $i = 0$, so every backend reproduces the
same binary64 result bit for bit.

**No reset.** Removing the neuronal reset is the paper's core premise:
no PSN variant clears state on firing, and firing never touches the
retained inputs. `reset()` only re-zeroes the retained inputs as an API
convenience.

### Variant selection

The paper defines three variants. The T-order PSN (Eqs. 9–10,
$H = WX$ with $W \in \mathbb{R}^{T\times T}$) and the k-order masked
PSN (Eqs. 11–12) are whole-sequence training formulations; the k-order
sliding PSN (Eqs. 14–15) is the only member expressible as a per-step
streaming neuron, so it is the canonical identity behind this class.

### Defaults

The paper trains $W$ and $V_{th}$ per task and publishes no universal
default. The uniform $W_i = 1/k$, $V_{th} = 1.0$ and $k = 8$ defaults
are repository defaults, documented as such in the descriptor.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_size` | 8 | Window order k: number of retained inputs |
| `v_threshold` | 1.0 | Firing threshold compared against H[t] |
| `weights` | uniform 1/k | Learnable weight vector W (length k) |

---

## Validation and atomicity

- Construction rejects a non-positive or non-integer `kernel_size`, a
  weight vector whose length differs from `kernel_size`, non-finite
  weights, and a non-finite threshold with a typed `ValueError`.
- `step()` rejects a non-finite input and a non-finite hidden state
  candidate with a typed `ValueError`; the pre-step state is preserved
  exactly (atomic rejection). The configuration is re-validated on
  every step, so post-construction mutation cannot smuggle invalid
  state through.

---

## Behaviour

- **Warm-up equals zero padding.** Before $k$ inputs have arrived, the
  missing history contributes exactly zero, so with the uniform kernel
  the hidden state is the sum of received inputs divided by $k$.
- **Constant drive.** With the uniform default kernel and constant
  input $I$, $H \to I$ once the window fills; the neuron fires every
  step when $I \ge V_{th}$ and never below it. There is no
  post-spike refractory dip, because there is no reset.
- **Learnable filter.** Non-uniform weights implement recency bias,
  temporal pattern detection, or change detection; the weights are the
  learnable parameters in the paper.

---

## Backend inventory

| Backend | Location | Parity |
|---------|----------|--------|
| Python | `src/sc_neurocore/neurons/models/psn.py` | reference |
| Rust (engine) | `engine/src/neurons/rate/parallel_spiking.rs` | bit-exact (atol = 0) |
| Rust (safety) | `src/sc_neurocore/accel/rust/safety/psn.rs` | bit-exact (atol = 0) |
| Go | `src/sc_neurocore/accel/go/services/psn.go` | bit-exact (atol = 0) |
| Julia | `src/sc_neurocore/accel/julia/neurons/psn.jl` | bit-exact (atol = 0) |

Executed parity: `tests/test_psn_backends.py` drives all four compiled
backends through 64 varied steps with a non-uniform weight vector and
compares hidden state and events at zero tolerance. Mojo and silicon
lanes are not implemented.

The engine class registers in the NetworkRunner under
`"ParallelSpiking"`/`"ParallelSpikingNeuron"`; the runner's voltage
readout reports the hidden state.

---

## Preserved historical identity

The structurally different recurrence formerly published under this
name — spike-triggered buffer clearing and circular kernel pairing —
is preserved count-neutrally as
[`SCResettingParallelSpikingNeuron`](sc_resetting_psn.md) with frozen
bit-exact anchors. Existing code that relied on the old behaviour
should construct that class instead.

---

## Test coverage

| Suite | What is verified |
|-------|------------------|
| `tests/test_model_psn_atomicity.py` | paper-equation oracle (bit-exact over 64 varied steps), Θ(0) = 1, warm-up zero padding, no-reset behaviour, typed rejections, atomicity |
| `tests/test_psn_backends.py` | four executed backends at zero tolerance, binding error paths, descriptor and page custody |
