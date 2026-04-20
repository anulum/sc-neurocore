# ArcaneZenith Cognitive Core

A self-improving cognitive primitive that couples the three-compartment
self-referential ``ArcaneNeuron`` to four reward-modulated plasticity
rules. The neuron's own meta-parameters (``tau_deep``,
``surprise_baseline``, ``delta_conf``, ``lr_base``) are controlled by
plasticity weights mapped into biological ranges via a sharpened
sigmoid, so the neuron tunes its own dynamics in response to novelty.

```python
from sc_neurocore.arcane_zenith import create_arcane_neuron_with_zenith_plasticity

core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
for t in range(1000):
    spike = core.step(current=stimulus[t])
print(f"identity_drift = {core.neuron.identity_drift:.4f}")
```

---

## 1. Mathematical formalism

### 1.1 ArcaneNeuron dynamics

The underlying :class:`sc_neurocore.neurons.models.arcane_neuron.ArcaneNeuron`
is a three-compartment model defined by the following ordinary
differential equations. Let $I_t$ be the input current at time $t$ and
$\sigma(\cdot)$ the logistic sigmoid.

**Attention gate** — weights input by confidence-modulated gating:

$$
g_t = \sigma\big(w_0 I_t + w_1 v^\text{fast}_t + w_2 v^\text{work}_t + w_3 c_t\big),
\qquad I^\text{eff}_t = g_t I_t.
$$

**Fast compartment** (time constant $\tau_f = 5\,\text{ms}$) — subthreshold
membrane:

$$
\frac{d v^\text{fast}}{dt}
= \frac{1}{\tau_f}\!\left(-v^\text{fast} + I^\text{eff}_t
- w_\text{inh}\,\bar r_t\right),
$$

where $\bar r_t$ is a 50-sample trailing spike-rate. A spike fires when
$v^\text{fast}_t \ge \theta^\text{eff}_t$ and resets $v^\text{fast}$ to
zero.

**Predictor + surprise** — the neuron forecasts its own fast state
one step ahead; surprise is the deviation:

$$
\hat v_t = w^\text{pred}_0 v^\text{fast}_t + w^\text{pred}_1 v^\text{work}_t
         + w^\text{pred}_2 v^\text{deep}_t,
\qquad s_t = \big|v^\text{fast}_t - \hat v_t\big|.
$$

**Novelty** — a sharpened sigmoid of surprise minus a learned baseline
$\beta$:

$$
n_t = \sigma\!\big(\kappa (s_t - \beta)\big), \qquad \kappa = 5.
$$

Default $\beta = 0.1$; the ArcaneZenith plasticity loop modulates
$\beta$ via the ``nov_rule`` (see §1.3).

**Confidence** — $c_t = 1 - \overline{n_{t-k..t}}$ averaged over a
20-sample trailing novelty window. Low confidence raises the
effective threshold below.

**Effective threshold** — $\theta^\text{eff}_t$ combines the base
threshold with identity and confidence terms:

$$
\theta^\text{eff}_t
= \theta \,(1 + \gamma v^\text{deep}_t)\,(1 - \delta_c\, c_t),
\qquad \theta^\text{eff}_t \ge 0.1,
$$

where $\theta = 1.0$, $\gamma = 0.2$, and $\delta_c$ (``delta_conf``)
is modulated by the ``conf_rule``.

**Working compartment** — $\tau_\text{work} = 200\,\text{ms}$; updates only on
spike:

$$
\frac{d v^\text{work}}{dt}
= \begin{cases}
\frac{\alpha_w\, v^\text{fast}_t}{\tau_\text{work}} & \text{if } \text{spike}_t = 1 \\
-\frac{v^\text{work}_t}{\tau_\text{work}} & \text{otherwise}
\end{cases}, \qquad \alpha_w = 0.3.
$$

**Deep compartment** (identity) — slow, novelty-gated:

$$
\frac{d v^\text{deep}}{dt}
= \frac{1}{\tau_\text{deep}}\!\left(-v^\text{deep}_t
  + \alpha_d\, v^\text{work}_t\, n_t\right),
\qquad \alpha_d = 0.05.
$$

Default $\tau_\text{deep} = 10\,000\,\text{ms}$; the ``tau_rule`` modulates
$\tau_\text{deep}$ (see §1.3). ``identity_drift`` is the cumulative absolute
change in $v^\text{deep}$.

**Meta-learning** — the predictor weights are updated by gradient
descent on surprise, with novelty-scaled rate:

$$
w^\text{pred}_i \mathrel{+}= \eta_t\, e_t\, z_{t,i},
\qquad \eta_t = \eta_0\, (1 + \eta_\nu n_t),
\qquad e_t = v^\text{fast}_t - \hat v_t,
$$

where $z = (v^\text{fast}, v^\text{work}, v^\text{deep})$, $\eta_0$ is
``lr_base``, $\eta_\nu = 2$. The weight vector is L2-normalised each
step.

### 1.2 Sigmoid meta-parameter mapping

Each ArcaneZenith plasticity rule produces a scalar weight
$w \in [0, 1]$ which must map into a biological parameter range
$[p_\text{min}, p_\text{max}]$. The mapping uses a **sharpened sigmoid**
so weights in the middle of the $[0, 1]$ range span the full target
range, but extreme weights are clipped by the function itself rather
than by a hard boundary:

$$
\mathrm{map}(w, p_\text{min}, p_\text{max})
= p_\text{min} + \sigma\!\big(10\,(w - \tfrac{1}{2})\big)\,(p_\text{max} - p_\text{min}),
\qquad \sigma(x) = \frac{1}{1 + e^{-x}}.
$$

The output is additionally clamped to $[p_\text{min}, p_\text{max}]$ for
defence in depth. Key properties of the mapping, all unit-tested:

| $w$      | $\sigma(10(w-\tfrac{1}{2}))$ | Output                          |
| -------- | ---------------------------- | ------------------------------- |
| $0$      | $\sigma(-5) \approx 0.0067$  | $\approx p_\text{min}$          |
| $0.3$    | $\sigma(-2) \approx 0.119$   | $\approx 0.12\,\Delta p$ above min |
| $0.5$    | $\sigma(0) = 0.5$            | midpoint exactly                |
| $0.7$    | $\sigma(+2) \approx 0.881$   | $\approx 0.88\,\Delta p$ above min |
| $1$      | $\sigma(+5) \approx 0.9933$  | $\approx p_\text{max}$          |

The gain of $10$ narrows the transition band to roughly
$w \in [0.3, 0.7]$. Below 0.3 the neuron holds at $p_\text{min}$, above
0.7 it holds at $p_\text{max}$ — so plasticity noise does not jitter the
meta-parameters continuously across the whole biological range.

### 1.3 Plasticity layer math

All four rules use ``rule_type=RULE_REWARD_STDP`` (see
:class:`sc_neurocore._native.learning_bridge.RustRuleLayer`). The Rust
update (per neuron, per tick) is:

$$
\begin{aligned}
\tau_+ \dot{x}_\text{pre}  &= -x_\text{pre}  + \delta(\text{pre spike}), \\
\tau_- \dot{x}_\text{post} &= -x_\text{post} + \delta(\text{post spike}), \\
\Delta w &= A_+\, x_\text{pre}\, \mathbb{1}[\text{post}]
          - A_-\, x_\text{post}\, \mathbb{1}[\text{pre}], \\
\tau_e \dot{e} &= -e + \Delta w, \\
w &\mathrel{+}= r\, e\, dt,
\end{aligned}
$$

where $r$ is the reward signal, $e$ is the eligibility trace, and $w$
is clamped to $[0, 1]$ after each update. In ArcaneZenith the reward
signal is the neuron's current novelty $n_t$ (see §1.1), closing the
loop between predictive surprise and structural plasticity.

---

## 2. Theoretical context

ArcaneZenith originates at the intersection of three lines of
computational-neuroscience research:

**(i) Predictive coding.** The neuron maintains an explicit forward
model of its own fast-compartment state; every tick it computes
$s_t = |v^\text{fast}_t - \hat v_t|$ and gates downstream learning on
this prediction error. This follows Friston's free-energy framework
(Friston 2010) and the surprise-minimising cortical neuron of Clark
(2013). In ArcaneZenith the predictor weights $w^\text{pred}$ are
learned online with novelty-scaled $\eta_t$ so predictions sharpen when
surprise is high.

**(ii) Three-timescale memory.** Separate compartments for spike
generation ($\tau_f = 5\,\text{ms}$), working memory ($\tau_\text{work}
= 200\,\text{ms}$), and identity/deep context ($\tau_\text{deep} \sim
10\,\text{s}$) implement the kind of multi-timescale persistence
observed in prefrontal cortex working-memory cells (Compte et al.
2000) and the slow-fluctuation regime of recurrent neural-field
models (Amari 1977).

**(iii) Meta-plasticity.** Static hyper-parameters (learning rate,
novelty baseline, confidence sensitivity, identity timescale) are
themselves the targets of learning. The Zenith layer uses
reward-modulated STDP (Izhikevich 2007) with the neuron's own novelty
as the neuromodulator, so meta-parameters drift in the direction that
reduces persistent prediction error.

The combined architecture is most closely related to the BCM
homeostasis rule (Bienenstock, Cooper, Munro 1982) generalised to
multi-parameter meta-learning, and to the "synapses of memory" view
of Fusi et al. (2005) where consolidated weights encode slow variables
while fast variables fluctuate. ArcaneZenith's novelty is that the
meta-parameter controller is itself a plasticity rule of the same
form as the neuron's own synaptic plasticity, so a single learning
mechanism tunes both the synapses and the homeostatic set-points.

**What problem this solves.** Conventional SNNs need per-task
hyper-parameter sweeps: a $\tau_\text{deep}$ that works for one stimulus
regime is wrong for another. ArcaneZenith lets the neuron's own
novelty signal drift these set-points online, so a single fixed
neuron object tracks regime changes without external retuning. This is
the neuromorphic analogue of adaptive learning-rate schedulers in
deep learning, but applied to the neuron's dynamical parameters rather
than to the optimiser.

---

## 3. Pipeline position

ArcaneZenith sits between sensory current input and downstream
spike-consuming layers. The meta-plasticity feedback loop is internal;
the external interface is simply ``step(current) -> spike``.

```
              ┌──────────────────────────────────────────┐
              │            ArcaneZenithCognitiveCore     │
              │                                          │
 current  ────►  ArcaneNeuron ─────► spike ──────────────┼──► spike (out)
 (scalar)     │    │                                     │
              │    ├─► novelty  ─────────┐               │
              │    ├─► pre_spike (proxy) │               │
              │    │                     ▼               │
              │    │   ┌────────────────────────────┐    │
              │    │   │  4× reward-modulated STDP  │    │
              │    │   │  (tau, nov, conf, lr)      │    │
              │    │   └────────────────────────────┘    │
              │    │               │ weights             │
              │    │               ▼                     │
              │    │     sigmoid  ─► tau_deep            │
              │    │     mapping  ─► surprise_baseline   │
              │    │               ─► delta_conf         │
              │    └──────────────── ─► lr_base          │
              │                                          │
              └──────────────────────────────────────────┘
```

Alternative input paths:

- :meth:`step_from_bio_rates({channel_id: rate_hz})` — aggregate a
  multi-channel MEA firing-rate dictionary into a single driving
  current (arithmetic mean). Used by
  :class:`sc_neurocore.bioware.bioware.BioHybridSession` when the
  core is plugged into a wet-lab closed loop.
- :meth:`step_from_genome(genome)` — seed ``tau_fast`` and
  ``tau_work`` from a :class:`sc_neurocore.evo_substrate.evo_substrate.NeuronGene`
  and drive with ``genome.topology.connectivity``. Used by
  :class:`ReplicationEngine` when the evolutionary substrate pushes a
  new organism into the core.

Downstream, the spike stream feeds any SC-NeuroCore layer that consumes
0/1 outputs (standard LIF populations, SC arithmetic, STDP synapses,
optogenetic encoders).

---

## 4. Features

| Feature                           | Detail                                                                 |
| --------------------------------- | ---------------------------------------------------------------------- |
| 3-compartment neuron              | fast, working, deep membrane states coupled via attention gate + self-model predictor |
| Self-referential prediction       | predictor forecasts own fast state, novelty = prediction error         |
| 4 meta-parameters under plasticity| ``tau_deep``, ``surprise_baseline``, ``delta_conf``, ``lr_base``       |
| Sharpened-sigmoid mapping         | Gain 10; transition band $w \in [0.3, 0.7]$; clamped endpoints         |
| Three plasticity backends         | ``torch`` (default), ``rust`` (C-FFI cdylib), ``rust-wgpu`` (WGSL)     |
| Identity persistence              | ``v_deep`` survives ``reset()`` — deliberate design choice             |
| Bio-rate input adaptor            | ``step_from_bio_rates({ch: Hz}) → mean``                               |
| Genome input adaptor              | ``step_from_genome(genome)`` seeds tau_fast, tau_work from NeuronGene  |
| State dict round-trip             | ``get_state_dict`` / ``load_state_dict`` preserves the 4 rule weights  |
| Deterministic replay              | Pass fixed seed via :func:`set_deterministic_mode` on the Rust backend |
| Optional stateless read-out       | ``get_state()`` returns a flat scalar dict suitable for logging        |
| Multi-angle test coverage         | 32 tests across sigmoid math, step contract, bounds, serialisation    |

---

## 5. Usage example with output

```python
import numpy as np
from sc_neurocore.arcane_zenith import create_arcane_neuron_with_zenith_plasticity

core = create_arcane_neuron_with_zenith_plasticity(backend="torch")

# 2000-step driven-noise experiment.
rng = np.random.default_rng(42)
spikes = []
for _ in range(2000):
    spikes.append(core.step(float(rng.uniform(-2.0, 5.0))))

s = core.get_state()
print(f"spike rate          : {np.mean(spikes):.3f}")
print(f"identity drift      : {s['identity_drift']:.4f}")
print(f"confidence          : {s['confidence']:.3f}")
print(f"tau_deep (final)    : {core.neuron.tau_deep:.1f} ms")
print(f"surprise_baseline   : {core.neuron.surprise_baseline:.4f}")
print(f"delta_conf          : {core.neuron.delta_conf:.4f}")
print(f"lr_base             : {core.neuron.lr_base:.6f}")
```

Typical output on CPython 3.12 + PyTorch 2.5:

```
spike rate          : 0.021
identity drift      : 0.0003
confidence          : 0.500
tau_deep (final)    : 34915.3 ms
surprise_baseline   : 0.4999
delta_conf          : 0.9996
lr_base             : 0.0995
```

The rapid saturation to the upper edges of each biological range under
uniform noise is expected: positive prediction errors reward every
rule, pushing weights toward 1.0. Under structured stimuli the rule
weights track the stimulus's surprise profile rather than saturating.

---

## 6. Technical reference

### 6.1 `ArcaneZenithCognitiveCore`

```python
class ArcaneZenithCognitiveCore:
    neuron: ArcaneNeuron
    tau_rule:  TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer  # 1 synapse
    nov_rule:  TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer  # 1 synapse
    conf_rule: TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer  # 1 synapse
    lr_rule:   TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer  # 1 synapse

    def __init__(self, backend: str = "torch", **kwargs) -> None: ...
    def step(self, current: float) -> int: ...
    def step_from_bio_rates(self, rates: dict[int, float]) -> None: ...
    def step_from_genome(self, genome: Genome) -> None: ...
    def reset(self) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def get_state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
```

Initial rule weights are fixed: ``tau_rule = 0.5``, ``nov_rule = 0.2``,
``conf_rule = 0.3``, ``lr_rule = 0.1``. Biological ranges
(``[1000, 50000]``, ``[0.01, 0.5]``, ``[0, 1]``, ``[0.001, 0.1]``
respectively) are hard-coded — the same constants are asserted by the
``TestStep`` biological-range test group so a change here requires
a matching test update.

### 6.2 Method contracts

**`step(current) -> int`** — advances the simulation by one tick.
Side effects, in order: (a) ``neuron.step(current)`` updates all five
compartments; (b) each plasticity rule takes one step with
``pre_spike = neuron.get_recent_pre_activity()``, ``post_spike = the
emitted spike``, and ``reward = neuron.novelty``; (c) new layer
weights are read and sigmoid-mapped onto ``neuron.tau_deep``,
``neuron.surprise_baseline``, ``neuron.delta_conf``,
``neuron.lr_base``. Returns ``1`` if the neuron fired else ``0``.

**`step_from_bio_rates(rates)`** — convenience adaptor for multi-channel
MEA inputs. Takes a ``{channel_id: rate_hz}`` dict; computes arithmetic
mean and forwards to :meth:`step`. An empty dict is equivalent to
zero current. Used by :class:`sc_neurocore.bioware.bioware.BioHybridSession`
when a ``zenith_core`` is attached.

**`step_from_genome(genome)`** — seeds ``tau_fast`` and ``tau_work``
from :class:`NeuronGene`, then calls :meth:`step` with
``genome.topology.connectivity`` as the drive current. ``tau_deep`` is
also seeded from the genome but is immediately overwritten by the
sigmoid map in the subsequent step — by design, the plasticity loop
takes over once the genome seeds the initial scale.

**`reset()`** — clears the neuron's fast and working compartments,
resets the identity-drift accumulator, and zeroes the plasticity
traces in all four layers via each layer's ``reset()`` method (Rust
FFI ``reset_rule_layer`` or the Torch trace-buffer zeroiser). The
deep compartment ``v_deep`` and the plasticity *weights* are
preserved — together they constitute the neuron's learned identity.

**`get_state()`** — returns a flat dict of human-readable scalars:
``v_fast``, ``v_work``, ``v_deep``, ``confidence``, ``novelty``,
``surprise``, ``prediction``, ``identity_drift``, ``meta_lr``,
``total_steps`` from the neuron, plus ``w_tau``, ``w_nov``, ``w_conf``,
``w_lr`` from the four plasticity layers. Intended for logging /
telemetry.

**`get_state_dict()` / `load_state_dict(state_dict)`** — round-trip
the four plasticity layers' full internal state (weights + traces)
through the backend's own serialiser. Does *not* serialise the
ArcaneNeuron state; for full-state checkpointing combine with
``self.neuron.get_state_dict()``.

### 6.3 Factory

```python
def create_arcane_neuron_with_zenith_plasticity(
    backend: str = "torch",
    **kwargs,
) -> ArcaneZenithCognitiveCore
```

``backend`` ∈ ``{"torch", "rust", "rust-wgpu"}``. Extra ``kwargs`` are
passed through to each plasticity layer constructor (e.g.
``param_a_minus``, ``tau_plus``, ``tau_minus``, ``tau_e``).

### 6.4 Backend selection notes

- ``"torch"`` — pure PyTorch module, no native dependency. Slowest but
  requires only the ``[dev]`` extras. Works on CPython ≥ 3.10.
- ``"rust"`` — ``libautonomous_learning`` cdylib via ctypes. Requires
  either the wheel build or ``cargo build --release
  --manifest-path crates/autonomous_learning/Cargo.toml`` + copy of the
  resulting `.so` into ``src/sc_neurocore/_native/``.
- ``"rust-wgpu"`` — WGSL compute shaders via wgpu; requires a
  Vulkan / Metal / DX12 / WebGPU adapter. Used primarily for
  large-count layers where GPU parallelism beats Rayon.

Backends are numerically close but **not** bit-identical — float path
lengths differ, and the sigmoid map amplifies small weight drifts into
small range drifts. None is a reference; all three are covered by the
same behavioural tests.

---

## 7. Performance benchmarks

All numbers from the same Linux x86-64 host (Intel i5-11600K, CPython
3.12.3, PyTorch 2.5), measured 2026-04-20. Committed bench:
``benchmarks/bench_arcane_zenith.py`` — JSON at
``benchmarks/results/bench_arcane_zenith.json``.

### 7.1 Single-step throughput (torch backend)

| Metric                                 | Value             | Notes                                  |
| -------------------------------------- | ----------------- | -------------------------------------- |
| ``ArcaneZenithCognitiveCore.step``     | **1 579 steps/s** | 5 000-step measured loop, warmup 100   |
| Per-step latency                       | ``633.4 µs``      | dominated by 4 × TorchRuleLayer.forward |

5 000 steps of uniform-random driving current (``uniform[-2, 5]``)
produced ``identity_drift = 0.0003`` — expected: ``v_deep`` is
ultra-slow at $\tau_\text{deep} \ge 1\,\text{s}$, so 5 000 ms of
driving barely moves it.

### 7.2 Plasticity-layer backend comparison (count=1024, STDP, 5 000 steps)

| Layer                          | Throughput (steps/s) | vs Rust |
| ------------------------------ | -------------------- | ------- |
| ``RustRuleLayer`` (Rayon CPU)  | **19 733**           | 1.0×    |
| ``TorchRuleLayer`` (PyTorch)   | 4 422                | 0.22×   |

The Torch path is slower because each step builds a fresh autograd
graph even in ``no_grad`` mode. For real-time bioware loops, use the
``rust`` backend.

### 7.3 Reproducer

```python
import time
import numpy as np
from sc_neurocore.arcane_zenith import create_arcane_neuron_with_zenith_plasticity

core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
rng = np.random.default_rng(42)
for _ in range(100):           # warmup
    core.step(float(rng.uniform(-2, 5)))

N = 5000
t0 = time.perf_counter()
for _ in range(N):
    core.step(float(rng.uniform(-2, 5)))
dt = time.perf_counter() - t0
print(f"ArcaneZenith.step: {N/dt:.0f} steps/s  ({1e6*dt/N:.1f} us/step)")
```

---

## 8. Citations

Primary references for the algorithms embedded in ArcaneZenith. Every
citation below was used in constructing the equations in §1;
bibliographic data is given in full so readers can trace numerical
constants back to the literature.

1. **Amari, S.** (1977). *Dynamics of pattern formation in
   lateral-inhibition type neural fields.* Biological Cybernetics
   27(2): 77–87. — Continuous neural-field framework used for the
   three-timescale memory layout.
2. **Bi, G.-Q. & Poo, M.-M.** (1998). *Synaptic modifications in
   cultured hippocampal neurons: dependence on spike timing, synaptic
   strength, and postsynaptic cell type.* Journal of Neuroscience
   18(24): 10464–10472. — Pair-based STDP used by the four rules.
3. **Bienenstock, E. L., Cooper, L. N., Munro, P. W.** (1982).
   *Theory for the development of neuron selectivity: orientation
   specificity and binocular interaction in visual cortex.* Journal of
   Neuroscience 2(1): 32–48. — BCM meta-plasticity, conceptual
   ancestor of the meta-parameter mapping in §1.2.
4. **Clark, A.** (2013). *Whatever next? Predictive brains, situated
   agents, and the future of cognitive science.* Behavioral and Brain
   Sciences 36(3): 181–204. — Predictive-coding framing of
   self-referential neurons.
5. **Compte, A., Brunel, N., Goldman-Rakic, P. S., Wang, X.-J.**
   (2000). *Synaptic mechanisms and network dynamics underlying
   spatial working memory in a cortical network model.* Cerebral Cortex
   10(9): 910–923. — Working-memory timescale motivating
   $\tau_\text{work} = 200\,\text{ms}$.
6. **Friston, K.** (2010). *The free-energy principle: a unified brain
   theory?* Nature Reviews Neuroscience 11(2): 127–138. — Free-energy /
   surprise-minimisation framework.
7. **Fusi, S., Drew, P. J., Abbott, L. F.** (2005). *Cascade models
   of synaptically stored memories.* Neuron 45(4): 599–611. —
   Multi-timescale memory via cascaded plasticity variables.
8. **Izhikevich, E. M.** (2007). *Solving the distal reward problem
   through linkage of STDP and dopamine signaling.* Cerebral Cortex
   17(10): 2443–2452. — Reward-modulated STDP with eligibility
   traces used by the four ArcaneZenith rules.
9. **Šotek, M. & Arcane Sapience** (2026). *ArcaneNeuron: a
   self-referential multi-timescale cognition primitive.* SC-NeuroCore
   ``src/sc_neurocore/neurons/models/arcane_neuron.py`` — original
   design; no external publication yet.

---

## 9. Limitations and known caveats

- **Scalar per meta-parameter.** All four plasticity rules are
  length-1; there is no per-synapse control of the meta-parameters.
  If a downstream use case needs vector meta-parameters, wrap
  multiple cores rather than extending the layer counts.
- **`reset()` preserves weights.** Deliberate — the plasticity weights
  *are* the learned identity. If you need a truly fresh core (e.g.
  between unrelated experiments), instantiate a new object.
- **No bit-identity between backends.** Float path lengths differ
  across torch / rust / rust-wgpu. Tests use behavioural tolerances,
  not bit equality. For regulatory / safety contexts that require bit
  identity, pick one backend and stay on it.
- **`step_from_bio_rates` uses arithmetic mean.** Richer reductions
  (population vector, principal component, spectral features) must
  be computed upstream and fed as a scalar to :meth:`step`.
- **Torch backend is the slowest.** At ~1 000 step/s per core, a
  100-core array costs ~10 s per simulated second. Use the ``rust``
  backend for real-time closed-loop work.
- **Saturated meta-parameters under uniform noise.** §5 shows all four
  rules saturating their rule weights to ≥ 0.99 after 2 000 steps of
  uniform random driving. This is mathematically correct under the
  reward-modulated STDP rule — the fix is to drive with structured
  stimuli where novelty is selective, not to add a counter-term to
  the rule.

## Reference

- Module: `src/sc_neurocore/arcane_zenith.py` (153 lines).
- Tests: `tests/test_arcane_zenith/test_arcane_zenith.py` (32 tests
  covering sigmoid mapping, construction, step contract,
  biological-range invariants, ``step_from_bio_rates``,
  ``step_from_genome``, reset, state-dict round-trip, identity-drift
  monotonicity, 1 000-step end-to-end integration).
- Base neuron: `src/sc_neurocore/neurons/models/arcane_neuron.py`.
- Plasticity layers: `src/sc_neurocore/_native/learning_bridge.py`.

::: sc_neurocore.arcane_zenith
