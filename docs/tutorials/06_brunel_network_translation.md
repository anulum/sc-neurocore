<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Translating Brian2 Networks to Stochastic Computing

The Brunel (2000) balanced random network is the standard SNN benchmark:
N excitatory and N/4 inhibitory LIF neurons, 10% random connectivity,
Poisson external drive, asynchronous irregular (AI) firing at ~30 Hz.

This tutorial translates a Brian2 Brunel implementation into sc-neurocore,
covering parameter mapping, the 20-variant translator, and performance
tradeoffs.

**Reference:** Brunel, N. (2000). Dynamics of Sparsely Connected Networks
of Excitatory and Inhibitory Spiking Neurons. *J Comput Neurosci*, 8(3),
183--208.

**Prerequisites:** `pip install sc-neurocore numpy`. Brian2 optional.

## 1. The Brian2 Brunel model

```python
eqs = "dv/dt = -v / (tau * ms) : 1"
G = NeuronGroup(N, eqs, threshold="v > v_th", reset="v = v_reset",
                method="euler", dt=0.1*ms)
S_exc = Synapses(G[:N_E], G, on_pre="v_post += J")
S_inh = Synapses(G[N_E:], G, on_pre="v_post -= g*J")
PoissonInput(G, "v", N=C_E, rate=nu_ext*Hz, weight=J)
```

Parameters: `tau_mem=20 ms`, `v_threshold=20 mV`, `v_reset=10 mV`,
`J=0.1 mV`, `g=5`, `conn_prob=0.1`.
External rate: `nu_ext = eta * V_th / (J * C_E * tau_m)`, `eta=2`.

## 2. BrunelParams dataclass

`brunel_translator.py` captures the full Brian2 parameter set:

```python
from benchmarks.brunel_translator import BrunelParams

bp = BrunelParams(
    n_exc=800, n_inh=200, conn_prob=0.1,
    weight_exc=0.1, g_inh=5.0,
    v_threshold=20.0, v_reset=10.0, v_rest=0.0,
    tau_mem=20.0, dt=0.1,
)
# bp.n_total == 1000, bp.weight_inh == 0.5
```

## 3. Three translation strategies

### 3a. Voltage domain (V1) -- direct LIF mapping

`translate_v1_stochastic_lif` maps Brian2 parameters one-to-one onto
`StochasticLIFNeuron`. Both use Euler-discretised LIF with delta-PSC
voltage kicks (`v += w`):

```python
from benchmarks.brunel_translator import translate_v1_stochastic_lif
from sc_neurocore import StochasticLIFNeuron

params = translate_v1_stochastic_lif(bp)
neuron = StochasticLIFNeuron(**params["neuron_kwargs"])
# v_threshold=20.0, v_reset=10.0, tau_mem=20.0, dt=0.1, resistance=1.0

spike = neuron.step(0.0)  # leak-only; synaptic input via n.v += dv
```

The benchmark loop (see `snn_comparison.py::run_v1_stochastic_lif`)
builds a random connectivity matrix, applies Poisson external kicks
per timestep, sums recurrent synaptic input from previous spikes, and
calls `neuron.v += dv; neuron.step(0.0)` per neuron.

### 3b. Probability domain (V2) -- bitstream SC

`translate_v2_rate_matched` converts weights to probabilities for
`VectorizedSCLayer`. Synaptic transmission becomes AND-gate
multiplication: `P(out=1) = P(spike=1) * P(weight=1)`.

```python
from benchmarks.brunel_translator import translate_v2_rate_matched

params = translate_v2_rate_matched(bp)
# weight_prob = J / V_th = 0.005, ext_prob = rate * dt / 1000
# bitstream_length = 4096
```

### 3c. Fixed-point hardware (V3) -- FPGA-targeted

`translate_v3_fixed_point` maps to `FixedPointLIFNeuron`, a bit-true
model of the Verilog `sc_lif_neuron`. All values are Q8.8 fixed-point:

```python
from benchmarks.brunel_translator import translate_v3_fixed_point
from sc_neurocore import FixedPointLIFNeuron

params = translate_v3_fixed_point(bp)
# v_threshold_q = 5120 (20.0 * 256), leak_k = 1, j_exc_q = 25

neuron = FixedPointLIFNeuron(
    data_width=16, fraction=8,
    v_threshold=params["v_threshold_q"],
    v_reset=params["v_reset_q"],
)
spike, v_out = neuron.step(
    leak_k=params["leak_k"], gain_k=params["gain_k"], I_t=current_q,
)
```

V11 extends to Q16.12 (32-bit) for higher dynamic range.

## 4. The 20-variant taxonomy

| # | Backend | Domain | Target |
|---|---------|--------|--------|
| V1 | StochasticLIFNeuron | Voltage | CPU reference |
| V2 | VectorizedSCLayer | Probability | SC analysis |
| V3 | FixedPointLIF Q8.8 | Fixed-point | FPGA 16-bit |
| V4 | BitstreamSynapse + LIF | Hybrid | SC + analog |
| V5 | SCIzhikevichNeuron | Voltage | Burst dynamics |
| V6 | HomeostaticLIFNeuron | Voltage | Rate homeostasis |
| V7 | V1 + noise_std=1.0 | Voltage | Noise robustness |
| V8 | V1 + refractory=5 | Voltage | Biological detail |
| V9 | V1 + post-kick timing | Voltage | Brian2 timing match |
| V10 | V1 + exact exp leak | Voltage | Precision study |
| V11 | FixedPointLIF Q16.12 | Fixed-point | FPGA 32-bit |
| V12 | V1 + StochasticSTDPSynapse | Voltage | Online learning |
| V13 | BitstreamDotProduct | SC | Multi-channel sum |
| V14 | V4 + Sobol encoding | Hybrid | Low-discrepancy |
| V15 | JaxSCDenseLayer | Probability | GPU/TPU |
| V16 | SCRecurrentLayer | Reservoir | Temporal tasks |
| V17 | MemristiveDenseLayer | Probability | Device defects |
| V18 | V1 + Numba JIT | Voltage | CPU fast path |
| V19 | PyTorch CUDA | Voltage | GPU |
| V20 | Vectorized NumPy | Voltage | CPU vectorized |

V1/V7--V10/V18/V20 share the voltage-domain LIF and differ in integration
method or acceleration. V2/V13/V15--V17 operate purely in probability
domain. V3/V11 target RTL-faithful fixed-point. V4/V14 combine bitstream
synapses with analog neurons.

## 5. Running the benchmarks

```bash
# All 20 variants + Brian2 on 1K neurons
python benchmarks/snn_comparison.py --all --json results.json --markdown

# Publishable Brian2 head-to-head at multiple scales
python benchmarks/brian2_benchmark.py --scales 1000 10000 --repeats 3

# Quick parameter check
python -c "from benchmarks.brunel_translator import *; print(translate_v1_stochastic_lif(BrunelParams()))"
```

`brian2_benchmark.py` compares Brian2, V1, V3, V18, V20 and reports
mean +/- std wall time and peak memory across repeats.

## 6. Where SC wins and where Brian2 wins

**SC-NeuroCore advantages:**

- N <= 1000: V18/V20 match or beat Brian2 (code-generation overhead
  dominates at small scale).
- FPGA deployment: V3/V11 map directly to Verilog RTL. Brian2 has no
  hardware synthesis path.
- Fault tolerance: V4/V14/V17 model stuck-at faults and device
  variability. Brian2 treats synapses as ideal.
- Pre-silicon verification: `FixedPointLIFNeuron` is bit-true against
  the RTL without gate-level simulation.

**Brian2 advantages:**

- N > 1000: C++ code generation + sparse connectivity scale far better.
  At 10K neurons, Brian2 is 10--100x faster than V1's Python loop.
- Biological detail: conductance-based synapses, dendritic compartments,
  gap junctions. SC-NeuroCore models delta-PSC only.
- Ecosystem: NeuroML, PyNN, and the computational neuroscience toolchain.

**Honest framing:** SC-NeuroCore deploys spiking networks on digital
hardware (FPGA, ASIC) where stochastic arithmetic provides area/power
efficiency. Brian2 simulates biologically detailed neural circuits. The
Brunel benchmark is the common ground for comparing both.

## 7. Parameter mapping reference

| Brian2 | V1 (float) | V3 (Q8.8) | V2 (probability) |
|--------|-----------|-----------|-------------------|
| `tau` ms | `tau_mem` | `leak_k = int(dt/tau * 256)` | -- |
| `v_th` mV | `v_threshold` | `int(v_th * 256)` | -- |
| `v_reset` mV | `v_reset` | `int(v_reset * 256)` | -- |
| `J` mV | `weight_exc` (voltage kick) | `int(J * 256)` | `J / v_th` |
| `g*J` mV | `weight_inh` | `int(g*J * 256)` | `g * J / v_th` |
| `rate` Hz | `external_rate_hz` | Poisson lambda/step | `rate * dt / 1000` |

For hybrid V4: `w_prob = clip(J / v_th, 0.001, 0.999)` passed to
`BitstreamSynapse(w=w_prob, length=256)`, output scaled by
`popcount_scale = v_th / bitstream_length`.

## 8. Adding a custom variant

Write a `translate_vNN` function returning a parameter dict from
`BrunelParams`, then add a corresponding `run_vNN` in
`benchmarks/snn_comparison.py`:

```python
def translate_v21_custom(bp: BrunelParams) -> dict:
    return dict(
        neuron_kwargs=dict(
            v_threshold=bp.v_threshold, v_reset=bp.v_reset,
            v_rest=bp.v_rest, tau_mem=bp.tau_mem, dt=bp.dt,
            resistance=1.0, noise_std=0.0,
        ),
        weight_exc=bp.weight_exc, weight_inh=bp.weight_inh,
        external_rate_hz=bp.external_rate_hz,
        ext_weight=bp.weight_exc, delta_psc=True,
    )
```

## Further reading

- Tutorial 01: Stochastic Computing Fundamentals
- `docs/hardware/FPGA_TOOLCHAIN_GUIDE.md`: synthesising `sc_lif_neuron`
- `docs/benchmarks/BENCHMARK_REPORT.md`: full results across scales
