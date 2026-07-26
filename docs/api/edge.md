# Edge — bare-metal SC runtime + AER mesh

Pure-Python port of the `tinysc_riscv` bare-metal crate: a complete
**SC inference runtime** that runs on a RISC-V MCU without FPU, plus
the **AER UDP mesh router** used for multi-FPGA deployments. The Python
and Rust sides are bit-compatible — a weight blob produced in Python
deserialises byte-for-byte on the MCU, and an LFSR bitstream produced
on either side matches the other's output word-for-word.

```python
from sc_neurocore.edge import (
    # Packed bitstream primitives
    popcount32, popcount_slice, sc_and, sc_or, sc_xor, sc_sub, sc_mux,
    and_packed, mux_packed, probability, scc,
    # Pseudo-random / low-discrepancy encoders
    Lfsr16, SobolGenerator,
    # SC neurons
    LifNeuron, IzhikevichNeuron,
    # Network runner
    SCLayer, SCNetwork,
    # Telemetry + serialisation + board profiles
    TelemetryRing, LayerTelemetry, DeviceTelemetry,
    WeightHeader, LayerHeader, WEIGHT_MAGIC,
    serialize_weights, deserialize_weights,
    PowerProfile, Board,
    PowerThermalConfig, build_power_thermal_model_from_vivado_reports,
    WebDeploymentConfig, build_web_deployment,
)
from sc_neurocore.edge.aer_router import AERRoutingDaemon
```

---

## Browser Deployment Scaffold

`sc_neurocore.edge.web_deploy` emits a deterministic static web bundle for
`.nir`, `.pt`, `.pth`, and JSON model artefacts. It is the first slice of the
WASM/WebGPU deployment path: generation does not require a browser, WebGPU
driver, Node.js, or a native WASM toolchain, and the emitted `manifest.json`
records the runtime contract honestly.

```python
from sc_neurocore.edge import WebDeploymentConfig, build_web_deployment

manifest = build_web_deployment(
    "model.nir",
    "build/web",
    WebDeploymentConfig(dt=1.0, bitstream_length=256),
)
print(manifest.artefacts["html"])  # index.html
```

Generated layout:

```text
build/web/
  index.html
  manifest.json
  model/model.nir
  runtime/sc_neurocore_web.js
  runtime/sc_neurocore_webgpu.wgsl
```

The browser runtime loads the manifest, checks WebGPU availability, and exposes
the SC probability contract used by later WASM kernels. The WGSL kernel clamps
probabilities into `[0, 1]`; it is intentionally small so it can be used as a
capability and packaging test before computational kernels are added.

---

## 1. Mathematical formalism

### 1.1 Packed bitstream word

A bitstream of length $L$ is stored as $\lceil L/32 \rceil$ unsigned
32-bit words. The popcount of one word is Wilkes–Wheeler–Gill:

$$
\mathrm{popcount32}(x)
= \sum_{b=0}^{31} \bigl((x \gg b) \wedge 1\bigr),
$$

computed in six SWAR steps (shift-and-mask) — the same algorithm as
`core_engine::bitstream::popcount32` in the Rust crate.

### 1.2 SC arithmetic via bitwise ops

Let $A,B \in \{0,1\}^{L}$ be independent unipolar streams with bit
probabilities $p_{A},\,p_{B}$. The SC arithmetic operators are:

$$
\begin{aligned}
\text{multiply}         \quad & A \wedge B,
  & p_{A\wedge B} &= p_{A}\,p_{B} \\
\text{saturating add}   \quad & A \vee B,
  & p_{A\vee B} &= p_{A} + p_{B} - p_{A}\,p_{B} \\
\text{abs.\ difference} \quad & A \oplus B,
  & p_{A\oplus B} &= p_{A}(1-p_{B}) + p_{B}(1-p_{A}) \\
\text{sat.\ subtract}   \quad & A \wedge \neg B,
  & p_{A \setminus B} &= p_{A}(1 - p_{B}) \\
\text{scaled add (MUX)} \quad & (S \wedge A) \vee (\neg S \wedge B),
  & p_{\text{mux}} &= p_{S}\,p_{A} + (1-p_{S})\,p_{B}
\end{aligned}
$$

These identities are the foundation of stochastic computing
(Gaines, 1967) and make all SC layers computable with only AND / OR /
NOT / MUX cells.

### 1.3 Alaghi–Hayes SCC

$$
\mathrm{SCC}(A,B) =
\begin{cases}
\dfrac{p_{A \wedge B} - p_{A}p_{B}}{\min(p_{A},\,p_{B}) - p_{A}p_{B}},
  & p_{A \wedge B} \geq p_{A}p_{B} \\[4pt]
\dfrac{p_{A \wedge B} - p_{A}p_{B}}{p_{A}p_{B} - \max(0,\,p_{A}+p_{B}-1)},
  & \text{otherwise}
\end{cases}
$$

:func:`scc` implements the case-split exactly; output is bounded in
$[-1,\,+1]$.

### 1.4 Galois LFSR-16 encoder

:class:`Lfsr16` uses the polynomial
$x^{16} + x^{14} + x^{13} + x^{11} + 1$ (Galois form, taps = `0xD008`)
with state $r_{t} \in \{1,\,\ldots,\,65535\}$:

$$
b_{t} = r_{t} \wedge \{2^{0}\!+\!2^{2}\!+\!2^{3}\!+\!2^{5}\},
\qquad
r_{t+1} = (r_{t} \gg 1) \,\vee\, (b_{t} \ll 15).
$$

Period is 65 535 (maximal for 16 bits). The probability encoder
compares the 16-bit state to a threshold $\theta \in [0,\,65535]$:

$$
\mathrm{bit}_{t} = \mathbf{1}\bigl[r_{t} < \theta\bigr],
\qquad
p = \theta/65535.
$$

### 1.5 Sobol low-discrepancy sequence

:class:`SobolGenerator` implements 1-D Sobol with Joe–Kuo direction
numbers (dimension 1: $V_{k} = 2^{15-k}$ for $k = 1..16$). Using
Gray-code indexing, each step costs **one XOR**:

$$
x_{n+1} = x_{n} \oplus V_{c(n)},
\qquad
c(n) = \text{position of lowest 1-bit in } n.
$$

Bits generated with Sobol thresholding have discrepancy $O(\log^{d} N /
N)$ vs LFSR's $O(1/\sqrt{N})$, so SC pipelines fed from Sobol streams
hit target precision with $L$ up to 4× shorter.

### 1.6 LIF with popcount accumulator

:class:`LifNeuron` tracks membrane $V$ as a **running popcount** with
right-shift leak:

$$
V_{t+1} = V_{t} + \mathrm{popcount}(I_{t}) - (V_{t} \gg s), \qquad
V_{t+1} \geq \Theta \;\Rightarrow\; \text{spike},\;\; V_{t+1} \leftarrow 0,
$$

where $s$ is the leak-shift (typical $s=3$ gives $\tau_{\text{leak}} =
1/(1 - 2^{-3}) \approx 8$ ticks). No multiplier, no FPU needed.

### 1.7 Izhikevich in Q16.16

:class:`IzhikevichNeuron` runs Izhikevich's 2003 two-variable model
entirely in 32-bit integer Q16.16:

$$
\dot{V} = 0.04 V^{2} + 5 V + 140 - U + I,
\qquad
\dot{U} = a (b V - U),
$$

with reset $(V \geq 30 \Rightarrow V \leftarrow c,\; U \leftarrow U + d)$.
Q16.16 gives $\Delta = 2^{-16} \approx 1.5 \cdot 10^{-5}$ resolution on
$V$; the four presets (regular spiking, fast spiking, chattering,
intrinsic burst) match the published $(a,b,c,d)$ tuples exactly.

### 1.8 Weight-blob wire format

```
offset | size | field
-------+------+----------------------
  0    |  4   | magic  = 0x5343_574C ("SCWL")
  4    |  4   | version
  8    |  4   | n_layers
 12    |  4   | flags
 16    |  4   | layer[0].n_inputs
 20    |  4   | layer[0].n_outputs
 24    |  4   | layer[0].threshold
 28    |  4   | reserved
 32    |  *   | layer[0].weights (n_outputs × words_per_row × u32)
 ...
```

All multi-byte fields are little-endian. `words_per_row = ⌈n_inputs/32⌉`.

---

## 2. Theory (why these particular mechanics)

### 2.1 Why bit-exact with Rust

The Rust crate (`tinysc_riscv`) is what actually runs on the MCU; the
Python module is the design-space exploration surface. Any divergence
between the two introduces silent drift — a network simulated in
Python that no longer matches its MCU deployment. We therefore keep
the same integer arithmetic, the same LFSR polynomial, the same Q16.16
encoding, the same byte layout. `tests/test_edge/test_parity.py`
asserts byte-identity on 1024-bit LFSR streams, Izhikevich spike
trains, and weight blobs.

### 2.2 Why integer arithmetic everywhere

The target board family (Table in §4) includes GD32VF103 (32 kB RAM, no
FPU) and ESP32-C3 (400 kB RAM, no FPU). Running an SC network there
means:

- No `float` anywhere — membrane, thresholds, weights, plasticity
  rates are all integers.
- No heap allocation — weight blobs are loaded zero-copy from flash.
- Popcount is the only inner kernel that must be fast; the RISC-V
  `cpop` extension is a single-cycle instruction, and the WWG fallback
  is ~6 cycles on cores without it.

### 2.3 Why both LFSR and Sobol

LFSR is the default because it is the smallest stream generator that
exists (16 FFs + 4 XOR gates) and perfectly matches the Rust
`core_engine::bitstream::Lfsr16`. Sobol is the **precision** option: for
a target bit probability $p$, Sobol converges to $\hat{p} = p \pm
O(\log N / N)$ vs LFSR's $\pm O(1/\sqrt{N})$. On a 1024-bit stream at
$p=0.5$, LFSR's 1-σ error is ≈0.0156; Sobol's is ≈0.002. Sobol costs
more per sample (one XOR + one trailing-zero count) but lets a
bandwidth-constrained fabric cut $L$ by 4× for the same MSE.

### 2.4 Why SCLayer does OR-combine, not concatenation

The most common pitfall when cascading SC layers is to simply
*concatenate* the upstream spike-stream words. This destroys the
probability interpretation (the combined stream becomes biased by
word order). :meth:`SCNetwork._flatten_bitstreams` instead uses the
**SC saturating-add identity** $A \vee B$, which preserves the
probability semantics up to a correction term $p_{A}p_{B}$ that is
small when individual inputs have low density.

### 2.5 Why cascading re-encodes through LFSR

Between layers, the boolean spike vector from layer $L_{k}$ is
re-encoded to a new bitstream by :class:`Lfsr16`. This deliberate
*re-randomisation* prevents correlation buildup: a spike from layer
$k$ that fires twice in a row would otherwise produce a perfectly
correlated stream that cannot be multiplied downstream
(`SCC(A, A) = 1` gives `A ∧ A = A`, not `p²`). Re-encoding restores
independence at the cost of one LFSR-cycle per spike — free on any
core with a hardware XOR tree.

### 2.6 Why the AER router is Go, not Python

UDP mesh routing at AER rates (~Mpkts/s peak) needs concurrency and
zero GC pauses. Go's goroutine model handles the fan-out per channel
without the GIL bottleneck that Python would have. The Python
:class:`AERRoutingDaemon` is a supervisor: `go build` on demand,
spawn, `SIGTERM` on teardown.

---

## 3. Position in the pipeline

```
        ┌──────────────────────────────────────────────┐
        │                Python design                  │
        │  (train / evolve a network, export weights)   │
        └──────────────────┬───────────────────────────┘
                           │
                    serialize_weights()
                           │
                           ▼
                    ┌─────────────┐
                    │ .bin blob   │
                    │ (SCWL magic)│
                    └──────┬──────┘
                           │  zero-copy load on boot
                           ▼
       ┌────────────────────────────────────┐
       │         RISC-V MCU target           │
       │  ┌──────────────────────────────┐  │
       │  │  sc_neurocore::edge (Rust)   │  │
       │  │  Lfsr16 → SCLayer.forward    │  │
       │  │    → LifNeuron / Izhikevich  │  │
       │  │    → TelemetryRing           │  │
       │  └──────────────────────────────┘  │
       └─────────────────┬──────────────────┘
                         │ AER events (UDP)
                         ▼
                ┌─────────────────────┐
                │  AERRoutingDaemon   │  (Go)
                │     UDP mesh        │
                └─────────┬───────────┘
                          │
                          ▼
                 Downstream FPGA tiles
```

---

## 4. Supported boards

| Board       | RAM   | Flash  | Active µW @160 MHz | Sleep µW | Comment              |
| ----------- | ----- | ------ | ------------------: | -------: | -------------------- |
| ESP32-C3    | 400 kB | 4 MB   |             15 000 |        5 | WROOM-02-class        |
| ESP32-C6    | 512 kB | 4 MB   |             18 000 |        7 | adds 802.15.4         |
| ESP32-H2    | 320 kB | 4 MB   |             12 000 |        3 | low-power variant     |
| GD32VF103   |  32 kB | 128 kB |              8 000 |       10 | smallest, cheapest    |
| CH32V307    |  64 kB | 256 kB |             10 000 |        8 | high I/O count        |
| K210        | 8 MB  | 16 MB  |            300 000 |       50 | dual-core + KPU        |
| Generic     |  64 kB | 256 kB |             10 000 |       10 | conservative fallback  |

:meth:`PowerProfile.for_board` linearly scales `active_uw` with clock
frequency (reference 160 MHz). :meth:`MemoryFootprint.estimate` checks
that a chosen network fits before deployment.

---

## 5. Features

- 11 bitstream primitives (popcount + 5 SC ops + 3 packed variants +
  probability + SCC).
- Bit-compatible LFSR-16 and Gray-code Sobol encoders.
- Two SC neuron models (LIF, Izhikevich with 4 presets).
- Multi-layer SCNetwork runner with per-layer cascading re-encode.
- Zero-copy weight blob (`SCWL` magic, 16-byte headers, little-endian).
- Runtime telemetry ring buffer (per-layer + per-device).
- 7 pre-profiled RISC-V MCU targets + memory-footprint estimator.
- Cargo / memory.x config generators (:func:`generate_cargo_config`,
  :func:`generate_memory_x`) for no_std RISC-V builds.
- Go AER UDP mesh router with Python lifecycle supervisor.

---

## 6. Usage

### 6.1 Run a 2-layer SC network

```python
from sc_neurocore.edge import SCNetwork, SCLayer

net = SCNetwork(bit_length=1024)
net.add_layer(SCLayer(n_inputs=32, n_outputs=16))
net.add_layer(SCLayer(n_inputs=16, n_outputs=8))
spikes = net.run([0.5] * 32)    # bool[8]
```

`SCNetwork` rejects non-integral bit lengths, non-finite or non-real input
probabilities, malformed encoded bitstream widths, and words outside the
unsigned 32-bit domain before layer execution. `SCLayer` applies the same
integer and u32-domain checks to configuration, weights, and input words, so
corrupted blobs or subclassed encoders fail at the Python boundary rather than
falling through to bitwise operations.

### 6.2 Export + reload weights

```python
from sc_neurocore.edge import serialize_weights, deserialize_weights
blob = serialize_weights(net.export_weights())
open("weights.bin", "wb").write(blob)

# On the MCU side, Rust loads it zero-copy; in Python we round-trip:
layers = deserialize_weights(open("weights.bin", "rb").read())
net2 = SCNetwork.from_weights(layers, bit_length=1024)
```

Expected blob size for a `32→16→8` network:
header (16 B) + 2 × layer header (16 B) + weight words
($16 \cdot 1 \cdot 4 + 8 \cdot 1 \cdot 4 = 96$ B) = **144 B**.

### 6.3 Power budget

```python
from sc_neurocore.edge import Board, PowerProfile, MemoryFootprint

prof = PowerProfile.for_board(Board.ESP32_C3, clock_mhz=80)
print(prof.duty_cycled_uw(duty=0.2))  # µW at 20 % active

fp = MemoryFootprint.estimate(
    num_layers=2, neurons_per_layer=64, bs_words=32, board=Board.GD32VF103
)
print(fp.fits_in_ram, fp.stack_bytes)
```

### 6.4 FPGA report-derived power/thermal JSON

Deployment bundles can emit a pre-silicon model from architecture settings, or
a report-derived model once Vivado has produced routed reports. The
report-derived path records the Vivado headline power, static/dynamic split,
effective TJA, junction temperature, and implementation resource counts while
preserving the SC workload metadata used by the estimator.

```python
from sc_neurocore.edge import (
    PowerThermalConfig,
    write_power_thermal_model_from_vivado_reports,
)

write_power_thermal_model_from_vivado_reports(
    "vivado_project/sc_shd_pynq.runs/impl_1",
    "build/deployable_artifacts",
    PowerThermalConfig(
        target="zynq",
        layer_sizes=((700, 128), (128, 128), (128, 20)),
        bitstream_length=256,
        clock_mhz=100.0,
    ),
)
```

The emitted JSON uses `source_mode = "vivado_report_derived"`. It is still not
a substitute for physical PYNQ board measurement; it is the reproducible bridge
between routed Vivado reports and the deployment artefact directory.

Vivado project, cache, generated-IP, and run directories are intentionally
untracked. Regenerate them from the maintained HDL inputs with
`hdl/pynq/create_block_design.tcl`; publish deployable outputs as release
artefacts rather than committing AMD/Xilinx generated files to the source tree.

### 6.5 Launch the AER router

```python
from sc_neurocore.edge.aer_router import AERRoutingDaemon
router = AERRoutingDaemon(port=9000)
router.start(build=True)
# ... experiment drives UDP events to localhost:9000 ...
router.stop()
```

---

## 7. Verified benchmarks

Measured on Ubuntu 24.04 / CPython 3.12.3 / Intel i5-11600K @ 3.90 GHz,
single-thread, 2026-04-20. Committed script:
`benchmarks/bench_edge.py`. Raw JSON at
`benchmarks/results/bench_edge.json`.

| Operation                                      | Throughput    | Latency    |
| ---------------------------------------------- | ------------- | ---------- |
| `popcount32` (pure-Py)                         | 3.30 M ops/s  | 303.0 ns   |
| `popcount_slice` (1024 words)                  | 3 545 ops/s   | 282.1 µs   |
| `Lfsr16.encode` (1024-bit)                     | 3 680 ops/s   | 271.8 µs   |
| `SobolGenerator.encode` (1024-bit)             | 916 ops/s     | 1 092.3 µs |
| `LifNeuron.tick` (32-word input)               | 99 909 ops/s  | 10.0 µs    |
| `IzhikevichNeuron.tick`                        | 2.21 M ops/s  | 453.2 ns   |
| `SCNetwork.run` (32→16→8 @ 1024 bits)          | 65 runs/s     | 15.35 ms   |
| `serialize_weights` (2-layer, 144 B blob)      | 199 047 ops/s | 5.02 µs    |
| `deserialize_weights` (2-layer)                | 126 270 ops/s | 7.92 µs    |
| `scc` (32 words = 1024 bits)                   | 36 076 ops/s  | 27.7 µs    |

Figures above are `time.perf_counter` deltas from
`benchmarks/bench_edge.py`.

**Interpretation.**

- `popcount32` at 319 ns is slow relative to the Rust `cpop`
  instruction (~1 ns at 1 GHz) but the Python version is only used
  for R&D — on the MCU the Rust path is what runs.
- `Lfsr16.encode` vs `SobolGenerator.encode` on a 1024-bit stream:
  LFSR is ~4× faster per bitstream because the Sobol step needs
  `(idx & -idx).bit_length()` (trailing-zero count) per sample. Sobol
  pays that cost for better precision; the rule of thumb is to use
  Sobol only when $L \leq 256$ and precision is critical.
- `SCNetwork.run` at 14.34 ms for 32→16→8 is dominated by
  per-input LFSR re-encoding; a NumPy-vectorised version is available
  in `sc_neurocore.v3.engine` but is not bit-compatible with the MCU
  target and so is excluded from this module.
- `IzhikevichNeuron.tick` is 21× faster than `LifNeuron.tick` because
  the LIF tick runs a full popcount over 32 words per step, whereas
  Izhikevich just does integer MACs on two 32-bit state variables.

---

## 8. Citations

1. Gaines B.R. (1967). *Stochastic computing systems*. In *Advances in
   Information Systems Science*, vol. 2, Plenum, 37–172.
2. Alaghi A., Hayes J.P. (2013). *Exploiting correlation in stochastic
   circuit design*. ICCD-2013, 39–46. (SCC definition.)
3. Wilkes M.V., Wheeler D.J., Gill S. (1951). *The Preparation of
   Programs for an Electronic Digital Computer*. Addison-Wesley.
   (WWG popcount.)
4. Izhikevich E.M. (2003). *Simple model of spiking neurons*. IEEE
   TNN 14(6):1569–1572. (Two-variable Q16.16 model.)
5. Joe S., Kuo F.Y. (2008). *Constructing Sobol' sequences with better
   two-dimensional projections*. SIAM J. Sci. Comput. 30:2635–2654.
6. RISC-V Foundation (2021). *RISC-V Bit-Manipulation Extension v1.0*.
   (`cpop` / `cpopw` popcount instructions.)
7. Šotek M. (2026). *SC-NeuroCore: bare-metal SC runtime port*.
   Internal report, ANULUM.

---

## 9. Known limitations

- **Python runner is not the hot path.** `SCNetwork.run` in Python
  costs ~14 ms per forward; the MCU Rust runner completes the same
  network in <200 µs on a 160 MHz ESP32-C3. Use the Python path for
  verification and weight-blob authoring, not for inference timing.
- **Single-dimension Sobol.** Only dimension 1 (V_k = 2^{15-k}) is
  wired; decorrelating N > 1 input streams still needs a phase-shifted
  LFSR bank (see `sc_neurocore.v3.engine`).
- **No 64-bit packing on the Python side.** LFSR packs to u32;
  SobolGenerator packs to u64 — the two encoders are therefore not
  drop-in replacements in a u32-word pipeline. Pick one per layer.
- **`SCLayer.forward` has no bias term.** Dense layers are weight-only;
  the bias must be folded into the threshold at quantisation time.
- **Integer Izhikevich is a best-effort Q16.16 discretisation.** The
  quadratic term `v*v >> 14` rounds differently from a floating-point
  reference; results match the published spike trains qualitatively
  (regular / fast / chattering / burst) but not bit-for-bit against a
  double-precision Izhikevich simulator.
- **AER router is IPv4 UDP only.** IPv6 and unix-domain sockets are
  not supported; on a Linux host with `net.ipv6.bindv6only=1` the
  router will fail to bind without explicit IPv4 address.
- **No RTT / loss bookkeeping in the Python supervisor.** Dropped AER
  packets are visible on the Go side (in `aer_router` logs) but the
  Python daemon does not surface the metric.

---

## 10. Rust parity — what the MCU runs

The Rust crate `tinysc_riscv` under `crates/tinysc_riscv/` is the
physical MCU counterpart to every module on this page. The parity rule
is byte-exact for the four externally visible surfaces:

| Surface             | Python type / function                        | Rust counterpart                              |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| Packed bitstream    | `popcount32`, `sc_and`, …                     | `bitstream::popcount32`, `bitstream::sc_and`  |
| LFSR stream         | `Lfsr16`                                      | `bitstream::Lfsr16`                           |
| Q16.16 Izhikevich   | `IzhikevichNeuron` + 4 presets                | `neuron::IzhikevichNeuron` + 4 presets        |
| Weight blob         | `serialize_weights` / `deserialize_weights`   | `weights::load_zero_copy`                     |

The parity tests live in `tests/test_edge/test_parity.py` and assert
word-for-word equality on 1024-bit LFSR streams seeded at `0xACE1`,
and byte-for-byte equality on a 2-layer `SCWL` blob. Any change to
the Python representation that would break parity must include a
matching Rust change in the same PR.

The Python side additionally provides three surfaces with no Rust
counterpart because they are R&D-only: :class:`SobolGenerator`
(precision experiments), :class:`PowerProfile` / :class:`Board`
(pre-deployment footprint estimation), and :func:`generate_cargo_config`
/ :func:`generate_memory_x` (cross-compilation aids). These never ship
to the MCU.

The generated Julia file under `accel/julia/edge/` and Mojo files under
`accel/mojo/kernels/` are non-authoritative mirror transcripts. The Python
`sc_neurocore.edge.sc_network` module owns the runtime validation contract
unless a maintained loader explicitly dispatches to another language.

---

## 11. Reproducibility

Every number in §7 is reproducible from a clean checkout by running

```bash
python benchmarks/bench_edge.py
```

which writes `benchmarks/results/bench_edge.json` alongside the stdout
table. Randomness is deterministic: `Lfsr16(0xACE1)` and
`SobolGenerator(0)` have no hidden entropy, and `SCNetwork.run`
initialises a fresh LFSR per call.

Variance between runs on the same host is dominated by scheduler
jitter and CPU-cache state; on the CEO workstation the inner hot
paths (popcount32, Izhikevich tick, weights serialise) stay within
±5 % of the numbers in §7 across repeated runs. `SCNetwork.run` can
drift up to ±15 % because its 15 ms per-forward budget is dominated by
Python-level allocation inside `_spikes_to_bitstreams`; pin the
benchmark to a single core with `taskset -c 0 python
benchmarks/bench_edge.py` for lower variance if you need to track
regressions.

---

## 12. Embedding hook — telemetry ring

:class:`TelemetryRing` is a fixed-capacity lock-protected ring buffer
that stores u32 samples (Python uses an ``int`` list internally; the
Rust counterpart uses ``[u32; N]`` on the MCU). The ring is
`threading.Lock`-guarded on the Python side for multi-thread writers;
the Rust side uses a single-producer assumption and needs no lock.

API:

- :meth:`TelemetryRing.push(value)` — overwrite-on-full append.
- :meth:`TelemetryRing.mean()` — arithmetic mean of the current
  window.
- :meth:`TelemetryRing.last()` — most recent value.
- :attr:`TelemetryRing.count` — number of live entries (<= capacity).
- :attr:`TelemetryRing.capacity` — construction-time fixed capacity
  (default 256).

:class:`LayerTelemetry` wraps two rings — ``spike_rate_ring`` and
``utilization_ring`` (both capacity 64 by default) — and records per-
tick activity via :meth:`LayerTelemetry.record_tick(n_spikes, n_neurons)`.
:class:`DeviceTelemetry` aggregates a dict of :class:`LayerTelemetry`
and exposes :meth:`DeviceTelemetry.summary` returning a JSON-ready
dict of per-layer `(spike_count, tick_count, mean_spike_rate,
mean_utilization)` plus device-level `(total_ticks, total_spikes,
error_count)`.

On the MCU side the rings are the same shape and the same field
semantics; the Rust `telemetry::DeviceTelemetry::summary()` returns the
same dict (serialised as JSON to the UART/WebSocket transport), so the
HIL debugger's frame schema is invariant across target. A shared-
memory fast path between Rust and Go is not yet wired — today the
data travels as JSON over UART or WebSocket.

---

## Reference

- Sources (package root):
  - `src/sc_neurocore/edge/__init__.py` (74 LOC, re-export surface)
  - `src/sc_neurocore/edge/bitstream.py` (109 LOC)
  - `src/sc_neurocore/edge/lfsr.py` (66 LOC)
  - `src/sc_neurocore/edge/sobol.py` (91 LOC)
  - `src/sc_neurocore/edge/neuron.py` (107 LOC, LIF + Izhikevich)
  - `src/sc_neurocore/edge/sc_network.py` (266 LOC)
  - `src/sc_neurocore/edge/weights.py` (129 LOC, SCWL format)
  - `src/sc_neurocore/edge/telemetry.py` (133 LOC)
  - `src/sc_neurocore/edge/power_estimator.py` (113 LOC)
  - `src/sc_neurocore/edge/deploy.py` (63 LOC)
  - `src/sc_neurocore/edge/aer_router.py` (48 LOC, Go supervisor)
- Go daemon: `src/sc_neurocore/accel/go/services/aer_router/main.go` +
  `main_test.go`.
- Benchmark: `benchmarks/bench_edge.py`.
- Parity tests vs Rust `tinysc_riscv`: `tests/test_edge/*.py`.

::: sc_neurocore.edge.aer_router
    options:
      show_root_heading: true

::: sc_neurocore.edge.bitstream
    options:
      show_root_heading: true

::: sc_neurocore.edge.lfsr
    options:
      show_root_heading: true

::: sc_neurocore.edge.sobol
    options:
      show_root_heading: true

::: sc_neurocore.edge.neuron
    options:
      show_root_heading: true

::: sc_neurocore.edge.sc_network
    options:
      show_root_heading: true

::: sc_neurocore.edge.weights
    options:
      show_root_heading: true

::: sc_neurocore.edge.power_estimator
    options:
      show_root_heading: true
