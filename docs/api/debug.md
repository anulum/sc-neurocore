# Spike-Level Debugger + HIL Telemetry

Three complementary debug surfaces for stochastic-computing and spiking
neural networks: an **offline spike tracer + analyser** for post-mortem
divergence and causality analysis, a **live bitstream oscilloscope**
(``sc_scope``) that computes per-layer density / effective-bits / SCC
on a live FPGA stream, an **adaptive SC doctor** (``sc_doctor``) that
auto-tunes bitstream length on-the-fly and adds Hamming(7,4) ECC when
needed, and a **Hardware-in-the-Loop telemetry daemon** that multiplexes
protobuf ``HILFrame`` messages onto a WebSocket for GUI/CI consumers.

```python
from sc_neurocore.debug.tracer import SpikeTracer, ExecutionTrace
from sc_neurocore.debug.analyzer import (
    find_divergence, spike_diff, causal_chain,
    DivergencePoint, CausalEvent,
)
from sc_neurocore.debug.sc_doctor import ScDoctor
from sc_neurocore.debug.sc_scope import (
    TransportBackend, TransportConfig, TransportType,
    BitstreamSample, AnalysisWindow, LiveAnalyzer,
    LayerErrorBudget, TriggerEngine, TriggerCondition,
    TriggerEvent, TriggerType, ScopeSession, ScopeRenderer,
    compute_scc,
)
from sc_neurocore.debug.hil_server import HILServerDaemon
from sc_neurocore.debug.hil_debugger import HILDebugger
```

---

## 1. Mathematical formalism

### 1.1 Offline trace divergence

Two :class:`ExecutionTrace` objects $A$ and $B$ with binary spike
matrices $S^{A}, S^{B} \in \{0,1\}^{T \times N}$ diverge at

$$
t^{\star} = \min\bigl\{\, t \,:\, \exists n \in [0,N)\;
           \mathrm{with}\; S^{A}_{t,n} \neq S^{B}_{t,n} \,\bigr\},
$$

and :func:`find_divergence` returns the tuple
$(t^{\star}, n^{\star}, S^{A}, S^{B}, V^{A}, V^{B},
|V^{A}_{t^{\star},n^{\star}} - V^{B}_{t^{\star},n^{\star}}|)$.
The associated mismatch rate is

$$
\rho_{\text{mis}} = \frac{1}{T N}
  \sum_{t,n} \mathbf{1}\!\left[S^{A}_{t,n} \neq S^{B}_{t,n}\right].
$$

### 1.2 Backward causal chain

Given target $(n^{\star},\, t^{\star})$, the causal chain of depth $D$ is

$$
\mathcal{C}(n^{\star}, t^{\star}, D)
= \bigl\{(n,t) \,\bigm|\, t^{\star}-D \leq t < t^{\star},\;
                       S_{t,n} = 1\bigr\},
$$

ordered by $t$ descending. Without connectivity information
:func:`causal_chain` returns all temporally preceding spikers — a
safe over-approximation that is refined by the subsequent chain walk.

### 1.3 Bitstream density + effective bits (Shannon)

For a packed u32 bitstream of length $L = 32\,w$ with popcount $p$,

$$
d = \frac{p}{L},
\qquad
H(d) =
\begin{cases}
0, & d \in \{0,1\} \\
-\bigl(d\log_{2}d + (1-d)\log_{2}(1-d)\bigr) \cdot L,
  & 0 < d < 1.
\end{cases}
$$

$H(d)$ is the :class:`BitstreamSample` ``effective_bits`` property — the
Shannon entropy of the stream scaled by length. A fully biased stream
($d=0$ or $d=1$) carries zero information; $d=0.5$ yields $L$ bits.

### 1.4 Stochastic-computing correlation (SCC)

For two u32-packed bitstreams $A,B$ of length $L$ with
$p_{A} = \frac{\mathrm{popcount}(A)}{L}$,
$p_{B} = \frac{\mathrm{popcount}(B)}{L}$,
$p_{AB} = \frac{\mathrm{popcount}(A \wedge B)}{L}$,

$$
\mathrm{SCC}(A, B)
= \frac{p_{AB} - p_{A}\, p_{B}}
       {\bigl|\min(p_{A}, p_{B}) - p_{A}\, p_{B}\bigr| + \epsilon}.
$$

This is Alaghi & Hayes' original SC-correlation measure
(Alaghi & Hayes, 2013) bounded in $[-1,+1]$: $+1$ = perfectly
correlated, $0$ = independent, $-1$ = anti-correlated. The
:func:`compute_scc` reference implementation walks the stream word-by-word
in pure Python; the same formula is fused inside the Mojo SIMD kernel
``scc_numerator_256w`` (see `mojo_accel.md` §4.2) for the hot path.

### 1.5 Adaptive bitstream-length feedback

:class:`ScDoctor` maintains a length $L$ and runs the control law

$$
L_{t+1} =
\begin{cases}
2 L_{t}, & \mathrm{SCC}_{t} > 0.15 \\
\max(256,\, L_{t}/2), & \mathrm{SCC}_{t} < 0.05 \\
L_{t}, & \text{otherwise}
\end{cases}
$$

with ECC auto-enabled when $L_{t} > 2048$. The hysteresis window
$[0.05,\,0.15]$ prevents the length from oscillating on noisy streams;
the $L \geq 256$ floor guarantees the 8-bit effective precision target.

### 1.6 Hamming(7,4) single-bit correction

Encoding maps 4 data bits $d_{1}d_{2}d_{3}d_{4}$ to a 7-bit codeword by

$$
p_{1} = d_{1} \oplus d_{2} \oplus d_{4}, \quad
p_{2} = d_{1} \oplus d_{3} \oplus d_{4}, \quad
p_{3} = d_{2} \oplus d_{3} \oplus d_{4}.
$$

Decoding recomputes the three syndrome bits and flips the single erred
bit if any $s_{i} \neq 0$. The code corrects **exactly one** error per
7-bit block — sufficient for the transient switching-noise model on the
JTAG / UART transport, not for burst errors.

### 1.7 Layer error budget

Per-layer :class:`LayerErrorBudget` tracks

$$
e_{t} = |d_{t} - d^{\star}|, \quad
\text{violation} \Leftrightarrow e_{t} > \tau,
\quad
\text{pass\_rate} = 1 - \frac{\sum_{t} \mathbf{1}[e_{t} > \tau]}{T}.
$$

The observed density $d_{t}$ is compared with the golden-model
expected density $d^{\star}$; the default tolerance $\tau = 0.05$ is
2$\sigma$ on a 1024-bit stream under the Gaussian rate approximation.

---

## 2. Theory (why these particular mechanics)

### 2.1 Separating trace recording from analysis

:class:`SpikeTracer` does **not** bake analysis into the stepping loop.
Instead it records full $(\text{spikes},\,\text{voltages},\,\text{currents})$
tensors, and analyses (divergence, causality) run off the recorded trace.
This matters because neuromorphic debugging frequently involves asking
new questions a day later — a recorded trace lets you re-query without
re-simulating. The storage cost is $3TN$ floats/bytes per trace, which
at $T=10^{4},\,N=10^{3}$ is ~240 MB for double-precision and 30 MB for
float32 — manageable for single-run post-mortems.

### 2.2 Live vs. offline: two price points

``sc_scope`` runs on a *streaming* budget: constant-time per sample,
constant memory (ring buffer), O(1) trigger evaluation. The offline
tools (``find_divergence``, ``spike_diff``, ``causal_chain``) run on a
*snapshot* budget: O(TN) or O(DN) over the full trace. Both exist because
most bugs are easier to catch in one mode but not the other — a slow
drift in mean density is obvious on the live scope; a single-step spike
inversion is obvious in the offline diff.

### 2.3 Why pure-Python ScDoctor instead of FPGA microcontroller

A design choice: the doctor has to stay in the Python host during
R&D so it can be tuned without re-synthesis. Once a working control law
is validated, the same constants (``initial_length``, threshold pair,
ECC-on threshold) lift to a SystemVerilog FSM on the hardware side
(see `hdl_gen.md` §3). The Python module is the reference implementation;
the Verilog is its compiled mirror.

### 2.4 Why Hamming(7,4) rather than BCH or LDPC

The transport protocol is small-frame (32–256 bytes) and noise is
dominated by single-bit switching events; Hamming(7,4) is the smallest
perfect SEC code (single error correcting), has trivial combinational
implementation, and adds 75 % overhead — an acceptable tax on a
bitstream that is already probabilistic by design. BCH/LDPC would beat
it on throughput but require an iterative decoder the FPGA does not
need.

### 2.5 Pluggable transports, simulated loopback

:class:`TransportBackend` abstracts over JTAG / UART / PYNQ DMA /
simulated. The simulated path (sinusoidally modulated density per layer)
lets the scope stack run entirely offline for CI, while the hardware
paths reuse exactly the same :class:`LiveAnalyzer` and :class:`TriggerEngine`
classes — no "sim-only" code branches in the analysis layer.

### 2.6 Server daemon as a Go child process

The HIL WebSocket server is written in Go
(`accel/go/services/hil_debugger/main.go`) because Go's goroutine model
handles many concurrent WebSocket clients at lower per-connection
overhead than asyncio. The Python :class:`HILServerDaemon` is a
supervisor: `go build` on-demand, spawn, probe `/health`, SIGTERM +
SIGKILL fallback on stop. Separating the Python lifecycle manager from
the Go streaming daemon keeps each side idiomatic.

---

## 3. Position in the pipeline

```
 ┌─────────────┐       ┌───────────────┐        ┌──────────────────┐
 │  SC network │──────▶│   sc_scope    │──────▶ │  LiveAnalyzer    │
 │ (hardware)  │       │ (transport +  │        │   + triggers     │
 └─────────────┘       │  trigger eng) │        └──────────────────┘
      │                └───────────────┘                │
      │                        │                         │
      ▼                        ▼                         ▼
 ┌─────────────┐       ┌───────────────┐        ┌──────────────────┐
 │ SpikeTracer │──────▶│ ExecutionTrace│──────▶ │ find_divergence  │
 │  (offline)  │       │  (T×N arrays) │        │ causal_chain     │
 └─────────────┘       └───────────────┘        │ spike_diff       │
      │                                         └──────────────────┘
      │
      ▼
 ┌─────────────┐       ┌───────────────┐
 │  ScDoctor   │──────▶│ Hamming(7,4)  │
 │ (adaptive)  │       │     ECC       │
 └─────────────┘       └───────────────┘
      │
      ▼ (Go side)
 ┌──────────────────────────────────────────────────────────────┐
 │   HILServerDaemon  →  WebSocket  →  GUI / CI / remote viewer │
 └──────────────────────────────────────────────────────────────┘
```

- **Upstream inputs.** `SpikeTracer` wraps a
  :class:`sc_neurocore.edge.sc_network.SCNetwork` (see `edge.md`) and
  intercepts its `step_all`. `sc_scope` reads packed-u32 bitstreams
  straight from an FPGA transport.
- **Downstream consumers.** `LiveAnalyzer` emits per-layer stats
  consumed by `ScopeRenderer` (text CLI) or the Go WebSocket fan-out.
  `ScDoctor` outputs a new bitstream length that the SC pipeline's
  length-scheduler reads on its next cycle.

---

## 4. Features

- Execution-trace recorder with per-population slicing
  (:class:`ExecutionTrace.population_spikes`).
- Divergence detection with voltage-diff reporting
  (:class:`DivergencePoint`).
- Causal-chain reconstruction up to a fixed backward depth
  (:func:`causal_chain` returning :class:`CausalEvent` records).
- Full spike-diff summary (total mismatches, per-neuron mismatches,
  first-divergence record).
- Four transports: ``SIMULATED`` (deterministic for CI), ``UART``,
  ``JTAG``, ``PYNQ_DMA``.
- Timestamped samples with density, popcount, Shannon effective bits.
- Ring-buffer analysis windows (64–∞ samples) with mean + std density,
  mean effective bits, sample rate.
- Five trigger types: ``DENSITY_ABOVE``, ``DENSITY_BELOW``,
  ``SPIKE_DETECTED``, ``SCC_ABOVE``, ``ERROR_BUDGET_VIOLATION``.
- Per-layer error budget with pass-rate metric.
- Adaptive bitstream length with hysteresis + Hamming(7,4) ECC.
- Go WebSocket daemon with `/health` readiness probe.
- Live CLI text renderer (:class:`ScopeRenderer`).

---

## 5. Usage

### 5.1 Offline: find first divergence

```python
from sc_neurocore.debug.tracer import SpikeTracer
from sc_neurocore.debug.analyzer import find_divergence, spike_diff

tracer_a = SpikeTracer(network_a)
tracer_b = SpikeTracer(network_b)
trace_a = tracer_a.run(duration=0.1, dt=0.001, seed=42)
trace_b = tracer_b.run(duration=0.1, dt=0.001, seed=42)

dp = find_divergence(trace_a, trace_b)
if dp is None:
    print("traces identical")
else:
    print(f"first divergence at t={dp.timestep} n={dp.neuron_id}")
    print(f"voltage_diff={dp.voltage_diff:.4f}")

summary = spike_diff(trace_a, trace_b)
print(f"mismatch_rate={summary['mismatch_rate']:.4e}")
print(f"per_neuron worst: neuron {summary['per_neuron_mismatches'].argmax()}")
```

Sample output on a 1000×32 trace with one injected spike flip:

```
first divergence at t=500 n=7
voltage_diff=0.0000
mismatch_rate=3.1250e-05
per_neuron worst: neuron 7
```

### 5.2 Live: scope session with triggers

```python
from sc_neurocore.debug.sc_scope import (
    TransportConfig, TransportType, ScopeSession,
    TriggerCondition, TriggerType,
)

cfg = TransportConfig(transport_type=TransportType.SIMULATED)
scope = ScopeSession(transport_config=cfg, num_layers=4, window_size=512)
scope.trigger_engine.add_trigger(
    TriggerCondition(TriggerType.DENSITY_ABOVE, threshold=0.8, layer_id=2)
)
scope.start()
for _ in range(1000):
    scope.step(num_words=32)
print(scope.analyzer.all_stats())
print(f"{len(scope.trigger_engine.events)} triggered events")
scope.stop()
```

### 5.3 Adaptive doctor in a live loop

```python
from sc_neurocore.debug.sc_doctor import ScDoctor

doctor = ScDoctor(initial_length=512, target_precision=0.95)
for sample in stream:
    scc = measured_scc(sample)
    doctor.adapt(scc)
    new_len = doctor.current_bitstream_length
    if doctor.error_correction_enabled:
        encoded_word = doctor.encode_ecc(sample.data4)
```

### 5.4 HIL daemon

```python
from sc_neurocore.debug.hil_debugger import HILDebugger

dbg = HILDebugger(port=8081)
ok = dbg.start()
# now GUI connects to ws://localhost:8081
dbg.stop()
```

---

## 6. API reference

### 6.1 Offline tracer + analyser

| Symbol                       | Purpose                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| :class:`ExecutionTrace`      | $(T\times N)$ spikes/voltages/currents + population metadata         |
| :class:`SpikeTracer`         | wraps a network and records a trace over `(duration, dt)`            |
| :class:`DivergencePoint`     | $(t, n, s^A, s^B, V^A, V^B, |V^A - V^B|)$                            |
| :class:`CausalEvent`         | $(t, n, I, V, \text{spiked})$                                        |
| :func:`find_divergence`      | first $(t^{\star}, n^{\star})$ where $S^A \neq S^B$                  |
| :func:`spike_diff`           | full summary dict                                                    |
| :func:`causal_chain`         | $D$-depth backward spike trace                                       |

### 6.2 ScDoctor

| Symbol                              | Purpose                                                        |
| ----------------------------------- | -------------------------------------------------------------- |
| :class:`ScDoctor`                   | adaptive length + ECC controller                               |
| :meth:`ScDoctor.adapt`              | hysteresis control law on measured SCC                         |
| :meth:`ScDoctor.encode_ecc`         | Hamming(7,4) encode of 4-bit word                              |
| :meth:`ScDoctor.decode_ecc`         | single-error-correcting decode                                 |

### 6.3 sc_scope transports + samples

| Symbol                        | Purpose                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| :class:`TransportType`        | `JTAG`, `UART`, `PYNQ_DMA`, `SIMULATED`                          |
| :class:`TransportConfig`      | port / baud / DMA base / length / timeout                        |
| :class:`TransportBackend`     | `connect`, `disconnect`, `read_bitstream`                        |
| :class:`BitstreamSample`      | `timestamp_ns`, `layer_id`, `neuron_id`, `words`, `density`, `effective_bits` |
| :class:`AnalysisWindow`       | ring-buffer stats over recent samples                            |
| :class:`LiveAnalyzer`         | multi-layer ingest + per-layer stats                             |
| :class:`LayerErrorBudget`     | per-layer tolerance tracker with pass rate                       |
| :func:`compute_scc`           | reference SCC numerator                                          |

### 6.4 sc_scope triggers

| Symbol                       | Purpose                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| :class:`TriggerType`         | `DENSITY_ABOVE`, `DENSITY_BELOW`, `SPIKE_DETECTED`, `SCC_ABOVE`, `ERROR_BUDGET_VIOLATION` |
| :class:`TriggerCondition`    | `(type, threshold, layer_id, enabled)`                            |
| :class:`TriggerEvent`        | `(type, timestamp_ns, layer_id, measured_value, threshold, sample)` |
| :class:`TriggerEngine`       | `add_trigger`, `evaluate`                                         |
| :class:`ScopeSession`        | glues transport + analyser + triggers together                    |
| :class:`ScopeRenderer`       | text-mode CLI renderer                                            |

### 6.5 HIL daemon

| Symbol                       | Purpose                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| :class:`HILServerDaemon`     | low-level Go process supervisor                                   |
| :class:`HILDebugger`         | thin convenience wrapper (start / stop / is_running)              |

The wire format is `vision2030.telemetry.HILFrame` — see `proto.md`
for the exact protobuf schema.

---

## 7. Verified benchmarks

All figures produced by `benchmarks/bench_debug.py` (committed). Measured
on Ubuntu 24.04 / CPython 3.12.3 / Intel i5-11600K @ 3.90 GHz,
single-thread, 2026-04-20. Raw JSON at
`benchmarks/results/bench_debug.json`.

| Operation                                  | Throughput        | Latency     |
| ------------------------------------------ | ----------------- | ----------- |
| `find_divergence` (T=1000, N=32)           | 382 ops/s         | 2.61 ms     |
| `spike_diff` (T=1000, N=32)                | 379 ops/s         | 2.64 ms     |
| `causal_chain` (depth=10)                  | 33 030 ops/s      | 30.28 µs    |
| `ScDoctor.adapt` (Rust dispatch)           | 3.62 M ops/s      | 276.3 ns    |
| `ScDoctor.encode_ecc` (Rust dispatch)      | 5.52 M ops/s      | 181.1 ns    |
| `ScDoctor.decode_ecc` (Rust dispatch)      | 5.22 M ops/s      | 191.4 ns    |
| `LiveAnalyzer.layer_stats`                 | 41 473 ops/s      | 24.11 µs    |
| `compute_scc` (256 u32 words, Rust)        | 690 046 ops/s     | 1.45 µs     |
| `TriggerEngine.evaluate` (2 conditions)    | 55 991 ops/s      | 17.86 µs    |
| `LayerErrorBudget.check`                   | 4.95 M ops/s      | 202.0 ns    |

**Interpretation.**

- Offline trace walking dominates at 1000-step traces: ~2.6 ms per
  full divergence scan. At 10 000 steps this scales linearly to ~26 ms;
  still acceptable for a CI gate.
- `ScDoctor.adapt` sits at 276 ns via Rust — the compute itself takes
  ~30 ns, the rest is PyO3 tuple pack/unpack overhead. Pure Python is
  actually faster on this one (~85 ns) because the function is
  branch-only with no array work. The Rust dispatch is kept to satisfy
  the multi-language rule and because it stops being the bottleneck as
  soon as the caller wants to batch many samples (a future
  `py_sc_doctor_adapt_batch` entrypoint would amortise the FFI cost).
- `ScDoctor.encode_ecc` / `decode_ecc`: Rust wins 1.7× / 3.1× over
  pure Python (181 ns / 191 ns vs 314 ns / 591 ns). Decode benefits
  more because the syndrome arithmetic + error-correction branch is
  heavier per call.
- `compute_scc` now dispatches to `stochastic_doctor_core.py_scc_packed`
  (PyO3 bridge over the Rust `scc_packed` kernel) when the compiled
  extension is importable. Measured speedup against the pure-Python
  fallback on a 256-word input: **174×** (676 085 ops/s vs 3 875 ops/s);
  results are bit-exact identical (Δ = 0.00e+00). The Mojo SIMD kernel
  `scc_numerator_256w` (`mojo_accel.md` §4.2) stays unwired for this
  code path because its subprocess startup dominates per-call cost; it
  remains the option of choice for bulk offline analyses.
- Trigger evaluation at 17 µs scales with number of conditions; the
  per-condition cost is ~8 µs because the engine also checks
  `layer_id` and `enabled` per sample.

Figures above are `time.perf_counter` deltas from
`benchmarks/bench_debug.py`.

---

## 8. Citations

1. Alaghi A., Hayes J.P. (2013). *Exploiting correlation in stochastic
   circuit design*. ICCD-2013, 39–46. (SCC definition §1.4.)
2. Hamming R.W. (1950). *Error detecting and error correcting codes*.
   Bell System Technical Journal 29(2):147–160. (Hamming(7,4) §1.6.)
3. Cover T.M., Thomas J.A. (2006). *Elements of Information Theory*,
   2nd ed. Wiley. (Shannon entropy §1.3.)
4. Alaghi A., Qian W., Hayes J.P. (2018). *The promise and challenge
   of stochastic computing*. IEEE TCAD 37(8):1515–1531. (Adaptive
   length control §1.5.)
5. Esposito R., Sbrolli M. et al. (2018). *On-line debugging of neural
   network accelerators with self-referencing trace infrastructure*.
   IEEE Embedded Systems Letters. (Trace-based HW debugging.)
6. Šotek M. (2026). *SC-NeuroCore: live bitstream oscilloscope and
   adaptive ECC controller*. Internal report, ANULUM.

---

## 9. Known limitations

- **Offline divergence scan is O(TN).** The current
  :func:`find_divergence` short-circuits on the first mismatch but walks
  the arrays in Python loops. For very large traces (T > $10^{5}$, N >
  $10^{3}$) the call reaches multi-hundred-millisecond territory. A
  ``numpy.argmax(diff)`` variant would trade the first-mismatch metadata
  for a ~40× speedup and is tracked as a future optimisation.
- **Causal chain has no connectivity prior.** Without reading the
  network's weight matrix, :func:`causal_chain` returns *all* temporally
  preceding spikers at each depth step. A connectivity-aware variant
  would narrow the chain to neurons that *actually* project to the
  target; the arrays needed for that walk are already recorded by
  :class:`SpikeTracer`, but the routing table is not yet exposed.
- **Mojo SCC not auto-dispatched.** The Mojo SIMD kernel
  `scc_numerator_256w` (1.07 µs standalone) is not wired into the live
  scope path because the pixi subprocess overhead per call exceeds the
  Python fallback for single-SCC computation. Use the Mojo kernel for
  bulk offline correlation matrices, not for streaming scope frames.
- **Hamming(7,4) is not burst-tolerant.** A double-bit error inside one
  7-bit codeword is silently miscorrected; the code detects neither.
  For UART links with higher burst rates, consider interleaving the
  encoded blocks before transmission.
- **HIL daemon single-port, single-process.** One
  :class:`HILServerDaemon` binds one TCP port. Multi-FPGA setups need
  multiple daemons on distinct ports; there is no built-in multiplexer.
- **Offline tracer is synchronous.** It steps the network serially on
  the main thread; no GIL release and no GPU tracing. For GPU-resident
  networks, record from :class:`sc_neurocore.arcane_zenith.ArcaneZenithCognitiveCore`
  which has its own event-hooked trace path.
- **No encrypted transport.** The WebSocket feed is plaintext; fine for
  localhost + loopback, not for remote dashboards over untrusted
  networks. TLS wrap the port with a reverse proxy if needed.
- **Trigger engine is O(conditions × samples).** Each incoming sample
  re-evaluates every enabled trigger linearly. With the typical ~10
  conditions used in practice, throughput stays in the 100 k/s range;
  beyond ~100 conditions, group triggers by `layer_id` in a dict and
  dispatch per layer instead of scanning the full list.
- **Python-side tracer copies state.** `SpikeTracer.run` allocates
  `(T × N)` `int8`/`float64` arrays up-front. On a 32 GB host this is
  fine up to $T \cdot N \approx 4 \cdot 10^{9}$ float64 entries but
  needs a chunking strategy for longer recordings. No streaming-to-disk
  writer ships in the current module.
- **Rust coverage complete.** :func:`compute_scc`,
  :meth:`ScDoctor.adapt`, :meth:`ScDoctor.encode_ecc`, and
  :meth:`ScDoctor.decode_ecc` all now dispatch to the
  `stochastic_doctor_core` PyO3 extension with pure-Python fallback.
  Honest measured per-op result (see §7): `compute_scc` 174× faster,
  `encode_ecc` 1.7× faster, `decode_ecc` 3.1× faster, `adapt`
  ~3× slower because the FFI overhead dominates the trivial branch
  body. A future batch entrypoint will amortise the FFI for `adapt`.
- **Rust coverage gap on trace walking.** `find_divergence`,
  `spike_diff`, and `causal_chain` are pure-NumPy Python. Moving them
  to `spike_stats_core` would deliver a ~40× speedup (per §9 above)
  and is tracked as a future task, not a current blocker.

---

## Reference

- Sources:
  - `src/sc_neurocore/debug/tracer.py` (155 LOC)
  - `src/sc_neurocore/debug/analyzer.py` (176 LOC)
  - `src/sc_neurocore/debug/sc_doctor.py` (101 LOC)
  - `src/sc_neurocore/debug/sc_scope.py` (538 LOC)
  - `src/sc_neurocore/debug/hil_server.py` (124 LOC)
  - `src/sc_neurocore/debug/hil_debugger.py` (34 LOC)
  - `src/sc_neurocore/debug/hil_client.py` (353 LOC)
- Go daemon: `src/sc_neurocore/accel/go/services/hil_debugger/`
  (`main.go` + `main_test.go`).
- Tests: `tests/test_debug/*.py` (1 781 LOC across 5 files).
- Benchmark: `benchmarks/bench_debug.py`.
- Wire protocol: [Protobuf Schemas](proto.md).

::: sc_neurocore.debug.tracer
    options:
      show_root_heading: true

::: sc_neurocore.debug.analyzer
    options:
      show_root_heading: true

::: sc_neurocore.debug.sc_doctor
    options:
      show_root_heading: true

::: sc_neurocore.debug.sc_scope
    options:
      show_root_heading: true

::: sc_neurocore.debug.hil_server
    options:
      show_root_heading: true

::: sc_neurocore.debug.hil_debugger
    options:
      show_root_heading: true
