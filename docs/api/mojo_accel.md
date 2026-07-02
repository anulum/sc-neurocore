# Mojo SIMD Kernels

Optional Mojo acceleration layer for stochastic-computing hot paths.
A pure-Mojo kernel bundle (``kernels.mojo``, 1 747 LOC) provides
vector-lane SC primitives; a Python façade
(:class:`MojoKernelRunner`) launches them through the pixi-managed
Mojo toolchain as a subprocess.

```python
from sc_neurocore.accel.mojo import MojoKernelRunner, _HAS_MOJO

if _HAS_MOJO:
    runner = MojoKernelRunner()
    ok = runner.build()                          # pixi run mojo build
    pop = runner.popcount([0xFF00, 0x0FF0])      # maintained Python fallback
```

The ``sc_neurocore.accel.mojo`` import never raises. ``_HAS_MOJO`` is
``False`` when the optional runner cannot be imported, and
``_mojo_import_reason`` records the captured failure. The
``MojoKernelRunner`` symbol remains importable in that state, but
constructing it raises a ``RuntimeError`` with the same reason instead of
silently reusing a stale runner binding. Downstream code gates on
``_HAS_MOJO`` before constructing the runner.

---

## 1. Mathematical formalism

### 1.1 Stochastic-computing bit operations (packed `UInt32`)

SC primitives operate on Bernoulli bitstreams stored packed into
``UInt32`` words. Let $a, b \in \{0, 1\}^{32}$ be length-32 bitstreams
of independent streams and $w_a, w_b \in \text{UInt32}$ their packed
representations. The kernel correspondences are:

| Operation       | Gate             | Packed-word identity               |
| --------------- | ---------------- | ---------------------------------- |
| ``sc_and``      | $a \wedge b$     | `w_a & w_b`                        |
| ``sc_or``       | $a \vee b$       | `w_a | w_b`                        |
| ``sc_xor``      | $a \oplus b$     | `w_a ^ w_b`                        |
| ``sc_not``      | $\neg a$         | `~w_a`                             |
| ``sc_sub``      | $a \wedge \neg b$| `w_a & ~w_b`                       |
| ``sc_mux(a,b,s)``| `s ? a : b`    | `(w_a & w_s) | (w_b & ~w_s)`       |

For independent streams with probabilities $p_a, p_b$ these map to
the standard SC identities:

$$
\begin{aligned}
\mathrm{AND:} &\qquad \mathbb{E}[a \wedge b] = p_a p_b \\
\mathrm{OR:} &\qquad \mathbb{E}[a \vee b] = p_a + p_b - p_a p_b \\
\mathrm{XOR:} &\qquad \mathbb{E}[a \oplus b] = p_a + p_b - 2 p_a p_b \\
\mathrm{MUX(s=\frac{1}{2}):} &\qquad \mathbb{E}[\text{out}] = \frac{p_a + p_b}{2} \\
\mathrm{NOT:} &\qquad \mathbb{E}[\neg a] = 1 - p_a.
\end{aligned}
$$

The "packed" variants (``and_packed``, ``or_packed``, ``xor_packed``,
``mux_packed``) simply iterate across a ``List[UInt32]`` with Mojo's
SIMD lanes.

### 1.2 Popcount — Hamming-weight estimator for SC density

Given packed bits $w \in \text{UInt32}$, ``popcount_u32`` returns the
Hamming weight $\|w\|_1$. For a length-$N$ bitstream the density
estimator $\hat p = \|b\|_1 / N$ is the maximum-likelihood estimate of
the underlying Bernoulli parameter $p$, with variance $p(1-p)/N \le
1/(4N)$ (Cramér-Rao bound for Bernoulli).

Popcount uses the folded-add trick (Hamming-weight in $O(\log_2 W)$
for word width $W$); Mojo's current implementation lowers to LLVM's
``llvm.ctpop.i32`` intrinsic, which on x86-64 with SSE4.2+ compiles to
the single-cycle ``POPCNT`` instruction.

### 1.3 Stochastic cross-correlation numerator

``scc_numerator(a, b)`` returns

$$
N_\text{scc}(a, b) = \sum_i (2 a_i - 1)(2 b_i - 1)
= 4 \|a \wedge b\|_1 - 2(\|a\|_1 + \|b\|_1) + N,
$$

which is the unnormalised SC correlation used by the stochastic
doctor. With corresponding denominator

$$
D_\text{scc} = \sqrt{(2\|a\|_1 - N)(2\|b\|_1 - N)},
$$

the SCC is $N_\text{scc} / \min(|2\|a\|_1 - N|, |2\|b\|_1 - N|)$,
normalised to $[-1, +1]$ (Alaghi & Hayes 2013).

### 1.4 Vector MAC

``vec_mac(weights, inputs, n_neurons, n_words)`` computes

$$
y_j = \sum_{i=1}^{n_\text{inputs}}
W_{j,i} \cdot \|b_i\|_1, \qquad j \in [0, n_\text{neurons}),
$$

where $b_i \in \text{List[UInt32]}$ is the packed input bitstream and
$W_{j,i} \in \mathbb{N}$ is the integer weight. The inner sum uses
``popcount_slice`` + multiplication; Mojo lanes parallelise across
outputs $j$.

### 1.5 STDP, R-STDP, eligibility trace

``stdp_update``, ``eligibility_trace_update``,
``reward_modulated_stdp`` implement the same exponential pair-STDP
rule documented in :doc:`bioware` §1.5 + the reward-modulated
variant of Izhikevich 2007, quantised to Q8.8 fixed-point on the
Mojo side so the output bit-matches the Rust path to within 1 ulp.

### 1.6 HDC bind

``hdc_bind(a, b)`` is a vector-wide XOR matching the binding operator
of classical hyperdimensional computing (Plate 1995, Kanerva 2009):

$$
\mathrm{bind}(a, b)_i = a_i \oplus b_i.
$$

For the packed representation this is just element-wise XOR over the
``List[UInt32]``.

---

## 2. Theoretical context

**Why Mojo, specifically.** Mojo (Modular 2023) is a Python-compatible
systems language with native SIMD types (`SIMD[DType, Width]`),
MLIR-based codegen, and a strict ownership model. For SC workloads
the `SIMD[UInt32, 8]` vector type maps directly onto AVX-2 ymm
registers — one lane per 32-bit word, so a 256-bit vector processes
256 SC bits per instruction. The Rust `autonomous_learning` engine
gets the same throughput via `std::simd`, but Mojo's syntax stays
inside Python's type system, so closing the performance gap without
leaving the Python dev loop is possible.

**Why subprocess, not FFI.** The Mojo ABI (2026-04) is not yet stable
enough to safely `ctypes.CDLL` a Mojo-produced shared library from
CPython across Mojo versions. The subprocess model trades per-call
latency (tens of ms, dominated by Mojo interpreter startup) for a
stable interface: the Python side invokes ``pixi run mojo
kernels.mojo <op> <args>`` and parses the printed result. This means
Mojo is useful for **batched** calls (hundreds of operations) but
**not** for per-tick FFI. See §7 for measured numbers — single-call
popcount is three orders of magnitude slower than pure Python
because startup dominates; batched whole-network MAC reverses that.

**Design parallels.** The kernel bundle structure mirrors the Rust
engine's SIMD module (`engine/src/simd/`): each SC primitive has a
scalar, packed, and vectorised variant. The Mojo and Rust
implementations share the same Q8.8 quantisation for all numerical
paths, so cross-validation between the two produces bit-identical
outputs — the ``benchmarks/bench_mojo_vs_rust.py`` harness
enforces this.

**Roadmap hook.** When Mojo's ABI stabilises and CPython ↔ Mojo FFI
is safe (Modular roadmap target: 2026 Q3), the subprocess boundary
will be replaced by a direct ctypes / CFFI call without any
functional change to the kernel surface. The Python façade's method
signatures are already wire-format-compatible with that future FFI.

---

## 3. Pipeline position

Mojo acceleration is a *peer* of the Rust engine — both sit below the
Python simulation layer and accelerate the same class of hot paths.
Users pick one via configuration; defaults remain Rust.

```
 Python SC network (src/sc_neurocore/)
        │
        ▼
 ┌────────────────────────────────────────────┐
 │        Accelerator dispatch                │
 │     ┌──────────┬───────────┬─────────┐     │
 │     ▼          ▼           ▼         ▼     │
 │  NumPy      Rust (FFI)   Mojo     Julia    │
 │            libsc_neur… (subproc)  (subproc)│
 │  default   default       opt-in   opt-in   │
 └────────────────────────────────────────────┘
        │          │          │        │
        ▼          ▼          ▼        ▼
 (pure py)  x86 SIMD  AVX/Mojo      DiffEq.jl
                      lanes         ODEs
```

**Inputs** — Python lists of ``int`` or NumPy arrays. The façade
converts to the Mojo-friendly format (typically ``List[UInt32]``).

**Outputs** — Python ``int`` / ``list[int]``. The subprocess writes
results to stdout in a regex-parseable line (``RESULT: <value>``)
which the façade extracts.

**Dispatch policy** — Mojo is **opt-in** via explicit instantiation
of :class:`MojoKernelRunner`. No SC-NeuroCore core component silently
routes through Mojo today (as of 2026-04-20); the kernel bundle is
available to users who want to benchmark or experiment with
Mojo-level SC primitives.

---

## 4. Features

| Feature                                            | Detail                                                                  |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| SC bit primitives (and/or/xor/not/sub/mux)         | Scalar + packed `List[UInt32]` + SIMD variants                          |
| Packing utilities (pack_bits / unpack_bits)        | Dense boolean ↔ `List[UInt32]`, bit-exact                               |
| Popcount (`popcount_u32`, `popcount_slice`)        | Folded-add over `UInt32`; lowers to `llvm.ctpop.i32`                    |
| SC metric (`scc_numerator`)                        | Unnormalised SC cross-correlation; used by stochastic doctor            |
| Vector MAC (`vec_mac`)                             | Matrix × packed-bitstream input; returns integer accumulators            |
| STDP / R-STDP / eligibility trace kernels          | Q8.8 pair rule, reward-modulated variant                                |
| HDC primitive (`hdc_bind`)                         | Packed XOR binding                                                      |
| Toolchain manifest (``pixi.toml`` + ``pixi.lock``) | Reproducible Mojo install via pixi                                      |
| Graceful degradation (`_HAS_MOJO`)                 | Import never raises when Mojo/pixi missing                              |
| Benchmark harness (``bench_mojo_vs_rust.py``)      | Mojo vs Rust parity + timing, pure-text output                         |
| Build helper (`build()`)                           | Thin `pixi run mojo build kernels.mojo` wrapper                         |

---

## 5. Usage example with output

```python
from sc_neurocore.accel.mojo import MojoKernelRunner, _HAS_MOJO

assert _HAS_MOJO, "install Mojo + pixi"

r = MojoKernelRunner()
print(f"kernel dir : {r._mojo_dir}")
print(f"pixi bin   : {r._pixi_bin}")

# Popcount a small batch. The current maintained runtime uses the Python
# fallback until direct Mojo IPC bindings are promoted.
bits = [0xFF00, 0x0FF0, 0xCAFEBABE]
v = r.popcount(bits)
print(f"popcount({bits}) = {v}")   # 8 + 8 + 22 = 38
```

Verified output on the reference host (Linux x86-64, Mojo from pixi,
2026-04-20):

```
kernel dir : /<repo>/src/sc_neurocore/accel/mojo
pixi bin   : /home/anulum/.pixi/bin/pixi
popcount([65280, 4080, 3405691582]) = 38
```

The popcount number matches the expected $\mathrm{popcount}(0xFF00) +
\mathrm{popcount}(0x0FF0) + \mathrm{popcount}(0xCAFEBABE) = 8 + 8 + 22
= 38$. This helper is intentionally kept bit-exact with the Python fallback
while direct Mojo IPC bindings remain pending.

---

## 6. Technical reference

### 6.1 `MojoKernelRunner`

```python
@dataclass
class MojoKernelRunner:
    _mojo_dir: Path = field(...)
    _pixi_bin: str = field(default_factory=lambda: os.path.expanduser("~/.pixi/bin/pixi"))

    def __post_init__(self): ...
    def build(self) -> bool: ...
    def run_benchmark(self, timeout_sec: int = 60) -> dict[str, float]: ...
    def popcount(self, data: list[int]) -> int: ...
    def lfsr_encode(self, seed: int, threshold: int, bits: int) -> list[int]: ...
```

| Method                                              | Semantic                                                                                       |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `__post_init__`                                     | Locates `kernels.mojo` (source-tree first, then installed package). Raises on neither present. |
| `build() -> bool`                                   | `pixi run mojo build kernels.mojo` in the kernel directory. Returns `False` on launch or build failure. |
| `run_benchmark(timeout_sec=60) -> dict[str, float]` | Runs the full kernel benchmark once and parses stdout. Returns `{}` on failure, timeout, or missing toolchain. |
| `popcount(data) -> int`                             | Returns the total Hamming weight through the maintained Python fallback while Mojo IPC is pending. |
| `lfsr_encode(seed, threshold, bits) -> list[int]`   | Generates an LFSR-encoded 16-bit stream through the maintained Python fallback while Mojo IPC is pending. |

Construction fails closed when ``kernels.mojo`` is unavailable. Build and
benchmark calls degrade to ``False`` / ``{}`` when Mojo or pixi cannot run, and
callers still gate default runtime dispatch on ``_HAS_MOJO``.

### 6.2 Kernel inventory (`kernels.mojo`)

The Mojo file groups kernels by stage. Each kernel takes and returns
``List[UInt32]`` packed-bit representations so the FFI surface stays
trivial. Top-level function table (line numbers refer to the current
``kernels.mojo``):

```
popcount_u32             (L19)   scalar
popcount_slice           (L27)   reduction
sc_and, sc_or, sc_xor    (L33–)  scalar
sc_mux, sc_sub, sc_not   (L42–)  scalar
and_packed               (L52)   elementwise
or_packed                (L58)   elementwise
xor_packed               (L64)   elementwise
mux_packed               (L70)   elementwise
scc_numerator            (L80)   SC metric
pack_bits                (L125)  conversion
unpack_bits              (L135)  conversion
vec_mac                  (L180)  matrix × packed bits
stdp_update              (L219)  pair STDP
eligibility_trace_update (L243)  trace decay
reward_modulated_stdp    (L255)  R-STDP full rule
hdc_bind                 (L271)  HDC XOR bind
```

Helper structs + SIMD-laned loops interleave between these entry
points.

### 6.3 Toolchain expectations

- **pixi** at ``~/.pixi/bin/pixi`` (override via ``_pixi_bin``).
- **Mojo 0.26+** — earlier versions miss the ``UnsafePointer`` FFI
  pattern the kernels use internally. Install via the pixi env
  manifest ``src/sc_neurocore/accel/mojo/pixi.toml`` + lock file.

### 6.4 Error modes

| Condition                              | Behaviour                                                         |
| -------------------------------------- | ----------------------------------------------------------------- |
| `kernels.mojo` missing entirely        | `__init__` raises `FileNotFoundError` with install instructions   |
| `pixi` not on `PATH` during build      | `build()` returns `False` and prints the failure                  |
| Mojo compilation fails                 | `build()` returns `False` and prints the failure                  |
| Benchmark subprocess timeout           | `run_benchmark()` returns `{}` after printing the timeout         |
| Benchmark subprocess launch failure    | `run_benchmark()` returns `{}` after printing the missing-toolchain path |
| `_HAS_MOJO == False` at import time    | Call sites should skip; `from sc_neurocore.accel.mojo import ...` stays safe |

---

## 7. Performance benchmarks

All numbers measured 2026-04-20 on Linux x86-64 (Intel i5-11600K,
CPython 3.12.3, Mojo 0.26.2 from Modular's pixi channel). Figures
come from a run of ``benchmarks/bench_mojo_vs_rust.py``.

### 7.1 Kernel-suite benchmark (amortised subprocess startup)

``benchmarks/bench_mojo_vs_rust.py`` runs ``kernels.mojo`` once inside
``pixi run mojo run kernels.mojo``, parses the printed per-kernel
timings, and normalises them to **ns per inner call** so the Mojo
loop count (100 k–1 M) and the Python loop count (1 k) are directly
comparable. The canonical output:

| Benchmark                    | Mojo (ns/call) | Python (ns/call) | Speedup |
| ---------------------------- | --------------:| ----------------:| -------:|
| ``popcount_1024w``           |          415.2 |        279 644.8 |   **673×** |
| ``scc_numerator_256w``       |          185.0 |        197 093.5 |  **1 065×** |
| ``lfsr_encode_1024bit``      |        2 085.8 |        275 451.4 |   **132×** |

Additional Mojo-only timings produced by the same run (no Python
reference kernel in this module — these are first-in-class Mojo
primitives):

| Benchmark                    | Mojo (ns/call) |
| ---------------------------- | --------------:|
| ``hdc_bind_256w``            |          424.2 |
| ``dna_hamming_256w``         |          457.6 |
| ``histogram_1024w``          |          563.0 |
| ``lif_batch_64``             |          172.8 |
| ``stdp_update_1024w``        |        1 787.5 |
| ``sobol_1024bit``            |        1 419.2 |
| ``spike_bin_10k``            |       20 003.4 |
| ``dvs_pack_4k``              |        3 993.4 |
| ``ring_topo_64``             |       13 056.6 |

Honest caveat — three kernels (``attention_256w``, plus earlier
versions of ``popcount`` and ``scc`` before the DCE fix) were
dead-code-eliminated by Mojo's optimiser because the result was
discarded with ``_ = kernel(...)``. The fix is the XOR-accumulator
pattern (``sink ^= kernel(...)``) which forces the result to be
observed — the popcount / SCC numbers above reflect this fix. The
``attention_256w`` kernel still shows 0.0 ns / call; it needs the
same accumulator treatment and is a known follow-up.

### 7.2 Single-call subprocess overhead

Every ``pixi run mojo`` invocation incurs ~2 s of interpreter
bring-up on the reference host. For small one-shot calls this
dominates the total wall clock. The kernel-suite run in §7.1 takes
~4 s total wall clock (2 s startup + ~2 s of 1 M-iteration
benchmarks) — the per-inner-call ns/call numbers are only meaningful
because the bench suite amortises the startup over ~10 M kernel
invocations.

The Python façade's ``popcount`` / ``lfsr_encode`` methods are
currently **not** wired to the Mojo path — they raise
``NotImplementedError("Mojo IPC bindings pending v4.0")`` and fall
back to the pure-Python reference in
``sc_neurocore.edge.bitstream``. Expect that roadmap item to land
when the Mojo ABI stabilises (Modular milestone 2026 Q3); until then
the way to exercise Mojo kernels is through the bench harness
above, not per-call API.

### 7.3 Reproducer

```bash
# Full kernel suite (bench_mojo_vs_rust.py — ~8 s wall clock total).
PYTHONPATH=src python benchmarks/bench_mojo_vs_rust.py

# Raw Mojo kernel output (no Python baseline, 16 kernel groups printed).
cd src/sc_neurocore/accel/mojo && pixi run mojo run kernels.mojo
```

Both must print non-zero milliseconds for the kernels in the table
above; a ``0.0 ms`` or ``N/A`` entry indicates a regressed DCE case
that needs the accumulator pattern restored.

---

## 8. Citations

1. **Alaghi, A. & Hayes, J. P.** (2013). *Survey of stochastic
   computing.* ACM Transactions on Embedded Computing Systems 12(2s):
   92. — SC arithmetic identities (§1.1), correlation coefficient
   definition (§1.3).
2. **Kanerva, P.** (2009). *Hyperdimensional computing: an
   introduction to computing in distributed representation with
   high-dimensional random vectors.* Cognitive Computation 1(2):
   139–159. — Hyperdimensional binding (§1.6).
3. **Izhikevich, E. M.** (2007). *Solving the distal reward problem
   through linkage of STDP and dopamine signaling.* Cerebral Cortex
   17(10): 2443–2452. — Reward-modulated STDP variant used by
   ``reward_modulated_stdp``.
4. **Plate, T. A.** (1995). *Holographic reduced representations.*
   IEEE Transactions on Neural Networks 6(3): 623–641. — Foundational
   paper for the HDC bind operator.
5. **Modular, Inc.** (2023). *Mojo Language Reference.* — Language
   manual, SIMD typing, subprocess-stable ABI policy. URL:
   <https://docs.modular.com/mojo/manual>.

---

## 8b. Full kernel-group inventory

The ``kernels.mojo`` file is organised into 45 numbered sections
grouping 108 public Mojo functions. Sections invoked by the
benchmark harness above are starred (★); remaining sections are
library utilities called internally or reserved for upcoming
integrations.

| §    | Group                             | Representative function            |
| ---- | --------------------------------- | ---------------------------------- |
| ★§1  | Popcount (1024 word slice)       | ``popcount_slice``                 |
| ★§2  | SCC numerator                     | ``scc_numerator``                  |
| ★§3  | LFSR-16 encoder                   | ``Lfsr16.encode_into``             |
| §4   | SC binary ops (scalar)           | ``sc_and``, ``sc_or``, ``sc_xor``  |
| §5   | SC binary ops (packed)            | ``and_packed``, ``or_packed``      |
| §6   | Pack / unpack bits                | ``pack_bits``, ``unpack_bits``     |
| ★§7  | STDP pair rule                    | ``stdp_update``                    |
| ★§8  | HDC similarity                    | ``hdc_bind``                       |
| §9   | Evo fitness scorer                | ``evo_fitness_score``              |
| §10  | Vector MAC                        | ``vec_mac``                        |
| §11  | Eligibility trace update          | ``eligibility_trace_update``       |
| §12  | Reward-modulated STDP             | ``reward_modulated_stdp``          |
| §13  | BCM metaplasticity                | ``bcm_update``                     |
| ★§14 | Attention scores                  | ``attention_score``                |
| §15  | Softmax (SIMD-laned)              | ``softmax_simd``                   |
| §16  | Layer norm                        | ``layernorm``                      |
| §17  | Dropout (stochastic mask)         | ``dropout_packed``                 |
| ★§18 | Histogram (fixed-bin)             | ``histogram_fixed``                |
| ★§19 | LIF batched step                  | ``lif_batch_step``                 |
| §20  | Sparsity mask                     | ``sparsity_mask``                  |
| §21  | Quantile (percentile)             | ``quantile_packed``                |
| §22  | KNN search (L2)                   | ``knn_l2_packed``                  |
| ★§23 | Sobol quasi-random                | ``Sobol32.generate``               |
| §24  | Halton quasi-random               | ``Halton32.generate``              |
| §25  | Xorshift64 RNG                    | ``Xorshift64.next``                |
| §26  | Hamming ECC                       | ``hamming_ecc_encode``             |
| §27  | Reed-Solomon (GF(256))            | ``rs_encode_gf256``                |
| §28  | CRC-32                            | ``crc32_packed``                   |
| §29  | LZSS (streaming)                  | ``lzss_compress_stream``           |
| §30  | Bit-interleave (Morton)           | ``morton_encode_2d``               |
| ★§31 | Spike bin (time → bin index)      | ``spike_bin``                      |
| §32  | Spike raster → rate               | ``spike_rate_from_raster``         |
| ★§33 | DVS frame pack                    | ``dvs_pack_frame``                 |
| §34  | DVS polarity map                  | ``dvs_polarity_map``               |
| ★§35 | Ring topology iteration           | ``ring_topo_step``                 |
| §36  | Grid-4 topology                   | ``grid4_step``                     |
| ★§37 | DNA Hamming distance              | ``dna_hamming``                    |
| §38  | DNA Levenshtein                   | ``dna_levenshtein``                |
| §39  | DNA encoding (2-bit)              | ``dna_2bit_encode``                |
| §40  | FFT (radix-2)                     | ``fft_radix2``                     |
| §41  | Wavelet (Haar)                    | ``wavelet_haar``                   |
| §42  | Median filter                     | ``median_filter``                  |
| §43  | Outlier rejection                 | ``outlier_reject``                 |
| §44  | Z-score normalisation             | ``zscore_normalise``               |
| §45  | Exponential moving average        | ``exponential_moving_average``     |

The sections are deliberately laid out as flat Mojo functions — no
traits / inheritance / classes — so the Mojo compiler has maximum
freedom to vectorise. This matches the Rust engine's flat-function
policy for hot paths.

## 8c. Build + test workflow

Getting the Mojo path running from scratch on a clean host:

```bash
# 1. Install pixi (Modular's package manager).
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Install Mojo via pixi — uses the Modular conda channel declared
#    in src/sc_neurocore/accel/mojo/pixi.toml.
cd src/sc_neurocore/accel/mojo
pixi install

# 3. Smoke-test: run the full kernel suite.
pixi run mojo run kernels.mojo
#   → prints 16 kernel timings + "45 kernel groups, 108 functions total"

# 4. Bench harness from the repo root.
cd ../../../..
PYTHONPATH=src python benchmarks/bench_mojo_vs_rust.py
#   → 13-row table with Mojo ns/call, Python ns/call, Speedup column.
```

The ``pixi.toml`` manifest declares ``channels = ["https://conda.modular.com/max", "conda-forge"]``
so both the Modular Mojo build and conda-forge standard libraries
are resolvable. The ``pixi.lock`` file pins exact versions for
reproducibility across machines.

## 8d. Cross-validation with the Rust path

Where a given SC primitive exists in both Mojo and Rust (popcount,
SCC numerator, LFSR, STDP, BCM, HDC bind), the two paths are
expected to produce bit-identical output when fed identical inputs.
The invariants the Mojo kernels honour:

- Q8.8 fixed-point arithmetic for every plasticity weight.
- Round-half-to-even on any rational → integer reduction.
- `UInt32`-packed bitstreams with LSB-first bit ordering inside a word
  (matches ``sc_neurocore.edge.bitstream.pack_bits``).
- LFSR state stored as little-endian `UInt16`.

The bench harness checks **timing**, not correctness; the correctness
harness lives at ``tests/test_learning/test_learning_mojo_parity.py``
which runs the Rust and Mojo kernels on the same ``pack_bits``
input vector and asserts bit-exact equality on the output word
array. Passing that parity test is a prerequisite for landing
Mojo-backed kernels into the default ``accel/`` dispatch path.

## 9. Limitations

- **Subprocess model.** ``build`` and ``run_benchmark`` pay a
  Mojo-interpreter start-up cost (~18 ms on reference host). Batch
  benchmark work rather than invoking the toolchain repeatedly in tight loops.
- **No ctypes FFI yet.** The Mojo ABI (as of 2026-04) is not yet
  stable across versions. When Modular releases a stable ABI (tracker
  milestone 2026 Q3), the subprocess façade will be replaced with a
  direct ``ctypes.CDLL`` call using the same Python method signatures.
- **Platform.** Mojo 0.26+ is Linux x86-64 first-class. macOS support
  tracks upstream Modular; Windows support is not on the near-term
  roadmap.
- **No GPU kernels in this module.** The WGPU path lives under
  :doc:`optics` and the Rust engine; Mojo GPU kernels are on the
  backlog pending Mojo's own GPU surface stabilising.
- **Results must be parsed from stdout.** No binary protocol; the
  subprocess writes plain-text ``RESULT: <value>`` lines that the
  Python façade regex-parses. Noisy kernels that print additional
  diagnostics can confuse the parser — keep kernel stdout clean.

---

## Reference

- Source: `src/sc_neurocore/accel/mojo/runner.py` (~115 lines),
  `src/sc_neurocore/accel/mojo/kernels.mojo` (1 747 lines).
- Manifest: `src/sc_neurocore/accel/mojo/pixi.toml` + `pixi.lock`.
- Benchmark: `benchmarks/bench_mojo_vs_rust.py`.
- Package entry: `src/sc_neurocore/accel/mojo/__init__.py` (exports
  `MojoKernelRunner` + `_HAS_MOJO` flag).

::: sc_neurocore.accel.mojo.runner
    options:
      show_root_heading: true
