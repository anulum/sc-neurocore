# Mojo SIMD Kernels

Optional Mojo acceleration layer for the stochastic computing hot
paths. A pure-Mojo kernel bundle (``kernels.mojo``) provides
vector-lane SC primitives; a thin Python wrapper
(:class:`MojoKernelRunner`) launches it through the pixi-managed Mojo
toolchain as a subprocess.

```python
from sc_neurocore.accel.mojo import MojoKernelRunner, _HAS_MOJO

if _HAS_MOJO:
    runner = MojoKernelRunner()
    ok = runner.build()              # pixi run mojo build
    pop = runner.popcount([0xFF00, 0x0FF0])
```

The import never raises — ``_HAS_MOJO`` is ``False`` when the runner
cannot be constructed (missing Mojo / pixi / kernel source). Downstream
code should gate on that flag.

---

## 1. `MojoKernelRunner`

Thin subprocess wrapper around `~/.pixi/bin/pixi` + the Mojo toolchain.

| Method                                            | Purpose                                                                                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `__init__()`                                      | Locates `kernels.mojo` — source-tree first, then installed package data. Raises if neither.    |
| `build() -> bool`                                 | Runs `pixi run mojo build kernels.mojo` in the kernel directory. Returns success flag.         |
| `run_benchmark(timeout_sec=60) -> Dict[str,float]`| Runs the built kernel bench, parses stdout, returns per-stage timing map.                      |
| `popcount(data: list[int]) -> int`                | Crosses the FFI boundary once to invoke `popcount_slice` on a packed `List[UInt32]`.           |
| `lfsr_encode(seed, threshold, bits) -> list[int]` | Generates a deterministic LFSR-encoded bitstream (length = ``bits``) for a given threshold.    |

All methods degrade gracefully when Mojo / pixi are absent — the
``_HAS_MOJO`` flag must be checked before use.

---

## 2. Kernel inventory — `kernels.mojo`

The Mojo file groups kernels by stage. Each kernel takes and returns
`List[UInt32]` packed-bit representations so the FFI surface stays
trivial.

### 2.1 Bit-level SC operators

``sc_and``, ``sc_or``, ``sc_xor``, ``sc_mux(a, b, sel)``, ``sc_sub``,
``sc_not`` — single-word SC primitives (unipolar). ``sc_mux`` selects
``a`` or ``b`` bit-wise based on ``sel``, matching the hardware
multiplexer pattern.

### 2.2 Packed variants

``and_packed``, ``or_packed``, ``xor_packed``, ``mux_packed`` — apply
the above element-wise over a `List[UInt32]`. All honour SIMD lanes
where Mojo can vectorise.

### 2.3 SC metric kernels

- ``popcount_u32(val)`` — Brian-Kernighan popcount on a single word.
- ``popcount_slice(data)`` — total population count of a packed array.
- ``scc_numerator(a, b)`` — stochastic cross-correlation numerator
  (pairwise co-activity) used by the SC doctor.

### 2.4 Packing utilities

- ``pack_bits(bits, n_bits)`` — dense boolean → `List[UInt32]`.
- ``unpack_bits(packed, n_bits)`` — inverse operation, bit-exact.

### 2.5 Neural kernels

- ``vec_mac(weights, inputs, n_neurons, n_words)`` — matrix-vector
  MAC over packed bit inputs; returns per-neuron integer accumulator.
- ``stdp_update(...)`` — Spike-Timing Dependent Plasticity weight
  update, pair-based.
- ``eligibility_trace_update(...)`` — eligibility-trace decay used by
  reward-modulated STDP (R-STDP).
- ``reward_modulated_stdp(...)`` — full R-STDP rule combining pair
  STDP with a global reward signal.

### 2.6 HDC primitive

- ``hdc_bind(a, b)`` — XOR-based binding of two hyperdimensional
  vectors (packed bits).

---

## 3. Benchmark harness

`benchmarks/bench_mojo_vs_rust.py` drives :class:`MojoKernelRunner`
side-by-side with the `sc_neurocore_engine` Rust path on the same
inputs. Output is a pure-text timings table so results can be tracked
in version control without binary artefacts.

---

## 4. Toolchain expectations

- pixi at ``~/.pixi/bin/pixi`` (override via ``_pixi_bin`` field on
  :class:`MojoKernelRunner` at construction).
- Mojo 0.26+ (earlier versions miss the ``UnsafePointer`` FFI pattern
  used by the kernels). Installation via pixi environment described
  in ``src/sc_neurocore/accel/mojo/pixi.toml``.

---

## 5. Limitations

- Subprocess-based — every invocation pays pixi + Mojo start-up cost.
  Batch operations through ``run_benchmark`` rather than per-call
  `popcount` invocations for tight loops.
- No differentiable / autograd integration — kernels are pure-inference
  SC primitives.
- Kernels are Linux x86_64 first-class; Mojo macOS / Windows support
  tracks upstream Modular.

---

## Reference

- Source: `src/sc_neurocore/accel/mojo/runner.py`,
  `src/sc_neurocore/accel/mojo/kernels.mojo`.
- Benchmark: `benchmarks/bench_mojo_vs_rust.py`.
- Toolchain manifest: `src/sc_neurocore/accel/mojo/pixi.toml` + `pixi.lock`.

::: sc_neurocore.accel.mojo.runner
    options:
      show_root_heading: true
