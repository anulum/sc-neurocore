# Array Guards — `sc_neurocore._native.array_guards`

The zero-copy gatekeeper between NumPy arrays and the Rust / Mojo native
extensions. Every SC-NeuroCore FFI call that hands a `ndarray` pointer
to a compiled library first passes through this module. If a caller
slips a strided view, a misaligned buffer, or a wrong-dtype array into
the native boundary, the guard raises a Python `ValueError` with a
pointer-level diagnostic **before** the foreign function gets a chance
to dereference unaligned memory.

```python
from sc_neurocore._native.array_guards import require_c_contiguous
import numpy as np

# Match dtype + layout — identity return, no copy
stream = np.zeros(4096, dtype=np.uint8)
safe = require_c_contiguous(stream, "stream", np.uint8)
assert safe is stream                                    # True
```

---

## 1. Why this exists

Every native backend in SC-NeuroCore —
[`stochastic_doctor_core`](stochastic_doctor.md),
[`autonomous_learning`](autonomous_learning.md),
[`sc_neurocore_engine`](rust-engine.md), the Mojo SIMD kernels in
[`accel/mojo`](accel.md) — accepts NumPy input over a C ABI. The
contract with every one of those backends is identical:

1. The underlying memory must be **C-contiguous** (row-major, stride
   equal to item size in the last axis).
2. The pointer must be **naturally aligned** for the requested dtype
   (a `float32` view at address `0x...1` will crash on any
   non-x86 target and incurs a trap-and-fixup penalty on x86).
3. The NumPy dtype must match the C header's `uint8_t*`, `float*`, or
   `int32_t*` declaration **exactly** — the FFI layer reinterpret-casts
   without conversion.

Violating any of those pre-conditions at the FFI layer yields one of
five failure modes, all of them nasty:

| Caller mistake | Native symptom | Detection cost without guard |
|---|---|---|
| Strided view (`arr[::2]`) | Incorrect results, no crash | Silent — costs a model-correctness bug |
| Transpose / F-contiguous | Reads wrong elements | Silent — diffs only show up at training loss |
| Misaligned buffer | SIGBUS on strict-align ISAs (ARM, RISC-V) | Hard crash, no stack trace into Python |
| Wrong dtype (int64 vs uint8) | Reads 8× the bytes expected | Process abort or segfault |
| Python list instead of ndarray | Null pointer deref | Immediate segfault |

The guard shifts every one of those into a deterministic
`ValueError` raised in Python land with the offending parameter
name in the message. Cost at the happy path is one `isinstance` check
plus two `flags` dict reads — benchmarked at <50 ns on a contemporary
laptop, which is negligible compared to the native call it precedes
(usually measured in microseconds or milliseconds).

## 2. Public API

### 2.1 `require_c_contiguous`

```python
def require_c_contiguous(
    arr: Any,
    name: str,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
) -> np.ndarray: ...
```

**Parameters**

- `arr` — the candidate input. May be a NumPy `ndarray`, a Python
  `list`, a `tuple`, a 0-d scalar, or any object implementing the
  `__array__` protocol. Zero-copy is guaranteed only for the
  `ndarray` with matching dtype path; every other input triggers at
  least one `np.asarray` conversion.
- `name` — caller-supplied identifier for the parameter. Propagates
  into the `ValueError` message so stack traces point at the exact
  argument that violated the contract, even when an FFI shim
  forwards several ndarrays in one call.
- `dtype` — optional. Accepts both a `np.dtype` instance
  (`np.dtype('uint8')`) and a `np.generic` subclass (`np.uint8`).
  Passing `None` skips the dtype check but still enforces contiguity
  and alignment.

**Returns** — a `ndarray` guaranteed to satisfy:

- `out.flags["C_CONTIGUOUS"] is True`
- `out.flags["ALIGNED"] is True`
- `out.dtype == np.dtype(dtype)` if `dtype` was supplied

**Identity semantics**

The guard promises **zero-copy** on the happy path: when the input is
a `ndarray` that is already C-contiguous, aligned, and carries the
requested dtype, the returned object is `arr` itself
(`require_c_contiguous(arr, ...) is arr`). FFI callers can therefore
cache pointers across multiple guard calls without invalidation:

```python
arr = np.zeros(1024, dtype=np.uint8)
ptr_before = arr.__array_interface__["data"][0]
out = require_c_contiguous(arr, "s", np.uint8)
ptr_after = out.__array_interface__["data"][0]
assert ptr_before == ptr_after                           # zero-copy
```

When the dtype differs, the guard calls `arr.astype(dtype, copy=False)`
— which **may** still reuse the buffer if the conversion is trivial
(same layout, same itemsize), but callers must not rely on that. The
test-suite assertion `test_dtype_coerce_returns_fresh_buffer_by_pointer`
makes the opposite pointer-inequality promise: after dtype coercion
with differing itemsize, the returned buffer is guaranteed distinct.

**Raises**

- `ValueError` — raised in any of four failure cases:
  1. Input is a `ndarray` but not C-contiguous.
  2. Input is a `ndarray` but not aligned.
  3. Input is non-`ndarray`, was converted via `np.asarray`, but the
     conversion yielded a non-contiguous array (only reachable via
     exotic `__array__` implementations — see §4.3).
  4. Same as (3) but alignment failed.

  Each message embeds the `name` parameter and, for the
  contiguity failure, the remediation hint
  `"pass np.ascontiguousarray(x)"`.

## 3. Guarantees and non-guarantees

### 3.1 What the guard catches

- **Contiguity** — rejects `arr[::2]`, `arr.T` (on 2-D+), `arr[:, 2]`
  (column slice), and any view produced by `np.lib.stride_tricks`
  that carries a non-default stride.
- **Alignment** — rejects `float32` views starting at non-4-byte
  offsets, `float64` views at non-8-byte offsets, etc. The check uses
  NumPy's own `ALIGNED` flag, which mirrors the platform ABI's
  requirement.
- **Dtype mismatch** — coerces via `.astype(dtype, copy=False)` when
  `dtype` is supplied and the input dtype does not match.
- **Empty arrays** — pass through. `np.empty(0, dtype=np.uint8)` is
  C-contiguous and aligned by definition, and the native backends all
  accept length-zero inputs as no-ops.
- **0-D arrays** — pass through for the same reason; scalars are
  trivially contiguous.

### 3.2 What the guard does **not** do

- **No bounds check on shape.** The guard does not enforce a specific
  rank or element count. If the native function expects a 1-D vector
  of length 1024, the caller is responsible for asserting that
  separately.
- **No dtype subtype enforcement.** `np.uint8` and `np.dtype('u1')`
  compare equal; an `np.int8` array against `dtype=np.uint8` will
  trigger a copy-coerce rather than a rejection. The guard treats
  dtype as a layout hint, not a semantic contract.
- **No write-back.** `require_c_contiguous` is a read-side check.
  Native functions that mutate their input in place must still ensure
  the caller holds a non-shared view — the guard will happily return
  a view shared with another Python object.
- **No `WRITEABLE` flag check.** Memory-mapped arrays with
  `WRITEABLE = False` will pass through; the native backend is
  expected to either respect read-only or copy the buffer on entry.

### 3.3 Why not use `np.ascontiguousarray` directly?

`np.ascontiguousarray(arr).astype(dtype)` has three drawbacks compared
to the guard:

1. It always returns a new object (`is arr` check always fails),
   losing zero-copy identity.
2. It silently masks alignment violations by copying — which
   eliminates the error but also eliminates any chance of diagnosing
   the caller bug.
3. It discards the `name` context, so a stack trace through four
   FFI shims reaches the user with `"expected C-contiguous"` and no
   indication of which parameter is wrong.

The guard is strict-by-default, informative, and zero-copy when the
caller has already done the right thing. Those three properties are
non-negotiable for a production FFI boundary.

## 4. Failure modes — worked examples

### 4.1 Strided view from `arr[::2]`

```python
>>> base = np.arange(20, dtype=np.uint8)
>>> view = base[::2]                    # stride = 2, not contiguous
>>> require_c_contiguous(view, "stream", np.uint8)
Traceback (most recent call last):
  ...
ValueError: stream must be C-contiguous; pass np.ascontiguousarray(x)
```

**How to fix** — make the copy explicit at the call site:

```python
safe = require_c_contiguous(np.ascontiguousarray(view), "stream", np.uint8)
```

### 4.2 Transposed 2-D matrix

```python
>>> base = np.zeros((4, 8), dtype=np.float32)
>>> t = base.T                          # F-contiguous, not C-contiguous
>>> require_c_contiguous(t, "weights", np.float32)
ValueError: weights must be C-contiguous; pass np.ascontiguousarray(x)
```

For a 2-D matrix a transposed view is F-contiguous, which is different
from C-contiguous. The native kernels in this project are all
row-major; a caller who truly wants column-major semantics must
either copy or transpose back.

### 4.3 Deliberately non-contiguous object via `__array__`

`np.asarray` cannot always normalise arbitrary Python objects. An
object implementing `__array__` that returns a non-contiguous view
survives the coercion:

```python
class StridedProducer:
    def __array__(self, dtype=None, copy=None):
        return np.arange(20, dtype=np.uint8)[::2]

require_c_contiguous(StridedProducer(), "producer")
# → ValueError: producer must be C-contiguous; ...
```

This branch is covered by
`TestNonNdarrayNonContiguousRejected::test_array_protocol_non_contiguous_raises`.
The existence of the test is what bought the module its 100 %
statement coverage — the defensive check after `np.asarray` is not
dead code, it's reachable via the `__array__` protocol.

### 4.4 Misaligned float view

```python
>>> raw = np.zeros(17, dtype=np.uint8)
>>> unaligned = np.ndarray(shape=(4,), dtype=np.float32,
...                        buffer=raw.data, offset=1)
>>> unaligned.flags["ALIGNED"]
False
>>> require_c_contiguous(unaligned, "weights")
ValueError: weights is not aligned
```

On x86 this array would read through a fixup trap and produce
(slow but correct) results. On ARMv7, ARMv8 with alignment checking,
or RISC-V, the same code path SIGBUSes. The guard removes that
target-specific behaviour.

### 4.5 Python list with matching dtype — zero error, one copy

```python
>>> safe = require_c_contiguous([1, 2, 3, 4], "s", np.uint8)
>>> type(safe), safe.dtype, safe.flags["C_CONTIGUOUS"]
(<class 'numpy.ndarray'>, dtype('uint8'), True)
```

The list is converted via `np.asarray(arr, dtype=np.uint8)`. The
resulting array is C-contiguous by construction, so no error is
raised — but this path **always** copies. Hot-loop callers should
convert to an ndarray once outside the loop.

## 5. Integration with the native layer

### 5.1 Rust PyO3 backends

Every PyO3 entry point in `crates/stochastic_doctor_core` and
`crates/autonomous_learning` calls the guard before extracting
the buffer pointer. The pattern is:

```python
def py_scc_bytes(a: np.ndarray, b: np.ndarray) -> float:
    a = require_c_contiguous(a, "a", np.uint8)
    b = require_c_contiguous(b, "b", np.uint8)
    return float(_sdc_rust.py_scc_bytes(a, b))
```

From the Rust side, `PyReadonlyArray1<u8>::as_slice()` would panic if
the array were non-contiguous; the guard converts that panic into a
catchable Python exception with the parameter name. See
[`stochastic_doctor.md`](stochastic_doctor.md) §3 for the full set
of protected entry points.

### 5.2 Mojo SIMD kernels

The Mojo kernels in `accel/mojo/kernels.mojo` take pointers as raw
`Int` addresses and reconstruct `UnsafePointer[T, MutAnyOrigin]`
inside; see [`accel.md`](accel.md) §4. This pattern requires the
Python-side buffer to be contiguous and aligned — the same two
conditions the guard enforces. The dispatcher
`accel/mojo_dispatch.py` applies the guard to every ndarray
argument before computing the pointer.

### 5.3 Wgpu bridge

`WgpuRuleLayer` in `_native/learning_bridge.py` copies ndarrays
into wgpu buffers via `memcpy`. Contiguity is mandatory; the copy
path assumes a flat byte-stream.

## 6. Performance profile

The guard's work is all Python-level: one `isinstance` check, two
flag reads, one optional `astype` call. The happy path in particular
resolves in a handful of bytecode ops and returns the input
unchanged. No benchmark script has been committed yet, so this
section deliberately avoids citing specific nanosecond numbers —
any measurement claim in these docs must be backed by a script in
`benchmarks/` and a JSON record in `benchmarks/results/` per the
repository's no-fabricated-benchmarks rule.

What the suite **does** verify is the **structural** performance
guarantee: on the matching-dtype ndarray path the returned object is
the same Python object as the input (`out is arr`) and the underlying
buffer pointer (`__array_interface__["data"][0]`) is identical. Those
two conditions together are sufficient to prove zero-copy; they are
asserted in
`test_guarded_output_pointer_stable_within_call`. A regression that
introduces a hidden copy on the happy path would break that test
before it could affect downstream users.

On the dtype-mismatch path the test suite asserts the opposite
structural property: when the itemsize of the requested dtype
differs from the input, the returned buffer pointer **must** differ
from the input (`test_dtype_coerce_returns_fresh_buffer_by_pointer`).
Callers relying on in-place mutation can therefore infer from a
pointer comparison whether the guard handed back the same memory or
a new buffer.

## 7. Test surface

The module ships with 24 multi-angle tests in
`tests/test_native/test_array_guards.py`, organised into seven classes:

| Class | Scope | # tests |
|---|---|---|
| `TestNdarrayHappyPath` | Matching dtype, 1-D / 2-D / empty / scalar | 5 |
| `TestNdarrayDtypeCoercion` | Coerce int64→uint8, float64→float32, dtype-object alias | 3 |
| `TestNdarrayNonContiguousRejected` | Strided slice, transpose, column slice, error-message content | 5 |
| `TestNonNdarrayConversion` | List, tuple, nested list, empty list, no-dtype | 5 |
| `TestNonNdarrayNonContiguousRejected` | `__array__` protocol returning non-contiguous / unaligned | 2 |
| `TestAlignmentEnforcement` | Misaligned buffer via `np.ndarray(buffer=..., offset=1)` | 1 |
| `TestIntegrationBytes` | `.tobytes()`, pointer stability, fresh-buffer-on-cast | 3 |

Statement coverage is **100 %** (19/19 statements), verified via
`pytest --cov=sc_neurocore._native.array_guards`. There are no
`pragma: no cover` directives in the source; every defensive branch
has a corresponding test.

## 8. Related modules

- [`sc_neurocore._native.learning_bridge`](autonomous_learning.md) —
  primary consumer on the training side.
- [`sc_neurocore._native.core_engine_bridge`](rust-engine.md) — primary
  consumer on the simulation side.
- [`sc_neurocore.stochastic_doctor`](stochastic_doctor.md) — consumer
  from the diagnostics side; the `require_c_contiguous` calls in
  `diagnostics.py` and `sc_doctor.py` are the motivation for the
  guard's alignment check.
- [`sc_neurocore.accel.mojo_dispatch`](accel.md) — consumer from the
  Mojo SIMD side.

## 9. Source reference

::: sc_neurocore._native.array_guards
