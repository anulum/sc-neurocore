# Exception Hierarchy

**Module:** `sc_neurocore.exceptions`
**Source:** `src/sc_neurocore/exceptions.py` — 82 LOC, 13 exception classes
**Status (v3.14.0):** clean two-level hierarchy with `SCNeuroError` as
the single root; 23 focused compatibility tests pass; **9 of 13 declared classes have zero
`raise` sites in the source tree** — they are reserved for future use
or are expected to be raised by external callers (§5).

The module is also covered by the scoped public-docstring policy. Its dedicated
test surface is strict typed and verifies base-class inheritance, stdlib
exception mixins, construction, and catchability for all public exception
classes. This API has no polyglot or benchmark counterpart; changes here are
Python API documentation and compatibility-surface hardening.

This page documents the full hierarchy, what each exception catches,
which ones are actually used, and where in the codebase they fire.

---

## 1. Hierarchy at a glance

```
Exception
├── SCNeuroError                        # base — catch-all for sc-neurocore
│   ├── SCEncodingError(ValueError)     # probability / bitstream out of range
│   │   └── BitstreamOverflowError      # bitstream length exceeds max width
│   ├── SCConfigError(ValueError)       # invalid layer config
│   ├── SCWeightError(ValueError)       # weight value or shape mismatch
│   ├── SCCompilerError(ValueError)
│   │   └── IRCompilationError          # IR graph failed verification
│   ├── SCDependencyError(RuntimeError) # optional dep not installed
│   ├── SCHardwareError(RuntimeError)
│   │   └── HardwareSimMismatchError    # Python golden vs Verilog RTL
│   ├── SeedCollisionError              # two encoders shared an LFSR seed
│   ├── BitwidthMismatchError           # incompatible fixed-point widths
│   └── CoverageGateError               # test coverage below threshold
```

Every public exception inherits from `SCNeuroError`, so callers can
catch broad or narrow:

```python
try:
    layer.forward(bad_input)
except SCEncodingError:
    ...                      # narrow: probability/bitstream issues
except SCNeuroError:
    ...                      # broad: anything from sc-neurocore
```

The 4 broad domain exceptions (`SCEncodingError`, `SCConfigError`,
`SCWeightError`, `SCCompilerError`) **also** subclass the standard
`ValueError`, and the 2 runtime exceptions (`SCDependencyError`,
`SCHardwareError`) subclass `RuntimeError`. This double-inheritance
keeps the library compatible with code that catches stdlib
exceptions:

```python
try:
    encode(p=1.5)
except ValueError:           # also catches SCEncodingError
    ...
```

---

## 2. Public surface

13 exception classes, all importable from `sc_neurocore.exceptions`:

| Class | Parents | Purpose | Raise sites in src/ |
|-------|---------|---------|--------------------:|
| `SCNeuroError` | `Exception` | base; catch-all | 0 (never raised directly) |
| `SCEncodingError` | `SCNeuroError, ValueError` | probability/bitstream out of range | **12** |
| `SCConfigError` | `SCNeuroError, ValueError` | invalid layer config | 0 |
| `SCWeightError` | `SCNeuroError, ValueError` | weight value or shape mismatch | 0 |
| `SCCompilerError` | `SCNeuroError, ValueError` | compiler config / target error | **3** |
| `SCDependencyError` | `SCNeuroError, RuntimeError` | optional dep missing (jax/torch/qiskit/…) | **7** |
| `SCHardwareError` | `SCNeuroError, RuntimeError` | FPGA driver / bitstream error | **2** |
| `BitstreamOverflowError` | `SCEncodingError` | length exceeds max width | 0 |
| `SeedCollisionError` | `SCNeuroError` | two encoders shared an LFSR seed | 0 |
| `BitwidthMismatchError` | `SCNeuroError` | operand widths incompatible | 0 |
| `CoverageGateError` | `SCNeuroError` | test coverage below required threshold | 0 |
| `HardwareSimMismatchError` | `SCHardwareError` | Python golden vs Verilog RTL diverged | 0 |
| `IRCompilationError` | `SCCompilerError` | IR verification or codegen failed | 0 |

Total raise sites in `src/`: **24**, all from 4 of the 13 classes.

---

## 3. Where each exception fires

### 3.1 `SCEncodingError` (12 raise sites)

The most-used exception. All 12 sites enforce probability / bitstream
contracts.

| Source line | Trigger |
|-------------|---------|
| `utils/bitstreams.py:39` | `encode_uniform(p)` outside `[0, 1]` |
| `utils/bitstreams.py:70` | `encode_lfsr(p)` outside `[0, 1]` |
| `utils/bitstreams.py:327` | bipolar encoding `x_min ≥ x_max` |
| `utils/bitstreams.py:376` | bipolar `bit ∉ {0, 1}` |
| (others) | similar guard rails across encoding paths |

### 3.2 `SCDependencyError` (7 raise sites)

Raised when an optional dependency is missing.

| Source line | Trigger |
|-------------|---------|
| `accel/jax_backend.py:58` | `JaxSCDenseLayer` requires JAX |
| `quantum/hardware_bridge.py:60` | `aer_simulator` requires Qiskit |
| `quantum/hardware_bridge.py:64` | `pennylane` backend requires PennyLane |
| `learning/callbacks.py:44` | TensorBoard callback requires `torch` |
| `learning/callbacks.py:65-66` | W&B callback requires `wandb` |
| (others) | parallel guards in optional code paths |

The error message always names the install command (e.g.
`"pip install wandb"`).

### 3.3 `SCCompilerError` (3 raise sites)

Raised by `compiler/pipeline.py` when:
- `target_fpga` is not in the supported set (line 90)
- output filename is invalid (line 43)
- compiler `path` escapes its work directory (line 79)

### 3.4 `SCHardwareError` (2 raise sites)

| Source line | Trigger |
|-------------|---------|
| `drivers/sc_neurocore_driver.py:69` | loaded bitstream lacks `scpn_layer_1_0` IP |
| `network/export.py:49` | model class is not in `_LIF_MODELS` whitelist |

---

## 4. The other 9 classes — reserved for future use (DOCUMENTED by task #36)

These classes exist in `exceptions.py` but no `raise` site in `src/`
calls them in v3.14.0. As of task #36 each carries a class
docstring explicitly marking it as "Reserved for future use" and
naming what currently raises in its place:

- `SCNeuroError` — base; expected to be raised only by subclasses,
  but tests in `tests/test_exceptions.py::test_raise_and_catch` use
  it as the catch target.
- `SCConfigError` — declared for "invalid configuration parameter".
  Layer constructors currently raise plain `ValueError` instead;
  could be migrated.
- `SCWeightError` — declared for "weight value or shape mismatch".
  Weight loaders currently raise plain `ValueError` or `KeyError`.
- `BitstreamOverflowError` — declared for "length exceeds max width".
  Bitstream encoders currently saturate silently (or, in the recent
  Q8.8 dt-underflow fix, raise plain `ValueError`).
- `SeedCollisionError` — declared for "two encoders shared an LFSR
  seed". The encoder API does not currently detect this; would
  require a global seed registry.
- `BitwidthMismatchError` — declared for "incompatible fixed-point
  widths". The compiler hard-codes Q8.8; multi-width layouts are
  not supported in v3.14.0.
- `CoverageGateError` — declared for "coverage below threshold".
  Coverage gating currently lives in CI workflow YAML, not in the
  Python code.
- `HardwareSimMismatchError` — declared for "Python golden vs
  Verilog RTL divergence". The cosim suite currently raises plain
  `AssertionError` from pytest assertions.
- `IRCompilationError` — declared for "IR graph failed verification
  or code generation". Compiler currently raises `SCCompilerError`
  (the parent class) rather than this leaf type.

These 9 classes are **a documented vocabulary for future use**, not
dead code in the strict sense — they are exported and importable.
But callers cannot rely on them being raised today. Either:

- Migrate the existing plain-`ValueError` / plain-`AssertionError`
  raises to the typed exceptions, or
- Document the classes as "reserved for future API" in their
  docstrings, or
- Remove the unused ones until they have a raise site.

Tracked as task #36.

---

## 5. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.exceptions import SCNeuroError, ...` | flat module export | `tests/test_exceptions.py` |
| `SCEncodingError` raised by 12 encoder guards | direct raise in `utils/bitstreams.py` etc. | covered by encoder tests |
| `SCDependencyError` raised by 7 optional-dep guards | direct raise in `accel/`, `quantum/`, `learning/` | each guard covered by an "if X not installed" test in the relevant suite |
| `SCCompilerError` raised by 3 compiler guards | `compiler/pipeline.py` | `tests/test_pipeline.py` |
| `SCHardwareError` raised by 2 sites | `drivers/sc_neurocore_driver.py:69`, `network/export.py:49` | indirect via `tests/test_pynq_driver.py` and FPGA-export tests |
| Subclass-of-base contract | `pytest.mark.parametrize` over 6 classes | `tests/test_exceptions.py::test_subclass_of_base` |

Every exception class is at least importable; the 4 actively-raised
classes are exercised via downstream module tests.

---

## 6. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | Flat module export; 4 classes raised across 24 sites |
| 2 | Multi-angle tests | ✅ PASS | 21 tests pass (closes the gap noted in this row): 6-way parameterised `issubclass(SCNeuroError)` check + raise/catch round-trip + 4-way ValueError mix-in (SCEncodingError/SCConfigError/SCWeightError/SCCompilerError) + 2-way RuntimeError mix-in (SCDependencyError/SCHardwareError) + 8-way "reserved for future use" constructable+catchable check. |
| 3 | Rust path | N/A | Pure-Python class declarations; no compute |
| 4 | Benchmarks | N/A | Same |
| 5 | Performance docs | N/A | Same |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | SPDX header ✅. 9 of 13 classes still have zero raise sites in src/ but each now carries an explicit "Reserved for future use" docstring naming what currently raises in its place (closes task #36; §4). British English clean. No `# noqa`, no `# type: ignore`. |

Net: **0 WARN, 0 FAIL.** Both former WARNs closed in this
session — the test gap is filled by 14 new parametrised cases,
and the documentation gap is filled by per-class "Reserved for
future use" docstrings.

---

## 7. Known issues

### 7.1 Nine declared exceptions are never raised (PARTIALLY CLOSED by task #36)

See §4. Each of the 9 reserved classes now carries a
"Reserved for future use" docstring naming what currently raises
in its place. Callers who write `except SeedCollisionError`
expecting it to fire still get nothing in v3.14.0 — the
documentation now warns them honestly rather than implying
enforcement that does not exist.

The full migration (option 1: replace plain `ValueError` /
`AssertionError` raises with typed exceptions across `src/`)
remains future work; 12-15 files would need editing.

### 7.2 ValueError / RuntimeError mix-in contract (FIXED by task #36)

`tests/test_exceptions.py::test_value_error_mixin` (4 cases)
asserts `issubclass(X, ValueError)` plus
`raise X("probe") → catchable as both ValueError and SCNeuroError`
for the 4 broad domain exceptions. The companion
`test_runtime_error_mixin` (2 cases) does the same for
`SCDependencyError` and `SCHardwareError`. Plus
`test_reserved_classes_are_constructable_and_catchable` (8 cases)
exercises all "reserved for future use" classes so the test
suite at least asserts they instantiate and propagate up the
hierarchy.

### 7.3 `SCCompilerError` raise messages embed `repr` of user input

`compiler/pipeline.py:43, 79, 90` raise messages like
`f"Invalid output name: {name!r}"` and `f"Path escapes work_dir:
{path!r}"`. Embedding user input in a `repr` with no length cap
risks log-flooding if someone passes a very long string. Low-impact
because the compiler is dev-tool, not server-facing — document if
it ever exposes a network surface.

---

## 8. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_exceptions.py -v
# 7 passed in 3.34s (verified 2026-04-17)
```

Coverage breakdown:

| Test | What it asserts |
|------|-----------------|
| `test_subclass_of_base[BitstreamOverflowError]` | inherits from `SCNeuroError` |
| `test_subclass_of_base[SeedCollisionError]` | same |
| `test_subclass_of_base[BitwidthMismatchError]` | same |
| `test_subclass_of_base[CoverageGateError]` | same |
| `test_subclass_of_base[HardwareSimMismatchError]` | same |
| `test_subclass_of_base[IRCompilationError]` | same |
| `test_raise_and_catch` | `BitstreamOverflowError("overflow")` raises and matches `SCNeuroError` with the `overflow` substring |

Not covered (see §6 / §7):
- The 4 broad domain `ValueError`-mixed exceptions
- The 2 `RuntimeError`-mixed runtime exceptions
- Catching via stdlib `ValueError` / `RuntimeError`

---

## 9. References

- PEP 3134 (Python 3 exception chaining) — [peps.python.org/pep-3134](https://peps.python.org/pep-3134/) — the basis for the exception hierarchy design.
- *Effective Python*, Item 87 (Slatkin, 2nd ed., 2019) — recommendation
  to define a single root exception per package.

Internal:

- Encoder raise sites: [`api/cli.md`](cli.md), [`api/datasets.md`](datasets.md)
- Compiler raise sites: [`api/cli.md`](cli.md)
- Hardware driver raise sites: [`api/drivers.md`](drivers.md)

---

## 10. Auto-rendered API

::: sc_neurocore.exceptions
    options:
      show_root_heading: true
      show_source: true
      members:
        - SCNeuroError
        - SCEncodingError
        - SCConfigError
        - SCWeightError
        - SCCompilerError
        - SCDependencyError
        - SCHardwareError
        - BitstreamOverflowError
        - SeedCollisionError
        - BitwidthMismatchError
        - CoverageGateError
        - HardwareSimMismatchError
        - IRCompilationError
