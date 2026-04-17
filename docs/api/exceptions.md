# Exception Hierarchy

**Module:** `sc_neurocore.exceptions`
**Source:** `src/sc_neurocore/exceptions.py` — 82 LOC, 13 exception classes
**Status (v3.14.0):** clean two-level hierarchy with `SCNeuroError` as
the single root; 7 tests pass; **9 of 13 declared classes have zero
`raise` sites in the source tree** — they are reserved for future use
or are expected to be raised by external callers (§5).

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

## 4. The other 9 classes — declared but unused

These classes exist in `exceptions.py` but no `raise` site in `src/`
calls them in v3.14.0:

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
| 2 | Multi-angle tests | ⚠️ WARN | 7 tests pass: 6-way parameterised `issubclass(SCNeuroError)` check + one `raise/catch` round-trip. **Tests do not cover the 4 broad domain exceptions independently** (SCEncodingError / SCConfigError / SCWeightError / SCCompilerError), nor the `RuntimeError` mix-in for SCDependencyError / SCHardwareError. |
| 3 | Rust path | N/A | Pure-Python class declarations; no compute |
| 4 | Benchmarks | N/A | Same |
| 5 | Performance docs | N/A | Same |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX header ✅. **9 of 13 classes have zero raise sites** — declared vocabulary without enforcement (§4). British English clean. No `# noqa`, no `# type: ignore`. |

Net: **2 WARN, 0 FAIL.** Both WARNs trace to "the vocabulary is
larger than the enforcement" — fix is to either raise the typed
exceptions or document the reserved-for-future status.

---

## 7. Known issues

### 7.1 Nine declared exceptions are never raised (task #36)

See §4. Three options for resolution:
1. Migrate plain `ValueError` / `AssertionError` raises to the
   typed exceptions throughout `src/`.
2. Add a "Reserved for future use" line to the docstring of each
   unused class.
3. Delete the unused ones until they have a raise site.

The current state misleads callers who write
`except SeedCollisionError` expecting it to fire — it never will in
v3.14.0.

### 7.2 No tests for `ValueError` / `RuntimeError` mix-in contract

`SCEncodingError` is a `ValueError` so legacy code catching
`ValueError` still works. No test asserts this. A one-line addition:

```python
def test_encoding_is_valueerror():
    assert issubclass(SCEncodingError, ValueError)
```

would lock the contract.

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
