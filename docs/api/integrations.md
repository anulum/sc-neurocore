# Framework Integrations

**Module:** `sc_neurocore.integrations`
**Source:** `src/sc_neurocore/integrations/` — 2 files, 157 LOC
**Status (v3.14.0):** one bridge present (Intel Lava / Loihi); 3 tests
exist but all skip without `lava-nc` installed; `__init__.py` does not
re-export anything; the bridge's own usage docstring references a class
name that does not exist (§3 honesty notice).

This page covers the entire integrations subpackage: today that means
the Lava bridge for deployment on Loihi 2 neuromorphic hardware (or
the Lava CPU simulation). Other planned bridges (PyTorch, ONNX,
Brian2) are not implemented in v3.14.0.

---

## 1. Public surface

`sc_neurocore.integrations.__init__` is **a docstring-only file** — it
declares the package purpose ("Framework integrations (Lava/Loihi,
etc.)") but does not re-export anything. To use the bridge, import
from the submodule directly:

```python
from sc_neurocore.integrations.lava_bridge import (
    SCtoLavaConverter,
    export_weights_loihi,
    loihi_threshold_from_sc,
    LoihiNetworkConfig,
)
```

The module-level `HAS_LAVA: bool` flag in `lava_bridge.py` is `True`
when the `lava-nc` library is importable. When `HAS_LAVA is False`,
the `SCDenseProcess` and `PySCDenseModel` classes (§4) are not
defined, but everything else still imports cleanly.

| Symbol | Defined in | Available without lava-nc? |
|--------|-----------|----------------------------|
| `HAS_LAVA` | `lava_bridge.py:37,39` | yes (always set) |
| `LoihiNetworkConfig` | `lava_bridge.py:66` (dataclass) | yes |
| `SCtoLavaConverter` | `lava_bridge.py:79` | yes |
| `export_weights_loihi` | `lava_bridge.py:42` | yes |
| `loihi_threshold_from_sc` | `lava_bridge.py:60` | yes |
| `SCDenseProcess` | `lava_bridge.py:121` (in `if HAS_LAVA` block) | **no — undefined when `lava-nc` absent** |
| `PySCDenseModel` | `lava_bridge.py:135` (in `if HAS_LAVA` block) | **no — undefined when `lava-nc` absent** |

Importing a name from the conditional block when `lava-nc` is missing
raises `ImportError` at use time (the `if HAS_LAVA` guard simply
omits the class definitions).

---

## 2. `lava-nc` dependency

`lava_bridge.py:26-39` performs a multi-import inside `try/except
ImportError`. The block depends on the following lava modules:

```python
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
```

If any one of these fails, `HAS_LAVA` is set to `False` and all 9
names go out of scope.

**Install instructions** (per the bridge's module docstring):
`pip install lava-nc`. Note that `lava-nc` officially supports Python
3.10 (the test file `test_lava_integration.py` documents this
constraint at line 11). On Python 3.12 — the system interpreter on
this workstation — `lava-nc` is not available, so all three Lava
tests skip.

---

## 3. Honesty notice — `LoihiDenseProcess` does not exist

`lava_bridge.py:15-18` says:

> Usage:
>     from sc_neurocore.integrations.lava_bridge import (
>         SCtoLavaConverter, export_weights_loihi, LoihiDenseProcess,
>     )

The actual class name in the source is **`SCDenseProcess`** (line
121), not `LoihiDenseProcess`. The example as written raises
`ImportError`. Either rename the class to `LoihiDenseProcess` (the
intent is clearer) or fix the docstring. Tracked as task #33.

The companion `PySCDenseModel` class also follows the
`SC*` prefix pattern — these names should probably stay paired.
Recommended fix: update the docstring to `SCDenseProcess`.

---

## 4. Weight + threshold quantisation

The two helper functions handle the SC ↔ Loihi numeric mapping.

### 4.1 `export_weights_loihi(weights, weight_bits=8, weight_exp=0)`

```python
max_val =  (1 << (weight_bits - 1)) - 1     # 127 for 8-bit
min_val = -(1 << (weight_bits - 1))         # -128
scaled = (weights * 2.0 - 1.0) * max_val    # [0, 1] → [-127, 127]
quantised = np.clip(np.round(scaled), min_val, max_val).astype(int32)
return quantised * (2 ** weight_exp)
```

SC weights live in `[0, 1]` (probability scale); Loihi expects signed
integers with configurable bit width. The mapping is **`weight - 0.5`
shifted to ±max**:
- `w = 0.0` → `-127`
- `w = 0.5` → `0`
- `w = 1.0` → `+127`

`weight_exp > 0` shifts the result left by `weight_exp` bits, giving
the caller a pre-scaled magnitude for Loihi's `wgtExp` field
(Loihi uses `effective_weight = stored_weight * 2**wgtExp`). The
default `weight_exp = 0` returns the raw quantised value.

The mapping treats `0.5` as the SC zero-point. This matches Loihi's
balanced bipolar encoding but differs from many SC frameworks where
the zero-point is `0` (unsigned probabilities). Document the
convention before importing existing SC weights.

### 4.2 `loihi_threshold_from_sc(sc_threshold, weight_bits=8)`

```python
max_val = (1 << (weight_bits - 1)) - 1
return int(np.round(sc_threshold * max_val))
```

Threshold scales linearly: `sc_threshold = 1.0` → `127`. Used by
`SCtoLavaConverter.convert_dense_layer` to set the per-output
`thresholds` array (defaults to `1.0 → 127` for every output).

---

## 5. `LoihiNetworkConfig` and `SCtoLavaConverter`

```python
@dataclass
class LoihiNetworkConfig:
    n_inputs: int
    n_outputs: int
    weights: np.ndarray
    thresholds: np.ndarray
    weight_bits: int = 8
    weight_exp: int = 0
    decay: int = 128
```

A POD struct that captures everything `PySCDenseModel.run_spk` needs
to instantiate a Loihi-style integrate-and-fire layer. `decay = 128`
is the default linear decay factor used in the integer membrane
update (§6).

### 5.1 `SCtoLavaConverter`

```python
class SCtoLavaConverter:
    def __init__(self, weight_bits: int = 8): ...

    def convert_dense_layer(self, sc_layer) -> LoihiNetworkConfig: ...
    def convert_training_model(self, spiking_net) -> list[LoihiNetworkConfig]: ...
```

`convert_dense_layer(sc_layer)` reads `sc_layer.weights` (a 2-D array
of shape `(n_out, n_in)`), calls `export_weights_loihi`, and builds
a single `LoihiNetworkConfig` with thresholds set uniformly to the
quantised value of `1.0`.

`convert_training_model(spiking_net)` walks the result of
`spiking_net.to_sc_weights()` (assumed to return an iterable of
`(n_out, n_in)` weight tensors, possibly PyTorch — handled by
`hasattr(w, "numpy")`) and produces one `LoihiNetworkConfig` per
layer. There is no explicit handling of layer-specific thresholds,
biases, or sparse connectivity — every layer gets the same threshold
default.

Both methods carry `# type: ignore[attr-defined]` markers because
`sc_layer.weights` and `spiking_net.to_sc_weights` are duck-typed.
A `Protocol` class (e.g. `class _ConvertibleLayer(Protocol)`) would
let the type-ignores go away.

---

## 6. `SCDenseProcess` and `PySCDenseModel` (Lava only)

When `HAS_LAVA is True`, two Lava classes are defined.

### 6.1 `SCDenseProcess(AbstractProcess)`

A Lava `Process` with:
- `s_in: InPort(shape=(n_inputs,))` — incoming spike vector
- `s_out: OutPort(shape=(n_outputs,))` — outgoing spike vector
- `weights: Var(shape=(n_out, n_in))` — quantised weight matrix
- `v: Var(shape=(n_outputs,))` — membrane state, initialised to zeros
- `threshold: Var(shape=(n_outputs,))` — per-output spike threshold
- `decay: Var(shape=(1,))` — single decay factor

### 6.2 `PySCDenseModel(PyLoihiProcessModel)`

The Python-side model implementation that runs in Lava's Loihi
simulator. The hot path is `run_spk` (`lava_bridge.py:143-149`):

```python
def run_spk(self) -> None:
    spikes_in = self.s_in.recv()
    current = self.weights @ spikes_in
    self.v[:] = (self.v * self.decay[0]) // 256 + current
    spikes_out = (self.v >= self.threshold).astype(int)
    self.v[spikes_out == 1] = 0
    self.s_out.send(spikes_out)
```

This is **integer current-based LIF**:
- `(v * decay) // 256` — linear decay scaled by the integer `decay`
  field (default 128 → keeps half the membrane each step)
- `+ current` — synaptic input from `weights @ spikes_in`
- threshold-and-reset — binary spike emission, hard reset to zero

The `// 256` divisor implies Loihi-style 16-bit fixed-point decay
scaling (decay value `d` in `[0, 255]` scales by `d / 256`). This
matches Loihi 1's CUBA integer model qualitatively but is **not a
verified Loihi 2 parity model** — Loihi 2 has additional features
(threshold adaptation, conductance-based COBA mode, programmable
neuron logic) that this bridge does not cover. Tracked as task #34.

The `@implements(proc=SCDenseProcess, protocol=LoihiProtocol)` and
`@requires(CPU)` decorators bind the model to the Lava simulator on
CPU. The bridge does not currently expose a hardware-deployment
helper for actual Loihi silicon.

---

## 7. Performance — not measured

`lava-nc` is not installable on Python 3.12, so end-to-end
benchmarks (Lava-CPU vs sc-neurocore Python) cannot be exercised in
this environment. Document the gap; do not fabricate numbers.

The pure-Python helpers (`export_weights_loihi`,
`loihi_threshold_from_sc`, `SCtoLavaConverter.convert_dense_layer`)
run in microseconds for typical layer sizes (≤1000 neurons).
Quantisation cost is dominated by `np.round` + `np.clip` — both
NumPy-vectorised.

---

## 8. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.integrations.lava_bridge import HAS_LAVA` | top-level `try/except` | `tests/test_lava_integration.py::test_lava_import` (skips without lava) |
| `SCtoLavaConverter.convert_dense_layer` → `export_weights_loihi` → `LoihiNetworkConfig` | direct call inside the converter | `test_sc_to_lava_converter` (skips without lava) |
| `PySCDenseModel.run_spk` integer LIF | bound by `@implements(proc=SCDenseProcess)` | `test_spike_train_parity` (skips without lava) |
| `loihi_threshold_from_sc` | called from `convert_dense_layer` | covered transitively |

The `__init__.py` re-export is **missing**, breaking the
`from sc_neurocore.integrations import X` pattern. Tracked as part
of task #33.

---

## 9. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ⚠️ WARN | `lava_bridge.py` internals wire correctly, but `__init__.py` exports nothing — see §1 |
| 2 | Multi-angle tests | ⚠️ WARN | 3 tests defined but **all skip** without `lava-nc` (Python 3.10 only). No tests cover `export_weights_loihi` / `loihi_threshold_from_sc` / `SCtoLavaConverter` independently of Lava. |
| 3 | Rust path | N/A | I/O + integer quantisation; downstream Lava is the compute layer |
| 4 | Benchmarks | ❌ FAIL | None — `lava-nc` unavailable in this env. Not fabricated. |
| 5 | Performance docs | ⚠️ WARN | §7 is honest but empty. |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX header on every file ✅. **`LoihiDenseProcess` in usage docstring does not exist** (§3). **`__init__.py` re-exports nothing**. **3+ undocumented `# type: ignore` markers** (lines 87, 101, 121, 135). |

Net: **4 WARN, 1 FAIL.** Most of the WARNs trace back to the `lava-nc`
Python-version constraint; the FAIL is the inability to benchmark
without it.

---

## 10. Known issues

### 10.1 `LoihiDenseProcess` referenced in docstring but undefined

See §3. Fix: update docstring to `SCDenseProcess`, or rename the
class. Tracked as task #33.

### 10.2 `__init__.py` re-exports nothing

See §1. Add explicit re-exports:

```python
from .lava_bridge import (
    HAS_LAVA,
    LoihiNetworkConfig,
    SCtoLavaConverter,
    export_weights_loihi,
    loihi_threshold_from_sc,
)

__all__ = [
    "HAS_LAVA",
    "LoihiNetworkConfig",
    "SCtoLavaConverter",
    "export_weights_loihi",
    "loihi_threshold_from_sc",
]
```

`SCDenseProcess` and `PySCDenseModel` cannot go in `__all__`
unconditionally because they are only defined when `HAS_LAVA is True`.
Either add them conditionally or document the
`from .lava_bridge import SCDenseProcess` pattern. Same task #33.

### 10.3 `PySCDenseModel.run_spk` is not a verified Loihi 2 parity model

The integer LIF in §6.2 matches Loihi 1 CUBA qualitatively but does
not verify against Loihi 2 reference behaviour. The bridge ships
this as "deployment on Intel Loihi 2" in the module docstring but
there is no parity test against Loihi 2 silicon or the official
Lava Loihi 2 simulator. Tracked as task #34.

### 10.4 Three `# type: ignore` markers without rationale

- `lava_bridge.py:87` — `# type: ignore[attr-defined]` on
  `sc_layer.weights` (duck-typed)
- `lava_bridge.py:101` — same on `spiking_net.to_sc_weights`
- `lava_bridge.py:121` — `# type: ignore[misc]` on
  `class SCDenseProcess(AbstractProcess)` (Lava base type)
- `lava_bridge.py:135` — same on `class PySCDenseModel(...)`

The first two can go away with a `Protocol` defining the duck-typed
methods. The latter two are Lava base-class typing limitations and
need a per-line rationale comment.

### 10.5 No `to_sc_weights` typing contract documented

`convert_training_model` calls `spiking_net.to_sc_weights()` and
assumes it returns an iterable of 2-D weight tensors. The type
contract is enforced by `# type: ignore` rather than by a `Protocol`
class or an `__abstractmethod__` declaration. Callers learn the
contract by reading the converter source.

### 10.6 Weight-zero-point convention may surprise users

`export_weights_loihi` treats SC `0.5` as the integer zero. SC
frameworks that use unsigned probabilities (zero = "always off")
will wrap to the negative half of the integer range. Document the
convention prominently in the function docstring or add an
`unsigned: bool = False` flag.

---

## 11. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_lava_integration.py -v
# 3 skipped (lava-nc not installed) - verified 2026-04-17
```

Test breakdown (all skip without `lava-nc`):

| Test | What it would check |
|------|---------------------|
| `test_lava_import` | `import lava.lib.dl.slayer` and `from lava.proc.lif.process import LIF` succeed |
| `test_sc_to_lava_converter` | Build a Dense + LIF chain, run 100 steps on Loihi1SimCfg, assert the membrane shape |
| `test_spike_train_parity` | Compare an `sc_neurocore.neurons.StochasticLIFNeuron` 100-step spike train against a Lava `LIF(shape=(1,), vth=256)` 100-step run; asserts both produce >0 spikes (does not assert exact count parity) |

What is NOT covered (independent of `lava-nc`):

- `export_weights_loihi` — no test of `[0, 1] → [-127, 127]`
  mapping, no test of `weight_exp` shift, no test of the
  zero-point convention
- `loihi_threshold_from_sc` — no test
- `SCtoLavaConverter.convert_dense_layer` — no test with a mock
  `sc_layer` (could be done without Lava)
- `LoihiNetworkConfig` dataclass round-trip — no test

These should all be addable as no-skip tests since they don't need
Lava.

---

## 12. References

- Davies M. *et al.* "Loihi: A Neuromorphic Manycore Processor with
  On-Chip Learning." *IEEE Micro* 38(1):82-99 (2018). The Loihi 1
  architecture this bridge mostly targets.
- Orchard G. *et al.* "Efficient Neuromorphic Signal Processing with
  Loihi 2." *2021 IEEE Workshop on Signal Processing Systems
  (SiPS)*. Loihi 2 architecture overview.
- Lava framework — [github.com/lava-nc/lava](https://github.com/lava-nc/lava) — Intel's open-source Python framework for neuromorphic computing.

Internal:

- Stochastic LIF neuron used in the parity test:
  [`api/neurons.md`](neurons.md)
- Compiler Q-format reference (Loihi uses 8-bit, sc-neurocore
  uses Q8.8 for FPGA): [`api/cli.md`](cli.md)

---

## 13. Auto-rendered API

::: sc_neurocore.integrations.lava_bridge
    options:
      show_root_heading: true
      show_source: true
      members:
        - HAS_LAVA
        - LoihiNetworkConfig
        - SCtoLavaConverter
        - export_weights_loihi
        - loihi_threshold_from_sc
