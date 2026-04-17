# Hardware Drivers

**Module:** `sc_neurocore.drivers`
**Source:** `src/sc_neurocore/drivers/` — 4 files, 260 LOC,
`__tier__ = "research"`
**Status (v3.14.0):** PYNQ-Z2 FPGA driver works in EMULATION mode and
correctly fails fast in HARDWARE mode without PYNQ; **the
`PhysicalTwinBridge` claims TCP/Serial hardware-in-the-loop but is a
mock** — see §3 honesty notice below; `verify_hardware_link` reaches
outside the project tree to import optional sister-repo modules.

This page covers the three public symbols that drivers expose, what
each one actually does, and what each one only claims to do.

---

## 1. Public surface

`sc_neurocore.drivers.__init__` re-exports 3 symbols and declares the
research tier:

| Symbol | Source file | Role |
|--------|-------------|------|
| `SC_NeuroCore_Driver` | `sc_neurocore_driver.py` | PYNQ-Z2 FPGA overlay + AXI-Lite register access |
| `PhysicalTwinBridge` | `physical_twin.py` | **Mocked** HIL bridge (see §3) |
| `verify_link` | `verify_hardware_link.py` | Diagnostic CLI that probes FPGA, Evo 2, Opentrons OT-2 |

Module-level constants:
- `__tier__ = "research"` — flag that the module is not stable API.
- `RealityHardwareError(ImportError)` — raised when hardware mode is
  requested but PYNQ libraries are missing.

---

## 2. `SC_NeuroCore_Driver`

```python
class SC_NeuroCore_Driver:
    def __init__(
        self,
        bitstream_path: str = "sc_neurocore.bit",
        mode: str = "HARDWARE",
    ) -> None: ...
```

Driver for the sc-neurocore FPGA overlay on PYNQ-Z2. Two modes:

- `mode="HARDWARE"` (default) — imports `pynq.Overlay`, loads the
  `.bit` file, and verifies the bitstream contains the
  `scpn_layer_1_0` IP block. Raises `RealityHardwareError` on any of:
  - PYNQ Python library missing
  - bitstream file not at the given path or at
    `/usr/local/lib/pynq/overlays/sc_neurocore/<bitstream>`
  - loaded overlay does not have the expected `scpn_layer_1_0` IP
    (raises `SCHardwareError` from `sc_neurocore.exceptions`)
- `mode="EMULATION"` — logs a warning and continues without touching
  any hardware. Used for development on x86 workstations.

Any other `mode` raises `ValueError`.

### 2.1 `write_layer_params(layer_id, params)`

Writes `gain` and/or `threshold` parameters to the named layer's
AXI-Lite registers in fixed-point Q16.16:

| Parameter | Register offset | Encoding |
|-----------|-----------------|----------|
| `gain` | `0x10` | `int(value * 65536)` |
| `threshold` | `0x14` | `int(value * 65536)` |

In EMULATION mode the call is a `logger.debug` no-op. In HARDWARE
mode it walks the overlay attribute namespace via
`getattr(overlay, f"scpn_layer_{layer_id}_0")` and raises
`ValueError` if the IP block is absent.

The Q16.16 encoding here differs from the Q8.8 used elsewhere in
sc-neurocore (compiler, network/export). Documented but worth
flagging if FPGA register IPs are migrated to Q8.8 in the future.

### 2.2 `run_step(input_vector)`

In HARDWARE mode raises `NotImplementedError` ("DMA transfer requires
PYNQ overlay") — i.e. no real DMA path is wired yet.

In EMULATION mode returns `self._rng.random(16)` — uses the
**per-instance** RNG seeded in `__init__` (default `seed=42`).
Two drivers built with the same seed produce identical output
sequences. Fixed by task #29; see §8.2 for the regression-test
breakdown.

### 2.3 Module-level test entry point

`if __name__ == "__main__"` at `sc_neurocore_driver.py:115-123` runs
a strict reality-check: instantiate in HARDWARE mode, expect
`RealityHardwareError` on x86. Used as a quick `python -m
sc_neurocore.drivers.sc_neurocore_driver` smoke test.

---

## 3. `PhysicalTwinBridge` — MOCK, not a real bridge

**Honesty notice.** The class docstring at `physical_twin.py:11-15`
says:

> Bridge for Hardware-In-the-Loop (HIL) Synchronization.
> Connects a Python Neuron to a physical PYNQ-Z2/FPGA neuron via
> TCP/Serial.

This is **not what the code does**. The constructor at
`physical_twin.py:17-23` does no networking at all:

```python
def __init__(self, ip="192.168.2.99", port=5000) -> None:
    self.ip = ip
    self.port = port
    self.connected = False
    print(f"Twin: Connecting to hardware at {ip}:{port}...")
    self.connected = True            # ← unconditionally set, no I/O
```

`sync_step` at `physical_twin.py:25-45` does no networking either:

```python
def sync_step(self, sw_v_mem: float, sw_spike: int) -> float:
    if not self.connected:
        return sw_v_mem
    # Simulate network latency
    # time.sleep(0.001)                                    ← commented out
    # Simulate hardware response (Mock)
    hw_v_mem = sw_v_mem + np.random.normal(0, 0.01)        ← Gaussian noise
    diff = abs(sw_v_mem - hw_v_mem)
    if diff > 0.1:
        print(f"Twin Warning: Divergence detected! …")
    return hw_v_mem
```

The "hardware response" is **`sw_v_mem + N(0, 0.01)`** — i.e. the
software value plus a Gaussian draw. There is no socket, no serial
port, no AXI-Lite read, no PYNQ call. The "divergence detector" can
only fire when `|N(0, 0.01)| > 0.1` (roughly a 0.4σ event scaled to
0.1 → essentially never).

Net effect: **`PhysicalTwinBridge` is a mock that misrepresents
itself as a hardware bridge.** The 45-line file contains no actual
`socket`, `serial`, `pynq`, `requests`, or any other I/O import.

What to do (tracked as task #30):

1. Implement the TCP path (the README, "192.168.2.99:5000", suggests
   the intent), or
2. Rename the class to `MockHILBridge` and update the docstring to
   say "deterministic noise model for HIL prototyping", or
3. Mark the class with a `_TODO_HIL = True` flag and document the
   honest mock status in the docstring.

In any case, callers must not assume `sync_step` reflects FPGA state.

`physical_twin.py:17` also carries `# type: ignore[no-untyped-def]`
without rationale — the IP/port defaults need explicit `str` and
`int` annotations.

---

## 4. `verify_link` — multi-target diagnostic CLI

`verify_hardware_link.py` exposes a single function `verify_link()`
that runs three sequential checks:

| Step | Target | Mechanism | Failure mode |
|------|--------|-----------|--------------|
| 1/3 | PYNQ-Z2 / FPGA bitstream | `SC_NeuroCore_Driver(mode="HARDWARE")` | `RealityHardwareError` → "Simulation Mode" message |
| 2/3 | Evo 2 genomic interface | `sys.path.append(...) → from scpn_evo2_real_interface import Evo2RealInterface` then `evo.connect()` | `ImportError` → "module not found"; `OSError`/`ConnectionError` → "Server unreachable" |
| 3/3 | Opentrons OT-2 robot | `from scpn_opentrions_verify import OpentronsVerifier` then `ot2.ping()` | `ImportError` → "module not found"; `OSError` → "ERROR" |

### 4.1 Cross-repo `sys.path.append` is fragile

`verify_hardware_link.py:37-43` reaches **outside the project tree**:

```python
sys.path.append(
    str(
        Path(__file__).resolve().parent
        / "../../../SCPN-CODEBASE/HolonomicAtlas/src/interfaces"
    )
)
from scpn_evo2_real_interface import Evo2RealInterface
```

This assumes the layout
`/path/to/sc-neurocore/src/sc_neurocore/drivers/...` next to a
sibling `SCPN-CODEBASE/HolonomicAtlas/src/interfaces/` directory.
Outside the GOTM monorepo that path will not exist — the import
falls through to the `ImportError` branch and prints "FAILURE:
Interface module not found in path." Tracked as task #31.

### 4.2 Module-level test entry point

`if __name__ == "__main__"` at line 71-72 runs `verify_link()`, so
`python -m sc_neurocore.drivers.verify_hardware_link` produces the
diagnostic table as a console output.

---

## 5. `RealityHardwareError`

```python
class RealityHardwareError(ImportError):
    """Raised when physical hardware is required but missing."""
```

Subclass of `ImportError`, raised by `_connect_to_fpga` when:
- PYNQ Python library not importable, or
- bitstream file not found at given path or fallback path, or
- bitstream loaded but lacks the expected IP block, or
- any `OSError` / `RuntimeError` during overlay construction.

The strict reality-check pattern means callers can `try / except
RealityHardwareError` to detect non-FPGA hosts and switch to
EMULATION cleanly.

---

## 6. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.drivers import SC_NeuroCore_Driver, ...` | `drivers/__init__.py:12-14` | `tests/test_pynq_driver.py` |
| HARDWARE mode dispatch | `_connect_to_fpga` in `__init__` | `test_driver_hardware_mode_fails_without_fpga` |
| EMULATION mode dispatch | logger warning + skip hardware path | `test_driver_emulation_mode` |
| `write_layer_params` AXI-Lite path | `getattr(overlay, ...)` | `test_driver_write_layer_params` (EMULATION only) |
| `run_step` EMULATION return | `np.random.rand(16)` | `test_driver_run_step` |
| `RealityHardwareError` propagation | raised on PYNQ import / file / IP failures | `test_driver_hardware_mode_fails_without_fpga` |
| `verify_link` CLI | `if __name__ == "__main__"` invokes it | not test-covered |

`PhysicalTwinBridge` is **not test-covered** — `tests/test_pynq_driver.py`
imports only `SC_NeuroCore_Driver`. Tracked as part of task #30.

---

## 7. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 3 symbols re-exported; HARDWARE/EMULATION dispatch tested |
| 2 | Multi-angle tests | ⚠️ WARN | 5 `SC_NeuroCore_Driver` tests pass (EMULATION mode + strict-fail). **`PhysicalTwinBridge` and `verify_link` have zero tests.** |
| 3 | Rust path | N/A | I/O + AXI-Lite shim; no compute kernel |
| 4 | Benchmarks | N/A | Hardware register writes / mock RNG; no meaningful benchmark |
| 5 | Performance docs | N/A | Same as above |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ❌ FAIL | **`PhysicalTwinBridge` misrepresents itself as a hardware bridge** (§3). **`run_step(EMULATION)` global-RNG anti-pattern FIXED by task #29** — driver now accepts `seed` parameter and uses per-instance RNG. **`verify_hardware_link.py` reaches outside the project tree via `sys.path.append`**. **3 undocumented `# type: ignore` / `# noqa` markers**. SPDX header on every file ✅. |

Net: **1 WARN, 1 FAIL.** The FAIL is honesty-driven, not
correctness-driven for the FPGA driver itself — but the
`PhysicalTwinBridge` claim is severe enough to fail the rule.
Task #29 closed in this session; the remaining FAIL items
(`PhysicalTwinBridge` mock, sys.path.append, type:ignore) stay
open under tasks #30 / #31.

---

## 8. Known issues (for the implementation)

### 8.1 `PhysicalTwinBridge` is a mock pretending to be hardware

See §3. Highest-priority fix in this module. Tracked as task #30.

### 8.2 `run_step` EMULATION RNG (FIXED by task #29)

`SC_NeuroCore_Driver.__init__` now accepts `seed: int = 42` and
constructs `self._rng = np.random.default_rng(seed)`. The
EMULATION `run_step` returns `self._rng.random(16)` instead of
`np.random.rand(16)`, so two drivers built with the same seed
produce bitwise-identical output sequences regardless of the
global numpy RNG state.

Regression coverage:
`tests/test_pynq_driver.py::TestRunStepDeterminism` (5 tests):
same-seed first call, same-seed 50-step sequence, distinct seeds
differ, global numpy seed does not leak in, default seed is 42.

### 8.3 `verify_hardware_link.py` `sys.path.append` outside project

See §4.1. Either (a) move the Evo 2 / Opentrons probes into a
dedicated optional extra (e.g. `pip install
sc-neurocore-engine[diagnostics]`), or (b) document the GOTM-monorepo
assumption in the function docstring, or (c) remove these checks
from the public API. Tracked as task #31.

### 8.4 `# type: ignore` markers without rationale

- `physical_twin.py:17` — `# type: ignore[no-untyped-def]` on the
  `__init__` signature with un-annotated `ip` / `port` defaults.
  Trivial fix: type as `str` / `int`.
- `sc_neurocore_driver.py:52` — `# type: ignore  # noqa: F401` on
  the `pynq` import. The `noqa: F401` is justified (the import is
  for side-effect — loading the library), but the bare `# type:
  ignore` (no error code) is too broad. Specify
  `# type: ignore[import-not-found]`.

### 8.5 Q16.16 in the FPGA driver vs Q8.8 elsewhere

`write_layer_params` encodes parameters as `int(value * 65536)`,
which is Q16.16 (16 integer + 16 fractional bits). The compiler
(`equation_compiler.py`) uses Q8.8. If the FPGA register IPs are
re-spun to Q8.8, this multiplication needs to change to `* 256`.
Document the Q-format choice in the IP-block contract.

### 8.6 `PhysicalTwinBridge` constructor prints to stdout

`print(f"Twin: Connecting to hardware at {ip}:{port}...")` at
`physical_twin.py:22` writes to stdout unconditionally on every
construction. This is library code; should use `logging` to allow
callers to silence it. Same pattern in
`physical_twin.py:43` (divergence warning). Tracked under task #30.

---

## 9. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_pynq_driver.py -v
# 5 passed in 1.51s (verified 2026-04-17)
```

Coverage breakdown (5 tests, all in flat module — no test class):

| Test | What it checks |
|------|----------------|
| `test_driver_emulation_mode` | EMULATION mode constructs without raising |
| `test_driver_write_layer_params` | EMULATION path no-ops cleanly with both `gain` and `threshold` |
| `test_driver_run_step` | EMULATION returns shape-(16,) ndarray |
| `test_driver_hardware_mode_fails_without_fpga` | HARDWARE on x86 raises `RealityHardwareError` |
| `test_driver_invalid_mode` | `mode="WHATEVER"` raises `ValueError` |

Not covered:

- `PhysicalTwinBridge` — no test of constructor or `sync_step` (§3
  honesty issue surfaces here)
- `verify_link` — no test (§4 path-handling issue)
- HARDWARE mode happy path — requires actual PYNQ-Z2; cannot be
  exercised on x86 (test_driver_hardware_mode_fails_without_fpga is
  the inverse)
- Bitstream-IP-block missing branch — not exercised; would need
  PYNQ + a bitstream lacking `scpn_layer_1_0`
- Q16.16 encoding correctness — `write_layer_params` is a no-op in
  EMULATION; the `int(value * 65536)` math is not asserted

---

## 10. References

- Xilinx PYNQ — [pynq.io](https://www.pynq.io/) — Python overlay
  framework for Zynq-class FPGAs.
- TUL PYNQ-Z2 board — board spec the FPGA driver is written against.
- AXI-Lite specification — [Arm IHI 0022](https://developer.arm.com/documentation/ihi0022/latest/) — register-mapped peripheral protocol used for `write_layer_params`.

Internal:

- Compiler (Q8.8 fixed-point): [`api/cli.md`](cli.md), [`api/compiler.md`](compiler.md)
- Exception hierarchy (`SCHardwareError`): planned `api/exceptions.md`

---

## 11. Auto-rendered API

::: sc_neurocore.drivers
    options:
      show_root_heading: true
      show_source: true
      members:
        - SC_NeuroCore_Driver
        - PhysicalTwinBridge
        - verify_link
