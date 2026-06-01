# Hardware Drivers

**Module:** `sc_neurocore.drivers`
**Source:** `src/sc_neurocore/drivers/` — 4 files, 399 LOC,
`__tier__ = "research"`
**Status (v3.15.1):** PYNQ-Z2 FPGA driver works in EMULATION mode and
correctly fails fast in HARDWARE mode without PYNQ. `PhysicalTwinBridge`
now exposes an honest deterministic EMULATION backend and an explicit
JSON-line TCP backend for hardware-twin services. `verify_hardware_link`
uses normal `PYTHONPATH` resolution for optional sister-repo probes and
does not mutate `sys.path`.

This page covers the three public symbols that drivers expose, what
each one actually does, and what each one only claims to do.

---

## 1. Public surface

`sc_neurocore.drivers.__init__` re-exports 3 symbols and declares the
research tier:

| Symbol | Source file | Role |
|--------|-------------|------|
| `SC_NeuroCore_Driver` | `sc_neurocore_driver.py` | PYNQ-Z2 FPGA overlay + AXI-Lite register access |
| `PhysicalTwinBridge` | `physical_twin.py` | Deterministic emulation bridge or explicit TCP hardware-twin client |
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
    (wrapped as `RealityHardwareError`)
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

## 3. `PhysicalTwinBridge`

`PhysicalTwinBridge` keeps the historical class name for API
compatibility, but it no longer pretends a mock is physical hardware.
It has two explicit modes:

- `mode="EMULATION"` (default) — deterministic local development mode.
  It uses a per-instance `numpy.random.default_rng(seed)` noise source,
  sets `connected=True` only for the local emulation backend, and writes
  no stdout on construction or divergence.
- `mode="TCP"` — real hardware-twin client mode. Each `sync_step(...)`
  opens a bounded TCP connection to `(ip, port)`, sends one compact
  JSON-line request, and requires a JSON-line reply containing numeric
  `v_mem`.

Constructor:

```python
class PhysicalTwinBridge:
    def __init__(
        self,
        ip: str = "192.168.2.99",
        port: int = 5000,
        *,
        mode: str = "EMULATION",
        timeout_s: float = 1.0,
        seed: int = 42,
        noise_sigma: float = 0.01,
        divergence_threshold: float = 0.1,
    ) -> None: ...
```

TCP request payload:

```python
{"spike":1,"v_mem":0.5}
```

TCP reply payload:

```python
{"v_mem": 0.875}
```

Malformed replies raise `ValueError`. Connection failures raise
`ConnectionError`. Divergence warnings go through module logging, not
stdout, so library callers can silence or route them.

Regression coverage:
`tests/test_pynq_driver.py::TestPhysicalTwinBridge` verifies no stdout
side effects, deterministic emulation, the compact JSON-line TCP
contract, and fail-closed malformed hardware replies.

---

## 4. `verify_link` — multi-target diagnostic CLI

`verify_hardware_link.py` exposes a single function `verify_link()`
that runs three sequential checks:

| Step | Target | Mechanism | Failure mode |
|------|--------|-----------|--------------|
| 1/3 | PYNQ-Z2 / FPGA bitstream | `SC_NeuroCore_Driver(mode="HARDWARE")` | `RealityHardwareError` → "Simulation Mode" message |
| 2/3 | Evo 2 genomic interface | `from scpn_evo2_real_interface import Evo2RealInterface` then `evo.connect()` | `ImportError` → "module not found"; `OSError`/`ConnectionError` → "Server unreachable" |
| 3/3 | Opentrons OT-2 robot | `from scpn_opentrions_verify import OpentronsVerifier` then `ot2.ping()` | `ImportError` → "module not found"; `OSError` → "ERROR" |

### 4.1 Cross-repo `sys.path.append` removed (FIXED by task #31)

The previous version mutated `sys.path` to reach into a sibling
`SCPN-CODEBASE/HolonomicAtlas/src/interfaces/` directory. That
behaviour was fragile (it assumed the GOTM monorepo layout) and
violated the principle that library code shouldn't manipulate
import paths.

`verify_link()` now imports `scpn_evo2_real_interface` and
`scpn_opentrions_verify` via standard `PYTHONPATH` resolution.
If the modules are not on the path the probe reports
`"FAILURE: <module> not on PYTHONPATH"` cleanly without changing
import state. The probe also accepts `extras: bool = True`
(default) — pass `extras=False` to skip both sibling-repo probes
and check only the FPGA subsystem.

Regression coverage:
`tests/test_pynq_driver.py::TestVerifyHardwareLink` (4 tests):
extras=False FPGA-only output, extras=True full output, default
is True, `verify_link` does not mutate `sys.path`.

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
| HARDWARE mode dispatch | `_connect_to_fpga` in `__init__` | `test_driver_hardware_mode_fails_without_fpga`, `test_driver_hardware_mode_uses_install_fallback_bitstream`, `test_driver_hardware_mode_rejects_overlay_without_expected_ip`, `test_driver_hardware_mode_wraps_overlay_runtime_errors` |
| EMULATION mode dispatch | logger warning + skip hardware path | `test_driver_emulation_mode` |
| `write_layer_params` AXI-Lite path | `getattr(overlay, ...)`, Q16.16 register writes | `test_driver_write_layer_params`, `test_driver_write_layer_params_hardware_q16_16_encoding`, `test_driver_write_layer_params_hardware_rejects_missing_layer` |
| `run_step` EMULATION return | per-instance `numpy.random.default_rng(seed)` | `test_driver_run_step`, `TestRunStepDeterminism` |
| `RealityHardwareError` propagation | raised on PYNQ import / file / IP failures | `test_driver_hardware_mode_fails_without_fpga` |
| `PhysicalTwinBridge` EMULATION/TCP boundary | deterministic local backend plus JSON-line TCP exchange | `TestPhysicalTwinBridge` |
| `verify_link` CLI | `if __name__ == "__main__"` invokes it | `TestVerifyHardwareLink` covers callable behaviour |

---

## 7. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 3 symbols re-exported; HARDWARE/EMULATION dispatch tested |
| 2 | Multi-angle tests | ✅ PASS | `SC_NeuroCore_Driver`, `verify_link`, and `PhysicalTwinBridge` all have focused tests covering emulation, failure boundaries, deterministic behaviour, and TCP contract handling. |
| 3 | Rust path | N/A | I/O + AXI-Lite shim; no compute kernel |
| 4 | Benchmarks | N/A | Hardware register writes and bounded TCP I/O; no meaningful benchmark without physical hardware |
| 5 | Performance docs | N/A | Same as above |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ✅ PASS | `PhysicalTwinBridge` now separates deterministic emulation from real TCP hardware-twin mode, `run_step(EMULATION)` uses a per-instance RNG, `verify_hardware_link.py` no longer mutates `sys.path`, the undocumented `physical_twin.py` type-ignore marker was removed, and the optional PYNQ import uses a narrow `type: ignore[import-not-found]`. SPDX header on every file ✅. |

Net: driver public-surface audit is now PASS for the locally testable
x86 scope. HARDWARE-mode happy-path evidence still requires a physical
PYNQ-Z2 board and remains part of the separate physical validation
backlog.

---

## 8. Known issues (for the implementation)

### 8.1 `PhysicalTwinBridge` TCP contract (FIXED by task #30)

`PhysicalTwinBridge` no longer sets physical connection state without
I/O. EMULATION mode is explicit and deterministic; TCP mode uses a
bounded JSON-line contract with fail-closed malformed-reply handling.

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

### 8.3 `verify_hardware_link.py` `sys.path.append` (FIXED by task #31)

`verify_link()` no longer mutates `sys.path`. It accepts an
`extras: bool = True` parameter — pass `extras=False` to skip
the two sibling-repo probes and check only the FPGA subsystem.
See §4.1.

### 8.4 Optional PYNQ import typing boundary (FIXED)

`sc_neurocore_driver.py` imports optional PYNQ symbols only inside
HARDWARE-mode connection setup. The import keeps `# noqa: F401`
because `allocate` is intentionally imported with the PYNQ runtime
surface, and now uses the narrow `# type: ignore[import-not-found]`
marker required for hosts where PYNQ is unavailable.

Regression coverage:
`tests/test_pynq_driver.py::TestDriverSourceHygiene` asserts that the
driver source keeps the narrow marker and does not reintroduce the old
broad `type: ignore` form.

### 8.5 Q16.16 in the FPGA driver vs Q8.8 elsewhere (DOCUMENTED)

`write_layer_params` encodes parameters as `int(value * 65536)`,
which is Q16.16 (16 integer + 16 fractional bits). The compiler
(`equation_compiler.py`) uses Q8.8. If the FPGA register IPs are
re-spun to Q8.8, this multiplication needs to change to `* 256`.
Document the Q-format choice in the IP-block contract.

Regression coverage:
`test_driver_write_layer_params_hardware_q16_16_encoding` constructs
a fake overlay and verifies the HARDWARE path writes `gain` to offset
`0x10` and `threshold` to offset `0x14` using Q16.16 integer values.
`test_driver_write_layer_params_hardware_rejects_missing_layer`
verifies that absent layer IPs fail closed.

## 9. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_pynq_driver.py -v
```

Coverage breakdown:

| Test | What it checks |
|------|----------------|
| `test_driver_emulation_mode` | EMULATION mode constructs without raising |
| `test_driver_write_layer_params` | EMULATION path no-ops cleanly with both `gain` and `threshold` |
| `test_driver_write_layer_params_hardware_q16_16_encoding` | HARDWARE path writes Q16.16 `gain` and `threshold` values to the expected AXI-Lite offsets |
| `test_driver_write_layer_params_hardware_rejects_missing_layer` | HARDWARE path raises when the target layer IP is absent |
| `test_driver_run_step` | EMULATION returns shape-(16,) ndarray |
| `test_driver_hardware_mode_fails_without_fpga` | HARDWARE on x86 raises `RealityHardwareError` |
| `test_driver_hardware_mode_uses_install_fallback_bitstream` | Missing local bitstream resolves to installed PYNQ overlay path and loads that path |
| `test_driver_hardware_mode_rejects_overlay_without_expected_ip` | Loaded overlays without `scpn_layer_1_0` fail closed as `RealityHardwareError` |
| `test_driver_hardware_mode_wraps_overlay_runtime_errors` | Overlay loader runtime failures are wrapped as `RealityHardwareError` |
| `test_driver_invalid_mode` | `mode="WHATEVER"` raises `ValueError` |
| `TestRunStepDeterminism` | Same-seed reproducibility, sequence reproducibility, seed separation, global RNG isolation, default seed |
| `TestVerifyHardwareLink` | FPGA-only probe mode, full probe mode, default extras behaviour, no `sys.path` mutation |
| `TestPhysicalTwinBridge` | No stdout side effects, deterministic emulation, compact JSON-line TCP contract, malformed-reply failure |

Not covered:

- HARDWARE mode happy path — requires actual PYNQ-Z2; cannot be
  exercised on x86 (test_driver_hardware_mode_fails_without_fpga is
  the inverse)

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
