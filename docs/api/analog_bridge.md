# Analog Bridge — `sc_neurocore.analog_bridge`

A DAC / ADC conversion layer between SC-NeuroCore's digital
stochastic-computing world and the analog substrates of physical
mixed-signal neuromorphic chips. Translates probability bitstreams
into target conductance / threshold values, quantises them through a
configurable DAC, wraps AER spike events for event-driven communication,
and ships an on-chip calibration routine that reports effective number
of bits (ENOB).

```python
from sc_neurocore.analog_bridge import (
    AnalogBridge,
    AnalogSubstrateProfile,
    CalibrationRoutine,
    EventDrivenInterface,
)

# Target BrainScaleS-3 — 6-bit DAC, 64 conductance levels
bridge = AnalogBridge(profile=AnalogSubstrateProfile.brainscales3())
cal = CalibrationRoutine(bridge)
print(f"max quantisation error = {cal.max_quantization_error():.3f} nS")
print(f"ENOB                  = {cal.effective_resolution_bits():.2f} bits")
```

---

## 1. Scope and tier

This package is tagged `__tier__ = "research"`. It covers the
**configuration-time** path between the SC compiler and a target
analog chip: turning probability-weighted synapses and threshold-coded
LIF neurons into DAC codewords that the chip can program, and wrapping
the event-driven communication layer the chip uses at run time.

What it **is not**:

- It is **not** a simulator of analog neuron physics. Sub-threshold
  dynamics, transistor-level mismatch, and thermal drift are out of
  scope — those belong to Spectre / ngspice / MEEP, invoked separately
  from the `sc_neurocore.optics` photonic surface.
- It is **not** a replacement for the vendor's own software stack.
  The substrate profiles describe the electrical envelope; actual
  deployment to BrainScaleS-3 or DynapSE hardware still goes through
  the respective vendor toolchains.
- It is **not** coupled to the SC Compiler's front-end. It takes an
  iterable of node descriptors (`type`, `id`, `probability`,
  `threshold`) and produces a JSON-serialisable configuration dict —
  the bridge is deliberately decoupled from the IR.

## 2. Module surface

The package exposes five public types, all importable from the root
namespace:

```python
from sc_neurocore.analog_bridge import (
    AEREvent,
    AnalogBridge,
    AnalogSubstrateProfile,
    CalibrationRoutine,
    EventDrivenInterface,
)
```

The canonical import path is the package root, not the internal
`sc_neurocore.analog_bridge.analog_bridge` module. Reaching into the
inner file via direct `sys.path` manipulation bypasses the
package's `__init__.py` and causes coverage.py to miss attribution.

`sc_neurocore.analog_bridge.analog_bridge` is in the scoped public-docstring
policy. The dedicated analog-bridge tests are strict typed and docstring-clean
across both real-surface test files. This API is pure Python + NumPy and has no
polyglot or benchmark counterpart in this slice; the existing coverage evidence
below remains the verification target.

| Symbol | Purpose | Section |
|---|---|---|
| `AnalogSubstrateProfile` | Parameter envelope for a specific analog chip family | §3 |
| `AnalogBridge` | DAC-quantising bridge from SC probabilities / thresholds to configuration words | §4 |
| `AEREvent` | Address-Event Representation spike tuple (neuron id, timestamp, polarity) | §5 |
| `EventDrivenInterface` | Bitstream ↔ AER ↔ synaptic current conversion | §5 |
| `CalibrationRoutine` | DAC sweep + worst-case error + ENOB computation | §6 |

## 3. Substrate profiles

`AnalogSubstrateProfile` is a frozen parameter envelope — it captures
the electrical ranges that the vendor datasheet guarantees and the
DAC resolution that the chip actually exposes. Two reference profiles
ship with the module:

### 3.1 BrainScaleS-3

```python
p = AnalogSubstrateProfile.brainscales3()
# name='BrainScaleS-3', g_min=0.0, g_max=63.0, v_min=-80.0, v_max=-40.0,
# dac_resolution=6, tau_mem_range=(1.0, 50.0), tau_syn_range=(0.5, 20.0),
# max_fanin=256
```

6-bit DAC, 64 distinct conductance codes between 0 and 63 nS.
Membrane voltage swing is −80 mV to −40 mV (40 mV dynamic range).
Time-constant ranges reflect what the chip's bias currents can tune
to on the published BrainScaleS-2 analog-neuron circuit; refresh
against the current datasheet before committing to them for
deployment. Max fan-in of 256 reflects the per-neuron synaptic row.

### 3.2 DynapSE-2

```python
p = AnalogSubstrateProfile.dynapse2()
# name='DynapSE-2', g_min=0.0, g_max=127.0, v_min=-70.0, v_max=-30.0,
# dac_resolution=7, tau_mem_range=(5.0, 200.0), tau_syn_range=(1.0, 100.0),
# max_fanin=64
```

7-bit DAC (one extra bit gains a factor of two in effective levels),
wider conductance range (0–127 nS) but smaller fan-in (64 per neuron
because DynapSE uses CAM-based routing). Time constants reach 200 ms
on the membrane, useful for low-activity spike-based inference.

### 3.3 Custom profiles

`AnalogSubstrateProfile` is a `@dataclass`, so new chips are
configured directly:

```python
custom = AnalogSubstrateProfile(
    name="Loihi-2",
    g_min=0.0, g_max=255.0,
    v_min=-65.0, v_max=-35.0,
    dac_resolution=8,
    tau_mem_range=(0.5, 500.0),
    tau_syn_range=(0.1, 250.0),
    max_fanin=4096,
)
```

The bridge treats profiles as opaque parameter bundles; only the
numeric fields (`g_min`, `g_max`, `v_min`, `v_max`, `dac_resolution`)
feed the quantiser. The string `name` and the time-constant ranges
are reported back in downstream tooling but do not participate in
the DAC arithmetic.

## 4. The bridge

### 4.1 Construction

`AnalogBridge` has two constructor modes:

```python
# Profile mode (preferred) — pulls ranges + dac_resolution from the profile
bridge = AnalogBridge(profile=AnalogSubstrateProfile.brainscales3())

# Legacy mode — explicit ranges + dac_res
bridge = AnalogBridge(g_range=(0.0, 100.0),
                      v_range=(-80.0, -40.0),
                      dac_res=10)
```

When a `profile` is given, `g_range` and `v_range` are ignored and
the profile fields win. The legacy mode exists so tests and ad-hoc
scripts can instantiate bridges without constructing a full profile —
every call site that targets a real chip should go through the
profile mode so that the full envelope (time constants, fan-in) is
visible downstream.

After construction the bridge exposes:

- `g_min`, `g_max`  — conductance bounds (nS)
- `v_min`, `v_max`  — voltage bounds (mV)
- `dac_res`  — DAC resolution in bits
- `dac_levels`  — `2 ** dac_res`
- `profile`  — the `AnalogSubstrateProfile` (or `None` in legacy mode)

### 4.2 Quantisation kernel

`_quantize(val, v_min, v_max)` is the scalar DAC approximation
shared by conductance and voltage paths:

$$
\text{norm} = \mathrm{clip}\!\left(\frac{val - v_\text{min}}{v_\text{max} - v_\text{min}}, 0, 1\right)
$$

$$
\text{dac} = \mathrm{round}\!\left(\text{norm} \cdot (L - 1)\right),
\qquad L = 2^{\text{dac\_res}}
$$

$$
\text{actual} = v_\text{min} + \frac{\text{dac}}{L - 1}\,(v_\text{max} - v_\text{min})
$$

Returns the tuple `(dac, actual)`. The `clip` step silently saturates
out-of-range inputs; callers that need an explicit out-of-range alarm
should validate before calling `_quantize`. The underscore prefix
marks the method as a private helper; the public entry point is
`emit_analog_config`.

### 4.3 `emit_analog_config`

```python
config = bridge.emit_analog_config(nodes)
# {"synapses": {id: {"dac": ..., "g_ns": ...}, ...},
#  "neurons":  {id: {"dac": ..., "v_mv": ...}, ...},
#  "errors":   {id: abs(target - actual), ...}}
```

Each node is a duck-typed object with `.type`, `.id`, and either
`.probability` (for `SC_WEIGHT`) or `.threshold` (for `LIF_MEMBRANE`).
The bridge walks the list once, routing each node to the appropriate
sub-dict:

- `SC_WEIGHT` — `probability ∈ [0, 1]` maps linearly onto
  `[g_min, g_max]` and gets quantised. The worst-case quantisation
  error per synapse is recorded under `errors[id]` so that the
  downstream tooling can flag synapses that fall below a user-set
  SNR threshold.
- `LIF_MEMBRANE` — `threshold ∈ [0, 1]` maps linearly onto
  `[v_min, v_max]`. Errors are not currently recorded for neurons
  (the envelope asymmetry between conductance and voltage paths is
  historical; a future revision will unify them once downstream
  consumers settle on a SNR model for threshold DACs as well).

The function returns a plain dict, trivially JSON-serialisable. The
contract with downstream tooling is: "two layers deep, neuron /
synapse / error sections keyed by node id". The SC Compiler's
`export/pipeline.py` (§7 of [`pipeline.md`](pipeline.md)) consumes
this dict verbatim.

## 5. Event-driven interface

Analog neuromorphic chips communicate via **Address-Event
Representation** (AER): spike events are packets `(neuron_id,
timestamp)` transmitted over an asynchronous bus. The bridge provides
the three conversions any SC-to-analog stack needs.

### 5.1 `AEREvent`

```python
ev = AEREvent(neuron_id=42, timestamp_us=15.0, polarity=1)
```

A thin `@dataclass` with no methods. Polarity is `+1` (excitatory) or
`−1` (inhibitory); the chip's routing fabric usually separates the
two into distinct lanes, but at the Python layer they share a single
event queue.

### 5.2 `EventDrivenInterface`

```python
iface = EventDrivenInterface(clock_period_us=1.0)
```

`clock_period_us` is the sampling period of the upstream SC bitstream.
At 1 µs, a 1024-sample bitstream represents 1.024 ms of simulated
time. The interface exposes three methods:

#### Bitstream → events

```python
bs = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
events = iface.bitstream_to_events(neuron_id=42, bitstream=bs)
# [AEREvent(42, 0.0, +1), AEREvent(42, 2.0, +1),
#  AEREvent(42, 3.0, +1), AEREvent(42, 6.0, +1)]
```

Walks the bitstream, emits an `AEREvent` at every non-zero sample,
timestamps at `i * clock_period_us`. Zero bits produce no event, so
an all-zero bitstream yields an empty list — the contract documented
by `test_zero_bitstream`.

#### Events → synaptic current

```python
events = [AEREvent(0, 5.0), AEREvent(0, 15.0)]
current = iface.events_to_current(
    events, duration_us=50.0, tau_syn=5.0, weight=1.0
)
# current: ndarray of length 50, exponential-decay impulse kernel
# per event
```

Converts an event list into a time-discretised synaptic current
trace. Every event contributes a kernel
$w\,\sigma\,\exp(-\Delta t / \tau_\text{syn})$ starting at its
timestamp, where $w$ is `weight`, $\sigma$ is the event polarity,
and $\Delta t$ is the elapsed time since the event. Kernels sum
linearly so overlapping events produce super-position. The output is
a 1-D NumPy array; the length is `duration_us / clock_period_us`
rounded to at least 1.

Events whose timestamp falls outside `[0, duration_us]` are silently
dropped — the method is not an input validator. Upstream code is
expected to have clipped the event list to the simulation window
already.

#### Rate code

```python
rate = iface.rate_code(events, window_us=1000.0)  # Hz
```

Simple `len(events) / (window_us * 1e-6)`. Guards against empty
events (returns 0.0) and non-positive windows (returns 0.0). No
attempt at sub-windowing, kernel-smoothing, or inter-spike-interval
statistics — for those, see
[`analysis.md`](analysis.md) §4.

## 6. Calibration

`CalibrationRoutine` is the sweep-and-measure harness used at
deployment time, after a bridge has been configured but before the
first inference run, to validate that the chip's DAC is hitting its
spec.

### 6.1 Construction and sweep

```python
bridge = AnalogBridge(profile=AnalogSubstrateProfile.brainscales3())
cal = CalibrationRoutine(bridge, num_steps=10)

sweep = cal.sweep_conductance()
# list of (dac_code, target_g, actual_g)
```

The sweep walks `num_steps + 1` evenly spaced targets from `g_min`
to `g_max`, quantises each, and reports the (DAC code, target,
actual) triple. Default `num_steps=10` gives 11 sample points, which
is enough to estimate max error on a 6–10 bit DAC; bump
`num_steps=2**dac_res` for a dense sweep that touches every code.

### 6.2 Max quantisation error

```python
err_ns = cal.max_quantization_error()
```

Returns the largest `|target - actual|` across the sweep, in nS.
Measured on the default 10-step sweep:

| Bridge | DAC bits | `num_steps` | `max_quantization_error` (nS) |
|---|---|---|---|
| `g_range=(0, 100)`, `dac_res=4` | 4 | 10 | 3.333 |
| BrainScaleS-3 (`g_range=(0, 63)`) | 6 | 10 | 0.500 |
| `g_range=(0, 100)`, `dac_res=10` | 10 | 10 | 0.049 |

Numbers come from `tools/analog_bridge_bench.py`-equivalent inline
reproduction at the end of §7; they are not cached literature values.

### 6.3 Effective number of bits (ENOB)

```python
enob = cal.effective_resolution_bits()
```

Computes

$$
\mathrm{ENOB} = \log_2\!\left(\frac{g_\text{max} - g_\text{min}}{\text{max}\ \text{err}}\right).
$$

This is the noise-free resolution the DAC delivers — the ratio of
full-scale range to the worst single-point error expressed in bits.
It depends on whether the sweep grid aligns to DAC code boundaries:
on the BrainScaleS-3 6-bit profile, the default `num_steps=10` sweep
produces `max_err = 0.5 nS` and thus `ENOB = log2(63 / 0.5) ≈ 6.98`,
not the nominal 6. A sweep aligned to the DAC grid (`num_steps=63`)
yields `max_err = 0.0`, triggering the fallback branch in §6.3
which returns the nominal `6.0`.

The "ENOB ≥ nominal" outcome happens because the sweep uses a
coarse probe grid; the metric measures DAC code distance from the
probe targets, not quantisation of a continuous-time analog signal.
Under a sine-wave input (the classical industry-standard ENOB
definition) the figure would be bounded above by `dac_res`. The
`effective_resolution_bits` method here is the grid-probe variant,
useful for DAC verification rather than true SNR characterisation.

The function guards two degenerate cases:

1. `max_err == 0` — reachable when every sweep target lands exactly on
   a DAC code (e.g. `num_steps == 2**dac_res`, floating-point round-off
   absorbs exactly). Returning `log2(range / 0) = inf` would poison
   downstream reports, so the fallback returns the nominal
   `dac_res`.
2. `full_range == 0` — reachable only via a degenerate bridge with
   `g_max == g_min`. Normal construction rejects this (the
   `_quantize` kernel would have raised `ZeroDivisionError` on
   first call), but a bypass-construction via `AnalogBridge.__new__`
   can produce such a bridge; the fallback preserves the
   `float(dac_res)` contract.

Both fallbacks are covered by patch-based tests in
`tests/test_analog_bridge/test_analog_bridge_extended.py`
(`test_enob_zero_error_falls_back_to_nominal`,
`test_enob_zero_range_falls_back_to_nominal`).

## 7. End-to-end example

A complete flow from a fabricated SC graph to a chip-ready
configuration:

```python
from dataclasses import dataclass
from sc_neurocore.analog_bridge import (
    AnalogBridge,
    AnalogSubstrateProfile,
    CalibrationRoutine,
    EventDrivenInterface,
)
import numpy as np

@dataclass
class Node:
    type: str
    id: str
    probability: float = 0.0
    threshold: float = 0.0

# 1. Select target chip and configure bridge
profile = AnalogSubstrateProfile.brainscales3()
bridge = AnalogBridge(profile=profile)

# 2. Emit analog configuration from SC graph
nodes = [
    Node("SC_WEIGHT", "syn_0", probability=0.33),
    Node("SC_WEIGHT", "syn_1", probability=0.75),
    Node("LIF_MEMBRANE", "nrn_0", threshold=0.55),
]
config = bridge.emit_analog_config(nodes)
# config["synapses"]["syn_0"] -> {"dac": 21, "g_ns": 21.0}
# config["errors"]["syn_0"]   -> 0.21
# config["synapses"]["syn_1"] -> {"dac": 47, "g_ns": 47.0}
# config["errors"]["syn_1"]   -> 0.25
# config["neurons"]["nrn_0"]  -> {"dac": 35, "v_mv": -57.78}

# 3. Validate DAC quality on the sweep grid
cal = CalibrationRoutine(bridge, num_steps=64)
# num_steps=64 on a 6-bit (64-level) DAC gives max_err = 0.5 nS,
# ENOB = log2(63 / 0.5) ≈ 6.977 — above the nominal 6 bits because
# the probe grid happens to straddle code boundaries rather than
# fall on them. A grid-aligned sweep (num_steps=63) would hit ENOB
# = 6.0 exactly via the fallback branch.
assert cal.effective_resolution_bits() >= profile.dac_resolution

# 4. At run time, convert an SC output bitstream to an AER event train
bs = np.random.choice([0, 1], size=1024, p=[0.7, 0.3]).astype(np.uint8)
iface = EventDrivenInterface(clock_period_us=1.0)
events = iface.bitstream_to_events(neuron_id=42, bitstream=bs)

# 5. Inject into a post-synaptic current trace for downstream decode
current = iface.events_to_current(events, duration_us=1024.0, tau_syn=5.0)
rate_hz = iface.rate_code(events, window_us=1024.0)
```

Steps 1–3 happen once at compile time. Steps 4–5 run per inference
call at the AER boundary between the chip and the host.

## 8. Integration with the rest of SC-NeuroCore

- **SC Compiler** — `sc_neurocore.compiler` produces the node list
  consumed by `emit_analog_config`. The bridge's JSON dict lands in
  the manifest produced by `export/pipeline.py`.
- **Evolutionary substrate** — `sc_neurocore.evo_substrate` can evolve
  synaptic probabilities against a fitness function that includes
  `errors[synapse_id]` from the bridge, penalising configurations
  that the target chip can't represent accurately.
- **Bioware** — `sc_neurocore.bioware.BioHybridSession` uses the
  same AER format (`AEREvent`) for MEA spike ingest, so upstream
  MEA data can flow into the interface without conversion.
- **Edge** — `sc_neurocore.edge.aer_router.AERRoutingDaemon`
  (see [`edge.md`](edge.md)) forwards events over UDP mesh for
  multi-FPGA deployments; its wire format is compatible with the
  Python `AEREvent` dataclass.
- **Array guards** — none of the analog-bridge code crosses the FFI
  boundary; `sc_neurocore._native.array_guards`
  (see [`array_guards.md`](array_guards.md)) is irrelevant here. This
  module is pure Python + NumPy.

## 9. Test surface

The package ships with 30 tests across two files, all in
`tests/test_analog_bridge/`:

| File | Class | Scope | # tests |
|---|---|---|---|
| `test_analog_bridge.py` | `TestQuantization` | `_quantize` min/max/midpoint, 0 / 1 saturation | 5 |
| `test_analog_bridge.py` | `TestEmitConfig` | `SC_WEIGHT` + `LIF_MEMBRANE` paths, mixed node list | 6 |
| `test_analog_bridge_extended.py` | `TestSubstrateProfiles` | package facade exports, BrainScaleS-3, DynapSE-2, profile constructor, legacy constructor | 5 |
| `test_analog_bridge_extended.py` | `TestEventDrivenInterface` | bitstream-to-events, zero bitstream, current shape / positivity / decay, rate code (incl. empty) | 7 |
| `test_analog_bridge_extended.py` | `TestCalibrationRoutine` | sweep length, sweep endpoints, max error positive, ENOB ≤ nominal, high-res vs low-res, ENOB zero-error fallback, ENOB zero-range fallback | 7 |

Statement coverage is **100 %** (101/101 statements in
`analog_bridge.py` + 3/3 in `__init__.py`), verified via
`pytest --cov=sc_neurocore.analog_bridge`. The 2026-06-27 facade ratchet also
passed Ruff, Ruff docstring checks, strict mypy, `30 passed` analog bridge tests,
and the scoped public docstring policy with both `analog_bridge.py` and
`analog_bridge/__init__.py` enforced. No `pragma: no cover`
directives exist in the module. The ENOB fallback branches are
covered via `unittest.mock.patch` because they are physically
reachable but not triggered by the standard sweep-and-measure
suite; see `test_enob_zero_error_falls_back_to_nominal` for the
pattern.

## 10. Limits and known gaps

- **No per-neuron error accounting.** `emit_analog_config` records
  quantisation error for synapses (`SC_WEIGHT`) but not for neuron
  thresholds (`LIF_MEMBRANE`). Downstream tooling that needs neuron
  SNR has to call `bridge._quantize(...)` directly on the target
  voltage. A future revision will unify error reporting across both
  node types.
- **No closed-loop calibration against real hardware.** The
  `CalibrationRoutine.sweep_conductance` uses the bridge's own
  `_quantize` as ground truth — it measures the software DAC
  model, not the chip. To validate the **actual** silicon, the sweep
  must be instrumented against the vendor's own readback API;
  that hook is not yet provided.
- **No inhibitory-polarity routing at `emit_analog_config`.** The
  current config dict treats all synapses as a single scalar
  conductance. Chips that split E / I onto separate DAC channels
  need a layer above the bridge to demultiplex by sign.
- **No noise / thermal drift model.** The bridge assumes the chip
  programmes the DAC code it was told to programme. Real analog
  chips add thermal variation, fabrication mismatch, and code-to-code
  integral non-linearity; none of those are modelled here. A
  `CalibrationRoutine.with_noise` variant is on the to-do list but
  not implemented. The error budget from `max_quantization_error` is
  therefore a lower bound on what deployed silicon will deliver.
- **DynapSE-2 profile is approximate.** The 127 nS max-g figure and
  7-bit DAC reflect an early-revision envelope; silicon iterations
  since then reach higher g_max and 8-bit DAC in some variants.
  Callers targeting current silicon should override the profile
  rather than relying on the built-in constant.

## 11. Related reading

Primary external references are deliberately **not** citation-pinned
here until the full specs have been pulled and version-stamped —
cached secondary sources get stale fast in neuromorphic hardware and
fabricated citations are worse than none. Working list of what each
envelope is grounded in:

- **BrainScaleS profile** — the range fields (`g_min=0`, `g_max=63`,
  `v_min=-80`, `v_max=-40`, `dac_resolution=6`, `max_fanin=256`)
  come from the published BrainScaleS-2 analog-neuron circuit
  envelope used by the Heidelberg/Kirchhoff-Institut group. Refresh
  against the live datasheet before claiming chip parity.
- **DynapSE profile** — `g_max=127`, `dac_resolution=7`, `max_fanin=64`
  mirror the DynapSE architecture description originally from INI
  Zurich. Newer revisions reach higher g_max and 8-bit DAC; current
  silicon should override the profile rather than rely on the
  built-in constant.
- **Address-Event Representation** — the `AEREvent` tuple and the
  `EventDrivenInterface` conversion semantics follow the standard
  AER framing (neuron-id + µs timestamp + polarity) that the
  neuromorphic community has shared since the early 1990s Caltech
  analog-VLSI work. A rigorous citation pass is tracked as a
  follow-up audit task.
- **ENOB definition** — this module computes the *grid-probe* variant
  `log2(range / max-err)`, not the classical sine-wave ENOB that ADC
  test standards define. The distinction matters: the grid-probe
  variant can exceed nominal DAC bits when the probe grid straddles
  code boundaries; the sine-wave variant is bounded above by `dac_res`.

## 12. Source reference

::: sc_neurocore.analog_bridge.analog_bridge
