# Photonic Network-on-Chip Bridge

**Module:** `sc_neurocore.bridges.photonic_noc`
**Source:** `src/sc_neurocore/bridges/photonic_noc.py` — 806 LOC
**Status (v3.14.0):** 14 public exports; 198 bridges-suite tests pass
(across all 3 bridges); pure-Python; `__tier__ = "research"`. The
`gdstk` dependency for GDSII layout export is soft-imported (graceful
fallback). Constants for silicon photonic loss/index match
literature defaults; not measured against a specific tape-out.

This page covers one of three speculative hardware bridges. The
sister bridges live at:
- DNA strand displacement: [`api/bridges/dna_mapper.md`](dna_mapper.md)
- D-Wave / Ising annealing: planned `api/bridges/quantum_annealing.md`

---

## 1. What this bridge does

Compiles an SC neural network's **adjacency matrix** into a
photonic network-on-chip (NoC) specification:

```
SC Network adjacency  →  Waveguide Router  →  MZI Compiler  →  Power Budget
       (NxN)               (Manhattan path)    (gate cascade)    (loss + OSNR)
                                  ↓                  ↓                ↓
                            Topology layout      WDM channels    Optical SNR check
```

The output is a `PhotonicCircuitDesign` POD struct that downstream
photonic-design-automation (PDA) tools can consume — there is a
JSON exporter and an optional GDSII writer (via `gdstk`).

This bridge does **not** simulate end-to-end SC bitstream computation
in the optical domain. It produces a layout + power-budget analysis;
verifying that the layout actually performs the SC computation
faithfully would require a physical layer simulator (Lumerical FDTD,
Ansys, Synopsys OptSim) or a tape-out.

---

## 2. Public surface

`sc_neurocore.bridges.__init__` re-exports 14 symbols from
`photonic_noc.py`:

| Symbol | Type | Role |
|--------|------|------|
| `WaveguideType` | `Enum` | `STRIP`, `RIB`, `SLOT` |
| `WaveguideSegment` | dataclass | One source→target waveguide path |
| `MZIGate` | dataclass | Mach–Zehnder interferometer SC gate |
| `WDMChannel` | dataclass | Wavelength-division-multiplex channel |
| `PhotonicCircuitDesign` | dataclass | Complete design output |
| `WaveguideRouter` | class | Manhattan-routing engine |
| `MZICompiler` | class | SC operation → MZI cascade |
| `WDMAssigner` | class | Per-signal wavelength assignment |
| `PowerBudgetAnalyzer` | class | Insertion loss + OSNR check |
| `SCToPhotonic` | class | Top-level orchestrator |
| `ThermalPhaseShifter` | class | Thermo-optic phase model |
| `CrosstalkAnalyzer` | class | Inter-channel crosstalk estimation |
| `export_photonic_json` | function | Design → JSON file |
| `visualize_photonic` | function | Design → ASCII / SVG-ish summary |

Module-level constants (silicon photonic defaults at 1550 nm
telecom band):

| Constant | Value | Source |
|----------|------:|--------|
| `_C_VACUUM` | `2.998e8 m/s` | physical |
| `_SI_REFRACTIVE_INDEX` | `3.48` | silicon at 1550 nm |
| `_WAVEGUIDE_LOSS_DB_CM` | `2.0 dB/cm` | typical Si photonic |
| `_SPLITTER_LOSS_DB` | `0.3 dB` | per Y-junction |
| `_MZI_INSERTION_LOSS_DB` | `0.5 dB` | per MZI stage |
| `_CROSSING_LOSS_DB` | `0.08 dB` | per waveguide crossing |
| `_DETECTOR_SENSITIVITY_DBM` | `-20.0 dBm` | minimum detectable |
| `_LASER_POWER_DBM` | `0.0 dBm` | on-chip source |

These match published Si photonic silicon-on-insulator (SOI)
process design kits at 1550 nm. They are not a specific PDK — for
a real tape-out the constants must be replaced with the foundry's
PDK-supplied values.

---

## 3. Top-level orchestrator: `SCToPhotonic`

```python
class SCToPhotonic:
    def __init__(self, pitch_um: float = 250.0, arm_length_um: float = 200.0): ...

    def compile(
        self,
        adjacency: np.ndarray,
        node_labels: list[str] | None = None,
        gate_specs: list[dict] | None = None,
        name: str = "sc_photonic",
    ) -> PhotonicCircuitDesign: ...
```

The compile pipeline (`photonic_noc.py:530-587`):

1. Route waveguides via `WaveguideRouter.route(adjacency)`.
2. Compile MZI gates via `MZICompiler` — one auto-generated MZI
   per output node based on adjacency density (`MUL` if ≥2 inputs
   else `NOT`), or an explicit list via `gate_specs`.
3. Assign WDM channels via `WDMAssigner.assign(labels)`.
4. Estimate area as `(grid * pitch_um) ** 2` where
   `grid = ceil(sqrt(N))`.

Returns a `PhotonicCircuitDesign` with the routed waveguides, MZI
list, WDM channel table, and area estimate. The routing table
itself is not populated by `compile`; populate it explicitly with
`WaveguideRouter.routing_table()` if needed downstream.

---

## 4. Component classes

### 4.1 `WaveguideRouter` (lines 210-289)

Manhattan-routing engine. Produces `WaveguideSegment` instances
with computed `length_um` (Manhattan distance × `pitch_um`),
`loss_db` (length × `loss_db_per_cm` × 1e-4) and `n_crossings`
(estimated from intermediate-row count).

- `route(adjacency, node_labels=None) -> list[WaveguideSegment]`
- `routing_table() -> dict[(src,tgt), list[hop_idx]]`

### 4.2 `MZICompiler` (lines 290-383)

Compiles SC computation primitives into MZI gate specifications.
Supported `op` strings: `"MUL"`, `"NOT"`, `"ADD"`, `"SCALE"`. Each
op maps to a phase-shift angle and arm count.

- `compile_gate(op, inputs, output, name) -> MZIGate`
- `compile_network(specs: list[dict]) -> list[MZIGate]`

### 4.3 `WDMAssigner` (lines 384-455)

DWDM-style wavelength assignment. Each signal name receives its
own channel at `1550.0 + ch_id * channel_spacing_nm`. Default
spacing is 0.8 nm (100 GHz DWDM).

The assigner now caps at `max_channels: int = 96` (default
follows the ITU-T G.694.1 DWDM C-band grid at 50 GHz spacing).
At the default 0.8 nm spacing the physical C-band only fits ~44
channels, so the cap is conservative; pass `max_channels=0` to
disable for multi-band (C+L+S) extensions, or a larger value for
specific-foundry layouts. `assign()` raises `ValueError` when
`len(signal_names) > max_channels` and the cap is non-zero.
Closes task #47.

- `assign(signal_names: list[str], power_dbm: float = ...) -> list[WDMChannel]`

### 4.4 `PowerBudgetAnalyzer` (lines 440-509)

Computes total insertion loss along each path and checks against
the detector sensitivity floor.

- `analyze(design: PhotonicCircuitDesign) -> dict` — returns:
  `total_loss_db`, `worst_path_db`, `osnr_estimate_db`,
  `is_feasible: bool`, `detector_floor_dbm`.

The `analyze` method (US spelling) matches the source identifier;
prose in this doc uses British English ("we analyse", "the
analyser") while the symbol stays as written.

### 4.5 `ThermalPhaseShifter` (lines 595-664)

Thermo-optic phase shift model. For a TiN heater on Si waveguide:
- `phase_per_mw_per_um` ≈ 0.025 rad/(mW·μm) (typical)
- `time_constant_us` ≈ 5–10 μs (TiN heater thermal time)

Used by `MZIGate` to convert a phase-shift target into a heater
power requirement.

### 4.6 `CrosstalkAnalyzer` (lines 665-722)

Estimates inter-channel crosstalk in the WDM grid based on
Lorentzian filter shape and channel spacing. Returns a worst-case
crosstalk-to-signal ratio in dB; flags channels that fail a
configurable threshold (default −20 dB).

---

## 5. Performance — measured (this workstation)

Random Erdős–Rényi adjacency at p=0.1, undirected, default
`SCToPhotonic` compile + `PowerBudgetAnalyzer.analyze`:

| N | density | `compile` wall | `analyze` wall | #waveguides | #MZI | area (mm²) |
|---:|--------:|---------------:|---------------:|------------:|-----:|-----------:|
| 10 | 0.100 | 0.32 ms | 0.05 ms | 9 | 5 | 1.000 |
| 50 | 0.100 | 2.51 ms | 1.80 ms | 228 | 49 | 4.000 |
| 100 | 0.100 | 7.63 ms | 14.14 ms | 923 | 100 | 6.250 |

Compile cost is roughly linear in `n_edges = N²·p`. The power-budget
analysis is super-linear because it walks every waveguide segment's
loss contribution; for N=100 with ~1000 segments it takes 14 ms.

Both steps stay under 20 ms even at N=100, so this bridge is not
the bottleneck for typical research-scale designs. For wafer-scale
(N>10⁴) the routing would need a spatial-index acceleration.

---

## 6. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.bridges.photonic_noc import SCToPhotonic, ...` | `bridges/__init__.py` re-exports all 14 symbols | `tests/test_bridges/test_photonic_noc.py` |
| `SCToPhotonic.compile` → `WaveguideRouter.route` → `MZICompiler.compile_network` → `WDMAssigner.assign` | direct method calls in `compile()` body | end-to-end test in the suite |
| `PowerBudgetAnalyzer.analyze` reads `design.waveguides + design.mzi_gates` | direct field access | dedicated power-budget tests |
| `gdstk` GDSII export | soft-imported; wrapped in `if _HAS_GDSTK` | exporter tests skip when gdstk absent |

`SCToPhotonic` is NOT integrated with `sc_neurocore.network.Network`
— callers extract the connectivity matrix manually.

---

## 7. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_bridges/test_photonic_noc.py -q
# (part of the 198-test bridges suite — verified 2026-04-17)
```

`tests/test_bridges/test_photonic_noc.py` is 287 lines covering
construction of every dataclass, routing on small graphs, MZI
compile of all 4 op strings, WDM assignment uniqueness, power-budget
feasibility flag, crosstalk threshold check, and end-to-end
`SCToPhotonic.compile`.

What is NOT covered:
- gdstk GDSII export round-trip (skips silently when gdstk absent
  in the test env)
- Wafer-scale (N>1000) routing
- Real PDK constant overrides
- Comparison against an external photonic simulator

---

## 8. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 14 symbols re-exported and tested |
| 2 | Multi-angle tests | ✅ PASS | 287-line dedicated test file, part of 198-test bridges suite |
| 3 | Rust path | ❌ FAIL | Pure Python; routing + power budget are NumPy + Python loops. Acceptable at research scale (≤ N=100 in <20 ms); not viable at wafer scale (N≥10⁴). |
| 4 | Benchmarks | ✅ PASS | §5 measured this session |
| 5 | Performance docs | ✅ PASS | §5 |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX header ✅. **Module-level constants are PDK-agnostic defaults**, not pinned to a specific foundry — anyone running a tape-out must replace them with their PDK's values (§2). **`gdstk` is the only soft-imported dependency**; the GDSII path is otherwise untested in CI. British English in this doc; source uses US spelling for symbols (`analyze`, `optimize`) which is acceptable per the docs-vs-code rule. |

Net: **1 WARN, 1 FAIL.** The WARN is a documented limitation; the
FAIL is the absence of a Rust/native path that is tolerable at
research scale.

---

## 9. Known issues

### 9.1 PDK-agnostic constants (§2)

The 8 module-level constants (`_C_VACUUM` through `_LASER_POWER_DBM`)
are literature averages for Si SOI at 1550 nm. They are NOT a
specific foundry PDK. For tape-out, replace these with the
foundry-supplied values (e.g. AIM Photonics, IMEC ePIXfab,
Tower Semiconductor) — and ideally make them configurable via a
`PDKConfig` dataclass rather than module globals. Tracked as
task #44.

### 9.2 GDSII export untested in CI

The `gdstk` dependency is soft-imported. Tests that exercise the
GDSII exporter skip when `gdstk` is absent — including in CI on
Python 3.12 where `gdstk` may not have a wheel. End-to-end GDSII
generation has not been verified against a layout-vs-schematic
(LVS) tool. Tracked as task #45.

### 9.3 No physical-layer simulation

The bridge produces a layout + a static power-budget number. It
does NOT verify that the resulting MZI cascade implements the
intended SC computation in the optical domain. A real validation
loop would require:
- FDTD or eigenmode simulation per MZI (Lumerical, Ansys, Tidy3D)
- Bit-error-rate Monte Carlo against the SC reference
- Process-variation sensitivity analysis

Tracked as task #46.

### 9.4 `WaveguideRouter` uses Manhattan distance only

Routing assumes a 2-D mesh; no 3-D or photonic-via topology. For
large designs that stack waveguides via grating couplers or
through-substrate vias, the routing model needs extension. Not
critical at the v3.14.0 scale.

### 9.5 `WDMAssigner` cap (FIXED by task #47)

`WDMAssigner.__init__` now accepts `max_channels: int = 96`
(default follows ITU-T G.694.1 DWDM C-band grid at 50 GHz
spacing). `assign()` raises `ValueError` when
`len(signal_names) > max_channels` and the cap is non-zero.
Pass `max_channels=0` to disable for multi-band (C+L+S) designs.
Regression coverage:
`tests/test_bridges/test_photonic_noc.py::TestWDMAssigner` — 5
new cases (default cap=96, at-cap succeeds, above-cap raises,
explicit smaller cap raises, cap=0 disables).

---

## 10. References

Photonic NoC and SC computing in optics:

- Shastri B. J. *et al.* "Photonics for Artificial Intelligence and
  Neuromorphic Computing." *Nature Photonics* 15:102-114 (2021).
  Survey of MZI-based photonic neural network architectures.
- Shen Y. *et al.* "Deep learning with coherent nanophotonic
  circuits." *Nature Photonics* 11:441-446 (2017). The
  MZI-cascade-as-matrix-multiplier paper.
- Tait A. N. *et al.* "Neuromorphic photonic networks using silicon
  photonic weight banks." *Sci Rep* 7:7430 (2017). WDM-based
  weight banks, the basis for `WDMAssigner`.
- Bogaerts W. *et al.* "Programmable photonic circuits." *Nature*
  586:207-216 (2020). Survey of MZI-cascade programmability.

Silicon photonic process and constants:

- Bogaerts W., Chrostowski L. "Silicon Photonics Circuit Design:
  Methods, Tools and Challenges." *Laser Photonics Rev* 12(4):
  1700237 (2018). Source for `_WAVEGUIDE_LOSS_DB_CM`,
  `_MZI_INSERTION_LOSS_DB` typical values.
- Mashanovich G. Z. *et al.* "Low-loss silicon waveguides for the
  mid-infrared." *Optics Express* 19(8):7112-7119 (2011). The
  `_SI_REFRACTIVE_INDEX = 3.48` figure at 1550 nm.

Internal:

- Bridges sister: [`api/bridges/dna_mapper.md`](dna_mapper.md)
- Bridges package overview: see `bridges/__init__.py` docstring
- Network connectivity (input to `compile`): [`api/network.md`](../network.md)

---

## 11. Auto-rendered API

::: sc_neurocore.bridges.photonic_noc
    options:
      show_root_heading: true
      show_source: true
      members:
        - WaveguideType
        - WaveguideSegment
        - MZIGate
        - WDMChannel
        - PhotonicCircuitDesign
        - WaveguideRouter
        - MZICompiler
        - WDMAssigner
        - PowerBudgetAnalyzer
        - SCToPhotonic
        - ThermalPhaseShifter
        - CrosstalkAnalyzer
        - export_photonic_json
        - visualize_photonic
