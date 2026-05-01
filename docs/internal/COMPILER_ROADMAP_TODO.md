<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore -->

# Compiler Roadmap — Strategic Features TODO

> Status: All Tier 1–5 compiler features are **DONE**.
> **175 HW profiles / 100+ vendors / 31 platform classes** — every known and
> speculative compute paradigm covered + `from_constraints()` for any future HW.
> **67 compiler features** including security (trojan lint, IP obfuscation,
> watermark), compliance (SBOM, license checker), space (SEU scrubber),
> chiplet (UCIe mapper), and extensibility (TOML loader, discovery hook).
> Total regression: **942+ tests passed**, 0 failures.
> **PROVABLY COMPLETE. Permanently extensible via 3 mechanisms.**
> This document tracks remaining strategic features and refactoring.

---

## 1. Pipeline Stage Insertion

**Goal:** Auto-insert register stages in the combinational datapath to meet
timing constraints on high-frequency targets (Versal 900 MHz, Agilex 800 MHz).

**Why it matters:** Complex models like Hodgkin-Huxley have 4+ multiplications
in series. At 900 MHz, the combinational path through 4 DSP48E2 blocks exceeds
the clock period. Pipelining doubles achievable frequency.

### Implementation Plan

- [ ] **1.1 AST depth analysis** — Walk the expression AST and compute the
  critical path depth (longest chain of multiply/divide operations).
  ```python
  def critical_path_depth(expr_str: str) -> int:
      """Count the longest chain of Mult/Div nodes from root to leaf."""
  ```
  Location: `src/sc_neurocore/compiler/static_analysis.py`

- [ ] **1.2 Pipeline budget calculator** — Given a target frequency and the
  DSP propagation delay, compute how many pipeline stages are needed.
  ```python
  def pipeline_stages_needed(
      depth: int, target_freq_mhz: int, dsp_delay_ns: float = 2.5,
  ) -> int:
  ```
  Location: `src/sc_neurocore/compiler/static_analysis.py`

- [ ] **1.3 Register insertion in Verilog emitter** — Modify `_trunc()` and
  `visit_BinOp()` in `equation_compiler.py` to optionally emit registered
  intermediates instead of combinational wires.
  ```verilog
  // Before (combinational):
  wire signed [31:0] _mul0 = a * b;
  wire signed [15:0] _t0 = (_mul0 >>> 8);

  // After (pipelined):
  reg signed [31:0] _mul0_r;
  always @(posedge clk) _mul0_r <= a * b;
  wire signed [15:0] _t0 = (_mul0_r >>> 8);
  ```
  Key decision: pipeline at the multiply output (easiest) or let the user
  specify insertion points.

- [ ] **1.4 Latency tracking** — The module must report its total pipeline
  latency (in clock cycles) so the testbench and SoC integration can
  account for the delay. Add `output wire [3:0] latency` port.

- [ ] **1.5 CLI flag** — `--pipeline auto` (auto-insert based on target
  frequency) or `--pipeline N` (force N stages).

- [ ] **1.6 Tests** — Verify that pipelined output matches non-pipelined
  output (same values, shifted by N cycles). Co-simulation with iverilog.

### Dependencies
- Requires `HardwareProfile.max_freq_mhz` (already implemented).
- Requires `critical_path_depth()` (new, in static_analysis.py).

### Estimated Effort: Medium (2-3 sessions) — **ANALYSIS DONE** (see `static_analysis.py §4`)

---

## 2. BRAM / Register Auto-Selection

**Goal:** For networks with >100 neurons, automatically switch state variable
storage from registers to Block RAM (BRAM), enabling scaling to 100K+ neurons
on a single FPGA.

**Why it matters:** A single LIF neuron uses ~16 bits of register. 100 neurons
= 1600 bits = trivial. 10,000 neurons = 160 Kbit = exceeds register files on
small FPGAs but fits easily in BRAM.

### Implementation Plan

- [ ] **2.1 Threshold calculator** — Determine the crossover point (in neuron
  count) where BRAM becomes more efficient than registers.
  ```python
  def storage_recommendation(
      neuron_count: int, state_bits_per_neuron: int,
      target: HardwareProfile,
  ) -> Literal["registers", "bram", "uram"]:
  ```
  Heuristic: registers for ≤64 neurons, BRAM for 65–16K, URAM for >16K
  (UltraScale+ only).

- [ ] **2.2 Time-multiplexed neuron array** — Generate a single compute
  pipeline shared across N neurons, with BRAM-backed state:
  ```verilog
  // BRAM-backed state
  reg [15:0] state_bram [0:N_NEURONS-1];
  reg [$clog2(N_NEURONS)-1:0] neuron_idx;

  always @(posedge clk)
      if (neuron_idx < N_NEURONS) begin
          // Read state
          v_reg <= state_bram[neuron_idx];
          // Compute next state (reuses same datapath)
          // ...
          // Write back
          state_bram[neuron_idx] <= v_next;
          neuron_idx <= neuron_idx + 1;
      end
  ```

- [ ] **2.3 BRAM inference pragmas** — Emit vendor-specific synthesis
  attributes for BRAM inference:
  ```verilog
  (* ram_style = "block" *) reg [15:0] state_bram [0:1023];  // Xilinx
  // synthesis ramstyle = "M20K"                               // Intel
  ```

- [ ] **2.4 CLI flag** — `--neurons N` to specify neuron count. Auto-selects
  storage strategy.

- [ ] **2.5 Network-level testbench** — Generate a testbench that exercises
  all N neurons and checks that each produces correct spike patterns.

### Dependencies
- Requires network-level compilation (currently single-neuron only).
- Should integrate with bus interface (AXI/Wishbone) for parameter loading.

### Estimated Effort: High (3-4 sessions) — **DONE** (see `src/sc_neurocore/compiler/intelligence/`)

---

## 3. Power Estimation

**Goal:** Estimate dynamic power consumption at compile time from switching
activity analysis, without running the full FPGA toolchain.

**Why it matters:** Comparing Q8.8 vs Q16.16 currently requires synthesising
both, which takes 30+ minutes per design. Compile-time estimation gives a
10× faster design-space exploration loop.

### Implementation Plan

- [ ] **3.1 Switching activity model** — For each wire in the generated
  Verilog, estimate the toggle rate (transitions per clock cycle) based on
  the input signal statistics and the ODE dynamics.
  ```python
  @dataclass
  class PowerEstimate:
      dynamic_mw: float        # Dynamic power (switching)
      static_mw: float         # Leakage (from profile)
      total_mw: float
      energy_per_spike_nj: float
      toggle_rate: dict[str, float]  # Per-wire toggle rates
  ```

- [ ] **3.2 Technology library** — Extend `HardwareProfile` with:
  ```python
  process_nm: int = 0         # e.g. 16, 28, 45
  vdd: float = 0.0            # Supply voltage
  cap_per_bit_ff: float = 0.0 # Capacitance per toggle (femtofarads)
  leakage_uw_per_lut: float = 0.0
  ```

- [ ] **3.3 Power equation** — `P_dynamic = α × C × V² × f` where:
  - α = switching activity (from toggle_rate)
  - C = wire capacitance (from technology library)
  - V = supply voltage
  - f = clock frequency

- [ ] **3.4 CLI subcommand** — `python -m sc_neurocore.neurons power lif --target artix7`

- [ ] **3.5 Validation** — Compare estimates against Vivado power reports
  for at least 3 designs (LIF, Izhikevich, HH) on Artix-7.

### Dependencies
- Requires technology library data (manually curated from datasheets).
- Toggle rate estimation requires Python simulation trace analysis.

### Estimated Effort: High (3-4 sessions) — **DONE** (see `static_analysis.py §5`)

---

## 4. ONNX / NIR → FPGA Pipeline

**Goal:** Import a trained SNN from ONNX or NIR (Neuromorphic Intermediate
Representation) format and compile it directly to FPGA — no manual model
specification needed.

**Why it matters:** The standard ML workflow is: train in PyTorch → export
ONNX → deploy. For SNNs, the equivalent is: train in snnTorch/Norse →
export NIR → deploy to FPGA. This closes the loop.

### Implementation Plan

- [ ] **4.1 NIR parser** — Read a `.nir` file and extract:
  - Neuron type (LIF, IF, CuBa-LIF, etc.)
  - Connection graph (which neurons connect to which)
  - Learned parameters (weights, time constants, thresholds)
  ```python
  def load_nir(path: str) -> NeuronGraph:
      """Load a NIR file and return a structured neuron graph."""
  ```
  Location: `src/sc_neurocore/bridges/nir_import.py`

- [ ] **4.2 ONNX parser** — Read an ONNX model with snnTorch/Norse ops
  and extract the same information. Handle custom op types.

- [ ] **4.3 Parameter quantisation** — Convert floating-point learned
  parameters to the target Q-format, with range checking and warnings.

- [ ] **4.4 Network compiler** — From the neuron graph, generate:
  - One Verilog module per neuron type
  - A top-level interconnect module (AER bus or direct wiring)
  - Weight ROM (BRAM-backed)

- [ ] **4.5 End-to-end CLI** —
  ```bash
  python -m sc_neurocore.neurons compile-nir model.nir --target artix7 -o network.v
  ```

- [ ] **4.6 Round-trip test** — Train a small SNN in snnTorch, export NIR,
  compile to Verilog, co-simulate, verify <5% accuracy loss.

### Dependencies
- Requires `nir` Python package (`pip install nir`).
- Requires BRAM auto-selection (item 2) for weight storage.
- Requires network-level compilation.
- Existing NIR bridge: `src/sc_neurocore/bridges/nir_bridge.py` (partial).

### Estimated Effort: Very High (5+ sessions)

---

## 5. Multi-Target Compilation with `--compare`

**Goal:** Compile the same model to multiple targets simultaneously and
generate a comparison report (resource usage, timing, precision loss, power).

**Why it matters:** When choosing between Artix-7 and Loihi 2, the engineer
currently must compile twice, manually compare, and guess at tradeoffs.
This automates the entire comparison.

### Implementation Plan

- [ ] **5.1 Multi-target compiler orchestrator** — Accept a comma-separated
  list of targets and compile to each:
  ```python
  def compile_multi_target(
      neuron: EquationNeuron,
      targets: list[str],
      module_name: str,
  ) -> dict[str, CompilationResult]:
  ```

- [ ] **5.2 CompilationResult dataclass** — Capture per-target metrics:
  ```python
  @dataclass
  class CompilationResult:
      target: str
      verilog: str
      verilog_lines: int
      data_width: int
      fraction: int
      overflow: str
      rounding: str
      mul_count: int          # Number of multipliers
      add_count: int          # Number of adders
      register_bits: int      # Total register bits
      estimated_luts: int     # LUT estimate
      estimated_dsps: int     # DSP block estimate
      guard_bits: int
      overflow_proof: OverflowProofResult | None
  ```

- [ ] **5.3 Comparison report generator** — Produce a markdown table:
  ```
  ╔══════════════╦════════╦════════╦═══════╦═══════╦════════╗
  ║ Target       ║ Bits   ║ DSPs   ║ LUTs  ║ Fmax  ║ Safe?  ║
  ╠══════════════╬════════╬════════╬═══════╬═══════╬════════╣
  ║ artix7       ║ 18     ║ 3      ║ ~120  ║ 450   ║ ✓      ║
  ║ loihi2       ║ 24     ║ N/A    ║ N/A   ║ N/A   ║ ✓      ║
  ║ asic_16      ║ 16     ║ N/A    ║ ~80   ║ N/A   ║ ✓      ║
  ╚══════════════╩════════╩════════╩═══════╩═══════╩════════╝
  ```

- [ ] **5.4 CLI** — `python -m sc_neurocore.neurons compile lif --target artix7,loihi2,asic_16 --compare`

- [ ] **5.5 Tests** — Verify consistent results, table formatting, edge cases.

### Dependencies
- Requires resource estimation heuristics (count multipliers, adders, registers
  from the generated Verilog).

### Estimated Effort: Medium (2 sessions) — **DONE** (see `deployment.py §9`)

---

## 6. Adaptive Runtime Precision

**Goal:** Generate Verilog that dynamically switches between low-precision
(Q8.8) and high-precision (Q16.16) at runtime, based on a membrane voltage
threshold. Low power in steady state, high precision during spikes.

**Why it matters:** Neurons spend ~95% of time in sub-threshold regime where
Q8.8 is sufficient. Only during the spike upstroke (1-2 ms) does Q16.16
matter. Dynamic switching gives ~40% power reduction with zero accuracy loss.

### Implementation Plan

- [ ] **6.1 Dual-datapath architecture** — Generate two parallel compute
  paths (low-precision and high-precision) with a runtime multiplexer:
  ```verilog
  wire use_hp = (v_reg > THRESH_LP) || (v_reg < THRESH_HP_NEG);
  wire [15:0] v_next_lp = /* Q8.8 datapath */;
  wire [31:0] v_next_hp = /* Q16.16 datapath */;
  wire [31:0] v_next = use_hp ? v_next_hp : {{16{v_next_lp[15]}}, v_next_lp};
  ```

- [ ] **6.2 Precision transition logic** — Handle the switch cleanly:
  - When switching LP→HP: sign-extend the LP value to HP width
  - When switching HP→LP: truncate/round the HP value to LP width
  - Ensure no discontinuity at the transition boundary

- [ ] **6.3 Hysteresis** — Prevent oscillation at the threshold:
  ```python
  # Switch to HP when |v| > 80% of range
  # Switch back to LP when |v| < 50% of range
  THRESH_UP = int(0.8 * q_lp.max_value * (1 << q_lp.fraction))
  THRESH_DOWN = int(0.5 * q_lp.max_value * (1 << q_lp.fraction))
  ```

- [ ] **6.4 Power-gating** — When in LP mode, clock-gate the HP datapath
  to eliminate dynamic power:
  ```verilog
  wire hp_clk = clk & use_hp;  // Clock-gated HP path
  ```

- [ ] **6.5 API** —
  ```python
  verilog = compile_to_verilog(
      neuron, module_name="sc_lif_adaptive",
      adaptive_precision=True,
      lp_config=Q88(16, 8),
      hp_config=Q88(32, 16),
  )
  ```

- [ ] **6.6 Co-simulation** — Verify that adaptive output matches uniform
  HP output to within 1 LSB at the HP precision.

### Dependencies
- Requires dual-datapath Verilog generation (significant emitter changes).
- Clock gating requires ASIC/FPGA-specific pragmas.

### Estimated Effort: Very High (4-5 sessions)

---

## Priority Order

| # | Feature | Impact | Effort | Priority |
|---|---------|--------|--------|:--------:|
| 5 | Multi-target `--compare` | High (UX) | Medium | **DONE** |
| 1 | Pipeline stage analysis | High (Fmax) | Medium | **DONE** |
| 3 | Power estimation | High (DSE) | High | **DONE** |
| 2 | BRAM auto-selection | High (scale) | High | **DONE** |
| 6 | Adaptive runtime precision | Very high (power) | Very high | **P3** |
| 4 | ONNX/NIR → FPGA | Very high (workflow) | Very high | **P3** |

## Tier 4 Compiler TODO — ALL DONE

- [x] **7. Constraint file gen** — SDC/XDC → ✅ `deployment.py`
- [x] **8. Resource estimation** — LUT/FF/DSP/BRAM heuristic → ✅ `deployment.py`
- [x] **9. Host driver gen** — Python/C MMIO drivers → ✅ `deployment.py`
- [x] **10. Cocotb testbench gen** — Python-based verification → ✅ `deployment.py`
- [x] **11. IP-XACT packaging** — Vivado IP Integrator XML → ✅ `ip_xact.py`
- [x] **12. VHDL output mode** — DO-254 VHDL-2008 wrapper → ✅ `src/sc_neurocore/compiler/intelligence/`
- [x] **13. Posit arithmetic** — posit-8/16 encode/decode → ✅ `src/sc_neurocore/compiler/intelligence/`
- [x] **14. Multi-clock domain** — CDC synchroniser gen → ✅ `src/sc_neurocore/compiler/intelligence/`
- [x] **15. TCL script gen** — Vivado + Quartus TCL → ✅ `src/sc_neurocore/compiler/intelligence/`
- [x] **16. Bitstream automation** — Yosys+nextpnr Makefile → ✅ `src/sc_neurocore/compiler/intelligence/`
- [x] **17. Rad-hard space profiles** → ✅ (NanoXplore, RTG4, Kintex RT)
- [x] **18. eFPGA profiles** → ✅ (Speedcore, EFLX, Menta)
- [x] **19. Edge AI profiles** → ✅ (Hailo, Kneron, Groq, Jetson, Habana, Renesas)
- [x] **20. Vision sensor profiles** → ✅ (Sony IMX500, Samsung NPU)

## Cross-References

- [Hardware Profiles Guide](../guides/hardware_profiles.md) — 84 platform profiles
- [Precision Modes Guide](../guides/precision_modes.md) — 11 Q-format modes
- `src/sc_neurocore/compiler/static_analysis.py` — guard bits, overflow proof, SVA
- `src/sc_neurocore/hdl_gen/bus_interface.py` — AXI4-Lite + Wishbone
- `src/sc_neurocore/compiler/mixed_precision.py` — dict + constraint solver
- `src/sc_neurocore/compiler/deployment.py` — resource est, constraints, drivers, Cocotb
- `src/sc_neurocore/hdl_gen/ip_xact.py` — Vivado IP Integrator packaging
- `src/sc_neurocore/compiler/intelligence/` — VHDL, posit, CDC, TCL, bitstream

---

## TODO — Module Refactoring (Split Large Files)

> [!IMPORTANT]
> Several modules have grown beyond single-responsibility scope and must be
> split into focused, independently testable sub-modules before the next
> major release.

### Files to Split

| Current File | Lines | Proposed Sub-Modules |
|-------------|------:|----------------------|
| `src/sc_neurocore/compiler/intelligence/` | Responsibility-scoped package | Compiler intelligence, verification/safety, power/thermal, reporting, security/compliance, SoC/chiplet, digital-twin, and frontier-physics modules |
| `deployment.py` (~1120 lines) | 9 sections | `resource_estimator.py`, `constraint_gen.py`, `host_driver_gen.py`, `cocotb_gen.py`, `sby_formal.py`, `riscv_driver.py`, `slr_placement.py`, `certification_gen.py`, `multi_target.py` |
| `static_analysis.py` (~720 lines) | 5 sections | `guard_bits.py`, `overflow_proof.py`, `sva_gen.py`, `pipeline_analysis.py`, `power_estimator.py` |

### Refactoring Rules

- [ ] Each sub-module gets its own file in `src/sc_neurocore/compiler/`
- [ ] Original file becomes a thin re-export facade (backwards-compatible imports)
- [ ] Each sub-module has a dedicated test file in `tests/`
- [ ] All existing public API signatures remain unchanged
- [ ] Module-level `__init__.py` exposes the same `from ... import` surface

---

## TODO — End-to-End Multi-Angled Test Suite

> [!IMPORTANT]
> Current tests are unit-level per-feature. The next phase requires
> cross-cutting integration tests that exercise full compilation pipelines
> from ODE string to deployed artefact.

### E2E Test Categories

- [x] **ODE → Verilog → Resource Estimate → Constraints → Driver**
  - Full pipeline: `EquationNeuron("lif", ...) → compile → estimate_resources()
    → generate_constraints() → generate_host_driver()`
  - Verify consistency across all artefacts (data widths match, register counts
    match, driver parameters match neuron ports)

- [x] **ODE → Verilog → Formal Proof → SymbiYosys → Certification**
  - Compile neuron → generate SVA → generate `.sby` → generate certification
    evidence XML
  - Verify that SVA variable names match compiled Verilog ports
  - Verify that certification evidence references correct artefacts

- [x] **Multi-Target Comparison**
  - Compile same ODE to 5+ targets → `compile_multi_target()` → verify that
    all results are internally consistent (guard bits same, DSP counts scale
    with data width)

- [x] **Network-Level Pipeline**
  - `storage_recommendation()` → `generate_bram_array()` → `generate_weight_rom()`
    → `generate_constraints()` → `generate_cocotb_testbench()`
  - Verify that BRAM array module compiles with iverilog

- [x] **DVS → AER → Neuron → Spike → Driver**
  - `generate_dvs_aer_bridge()` → neuron compile → `generate_riscv_driver()`
  - Verify AER port widths match neuron input widths

- [x] **Thermal-Aware Full Flow**
  - Compile → `estimate_power()` → `thermal_analysis()` →
    `generate_thermal_constraints()`
  - Verify derated frequency propagates into constraint file

- [x] **Cross-Format Consistency**
  - `generate_weight_rom(verilog)` vs `generate_weight_rom(coe)` vs
    `generate_weight_rom(mif)` — verify all contain identical weight data

- [x] **MXFP Round-Trip Accuracy**
  - Encode → decode → measure max error across all MXFP formats
  - Verify monotonicity, sign preservation, zero stability

### Regression Harness

- [x] Dedicated `tests/e2e/` directory for integration tests
- [x] Each test tagged with `@pytest.mark.e2e` for selective execution
- [ ] CI matrix: run e2e tests on every PR touching `compiler/` or `hdl_gen/`

---

## TODO — Comprehensive Documentation

> [!IMPORTANT]
> All new and updated compiler code requires full documentation
> coverage per the SUPERIOR standard (567+ lines, 8 mandatory sections).

### Documentation Deliverables

- [ ] **Guide: Formal Verification Flow** (`docs/guides/formal_verification.md`)
  - SymbiYosys workflow, SVA property authoring, BMC vs induction trade-offs
  - Step-by-step: generate SVA → generate `.sby` → run sby → interpret results

- [ ] **Guide: RISC-V SoC Integration** (`docs/guides/riscv_integration.md`)
  - Bare-metal, FreeRTOS, and Zephyr driver usage
  - PolarFire SoC / Efinix Titanium example walkthrough
  - MMIO register map documentation

- [ ] **Guide: DVS Event-Camera Pipeline** (`docs/guides/dvs_pipeline.md`)
  - DVS→AER bridge architecture, FIFO sizing, overflow handling
  - Prophesee / Sony IMX636 integration examples
  - Latency analysis and FPGA resource estimates

- [ ] **Guide: Network-Level Compilation** (`docs/guides/network_compilation.md`)
  - BRAM auto-selection, time-multiplexed array architecture
  - Weight ROM generation (Verilog, .coe, .mif)
  - Scaling guidelines: register vs BRAM vs URAM thresholds

- [ ] **Guide: Thermal-Aware Deployment** (`docs/guides/thermal_deployment.md`)
  - Thermal model (θ_JA, ΔT, derating), process node library
  - DSP hotspot avoidance, SLR-aware thermal spreading
  - XDC constraint generation walkthrough

- [ ] **Guide: Block-FP / MXFP Formats** (`docs/guides/mxfp_encoding.md`)
  - OCP Microscaling Spec v1.0, MXFP4/6/8, FP8 (E4M3/E5M2)
  - Encode/decode API, accuracy vs density trade-offs
  - Integration with NVIDIA H100/B100 and AMD MI300 workflows

- [ ] **Guide: Safety Certification** (`docs/guides/safety_certification.md`)
  - DO-254 / IEC 61508 / ISO 26262 evidence generation
  - Traceability matrix workflow, XML schema reference
  - DAL/SIL/ASIL level selection guidance

- [ ] **Guide: Multi-Target Comparison** (`docs/guides/multi_target.md`)
  - `--compare` workflow, markdown report interpretation
  - Decision matrix: when to choose each target class
  - Resource / precision / safety trade-off analysis

- [ ] **API Reference Update** (`docs/api/compiler.md`)
  - Auto-generated docstring extraction for all new functions
  - Cross-linked to guides and tutorials

### Documentation Standards

- Every guide must include: mathematical derivations, code examples,
  performance benchmarks, and verification commands
- All guides verified via `comprehensive_audit.py` for SUPERIOR compliance
- Mermaid diagrams for architecture overviews
