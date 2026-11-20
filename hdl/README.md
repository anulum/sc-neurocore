<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Hardware modules and native validation -->

# SC-NeuroCore HDL Modules

Synthesisable Verilog for the SC-NeuroCore neuromorphic datapath.
The source-pinned CI toolchain uses Yosys 0.70, SBY 0.70 and sv2v 0.0.13.

## RTL Modules

| Module | Description | Testbench | Formal Proof |
|--------|-------------|:---------:|:------------:|
| `sc_lif_neuron.v` | Fixed-point LIF neuron (16-bit state) | tb | sby |
| `sc_bitstream_encoder.v` | LFSR-based stochastic bitstream encoder | -- | sby |
| `sc_bitstream_synapse.v` | Bitwise AND synapse with popcount | -- | sby |
| `sc_dotproduct_to_current.v` | Popcount accumulator to synaptic current | tb | sby |
| `sc_firing_rate_bank.v` | Multi-neuron firing-rate counter bank | tb | sby |
| `sc_dense_layer_core.v` | Single dense-layer compute core | tb | sby |
| `sc_axil_cfg.v` | AXI-Lite configuration register file | tb | sby |
| `sc_dense_matrix_layer.v` | Matrix-level dense layer (multi-core) | tb | -- |
| `sc_dense_layer_top.v` | Top-level dense layer wrapper | -- | -- |
| `sc_neurocore_top.v` | AXI-Lite dense-layer measurement and pollable completion | tb | sby |

**8/10** modules have SymbiYosys formal jobs (`hdl/formal/*.sby`); job presence
does not establish that a proof has completed on the current source.
**7/10** modules have simulation testbenches (`hdl/tb_*.v`).

The integration wrappers `sc_dense_matrix_layer` and `sc_dense_layer_top`
have no formal job configuration; they require their own integration acceptance.

The full `sc_neurocore_top` regression and its AXI lifecycle bench run through
`pytest tests/test_neurocore_top_rtl.py -q`, included in the main CI test shards.
They use the actual encoders, neuron datapath, register bank and rate bank.
All configuration registers have deterministic reset values: zero drive,
leak and gain, a one-clock measurement window, 1 ms host metadata and its
Q16.16 unit reciprocal. Input and weight words are unsigned 16-bit LFSR
thresholds (`0xC000` is approximately 0.75); current endpoints, leak and gain
use signed Q8.8. Completion stays set after reads until reset or the next
accepted start, while the rate bank receives only the original one-clock
completion pulse. Neurons and LFSRs keep clocking between measurements;
start does not reinitialise their state. See the
[register map and lifecycle](../docs/hardware/SC_NEUROCORE_HARDWARE_MANUAL.md#3-register-map).

`sc_dotproduct_to_current` requires a positive `N_INPUTS` and maps each
population count to `y_min + trunc((y_max-y_min)*count/N_INPUTS)`. Division
truncates towards zero. Both endpoint orders and the complete signed
`DATA_WIDTH` range are supported; the binary point is preserved. The RTL
regression compares repository and packaged RTL with exact integer results for widths 1–64 and fan-ins 1–33,
including extreme endpoints and every reachable count. The formal harness
checks the full signed 16-bit domain for three inputs without an endpoint-order
assumption. Run `pytest tests/test_dotproduct_current_rtl.py -q` with Icarus
Verilog installed, or use `tb_sc_dotproduct_to_current.v` for a compact bench.

`sc_dense_matrix_layer` shares input bitstreams and assigns a distinct weight
row to each neuron. `PROBABILITY_WIDTH` controls packed encoder thresholds;
`DATA_WIDTH` and `FRACTION` control signed current and LIF words. The default
probability width equals the state width for existing direct RTL consumers.
IR emission selects 16-bit thresholds after converting the declared probability
format and supplies both encoder seed bases. See [compute graph hardware words](../docs/api/sc-ir.md#dense-hardware-words).
The native `test_ir_dense_hardware` regression compiles repository and packaged
copies with Icarus and checks currents, encoder bits, membrane state and spikes
cycle by cycle; `tests/test_ir_dense_rtl.py` repeats the installed Python-to-RTL
boundary against the bit-true Python neurons. These are simulations, not board
measurements or routed timing results.

## SHD inference core

`sc_shd_int_core.v` runs the SHD speech network with integer axonal delays
(140 counts → 128 → 128 → 20). `tools/shd_int_reference.py` is the integer
reference for that network, and the core matches it bit for bit on every
accumulator.

**Step and arithmetic:**
- One 2 ms input step is a fixed phase sequence. Layer 1 multiplies 4-bit counts
  by int8 weights; layers 2 and 3 add weight rows for spikes.
- Delays are 0..30 steps through 32-slot histories.
- The LIF update is `v − (v >>> 2) + acc·2^F`. A neuron fires at an integer
  threshold and its membrane then resets to 0.

**Sizing and refusal:**
- Every width comes from a written bound for inputs of at most 1024 steps and
  counts of at most 15, so nothing saturates.
- A longer or empty input raises `error` and does not run.

**Two modes, one result:**
- `ZERO_SKIP = 1` reads a weight row only for a non-zero input; `ZERO_SKIP = 0`
  reads every row.
- Both give identical accumulators. `row_reads` reports which rows were read.

**Weights and delays** are ROM contents loaded by `$readmemh` in a form
synthesis keeps. Bit-exactness of a synthesised netlist is verified
separately, by post-synthesis simulation.

**Tests:** `pytest tests/test_shd_int_core_cosim.py` cosimulates random and edge
networks against the reference with Icarus Verilog.

**Trained network:** `fixtures/shd_int_network.json`, with its provenance in
`fixtures/shd_int_network.README.md`. `tests/test_shd_int_network.py` runs it
on the core.

**Legacy model:** `sc_shd_top.v` and its Vmin/Q8.8 cells implement the earlier
binary-input model and are superseded by this core.

## Event-driven SHD core (`hdl/shd_event/`)

`hdl/shd_event/sc_shd_event_p.v` implements the event-driven integer SHD core
with `P` shared LIF lanes (`sc_lanes.v`), a 32-bucket axonal-delay time wheel
per hidden layer, and an idle group-word weight/delay load port (`load_we`,
`load_sel` 0..4, `load_addr`, `load_grp`, `load_data`). Imported from the
group's earlier design study as Golden Reference #0, it computes the exact same
20 signed accumulators as `tools/shd_int_reference.py`.

**Target leaf seam (filelist-only selection, zero target preprocessor branches):**
- `hdl/shd_event/generic.f` — behavioural 1P macro model (`tech/generic/sc_sram_macro.v`),
  tiled memory wrapper (`tech/generic/sc_mem.v`), and pass-through clock-gate leaf
  (`tech/generic/cells_clkgate.v`).
- `hdl/shd_event/ihp_sg13g2.f` — IHP SG13G2 1P macro selector and blackbox stubs
  (`tech/ihp_sg13g2/sc_sram_macro.v`, `tech/ihp_sg13g2/macro_stubs.v`), `sg13g2_lgcp_1`
  latch clock-gate leaf (`tech/ihp_sg13g2/cells_clkgate.v`), and shared
  `tech/generic/sc_mem.v`.
- `hdl/shd_event/xilinx_7series.f` — Xilinx 7-Series block-RAM-inferring memory leaf
  (`tech/xilinx_7series/sc_mem.v`) and shared `tech/generic/cells_clkgate.v`.

**Verification and synthesis evidence (simulation and synthesis only; no board or silicon run yet):**
- `tools/cosim_shd_event_core.py` and `pytest tests/test_shd_event_core.py` cosimulate
  `tb_sc_shd_event_p.v` over `generic.f` and `xilinx_7series.f` in Icarus Verilog and
  Verilator against `tools/shd_int_reference.py`.
- `hdl/reports/sc_shd_event_p_xc7z020_synth_xilinx.txt` records the Yosys 0.70
  `synth_xilinx` synthesis statistics for `xilinx_7series.f` (both hierarchical
  per-`sc_mem` block-RAM inference and flattened XC7Z020 top-level utilisation).
- `hdl/pynq/synth_shd_event_xc7z020.tcl` and
  `hdl/constraints/sc_shd_event_axi_ooc_100mhz.xdc` reproduce the Vivado v2025.2
  out-of-context synthesis and placed-and-routed implementation of
  `sc_shd_event_axi` (`P=8`, `THETA1=72`, `THETA2=51`, `V1_WIDTH=22`,
  `V2_WIDTH=17`, `CAP1=62`, `CAP2=39`, `WSHAPE=6`, `QSHAPE=12`) on
  `xc7z020clg400-1` at a `100 MHz` (`10.000 ns`) target:
  - **Synthesised (`vivado_sc_shd_event_axi_synth_{util,hier_util,timing}_xc7z020_100mhz.rpt`):**
    `12,974 / 53,200` LUTs (`24.39%`: `12,934` logic + `40` `RAMD64E`),
    `14,073 / 106,400` FFs (`13.23%`), `21 / 140` Block RAM Tiles
    (`19 RAMB36E1 + 4 RAMB18E1`, `15.00%`), `0` DSPs, `WNS = -6.887 ns`,
    `WHS = +0.173 ns` (`0` hold misses, `0` `check_timing` warnings).
  - **Placed and routed (`vivado_sc_shd_event_axi_routed_{util,hier_util,timing,drc}_xc7z020_100mhz.rpt`):**
    `12,957 / 53,200` LUTs (`24.36%`: wrapper `439`, `u_core` `12,520`),
    `14,082 / 106,400` FFs (`13.23%`: wrapper `906`, `u_core` `13,176`),
    `5,814 / 13,300` slices (`43.71%`), `21 / 140` Block RAM Tiles
    (`q1=9 RAMB36E1`, `q2=8 RAMB36E1`, `q3=2 RAMB36E1`, `w1=2 RAMB18E1`,
    `w2=2 RAMB18E1`), `0` DSPs, and setup `WNS = -3.343 ns` (`13.343 ns` implied
    minimum period: `74.95 MHz` `Fmax` and `+6.657 ns` setup slack at `50 MHz`,
    estimated from the 100 MHz run's WNS (not a run at that frequency); hold not
    closed out of context; routed `WHS = -0.689 ns` across `241` failing hold
    endpoints has worst path `S_AXI_WDATA[0]` -> `load_wdata_lo_reg_reg[0]/D`,
    where `0.500 + 0.924 - (1.660 + 0.035 + 0.418) = -0.689 ns` under
    `HD.CLK_SRC BUFGCTRL_X0Y0` and `set_input_delay -min 0.5`, and the summary
    report does not break the other `240` endpoints down). Routed DRC reports
    `0` errors and `44` warnings (`CHECK-3` ×2, `REQP-1839` ×20, and `REQP-1840`
    ×20 at their 20-violation cap where asynchronously reset registers drive
    `RAMB36E1`/`RAMB18E1` address and enable pins — not analysed by static
    timing — plus `RTSTAT-10` ×1 and expected OOC `ZPS7-1` ×1). Full Zynq PS+PL
    block-design timing, physical board execution, hardware latency, and power
    are not measured.
- **Measured clock-gate contract (`OPENROAD_CLKGATE` pass-through vs `sg13g2_lgcp_1` latch):**
  all 20 class accumulators and all 15 datapath, SRAM, wheel-push/pop, and overflow
  counters match bit-for-bit across both clock-gate leaves. Because `cg_root` gates `cclk`
  during `S_IDLE` and idle `S_WAIT` cycles (`in_input && !frame_valid`), under the real
  latch leaf `done` stays high across `S_IDLE` (versus a single-cycle pulse under the
  pass-through leaf) and the simulation-only `cycles` counter counts active `cclk` edges
  (`cycles_pass_through - 1` when `start` deasserts one cycle before the first `frame_valid`).
  Host wrappers must therefore edge-detect `done`.
- `hdl/shd_event/sc_shd_event_axi.v` wraps `sc_shd_event_p` in a 9-bit AXI4-Lite slave
  (`0x000..0x0FC`) keeping every `0x00..0x8C` offset compatible with `hdl/pynq/sc_shd_driver.py`
  except `0x08..0x10`. Core `rst_n` is driven by `S_AXI_ARESETN` alone, `done_sticky`
  edge-detects `done`, `hdl/formal/sc_shd_event_axi.sby` proves and covers the loader,
  reserved-offset, target-range, and stepping invariants, and `tools/cosim_shd_event_axi.py` /
  `tests/test_shd_event_axi.py` cosimulate count, binary, and negative AXI4-Lite flows against
  `tools/shd_int_reference.py`.
- `hdl/pynq/sc_shd_driver.py` adds integrity-checked `load_network` over
  `hdl/fixtures/shd_int_network.json` (`zlib.crc32` over the 64-bit little-endian group-word
  stream, verified against `LOAD_COUNT`, `LOAD_CRC32`, and `LOAD_STATUS.sealed`; detects
  accidental corruption without claiming authentication and validates per-selector
  `LOAD_TARGET` bounds before bus writes; `lanes` must equal the synthesised bitstream's `P`
  parameter — `8` on the default build — since a smaller value triggers `refused_range` while
  a larger value sends fewer words in the wrong layout and cannot be detected from the bus)
  plus bounded-poll event-mode inference (`run_event`, `classify_event`, and
  `classify(..., mode="event")`) routing boolean rasters to `SPIKE_IN_0..4` and `0..15`
  integer rasters to `FRAME_COUNT_0..17` with zero tail padding and zero writes to reserved
  `0x08..0x10` (`tests/test_shd_driver.py`).

### Event SHD AXI4-Lite Register Map (`hdl/shd_event/sc_shd_event_axi.v`)

| Offset | Access | Name | Description |
| --- | :---: | --- | --- |
| `0x00` | R/W | `CTRL` | W: `[0]=start` (requires idle, `sealed`, `1 <= T_ORIG <= 1024`); R: `[1]=busy`, `[2]=done` (sticky), `[3]=frame_ready`, `[4]=error`, `[5]=overflow`, `[6]=commit_refused` (sticky) |
| `0x04` | R/W | `T_ORIG` | `[15:0]` input step count (reset `250`, valid `1..1024`) |
| `0x08..0x10` | — | `RESERVED` | Reserved on the event bitstream (the event core's layer scaling is fixed at synthesis by its parameters); read `0`; writes refused with `SLVERR` |
| `0x14..0x24` | W | `SPIKE_IN_0..4` | Write-only (read `0`); 140 binary spike bits expanded to `{3'b000, b}` nibbles |
| `0x28` | W | `SPIKE_COMMIT` | Write-only (read `0`); commit 560-bit `frame_buf` when `frame_ready=1` and clear `frame_buf` upon capture |
| `0x2C..0x3C` | — | `RESERVED` | Read `0`; writes refused with `SLVERR` |
| `0x40..0x8C` | R | `OUT_V_0..19` | Twenty signed 32-bit class accumulators |
| `0x90` | R/W | `LOAD_CTRL` | W: `[0]=clear`, `[1]=auto_increment`, `[2]=seal`; R: `[1]=auto_increment` |
| `0x94` | R/W | `LOAD_TARGET` | `[2:0]=load_sel` (`0..4`), `[9:3]=load_grp`, `[17:10]=load_addr` |
| `0x98` | R/W | `LOAD_WDATA_LO` | Low 32 bits of 64-bit group word |
| `0x9C` | R/W | `LOAD_WDATA_HI` | High 32 bits; write commits 64-bit word when idle and `LOAD_TARGET` is in range, else refused with `SLVERR` |
| `0xA0` | R | `LOAD_COUNT` | Committed 64-bit group words since `LOAD_CTRL.clear` |
| `0xA4` | R | `LOAD_CRC32` | Reflected IEEE CRC-32 (`0xEDB88320`, little-endian bytes) over committed words (integrity check only, not authentication) |
| `0xA8` | R/W | `EXPECTED_CRC32` | Expected 32-bit CRC written before `LOAD_CTRL.seal` |
| `0xAC` | R | `LOAD_STATUS` | `[0]=sealed`, `[1]=crc_mismatch`, `[2]=refused_busy`, `[3]=refused_range` (sticky until `LOAD_CTRL.clear`) |
| `0xB0..0xB4` | — | `RESERVED` | Read `0`; writes refused with `SLVERR` |
| `0xB8..0xFC` | W | `FRAME_COUNT_0..17` | Write-only (read `0`); 140 4-bit channel counts (`FRAME_COUNT_17[31:16] == 0` enforced via `SLVERR`) |
