# FMEA — SNN Compiler Pipeline (IR → SystemVerilog → FPGA)

Failure Mode and Effects Analysis for the SC-NeuroCore IR compiler
targeting safety-adjacent neuromorphic deployments.

## Scope

IR graph construction → verification → SystemVerilog emission → Yosys synthesis → FPGA bitstream.

## Severity Scale

| Level | Description |
|-------|-------------|
| S1 | Silent wrong output — spikes differ from golden model |
| S2 | Detectable wrong output — assertion/checker catches it |
| S3 | Build failure — synthesis or simulation abort |
| S4 | Performance degradation — correct but slow |

## Failure Modes

| ID | Stage | Failure Mode | Cause | Effect | Severity | Detection | Mitigation |
|----|-------|-------------|-------|--------|----------|-----------|------------|
| F01 | IR Build | Duplicate node ID | User adds node twice | Graph integrity violation | S3 | `ir_verify` rejects | Builder enforces unique IDs |
| F02 | IR Build | Unconnected input port | Missing `add_edge` | Neuron receives zero current | S1 | Formal property check | Connectivity analysis in verifier |
| F03 | IR Build | Weight overflow | Q8.8 value > 127.996 | Wraparound to negative | S1 | Range check in `to_sc_weights()` | Clamp + warning on export |
| F04 | IR Verify | Invalid fan-in | >256 inputs per neuron | LUT explosion on FPGA | S4 | Verifier fan-in limit | Configurable max_fanin param |
| F05 | SV Emit | LFSR polynomial mismatch | Emit uses different tap config than Python | Bitstream decorrelation fails | S1 | Co-simulation checker | Single source of truth for polynomial |
| F06 | SV Emit | Signed/unsigned mismatch | Q8.8 treated as unsigned in Verilog | Wrong arithmetic | S1 | Formal equiv check | `$signed()` annotation in template |
| F07 | SV Emit | Reset state mismatch | Verilog initial values differ from Python | First-cycle divergence | S2 | Co-sim golden-model check | Shared reset constants |
| F08 | Synthesis | Timing violation | Critical path through weight matrix | Clock constraint failure | S3 | Yosys/Vivado STA report | Pipeline registers on MAC |
| F09 | Synthesis | Resource overflow | Network exceeds FPGA capacity | Place-and-route failure | S3 | Resource estimation pre-check | LUT/BRAM budget calculator |
| F10 | Runtime | Metastability | Async input crossing clock domain | Glitch on spike bus | S1 | Double-FF synchronizer | CDC lint (Spyglass/Verilator) |
| F11 | Runtime | Refractory counter underflow | Off-by-one in counter | Extra spike per burst | S1 | Property-based test (Hypothesis) | Counter ≥ 0 assertion in RTL |
| F12 | Training | Gradient explosion | Surrogate β too large | NaN weights after export | S1 | NaN guard in `to_sc_weights()` | Gradient clipping + β schedule |
| F13 | Runtime | Silent accumulator saturation | Mixed Q8.8/Q16.16 or block-floating MAC exceeds accumulator range | Saturated output could be mistaken for valid control signal | S1 | `overflow_vector`, same-cycle `trap_event_vector`, sticky `trap_vector`, optional `SC_NEUROCORE_ASSERTIONS` | Saturate output, raise immediate trap event, latch per-lane trap until host clear |

## Residual Risk

After all mitigations, residual risk concentrates in F05 (LFSR mismatch),
F06 (signed/unsigned), and F13 (operator response to a latched saturation
trap). F05 and F06 are covered by the co-simulation checker that compares
Python golden-model spike trains against Icarus Verilog output bit-for-bit
across 10 000 timesteps with 5 LFSR seeds. F13 is covered by the
precision-overflow trap contract: a transient accumulator saturation is
visible immediately through `trap_event_vector` and persists through
`trap_vector` until host clear, preventing silent wash-out.

## Review Schedule

Re-assess after each compiler feature addition or Verilog template change.
