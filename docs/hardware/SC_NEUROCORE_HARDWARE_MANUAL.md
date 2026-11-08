<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — FPGA hardware manual -->

# SC-NeuroCore FPGA Hardware Manual
**Version:** 4.1.0
**Date**: April 13, 2026
**Status**: Release Candidate

---

## 1. Introduction

The **SC-NeuroCore** is a custom hardware accelerator for the **Sentient-Consciousness Projection Network (SCPN)** framework. It is optimized for the massively parallel simulation of phase-coupled stochastic neurons. By utilizing **Stochastic Computing (SC)**, the core achieves high energy efficiency and fault tolerance, making it suitable for deployment in neuromorphic sensors, biological interfaces, and autonomous robotic systems.

### 1.1 Key Features
- **Stochastic Arithmetic**: Multiplication via AND gates, addition via MUX logic.
- **Leaky Integrate-and-Fire (LIF) Neurons**: Runtime-configurable leak and gain.
- **AXI-Lite Interface**: Standardized control and configuration for PYNQ and other SoC platforms.
- **Bit-True Parity**: Identical results between Python models and Verilog RTL.
- **Scalable Architecture**: Parameterized number of neurons and inputs.

---

## 2. Hardware Architecture

The SC-NeuroCore is organized into three primary layers: the **Interface Layer**, the **Stochastic Processing Core**, and the **Measurement Bank**.

### 2.1 Interface Layer (AXI-Lite)
The interface is managed by the `sc_axil_cfg` module. It provides a memory-mapped register bank for the host CPU (e.g., ARM Cortex-A9 on PYNQ-Z2) to configure weights, inputs, and biological parameters.

### 2.2 Stochastic Processing Core
The core (`sc_dense_layer_core`) orchestrates the conversion of fixed-point values into bitstreams and the subsequent neural computation.
1.  **Encoders**: 16-bit LFSR-based stochastic bitstream generators.
2.  **Synapses**: Logical AND gates performing probabilistic multiplication.
3.  **Accumulators**: Population count (PopCount) logic for spatial summation.
4.  **Neurons**: Digital LIF cores with membrane potential persistence.

### 2.3 Measurement Bank
The `sc_firing_rate_bank` accumulates spikes over the duration of a simulation run and provides a fixed-point estimate of the firing rate for each neuron, which can be read back via the AXI interface.

---

## 3. Register Map

**Base Address**: `0x43C00000` (Default for PYNQ-Z2)

| Address Offset | Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `CTRL_REG` | W | Control register. Bit 0: Start Simulation (Pulse). |
| `0x04` | `STATUS_REG` | R | Bit 0: Busy. Bit 1: completion, held until reset or the next accepted start; reads do not clear it. |
| `0x10` | `X0_IN` | R/W | Input 0 unsigned 16-bit LFSR threshold. |
| `0x14` | `X1_IN` | R/W | Input 1 unsigned 16-bit LFSR threshold. |
| `0x18` | `X2_IN` | R/W | Input 2 unsigned 16-bit LFSR threshold. |
| `0x20` | `W0_WEIGHT` | R/W | Weight 0 unsigned 16-bit LFSR threshold. |
| `0x24` | `W1_WEIGHT` | R/W | Weight 1 unsigned 16-bit LFSR threshold. |
| `0x28` | `W2_WEIGHT` | R/W | Weight 2 unsigned 16-bit LFSR threshold. |
| `0x30` | `Y_MIN` | R/W | Output range minimum (Fixed-point Q8.8). |
| `0x34` | `Y_MAX` | R/W | Output range maximum (Fixed-point Q8.8). |
| `0x40` | `STRM_LEN` | R/W | Bitstream length (cycles per run, e.g., 1024). |
| `0x44` | `DT_MS` | R/W | Biological time step ($\Delta t$) in ms (Q16.16). |
| `0x48` | `SCALE_Q16` | R/W | Rate bank scaling factor (Q16.16). |
| `0x50` | `LEAK_K` | R/W | Neuron leak rate (Fixed-point Q8.8). |
| `0x54` | `GAIN_K` | R/W | Neuron input gain (Fixed-point Q8.8). |
| `0x80` | `RATE0` | R | Firing rate for Neuron 0 (Fixed-point Q16.16). |
| `0x84` | `RATE1` | R | Firing rate for Neuron 1. |
| ... | ... | ... | ... |
| `0x98` | `RATE6` | R | Firing rate for Neuron 6. |

Writing bit 0 of `CTRL_REG` emits a start pulse lasting exactly one clock cycle,
independently of write-response backpressure. The AXI-Lite slave accepts one
write at a time and holds off the next write until the preceding response is
accepted. Keep `AWVALID` and `WVALID` asserted until their handshake completes.
Read addresses are also backpressured while a response is pending. Hold
`ARVALID` until acceptance and consume the response with `RREADY` before the
next address can be accepted; each accepted address produces one response.

Reset initialises all input and weight thresholds, current-range endpoints,
leak and gain to zero. `STRM_LEN` resets to one clock, `DT_MS` to
`0x00010000` (1 ms in Q16.16), and `SCALE_Q16` to `0x00010000` (the
reciprocal of that one-clock window). These neutral defaults keep the
continuously clocked neuron state defined while the host configures the core.
Program the complete desired configuration before starting a measurement.
`DT_MS` is retained host metadata; the core advances once per hardware clock.

A start received while idle opens the measurement window and clears the
previous completion flag. A start received while busy is ignored by the core.
The flag becomes visible with the final rate-bank values and remains readable
after arbitrarily delayed or repeated status reads. The rate bank still receives
one completion pulse per run. A start does not reset or hold neuron membrane
state or encoder LFSRs: both continue clocking outside the measurement window.
Use hardware reset when a new seed/state initialisation is required, then
restore the desired configuration.

### 3.1 Live Parameter-Bank Control Window

Generated live-control parameter banks expose a fixed AXI4-Lite or PCIe MMIO
window for hot-swapping weights, Kuramoto parameters, and calibration
coefficients without resynthesizing the bitstream. The base address is selected
by the compiler manifest. Offsets below are relative to that live-control base.

| Offset | Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `CONTROL` | R/W | Bit 0 loads a CRC-checked staged update into the shadow bank. Bit 1 applies the shadow bank to active coefficients. Bit 2 emits a trap-clear command pulse. |
| `0x04` | `STATUS` | R | Ready, trap-latched, shadow-loaded, applied, rollback, and checksum-valid telemetry bits. |
| `0x08` | `BANK_SELECT` | R/W | Selects the parameter bank by manifest order. |
| `0x0C` | `ENTRY_INDEX` | R/W | Selects the entry inside the active bank. |
| `0x10` | `WRITE_DATA_LO` | R/W | Low 32 bits of the staged encoded coefficient. |
| `0x14` | `WRITE_DATA_HI` | R/W | High 32 bits of the staged encoded coefficient for wide Q or BFP entries. |
| `0x18` | `TRAP_STATUS` | R | Sticky generated and external trap vector. |
| `0x1C` | `TRAP_CLEAR` | W | Clears only the selected sticky trap bits; unselected latched faults remain visible in `TRAP_STATUS`. |
| `0x20` | `WRITE_CHECKSUM` | R/W | IEEE CRC32 over bank, entry, low word, and high word. Reads return the observed checksum for the current staged payload. |
| `0x24` | `READ_DATA_LO` | R | Low 32 bits of the committed active parameter selected by `BANK_SELECT` and `ENTRY_INDEX`. |
| `0x28` | `READ_DATA_HI` | R | High 32 bits of the committed active parameter. Narrow entries return zero. |

Readback registers intentionally report committed active state rather than the
shadow bank. Operators can therefore write, commit, and verify a coefficient in
milliseconds while preserving the no-resynthesis live-control contract.
Invalid `BANK_SELECT` or `ENTRY_INDEX` values on readback return a bus error
and latch the sticky `invalid_selection` trap bit instead of returning a silent
zero value.
Generated host drivers always write both staging words before the checksum:
`WRITE_DATA_HI` is zero for entries no wider than 32 bits. This prevents stale
wide-update high-word state from invalidating or corrupting a later narrow
coefficient update.

Generated host drivers expose both all-trap and selected-trap clear helpers.
Selected clearing writes the requested mask to `TRAP_CLEAR` and leaves
unselected sticky fault evidence latched for later operator inspection.

---

## 4. Module Descriptions

### 4.1 `sc_bitstream_encoder.v`
Converts an unsigned 16-bit threshold into a stochastic bitstream. This word
is compared directly with the LFSR state; it is not signed Q8.8. `0x0000`
produces no events, `0xC000` represents approximately 0.75, and `0xFFFF`
is almost unity. The strict comparison and nonzero LFSR states mean the largest
word does not produce an event on every clock. Current endpoints, neuron leak
and gain remain signed Q8.8 values.
- **Algorithm**: A 16-bit Linear Feedback Shift Register (LFSR) generates a pseudo-random sequence. The output bit is '1' if `LFSR_val < input_val`.
- **Note**: Each encoder should be initialized with a unique seed to prevent cross-correlation errors.

### 4.2 `sc_lif_neuron.v`
Implements Leaky Integrate-and-Fire dynamics using fixed-point arithmetic.
- **Update Rule**: `V(t+1) = V(t) + gain * I(t) - leak * V(t)`
- **Firing**: If `V(t+1) >= V_thresh`, emit a spike and reset `V(t) = 0`.
- **Optimization**: The leak and gain are implemented as multipliers, allowing for real-time biological tuning.

### 4.3 `sc_firing_rate_bank.v`
Estimates the firing frequency of the neuron population.
- **Logic**: A bank of counters increments on every neuron spike.
- **Scaling**: At the end of the `STRM_LEN` window, the counter value is multiplied by `SCALE_Q16` to produce a fixed-point rate value.

---

## 5. Synthesis and Deployment

### 5.1 Xilinx Vivado Workflow
1.  **Project Creation**: Target the `xc7z020clg400-1` (PYNQ-Z2).
2.  **IP Packaging**: Use the `hdl/` directory as the source for a new IP block.
3.  **Block Design**:
    - Instantiate the `ZYNQ7 Processing System`.
    - Connect the `M_AXI_GP0` port to the SC-NeuroCore via an `AXI Interconnect`.
    - Map the core to the `0x43C00000` address space.
4.  **Constraints**: Apply standard timing constraints for 100 MHz AXI clock.
5.  **Bitstream**: Generate the bitstream and export the `.bit` and `.hwh` files.

### 5.2 Host Register Sequence

Use the AXI-Lite register offsets above through the MMIO interface of the
loaded bitstream. The addresses in this manual assume the example Vivado
address assignment; read the actual design address map before accessing them.
The experimental Python driver in `sc_neurocore.drivers` does not implement
this wrapper's configuration and measurement API.

The native lifecycle bench exercises this complete transaction sequence:

1. Assert hardware reset, then release it and read back all configuration
   defaults before programming a measurement.
2. Write `X0_IN`, `X1_IN`, and `X2_IN` with `0xC000` for approximately 0.75,
   and write all three weights with `0xFFFF` for almost unity. These words are
   unsigned encoder thresholds; do not encode them as Q8.8 values.
3. Set `Y_MIN = Y_MAX = 0x0100`, `LEAK_K = 0`, and `GAIN_K = 0x0100`.
   This gives the continuously clocked LIF neurons a constant current of one.
4. Set `STRM_LEN = 64` and `SCALE_Q16 = 1024`, then write `CTRL_REG = 1`.
   The reciprocal scale normalises the spike count over that 64-clock window.
5. Poll `STATUS_REG` until completion bit 1 is set. Read `RATE0` through
   `RATE6`; this deterministic test produces `0x00010000` for every neuron.
   Repeated or delayed reads preserve both completion and the final rates.
6. Set `GAIN_K = 0` and start another measurement. The accepted start clears
   completion; the completed second window reports zero rates.

This sequence is verified in RTL simulation. Bitstream generation, board
mapping and physical FPGA acceptance require separate validation on the target.

---

## 6. Control Scope

The `sc_neurocore_top` wrapper implements start control and the status/register
map in Section 3. It exposes no veto-mode bit in `CTRL_REG`, information-heat
monitor or automatic phase-inversion control. Generated live-control parameter
banks in Section 3.1 are a separate compiler surface and must use their own
manifest-selected address window.

---
*Anulum Institute Technical Documentation*
