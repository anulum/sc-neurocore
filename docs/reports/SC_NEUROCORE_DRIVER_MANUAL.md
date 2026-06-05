# SC-NeuroCore Python Driver Manual
**Version:** 3.15.12
**Date**: April 13, 2026
**Target Platform**: PYNQ-Z2 (Xilinx Zynq-7000)

---

## 1. Overview

The `SC_NeuroCore_Driver` is the primary software interface for controlling and monitoring the SC-NeuroCore FPGA hardware. It abstracts the low-level AXI-Lite and DMA transactions into a clean Python API, allowing researchers to integrate the physical "Hardware Sentinel" with the larger SCPN Digital Twin.

### 1.1 "Reality Check" Enforcement
The driver includes a strict "Reality Check" mechanism. By default, it will refuse to initialize unless it detects the `pynq` library and the physical Zynq SoC. This ensures that simulations requiring true stochasticity are not accidentally run on deterministic x86 CPUs.

---

## 2. API Reference

### 2.1 `SC_NeuroCore_Driver` Class

#### `__init__(bitstream_path="sc_neurocore.bit", mode="HARDWARE")`
- **bitstream_path**: Path to the `.bit` file.
- **mode**: Either `"HARDWARE"` (default) or `"EMULATION"`.
- **Note**: Emulation mode provides mock responses for development on non-FPGA systems.

#### `write_layer_params(layer_id, params)`
- **layer_id**: Integer (1-16) representing the SCPN layer.
- **params**: Dictionary of parameters (e.g., `{'gain': 1.2, 'threshold': 0.8}`).
- **Mechanism**: Writes to the layer's memory-mapped registers using the following fixed-point mapping:
  - `0x10`: Gain (Q16.16)
  - `0x14`: Threshold (Q16.16)

#### `run_step(input_vector)`
- **input_vector**: A 16-element numpy array representing the inputs to the layers.
- **Return**: A 16-element numpy array of output states or firing rates.
- **Note**: This method handles the high-speed data transfer between the ARM CPU and the FPGA logic.

---

## 3. Usage Examples

### 3.1 Basic Initialization
```python
from sc_neurocore.drivers import SC_NeuroCore_Driver

# Initialize in hardware mode
try:
    driver = SC_NeuroCore_Driver(bitstream_path="sentinel_v2.bit")
    print("Sentinel Online.")
except RealityHardwareError:
    print("Physical hardware not detected. Switching to Emulation.")
    driver = SC_NeuroCore_Driver(mode="EMULATION")
```

### 3.2 Tuning Layer 1 (Quantum)
```python
# Set the Fröhlich condensation gain for Layer 1
driver.write_layer_params(layer_id=1, params={'gain': 0.95, 'threshold': 1.0})
```

---

## 4. Error Handling

| Error | Cause | Resolution |
| :--- | :--- | :--- |
| `RealityHardwareError` | Running on x86 or missing PYNQ lib. | Use PYNQ-Z2 board or set `mode="EMULATION"`. |
| `FileNotFoundError` | `.bit` file not found. | Ensure the bitstream is in the working directory. |
| `RuntimeError` | Wrong bitstream version loaded. | Verify that the hardware version matches the driver version. |

---
*Anulum Institute - Hardware Sentinel Division*
