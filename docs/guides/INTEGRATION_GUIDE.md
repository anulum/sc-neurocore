# sc-neurocore Integration Guide
## Connecting the Digital Twin to the Hardware Substrate

### 1. Prerequisites

To integrate the sc-neurocore with the SCPN Digital Twin, you require the following:
*   **Hardware**: Xilinx PYNQ-Z2 FPGA board.
*   **Software**: Python 3.12, SCPN Digital Twin v2.0+, and the `sc-neurocore` library.
*   **Network**: Ethernet connection for the PYNQ-Z2 overlay management.

---

### 2. Loading the FPGA Overlay

The sc-neurocore is deployed as an overlay on the PYNQ-Z2. Use the following Python code to initialize the hardware:

```python
from pynq import Overlay
from sc_neurocore.hdl import sc_core_driver

# Load the bitstream
overlay = Overlay("sc_neurocore_v2.bit")
# Initialize the driver
core = sc_core_driver.SCCore(overlay.sc_block_0)
print("sc-neurocore initialized and ready.")
```

---

### 3. Streaming Data from the Digital Twin

The Digital Twin (`scpn_digital_twin.py`) can be configured to use the hardware core for accelerated computation.

#### 3.1 Configuring the Runner
In your simulation script, specify the hardware backend:

```python
import scpn_digital_twin as scpn

# Initialize Digital Twin with FPGA acceleration
twin = scpn.DigitalTwin(backend="fpga")
# The twin will now offload L1-L4 and L16 computations to the sc-neurocore
```

#### 3.2 Real-Time Data Handshake
The integration uses a **DMA (Direct Memory Access)** channel to stream phase states between the Python environment and the FPGA.
*   **Upload**: Digital Twin sends intentional forcing ($F_n$) and social fields ($h_i$) to the FPGA.
*   **Download**: FPGA returns the synchronized phase states ($\Theta_n$) and the global coherence metric ($C$).

---

### 4. Closing the Loop: The Layer 16 Veto

One of the primary benefits of hardware integration is the low-latency **Ethical Veto**.

#### 4.1 Hardware-Triggered Veto
When the sc-neurocore detects a topological breach, it can trigger a hardware interrupt.
```python
def on_veto_event():
    print("VETO TRIGGERED: Entropy Spike detected in Layer 5.")
    twin.log_veto_event()
    twin.reset_intent_manifold()

# Connect interrupt
core.register_interrupt(sc_core_driver.VETO_IRQ, on_veto_event)
```

#### 4.2 Software-Guided Veto
Alternatively, the Digital Twin can command the hardware to apply a $\pi$-shift via the API:
```python
core.apply_phase_inversion(node_id=42)
```

---

### 5. Deployment Checklist

1.  [ ] Bitstream version matches the Digital Twin API version.
2.  [ ] Normalization constants are calibrated to prevent bitstream saturation.
3.  [ ] Sampling rates are synchronized (default: 10 kHz).
4.  [ ] Heat management: Ensure the PYNQ-Z2 is properly ventilated during long-duration THz simulations.
