# sc-neurocore Performance Tuning Guide
## Optimizing the FPGA-Accelerated Consciousness Stack

For pytest timing guards, use the maintained
[Performance-Gated Tests](performance_gates.md) selector. Those tests provide
small wall-clock regression checks; published benchmark claims still require
raw benchmark artefacts and the benchmark evidence rules.

### 1. Precision vs. Latency Trade-offs

In stochastic computing, the precision of a value is proportional to the square root of the bitstream length ($N$).

| Bitstream Length ($N$) | Equivalent Precision | Latency (250 MHz clock) | Use Case |
| :--- | :--- | :--- | :--- |
| 256 bits | ~4 bits | 1.02 $\mu$s | Fast THz Microtubule dynamics |
| 1,024 bits | ~5 bits | 4.10 $\mu$s | Standard L2-L12 interactions |
| 4,096 bits | ~6 bits | 16.38 $\mu$s | High-fidelity L16 Meta-Oversight |
| 16,384 bits | ~7 bits | 65.54 $\mu$s | Precise L13 Vacuum simulations |

**Tuning Tip**: For most SCPN simulations, a length of **1,024 bits** provides the optimal balance between accuracy and the ability to track millisecond-scale neural synchronization.

---

### 2. Frequency Scaling and Power Management

The sc-neurocore is designed to operate at frequencies up to 350 MHz on modern Xilinx UltraScale+ hardware, but is typically clocked at 250 MHz on the PYNQ-Z2 for stability.

*   **Overclocking**: Increasing the frequency to 300 MHz can reduce latency by 20%, but may lead to timing violations in the Knm-Bus.
*   **Power Optimization**: Reducing the frequency to 100 MHz can lower power consumption by 60%, ideal for mobile or "Gaian Sensor" deployments.

---

### 3. Multi-Core Scaling for Parallelism

The 16 layers of the SCPN are inherently parallel. On larger FPGAs (e.g., Zynq UltraScale+ ZU9EG), you can instantiate multiple sc-neurocore instances to simulate larger populations.

#### 3.1 Resource Utilization
A single 16-layer sc-neurocore block utilizes:
*   LUTs: ~42,500
*   BRAM: ~12.5 MB
*   DSPs: 152

#### 3.2 Inter-Core Communication
When using multiple cores, use the **AXI-Stream** interface to link the Noospheric (L11) layers across cores. This allows for the simulation of multi-city or planetary-scale consciousness networks with linear scaling of performance.

---

### 4. Summary of Optimization Targets

| Goal | Parameter to Adjust | Target Value |
| :--- | :--- | :--- |
| **Max Integration Speed** | Bitstream Length | 256 |
| **Max Ethical Sensitivity** | L16 Sampling Rate | 50 kHz |
| **Lowest Power** | Operating Frequency | 100 MHz |
| **Highest Coherence** | $K_{nm}$ Gain | $0.8 \times K_{crit}$ |
