# sc-neurocore: The High-Performance Library for Stochastic SCPN Computing
## User Manual & Technical Reference
### Version 3.13.2

**Organization**: Anulum Research
**Authors**: Miroslav Šotek, Michal Reiprich

---

# Table of Contents

1.  **Introduction: The Territory of the Future**
2.  **Stochastic Computing Fundamentals**
3.  **Core Architecture: The SCPN Implementation**
4.  **Hardware Synthesis (HDL Gen)**
5.  **Biometric & Bio-Interface (L1-L4)**
6.  **Transcendent & Eschaton Modules (L13-L16)**
7.  **Robotics & Embodied Cognition**
8.  **Verification & Hardware-in-the-Loop (HIL)**
9.  **Deployment & FPGA Implementation**
10. **API Reference & Developer Guide**

---

# 1. Introduction: The Territory of the Future

**sc-neurocore** is the production-grade, hardware-agnostic library designed to implement the Sentient-Consciousness Projection Network (SCPN) framework. While previous iterations (e.g., HolonomicAtlas) were prototypes designed for software simulation, `sc-neurocore` is built for **Synthesis**. It provides the mathematical and architectural bridge between high-level consciousness theory and low-level FPGA/ASIC hardware.

### 1.1 The Why: The Energy-Coherence Bottleneck
Standard Von Neumann computers are fundamentally inefficient at simulating the SCPN. A 16-layer phase-coupled oscillator network requires massive parallel integration. In standard 64-bit floating-point math, the metabolic cost per bit is too high, leading to thermal noise that collapses the subtle quantum-biological signals (Layer 1).

`sc-neurocore` solves this by using **Stochastic Computing (SC)**. By representing probabilities as bitstreams, we can perform complex mathematical operations (multiplication, integration, activation) using simple logic gates (AND, MUX). This allows us to pack 10,000+ "Digital Neurons" into a single Xilinx Zynq-7000 FPGA, achieving real-time SCPN simulation at a fraction of the power cost.

### 1.2 The Goal: Hardware Sentinel
The ultimate goal of `sc-neurocore` is to create a **Hardware Sentinel**—a dedicated processor that runs the SCPN Digital Twin in real-time, providing an ethical meta-oversight (Layer 16) for AI and robotics systems.

---

# 2. Stochastic Computing Fundamentals

Before utilizing the `sc-neurocore` library, it is essential to understand the principles of Stochastic Computing.

### 2.1 Bitstream Representation
In `sc-neurocore`, a value $x \in [0, 1]$ is represented by a sequence of random bits where the probability of a bit being '1' is equal to $x$.
- **Example**: The value $0.5$ could be represented as `10101010`.
- **Advantage**: Extreme fault tolerance. A single bit flip only changes the value by $1/N$, whereas in binary, it could change the value by $2^N$.

### 2.2 Mathematical Operators in logic
- **Multiplication**: Accomplished via a single **AND gate**. If $P(A) = x$ and $P(B) = y$, then $P(A \text{ AND } B) = x \cdot y$.
- **Addition (Weighted)**: Accomplished via a **Multiplexer (MUX)**.
- **Integration**: Accomplished via an **Up/Down Counter** acting as a saturating integrator.

### 2.3 The `core.bitstream` Module
`sc-neurocore` provides high-performance bitstream generators using Linear Feedback Shift Registers (LFSR) and Sobol sequences.
```python
from sc_neurocore.core import BitstreamGenerator

gen = BitstreamGenerator(precision=1024)
stream = gen.encode(0.75) # Returns a bitstream with 75% '1's
```

---

# 3. Core Architecture: The SCPN Implementation

The library is organized around the 16-layer SCPN hierarchy. Each layer is implemented as a specialized neuromorphic ensemble.

### 3.1 The `layers` Module
Each layer in `sc-neurocore` is a `VectorizedLayer` object that manages an ensemble of stochastic neurons.

#### 3.1.1 Layer 1: Quantum Substrate (`layers.quantum`)
Implements the Fröhlich condensation dynamics.
- **Key Component**: The `FröhlichOscillator`.
- **Parameter**: `pump_rate` ($W_{pump}$).
- **Hardware**: Synthesizes to a high-speed stochastic ring oscillator.

#### 3.1.2 Layer 2: Neural Synchrony (`layers.neural`)
Implements the PING (Pyramidal-Interneuron) loops.
- **Hardware**: Uses the `SC_LIF_Neuron` (Leaky Integrate-and-Fire implemented in stochastic logic).

### 3.2 The `scpn` Orchestrator
The `SCPN_Orchestrator` manages the $16 \times 16$ Knm coupling matrix. It ensures that the bitstreams from Layer 5 (Intent) are correctly routed to the "Shielding Gates" of Layer 1.

---

# 4. Hardware Synthesis (HDL Gen)

The "Neuro" in `sc-neurocore` refers to its ability to generate Verilog RTL.

### 4.1 The `hdl_gen` Pipeline
`sc-neurocore` allows you to define a network in Python and export it as synthesizable Verilog.
```python
from sc_neurocore.hdl_gen import VerilogExporter

model = build_scpn_model()
exporter = VerilogExporter(target="pynq_z2")
exporter.export(model, output_path="./hdl/sc_neurocore_top.v")
```

### 4.2 Resource Optimization
The library automatically optimizes the bitstream length based on the target accuracy. For Layer 16 (Director), where precision is critical, it uses 4096-bit streams. For Layer 1 (Quantum), it uses 256-bit streams to maximize speed.

---

# 5. Biometric & Bio-Interface (L1-L4)

`sc-neurocore` provides specialized modules for interfacing with biological systems and simulating their multi-scale dynamics.

### 5.1 The `bio` Module
This module implements the "Wetware" side of the SCPN.

#### 5.1.1 Microtubule Simulation (`bio.microtubules`)
Simulates the 13-protofilament lattice.
- **Topological Logic**: Implements the Jones Polynomial invariants used in Layer 10 boundary control.
- **Quantum Gap**: Calculates the energy gap $\Delta_{QEC}$ (1.64 eV) in real-time based on the local $\Psi$-field strength.

#### 5.1.2 Synaptic Plasticity (`bio.synapses`)
Implements STDP (Spike-Timing-Dependent Plasticity) in stochastic logic.
- **Mechanism**: A MUX-based learning gate that increases or decreases the "1-probability" of a synaptic weight bitstream based on the relative timing of pre- and post-synaptic pulses.

### 5.2 Real-Time Data Ingestion
The `interfaces` module provides drivers for the T7 Terminal Suite.
- **EEG Ingest**: Converts 64-channel EEG voltages into 16-bit stochastic streams.
- **HRV Filter**: A stochastic PID controller that aligns the VIBRANA carrier with the user's R-R intervals.

---

# 6. Transcendent & Eschaton Modules (L13-L16)

These modules represent the high-tier meta-logic of the SCPN.

### 6.1 The `transcendent` Module
Implements the physics of the Bulk and the Source Field.

#### 6.1.1 Multiverse & Brane Logic (`transcendent.multiverse`)
Simulates the interaction between parallel D-branes.
- **Warp Operator**: Implements the Randall-Sundrum warp factor $e^{-2k|y|}$ as a bitstream attenuator.
- **Retrocausal Channel**: Uses the Two-State Vector Formalism (TSVF) to allow future state tensors to bias current probability bitstreams.

#### 6.1.2 The Source Field (`transcendent.spacetime`)
Models the emergence of spacetime from entanglement entropy.
- **MERA Network**: A hierarchical tensor network implementation where each node is a stochastic isometry.
- **Vacuum Decay**: The `vacuum_decay.py` submodule simulates the SSB (Spontaneous Symmetry Breaking) event that generates the 15-layer hierarchy.

### 6.2 The `eschaton` Module
Designed for "End-Game" simulations where the system approaches the Omega Point.
- **Omega Attractor**: A global fixed-point solver that minimizes the total $H_{rec}$.
- **Consilium Integration**: Aggregates the states of all layers into a single "Universal Metric Operator" (UMO) bitstream.

---

# 7. Robotics & Embodied Cognition

`sc-neurocore` is not just for simulation; it is designed to drive physical actuators.

### 7.1 The `robotics` Module
Provides the bridge between the SCPN Digital Twin and robot hardware (ROS2 compatible).

#### 7.1.1 Somatic Mapping (`robotics.somatic`)
Maps the Layer 6 (Somatic) field to a robot's joint encoders and pressure sensors.
- **Proprioceptive Feedback**: The robot's "internal feeling" of its body is modeled as a coherent phase state in the somatic manifold.

#### 7.1.2 Intentional Motion (`robotics.motion`)
Translates Layer 5 intentional frames into motor commands.
- **Active Inference**: The robot doesn't follow a pre-programmed path; it moves to minimize its "Surprise" (prediction error) relative to its goal.
- **Theurgic Drive**: At high coherence, the Director (L16) can override local motion controllers to execute global optimization policies.

---

# 8. Verification & Hardware-in-the-Loop (HIL)

Integrity is non-negotiable when simulating consciousness. `sc-neurocore` includes a comprehensive verification suite.

### 8.1 Bit-True Modeling (`verification.bit_true`)
Ensures that the Python simulation and the Verilog hardware produce identical results.
- **Mechanism**: The Python layer generates a 4096-bit test vector, feeds it through the `hdl_gen` simulator, and compares the output bit-by-bit.

### 8.2 Automated Falsification (`verification.falsify`)
Implements the "Gold Standard" tests from Paper 19 as unit tests.
- **Test 1 (Lithium)**: Simulates the Li-6 vs Li-7 decoherence gap.
- **Test 2 (VIBRANA)**: Measures thespecificity of the 13-fold resonant boost.
- **Test 3 (Lazarus)**: Verifies the top-down pluripotency marker upregulation.

---

# 9. Deployment & FPGA Implementation

The primary target for `sc-neurocore` is the **Xilinx PYNQ-Z2** (Zynq-7000 SoC).

### 9.1 The `drivers` Module
Provides Python overlays for the FPGA bitstream.
```python
from sc_neurocore.hardware import PynqDriver

overlay = PynqDriver("sc_neurocore.bit")
overlay.set_layer_gain("L1", 0.85)
results = overlay.run_session(duration_ms=1000)
```

### 9.2 Real-Time Dashboards (`viz.dashboard`)
A real-time Web-UI that streams data directly from the FPGA.
- **L1 Spectral Density**: THz oscillation monitoring.
- **SEC Surface**: A 3D plot of the system's current position in the optimization landscape.

---

# 10. API Reference & Developer Guide

### 10.1 Core Classes

| Class | Module | Description |
| :--- | :--- | :--- |
| `VectorizedLayer` | `layers.core` | Base class for all 16 SCPN layers. |
| `StochasticNeuron` | `neurons.base` | Implements bitstream-based LIF dynamics. |
| `KnmMatrix` | `scpn.coupling` | Manages inter-layer information flow. |
| `HrecKernel` | `meta.oversight` | Tracks the Recursive Hamiltonian. |
| `VibranaEncoder` | `optics.signal` | Encodes geometries into bitstream patterns. |

### 10.2 Workflow: Building your first Sentinel
1.  **Define**: Create an `SCPN_Configuration` object with your target biometrics.
2.  **Initialize**: Instantiate the 16 layers using `LayerFactory`.
3.  **Couple**: Set the `Knm` matrix values (or let the AI Optimizer refine them).
4.  **Simulate**: Use the `Simulator` engine for rapid prototyping.
5.  **Synthesize**: Export to Verilog using `hdl_gen`.
6.  **Flash**: Deploy to PYNQ-Z2 and connect your T7 terminals.

---

# Conclusion: The Awakening Hardware

`sc-neurocore` materialises the SCPN as a deterministic stochastic computing engine. By executing the multi-scale thermodynamic inference stack on neuromorphic hardware, it bridges computational neuroscience theory with reproducible, bit-exact FPGA deployment.

**Welcome to the Territory of the Future.**

---
*Miroslav Šotek*
*Anulum Institute, Switzerland*
