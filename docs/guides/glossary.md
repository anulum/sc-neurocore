# Glossary

Domain-specific terms used throughout SC-NeuroCore documentation.

---

## Stochastic Computing

| Term | Definition |
|------|-----------|
| **Bitstream** | Temporal sequence of {0,1} bits where the proportion of 1s encodes a probability p ∈ [0,1]. |
| **Bitstream length (L)** | Number of bits in a stochastic bitstream. Precision scales as O(1/√L). |
| **Unipolar SC** | Bitstream encodes values in [0,1]. Multiplication via AND gate. |
| **Bipolar SC** | Bitstream encodes values in [-1,1] via p=(x+1)/2. Multiplication via XNOR gate. |
| **Popcount** | Count of 1-bits in a packed word. Used to decode bitstream probability. |
| **CORDIV** | Stochastic division circuit (Li et al. 2014). Sequential state machine: x=1→z=1, y=1→z=0, else hold. |
| **MUX** | 2:1 multiplexer for scaled addition: z = s·x + (1-s)·y. |
| **LFSR** | Linear Feedback Shift Register. Pseudo-random number generator for bitstream encoding. |
| **Sobol sequence** | Low-discrepancy quasi-random sequence. Gives O(1/L) convergence vs O(1/√L) for Bernoulli. |
| **Decorrelation** | Requirement that bitstreams use independent random sources. Correlated inputs bias results. |
| **Q8.8** | Signed fixed-point format: 8 integer bits + 8 fractional bits. Range [-128, +127.996], step 1/256. |
| **SC-aware pruning** | Remove weights whose bitstream contribution falls below noise floor at target L. |

## Spiking Neural Networks

| Term | Definition |
|------|-----------|
| **LIF** | Leaky Integrate-and-Fire neuron. τ_m dV/dt = -(V-V_rest) + R·I. Spike when V > V_th. |
| **Izhikevich** | Two-variable neuron model: dv/dt = 0.04v² + 5v + 140 - u + I. 20+ firing patterns. |
| **Hodgkin-Huxley** | Biophysical neuron with Na⁺/K⁺ ion channel gating (1952 Nobel Prize model). |
| **AdEx** | Adaptive Exponential integrate-and-fire. LIF + exponential spike initiation + adaptation. |
| **FitzHugh-Nagumo** | Reduced 2-variable model: fast voltage + slow recovery. Limit cycle dynamics. |
| **Hindmarsh-Rose** | 3-variable bursting model with slow modulation variable z. |
| **Surrogate gradient** | Smooth approximation of the Heaviside spike function for backpropagation (FastSigmoid, ATan). |
| **STDP** | Spike-Timing-Dependent Plasticity. Pre-before-post → potentiation. Post-before-pre → depression. |
| **STP** | Short-Term Plasticity. Facilitation (repeated spikes increase release) or depression (vesicle depletion). |
| **E-prop** | Eligibility propagation. Three-factor rule: pre × post × error. Online credit assignment. |
| **R-STDP** | Reward-modulated STDP. Eligibility traces gated by global reward signal. |
| **BCM** | Bienenstock-Cooper-Munro rule. Sliding threshold metaplasticity. |
| **Population** | Group of identical neurons managed as a unit. |
| **Projection** | Synaptic connectivity between populations. Stored as CSR sparse matrix. |
| **CSR** | Compressed Sparse Row format: (indptr, indices, data) arrays for sparse matrices. |
| **Spike gating** | Skip idle neurons during simulation. Compute ∝ active count. |

## FPGA and Hardware

| Term | Definition |
|------|-----------|
| **LUT** | Look-Up Table. Basic FPGA logic element. One LIF neuron ≈ 30–50 LUTs. |
| **FF** | Flip-Flop. 1-bit register on FPGA. |
| **DSP48** | Dedicated multiply-accumulate block on Xilinx FPGAs. |
| **BRAM** | Block RAM. On-chip memory for weight storage. |
| **Yosys** | Open-source FPGA synthesis tool. SC-NeuroCore uses it for ice40/ECP5. |
| **Vivado** | Xilinx FPGA synthesis and implementation tool. For Artix-7/Zynq. |
| **AER** | Address-Event Representation. Spike encoding: (neuron_id, timestamp) pairs. |
| **Event-driven** | Neurons compute only on input events. Power ∝ spike rate, not clock rate. |
| **Clock-driven** | Every neuron updates every clock cycle. Power ∝ N × f_clk. |
| **SymbiYosys** | Formal verification frontend for Yosys. Proves assertions via SMT solvers. |
| **SystemVerilog** | Hardware description language. SC-NeuroCore's equation compiler emits this. |

## SCPN and Consciousness Modelling

| Term | Definition |
|------|-----------|
| **SCPN** | Stochastic Computational Phase Network. 16-layer hierarchical oscillator model. |
| **K_nm** | Inter-layer coupling matrix. Symmetric, zero-diagonal, calibrated to PhysioNet PLV (r=0.951). |
| **Ω_N** | Natural frequencies for 16 SCPN layers. L2 ≈ 40 Hz (gamma), L5 ≈ 1 Hz (intentional). |
| **Kuramoto model** | Coupled oscillator model: dθ/dt = ω + Σ K sin(Δθ). Phase synchronisation. |
| **Order parameter R** | Kuramoto coherence: R = |⟨exp(iθ)⟩|. R=0 incoherent, R=1 fully synchronised. |
| **BKT transition** | Berezinskii-Kosterlitz-Thouless. Topological phase transition via vortex unbinding. |
| **FIM** | Fisher Information Metric. Self-observation feedback: ΔW -= λ·(activity-μ)/N. |
| **Φ** | Integrated information (Tononi IIT). Measures how much a system is "more than its parts." |
| **Sheaf defect** | Obstruction to global phase coherence. Zero when synchronised. |
| **Winding number** | Topological invariant counting phase wraps around S¹. |
| **Ricci curvature** | Ollivier-Ricci curvature on graphs. Positive = community, negative = bottleneck. |
| **Lazarus protocol** | Checkpoint save/load/merge for identity substrate. Preserves weights + voltages + traces. |

## NIR and Interoperability

| Term | Definition |
|------|-----------|
| **NIR** | Neuromorphic Intermediate Representation. Framework-agnostic SNN graph format. |
| **Norse** | PyTorch-based SNN library (Denmark). LIF/LI neurons with autograd. |
| **snnTorch** | PyTorch SNN library (Eshraghian, UCSC). Leaky/Synaptic neurons. |
| **SpikingJelly** | Chinese SNN framework. Event-driven training. |
| **Brian2** | Python SNN simulator (Goodman, Brette). ODE-based, C++ codegen. |
| **Lava** | Intel's neuromorphic framework for Loihi chips. |
| **SpiNNaker** | Manchester ARM-based neuromorphic platform. |
