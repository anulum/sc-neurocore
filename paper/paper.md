---
title: "SC-NeuroCore: A Deterministic Stochastic Computing Framework for Neuromorphic Hardware Design"
tags:
  - Python
  - Rust
  - neuromorphic computing
  - stochastic computing
  - FPGA
  - spiking neural networks
  - hardware simulation
authors:
  - name: Miroslav Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
affiliations:
  - name: Anulum Research, Independent Researcher
    index: 1
date: 9 March 2026
bibliography: paper.bib
---

# Summary

SC-NeuroCore is an open-source framework for designing, simulating, and
deploying neuromorphic circuits based on stochastic computing (SC). It
provides bit-true Python simulation that matches synthesisable Verilog RTL
cycle-exactly, a high-performance Rust engine with SIMD acceleration,
and an IR compiler that emits SystemVerilog for FPGA targets. Criterion
benchmarks on an Intel i7-10700K with AVX-512 measure 41.3 Gbit/s
bitstream packing and 224 Mstep/s LIF neuron throughput; lower SIMD tiers
(AVX2, NEON) are supported with proportionally reduced throughput.

The framework includes an end-to-end MNIST-on-FPGA demo that trains a
digit classifier, quantises weights to Q8.8 fixed-point, simulates
inference with stochastic bitstreams matching the RTL encoding, and
exports Verilog weight parameters for synthesis---achieving 94.0%
accuracy under stochastic computing at a bitstream length of 1024
(vs 94.2% float baseline).

# Statement of Need

Stochastic computing encodes values as random bit-streams and performs
arithmetic with single logic gates---an AND gate multiplies two
probabilities, a multiplexer adds them [@alaghi2013]. This makes SC
circuits extremely area-efficient and inherently fault-tolerant, properties
that are attractive for edge neuromorphic inference where power and silicon
area are constrained [@smithson2019].

The target audience is twofold: (a) hardware designers prototyping
neuromorphic edge devices who need a bit-true simulation-to-synthesis
path, and (b) SNN researchers who want cycle-accurate hardware models
rather than abstract differential-equation solvers.

No existing open-source tool provides an integrated SC design flow.
Researchers must manually translate SC algorithms into HDL, write ad-hoc
testbenches, and hope the stochastic behaviour of their Python model
matches the hardware. SC-NeuroCore closes this gap:

1. **Bit-true simulation**: A Python model whose LFSR seeds, fixed-point
   arithmetic (Q8.8), and overflow semantics match the Verilog RTL
   bit-for-bit. Co-simulation scripts verify equivalence automatically.

2. **Performance**: A Rust engine (`sc_neurocore_engine`) accelerates
   packed-bitstream AND, popcount, and LFSR operations via
   AVX-512/AVX2/NEON SIMD. A full 1024-bit Bernoulli encode cycle
   completes in 398 ns; prepacked dense forward (64$\to$32) runs at
   43.8$\times$ vs Python.

3. **Hardware target**: An IR compiler lowers network descriptions to
   SystemVerilog with AXI-Lite configuration, targeting Xilinx and Intel
   FPGAs. A Yosys synthesis script (`tools/yosys_synth.py`) reports
   LUT/FF counts on Xilinx 7-series; SymbiYosys formal verification
   covers 11 properties across the encoder, neuron, and synapse modules.

4. **Modular architecture**: A tiered module system separates
   production-ready core primitives (neurons, synapses, layers, HDL
   generation) from research extensions, letting users install only
   what they need.

Existing neuromorphic simulators---NEST [@gewaltig2007], Brian2
[@stimberg2019], and Lava [@lava2021]---target event-driven spiking
network simulation at the differential-equation level. Python SNN
libraries such as snnTorch [@eshraghian2023] and Norse [@pehle2021norse]
provide gradient-based training of spiking networks on GPU but operate
on continuous-valued membrane potentials, not hardware bit-streams. None
of these tools model stochastic bitstream-level computation or emit
synthesisable RTL.

SC-NeuroCore operates at a different abstraction: individual AND/OR
gates on bit-streams, enabling direct correspondence to synthesised
hardware. A Brunel balanced-network benchmark [@brunel2000] enables
cross-simulator wall-clock comparison: on a 1000-neuron network Brian2
completes in 1.6 s (Cython codegen) vs SC-NeuroCore's 5.2 s (Numba
JIT)---Brian2's sparse C++ codegen scales better above $\sim$1K neurons,
while SC-NeuroCore targets FPGA-scale networks ($\leq$1K) where bit-exact
RTL co-simulation matters and Brian2 has no hardware path. The framework
follows NeuroBench [@yik2023neurobench] methodology for standardised
reporting.

# Architecture

SC-NeuroCore is structured as three layers:

**Python API** (`pip install sc-neurocore`): Provides `BitstreamEncoder`,
`BitstreamSynapse`, `StochasticLIFNeuron`, `SCDenseLayer`,
`VectorizedSCLayer`, and 22 other public symbols. All primitives use a
16-bit maximal-length LFSR (polynomial
$x^{16}+x^{14}+x^{13}+x^{11}+1$, period 65 535) with decorrelated seed
assignment [@golomb1967shift]. Fixed-point arithmetic uses Q8.8 signed
two's complement with explicit bit-width masking.

**Rust Engine** (`sc_neurocore_engine`): A PyO3-bound Rust crate providing
SIMD-accelerated `vec_and`, `vec_popcount`, LFSR stepping, and HDC
(hyper-dimensional computing) vector operations. Cross-compiled wheels are
published for Linux, macOS, and Windows across Python 3.10--3.13.

**Verilog RTL** (`hdl/`): Ten synthesisable modules including
`sc_lif_neuron.v` (Q8.8 leaky integrate-and-fire with configurable
threshold and refractory period), `sc_dense_matrix_layer.v` (per-neuron
weight matrix for classification tasks such as MNIST), and
`sc_neurocore_top.v` (AXI-Lite wrapper). Yosys synthesis of the full
`sc_neurocore_top` (3-input, 7-neuron) yields 7 382 LUTs and 2 442 FFs
on Xilinx 7-series; the MNIST 16$\to$10 configuration is estimated at
$\sim$56K LUTs, fitting an Artix-7 100T.

# Key Features

- **Co-simulation**: Python golden model generates stimulus vectors;
  Icarus Verilog runs the RTL; a checker script verifies bit-exact
  equivalence across all LFSR seeds and neuron states.
- **GPU acceleration**: Optional CuPy backend for packed-bitstream
  operations, with transparent CPU fallback.
- **IR compiler**: Builds a dataflow graph from a Python network
  description, performs verification passes (seed uniqueness, bit-width
  consistency), and emits SystemVerilog.
- **Property-based testing**: Hypothesis-driven tests verify stochastic
  invariants (bitstream roundtrip accuracy, LFSR determinism, neuron
  output constraints) across randomised inputs.
- **MNIST-on-FPGA demo**: End-to-end pipeline from `sklearn` training
  through Q8.8 quantisation and SC simulation to Verilog weight export,
  demonstrating 94.0% SC accuracy with sign-magnitude encoding.
- **Formal verification**: SymbiYosys proofs for LIF neuron
  (5 properties), bitstream synapse (4 properties), and encoder
  (2 properties).

# Quality Assurance

SC-NeuroCore maintains 1 100+ Python and 108 Rust tests with 98% line
coverage, enforced by CI on every push. The test suite includes unit
tests, integration tests, 18 property-based tests (Hypothesis),
cross-layer coupling tests, and hardware co-simulation checks. Static
analysis comprises Ruff linting, Bandit security scanning, and SPDX
license header validation. Eleven CI workflows---all with SHA-pinned
GitHub Actions---cover lint, test, build, benchmark, documentation,
CodeQL, and OpenSSF Scorecard.

# Acknowledgements

The stochastic computing primitives build on foundational work by
Alaghi and Hayes [-@alaghi2013] and the survey by Smithson et al.
[-@smithson2019]. The neuromorphic neuron models follow the formulation
in Gerstner et al. [-@gerstner2014]. The Izhikevich neuron model
[@izhikevich2003] is supported alongside LIF via a half-step integration
scheme.

# References
