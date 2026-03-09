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
date: 7 March 2026
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
(AVX2, NEON) are supported with proportionally reduced throughput. The
framework spans the full design flow from algorithm exploration through
hardware-software co-simulation to bitstream generation.

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
matches the hardware. SC-NeuroCore closes this gap by offering:

1. **Bit-true simulation**: A Python model whose LFSR seeds, fixed-point
   arithmetic (Q8.8), and overflow semantics match the Verilog RTL
   bit-for-bit. Co-simulation scripts verify equivalence automatically.

2. **Performance**: A Rust engine (`sc_neurocore_engine`) accelerates
   packed-bitstream AND, popcount, and LFSR operations via
   AVX-512/AVX2/NEON SIMD, with throughput figures reported in the
   Summary above. A full 1024-bit Bernoulli encode cycle completes in
   398 ns on the same i7-10700K test platform.

3. **Hardware target**: An IR compiler lowers network descriptions to
   SystemVerilog with AXI-Lite configuration, targeting Xilinx and Intel
   FPGAs. A Yosys synthesis script (`tools/yosys_synth.py`) is provided
   for automated LUT/FF reporting on Xilinx 7-series; formal verification
   covers the bitstream encoder (2 properties), LIF neuron (5 properties),
   and synapse (4 properties).

4. **Modular architecture**: A tiered module system separates
   production-ready core primitives (neurons, synapses, layers, HDL
   generation) from research extensions (robotics, quantum bridges,
   hyper-dimensional computing), letting users install only what they need.

Existing neuromorphic simulators---NEST [@gewaltig2007], Brian2
[@stimberg2019], and Lava [@lava2021]---target event-driven spiking
network simulation at the differential-equation level. Python SNN
libraries such as snnTorch [@eshraghian2023] and Norse [@pehle2021norse]
provide gradient-based training of spiking networks on GPU but operate
on continuous-valued membrane potentials, not hardware bit-streams. None
of these tools model stochastic bitstream-level computation.
SC-NeuroCore operates at a different abstraction: individual AND/OR
gates on bit-streams, enabling direct correspondence to synthesised
hardware. The Izhikevich neuron model [@izhikevich2003] is supported
alongside LIF via a half-step integration scheme for numerical stability
on the quadratic voltage term. The framework includes a Brunel balanced
network benchmark [@brunel2000] for cross-simulator wall-clock
comparison and follows the NeuroBench [@yik2023neurobench] methodology
for standardised reporting. The LFSR design draws on maximal-length
sequence theory [@golomb1967shift].

# Architecture

SC-NeuroCore is structured as three layers:

**Python API** (`pip install sc-neurocore`): Provides `BitstreamEncoder`,
`BitstreamSynapse`, `BitstreamDotProduct`, `StochasticLIFNeuron`,
`SCDenseLayer`, `VectorizedSCLayer`, and 22 other public symbols. All
primitives use a 16-bit maximal-length LFSR (polynomial
$x^{16}+x^{14}+x^{13}+x^{11}+1$, period 65 535) with decorrelated seed
assignment. Fixed-point arithmetic uses Q8.8 signed two's complement with
explicit bit-width masking.

**Rust Engine** (`sc_neurocore_engine`): A PyO3-bound Rust crate providing
SIMD-accelerated `vec_and`, `vec_popcount`, LFSR stepping, and HDC
(hyper-dimensional computing) vector operations. Cross-compiled wheels are
published for Linux, macOS, and Windows across Python 3.10--3.12.

**Verilog RTL** (`hdl/`): Eight synthesisable modules including
`sc_lif_neuron.v` (Q8.8 leaky integrate-and-fire with configurable
threshold and refractory period), `sc_dense_layer_core.v` (full pipeline
with decorrelated LFSR seeds), and `sc_neurocore_top.v` (AXI-Lite wrapper
for register-based configuration).

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
- **Tiered packaging**: `pip install sc-neurocore` ships core + simulation
  modules (142 files, 171 KB wheel). Research and frontier modules are
  available from source.

# Quality Assurance

SC-NeuroCore maintains 978 Python and 124 Rust tests with 98% line coverage, enforced by CI
on every push. The test suite includes unit tests, integration tests,
property-based tests (Hypothesis), cross-layer coupling tests, quantum
error correction tests, and hardware co-simulation checks. Static analysis
comprises Black formatting, Ruff linting, Bandit security scanning, and
SPDX license header validation. OpenSSF Scorecard and Best Practices
badges track supply-chain security posture.

# Acknowledgements

The stochastic computing primitives build on foundational work by
Alaghi and Hayes [-@alaghi2013] and the survey by Smithson et al.
[-@smithson2019]. The neuromorphic neuron models follow the formulation in
Gerstner et al. [-@gerstner2014].

# References
