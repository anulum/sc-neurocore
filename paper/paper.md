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
  - given-names: Miroslav
    surname: Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
affiliations:
  - name: Anulum Research, Independent Researcher
    index: 1
date: 10 March 2026
bibliography: paper.bib
---

# Summary

SC-NeuroCore is an open-source framework for designing, simulating, and
deploying neuromorphic circuits based on stochastic computing (SC). It
provides bit-true Python simulation that matches synthesisable Verilog RTL
cycle-exactly, a high-performance Rust engine with SIMD acceleration, and
an IR compiler that emits SystemVerilog for FPGA targets. The framework
bridges GPU-based SNN training to hardware deployment: networks trained
with surrogate gradients in PyTorch are quantised and exported to
stochastic bitstream weights for FPGA synthesis.

An end-to-end MNIST demo trains a spiking digit classifier, quantises
weights to Q8.8 fixed-point, simulates inference with stochastic
bitstreams matching the RTL encoding, and exports Verilog weight
parameters---achieving 94.0% accuracy under SC at bitstream length 1024
(vs 94.2% float baseline). A convolutional SNN variant reaches 99.2%
via surrogate gradient training with learnable neuron parameters.

# Statement of Need

Stochastic computing encodes values as random bit-streams and performs
arithmetic with single logic gates---an AND gate multiplies two
probabilities, a multiplexer adds them [@alaghi2013]. SC circuits are
area-efficient and fault-tolerant, attractive for edge neuromorphic
inference where power and silicon area are constrained [@smithson2019].

The target audience is (a) hardware designers prototyping neuromorphic
edge devices who need a bit-true simulation-to-synthesis path, and
(b) SNN researchers who want cycle-accurate hardware models rather than
abstract differential-equation solvers.

No existing open-source tool provides an integrated SC design flow.
Researchers must manually translate SC algorithms into HDL, write ad-hoc
testbenches, and hope the stochastic behaviour of their Python model
matches the hardware. SC-NeuroCore closes this gap with bit-true
simulation, SIMD-accelerated Rust kernels, an IR compiler targeting
Xilinx and Intel FPGAs, and a PyTorch surrogate gradient training module
that bridges float-domain learning to SC bitstream deployment.

# State of the Field

Neuromorphic simulators---NEST [@gewaltig2007], Brian2
[@stimberg2019], and Lava [@lava2021]---target event-driven spiking
network simulation at the differential-equation level. Python SNN
training libraries snnTorch [@eshraghian2023] and Norse
[@pehle2021norse] provide gradient-based training on GPU but operate
on continuous-valued membrane potentials, not hardware bit-streams. None
model stochastic bitstream-level computation or emit synthesisable RTL.

SC-NeuroCore operates at a different abstraction: individual AND/OR
gates on bit-streams with direct correspondence to synthesised hardware.
A Brunel balanced-network benchmark [@brunel2000] on AMD EPYC 9575F
shows SC-NeuroCore's Numba JIT backend completes a 1000-neuron
simulation in 0.35 s versus Brian2's 1.38 s (4.0$\times$ speedup),
with firing rates matching within 1% (100 Hz). At 10 000 neurons Brian2
is 1.35$\times$ faster (5.9 s vs 4.4 s), as its compiled C++ codegen
scales better for large sparse networks. SC-NeuroCore targets
FPGA-scale networks ($\leq$5K neurons) where bit-exact RTL
co-simulation matters and Brian2 has no hardware path.

For surrogate gradient training, SC-NeuroCore's `training` module
matches snnTorch on a standard FC-SNN benchmark (95.5% vs 95.8% MNIST,
identical 784$\to$128$\to$128$\to$10 architecture, 10 epochs). With
learnable membrane time constants [@fang2021] the FC-SNN reaches 97.7%;
a convolutional SNN architecture reaches 99.2%. The `to_sc_weights()`
method exports trained float weights normalised to [0, 1] for SC
bitstream deployment---a train-to-hardware path that snnTorch and Norse
do not provide.

# Software Design

SC-NeuroCore is structured as three layers, each independently usable:

**Python API** (`pip install sc-neurocore`): Provides `BitstreamEncoder`,
`BitstreamSynapse`, `StochasticLIFNeuron`, `SCDenseLayer`,
`VectorizedSCLayer`, and 22 other public symbols. All primitives use a
16-bit maximal-length LFSR (polynomial
$x^{16}+x^{14}+x^{13}+x^{11}+1$, period 65 535) with decorrelated seed
assignment [@golomb1967shift]. Fixed-point arithmetic uses Q8.8 signed
two's complement with explicit bit-width masking. An optional `training`
subpackage (requiring PyTorch) provides LIF, adaptive LIF [@bellec2020],
and recurrent LIF cells with surrogate gradient backward passes,
supporting learnable membrane and threshold parameters.

**Rust Engine** (`sc_neurocore_engine`): A PyO3-bound Rust crate providing
SIMD-accelerated `vec_and`, `vec_popcount`, LFSR stepping, and HDC
vector operations. Runtime feature detection selects AVX-512, AVX2, or
NEON paths. A Criterion benchmark measures 41.3 Gbit/s bitstream packing
on AVX-512 (Intel i7-10700K). Cross-compiled wheels target Linux, macOS,
and Windows across Python 3.10--3.13.

**Verilog RTL** (`hdl/`): Ten synthesisable modules including
`sc_lif_neuron.v` (Q8.8 LIF with configurable threshold and refractory
period), `sc_dense_matrix_layer.v` (per-neuron weight matrix), and
`sc_neurocore_top.v` (AXI-Lite wrapper). Yosys synthesis of
`sc_neurocore_top` (3-input, 7-neuron) yields 7 382 LUTs on Xilinx
7-series. The MNIST 16$\to$10 configuration is estimated at $\sim$56K
LUTs, fitting an Artix-7 100T.

The key design trade-off is determinism over speed: where Brian2 uses
compiled C++ codegen for maximal throughput, SC-NeuroCore maintains
bit-exact correspondence between Python simulation and Verilog RTL at
every timestep. This enables co-simulation workflows where the Python
golden model generates stimulus vectors, Icarus Verilog runs the RTL,
and a checker script verifies bit-exact equivalence across all LFSR
seeds and neuron states. SymbiYosys formal verification covers
11 properties across the encoder, neuron, and synapse modules.

# Research Impact Statement

SC-NeuroCore is published on PyPI with a Zenodo-archived DOI
[@scneurocore_zenodo]. The framework fills a gap at the intersection of
two active research areas---stochastic computing [@alaghi2013] and
neuromorphic hardware [@smithson2019]---that currently lacks open-source
tooling for the simulation-to-synthesis path.

Credible near-term significance rests on three capabilities no competing
tool provides: (1) bit-true SC simulation matching synthesisable RTL,
(2) an IR compiler emitting SystemVerilog for FPGA targets, and
(3) a surrogate gradient training module with a direct export path to
SC bitstream weights. The Brunel benchmark results and MNIST-on-FPGA
demo provide reproducible baselines following NeuroBench methodology
[@yik2023neurobench].

The project originated in the God of the Math Collection research
monorepo (active since December 2025) and was extracted to a standalone
repository in February 2026 for independent packaging and CI. The parent
project encompasses SCPN (Self-Consistent Phenomenological Network)
theoretical work across multiple repositories, of which SC-NeuroCore
implements the neuromorphic hardware layer.

# Quality Assurance

SC-NeuroCore maintains 1 080 Python and 108 Rust tests with 98% line
coverage, enforced by CI on every push. The test suite includes unit
tests, integration tests, 18 property-based tests (Hypothesis),
cross-layer coupling tests, and hardware co-simulation checks. Static
analysis comprises Ruff linting, Bandit security scanning, and SPDX
license header validation. Eleven CI workflows---all with SHA-pinned
GitHub Actions---cover lint, test, build, benchmark, documentation,
CodeQL, and OpenSSF Scorecard.

# AI Usage Disclosure

Generative AI (Claude, Anthropic; model versions claude-sonnet-4-20250514 and
claude-opus-4-20250514) was used during development for code generation,
refactoring, test writing, benchmark scripting, and drafting portions of
this paper. All generated code was reviewed, tested, and validated by
the human author. Core architectural decisions---the LFSR seed
assignment scheme, Q8.8 fixed-point encoding, IR compiler design, SIMD
dispatch strategy, and surrogate gradient function selection---were made
by the human author. The AI did not make independent design choices.

# Acknowledgements

The stochastic computing primitives build on foundational work by
Alaghi and Hayes [-@alaghi2013] and the survey by Smithson et al.
[-@smithson2019]. The neuromorphic neuron models follow the formulation
in Gerstner et al. [-@gerstner2014]. The Izhikevich neuron model
[@izhikevich2003] is supported alongside LIF. The adaptive LIF cell
follows Bellec et al. [-@bellec2020], and the learnable membrane time
constant approach follows Fang et al. [-@fang2021]. This work was
self-funded with no external financial support.

# References
