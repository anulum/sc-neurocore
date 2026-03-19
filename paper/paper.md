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
date: 17 March 2026
bibliography: paper.bib
---

# Summary

SC-NeuroCore is an open-source framework for designing, simulating, and
deploying neuromorphic circuits based on stochastic computing (SC). It
provides bit-true Python simulation that matches synthesisable Verilog RTL
cycle-exactly, a high-performance Rust SIMD engine with PyO3 bindings, and
an IR compiler that emits SystemVerilog for FPGA targets. The framework
bridges GPU-based SNN training to hardware deployment: networks trained
with surrogate gradients in PyTorch are quantised to Q8.8 fixed-point and
exported to stochastic bitstream weights for FPGA synthesis (\autoref{fig:pipeline}).

![SC-NeuroCore train-to-hardware pipeline. Float-domain surrogate gradient training produces SNN weights, which are quantised to Q8.8 fixed-point, simulated as SC bitstreams with bit-exact RTL correspondence, compiled to SystemVerilog via the IR compiler, and synthesised for FPGA targets. The Rust SIMD engine accelerates all simulation stages. A bidirectional co-simulation link verifies Python--Verilog equivalence at every timestep.\label{fig:pipeline}](figures/pipeline.png){ width=100% }

# Statement of Need

Stochastic computing encodes values as random bit-streams and performs
arithmetic with single logic gates---an AND gate multiplies two
probabilities, a multiplexer adds them [@alaghi2013]. SC circuits are
area-efficient and fault-tolerant, attractive for edge neuromorphic
inference where power and silicon area are constrained [@smithson2019].

No existing open-source tool provides an integrated SC design flow.
Researchers must manually translate SC algorithms into HDL, write ad-hoc
testbenches, and hope the stochastic behaviour of their Python model
matches the hardware. SC-NeuroCore closes this gap with bit-true
simulation, SIMD-accelerated Rust kernels, an IR compiler targeting
Xilinx and Intel FPGAs, and a surrogate gradient training module
that exports directly to SC bitstream weights---a train-to-hardware path
that snnTorch [@eshraghian2023], Norse [@pehle2021norse], Brian2
[@stimberg2019], NEST [@gewaltig2007], and Lava [@lava2021] do not provide.

The target audience is (a) hardware designers prototyping neuromorphic
edge devices who need a bit-true simulation-to-synthesis path, and
(b) SNN researchers who want cycle-accurate hardware models rather than
abstract differential-equation solvers.

# State of the Field

Neuromorphic simulators---NEST, Brian2, and Lava---target event-driven
spiking network simulation at the differential-equation level. SNN
training libraries snnTorch and Norse provide gradient-based training on
GPU but operate on continuous-valued membrane potentials, not hardware
bit-streams. None model stochastic bitstream-level computation or emit
synthesisable RTL.

SC-NeuroCore operates at a different abstraction: individual AND/OR
gates on bit-streams with direct correspondence to synthesised hardware.
A Brunel balanced-network benchmark [@brunel2000] shows SC-NeuroCore's
Numba JIT backend completes a 1 000-neuron simulation in 0.35 s versus
Brian2's 1.38 s (4.0$\times$ speedup), with firing rates matching
within 1% (\autoref{fig:raster}). At 10 000 neurons Brian2 is
1.35$\times$ faster (5.9 s vs 4.4 s), as its compiled C++ codegen
scales better for large sparse networks. SC-NeuroCore targets
FPGA-scale networks ($\leq$5K neurons) where bit-exact RTL
co-simulation matters.

![Spike raster from a 5-neuron LIF network driven by sinusoidal input, simulated with SC-NeuroCore's stochastic bitstream encoding. Each neuron uses a decorrelated 16-bit LFSR seed.\label{fig:raster}](figures/spike_raster.png){ width=90% }

For surrogate gradient training, SC-NeuroCore's `training` module
matches snnTorch on a standard FC-SNN benchmark (95.5% vs 95.8% MNIST,
identical 784$\to$128$\to$128$\to$10 architecture, 10 epochs). With
learnable membrane time constants [@fang2021] the FC-SNN reaches 97.7%;
a convolutional SNN architecture reaches 99.49%. The `to_sc_weights()`
method exports trained float weights normalised to [0, 1] for SC
bitstream deployment.

# Software Design

SC-NeuroCore is structured in five layers, each independently usable:

**Python API** (`pip install sc-neurocore`): 38 public symbols including
`BitstreamEncoder`, `StochasticLIFNeuron`, `SCDenseLayer`, and
`VectorizedSCLayer`. All SC primitives use a 16-bit maximal-length LFSR
(polynomial $x^{16}+x^{14}+x^{13}+x^{11}+1$, period 65 535) with
decorrelated seed assignment [@golomb1967shift]. Fixed-point arithmetic
uses Q8.8 signed two's complement. An optional `training` subpackage
provides LIF, adaptive LIF [@bellec2020], and recurrent LIF cells with
surrogate gradient backward passes and learnable membrane parameters.
A library of 122 neuron models---from McCulloch-Pitts [@mcculloch1943]
through Hodgkin-Huxley [@hodgkin1952], Izhikevich [@izhikevich2003],
and 9 hardware chip emulators (Loihi, TrueNorth, BrainScaleS, SpiNNaker,
Akida)---covers 82 years of computational neuroscience.

**Rust Engine** (`sc_neurocore_engine`): A PyO3-bound Rust crate
providing SIMD-accelerated bitstream operations, 111 neuron model
implementations, and a `NetworkRunner` with CSR-sparse projections and
Rayon-parallel population stepping scaling to 100K+ neurons. Runtime
feature detection selects AVX-512, AVX2, or NEON paths. A Criterion
benchmark measures 41.3 Gbit/s bitstream packing on AVX-512.
Cross-compiled wheels target Linux, macOS, and Windows across
Python 3.10--3.14.

**Network Simulation** (`sc_neurocore.network`): A
Population-Projection-Network engine with three backends (Python/NumPy,
Rust NetworkRunner, MPI via mpi4py), six topology generators, a model
zoo with 10 pre-built configurations, 3 pre-trained weight sets, and
126 spike train analysis functions covering the combined scope of
Elephant [@elephant2023] and PySpike.

**Verilog RTL** (`hdl/`): 17 synthesisable modules including
`sc_lif_neuron.v` (Q8.8 LIF), `sc_dense_matrix_layer.v`, and
`sc_neurocore_top.v` (AXI-Lite wrapper). Yosys synthesis of
`sc_neurocore_top` yields 7 382 LUTs on Xilinx 7-series. SymbiYosys
formal verification covers 64 properties across 7 modules.

**IR Compiler**: Parses a graph-based intermediate representation,
verifies structural invariants, and emits synthesisable SystemVerilog
targeting Xilinx and Intel FPGAs.

A minimal end-to-end example:

```python
from sc_neurocore import BitstreamEncoder, StochasticLIFNeuron
enc = BitstreamEncoder(data_width=16, fraction=8)
neuron = StochasticLIFNeuron()
for t in range(100):
    spike, v = neuron.step(leak_k=1, gain_k=256, i_t=50, noise_in=0)
```

The key design trade-off is determinism over speed: SC-NeuroCore
maintains bit-exact correspondence between Python simulation and Verilog
RTL at every timestep, enabling co-simulation workflows where a checker
script verifies bit-exact equivalence across all LFSR seeds and neuron
states.

# Availability

SC-NeuroCore is available on [PyPI](https://pypi.org/project/sc-neurocore/)
(`pip install sc-neurocore`) and
[GitHub](https://github.com/anulum/sc-neurocore) under
AGPL-3.0-or-later with a commercial license option.
[Documentation](https://anulum.github.io/sc-neurocore/) is hosted on
GitHub Pages. The repository includes a
[contributing guide](https://github.com/anulum/sc-neurocore/blob/main/CONTRIBUTING.md),
24 tutorials, 14 worked examples, and 5 Jupyter notebooks including an
interactive neuron model explorer.
A Zenodo-archived DOI is available [@scneurocore_zenodo].

# Quality Assurance

SC-NeuroCore maintains 1 785 Python and 336 Rust tests with 100% line
coverage enforced by CI on every push. The test suite includes unit
tests, integration tests, property-based tests (Hypothesis),
cross-layer coupling tests, and hardware co-simulation checks.
Static analysis comprises Ruff linting, Bandit security scanning,
SPDX license header validation, and CodeQL. Thirteen CI workflows---all
with SHA-pinned GitHub Actions---guard every merge. OpenSSF Scorecard
monitors supply-chain security.

# AI Usage Disclosure

Generative AI (Claude, Anthropic; model versions claude-sonnet-4-20250514,
claude-opus-4-20250514, and claude-opus-4-6) was used during development
for code generation, refactoring, test writing, and benchmark scripting.
All generated code was reviewed, tested, and validated by the human
author. Core architectural decisions---LFSR seed assignment, Q8.8
encoding, IR compiler design, SIMD dispatch strategy, and surrogate
gradient function selection---were made by the human author.

# Acknowledgements

The SC primitives build on Alaghi and Hayes [-@alaghi2013] and Smithson
et al. [-@smithson2019]. Neuron models follow Gerstner et al.
[-@gerstner2014], Izhikevich [-@izhikevich2003], Bellec et al.
[-@bellec2020], and Fang et al. [-@fang2021]. Benchmarks follow
NeuroBench methodology [@yik2023neurobench]. This work was self-funded.

# References
