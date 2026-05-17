---
title: "SC-NeuroCore: Stochastic Computing Simulation and Hardware-Oriented Compilation for Spiking Neural Networks"
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
date: 14 May 2026
bibliography: paper.bib
---

# Summary

SC-NeuroCore is an open-source Python and Rust framework for research on
stochastic-computing implementations of spiking neural networks. Stochastic
computing represents values as bitstreams and replaces arithmetic operations
with simple digital logic, which makes it attractive for low-area
neuromorphic hardware. SC-NeuroCore provides deterministic bitstream
simulation, fixed-point export utilities, hardware-oriented intermediate
representations, and Verilog generation paths that help researchers compare
software SNN behaviour with hardware-realistic execution.

![SC-NeuroCore train-to-hardware pipeline. Float-domain surrogate gradient training produces SNN weights, which are quantised to fixed-point values, simulated as stochastic bitstreams, lowered to RTL-oriented representations, and checked against hardware-facing artefacts.\label{fig:pipeline}](figures/pipeline.png){ width=100% }

# Statement of Need

Neuromorphic researchers typically work with simulators such as NEST
[@gewaltig2007], Brian2 [@stimberg2019], Lava [@lava2021], snnTorch
[@eshraghian2023], and Norse [@pehle2021norse]. These tools are effective for
event-driven simulation, differentiable SNN training, and neuromorphic
software workflows, but they do not provide an integrated path for reasoning
about stochastic bitstream arithmetic and hardware-oriented SNN compilation.
Hardware designers working with stochastic computing often have to maintain
separate Python models, HDL implementations, and verification scripts, which
makes it easy for stochastic encodings, fixed-point assumptions, and RTL
behaviour to diverge.

SC-NeuroCore addresses this gap for researchers who need a reproducible bridge
between SNN training, stochastic arithmetic, and hardware-facing artefacts. It
targets two audiences: computational neuroscience and machine-learning
researchers who need hardware-realistic SNN experiments, and digital hardware
researchers who need bitstream-level test or synthesis artefacts for FPGA and
ASIC exploration. The project is not yet being submitted to JOSS; this paper is
being maintained as a pre-submission draft until the remaining production
hardening, validation, and documentation items are complete.

# State of the Field

Stochastic computing has a long history as a compact arithmetic model
[@alaghi2013] and has been surveyed for neural-network hardware
[@smithson2019]. Spiking-neuron simulation has a separate ecosystem centred on
differential-equation solvers, event queues, and differentiable training. The
research gap is at the boundary: a trained SNN can be simulated or exported in
many formats, but few open tools expose the stochastic bitstream length,
encoding, random-stream correlation, fixed-point precision, and HDL metadata
that determine whether a design remains valid after hardware lowering.

SC-NeuroCore is intentionally positioned as a hardware-facing complement to
the existing SNN software ecosystem rather than a replacement for it. It
includes Python reference paths, a Rust acceleration crate, NIR import/export
work, RTL-oriented generators, and reproducible benchmark scripts. Current
public claims are limited to committed artefacts and CI-verifiable behaviour;
physical FPGA measurements, dynamic power, and unresolved model-fidelity audit
items remain open validation work.

![Spike raster from a small LIF network driven by sinusoidal input, simulated with stochastic bitstream encoding.\label{fig:raster}](figures/spike_raster.png){ width=90% }

# Software Design

The design separates four concerns that are often coupled in hardware SNN
experiments. First, Python reference modules express stochastic encoders,
spiking layers, neuron models, and export utilities in a form that can be
tested without vendor tools. Second, a Rust crate provides accelerated
bitstream and model kernels where the behaviour has parity tests against the
Python implementation. Third, compiler and NIR-bridge modules preserve graph,
precision, and hardware metadata so that model structure can be validated
before HDL generation. Fourth, HDL and co-simulation collateral provide a
hardware-facing evidence path for selected designs.

This separation trades maximal speed for auditability. Researchers can inspect
the same stochastic assumptions at Python, Rust, metadata, and HDL boundaries,
and unsupported paths are expected to fail closed rather than silently emit
unchecked artefacts. The project uses `pyproject.toml` packaging, AGPL-3.0-or-
later licensing with a commercial licence option, a `CITATION.cff` file, and a
Zenodo software DOI [@scneurocore_zenodo]. The current package version is
3.14.0, and the current Python coverage gate is 96%; 100% remains a project
target, not a present release claim.

# Research Impact Statement

The near-term research value of SC-NeuroCore is its reproducible experimental
surface for stochastic-computing SNN hardware studies. It provides a single
repository where bitstream encoders, SNN training/export utilities, NIR
adapters, Rust kernels, HDL generators, formal checks, benchmark scripts, and
documentation can be inspected together. This makes it suitable for
experiments that compare accuracy, bitstream length, fixed-point precision,
area estimates, and hardware metadata under controlled assumptions.

The project has active collaboration work around SHD deployment evidence and
stochastic SNN compression, but public claims will remain bounded until the
current open validation queue is complete. In particular, physical PYNQ-Z2
measurements, dynamic power, final deployable SHD artefacts, and unresolved
model-fidelity audit items are treated as pre-submission blockers rather than
completed JOSS evidence.

# AI Usage Disclosure

Generative AI tools have been used during software development, documentation
drafting, code review, CI triage, and paper editing. AI-generated changes are
not accepted as authoritative by themselves: the project requires human review,
local or CI verification, source inspection, and evidence-backed documentation
before claims are promoted to public release text.

# Acknowledgements

The stochastic-computing primitives build on Alaghi and Hayes
[-@alaghi2013] and Smithson et al. [-@smithson2019]. Neuron and SNN modelling
context draws on McCulloch and Pitts [-@mcculloch1943], Hodgkin and Huxley
[-@hodgkin1952], Izhikevich [-@izhikevich2003], Gerstner et al.
[-@gerstner2014], Bellec et al. [-@bellec2020], Fang et al. [-@fang2021], and
NeuroBench methodology [@yik2023neurobench]. This work is self-funded.

# References
