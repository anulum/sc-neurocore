# Product Overview

SC-NeuroCore is a stochastic-computing and neuromorphic hardware co-design toolkit. It helps teams move from numerical neuron and network experiments to auditable implementation artefacts: stochastic bitstreams, fixed-point representations, optional accelerated simulation, generated RTL, synthesis reports, and readiness evidence.

## Plain-language summary

Most neural-network software optimises floating-point training and inference. SC-NeuroCore focuses on a different question: when can neural computation be represented as probability, spike timing, or stochastic bitstreams, and when can that representation be moved toward energy-aware hardware?

The project gives researchers and engineers one place to explore:

- stochastic encoders and bitstream arithmetic;
- spiking neuron and synapse models;
- network simulation and training paths;
- hardware export and co-simulation workflows;
- benchmark and parity evidence;
- evidence categories for industrial readiness.

## Core workflow

1. Model a neuron, synapse, network, or stochastic operator in Python.
2. Validate numerical behaviour with module-specific tests, parity artefacts, or benchmark harnesses.
3. Select an execution path: pure Python, optional Rust engine, optional training backend, or hardware-generation route.
4. Export evidence: JSON benchmark results, generated RTL, synthesis reports, notebook evidence, or readiness assessments.
5. Promote only claims that are backed by committed artefacts.

## Why stochastic computing matters

Stochastic computing represents values as probabilities over bitstreams. That can trade exact arithmetic for compact logic, fault tolerance, and hardware-friendly approximate computation. In neuromorphic systems, this aligns naturally with spikes, rates, stochastic synapses, and noisy physical substrates.

SC-NeuroCore treats stochasticity as a measurable engineering parameter rather than a demo effect. The documentation and benchmark policy require error bounds, parity checks, or committed artefacts before a result becomes a public claim.

## What the repository contains

| Layer | Purpose |
| --- | --- |
| Public Python package | Stable entry points for simulation, encoding, layers, selected hardware helpers, and documented APIs. |
| Optional Rust engine | Acceleration for selected hot paths when the local environment provides the engine. |
| Training and interop extras | PyTorch, NIR, JAX, CuPy, quantum, and Studio paths installed only when requested. |
| HDL and hardware evidence | Verilog/SystemVerilog generation, primitive RTL, synthesis reports, and deployment guides. |
| Research notebooks | Executable walkthroughs and evidence notebooks for selected workflows. |
| Polyglot research mirrors | Julia, Go, Mojo, and Rust counterparts for selected kernels and comparison benchmarks. |

## Evidence boundary

SC-NeuroCore is a broad research and engineering stack, not a blanket certification claim. The public documentation distinguishes:

- stable package surfaces from source-only research modules;
- measured benchmark artefacts from planned comparison gaps;
- synthesis/resource evidence from power or energy evidence;
- readiness assessment from regulated deployment approval;
- optional dependency paths from default installation guarantees.

## Where to start

- New users: [Getting Started](guides/getting-started.md).
- Structured learning: [Learning Path](LEARNING_PATH.md).
- Notebook users: [Notebook Guide](guides/notebook_guide.md).
- API consumers: [API Reference Index](api/API_REFERENCE.md).
- Benchmark readers: [Benchmarks](benchmarks/BENCHMARKS.md) and [Cross-Framework Benchmark Evidence](benchmarks/cross_framework.md).
- Commercial or industrial evaluators: [Applications and Market](applications_and_market.md) and [Industrial Applications](api/industrial_applications.md).
