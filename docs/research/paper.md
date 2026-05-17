# SC-NeuroCore JOSS Pre-Submission Draft

**Miroslav Šotek** · [ORCID 0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
· Anulum Research, Independent Researcher

**Date:** 14 May 2026

**Status:** Pre-submission draft. Formal JOSS submission is postponed until the
remaining production-hardening, validation, coverage, documentation, and
hardware-evidence TODOs are complete.

**Software DOI:** [10.5281/zenodo.18906614](https://doi.org/10.5281/zenodo.18906614)

The canonical JOSS source is `paper/paper.md` at the repository root. This page
mirrors the current submission stance for the public documentation site: the
paper is being kept in JOSS format, but it is not yet being submitted.

## Current Evidence Boundary

SC-NeuroCore is an open-source Python and Rust framework for stochastic-
computing experiments with spiking neural networks. It provides deterministic
bitstream simulation, fixed-point export utilities, hardware-oriented
intermediate representations, NIR bridge work, Rust acceleration paths, and
Verilog-generation collateral.

Public claims are limited to committed artefacts and CI-verifiable behaviour.
The current package version is **3.14.0**, and the current Python coverage gate
is **96%**. The project target remains 100% coverage, but 100% is not a current
release claim.

## Submission Blockers

The JOSS submission/review item remains open until:

- numeric claims in the paper and public docs are regenerated from current CI,
  release, benchmark, and artefact evidence;
- physical FPGA claims are backed by PYNQ-Z2 board logs for power, thermal,
  latency, and parity;
- unresolved model-fidelity and documentation audit items are closed or
  explicitly scoped;
- core installation, examples, tests, contribution paths, and support channels
  have been checked from a clean reviewer-style environment.

## JOSS Structure

The draft follows the current JOSS paper structure:

- Summary
- Statement of Need
- State of the Field
- Software Design
- Research Impact Statement
- AI Usage Disclosure
- Acknowledgements
- References

## Research Impact Position

The near-term research value is a reproducible experimental surface for
stochastic-computing SNN hardware studies: bitstream encoders, training/export
utilities, NIR adapters, Rust kernels, HDL generators, formal checks, benchmark
scripts, and documentation live in one inspectable repository. Active
collaboration work around SHD deployment evidence and stochastic SNN
compression remains pre-submission evidence until the validation queue is
complete.
