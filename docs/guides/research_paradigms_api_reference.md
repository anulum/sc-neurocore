<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore -->

# Research Paradigms API Reference

This quick reference covers the advanced co-design APIs for speculative and
deep-future platforms including wetware, reversible logic, molecular, and
microfluidic paradigms.

## Advanced Intelligence Features (`sc_neurocore.compiler.intelligence`)

### `dispatch_omni_paradigm(equations: dict[str, str]) -> OmniDispatchMap`
Partitions a single SNN model across thermodynamic, optical, CMOS, and quantum hardware simultaneously based on equation AST structure.

### `synthesize_reversible_logic(equations: dict[str, str], bits: int = 16) -> ReversibleNetlist`
Translates differential equations into Toffoli and Fredkin logic gates for zero-energy Landauer-limit hardware.

### `map_wetware_mea(populations: int, connectivity: float) -> MEAMapping`
Compiles spatial topology into physical Multi-Electrode Array (MEA) stimulation frequencies and voltages for biological organoids.

### `synthesize_morphology(equations: dict[str, str], max_generations: int = 10) -> Morphology`
Uses evolutionary algorithms to design a completely custom physical routing topology matching the SNN.

### `enforce_cognitive_bounds(equations: dict[str, str], state_bounds: dict[str, tuple[float, float]]) -> CognitiveBounds`
Injects hardware kill-switches into the RTL to prevent runaway states in extreme-scale architectures.

### `generate_adiabatic_clocks(phases: int, freq_mhz: float) -> list[AdiabaticPhase]`
Generates multi-phase trapezoidal resonant clock timings for energy-recovery logic.

### `route_holographic_interconnects(num_neurons: int, connections: int) -> HolographicRouter`
Calculates phase array matrices for free-space optical holographic network projections.

## Hardware Profiles (`sc_neurocore.compiler.platforms`)

**Wetware / Biological**
* `cortical_labs_dishbrain`
* `finalspark_neuroplatform`

**Molecular / Chemical**
* `biomemory_dna`
* `catalog_dna_compute`

**Reversible / Adiabatic**
* `superconducting_aqfp`
* `scrl_logic`

**Microfluidic / Mechanical**
* `nanofluidic_logic`
* `mems_neuromorphic`
