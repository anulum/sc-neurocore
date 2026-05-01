<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore -->

# Adaptive Reliability API Reference

This document provides the complete API reference for adaptive reliability
compiler intelligence features and frontier platform classes.

## Compiler Intelligence Features (§68–§76)

### §68. Approximate Computing Configurator
**`configure_approximation(equations, max_error_pct=5.0)`**
Trades precision for energy by identifying variables that can tolerate reduced bit-widths or aggressive truncation.

### §69. Energy Harvesting Modeler
**`model_energy_harvest(power_mw, harvester_type="solar", environment="indoor")`**
Analyses the feasibility of running the compiled design on batteryless edge devices using ambient energy.

### §70. Aging-Aware Timing Predictor
**`predict_aging(fmax_mhz, years=10, temperature_c=85.0)`**
Predicts circuit degradation over time due to NBTI and HCI, generating derated timing bounds.

### §71. DVFS Controller Generator
**`generate_dvfs_controller(module_name)`**
Generates a dynamic voltage and frequency scaling (DVFS) state machine that scales performance based on network spike rate.

### §72. Pareto Frontier Explorer
**`explore_pareto(equations)`**
Sweeps compilation parameters to find the non-dominated set of configurations balancing power, area, and latency.

### §73. Post-Quantum IP Protector
**`protect_ip_pqc(module_name, equations)`**
Applies NIST-standardized post-quantum cryptography (CRYSTALS-Dilithium) to sign the generated IP.

### §74. Statistical Fault Injector
**`run_fault_campaign(equations)`**
Runs a statistical fault injection campaign to measure the Soft Error Rate (SER) and Silent Data Corruption (SDC) rate.

### §75. Static Timing Closer
**`verify_timing_closure(equations, target_freq_mhz=250.0)`**
Performs formal static timing analysis (STA) on the generated RTL to verify critical path delays.

### §76. Telemetry Ingestor
**`ingest_telemetry(hw_data, twin_data, drift_threshold=0.05)`**
Closes the loop between physical hardware and the compiler's digital twin by ingesting physical telemetry.

## Frontier Hardware Platforms

The platform registry includes 4 speculative compute paradigms (8 profiles):

1.  **Thermodynamic (`thermodynamic`)**: Harnesses ambient thermal noise (kBT) for computing via thermal equilibration (`extropic_epu`, `normal_cn101`).
2.  **Probabilistic (`probabilistic`)**: Uses unstable nanomagnets (stochastic MTJ) to create fluctuating p-bits (`purdue_pbit`, `tohoku_sot_pbit`).
3.  **Polariton (`polariton`)**: Uses exciton-polariton Bose-Einstein condensates for ultrafast computation (`marvell_polariton`, `stanford_polariton`).
4.  **Metamaterial (`metamaterial`)**: Uses acoustic or optical metamaterials for passive spatial computing (`mit_metamaterial`, `penn_acoustic_meta`).

See the [Compiler Intelligence Guide](compiler_intelligence.md) and [Frontier Platforms Guide](frontier_platforms.md) for detailed usage examples and physics properties.
