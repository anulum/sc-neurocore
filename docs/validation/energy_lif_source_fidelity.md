# EnergyLIF source-fidelity boundary

The source identity is pinned to Fardet and Levina's 2020 eLIF equations and author Brian implementation (`elif-madexp`): coupled RK4, 0.1 ms sampling, strict dual threshold, voltage reset, and energy-cost subtraction. The primary evidence is the DOI-bound 512-step receipt plus five-runtime trajectory parity.

`SCNormalizedEnergyLIFNeuron` is the separately named project recurrence that previously occupied the source name. Its exact-flow equations, level threshold, normalized energy ceiling, and three-event frozen receipt remain intact. It is an SC modification and does not increment the 155-model source catalogue.

The RTL artefacts specialize the two default profiles to signed Q32.32. Co-simulation covers the enrolled traces, Yosys proves synthesizability, the source job proves bounded reset only, and the SC job proves bounded reset/envelope assertions. Neither RTL lane establishes universal real-number equivalence or higher silicon evidence.
