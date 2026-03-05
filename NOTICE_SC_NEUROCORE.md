© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

## Scope

- This notice should be included in SC-NeuroCore code and documentation where technically possible.
- Use file-appropriate comment syntax for code/config files.
- Keep this notice intact in generated session logs and benchmark reports.

## Compiled Bitstream Licensing

SystemVerilog/Verilog source files in `hdl/` are licensed under AGPLv3.
Compiled FPGA bitstreams generated from these sources are considered
**derived works** under AGPLv3 Section 1. Distribution of compiled
bitstreams (whether standalone or embedded in hardware products) triggers
AGPLv3 obligations including source disclosure.

For deployment of compiled bitstreams in closed-source or commercial
products without AGPLv3 obligations, a separate commercial license is
required. Contact [protoscience@anulum.li](mailto:protoscience@anulum.li).

## Python Simulation Environment

The pure-Python pathway (`src/sc_neurocore/`) serves as a **deterministic
digital twin simulation environment** — not a fallback. It provides
cycle-exact bit-true models for rapid prototyping, verification, and
co-simulation against the Rust engine and Verilog RTL.
