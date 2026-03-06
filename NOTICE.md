# NOTICE — Licensing & Commercial Use

© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li

## Dual Licensing

SC-NeuroCore is available under two license options:

### Open Source — AGPL-3.0-or-later

The full source code is licensed under the
[GNU Affero General Public License v3.0](LICENSE). Under AGPL-3.0:

- You may use, modify, and distribute SC-NeuroCore freely.
- If you run a modified version as a network service, you must make
  your source code available to users of that service.
- All derivative works must also be licensed under AGPL-3.0.

### Commercial License

A proprietary license is available for organisations that:

- Cannot comply with AGPL-3.0 copyleft requirements
- Need to embed SC-NeuroCore in closed-source products
- Require SLA-backed support, indemnification, or custom builds

Contact [protoscience@anulum.li](mailto:protoscience@anulum.li) for terms.

## AGPL-3.0 Boundary Clarification

| Use Case | License Required |
|----------|-----------------|
| Academic research with published results | AGPL-3.0 (free) |
| Internal simulation not exposed as a service | AGPL-3.0 (free) |
| SaaS / cloud service using SC-NeuroCore | AGPL-3.0 (must release source) **or** Commercial |
| Embedded in closed-source product | Commercial |
| FPGA bitstreams generated from HDL templates | Commercial (HDL templates are AGPL-covered) |

## Third-Party Components

| Component | License | Location |
|-----------|---------|----------|
| NumPy | BSD-3-Clause | runtime dependency |
| SciPy | BSD-3-Clause | runtime dependency |
| Numba | BSD-2-Clause | runtime dependency |
| Matplotlib | PSF-based | runtime dependency |
| PyO3 | MIT/Apache-2.0 | Rust engine binding |

See `sbom.json` (attached to each GitHub Release) for the full software
bill of materials.
