<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Hardware abstraction API -->

# Hardware Abstraction

The `sc_neurocore.hardware` package provides device specifications, resource
estimation, constraint checks, neuron-to-core mapping and deployment packaging.
Its public facade exports `DeviceSpec`, `DeviceFamily`, `get_device`,
`ResourceEstimator`, `ConstraintChecker`, `Mapper` and `Deployer`, together with
their associated result types.

```python
from sc_neurocore.hardware import DeviceFamily, get_device

spec = get_device(DeviceFamily.FPGA_GENERIC)
print(spec.family.name)  # FPGA_GENERIC
```

These are software abstractions and metadata. A target description or deployment
package does not establish communication with a physical device, calibrated
measurements or qualification of that target. Model-specific emulation and
compiler target profiles have separate APIs; see
[target profiles](../guides/hardware_profiles.md) and the
[Studio hardware evidence guide](../studio/manual/hardware.md).

::: sc_neurocore.hardware
    options:
      show_root_heading: true
