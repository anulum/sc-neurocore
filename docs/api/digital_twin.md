# Digital Twin

Simulate FPGA imperfections during training for deployment confidence.

- `FPGADigitalTwin` — Models process variation, quantization noise, thermal drift, and routing delays. Train against the twin to produce hardware-robust SNNs. Compare ideal vs twin output to estimate deployment accuracy loss.

```python
from sc_neurocore.digital_twin import FPGADigitalTwin
```

See [Tutorial 48: Digital Twin](../tutorials/48_digital_twin.md).

::: sc_neurocore.digital_twin
    options:
      show_root_heading: true
