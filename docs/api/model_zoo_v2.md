# Model Zoo

Plugin-based neuron model zoo with auto-Verilog generation and
auto-documentation. Ships with LIF, Izhikevich, AdEx, and Hodgkin-Huxley.

## Quick Start

```python
from sc_neurocore.model_zoo.model_zoo import (
    PluginRegistry, VerilogGenerator, DocGenerator,
    LIFPlugin, IzhikevichPlugin, AdExPlugin,
)
```

::: sc_neurocore.model_zoo.model_zoo
