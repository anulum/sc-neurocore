# SC-NeuroCore User Guide

## Introduction
SC-NeuroCore is a library for simulating Stochastic Computing (SC) based neural networks. It provides software models that align with hardware implementations for FPGA deployment.

## Installation
Ensure the package is in your python path.
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/sc-neurocore/src
```

## Quick Start

### 1. Creating a Neuron
```python
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

# Create a neuron with a 20ms time constant
neuron = StochasticLIFNeuron(tau_mem=20.0, v_threshold=1.0)

# Run for 100 steps with constant input
for _ in range(100):
    spike = neuron.step(input_current=0.1)
    if spike:
        print("Spike!")
```

### 2. Building a Stochastic Network
Connect a source to a neuron via stochastic synapses.

```python
from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

# Define inputs and weights
inputs = [0.8, 0.5, 0.2]
weights = [1.0, 0.5, 0.0]

# Source generates stochastic currents
source = BitstreamCurrentSource(
    x_inputs=inputs, 
    weight_values=weights,
    x_min=0.0, x_max=1.0,
    w_min=0.0, w_max=1.0
)

neuron = StochasticLIFNeuron()

# Simulation loop
for t in range(100):
    current = source.step()
    neuron.step(current)
```

## Hardware Deployment (PYNQ)
To run on PYNQ-Z1/Z2:

1.  Synthesize the bitstream using `03_CODE/sc-neurocore/scripts/build_bitstream.tcl`.
2.  Copy `sc_neurocore.bit` and `sc_neurocore.hwh` to the board.
3.  Use the provided driver (or `HardwareInterface` adapter in `hardware_tests`) to control the IP.

```python
from hardware_tests.interfaces.pynq_adapter import HardwareInterface

hw = HardwareInterface("sc_neurocore.bit")
hw.reset_core()
hw.load_weights([0.5, 0.8, 0.1])
hw.start_core()
```
