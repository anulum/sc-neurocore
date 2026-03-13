# Neurons

Spiking neuron models — the fundamental compute units of SC-NeuroCore.

All neurons implement the `BaseNeuron` interface (`step`, `reset_state`,
`get_state`). Choose by fidelity vs speed:

| Class | Domain | Speed |
|-------|--------|-------|
| `StochasticLIFNeuron` | Software simulation | Fast |
| `FixedPointLIFNeuron` | Bit-true hardware model | Medium |
| `SCIzhikevichNeuron` | Rich dynamics (bursting, chattering) | Medium |
| `HomeostaticLIFNeuron` | Self-regulating firing rate | Fast |
| `StochasticDendriticNeuron` | Multi-compartment dendritic processing | Slow |

## Base

::: sc_neurocore.neurons.base.BaseNeuron

## Leaky Integrate-and-Fire

::: sc_neurocore.neurons.stochastic_lif.StochasticLIFNeuron

## Fixed-Point LIF (Hardware Model)

::: sc_neurocore.neurons.fixed_point_lif.FixedPointLIFNeuron

::: sc_neurocore.neurons.fixed_point_lif.FixedPointLFSR

::: sc_neurocore.neurons.fixed_point_lif.FixedPointBitstreamEncoder

## Izhikevich

::: sc_neurocore.neurons.sc_izhikevich.SCIzhikevichNeuron

## Homeostatic LIF

::: sc_neurocore.neurons.homeostatic_lif.HomeostaticLIFNeuron

## Dendritic

::: sc_neurocore.neurons.dendritic.StochasticDendriticNeuron
