# Synapses

Stochastic-computing synapses implement weighted connections between
neurons using bitstream multiplication (AND gates).

| Class | Learning | Use case |
|-------|----------|----------|
| `BitstreamSynapse` | None (static weight) | Inference, fixed networks |
| `StochasticSTDPSynapse` | Hebbian STDP | Unsupervised learning |
| `RewardModulatedSTDPSynapse` | Three-factor R-STDP | Reinforcement learning |
| `BitstreamDotProduct` | None | Multi-input weighted sum |

## Static Synapse

::: sc_neurocore.synapses.sc_synapse.BitstreamSynapse

## STDP Synapse

::: sc_neurocore.synapses.stochastic_stdp.StochasticSTDPSynapse

## Reward-Modulated STDP

::: sc_neurocore.synapses.r_stdp.RewardModulatedSTDPSynapse

## Dot Product

::: sc_neurocore.synapses.dot_product.BitstreamDotProduct
