# Synapses

Stochastic-computing synapses implement weighted connections between
neurons using bitstream multiplication (AND gates).

| Class | Learning | Use case |
|-------|----------|----------|
| `BitstreamSynapse` | None (static weight) | Inference, fixed networks |
| `StochasticSTDPSynapse` | Hebbian STDP | Unsupervised learning |
| `RewardModulatedSTDPSynapse` | Three-factor R-STDP | Reinforcement learning |
| `BitstreamDotProduct` | None | Multi-input weighted sum |
| `TripletSTDP` | Pfister-Gerstner 2006 | Rate-dependent cortical plasticity |
| `BCMSynapse` | Sliding threshold | Metaplasticity, selectivity |
| `ClopathSTDP` | Voltage-based | Unifies rate + timing plasticity |
| `TripartiteSynapse` | Astrocyte-modulated | Neuron-glia-synapse coupling |
| `GapJunction` | Electrical coupling | Interneuron synchrony |

## Static Synapse

::: sc_neurocore.synapses.sc_synapse.BitstreamSynapse

## STDP Synapse

::: sc_neurocore.synapses.stochastic_stdp.StochasticSTDPSynapse

## Reward-Modulated STDP

::: sc_neurocore.synapses.r_stdp.RewardModulatedSTDPSynapse

## Dot Product

::: sc_neurocore.synapses.dot_product.BitstreamDotProduct

## Triplet STDP (Pfister-Gerstner 2006)

::: sc_neurocore.synapses.triplet_stdp.TripletSTDP

## BCM Metaplasticity (Bienenstock-Cooper-Munro 1982)

::: sc_neurocore.synapses.bcm.BCMSynapse

## Voltage-Based STDP (Clopath et al. 2010)

::: sc_neurocore.synapses.clopath_stdp.ClopathSTDP

## Tripartite Synapse (Astrocyte Coupling)

::: sc_neurocore.synapses.tripartite.TripartiteSynapse

## Gap Junction (Electrical Synapse)

::: sc_neurocore.synapses.gap_junction.GapJunction
