# Neuro-Symbolic (Predictive Coding)

Predictive coding primitives with hyperdimensional symbol binding.
Formally verifiable inference over SC bitstreams.

## Quick Start

```python
from sc_neurocore.neuro_symbolic import (
    NeuroSymbolicPredictiveAgent,
    PredictiveAgentConfig,
)

agent = NeuroSymbolicPredictiveAgent(
    PredictiveAgentConfig(
        input_dim=4,
        hidden_dim=2,
        symbols=("left", "right", "rest"),
    )
)
result = agent.observe([0.25, -0.2, 0.1, -0.1], top_k=2)
print(result.signature.popcount)
```

## High-Level Agent

::: sc_neurocore.neuro_symbolic.agent

The high-level agent keeps the existing predictive-coding and
hyperdimensional-symbol implementation as the underlying engine. Its SC-facing
contract is explicit:

- prediction error is encoded as `xor_bits`;
- integer error magnitude is `popcount`;
- `normalised_popcount` is the hardware-friendly magnitude proxy;
- optional `learn=True` applies one predictive-coding update after inference.

## Low-Level Primitives

::: sc_neurocore.neuro_symbolic.predictive_coding
