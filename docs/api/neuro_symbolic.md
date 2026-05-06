<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (C) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore documentation -->

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

## Self-Verification Trace

The self-verification layer turns a neuro-symbolic inference result into
checked obligations rather than a narrative explanation:

```python
from sc_neurocore.neuro_symbolic import build_self_verification_trace

observation = [0.25, -0.2, 0.1, -0.1]
result = agent.observe(observation, top_k=2)
verification = build_self_verification_trace(result, observation=observation)
assert verification.passed
print(verification.result_digest)
```

The trace checks prediction/error consistency, SC XOR/popcount consistency,
reasoning-trace bounds, confidence/similarity ranges, sorted symbolic scores,
and emits a stable SHA-256 digest for audit logs.

::: sc_neurocore.neuro_symbolic.self_verification

## Low-Level Primitives

::: sc_neurocore.neuro_symbolic.predictive_coding
