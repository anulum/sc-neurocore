<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Project — SC-NeuroCore local LLM bridge guide
-->

# Local LLM Bridge

SC-NeuroCore now includes an opt-in bridge for locally hosted language models.
It is designed for:

- local spike-raster summarisation
- local explanation of SC network activity
- prompt-based analysis without any cloud dependency

## Scope

The bridge is intentionally narrow.

It does **not** train a language model.
It does **not** add a mandatory runtime dependency.
It does **not** alter any existing simulation path.

It provides a local client around two endpoint styles:

- Ollama chat API
- OpenAI-compatible chat-completions API hosted locally

## Module

- `sc_neurocore.bridges.local_llm`

Key symbols:

- `LocalLLMConfig`
- `LocalLLMProvider`
- `LocalLLMBridge`
- `LocalLLMResponse`
- `SpikePromptAdapter`

## Example: Ollama

```python
import numpy as np

from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMProvider,
)

raster = np.array(
    [
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 0],
    ],
    dtype=bool,
)

bridge = LocalLLMBridge(
    LocalLLMConfig(
        base_url="http://127.0.0.1:11434",
        provider=LocalLLMProvider.OLLAMA,
        model="qwen2.5:7b-instruct",
    )
)

response = bridge.analyse_spike_raster(
    raster,
    dt_ms=1.0,
    neuron_labels=["n0", "n1", "n2"],
)

print(response.text)
```

## Example: OpenAI-compatible local server

```python
from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMProvider,
)

bridge = LocalLLMBridge(
    LocalLLMConfig(
        base_url="http://127.0.0.1:8000",
        provider=LocalLLMProvider.OPENAI_COMPAT,
        model="local-model",
    )
)

response = bridge.chat("Summarise this network state in three bullet points.")
print(response.text)
```

## Practical notes

- Use `PYTHONPATH=src:bridge` when launching from a source checkout so the
  maintained bridge package is importable.
- The bridge is local-only. It does not talk to hosted APIs.
- If the endpoint is down or returns malformed JSON, `LocalLLMError` is raised.
- Prompt shaping is deliberately simple. Extend `SpikePromptAdapter` rather
  than embedding prompt logic throughout the codebase.

## Explainability workflow

The bridge is wired into the deterministic explainability layer as an opt-in
enhancer.

```python
from sc_neurocore.bridges.local_llm import LocalLLMBridge, LocalLLMConfig, LocalLLMProvider
from sc_neurocore.explainability.explainability import ExplainabilityEngine

bridge = LocalLLMBridge(
    LocalLLMConfig(
        base_url="http://127.0.0.1:11434",
        provider=LocalLLMProvider.OLLAMA,
        model="gemma3:1b",
    )
)

engine = ExplainabilityEngine(seed=0xACE1)
node, explanation = engine.explain_spike_with_local_llm(
    "n0",
    threshold_q16=32768,
    bitstream_length=256,
    spike_threshold_count=100,
    bridge=bridge,
)
```

The deterministic replay path still runs first.
The local model only rewrites the explanation text.
It does not replace the numeric decision path or provenance chain.

## Safety boundary

This bridge is a research-tier interface, not a control loop.

Use it for:

- explanation
- summarisation
- offline operator support

Do not use it as the sole authority for:

- safety-critical control decisions
- verification of mathematical correctness
- replacement of deterministic model checks
